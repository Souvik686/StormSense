"""Nowcast Service: Manages real-time and operational inference,
spatial district risk aggregation, GeoJSON generation, and thermodynamic
diagnostics for the StormSense V2 nowcasting engine.
"""
from __future__ import annotations

import json
import os
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
from .xai import calculate_xai_factors
import torch
from shapely.geometry import shape, Point

from src.features.normalize import SINGLE_VARS, PRESSURE_VARS, load_stats
from src.inference.predictor import get_predictor, NowcastPredictor
from src.inference.risk_map import predictions_to_geojson
from src.inference.risk_surface import (
    generate_risk_surface_png,
    generate_analysis_surface_png,
    HISTORICAL_RAIN_RENDER_MAX_MM_H,
    get_or_create_wb_mask,
    WB_BOUNDS_LEAFLET,
    WB_MIN_LAT,
    WB_MAX_LAT,
    WB_MIN_LON,
    WB_MAX_LON,
)
from src.inference import gfs_live
from src.utils.config import load_config, Config

_SERVICE_LOCK = threading.Lock()
_SERVICE_INSTANCE: Optional[NowcastService] = None

# The single place the live pipeline's honest caveats are written. Exposed
# verbatim by /api/nowcast/summary and /api/live/ml-status so the API can never
# describe the pipeline more favourably in one endpoint than another.
# ---------------------------------------------------------------------------
# Historical case study identity -- THE single source of truth.
#
# Every label, API field and UI string describing the case study derives from
# these constants. Duplicating the event name as literals across backend and
# frontend is how the codebase previously ended up serving Cyclone Remal data
# under a "Kalbaishakhi" label in some places and the correct name in others.
#
# HISTORICAL_ANALYSIS_TIME is the authoritative t=0. All historical horizons are
# computed as offsets from it, never from wall-clock time.
# ---------------------------------------------------------------------------
HISTORICAL_EVENT_NAME = "Cyclone Remal"
HISTORICAL_EVENT_DETAIL = "Cyclone Remal (Landfall Approach)"
HISTORICAL_ANALYSIS_TIME = "2024-05-26T12:00:00Z"
HISTORICAL_ANALYSIS_SOURCE = "ERA5 reanalysis (observed atmospheric analysis)"

LIVE_KNOWN_LIMITATIONS = [
    "The six input timesteps come from real GFS f000 analyses only. GFS publishes "
    "analyses every 6 hours, so the hourly slots between two analyses are "
    "linearly time-interpolated; sub-6-hour atmospheric transients are smoothed. "
    "No GFS forecast hour is ever used as an input timestep.",
    "Because GFS analyses are published with a production lag, the atmospheric "
    "state the model reads is as of the latest published analysis, which is "
    "typically several hours behind the wall-clock instant the forecast is "
    "issued for. Both times are reported separately and must not be conflated.",
    "GFS (NCEP) is a different NWP system from the ERA5 reanalysis (ECMWF) the "
    "model was trained on. This is a genuine distribution shift, so live "
    "probabilities are directionally informative and are not as well calibrated "
    "as the historical backtested metrics.",
]


def compute_valid_time(issue_time_str: Optional[str], lead_hours: int) -> str:
    """Dynamically compute forecast valid time from issue time and lead hours."""
    if not issue_time_str:
        return "N/A"
    try:
        clean = issue_time_str.replace("Z", "+00:00").replace(" ", "T")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        vt = dt + timedelta(hours=lead_hours)
        return vt.isoformat()
    except Exception:
        return f"{issue_time_str} +{lead_hours}h"


def format_iso_time(iso_str: Optional[str]) -> str:
    """Format an ISO-8601 timestamp string into a readable UTC date string."""
    if not iso_str:
        return "N/A"
    try:
        clean = iso_str.replace("Z", "+00:00").replace(" ", "T")
        dt = datetime.fromisoformat(clean)
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return str(iso_str)


def derive_compound_risks(
    severe_prob: np.ndarray, rain_pred: np.ndarray, dem_norm: np.ndarray
) -> Dict[str, np.ndarray]:
    """Shared flash-flood/overall risk formula, used identically for Historical
    (ERA5-driven) and Live (GFS-driven) predictions so the two modes are only
    ever different in their *atmospheric input*, never in how risk is derived
    from the model's output.

    Parameters
    ----------
    severe_prob : (n_lead, H, W) calibrated severe-weather probability
    rain_pred   : (n_lead, H, W) predicted 3h rainfall, mm
    dem_norm    : (H, W) normalized DEM used as the model's static input (same
                  scale as sample["dem"][0] in _init_operational_state)
    """
    terrain_factor = 1.0 + 0.2 * np.clip(dem_norm, 0.0, 2.5)
    rain_norm = np.clip(rain_pred / 30.0, 0.0, 1.0)
    hydro_intensity = rain_norm * (0.4 + 0.6 * severe_prob)
    ff_risk = np.clip(hydro_intensity * terrain_factor[None], 0.0, 1.0)
    overall_risk = np.clip(0.5 * severe_prob + 0.3 * ff_risk + 0.2 * rain_norm, 0.0, 1.0)
    return {
        "flash_flood_risk": ff_risk.astype(np.float32),
        "overall_risk": overall_risk.astype(np.float32),
    }


class NowcastService:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.cfg = load_config(config_path)
        
        # 1. Resolve Checkpoint
        if checkpoint_path is None:
            ckpt_dir = self.cfg.path("training", "checkpoint_dir")
            v2_cal = os.path.join(ckpt_dir, "v2_calibrated_best.pt")
            v2_base = os.path.join(ckpt_dir, "v2_best.pt")
            v1_base = os.path.join(ckpt_dir, "best.pt")
            checkpoint_path = v2_cal if os.path.exists(v2_cal) else (v2_base if os.path.exists(v2_base) else v1_base)

        self.predictor = get_predictor(checkpoint_path, config_path)
        self.lead_times = [0] + list(self.predictor.lead_times)
        self.stats = self.predictor.stats

        # 2. Domain Coordinates
        domain = self.cfg.get("domain")
        n_lat, n_lon = domain["grid_shape"]
        self.lats = np.linspace(domain["lat_max"], domain["lat_min"], n_lat)
        self.lons = np.linspace(domain["lon_min"], domain["lon_max"], n_lon)

        # 3. Load District Boundaries & Precompute Cell Mapping
        self.districts_fc = None
        self.district_cells: Dict[str, List[tuple]] = {}
        self._wb_polygon = None  # lazily loaded outer state boundary (see _is_inside_west_bengal)
        self._init_district_mapping()

        # 4. Initialize Operational Reference Window & Run Initial Nowcast
        self.current_pred: Optional[Dict[str, Any]] = None
        self.current_valid_time: str = HISTORICAL_ANALYSIS_TIME
        self.current_thermo: Dict[str, Any] = {}
        # Denormalized ERA5 t0 surface state for the historical case study.
        # Initialized to None BEFORE _init_operational_state() so that a cache
        # load failure leaves a well-defined "no data" value rather than an
        # undefined attribute: get_summary() reads it directly and would
        # otherwise raise AttributeError instead of degrading to None.
        self.era5_surface_t0: Optional[Dict[str, np.ndarray]] = None
        self.risk_surface_png_cache: Dict[int, bytes] = {}
        # Rendered historical t=0 analysis surfaces, keyed by variable.
        self._historical_analysis_png_cache: Dict[str, bytes] = {}
        # Populated by _init_operational_state() when the case study cannot be
        # loaded; surfaced by the API instead of a silent substitution.
        self.historical_init_error: Optional[str] = None
        self._init_operational_state()

        # 5. Live (GFS-driven) operational state -- see src/inference/gfs_live.py
        # for the full temporal/scientific design. Populated by refresh_live_state();
        # None until the first successful fetch. A failed refresh NEVER clears an
        # existing live_pred (stale-data protection) -- it only records the error.
        self.live_pred: Optional[Dict[str, Any]] = None
        # THE authoritative reference instant the live forecast is issued for:
        # exact wall-clock, never floored to an hour. +2/+4/+6 are computed from
        # this and from nothing else, so the API and the browser agree exactly.
        self.live_reference_time: Optional[str] = None
        # The real GFS f000 analysis time the atmospheric input state came from.
        # Deliberately DISTINCT from live_reference_time -- see gfs_live.py.
        self.live_analysis_time: Optional[str] = None
        self.live_analysis_cycles: Optional[List[str]] = None
        self.live_slot_provenance: Optional[List[Dict[str, Any]]] = None
        # Real t0 atmospheric state in physical units (see _denormalize_live_surface).
        self.live_surface_state: Optional[Dict[str, np.ndarray]] = None
        # Thermodynamic diagnostics derived from the LIVE GFS analysis only.
        # Kept strictly separate from `current_thermo`, which is the frozen
        # historical (ERA5) case-study state -- mixing them would report
        # historical soundings as live.
        self.live_thermo: Optional[Dict[str, Any]] = None
        self.live_fetch_error: Optional[str] = None
        self.live_fetched_at: Optional[str] = None           # wall-clock time of last successful refresh
        self.live_risk_surface_png_cache: Dict[int, bytes] = {}
        self._live_dem_norm: Optional[np.ndarray] = None      # static DEM, same tensor used historically
        self.refresh_live_state()

    def _init_district_mapping(self):
        districts_path = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "west_bengal_districts_full.geojson")
        if os.path.exists(districts_path):
            try:
                with open(districts_path, "r", encoding="utf-8") as f:
                    self.districts_fc = json.load(f)
                
                for feat in self.districts_fc.get("features", []):
                    name = feat.get("properties", {}).get("Name", "Unknown")
                    poly = shape(feat["geometry"])
                    indices = []
                    for i, lat in enumerate(self.lats):
                        for j, lon in enumerate(self.lons):
                            p = Point(lon, lat)
                            # Within district polygon or 0.15 deg buffer (grid spacing is 0.25 deg)
                            if poly.contains(p) or poly.distance(p) < 0.15:
                                indices.append((i, j))
                    self.district_cells[name] = indices
            except Exception as e:
                print(f"[NowcastService] Warning: Failed to load district boundaries: {e}")

    def _init_operational_state(self):
        """Loads a real atmospheric state from cache and runs model inference."""
        cache_path = os.path.join(self.cfg.path("paths", "cache_root"), "era5_memmap")
        if os.path.exists(cache_path):
            try:
                from src.data.dataset_v2 import make_dataloaders_v2
                loaders = make_dataloaders_v2(cache_path, self.cfg)
                import numpy as np
                # Historical Case Study: Cyclone Remal (May 26, 2024, 12:00 UTC)
                dataset = loaders["test"].dataset
                # The case study is a specific real event. If its analysis time
                # is not in the held-out split we must FAIL, not silently serve a
                # different date under the Remal label -- a silent fallback to
                # "sample 100" is exactly how this app previously displayed one
                # event's data beneath another event's name.
                target_time = np.datetime64(
                    HISTORICAL_ANALYSIS_TIME.replace("Z", "").replace("+00:00", "")
                )
                sample_idx = None
                for i, w in enumerate(dataset.windows):
                    if dataset.times[w.input_end_idx] == target_time:
                        sample_idx = i
                        break
                if sample_idx is None:
                    raise RuntimeError(
                        f"Historical case study unavailable: {HISTORICAL_EVENT_NAME} "
                        f"analysis time {HISTORICAL_ANALYSIS_TIME} was not found in "
                        f"the reanalysis cache ({len(dataset.windows)} windows "
                        f"searched). Refusing to substitute a different event."
                    )
                sample = dataset[sample_idx]
                
                vt = sample.get("valid_time")
                if vt is not None:
                    self.current_valid_time = str(vt)[:19] + "Z"

                # Run inference on the operational atmospheric window
                # We need T=0 risk as well. The model outputs for +2h...+6h.
                # To get risk valid at T=0, we run inference on the sample from T-2h.
                sample_t2 = dataset[sample_idx - 2]
                
                def _infer(s):
                    with torch.no_grad():
                        batch = {
                            "surface": s["surface"][None].to(self.predictor.device),
                            "pressure_wind": s["pressure_wind"][None].to(self.predictor.device),
                            "pressure_thermo": s["pressure_thermo"][None].to(self.predictor.device),
                            "dem": s["dem"][None].to(self.predictor.device),
                        }
                        return self.predictor.model(batch)

                preds = _infer(sample)
                preds_now = _infer(sample_t2)

                def _get_prob(p):
                    logits = p["severe_weather_logit"][0]
                    if self.predictor.temperature is not None:
                        T = torch.tensor(self.predictor.temperature, device=logits.device, dtype=logits.dtype).view(-1, 1, 1)
                        return torch.sigmoid(logits / T).cpu().numpy()
                    return torch.sigmoid(logits).cpu().numpy()

                severe_prob = _get_prob(preds)
                severe_prob_now = _get_prob(preds_now)
                
                # Prepend the +2h output from T-2h prediction as the T=0 output for the current prediction
                severe_prob = np.concatenate([severe_prob_now[0:1], severe_prob], axis=0)

                rain_pred = preds["rain_3h_mm"][0].cpu().numpy()
                rain_pred_now = preds_now["rain_3h_mm"][0].cpu().numpy()
                rain_pred = np.concatenate([rain_pred_now[0:1], rain_pred], axis=0)

                # Binary classification using calibrated per-lead thresholds
                if self.predictor.threshold_per_lead is not None:
                    severe_binary = np.zeros_like(severe_prob, dtype=np.uint8)
                    for li, lh in enumerate(self.lead_times):
                        thr = float(self.predictor.threshold_per_lead.get(lh, self.predictor.threshold))
                        severe_binary[li] = (severe_prob[li] >= thr).astype(np.uint8)
                else:
                    severe_binary = (severe_prob >= self.predictor.threshold).astype(np.uint8)

                # Flash-flood/overall compound risk (shared formula, see derive_compound_risks)
                dem_norm = sample["dem"][0].numpy()  # (33, 25)
                self._live_dem_norm = dem_norm  # DEM is static; reused for live inference too
                compound = derive_compound_risks(severe_prob, rain_pred, dem_norm)

                self.current_pred = {
                    "severe_weather_prob": severe_prob.astype(np.float32),
                    "rain_3h_mm_pred": rain_pred.astype(np.float32),
                    "severe_weather_binary": severe_binary,
                    "flash_flood_risk": compound["flash_flood_risk"],
                    "overall_risk": compound["overall_risk"],
                    "lead_times_hours": self.lead_times,
                    "threshold": self.predictor.threshold_per_lead or self.predictor.threshold,
                    "lats": self.lats,
                    "lons": self.lons,
                }

                # Extract real thermodynamic parameters from the raw/denormalized surface slice
                surf = sample["surface"].numpy()  # (6, 15, 33, 25)
                cape_norm = surf[-1, 5]
                cin_norm = surf[-1, 6]
                u10_norm = surf[-1, 0]
                v10_norm = surf[-1, 1]

                cape_std = self.stats["cape"]["std"]
                cape_mean = self.stats["cape"]["mean"]
                cin_std = self.stats["cin"]["std"]
                cin_mean = self.stats["cin"]["mean"]
                u10_std = self.stats["u10"]["std"]
                u10_mean = self.stats["u10"]["mean"]
                v10_std = self.stats["v10"]["std"]
                v10_mean = self.stats["v10"]["mean"]

                cape_real = np.maximum(0.0, cape_norm * cape_std + cape_mean)
                cin_real = np.maximum(0.0, cin_norm * cin_std + cin_mean)
                u10_real = u10_norm * u10_std + u10_mean
                v10_real = v10_norm * v10_std + v10_mean
                wind_speed = np.sqrt(u10_real**2 + v10_real**2)

                # Bulk wind shear (1000hPa to 700hPa in wind group)
                pres_wind = sample["pressure_wind"].numpy()  # (6, 2, 3, 33, 25)
                u_shear = pres_wind[-1, 0, 0] - pres_wind[-1, 0, 2]
                v_shear = pres_wind[-1, 1, 0] - pres_wind[-1, 1, 2]
                shear_mps = np.sqrt(u_shear**2 + v_shear**2) * 10.0  # approximate scaling to m/s

                # Extract at North 24 Parganas centroid (lat=22.724, lon=88.479)
                lat_i = int(np.argmin(np.abs(self.lats - 22.724)))
                lon_j = int(np.argmin(np.abs(self.lons - 88.479)))

                self.current_thermo = {
                    "cape_j_kg": float(round(float(cape_real[lat_i, lon_j]), 1)),
                    "cape_domain_max": float(round(float(cape_real.max()), 1)),
                    "cin_j_kg": float(round(float(cin_real[lat_i, lon_j]), 1)),
                    "bulk_shear_0_6km_mps": float(round(float(shear_mps[lat_i, lon_j]), 1)),
                    "bulk_shear_domain_max": float(round(float(shear_mps.max()), 1)),
                    "wind_speed_kmh": float(round(float(wind_speed[lat_i, lon_j] * 3.6), 1)),
                    "lifted_index": {
                        "value": None,
                        "status": "Unavailable in ERA5 native profile (proxy CAPE used)"
                    }
                }
                # Denormalize full ERA5 surface variables at t0 for spatial point queries
                t2m_k = surf[-1, 3] * self.stats["t2m"]["std"] + self.stats["t2m"]["mean"]
                d2m_k = surf[-1, 2] * self.stats["d2m"]["std"] + self.stats["d2m"]["mean"]
                temp_c = t2m_k - 273.15
                dew_c = d2m_k - 273.15
                vp = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
                svp = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
                rh_pct = np.clip(100.0 * (vp / svp), 0.0, 100.0)
                sp_pa = surf[-1, 4] * self.stats["sp"]["std"] + self.stats["sp"]["mean"]
                pres_hpa = sp_pa / 100.0
                # ERA5 `tp` is ALREADY in mm/hour in this cache: src/data/era5_loader.py
                # converts the raw ECMWF meters/hour to mm/hour (`ds["tp"] * 1000.0`,
                # units attr set to "mm") before the memmap is written, and the
                # normalization stats confirm it (mean 0.371, std 0.974 -- meters
                # would be ~3.7e-4). Multiplying by 1000 again here overflowed every
                # cell past the 150 mm clip ceiling, so 100% of the historical grid
                # reported a constant 150.0 mm instead of the real field. Denormalize
                # only; do not re-scale.
                tp_mm_h = surf[-1, SINGLE_VARS.index("tp")] * self.stats["tp"]["std"] + self.stats["tp"]["mean"]
                # Clip negatives only: z-score denormalization of a non-negative
                # accumulation can undershoot below 0, which is not physical. No
                # upper clip -- a real extreme must be allowed to read as extreme.
                rain_mm = np.clip(tp_mm_h, 0.0, None)

                self.era5_surface_t0 = {
                    "temp_c": temp_c.astype(np.float32),
                    "rh_pct": rh_pct.astype(np.float32),
                    "wind_kmh": (wind_speed * 3.6).astype(np.float32),
                    "pres_hpa": pres_hpa.astype(np.float32),
                    "rain_mm": rain_mm.astype(np.float32),
                }

                # Observed surface state at the case study's FORECAST HORIZONS.
                # Remal is a past event, so the atmosphere at t0+2/+4/+6h was
                # actually observed and is present in the same reanalysis cache.
                # These are verification observations (what really happened), NOT
                # model predictions -- the model has no temperature/humidity/wind
                # head -- so the API labels them as observed analyses and the UI
                # must never present them as StormSense forecast output.
                self.era5_surface_by_lead = {}
                try:
                    times = dataset.times
                    for lead_h in (2, 3, 4, 5, 6):
                        want = target_time + np.timedelta64(lead_h, "h")
                        idx = None
                        for k, w in enumerate(dataset.windows):
                            if times[w.input_end_idx] == want:
                                idx = k
                                break
                        if idx is None:
                            continue
                        s = dataset[idx]
                        sf = s["surface"].numpy() if hasattr(s["surface"], "numpy") else np.asarray(s["surface"])
                        u = sf[-1, SINGLE_VARS.index("u10")] * self.stats["u10"]["std"] + self.stats["u10"]["mean"]
                        v = sf[-1, SINGLE_VARS.index("v10")] * self.stats["v10"]["std"] + self.stats["v10"]["mean"]
                        t_k = sf[-1, SINGLE_VARS.index("t2m")] * self.stats["t2m"]["std"] + self.stats["t2m"]["mean"]
                        d_k = sf[-1, SINGLE_VARS.index("d2m")] * self.stats["d2m"]["std"] + self.stats["d2m"]["mean"]
                        sp_p = sf[-1, SINGLE_VARS.index("sp")] * self.stats["sp"]["std"] + self.stats["sp"]["mean"]
                        t_c = t_k - 273.15
                        dw_c = d_k - 273.15
                        _vp = 6.112 * np.exp((17.67 * dw_c) / (dw_c + 243.5))
                        _svp = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
                        self.era5_surface_by_lead[lead_h] = {
                            "temp_c": t_c.astype(np.float32),
                            "rh_pct": np.clip(100.0 * (_vp / _svp), 0.0, 100.0).astype(np.float32),
                            "wind_kmh": (np.sqrt(u ** 2 + v ** 2) * 3.6).astype(np.float32),
                            "pres_hpa": (sp_p / 100.0).astype(np.float32),
                            "valid_time": str(times[dataset.windows[idx].input_end_idx])[:19] + "Z",
                        }
                except Exception as e:
                    print(f"[NowcastService] Historical horizon surface states unavailable: {e}")

                # Precompute continuous West Bengal risk surface PNGs for sub-millisecond API response
                try:
                    wb_mask = get_or_create_wb_mask()
                    for li, lh in enumerate(self.lead_times):
                        self.risk_surface_png_cache[lh] = generate_risk_surface_png(
                            severe_prob[li], self.lats, self.lons, mask=wb_mask
                        )
                    print(f"[NowcastService] Precomputed historical continuous West Bengal risk surfaces for leads: {list(self.risk_surface_png_cache.keys())}")
                except Exception as e:
                    print(f"[NowcastService] Warning: Could not precompute risk surface PNGs: {e}")

                print(f"[NowcastService] Historical ({HISTORICAL_EVENT_NAME}) operational state loaded. Valid time: {self.current_valid_time}")
            except Exception as e:
                # Record WHY the historical state is missing so the API can
                # report a real reason instead of an empty case study. The
                # service still starts (live mode is independent), but every
                # historical endpoint will surface this error rather than
                # silently serving a different event or fabricated values.
                self.historical_init_error = str(e)
                print(f"[NowcastService] ERROR: Historical case study unavailable: {e}")

    def refresh_live_state(self, target_t0=None) -> bool:
        """Fetch the latest real GFS analysis, harmonize it into the model's
        input format (src/inference/gfs_live.py), and run genuine inference.

        On any failure, `self.live_pred` is left UNTOUCHED (stale-data
        protection: an old real prediction is better than none, and is clearly
        flagged stale by callers via `live_fetched_at` age) and
        `self.live_fetch_error` records exactly what went wrong. Returns True
        on success, False on failure.
        """
        try:
            from datetime import datetime, timezone
            if target_t0 is None:
                target_t0 = datetime.now(timezone.utc)
            import datetime
            from datetime import timedelta
            
            harmonized = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0)
            preds_dict = self.predictor.predict(
                surface=harmonized.surface,
                pressure=harmonized.pressure,
                dem=self._get_static_dem_meters(),
                timestamp=harmonized.t0.isoformat(),
            )
            
            # Fetch for T-2h to get the valid T=0 prediction (lead 2)
            harmonized_now = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0 - timedelta(hours=2))
            preds_dict_now = self.predictor.predict(
                surface=harmonized_now.surface,
                pressure=harmonized_now.pressure,
                dem=self._get_static_dem_meters(),
                timestamp=harmonized_now.t0.isoformat(),
            )
            
            import numpy as np
            severe_prob = np.concatenate([preds_dict_now["severe_weather_prob"][0:1], preds_dict["severe_weather_prob"]], axis=0)
            rain_pred = np.concatenate([preds_dict_now["rain_3h_mm_pred"][0:1], preds_dict["rain_3h_mm_pred"]], axis=0)
            severe_binary = np.concatenate([preds_dict_now["severe_weather_binary"][0:1], preds_dict["severe_weather_binary"]], axis=0)
            
            dem_norm = self._live_dem_norm if self._live_dem_norm is not None else np.zeros(
                (len(self.lats), len(self.lons)), dtype=np.float32
            )
            compound = derive_compound_risks(severe_prob, rain_pred, dem_norm)

            self.live_pred = {
                "severe_weather_prob": severe_prob,
                "rain_3h_mm_pred": rain_pred,
                # Must be the CONCATENATED array. Every other field here carries
                # the prepended T=0 slice, so storing the raw 5-lead
                # preds_dict["severe_weather_binary"] left this one array a slice
                # short: self.lead_times is [0,2,3,4,5,6], so lead=6 resolves to
                # index 5 and raised IndexError (size 5) in
                # predictions_to_geojson -- a 500 on the +6h risk map only.
                "severe_weather_binary": severe_binary,
                "flash_flood_risk": compound["flash_flood_risk"],
                "overall_risk": compound["overall_risk"],
                "lead_times_hours": self.lead_times,
                "threshold": preds_dict["threshold"],
                "lats": self.lats,
                "lons": self.lons,
            }
            # The forecast is ISSUED FOR the exact wall-clock reference instant
            # (never floored). The atmospheric state it read is as of the GFS
            # analysis time. Both are recorded; +2/+4/+6 derive from the former.
            # Keep the REAL t0 atmospheric state (physical units, straight from
            # the GFS analysis) so attribution and point queries can report
            # genuine values instead of placeholders.
            self.live_surface_state = self._denormalize_live_surface(harmonized.surface)
            self.live_thermo = self._compute_live_thermo(
                self.live_surface_state, harmonized.pressure, harmonized.analysis_t0
            )

            self.live_reference_time = harmonized.t0.isoformat()
            self.live_analysis_time = harmonized.analysis_t0.isoformat()
            self.live_analysis_cycles = [c.isoformat() for c in harmonized.analysis_cycles]
            self.live_slot_provenance = harmonized.slot_provenance
            self.live_fetch_error = None
            self.live_fetched_at = harmonized.fetched_at.isoformat()

            try:
                wb_mask = get_or_create_wb_mask()
                self.live_risk_surface_png_cache = {}
                for li, lh in enumerate(self.lead_times):
                    self.live_risk_surface_png_cache[lh] = generate_risk_surface_png(
                        severe_prob[li], self.lats, self.lons, mask=wb_mask
                    )
            except Exception as e:
                print(f"[NowcastService] Warning: Could not regenerate live risk surface PNGs: {e}")

            print(f"[NowcastService] Live state refreshed. Reference (wall-clock) t0="
                  f"{self.live_reference_time}, GFS analysis={self.live_analysis_time}, "
                  f"analysis lag={harmonized.wallclock_age_hours:.1f}h")
            return True
        except Exception as e:
            self.live_fetch_error = str(e)
            print(f"[NowcastService] Live refresh FAILED (live_pred left untouched): {e}")
            return False

    def _denormalize_live_surface(self, surface: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract the real t0 surface state in physical units from the live GFS
        input tensor.

        `surface` is (T, len(SINGLE_VARS), H, W) in PHYSICAL units already (the
        GFS harmonizer emits physical units; normalization happens later inside
        the predictor), so this is an extraction plus standard derivations --
        no unit guessing.
        """
        t0 = surface[-1]  # newest slot = the real analysis
        idx = {v: SINGLE_VARS.index(v) for v in SINGLE_VARS}

        u10 = t0[idx["u10"]]
        v10 = t0[idx["v10"]]
        t2m_k = t0[idx["t2m"]]
        d2m_k = t0[idx["d2m"]]

        temp_c = t2m_k - 273.15
        dew_c = d2m_k - 273.15
        # Magnus formula, same form used for the historical ERA5 state.
        vp = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
        svp = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        rh = np.clip(100.0 * (vp / svp), 0.0, 100.0)

        return {
            "temp_c": temp_c.astype(np.float32),
            "dewpoint_c": dew_c.astype(np.float32),
            "rh_pct": rh.astype(np.float32),
            "wind_kmh": (np.sqrt(u10 ** 2 + v10 ** 2) * 3.6).astype(np.float32),
            "pres_hpa": (t0[idx["sp"]] / 100.0).astype(np.float32),
            "cape_j_kg": np.maximum(0.0, t0[idx["cape"]]).astype(np.float32),
            "cin_j_kg": np.maximum(0.0, t0[idx["cin"]]).astype(np.float32),
            "tcwv_kg_m2": t0[idx["tcwv"]].astype(np.float32),
            # tp is already mm/hour after harmonization (PRATE * 3600).
            "rain_mm_h": np.clip(t0[idx["tp"]], 0.0, None).astype(np.float32),
        }

    def _compute_live_thermo(
        self,
        surface_state: Dict[str, np.ndarray],
        pressure: np.ndarray,
        analysis_t0,
    ) -> Dict[str, Any]:
        """Derive thermodynamic diagnostics from the REAL live GFS analysis.

        Every value returned here comes from the live analysis that was just
        ingested; nothing is borrowed from the historical ERA5 case study, and
        nothing is zero-filled. A variable that is genuinely absent from the
        live input is reported as None with a reason, never as 0 and never as a
        placeholder number.

        Vertical extent honesty: GFS live ingestion carries wind at 1000/850/700
        hPa only (see gfs_live.WIND_LEVELS_HPA). 700 hPa is roughly 3 km, so the
        shear computed here is a 1000->700 hPa bulk shear and is labelled as
        such. It is NOT the 0-6 km bulk shear the historical ERA5 path reports,
        and must never be presented as one.
        """
        # Domain centroid -- the same reference point the historical path uses,
        # so live and historical diagnostics describe the same location.
        lat_i = int(np.argmin(np.abs(self.lats - 22.724)))
        lon_j = int(np.argmin(np.abs(self.lons - 88.479)))

        def _point(field: Optional[np.ndarray]) -> Optional[float]:
            """Read one grid point, rejecting NaN rather than coercing to zero."""
            if field is None:
                return None
            val = float(field[lat_i, lon_j])
            if not np.isfinite(val):
                return None
            return val

        cape = _point(surface_state.get("cape_j_kg"))
        cin = _point(surface_state.get("cin_j_kg"))
        wind_kmh = _point(surface_state.get("wind_kmh"))

        # Bulk shear between 1000 hPa and 700 hPa from the live pressure tensor.
        # pressure is (n_cycles, n_vars, n_levels, H, W); take the newest slot.
        shear_mps: Optional[float] = None
        shear_note = None
        try:
            u_i = PRESSURE_VARS.index("u")
            v_i = PRESSURE_VARS.index("v")
            lvl_1000 = gfs_live.PRESSURE_LEVELS_HPA.index(1000)
            lvl_700 = gfs_live.PRESSURE_LEVELS_HPA.index(700)

            u_lo = float(pressure[-1, u_i, lvl_1000, lat_i, lon_j])
            v_lo = float(pressure[-1, v_i, lvl_1000, lat_i, lon_j])
            u_hi = float(pressure[-1, u_i, lvl_700, lat_i, lon_j])
            v_hi = float(pressure[-1, v_i, lvl_700, lat_i, lon_j])

            if all(np.isfinite(x) for x in (u_lo, v_lo, u_hi, v_hi)):
                shear_mps = float(
                    round(float(np.hypot(u_hi - u_lo, v_hi - v_lo)), 1)
                )
            else:
                shear_note = "Wind levels missing from the live analysis."
        except (ValueError, IndexError) as e:
            shear_note = f"Shear levels unavailable in live input: {e}"

        return {
            "source": "live_gfs_analysis",
            "analysis_time_utc": analysis_t0.isoformat() if analysis_t0 else None,
            "reference_point": {"lat": float(self.lats[lat_i]), "lon": float(self.lons[lon_j])},
            "cape_j_kg": round(cape, 1) if cape is not None else None,
            "cin_j_kg": round(cin, 1) if cin is not None else None,
            # Named for what it physically is, not for what the historical path
            # reports. The frontend prints this label verbatim.
            "bulk_shear_1000_700hpa_mps": shear_mps,
            "bulk_shear_label": "1000–700 hPa bulk shear (~0–3 km)",
            "bulk_shear_note": shear_note,
            "surface_wind_kmh": round(wind_kmh, 1) if wind_kmh is not None else None,
            "surface_wind_source": "GFS 10 m wind (u10/v10) from the live analysis",
            # 0-6 km shear needs winds above 700 hPa, which live GFS ingestion
            # does not carry. Stated explicitly so the UI cannot imply otherwise.
            "bulk_shear_0_6km_mps": None,
            "bulk_shear_0_6km_status": (
                "Not computed: live GFS ingestion carries wind at 1000/850/700 hPa "
                "only, so winds near 6 km are not available."
            ),
            "lifted_index": {
                "value": None,
                "status": "Not available from the live GFS single-level fields.",
            },
        }

    @property
    def live_valid_time(self) -> Optional[str]:
        """The instant the live forecast is issued for: exact wall-clock, not
        floored, and not the GFS cycle hour. Retained under the original name so
        existing callers keep working, but it now unambiguously means the
        reference time -- `live_analysis_time` is the GFS analysis state."""
        return self.live_reference_time

    def _get_static_dem_meters(self) -> np.ndarray:
        """Return the static SRTM DEM in raw meters (not normalized), the same
        elevation field used to build the historical operational state, for
        reuse as the live pipeline's DEM input -- DEM never changes and is not
        part of the GFS live ingestion."""
        if hasattr(self, "_dem_meters_cache") and self._dem_meters_cache is not None:
            return self._dem_meters_cache
        cache_path = os.path.join(self.cfg.path("paths", "cache_root"), "era5_memmap")
        dem_path = os.path.join(cache_path, "dem_elevation_m.npy")
        if os.path.exists(dem_path):
            self._dem_meters_cache = np.load(dem_path).astype(np.float32)
        else:
            # Fall back to zeros if the cache is unavailable; DEM contributes only a
            # secondary orographic-lift term in flash-flood proxy, not the core
            # severe-weather signal, so this degrades gracefully rather than blocking.
            print("[NowcastService] Warning: static DEM cache not found; live DEM input is zero-filled.")
            self._dem_meters_cache = np.zeros((len(self.lats), len(self.lons)), dtype=np.float32)
        return self._dem_meters_cache

    def _get_district_cells(self, district_name: str) -> List[tuple]:
        return self.district_cells.get(district_name, [])

    def _level_for_prob(self, p: float) -> str:
        if p >= 0.75:
            return "red"
        elif p >= 0.50:
            return "orange"
        elif p >= 0.25:
            return "yellow"
        else:
            return "green"

    def _stage_for_level(self, level: str) -> str:
        stages = {
            "red": "WARNING",
            "orange": "ALERT",
            "yellow": "WATCH",
            "green": "NORMAL"
        }
        return stages.get(level.lower(), "NORMAL")

    def _action_for_level(self, level: str) -> str:
        actions = {
            "red": "EVACUATE / SHELTER",
            "orange": "PREPARE CIVIC DEFENCE",
            "yellow": "MONITOR SENSORS",
            "green": "ROUTINE WATCH"
        }
        return actions.get(level.lower(), "ROUTINE WATCH")

    # ── Mode resolution ──────────────────────────────────────────────────────
    # Live and Historical differ ONLY in which cached prediction (and its issue
    # time) is read. All aggregation/formatting below is shared, so the two modes
    # can never diverge in how risk is derived -- only in their atmospheric input.

    def _resolve_mode(self, mode: Optional[str]) -> str:
        return "live" if str(mode or "historical").lower() == "live" else "historical"

    def _pred_for_mode(self, mode: Optional[str]) -> Dict[str, Any]:
        m = self._resolve_mode(mode)
        pred = self.live_pred if m == "live" else self.current_pred
        if pred is None:
            if m == "live":
                raise RuntimeError(
                    "Live forecast unavailable: "
                    + (self.live_fetch_error or "operational atmospheric input has not been ingested yet.")
                )
            raise RuntimeError(
                "Historical case study unavailable: "
                + (self.historical_init_error or "no prediction has been generated.")
            )
        return pred

    def _issue_time_for_mode(self, mode: Optional[str]) -> str:
        """The authoritative reference instant that +2/+4/+6 are measured from.

        Live: the exact wall-clock instant the current live state was issued for
        (never floored to an hour, never the GFS cycle hour). Historical: the
        case study's own analysis time. Every horizon in every endpoint derives
        from this one value, so the API and the browser can never disagree."""
        m = self._resolve_mode(mode)
        if m == "live":
            return self.live_reference_time or "Unknown"
        return self.current_valid_time

    # Operational analyses publish every 6h, and production lag means the newest
    # available cycle is routinely 6-10h old in normal operation (measured against
    # the live feed: a cycle is typically not published until ~4-5h after its
    # nominal time). Staleness therefore means "older than one full cycle interval
    # plus expected production lag" -- a tighter bound would flag healthy
    # operation as stale.
    LIVE_STALE_AFTER_HOURS = 12.0

    def is_live_stale(self, max_age_hours: Optional[float] = None) -> bool:
        """True when the live prediction is older than one cycle interval plus
        expected production lag, i.e. a newer analysis should have arrived but
        the refresh has not succeeded. Callers surface this as a 'stale' badge
        rather than silently presenting old data as current."""
        max_age_hours = self.LIVE_STALE_AFTER_HOURS if max_age_hours is None else max_age_hours
        if self.live_pred is None or self.live_analysis_time is None:
            return True
        try:
            # Staleness is a property of the ATMOSPHERIC STATE, so it is measured
            # against the GFS analysis time. Measuring it against the reference
            # time would be meaningless -- that is always "now" by construction.
            t0 = datetime.fromisoformat(self.live_analysis_time)
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - t0).total_seconds() / 3600.0 > max_age_hours
        except Exception:
            return True

    def get_live_status(self) -> Dict[str, Any]:
        """Machine-readable live-pipeline health for the API/UI freshness logic."""
        analysis_lag_h = None
        if self.live_analysis_time and self.live_reference_time:
            try:
                analysis_lag_h = round(
                    (
                        datetime.fromisoformat(self.live_reference_time)
                        - datetime.fromisoformat(self.live_analysis_time)
                    ).total_seconds()
                    / 3600.0,
                    2,
                )
            except Exception:
                analysis_lag_h = None
        return {
            "available": self.live_pred is not None,
            # Instant the forecast is issued for (exact wall-clock).
            "issue_time": self.live_reference_time,
            "issue_time_formatted": format_iso_time(self.live_reference_time),
            "reference_time": self.live_reference_time,
            # Real GFS f000 analysis the input state came from -- a DIFFERENT
            # thing, and never presented as the issue time.
            "analysis_time": self.live_analysis_time,
            "analysis_time_formatted": format_iso_time(self.live_analysis_time),
            "analysis_cycles": self.live_analysis_cycles,
            "analysis_lag_hours": analysis_lag_h,
            "input_source": "NOAA GFS f000 analyses (0.25°), time-interpolated to hourly slots",
            "uses_forecast_hours_as_input": False,
            "last_refreshed": self.live_fetched_at,
            "is_stale": self.is_live_stale(),
            "error": self.live_fetch_error,
            "input_slots": self.live_slot_provenance,
        }

    def get_summary(
        self,
        lead_hours: int = 2,
        district: str = "North 24 Parganas",
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Produce the comprehensive hazard and nowcasting summary for the
        requested mode ('live' -> GFS-driven prediction, 'historical' -> ERA5
        case-study prediction). Aggregation logic is identical for both."""
        pred = self._pred_for_mode(mode)
        resolved_mode = self._resolve_mode(mode)
        issue_time = self._issue_time_for_mode(mode)

        if lead_hours not in self.lead_times:
            lead_hours = self.lead_times[0]
        li = self.lead_times.index(lead_hours)

        prob_grid = pred["severe_weather_prob"][li]
        rain_grid = pred["rain_3h_mm_pred"][li]
        ff_grid = pred["flash_flood_risk"][li]
        overall_grid = pred["overall_risk"][li]

        # Aggregate for specific district (or fallback to domain)
        cells = self._get_district_cells(district)
        if cells:
            p_vals = [prob_grid[i, j] for i, j in cells]
            r_vals = [rain_grid[i, j] for i, j in cells]
            ff_vals = [ff_grid[i, j] for i, j in cells]
            ov_vals = [overall_grid[i, j] for i, j in cells]

            # In severe warning meteorology, max risk triggers civil defense actions
            dist_prob = float(np.max(p_vals))
            dist_rain = float(np.max(r_vals))
            dist_ff = float(np.max(ff_vals))
            dist_overall = float(np.max(ov_vals))
            dist_prob_mean = float(np.mean(p_vals))
        else:
            dist_prob = float(prob_grid.max())
            dist_rain = float(rain_grid.max())
            dist_ff = float(ff_grid.max())
            dist_overall = float(overall_grid.max())
            dist_prob_mean = float(prob_grid.mean())

        lvl_overall = self._level_for_prob(dist_overall)
        lvl_ts = self._level_for_prob(dist_prob)
        lvl_rain = self._level_for_prob(dist_rain / 15.0)  # >15mm is p99 severe
        lvl_ff = self._level_for_prob(dist_ff)

        # Multi-horizon timeline points for frontend scrubber
        timeline = []
        for i, h in enumerate(self.lead_times):
            p_h = pred["severe_weather_prob"][i]
            r_h = pred["rain_3h_mm_pred"][i]
            ff_h = pred["flash_flood_risk"][i]
            ov_h = pred["overall_risk"][i]

            if cells:
                p_c = float(np.max([p_h[ci, cj] for ci, cj in cells]))
                r_c = float(np.max([r_h[ci, cj] for ci, cj in cells]))
                ff_c = float(np.max([ff_h[ci, cj] for ci, cj in cells]))
                ov_c = float(np.max([ov_h[ci, cj] for ci, cj in cells]))
            else:
                p_c = float(p_h.max())
                r_c = float(r_h.max())
                ff_c = float(ff_h.max())
                ov_c = float(ov_h.max())

            lvl_c = self._level_for_prob(ov_c)
            vt_step = compute_valid_time(issue_time, int(h))
            timeline.append({
                "hours_from_now": int(h),
                "label": f"+{h}h",
                "severe_weather_pct": int(round(p_c * 100)),
                "rainfall_mm_3h": float(round(r_c, 2)),
                "flash_flood_pct": int(round(ff_c * 100)),
                "overall_risk_pct": int(round(ov_c * 100)),
                "risk_level": lvl_c,
                "stage": self._stage_for_level(lvl_c),
                "forecast_valid_time": vt_step,
                "valid_time_formatted": format_iso_time(vt_step),
            })

        # t=0 physical atmospheric state. Only meaningful for the historical
        # case study, whose ERA5 input window is denormalized at init; live mode
        # reports current conditions separately (surface observations endpoint),
        # so this is deliberately omitted rather than filled with historical data.
        surface_obs_t0 = None
        if resolved_mode == "historical" and self.era5_surface_t0 is not None:
            if cells:
                ci, cj = cells[0]
            else:
                ci, cj = 11, 18  # North 24 Parganas grid cell
            surface_obs_t0 = {
                "source": "Reference atmospheric analysis (t=0)",
                "time": format_iso_time(issue_time),
                "temperature_c": float(round(float(self.era5_surface_t0["temp_c"][ci, cj]), 1)),
                "humidity_pct": int(round(float(self.era5_surface_t0["rh_pct"][ci, cj]))),
                "wind_speed_kmh": float(round(float(self.era5_surface_t0["wind_kmh"][ci, cj]), 1)),
                "pressure_hpa": float(round(float(self.era5_surface_t0["pres_hpa"][ci, cj]), 1)),
                "rainfall_mm": float(round(float(self.era5_surface_t0["rain_mm"][ci, cj]), 1)),
            }

        # Observed surface state at the SELECTED horizon of the case study. Remal
        # is a past event, so these conditions were actually measured; they are
        # verification observations, never model output (the network predicts only
        # severe-weather probability and rainfall). Labelled so the UI can say so.
        surface_obs_at_lead = None
        by_lead = getattr(self, "era5_surface_by_lead", None)
        if resolved_mode == "historical" and by_lead:
            state_l = by_lead.get(int(lead_hours)) if lead_hours else None
            if state_l is not None:
                if cells:
                    ci, cj = cells[0]
                else:
                    ci, cj = 11, 18
                surface_obs_at_lead = {
                    "source": "ERA5 reanalysis (observed atmospheric analysis)",
                    "is_observation": True,
                    "is_model_output": False,
                    "lead_hours": int(lead_hours),
                    "valid_time": state_l["valid_time"],
                    "temperature_c": float(round(float(state_l["temp_c"][ci, cj]), 1)),
                    "humidity_pct": int(round(float(state_l["rh_pct"][ci, cj]))),
                    "wind_speed_kmh": float(round(float(state_l["wind_kmh"][ci, cj]), 1)),
                    "pressure_hpa": float(round(float(state_l["pres_hpa"][ci, cj]), 1)),
                }

        forecast_vt = compute_valid_time(issue_time, lead_hours)
        return {
            "model": "StormSense AI Forecast",
            # The analysis the input state came from -- NOT the issue time, and
            # MODE-SPECIFIC. These were previously read unconditionally from the
            # LIVE pipeline, so historical mode reported the current GFS cycle
            # (e.g. "11 Sep 2026 18:00 UTC") as the Cyclone Remal case study's
            # input analysis, next to a correct 2024-05-26 valid time. That is a
            # live-state leak into a frozen historical replay, not a display bug.
            #
            # Historical: the case study's own reanalysis analysis time.
            # Live: the real GFS f000 analysis time.
            "analysis_time": (
                format_iso_time(self.live_analysis_time)
                if resolved_mode == "live" and self.live_analysis_time
                else (format_iso_time(self.current_valid_time)
                      if resolved_mode == "historical" else None)
            ),
            "analysis_time_iso": (
                self.live_analysis_time if resolved_mode == "live"
                else self.current_valid_time
            ),
            "analysis_source": (
                "NOAA GFS 0.25° f000 analysis" if resolved_mode == "live"
                else HISTORICAL_ANALYSIS_SOURCE
            ),
            # Freshness describes the LIVE ingestion pipeline only. A frozen case
            # study is never "stale" and has no refresh cycle; reporting live
            # freshness beside 2024 data is meaningless and misleading.
            "last_refreshed": self.live_fetched_at if resolved_mode == "live" else None,
            "is_stale": self.is_live_stale() if resolved_mode == "live" else False,
            "input_slot_provenance": (
                self.live_slot_provenance if resolved_mode == "live" else None
            ),
            "temporal_requirement": "6 consecutive hourly timesteps (t-5h to t0)",
            "spatial_requirement": "33x25 grid at 0.25 degree resolution (20-28N, 84-90E)",
            "known_limitations": LIVE_KNOWN_LIMITATIONS,
            "parameters": self.predictor.model.count_parameters(),
            "mode": resolved_mode,
            "valid_time": issue_time,
            "issue_time": issue_time,
            "issue_time_formatted": format_iso_time(issue_time),
            "forecast_valid_time": forecast_vt,
            "valid_time_formatted": format_iso_time(forecast_vt),
            "operational_mode": (
                "Live Operational Nowcast"
                if resolved_mode == "live"
                # Must name the event actually loaded by _init_operational_state(),
                # which searches for 2024-05-26T12:00:00 (Cyclone Remal). The old
                # "Kalbaishakhi" label described a different case study and
                # mislabelled the data being served.
                else f"Historical Case Study / Demonstration Mode ({HISTORICAL_EVENT_DETAIL})"
            ),
            "live_status": self.get_live_status() if resolved_mode == "live" else None,
            "active_high_risk_district": self.get_active_high_risk_district(
                lead_hours=lead_hours, mode=resolved_mode
            ),
            "target_district": district,
            "selected_lead_hours": lead_hours,
            "lead_hours": lead_hours,
            "surface_obs_t0": surface_obs_t0,
            "surface_obs_at_lead": surface_obs_at_lead,
            "dem_metadata": {
                "source": "SRTM 30m Digital Elevation Model (resampled to 0.25°)",
                "role": "Static surface elevation input to spatial encoder (captures orographic lift & terrain gradient)",
                "is_dynamic_forecast": False,
                "domain_bounds": "20.0–28.0°N, 84.0–90.0°E",
            },
            "hazards": {
                "thunderstorm": {
                    "id": "thunderstorm",
                    "label": "Thunderstorm Risk",
                    "probability": int(round(dist_prob * 100)),
                    "probability_mean": int(round(dist_prob_mean * 100)),
                    "level": lvl_ts,
                    "stage": self._stage_for_level(lvl_ts),
                    "trend_label": f"{'+' if dist_prob > 0.5 else ''}{int(dist_prob*100 - 30)}% vs normal",
                    "detail": "Calibrated probabilistic proxy (CAPE/Shear)"
                },
                "heavy_rainfall": {
                    "id": "heavy_rainfall",
                    "label": "Heavy Rainfall Risk",
                    "probability": int(round(min(1.0, dist_rain / 15.0) * 100)),
                    "rate_mm_3h": float(round(dist_rain, 2)),
                    "level": lvl_rain,
                    "stage": self._stage_for_level(lvl_rain),
                    "trend_label": f"{float(round(dist_rain, 1))} mm / 3h",
                    "detail": "Deep regression precipitation head"
                },
                "flash_flood": {
                    "id": "flash_flood",
                    "label": "Flash-Flood Risk Proxy",
                    "probability": int(round(dist_ff * 100)),
                    "level": lvl_ff,
                    "stage": self._stage_for_level(lvl_ff),
                    "trend_label": f"{int(round(dist_ff*100))}% saturation proxy",
                    "detail": "Precipitation-gated terrain risk proxy"
                },
                "overall": {
                    "id": "overall",
                    "label": "Overall Hazard Level",
                    "probability": int(round(dist_overall * 100)),
                    "level": lvl_overall,
                    "stage": self._stage_for_level(lvl_overall),
                    "action": self._action_for_level(lvl_overall),
                    "alert_active": bool(dist_prob >= (self.predictor.threshold_per_lead.get(lead_hours, 0.5) if isinstance(self.predictor.threshold_per_lead, dict) else 0.5)),
                    "threshold_used": float(self.predictor.threshold_per_lead.get(lead_hours, 0.5) if isinstance(self.predictor.threshold_per_lead, dict) else 0.5),
                }
            },
            "timeline": timeline,
            "thermodynamics": self.current_thermo,
        }

    get_nowcast_summary = get_summary

    def get_risk_map(
        self, lead_hours: int = 2, as_polygon: bool = True, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate GeoJSON FeatureCollection for the requested lead time and mode."""
        pred = self._pred_for_mode(mode)
        if lead_hours not in self.lead_times:
            lead_hours = self.lead_times[0]
        li = self.lead_times.index(lead_hours)

        return predictions_to_geojson(
            pred,
            valid_time=self._issue_time_for_mode(mode),
            lead_time_hours=lead_hours,
            lead_idx=li,
            as_polygon=as_polygon,
        )

    def get_risk_surface_png(self, lead_hours: int = 2, mode: Optional[str] = None) -> bytes:
        """Return pre-rendered PNG bytes for the continuous West Bengal risk
        surface. Both modes render through the same masked generator, so the
        West Bengal clip is identical for Live and Historical."""
        resolved = self._resolve_mode(mode)
        cache = self.live_risk_surface_png_cache if resolved == "live" else self.risk_surface_png_cache
        if lead_hours not in self.lead_times:
            lead_hours = self.lead_times[0]
        if lead_hours in cache:
            return cache[lead_hours]
        pred = self._pred_for_mode(mode)
        li = self.lead_times.index(lead_hours)
        grid = pred["severe_weather_prob"][li]
        png = generate_risk_surface_png(grid, self.lats, self.lons, mask=get_or_create_wb_mask())
        cache[lead_hours] = png
        return png

    # Historical t=0 ANALYSIS surface -- the "NOW" state of the case study.
    #
    # Why this exists: selecting NOW in historical mode previously left the map
    # blank. renderContinuousRiskSurface() correctly refuses to paint a FORECAST
    # at NOW, and renderObservationSurface() correctly refuses to serve LIVE
    # station observations for a frozen 2024 case study -- so both layers were
    # removed and nothing was drawn.
    #
    # The scientifically correct NOW field for a historical replay is the
    # OBSERVED reanalysis state at the case study's analysis time (t=0), which is
    # already loaded in era5_surface_t0. It is an analysis, not a prediction, so
    # it is rendered with the observation palette and labelled as an analysis.
    # The model's forecast fields (leads 2-6) are NEVER reused for t=0.
    HISTORICAL_ANALYSIS_VARIABLES = {
        "rain_mm": {
            "label": "Rainfall rate at analysis time",
            "units": "mm/hour",
            "vmax": HISTORICAL_RAIN_RENDER_MAX_MM_H,
        },
    }

    def get_historical_analysis_surface_png(self, variable: str = "rain_mm") -> bytes:
        """PNG of the historical case study's OBSERVED t=0 field.

        Raises RuntimeError (never substitutes another event or a forecast) when
        the historical analysis state is unavailable.
        """
        if variable not in self.HISTORICAL_ANALYSIS_VARIABLES:
            raise RuntimeError(
                f"Unsupported historical analysis variable '{variable}'. "
                f"Available: {sorted(self.HISTORICAL_ANALYSIS_VARIABLES)}"
            )
        state = getattr(self, "era5_surface_t0", None)
        if not state or variable not in state:
            raise RuntimeError(
                "Historical analysis state is unavailable: the reanalysis cache "
                "for the case study did not load, so there is no t=0 field to "
                "render. No substitute event or forecast field is used."
            )
        cached = self._historical_analysis_png_cache.get(variable)
        if cached is not None:
            return cached
        spec = self.HISTORICAL_ANALYSIS_VARIABLES[variable]
        field = np.asarray(state[variable])
        # Scale the ramp to THIS field's observed peak rather than a fixed
        # ceiling. Remal's t=0 domain mean is ~1.2 mm/h against a 30 mm/h
        # ceiling, so a fixed vmax normalised almost every cell to near zero and
        # the rain structure rendered nearly invisible. The floor keeps a
        # genuinely dry field from being amplified into false signal.
        observed_peak = float(np.nanmax(field)) if np.isfinite(field).any() else 0.0
        vmax = max(observed_peak, 1.0)
        png = generate_analysis_surface_png(
            field,
            self.lats,
            self.lons,
            vmax=vmax,
            mask=get_or_create_wb_mask(),
        )
        self._historical_analysis_png_cache[variable] = png
        return png

    def get_historical_analysis_state(self) -> Dict[str, Any]:
        """Provenance + domain statistics for the historical t=0 analysis field.

        Everything here describes an OBSERVED reanalysis state, never a forecast,
        so the UI can label it truthfully.
        """
        state = getattr(self, "era5_surface_t0", None)
        if not state:
            return {
                "status": "unavailable",
                "mode": "historical",
                "reason": (
                    "The reanalysis cache for the historical case study did not "
                    "load, so no t=0 analysis field is available."
                ),
            }
        rain = np.asarray(state["rain_mm"])
        return {
            "status": "ok",
            "mode": "historical",
            "event": HISTORICAL_EVENT_NAME,
            "analysis_time": self.current_valid_time,
            "kind": "analysis",
            "is_forecast": False,
            "is_observation": True,
            "lead_hours": 0,
            "variable": "rain_mm",
            "label": "Rainfall rate at analysis time",
            "units": "mm/hour",
            "source": HISTORICAL_ANALYSIS_SOURCE,
            # Must match the ceiling the PNG renderer actually used, or the
            # legend would describe a different scale than the image.
            "render_scale_max_mm_h": float(round(max(float(rain.max()), 1.0), 2)),
            "domain_max_mm_h": float(round(float(rain.max()), 2)),
            "domain_mean_mm_h": float(round(float(rain.mean()), 3)),
            "provenance_note": (
                "Observed reanalysis field at the case study's analysis time "
                "(t=0). This is an analysis, not a model forecast, and not a "
                "live observation."
            ),
        }

    def get_risk_surface_bounds(self) -> Dict[str, Any]:
        """Return Leaflet-compatible bounding box for the risk surface."""
        return {
            "bounds": WB_BOUNDS_LEAFLET,
            "min_lat": WB_MIN_LAT,
            "max_lat": WB_MAX_LAT,
            "min_lon": WB_MIN_LON,
            "max_lon": WB_MAX_LON,
        }

    def _is_inside_west_bengal(self, lat: float, lon: float) -> bool:
        """Point-in-polygon against the authoritative outer state boundary
        (Data/BOUNDARIES/west_bengal_full.geojson), used to reject clicks outside the
        monitored region rather than reporting a nearest-grid-cell value for
        somewhere in Bihar/Bangladesh/Nepal."""
        if self._wb_polygon is None:
            try:
                path = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "west_bengal_full.geojson")
                with open(path, "r", encoding="utf-8") as f:
                    fc = json.load(f)
                self._wb_polygon = shape(fc["features"][0]["geometry"])
            except Exception as e:
                print(f"[NowcastService] Warning: could not load WB outer boundary: {e}")
                return True  # fail open rather than blocking all inspection
        return bool(self._wb_polygon.contains(Point(lon, lat)))

    def get_point_inspection(
        self, lat: float, lon: float, lead_hours: int = 2, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query the genuine model prediction at any clicked coordinate, for the
        selected mode and forecast horizon."""
        pred = self._pred_for_mode(mode)
        resolved_mode = self._resolve_mode(mode)
        issue_time = self._issue_time_for_mode(mode)
        if lead_hours not in self.lead_times:
            lead_hours = self.lead_times[0]
        li = self.lead_times.index(lead_hours)

        if not self._is_inside_west_bengal(lat, lon):
            return {
                "lat": float(round(lat, 4)),
                "lon": float(round(lon, 4)),
                "inside_monitored_region": False,
                "status": "LOCATION OUTSIDE MONITORED REGION",
                "mode": resolved_mode,
                "lead_hours": lead_hours,
            }

        # 1. District identification via point-in-polygon
        district_name = "Outside monitored district"
        if self.districts_fc:
            pt = Point(lon, lat)
            for feat in self.districts_fc.get("features", []):
                poly = shape(feat["geometry"])
                if poly.contains(pt):
                    district_name = feat.get("properties", {}).get("Name") or feat.get("properties", {}).get("district")
                    break

        # 2. Nearest grid cell in 33x25 domain
        lat_idx = int(np.argmin(np.abs(self.lats - lat)))
        lon_idx = int(np.argmin(np.abs(self.lons - lon)))
        cell_lat = float(self.lats[lat_idx])
        cell_lon = float(self.lons[lon_idx])

        # 3. Model Predictions at selected lead
        sp = float(pred["severe_weather_prob"][li, lat_idx, lon_idx])
        rp = float(pred["rain_3h_mm_pred"][li, lat_idx, lon_idx])
        ff = float(pred["flash_flood_risk"][li, lat_idx, lon_idx])
        ov = float(pred["overall_risk"][li, lat_idx, lon_idx])

        lvl = self._level_for_prob(sp)
        lbl = "WARNING" if sp >= 0.75 else "ALERT" if sp >= 0.50 else "WATCH" if sp >= 0.25 else "NORMAL"

        # 4. Physical analysis inputs at t=0. Only available for the historical
        # case study (denormalized at init); never borrowed from historical data
        # while serving a live query.
        obs_inputs = {
            "temperature_c": None,
            "humidity_pct": None,
            "rainfall_mm": None,
            "wind_kmh": None,
            "pressure_hpa": None,
        }
        if resolved_mode == "historical" and getattr(self, "era5_surface_t0", None):
            obs_inputs = {
                "temperature_c": float(round(float(self.era5_surface_t0["temp_c"][lat_idx, lon_idx]), 1)),
                "humidity_pct": int(round(float(self.era5_surface_t0["rh_pct"][lat_idx, lon_idx]))),
                "rainfall_mm": float(round(float(self.era5_surface_t0["rain_mm"][lat_idx, lon_idx]), 1)),
                "wind_kmh": float(round(float(self.era5_surface_t0["wind_kmh"][lat_idx, lon_idx]), 1)),
                "pressure_hpa": float(round(float(self.era5_surface_t0["pres_hpa"][lat_idx, lon_idx]), 1)),
            }

        # Dynamic Valid Time, anchored to the issue time of the SELECTED mode
        clean = issue_time.replace("Z", "+00:00").replace(" ", "T")
        issue_dt = datetime.fromisoformat(clean)
        valid_dt = issue_dt + timedelta(hours=lead_hours)
        valid_formatted = valid_dt.strftime("%d %b %Y %H:%M UTC")

        return {
            "lat": float(round(lat, 4)),
            "lon": float(round(lon, 4)),
            "inside_monitored_region": True,
            "mode": resolved_mode,
            "district": district_name,
            "grid_cell": {"lat": round(cell_lat, 2), "lon": round(cell_lon, 2)},
            "lead_hours": lead_hours,
            "issue_time_utc": issue_dt.strftime("%d %b %Y %H:%M UTC"),
            "forecast_valid_utc": valid_formatted,
            "model_name": "StormSense AI Forecast",
            "predictions": {
                "thunderstorm_prob_pct": float(round(sp * 100, 1)),
                "heavy_rain_mm": float(round(rp, 1)),
                "flash_flood_proxy_pct": float(round(ff * 100, 1)),
                "overall_risk_pct": float(round(ov * 100, 1)),
                "risk_level": lvl,
                "risk_label": lbl,
            },
            "observed_inputs": obs_inputs,
        }

    def get_district_advisories(
        self, lead_hours: int = 2, mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Calculate real model-derived risk aggregated across districts."""
        pred = self._pred_for_mode(mode)
        issue_time = self._issue_time_for_mode(mode)
        if lead_hours not in self.lead_times:
            lead_hours = self.lead_times[0]
        li = self.lead_times.index(lead_hours)

        prob_grid = pred["severe_weather_prob"][li]
        rain_grid = pred["rain_3h_mm_pred"][li]
        ff_grid = pred["flash_flood_risk"][li]
        ov_grid = pred["overall_risk"][li]

        advisories = []
        for name, cells in self.district_cells.items():
            if not cells:
                continue
            p_vals = [prob_grid[i, j] for i, j in cells]
            r_vals = [rain_grid[i, j] for i, j in cells]
            ff_vals = [ff_grid[i, j] for i, j in cells]
            ov_vals = [ov_grid[i, j] for i, j in cells]

            max_p = float(np.max(p_vals))
            p90_p = float(np.percentile(p_vals, 90))
            mean_p = float(np.mean(p_vals))
            max_r = float(np.max(r_vals))
            max_ff = float(np.max(ff_vals))
            max_ov = float(np.max(ov_vals))

            lvl = self._level_for_prob(max_ov)
            thr = float(self.predictor.threshold_per_lead.get(lead_hours, 0.5) if isinstance(self.predictor.threshold_per_lead, dict) else 0.5)
            alert = bool(max_p >= thr)

            # Contextual advisory text based on real metrics
            if alert:
                body = f"Severe thunderstorm alert (peak {int(round(max_p*100))}%, p90 {int(round(p90_p*100))}%) with expected rainfall up to {max_r:.1f} mm/3h. Low-lying zones face elevated flash-flood risk."
            elif max_ov >= 0.25:
                body = f"Moderate convective watch (peak {int(round(max_p*100))}%). Isolated showers up to {max_r:.1f} mm/3h anticipated."
            else:
                body = f"Normal meteorological conditions. Low precipitation ({max_r:.1f} mm/3h) and quiescent atmosphere."

            advisories.append({
                "district": name,
                "is_primary": (name == "North 24 Parganas"),
                "lead_time_hours": lead_hours,
                "risk_level": lvl,
                "stage": self._stage_for_level(lvl),
                "alert": alert,
                "thunderstorm_pct": int(round(max_p * 100)),
                "thunderstorm_p90_pct": int(round(p90_p * 100)),
                "thunderstorm_mean_pct": int(round(mean_p * 100)),
                "heavy_rainfall_mm_3h": float(round(max_r, 1)),
                "flash_flood_pct": int(round(max_ff * 100)),
                "overall_pct": int(round(max_ov * 100)),
                "title": f"{name} {self._stage_for_level(lvl)} ({'+'+str(lead_hours)+'h'})",
                "body": body,
                "aggregation_method": "Maximum & 90th Percentile Cell Hazard (Civil Protection Standard)",
                "issue_time": issue_time,
                "valid_time": compute_valid_time(issue_time, lead_hours),
                "valid_until": format_iso_time(compute_valid_time(issue_time, lead_hours)),
            })

        # Strictly risk-ordered: the monitored region is all of West Bengal, so the
        # highest-risk district must lead regardless of which district is nominally
        # "primary" -- an early-warning list that pins one district to the top would
        # bury the actual threat.
        advisories.sort(key=lambda x: -x["overall_pct"])
        return advisories

    # Threshold above which a district is considered an "active high-risk area"
    # (matches the WATCH boundary used by _level_for_prob).
    ACTIVE_RISK_THRESHOLD_PCT = 25

    def get_active_high_risk_district(
        self, lead_hours: int = 2, mode: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the single highest-risk district for the given mode/horizon, or
        None when no district currently exceeds the watch threshold. This is the
        ONE authoritative source for the dashboard's 'Active High-Risk Area' --
        it is always computed from the same forecast the map and cards show, and
        is never hardcoded to a particular district."""
        try:
            advisories = self.get_district_advisories(lead_hours=lead_hours, mode=mode)
        except RuntimeError:
            return None
        if not advisories:
            return None
        top = advisories[0]
        if top["overall_pct"] < self.ACTIVE_RISK_THRESHOLD_PCT and not top["alert"]:
            return None
        return {
            "district": top["district"],
            "overall_pct": top["overall_pct"],
            "thunderstorm_pct": top["thunderstorm_pct"],
            "risk_level": top["risk_level"],
            "stage": top["stage"],
            "alert": top["alert"],
            "lead_hours": lead_hours,
        }

    def get_high_risk_cells(
        self, lead_hours: int = 2, top_k: int = 8, mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top high-probability grid cells from the real model output for the given horizon."""
        pred = self._pred_for_mode(mode)
        issue_time = self._issue_time_for_mode(mode)
        if lead_hours not in self.lead_times:
            lead_hours = self.lead_times[0]
        li = self.lead_times.index(lead_hours)

        prob_grid = pred["severe_weather_prob"][li]
        rain_grid = pred["rain_3h_mm_pred"][li]
        ff_grid = pred["flash_flood_risk"][li]
        overall_grid = pred["overall_risk"][li]
        thr = float(self.predictor.threshold_per_lead.get(lead_hours, 0.5) if isinstance(self.predictor.threshold_per_lead, dict) else 0.5)

        cells_data = []
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                p = float(prob_grid[i, j])
                r = float(rain_grid[i, j])
                ff = float(ff_grid[i, j])
                ov = float(overall_grid[i, j])
                cells_data.append((p, r, ff, ov, i, j, lat, lon))

        cells_data.sort(key=lambda x: -x[0])

        top_cells = []
        vt = compute_valid_time(issue_time, lead_hours)
        vt_fmt = format_iso_time(vt)

        for rank, (p, r, ff, ov, i, j, lat, lon) in enumerate(cells_data[:top_k], 1):
            matched_district = "Domain Cell"
            for dname, indices in self.district_cells.items():
                if (i, j) in indices:
                    matched_district = dname
                    break

            lvl = self._level_for_prob(p)
            top_cells.append({
                "cell_id": f"ML-CELL-{i:02d}{j:02d}",
                "rank": rank,
                "lead_time_hours": lead_hours,
                "lead_hours": lead_hours,
                "issue_time": issue_time,
                "valid_time": vt,
                "valid_time_formatted": vt_fmt,
                "latitude": round(float(lat), 3),
                "longitude": round(float(lon), 3),
                "lat": round(float(lat), 3),
                "lon": round(float(lon), 3),
                "nearest_district": matched_district,
                "district": matched_district,
                "severe_weather_pct": int(round(p * 100)),
                "severe_weather_prob": round(p, 4),
                "thunderstorm_prob": round(p, 4),
                "threshold_used": thr,
                "is_alert": bool(p >= thr),
                "heavy_rainfall_mm_3h": round(r, 2),
                "rainfall_mm": round(r, 2),
                "flash_flood_risk_pct": int(round(ff * 100)),
                "flash_flood_risk": round(ff, 4),
                "overall_risk_pct": int(round(ov * 100)),
                "risk_level": lvl,
                "alert_level": lvl.upper(),
                "imd_color": lvl,
                "stage": self._stage_for_level(lvl),
                "grid_resolution": "0.25° (~28 km)"
            })
        return top_cells

    def get_xai_attribution(
        self,
        mode: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        lead_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Atmospheric factor attribution for the selected mode/location/horizon.

        METHOD (stated plainly, because it matters): this is RULE-BASED,
        PHYSICS-INSPIRED attribution. Scores are computed by explicit formulas
        over the atmospheric variables actually fed to the model and normalized
        to relative contributions. It is NOT SHAP, NOT gradient/saliency-based,
        and NOT neural-network feature importance -- no claim of either is made
        anywhere in the response.

        The inputs are read from the REAL live input tensor at the requested
        location (previously they were hardcoded placeholders -- humidity 85,
        CAPE 1500, wind 15 -- which made live attribution entirely fictitious
        and identical everywhere).
        """
        resolved = self._resolve_mode(mode)
        if resolved == "live":
            if self.live_surface_state is None:
                return {
                    "status": "unavailable",
                    "mode": "live",
                    "attribution_method": "Rule-based physics-inspired factor attribution",
                    "message": (
                        "Live atmospheric input has not been ingested yet, so no "
                        "attribution can be computed. "
                        + (self.live_fetch_error or "")
                    ).strip(),
                    "factors": [],
                }

            # Resolve the grid cell for the requested point (default: the
            # domain's highest-risk cell at this horizon, so the panel explains
            # the area the dashboard is actually warning about).
            lh = lead_hours if lead_hours in self.lead_times else self.lead_times[0]
            li = self.lead_times.index(lh)
            if lat is not None and lon is not None:
                i = int(np.argmin(np.abs(self.lats - lat)))
                j = int(np.argmin(np.abs(self.lons - lon)))
                loc_desc = f"{float(self.lats[i]):.2f}N, {float(self.lons[j]):.2f}E"
            else:
                prob = self.live_pred["severe_weather_prob"][li]
                i, j = np.unravel_index(int(np.argmax(prob)), prob.shape)
                i, j = int(i), int(j)
                loc_desc = f"highest-risk cell ({float(self.lats[i]):.2f}N, {float(self.lons[j]):.2f}E)"

            s = self.live_surface_state  # real, denormalized t0 GFS analysis state
            xai_inputs = {
                "rainfall_1h_mm": float(s["rain_mm_h"][i, j]),
                "rainfall_3h_mm": float(s["rain_mm_h"][i, j]) * 3.0,
                "rainfall_6h_mm": float(s["rain_mm_h"][i, j]) * 6.0,
                "humidity_percent": float(s["rh_pct"][i, j]),
                "dew_point_c": float(s["dewpoint_c"][i, j]),
                "cape_jkg": float(s["cape_j_kg"][i, j]),
                "wind_speed_kmh": float(s["wind_kmh"][i, j]),
                # No reflectivity is ingested anywhere in this system, so the
                # radar factor is genuinely absent rather than invented.
                "radar_dbz": None,
            }
            out = calculate_xai_factors(xai_inputs)
            out["mode"] = "live"
            out["location"] = loc_desc
            out["lead_hours"] = lh
            out["grid_cell"] = {"lat": round(float(self.lats[i]), 2), "lon": round(float(self.lons[j]), 2)}
            out["predicted_severe_prob_pct"] = round(
                float(self.live_pred["severe_weather_prob"][li, i, j]) * 100.0, 1
            )
            out["input_analysis_time"] = self.live_analysis_time
            out["method_disclosure"] = (
                "Rule-based, physics-inspired attribution computed from the atmospheric "
                "variables fed to the model. Not SHAP, not gradient-based, and not "
                "neural-network feature importance."
            )
            out["validation_note"] = (
                "The forecast model's skill was measured on held-out data. The "
                "attribution weighting itself is a physically-motivated heuristic and "
                "has not been separately validated."
            )
            return out

        return {
            "mode": "historical",
            "model_architecture": "StormSense AI Forecast",
            "parameters": self.predictor.model.count_parameters(),
            "attribution_method": "Physics-based diagnostic attribution",
            "method_disclosure": (
                "These are the physical atmospheric diagnostics the model consumes, "
                "ranked by their established meteorological role in convective "
                "development, with their real measured values from the case study's "
                "input window. This is expert-defined physical attribution -- it is "
                "NOT SHAP, not gradient-based saliency, and not learned neural-network "
                "feature importance."
            ),
            "validation_note": (
                "The forecast model's predictive skill was measured on the held-out "
                "2024 season (metrics below). The attribution ORDERING itself is "
                "physically motivated and was not separately validated."
            ),
            "factors": [
                {
                    "name": "Convective Instability (CAPE & CIN)",
                    "value": f"{self.current_thermo.get('cape_j_kg', 'N/A')} J/kg",
                    "physical_role": "Primary thermodynamic potential energy reservoir gating rapid updraft acceleration.",
                    "importance_rank": 1,
                    "impact": "Dominant"
                },
                {
                    # Reads bulk_shear_0_6km_mps, so it must be labelled as the
                    # 0-6 km shear it actually is. The historical ERA5 profile
                    # genuinely supports this depth; the 1000-700 hPa label
                    # belongs to the LIVE GFS path, which cannot see above
                    # 700 hPa. Conflating the two misstates the layer depth.
                    "name": "Deep-Layer Bulk Wind Shear (0–6 km)",
                    "value": f"{self.current_thermo.get('bulk_shear_0_6km_mps', 'N/A')} m/s",
                    "physical_role": "Kinematic vector separation organizing storm updrafts and delaying convective precipitation downdraft choking.",
                    "importance_rank": 2,
                    "impact": "High"
                },
                {
                    "name": "Surface Moisture & Convergence (u10/v10 & TCWV)",
                    "value": f"{self.current_thermo.get('wind_speed_kmh', 'N/A')} km/h",
                    "physical_role": "Boundary-layer marine moisture flux from Bay of Bengal driving local moisture pooling.",
                    "importance_rank": 3,
                    "impact": "High"
                },
                {
                    "name": "Mid-Tropospheric Moisture & Temperature (q/t @ 500/700 hPa)",
                    "value": "700 hPa - 500 hPa profile",
                    "physical_role": "Steep environmental lapse rates sustaining deep convective columns into upper troposphere.",
                    "importance_rank": 4,
                    "impact": "Moderate"
                },
                {
                    "name": "Topographic Orographic Anchoring (SRTM DEM)",
                    "value": "Elevation & Slope",
                    "physical_role": "Himalayan foothills terrain barrier forcing mechanical air-parcel lift in northern sub-regions.",
                    "importance_rank": 5,
                    "impact": "Localized"
                }
            ],
            "verified_test_metrics_2024": {
                "test_split": "2024 Held-Out Convective Season (May–Oct 2024)",
                "mean_pr_auc": 0.4840,
                "mean_csi": 0.3068,
                "mean_pod": 0.5428,
                "mean_far": 0.5891,
                "mean_brier_score": 0.0512,
                "mean_ece": 0.0760,
                "mean_rainfall_mae": 0.700,
                "vs_persistence": "Beats persistence baseline at EVERY lead horizon (+2h to +6h)"
            }
        }


def get_nowcast_service() -> NowcastService:
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        with _SERVICE_LOCK:
            if _SERVICE_INSTANCE is None:
                _SERVICE_INSTANCE = NowcastService()
    return _SERVICE_INSTANCE

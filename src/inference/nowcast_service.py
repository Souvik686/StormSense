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
from src.inference import risk_thresholds
from src.utils.config import load_config, Config

_SERVICE_LOCK = threading.Lock()
_SERVICE_INSTANCE: Optional[NowcastService] = None

# Honest caveats about the live pipeline, exposed verbatim by
# /api/nowcast/summary and /api/live/ml-status so both endpoints describe the
# pipeline consistently.

# Historical case study identity. Every label, API field and UI string that
# describes the case study reads from these constants rather than
# hardcoding the event name in multiple places.
HISTORICAL_EVENT_NAME = "Cyclone Remal"
HISTORICAL_EVENT_DETAIL = "Cyclone Remal (Landfall Approach)"
HISTORICAL_ANALYSIS_TIME = "2024-05-26T12:00:00Z"
HISTORICAL_ANALYSIS_SOURCE = "ERA5 reanalysis (observed atmospheric analysis)"


# Demo horizon mapping: the user-facing label -> real model lead used by the
# current deployment.
#
# The buttons read NOW / +2h / +4h / +6h, offsets from the current instant.
# The model forecasts forward from a GFS analysis that is always somewhat in
# the past, so serving lead 2 under the label "+2h from now" would advertise
# a forecast that is actually valid hours earlier. Instead the mapping is
# fixed to four adjacent real leads of the active model:
#
#       LABEL      REAL LEAD      valid at analysis_t0 + lead
#       NOW    ->       3h
#       +2h    ->       4h
#       +4h    ->       5h
#       +6h    ->       6h
#
# This is an application-level mapping only; it does not change any
# scientific evaluation. The backend always retains the real lead and target
# timestamp alongside the label -- see _effective_lead_hours() and the
# `demo_horizon` block returned by the point/summary endpoints.
#
# Keys are the lead the API is called with (0 == NOW); values are the real
# model lead actually served. STORMSENSE_DEMO_HORIZONS=0 disables the mapping
# and serves each requested lead as itself.
DEMO_HORIZON_MAP: Dict[int, int] = {0: 3, 2: 4, 4: 5, 6: 6}

# Optional wall-clock resolution, off by default.
#
# The fixed map above ages with the GFS analysis: since the analysis is
# 3.6-9.6h old depending on where the 6-hourly cycle currently sits, the true
# offset between a label and the current instant drifts as the cycle ages.
# Acceptable because the labels name four successive hourly forecasts rather
# than promising exact offsets, and the true target timestamp is always
# displayed alongside them.
#
# Enabling dynamic mode instead resolves each label to the model lead whose
# valid time is nearest `wall_clock + horizon`, recomputed per request, at
# the cost of the four horizons no longer being adjacent leads.
# src/inference/target_time.py performs the same arithmetic for
# /api/nowcast/wallclock-horizons.
def demo_horizons_dynamic() -> bool:
    """Whether horizon labels track the wall clock (True) or use the fixed map.

    Off by default: STORMSENSE_DEMO_HORIZONS_DYNAMIC=1 switches to resolving
    each label against the current time instead of the fixed lead map.
    """
    v = os.environ.get("STORMSENSE_DEMO_HORIZONS_DYNAMIC", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


# Widened from target_time's 0.5h because these four labels are coarse by
# construction ("+2h" is not a promise of 120.0 minutes) and a 1h tolerance is
# what lets hourly leads cover the whole cycle. The REAL target time is always
# reported alongside, so the residual error is visible rather than hidden.
DEMO_HORIZON_TOLERANCE_HOURS = 1.0

# The label shown to the user for each REQUESTED lead. Deliberately unchanged
# by the mapping: the user sees NOW/+2h/+4h/+6h regardless of which real lead
# backs them.
DEMO_HORIZON_LABELS: Dict[int, str] = {0: "NOW", 2: "+2h", 4: "+4h", 6: "+6h"}


def demo_horizons_enabled() -> bool:
    """Whether the temporary label->lead mapping is active.

    On by default. STORMSENSE_DEMO_HORIZONS=0/false/no turns it off, which makes
    every requested lead serve itself -- the behaviour to use for any scientific
    evaluation, so a backtest can never inherit the demo relabelling.
    """
    v = os.environ.get("STORMSENSE_DEMO_HORIZONS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")

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
    "A GFS f000 analysis contains no precipitation field at all, so the "
    "precipitation input is taken from APCP accumulated over the 6-hour window "
    "ENDING at the analysis time (from the preceding cycle). That window has "
    "already elapsed, so it carries no future information, but it is a 6-hourly "
    "mean rather than an instantaneous rate.",
    "Convective inhibition (CIN) is the most shifted input: GFS and ERA5 store it "
    "with opposite sign conventions, which is corrected, but even corrected the "
    "two agree only weakly (correlation ~0.20, magnitudes differing ~3x).",
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
        self._apply_gfs_threshold_override()

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
        # Which real model lead currently backs the NOW slot, and its TRUE valid
        # time. Populated by refresh_live_state(); see the re-anchoring block
        # there. None until the first successful live refresh.
        self.live_now_anchor: Optional[Dict[str, Any]] = None
        # Provenance of the live `tp` field: which elapsed APCP window and source
        # cycle it came from, or an "unavailable" record. None until first refresh.
        self.live_precip_provenance: Optional[Dict[str, Any]] = None
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
                # is not in the held-out split, fail rather than silently
                # serving a different date under the Remal label.
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
                # (routed through _effective_threshold_for_lead so the
                # instance-local GFS override, when enabled, applies here too)
                if self.predictor.threshold_per_lead is not None:
                    severe_binary = np.zeros_like(severe_prob, dtype=np.uint8)
                    for li, lh in enumerate(self.lead_times):
                        thr = self._effective_threshold_for_lead(lh)
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
            
            # TIMESTAMP SEMANTICS (load-bearing -- do not revert to `t0`).
            #
            # The predictor turns `timestamp` into the model's hour-of-day /
            # day-of-year channels, and it derives the six slots' hours as
            # (ts.hour + [-5..0]) -- i.e. it assumes `timestamp` is the instant
            # the LAST INPUT SLOT was observed. Training builds those channels
            # from the real timestamps of the six input slots
            # (src/data/dataset_v2.py: `times_window = self.times[start:end]`).
            #
            # The six live slots end at `analysis_t0` (the real GFS f000
            # analysis), NOT at `t0` (wall-clock). Passing `t0` told the model
            # the atmosphere it is reading was observed `wallclock_age_hours`
            # later than it actually was -- routinely 6-11h, because of GFS
            # production lag. That desynchronized the diurnal channels from the
            # physical fields and moved the prediction in BOTH directions
            # (measured on the 17 Sep 2026 cached cycles: -1.9 pp to +10.1 pp at
            # Kolkata; domain-max 8.7% -> 85.9% on identical tensors). It also
            # made the map drift on its own between GFS cycles, because the
            # 5-minute refresh re-ran the same tensors with an advancing clock.
            #
            # `analysis_t0` is the only value consistent with how the channels
            # were trained. The wall-clock reference instant is still reported
            # separately as `live_reference_time` -- this changes what the model
            # is TOLD about its input, not what the forecast is issued for.
            harmonized = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0)
            preds_dict = self.predictor.predict(
                surface=harmonized.surface,
                pressure=harmonized.pressure,
                dem=self._get_static_dem_meters(),
                timestamp=harmonized.analysis_t0.isoformat(),
            )

            # NOW re-anchoring: a model whose analysis is `age` hours old cannot
            # observe the present, but it can forecast it, so NOW is served by a
            # real forecast lead rather than by re-running inference.
            #
            # The lead is resolved by _resolve_demo_lead(0), the same source of
            # truth _effective_lead_hours() uses, so the anchor reported here
            # and the lead the point/summary endpoints actually serve can never
            # disagree. Under the fixed demo mapping that is a constant lead;
            # with dynamic resolution enabled it is the lead whose valid time is
            # nearest wall-clock. Only if neither applies does this fall back to
            # choosing the nearest-valid-time lead directly.
            import numpy as np

            model_leads = list(self.predictor.lead_times)
            analysis_t0 = harmonized.analysis_t0
            now_ref = target_t0

            anchor_lead = None
            _resolver = getattr(self, "_resolve_demo_lead", None)
            if demo_horizons_enabled() and callable(_resolver):
                try:
                    _cand = _resolver(0)
                except Exception:
                    _cand = None
                if _cand is not None and _cand in model_leads:
                    anchor_lead = int(_cand)

            if anchor_lead is None:
                # No demo mapping in force: fall back to the lead whose true
                # valid time sits closest to the current instant.
                deltas = [
                    (abs((analysis_t0 + timedelta(hours=L) - now_ref).total_seconds()), L)
                    for L in model_leads
                ]
                _, anchor_lead = min(deltas)
                anchor_lead = int(anchor_lead)

            anchor_idx = model_leads.index(anchor_lead)
            anchor_valid = analysis_t0 + timedelta(hours=anchor_lead)
            anchor_offset_h = (anchor_valid - now_ref).total_seconds() / 3600.0

            self.live_now_anchor = {
                "source_lead_hours": int(anchor_lead),
                "valid_time_utc": anchor_valid.isoformat(),
                "analysis_time_utc": analysis_t0.isoformat(),
                "wall_clock_utc": now_ref.isoformat(),
                "offset_from_wall_clock_hours": round(anchor_offset_h, 3),
                "reaches_wall_clock": bool(anchor_valid >= now_ref),
                "note": (
                    "NOW is a model forecast, not an observation. Served by "
                    f"lead +{anchor_lead}h from analysis "
                    f"{analysis_t0.isoformat()} (available leads {model_leads})."
                ),
            }

            severe_prob = np.concatenate(
                [preds_dict["severe_weather_prob"][anchor_idx:anchor_idx + 1],
                 preds_dict["severe_weather_prob"]], axis=0)
            rain_pred = np.concatenate(
                [preds_dict["rain_3h_mm_pred"][anchor_idx:anchor_idx + 1],
                 preds_dict["rain_3h_mm_pred"]], axis=0)
            severe_binary = np.concatenate(
                [preds_dict["severe_weather_binary"][anchor_idx:anchor_idx + 1],
                 preds_dict["severe_weather_binary"]], axis=0)
            
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
            # Where `tp` came from (elapsed-APCP window + source cycle), so the
            # API can state it rather than implying f000 carried precipitation.
            self.live_precip_provenance = getattr(harmonized, "precip_provenance", None)
            self.live_analysis_cycles = [c.isoformat() for c in harmonized.analysis_cycles]
            self.live_slot_provenance = harmonized.slot_provenance
            self.live_fetch_error = None
            self.live_fetched_at = harmonized.fetched_at.isoformat()

            try:
                wb_mask = get_or_create_wb_mask()
                self.live_risk_surface_png_cache = {}
                # Key by the requested lead but render the lead actually served
                # (the mapping get_risk_surface_png() applies). Using the same
                # _lead_index() resolver as the request path keeps the cached
                # bytes and a cache-miss render consistent.
                for _i, _lh in enumerate(self.predictor.lead_times):
                    # Keyed by (effective lead, array index) -- the same shape
                    # get_risk_surface_png() looks up, so a warm cache and a
                    # cold render can never disagree.
                    self.live_risk_surface_png_cache[("eff", int(_lh), int(_i) + 1)] = (
                        generate_risk_surface_png(
                            severe_prob[_i + 1], self.lats, self.lons, mask=wb_mask)
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
        # Delegates to THE single source of truth (src/inference/risk_thresholds.py)
        # so the popup, the GeoJSON and the painted surface can never drift apart.
        return risk_thresholds.level_for_prob(p)

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

    def model_identity(self) -> Dict[str, Any]:
        """Who the ACTIVE model actually is -- derived, never hardcoded.

        Every page that names the model (AI Nowcast, XAI, benchmark, system
        info) reads this, so the app cannot describe itself as V2 while serving
        V4. Name, lead set and parameter count all come from the loaded
        checkpoint and config, so swapping the checkpoint updates the UI with no
        further edits.

        `deployment_status` states plainly that V4 is an application/demo
        deployment. Per reports/V4_DECISION_CRITERIA.md Rule 2, no V4 lead met
        the operational requirements (POD >= 0.30: 0 of 13 leads; best V4 GFS
        POD 22.89% at +16h), so presenting it as operationally promoted would be
        false. That conclusion is recorded here rather than in a UI string so it
        cannot drift out of sync with the reports.
        """
        leads = list(self.predictor.lead_times)
        ckpt = str(getattr(self.predictor, "checkpoint_path", "") or "")
        is_v4 = len(leads) == 13 and min(leads) == 4 and max(leads) == 16

        if is_v4:
            version, name = "V4", "StormSense V4 Wall-Clock Nowcaster"
        elif leads and max(leads) <= 6:
            version, name = "V2", "StormSense V2 Nowcaster"
        else:
            version, name = "custom", "StormSense Nowcaster"

        return {
            "version": version,
            "model_name": name,
            "architecture": (
                "ConvGRU encoder (surface / wind / thermodynamic branches) with a "
                "DEM-conditioned multi-scale fusion trunk and a shared "
                "multi-horizon decoder head."
            ),
            "checkpoint_path": ckpt,
            "config_path": str(getattr(self.predictor, "config_path", "") or ""),
            "model_lead_times_hours": leads,
            "model_parameters": int(self.predictor.model.count_parameters()),
            "is_calibrated": bool(getattr(self.predictor, "is_calibrated", False)),
            "calibration": "per-lead temperature scaling, fitted on validation only",
            "training_epoch": int(getattr(self.predictor, "training_epoch", -1)),
            "demo_horizon_mapping": (
                {str(k): v for k, v in DEMO_HORIZON_MAP.items()}
                if demo_horizons_enabled() else None
            ),
            # Whether the GFS-domain threshold override (leads 10-15 only) is
            # currently active, and exactly which leads it touched. None when
            # STORMSENSE_GFS_THRESHOLD_OVERRIDE is unset -- the default -- so
            # the checkpoint's original ERA5-fitted thresholds are in force.
            "gfs_threshold_override": getattr(self, "_gfs_threshold_override_applied", None),
            "deployment_status": (
                "application/demo deployment" if is_v4 else "production"
            ),
            "promotion_note": (
                "V4 is deployed here as an application/demo deployment. It did NOT "
                "pass the operational promotion criteria in "
                "reports/V4_DECISION_CRITERIA.md: no V4 lead met Rule 2 "
                "(POD >= 0.30 on held-out 2024 GFS -- 0 of 13 leads; best V4 GFS "
                "POD 22.89% at +16h). V4 was adopted because it is the only model "
                "whose lead set can reach the wall-clock horizons at all, not "
                "because it scored better than V2 on V2's own ground."
                if is_v4 else
                "Incumbent production model."
            ),
        }

    def _is_servable_lead(self, lead_hours: int, mode: Optional[str] = None) -> bool:
        """Can this REQUESTED lead be served at all?

        True for any real model lead, and additionally for the demo-mapped
        request leads (0/2/4/6), which are keys into DEMO_HORIZON_MAP rather
        than model leads -- under V4 the model has no lead 2, yet "+2h" is a
        legitimate request served by the real V4 +6h forecast. The callers'
        `lead not in self.lead_times` guards would otherwise silently coerce it
        to the first model lead and paint the wrong horizon.
        """
        if lead_hours in self.lead_times:
            return True
        if self._resolve_mode(mode) != "live" and lead_hours > 0:
            # Snapped to the nearest real lead by _effective_lead_hours.
            return True
        return (
            self._resolve_mode(mode) == "live"
            and demo_horizons_enabled()
            and int(lead_hours) in DEMO_HORIZON_MAP
            and self._resolve_demo_lead(int(lead_hours)) is not None
        )

    def _model_lead_times(self) -> list:
        """The loaded model's real leads, or [] when no predictor is attached.

        Tolerates objects that hold only timing state (test stubs, partially
        constructed services) so lead resolution degrades instead of raising.
        """
        pred = getattr(self, "predictor", None)
        try:
            return list(getattr(pred, "lead_times", []) or [])
        except Exception:
            return []

    def _resolve_demo_lead(self, requested: int) -> Optional[int]:
        """The model lead that best answers "requested hours from NOW", or None.

        Dynamic path: the lead whose valid time (`analysis + lead`) is nearest
        `wall_clock + requested`, within DEMO_HORIZON_TOLERANCE_HOURS. Recomputed
        per request, so the label keeps tracking the clock as the analysis ages
        instead of drifting with it.

        Falls back to the fixed DEMO_HORIZON_MAP when dynamic resolution is off,
        when there is no live analysis to measure against, or when no lead lands
        within tolerance -- so a horizon always resolves to something real.
        """
        model_leads = self._model_lead_times()
        fixed = DEMO_HORIZON_MAP.get(int(requested))
        fallback = fixed if (fixed in model_leads) else None

        if not demo_horizons_dynamic() or not model_leads:
            return fallback
        if not self.live_analysis_time:
            return fallback
        try:
            analysis = datetime.fromisoformat(
                self.live_analysis_time.replace("Z", "+00:00"))
        except Exception:
            return fallback

        now = datetime.now(timezone.utc)
        target = now + timedelta(hours=float(requested))
        # Distance of each lead's TRUE valid time from the target instant.
        best, best_err = None, None
        for L in model_leads:
            err = abs(((analysis + timedelta(hours=float(L))) - target).total_seconds() / 3600.0)
            if best_err is None or err < best_err:
                best, best_err = int(L), err
        if best is None or best_err > DEMO_HORIZON_TOLERANCE_HOURS:
            return fallback
        return best

    def _nearest_model_lead(self, lead_hours: int) -> int:
        """The real model lead closest to `lead_hours`, excluding the NOW slot.

        Used in HISTORICAL mode, where the demo mapping deliberately does not
        apply (a frozen replay has no wall-clock to anchor to) but the requested
        horizon may still not be a model lead -- V4 has no +2h or +3h at all.
        Snapping to the nearest real lead keeps the case study usable at the
        classic +2/+4/+6 horizons instead of collapsing every one of them to the
        t0 analysis, and the served lead is always reported back so the
        substitution is visible rather than silent.
        """
        real = [L for L in self.lead_times if L != 0]
        if not real:
            return 0
        return min(real, key=lambda L: (abs(L - lead_hours), L))

    # GFS-domain decision-threshold override for leads 10-15.
    #
    # The checkpoint's thresholds (0.559-0.650) were fitted on ERA5 validation,
    # but GFS input shifts key variables enough that the ERA5-optimal operating
    # point is not the GFS-optimal one. A refit against held-out GFS data
    # (fit on 2023, scored on untouched 2024 -- see
    # reports/v4_gfs_threshold_fit_diag.json) roughly doubles mean POD.
    #
    # Restricted to these six leads because leads 5 and 6 fit to a near-zero
    # threshold (0.01, effectively always-positive) and leads 4, 7, 8, 9 and 16
    # showed negligible or negative CSI change under the refit. Leads 4-9 and
    # 16 keep the checkpoint's original ERA5-fitted thresholds.
    _GFS_THRESHOLD_OVERRIDE = {10: 0.03, 11: 0.16, 12: 0.43, 13: 0.25, 14: 0.29, 15: 0.28}

    def _apply_gfs_threshold_override(self) -> None:
        """Compute the override map without touching self.predictor.

        `self.predictor` is a module-level singleton shared by every
        NowcastService instance and by any evaluation script that loads the
        same checkpoint. Reassigning its `threshold_per_lead` dict would leak
        the override into every other consumer of the cache, including
        backtests that must run against the checkpoint's real thresholds. The
        override therefore lives only in `self._gfs_threshold_override_applied`
        on this instance; `_effective_threshold_for_lead()` below checks it
        first and falls back to the read-only `self.predictor.threshold_per_lead`.
        """
        self._gfs_threshold_override_applied: Dict[int, float] = {}
        if os.environ.get("STORMSENSE_GFS_THRESHOLD_OVERRIDE", "0").strip().lower() not in ("1", "true", "yes", "on"):
            return
        if not isinstance(getattr(self.predictor, "threshold_per_lead", None), dict):
            return
        applied = {
            lead: float(thr) for lead, thr in self._GFS_THRESHOLD_OVERRIDE.items()
            if lead in self.predictor.lead_times
        }
        self._gfs_threshold_override_applied = applied
        print(f"[NowcastService] GFS-domain threshold override ACTIVE for leads "
              f"{sorted(applied.keys())}: {applied}. Diagnostic fit: "
              f"reports/v4_gfs_threshold_fit_diag.json. Leads outside this set "
              f"keep the checkpoint's original ERA5-fitted thresholds. This "
              f"instance only -- the shared predictor cache is untouched.")

    def _effective_threshold_for_lead(self, lead_hours: int) -> float:
        """The decision threshold for `lead_hours`, INSTANCE-LOCAL override
        first, then the checkpoint's own per-lead threshold, then its scalar
        fallback. This is the one place threshold_per_lead should be read from
        for any code that must respect the GFS override; call sites that read
        `self.predictor.threshold_per_lead` directly do not see it."""
        override = getattr(self, "_gfs_threshold_override_applied", None)
        if override and int(lead_hours) in override:
            return override[int(lead_hours)]
        tpl = getattr(self.predictor, "threshold_per_lead", None)
        fallback = float(getattr(self.predictor, "threshold", 0.5) or 0.5)
        if isinstance(tpl, dict):
            return float(tpl.get(lead_hours, tpl.get(str(lead_hours), fallback)))
        return fallback

    def _lead_index(self, lead_hours: int, mode: Optional[str] = None) -> int:
        """Index into the prediction arrays for the lead ACTUALLY served.

        THE single place a requested lead becomes an array index. Every consumer
        (summary, risk map, risk surface, point inspection, district advisories,
        high-risk cells, XAI) goes through this, so the demo horizon mapping and
        the NOW re-anchoring can never apply to a timestamp while the array
        index still points at the unmapped lead -- which is exactly how a map
        ends up painted with one horizon while its caption states another.

        Falls back to the requested lead's own index, then to index 0, so an
        unmappable request degrades instead of raising.
        """
        eff = self._effective_lead_hours(lead_hours, mode)
        if eff in self.lead_times:
            return self.lead_times.index(eff)
        if lead_hours in self.lead_times:
            return self.lead_times.index(lead_hours)
        return 0

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

    def live_now_provenance(self) -> Dict[str, Any]:
        """What the LIVE lead=0 ("NOW") field actually is, stated plainly.

        NOW is not a true t=0 analysis-time diagnosis. The model has no lead-0
        head, so NOW is produced by running inference on the input window ending
        2 hours before the forecast window and taking that run's +2h output
        (`refresh_live_state`). In Historical mode that works exactly as
        intended, because ERA5 is HOURLY: `dataset[idx-2]` really is two hours
        earlier, and its +2h head lands precisely on the case-study t0.

        Live GFS is different, and this is the honest caveat: f000 analyses are
        published only every 6 hours, so `target_t0 - 2h` usually floors to the
        SAME cycle as `target_t0`. Measured over a full day, the two fetches
        resolve to the same analysis for 16 of 24 wall-clock hours, and in those
        hours the NOW field is bit-identical to the +2h field (verified: equal
        SHA-256 over the 33x25 array). Its true validity is
        `analysis_time + 2h`, which is typically 5-10 h BEHIND wall clock.

        Rather than fabricate a t=0 field the data cannot support, the API
        reports this so the UI can label NOW truthfully.
        """
        out: Dict[str, Any] = {
            "is_true_t0_analysis": False,
            "construction": (
                "Inference on the input window ending 2 h before the forecast "
                "window; its +2 h head is served as NOW. The model has no lead-0 head."
            ),
            "analysis_time_utc": self.live_analysis_time,
            "reference_time_utc": self.live_reference_time,
            "field_valid_time_utc": None,
            "identical_to_plus_2h": None,
            "caveat": None,
        }
        try:
            if self.live_analysis_time:
                a = datetime.fromisoformat(self.live_analysis_time)
                if a.tzinfo is None:
                    a = a.replace(tzinfo=timezone.utc)
                vt = a + timedelta(hours=2)
                out["field_valid_time_utc"] = vt.isoformat()
                if self.live_reference_time:
                    r = datetime.fromisoformat(self.live_reference_time)
                    if r.tzinfo is None:
                        r = r.replace(tzinfo=timezone.utc)
                    out["field_age_vs_reference_hours"] = round(
                        (r - vt).total_seconds() / 3600.0, 2
                    )
            pred = self.live_pred
            if pred is not None:
                sp = pred["severe_weather_prob"]
                if sp.shape[0] >= 2:
                    out["identical_to_plus_2h"] = bool(
                        np.array_equal(sp[0], sp[1])
                    )
            if out["identical_to_plus_2h"]:
                out["caveat"] = (
                    "This NOW field is identical to the +2 h field. GFS publishes "
                    "analyses only every 6 h, so the 2-hour-earlier input window "
                    "resolved to the same analysis cycle. Read NOW as the earliest "
                    "available forecast step, not as a real-time observation."
                )
            else:
                out["caveat"] = (
                    "NOW derives from an earlier analysis cycle than the forecast "
                    "window, so it is a distinct field; its validity is still "
                    "analysis time + 2 h, not wall clock."
                )
        except Exception as e:
            out["caveat"] = f"Provenance could not be fully determined: {e}"
        return out

    def _issue_time_for_mode(self, mode: Optional[str]) -> str:
        """The authoritative reference instant that +2/+4/+6 are measured from.

        This is the ANALYSIS time -- the instant the atmospheric state the model
        read was actually observed -- in both modes. A forecast's valid time is
        `analysis_t0 + lead`, never `wall_clock + lead`: the model integrates
        forward from the state it was given, and GFS analyses are routinely
        4-10h old (production lag), so the two differ by that age. The
        wall-clock reference instant remains available as `live_reference_time`
        for provenance; it is not a valid-time basis."""
        m = self._resolve_mode(mode)
        if m == "live":
            return self.live_analysis_time or self.live_reference_time or "Unknown"
        return self.current_valid_time

    def _effective_lead_hours(self, lead_hours: int, mode: Optional[str] = None) -> int:
        """Map a REQUESTED lead to the real model lead whose output is served.

        Only lead 0 (NOW) is remapped, and only in live mode, where NOW is
        re-anchored onto the model lead whose valid time is nearest wall-clock
        (see refresh_live_state). Every other lead is its own model lead. Valid
        times must be computed from THIS value, or NOW would advertise
        `analysis_t0 + 0` -- the analysis instant -- while actually serving a
        forecast valid hours later.

        Historical mode keeps lead 0 == the case study's t0 analysis: that mode
        replays a frozen event, so there is no wall-clock to re-anchor to.

        DEMO HORIZON MAPPING takes precedence when enabled: the four user-facing
        labels NOW/+2h/+4h/+6h are served by real V4 leads 5/6/7/8 (see
        DEMO_HORIZON_MAP). It applies in LIVE mode only -- historical mode is a
        frozen replay whose horizons are genuine offsets from the case study's
        own analysis time, so relabelling them there would corrupt the replay.

        Everything downstream (valid times, per-lead decision thresholds, the
        risk surface, alerts) computes from THIS return value, so the real lead
        is what actually drives the product while the label stays fixed."""
        is_live = self._resolve_mode(mode) == "live"

        if is_live and demo_horizons_enabled():
            mapped = DEMO_HORIZON_MAP.get(int(lead_hours))
            # Only honour the mapping if the loaded model really has that lead;
            # otherwise fall through rather than serve a lead that does not
            # exist. getattr(), because callers legitimately exercise this
            # method on objects that carry only the timing state (the anchoring
            # tests drive it with a stub that has no predictor at all) -- the
            # mapping must degrade there, not raise.
            # Inlined rather than delegated so this method has NO dependency on
            # sibling attributes beyond `predictor`: the anchoring tests borrow
            # it as an unbound function onto a minimal stub, which is a useful
            # property to keep (it pins the timing logic without loading torch).
            try:
                _model_leads = list(getattr(getattr(self, "predictor", None),
                                            "lead_times", []) or [])
            except Exception:
                _model_leads = []
            if mapped is not None and _model_leads:
                # Prefer the DYNAMIC lead (nearest valid time to wall_clock +
                # horizon) so the label keeps meaning what it says as the
                # analysis ages; _resolve_demo_lead falls back to `mapped`.
                # getattr keeps this working on the minimal stub the anchoring
                # tests drive, which has no such method.
                _dyn = None
                _resolver = getattr(self, "_resolve_demo_lead", None)
                if callable(_resolver):
                    try:
                        _dyn = _resolver(int(lead_hours))
                    except Exception:
                        _dyn = None
                if _dyn is not None and _dyn in _model_leads:
                    return int(_dyn)
                if mapped in _model_leads:
                    return int(mapped)

        if lead_hours != 0:
            # Historical mode: the demo mapping does not apply, but the request
            # may still name a horizon this model has no lead for. Snap to the
            # nearest real lead rather than returning a lead that does not exist
            # (which _lead_index would then fall back to index 0 for, silently
            # painting the analysis instant under a "+4h" label).
            if not is_live and lead_hours not in getattr(self, "lead_times", [lead_hours]):
                return self._nearest_model_lead(lead_hours)
            return lead_hours
        if not is_live:
            return 0
        anchor = getattr(self, "live_now_anchor", None)
        if anchor and anchor.get("source_lead_hours") is not None:
            return int(anchor["source_lead_hours"])
        return 0

    def demo_horizon_info(self, lead_hours: int, mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Full provenance for a demo-mapped horizon, or None when not mapped.

        Reports the user-facing label, the real V4 lead behind it, the analysis
        time, the TRUE forecast target timestamp, and how far that target sits
        from the current instant. This is what keeps Part 4's requirement --
        real timestamps internally -- checkable rather than asserted: a consumer
        can always recover exactly which lead produced the number on screen.
        """
        if self._resolve_mode(mode) != "live" or not demo_horizons_enabled():
            return None
        if int(lead_hours) not in DEMO_HORIZON_MAP:
            return None
        # The lead ACTUALLY served -- dynamic when available, fixed otherwise.
        mapped = self._resolve_demo_lead(int(lead_hours))
        if mapped is None or mapped not in self._model_lead_times():
            return None

        analysis_iso = self.live_analysis_time
        target_iso = None
        age_h = None
        offset_h = None
        now = datetime.now(timezone.utc)
        if analysis_iso:
            try:
                a = datetime.fromisoformat(analysis_iso.replace("Z", "+00:00"))
                target = a + timedelta(hours=mapped)
                target_iso = target.isoformat()
                age_h = round((now - a).total_seconds() / 3600.0, 3)
                offset_h = round((target - now).total_seconds() / 3600.0, 3)
            except Exception:
                pass

        return {
            "enabled": True,
            "requested_lead_hours": int(lead_hours),
            "user_facing_label": DEMO_HORIZON_LABELS.get(int(lead_hours), f"+{lead_hours}h"),
            "actual_model_lead_hours": int(mapped),
            "analysis_time_utc": analysis_iso,
            "analysis_age_hours": age_h,
            "forecast_target_time_utc": target_iso,
            "target_offset_from_now_hours": offset_h,
            "mapping": {str(k): v for k, v in DEMO_HORIZON_MAP.items()},
            # THE TIME THE UI SHOWS: exactly `now + horizon`.
            #
            # The user reads "+2h" as "two hours from the current clock", so the
            # headline timestamp is computed from the clock, not from the model
            # grid. The model cannot produce a forecast valid at an arbitrary
            # instant -- V4's leads are whole hours from a :00 analysis, so its
            # valid times land on the hour -- and `forecast_target_time_utc`
            # above remains that REAL instant. The two are reported separately
            # and `display_vs_target_minutes` is the gap between them, so the
            # headline can be exact without the underlying forecast being
            # misrepresented as valid at a time it is not.
            "display_time_utc": (now + timedelta(hours=float(lead_hours))).isoformat(),
            "display_offset_hours": float(lead_hours),
            "display_vs_target_minutes": (
                round((datetime.fromisoformat(target_iso)
                       - (now + timedelta(hours=float(lead_hours)))).total_seconds() / 60.0)
                if target_iso else None
            ),
            "resolution": "dynamic" if demo_horizons_dynamic() else "fixed",
            "fixed_fallback_lead_hours": DEMO_HORIZON_MAP.get(int(lead_hours)),
            "label_error_hours": offset_h - float(lead_hours) if offset_h is not None else None,
            "note": (
                "Application/demo horizon mapping. The label is a fixed UI string; "
                f"the value served is the real V4 +{mapped}h forecast valid at "
                f"{target_iso}. This is not a claim that this lead met the "
                "operational promotion criteria in reports/V4_DECISION_CRITERIA.md."
            ),
        }

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
            "precip_provenance": getattr(self, "live_precip_provenance", None),
        }

    # User-facing horizons, in hours from the CURRENT INSTANT (not from the
    # analysis). These are what the UI offers; whether each can actually be
    # honoured depends on how old the analysis is and which leads the loaded
    # checkpoint provides -- see resolve_wallclock_horizons().
    WALLCLOCK_HORIZONS_HOURS = [0, 2, 4, 6]

    def resolve_wallclock_horizons(
        self, wall_clock: Optional[datetime] = None, allow_interpolation: bool = False
    ) -> Dict[str, Any]:
        """Which user-facing NOW/+2/+4/+6 horizons the live pipeline can honestly
        serve right now, and from which model lead.

        This is the honest answer to "does '+2 HOURS' mean two hours from now?".
        A horizon is reported available only when some model lead's TRUE valid
        time (`analysis_time + lead`) lands within the labelling tolerance of
        `wall_clock + horizon`. Otherwise it is reported unavailable WITH a
        reason, and must be shown that way rather than filled with a neighbouring
        lead under the wrong label.

        Returns {} when there is no live state to reason about.
        """
        from src.inference.target_time import resolve_all

        if wall_clock is None:
            wall_clock = datetime.now(timezone.utc)
        if not self.live_analysis_time:
            return {
                "available": False,
                "reason": "No live GFS analysis has been loaded yet.",
                "horizons": {},
            }
        try:
            analysis = datetime.fromisoformat(self.live_analysis_time)
        except Exception as e:
            return {"available": False, "reason": f"Unparseable analysis time: {e}",
                    "horizons": {}}

        # The model's own leads, straight from the loaded checkpoint. Lead 0 in
        # self.lead_times is the service's synthetic NOW slot, not a real model
        # lead, so it is excluded here.
        model_leads = [float(L) for L in self.predictor.lead_times]
        resolved = resolve_all(
            wall_clock, analysis, self.WALLCLOCK_HORIZONS_HOURS, model_leads,
            allow_interpolation=allow_interpolation,
        )
        out = {k: v.to_dict() for k, v in resolved.items()}
        n_ok = sum(1 for v in resolved.values() if v.available)
        return {
            "available": n_ok > 0,
            "n_supported": n_ok,
            "n_requested": len(resolved),
            "model_leads_hours": model_leads,
            "analysis_time_utc": self.live_analysis_time,
            "wall_clock_utc": wall_clock.isoformat(),
            "analysis_age_hours": round(
                (wall_clock - analysis).total_seconds() / 3600.0, 3),
            "horizons": out,
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

        if not self._is_servable_lead(lead_hours, mode):
            lead_hours = self.lead_times[0]
        li = self._lead_index(lead_hours, mode)

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
            # "Whole State" must mean WEST BENGAL, not the whole model domain.
            #
            # The model grid spans 20-28N / 84-90E, which deliberately includes
            # synoptic context OUTSIDE West Bengal (Odisha, Jharkhand, Bihar,
            # Bangladesh, Nepal). Taking a plain .max() over that array reported
            # a peak from a neighbouring state under a card labelled "WEST
            # BENGAL PEAK". Measured on live 2026-09-18 12Z: the thunderstorm
            # card read 31% from a cell at 21.00N, 85.25E (Odisha) while the
            # true West Bengal peak was 7.83% -- so the dashboard showed an
            # amber 31% while the risk map was correctly all-green, which is
            # exactly the contradiction that prompted this audit.
            #
            # `_wb_cell_mask()` is the same authoritative outer boundary used to
            # reject out-of-state map clicks, so the number on the card and the
            # colour on the map are now derived from the same geography.
            wb_mask = self._wb_cell_mask()
            if wb_mask is not None and wb_mask.any():
                dist_prob = float(prob_grid[wb_mask].max())
                dist_rain = float(rain_grid[wb_mask].max())
                dist_ff = float(ff_grid[wb_mask].max())
                dist_overall = float(overall_grid[wb_mask].max())
                dist_prob_mean = float(prob_grid[wb_mask].mean())
            else:
                # Boundary unavailable: fall back to the full domain rather than
                # returning nothing, and say so instead of silently mislabelling.
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
            vt_step = compute_valid_time(issue_time, self._effective_lead_hours(int(h), mode))
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

        forecast_vt = compute_valid_time(issue_time, self._effective_lead_hours(lead_hours, mode))
        return {
            "model": "StormSense AI Forecast",
            # The analysis the input state came from -- not the issue time, and
            # mode-specific, so live pipeline state can never leak into a
            # frozen historical replay. Historical: the case study's own
            # reanalysis analysis time. Live: the real GFS f000 analysis time.
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
            # Which real model lead backs the NOW slot, and its TRUE valid time.
            # Live-only: historical NOW is the case study's t0 analysis and is
            # never re-anchored. Consumers must use this to label NOW as a
            # forecast valid at `valid_time_utc`, not as an observation of the
            # present instant.
            "now_anchor": (
                self.live_now_anchor if resolved_mode == "live" else None
            ),
            # Which real V4 lead and TRUE target timestamp back the requested
            # horizon label. None when the demo mapping is off or in historical
            # mode. See DEMO_HORIZON_MAP.
            "demo_horizon": self.demo_horizon_info(lead_hours, mode),
            "effective_model_lead_hours": int(self._effective_lead_hours(lead_hours, mode)),
            "model_identity": self.model_identity(),
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
                    "alert_active": bool(dist_prob >= self._effective_threshold_for_lead(lead_hours)),
                    "threshold_used": self._effective_threshold_for_lead(lead_hours),
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
        if not self._is_servable_lead(lead_hours, mode):
            lead_hours = self.lead_times[0]
        li = self._lead_index(lead_hours, mode)

        return predictions_to_geojson(
            pred,
            valid_time=self._issue_time_for_mode(mode),
            lead_time_hours=lead_hours,
            lead_idx=li,
            as_polygon=as_polygon,
            # Live: the real GFS f000 analysis the input came from, which lags
            # the issue time by the production lag. Historical: the case study's
            # analysis time IS its issue time, so the two coincide and no
            # separate age is reported.
            analysis_time=(
                self.live_analysis_time
                if self._resolve_mode(mode) == "live"
                else None
            ),
        )

    def get_risk_surface_png(self, lead_hours: int = 2, mode: Optional[str] = None) -> bytes:
        """Return pre-rendered PNG bytes for the continuous West Bengal risk
        surface. Both modes render through the same masked generator, so the
        West Bengal clip is identical for Live and Historical."""
        resolved = self._resolve_mode(mode)
        cache = self.live_risk_surface_png_cache if resolved == "live" else self.risk_surface_png_cache
        if not self._is_servable_lead(lead_hours, mode):
            lead_hours = self.lead_times[0]

        # Key the cache by the lead ACTUALLY SERVED, not by the requested one.
        #
        # Under dynamic resolution the lead behind a label changes as the
        # analysis ages (a "+2h" served by lead 6 at 00:10Z is served by lead 7
        # an hour later), so a cache keyed by the REQUEST would keep returning
        # the first lead's image under a label that has since moved. Keying by
        # the effective lead makes a stale entry impossible: a changed mapping
        # simply misses and renders the correct field.
        li = self._lead_index(lead_hours, mode)
        cache_key = ("eff", int(self._effective_lead_hours(lead_hours, mode)), int(li))
        if cache_key in cache:
            return cache[cache_key]
        pred = self._pred_for_mode(mode)
        grid = pred["severe_weather_prob"][li]
        png = generate_risk_surface_png(grid, self.lats, self.lons, mask=get_or_create_wb_mask())
        cache[cache_key] = png
        return png

    # Historical t=0 analysis surface -- the "NOW" state of the case study.
    #
    # The correct NOW field for a historical replay is the observed reanalysis
    # state at the case study's analysis time (t=0), already loaded in
    # era5_surface_t0. It is an analysis, not a prediction, so it is rendered
    # with the observation palette and labelled as an analysis. The model's
    # forecast fields (leads 2-6) are never reused for t=0.
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

    def _wb_cell_mask(self) -> Optional[np.ndarray]:
        """Boolean (n_lat, n_lon) mask of model cells that lie inside West Bengal.

        Used so that any "whole state" aggregate describes West Bengal and not
        the wider 20-28N / 84-90E model domain, which also covers Odisha,
        Jharkhand, Bihar, Bangladesh and Nepal.

        A cell counts as inside when its centre is within the outer state
        boundary, or within half a grid step (0.125 deg) of it -- the same
        tolerance `_init_district_mapping` uses, so a coastal or border cell
        whose centre falls just outside the polygon still contributes. Returns
        None when the boundary cannot be loaded, so callers can degrade
        explicitly rather than silently aggregating the wrong area.
        """
        if getattr(self, "_wb_cell_mask_cache", None) is not None:
            return self._wb_cell_mask_cache
        if self._wb_polygon is None:
            # Reuse the same loader/caching path as the click test.
            self._is_inside_west_bengal(float(self.lats[0]), float(self.lons[0]))
        poly = self._wb_polygon
        if poly is None:
            return None
        mask = np.zeros((len(self.lats), len(self.lons)), dtype=bool)
        for i, la in enumerate(self.lats):
            for j, lo in enumerate(self.lons):
                p = Point(float(lo), float(la))
                if poly.contains(p) or poly.distance(p) < 0.125:
                    mask[i, j] = True
        self._wb_cell_mask_cache = mask
        return mask

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
        if not self._is_servable_lead(lead_hours, mode):
            lead_hours = self.lead_times[0]
        li = self._lead_index(lead_hours, mode)

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
        lbl = risk_thresholds.stage_for_prob(sp)

        # DISPLAY BAND vs DECISION THRESHOLD -- these are different quantities and
        # on GFS input they can disagree visibly.
        #
        # The colour bands are fixed at 0.25/0.50/0.75 (risk_thresholds.py). The
        # model's own per-lead operating point is a separate number, and once
        # thresholds are fitted in the GFS domain some of them fall BELOW 0.25:
        # measured on held-out 2024 GFS with GFS-refit thresholds, 20.5% of
        # alerting cells at +8h (16.6% at +10h) had p < 0.25 and were therefore
        # painted "Normal" green while the model was in fact detecting an event.
        #
        # Rather than bend the display bands -- which would desynchronise the map,
        # the legend, the GeoJSON and the popup, all of which share those edges --
        # the alert state is reported explicitly alongside the band so a caller can
        # never mistake "green" for "not alerting".
        # NOW is served by a re-anchored real model lead in live mode, so the
        # threshold must be looked up for THAT lead, not for the literal 0.
        # Routed through _effective_threshold_for_lead so the instance-local
        # GFS threshold override (leads 10-15, opt-in) applies here too.
        _eff_lead = self._effective_lead_hours(lead_hours, mode)
        _thr = self._effective_threshold_for_lead(_eff_lead)
        _alert = bool(sp >= _thr)

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

        # Dynamic Valid Time = analysis_t0 + the EFFECTIVE lead actually served.
        # For NOW (lead 0) in live mode the effective lead is the re-anchored
        # model lead, so the reported valid time matches the field returned
        # rather than the analysis instant.
        _eff_lead = self._effective_lead_hours(lead_hours, mode)
        clean = issue_time.replace("Z", "+00:00").replace(" ", "T")
        issue_dt = datetime.fromisoformat(clean)
        valid_dt = issue_dt + timedelta(hours=_eff_lead)
        valid_formatted = valid_dt.strftime("%d %b %Y %H:%M UTC")

        return {
            "lat": float(round(lat, 4)),
            "lon": float(round(lon, 4)),
            "inside_monitored_region": True,
            "mode": resolved_mode,
            "district": district_name,
            "grid_cell": {"lat": round(cell_lat, 2), "lon": round(cell_lon, 2)},
            "lead_hours": lead_hours,
            # The real model lead behind this value. Differs from `lead_hours`
            # only for NOW in live mode, where it names the re-anchored lead.
            "effective_model_lead_hours": int(_eff_lead),
            "issue_time_utc": issue_dt.strftime("%d %b %Y %H:%M UTC"),
            "forecast_valid_utc": valid_formatted,
            "now_anchor": (
                self.live_now_anchor
                if (lead_hours == 0 and resolved_mode == "live") else None
            ),
            "demo_horizon": self.demo_horizon_info(lead_hours, mode),
            "model_name": "StormSense AI Forecast",
            "predictions": {
                "thunderstorm_prob_pct": float(round(sp * 100, 1)),
                "heavy_rain_mm": float(round(rp, 1)),
                "flash_flood_proxy_pct": float(round(ff * 100, 1)),
                "overall_risk_pct": float(round(ov * 100, 1)),
                "risk_level": lvl,
                "risk_label": lbl,
                # The model's own detection decision for this cell, reported
                # separately from the colour band. `risk_level` can read "green"
                # while `alert_active` is true, because the display bands are
                # fixed at 0.25/0.50/0.75 while the per-lead operating point is
                # whatever validation fitted -- on GFS some of those land below
                # 0.25. Consumers that care whether the model is detecting an
                # event must read this, not the colour.
                "alert_active": _alert,
                "decision_threshold": float(round(_thr, 4)),
                "threshold_lead_hours": int(_eff_lead),
                "band_edges": [risk_thresholds.WATCH_MIN,
                               risk_thresholds.ALERT_MIN,
                               risk_thresholds.WARNING_MIN],
            },
            "observed_inputs": obs_inputs,
        }

    def get_district_advisories(
        self, lead_hours: int = 2, mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Calculate real model-derived risk aggregated across districts."""
        pred = self._pred_for_mode(mode)
        issue_time = self._issue_time_for_mode(mode)
        if not self._is_servable_lead(lead_hours, mode):
            lead_hours = self.lead_times[0]
        li = self._lead_index(lead_hours, mode)
        _eff_lead = self._effective_lead_hours(lead_hours, mode)

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
            # BUG FIXED: this looked up the threshold by the RAW requested
            # `lead_hours` (0/2/4/6 under the demo mapping), not by `_eff_lead`
            # -- the real V4 lead the risk grids above are actually drawn from
            # (via `li = self._lead_index(lead_hours, mode)`). For NOW
            # (lead_hours=0) that key does not exist in threshold_per_lead at
            # all, so it silently fell back to a flat 0.5 instead of the real
            # per-lead calibrated threshold (0.559-0.650), which could disagree
            # with the risk-map/point-inspection alert state for the SAME cell
            # and SAME selected horizon -- the dashboard-vs-advisories mismatch.
            # Also routed through _effective_threshold_for_lead so the
            # instance-local GFS override (leads 10-15) applies here too.
            thr = self._effective_threshold_for_lead(_eff_lead)
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
                "valid_time": compute_valid_time(issue_time, _eff_lead),
                "valid_until": format_iso_time(compute_valid_time(issue_time, _eff_lead)),
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
        if not self._is_servable_lead(lead_hours, mode):
            lead_hours = self.lead_times[0]
        li = self._lead_index(lead_hours, mode)

        _eff_lead = self._effective_lead_hours(lead_hours, mode)
        prob_grid = pred["severe_weather_prob"][li]
        rain_grid = pred["rain_3h_mm_pred"][li]
        ff_grid = pred["flash_flood_risk"][li]
        overall_grid = pred["overall_risk"][li]
        # Same fix as get_district_advisories: look up the threshold by the
        # EFFECTIVE lead the grids above were drawn from, not the raw request.
        thr = self._effective_threshold_for_lead(_eff_lead)

        # Rank cells INSIDE WEST BENGAL only. The model grid spans 20-28N/84-90E
        # and also covers Odisha, Jharkhand, Bihar, Bangladesh and Nepal, so an
        # unrestricted ranking filled the "predicted high-risk ML cells" list
        # with neighbouring-state cells labelled "Domain Cell" (measured: the
        # top entries read 31.0% / 30.1% / 27.5% from Odisha while the true West
        # Bengal peak was 7.83%). That made the panel contradict every other
        # West-Bengal-scoped number on the page.
        wb_mask = self._wb_cell_mask()
        cells_data = []
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                if wb_mask is not None and not wb_mask[i, j]:
                    continue
                p = float(prob_grid[i, j])
                r = float(rain_grid[i, j])
                ff = float(ff_grid[i, j])
                ov = float(overall_grid[i, j])
                cells_data.append((p, r, ff, ov, i, j, lat, lon))

        cells_data.sort(key=lambda x: -x[0])

        top_cells = []
        vt = compute_valid_time(issue_time, _eff_lead)
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

        This is rule-based, physics-inspired attribution: scores are computed
        by explicit formulas over the atmospheric variables actually fed to the
        model, normalized to relative contributions. It is not SHAP, not
        gradient/saliency-based, and not neural-network feature importance --
        no such claim is made anywhere in the response.

        The inputs are read from the real live input tensor at the requested
        location.
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
            lh = lead_hours if self._is_servable_lead(lead_hours, "live") else self.lead_times[0]
            # Resolve through the SAME mapping every other endpoint uses, so the
            # explanation is built from the identical model output the map and
            # the cards are showing -- including the demo horizon mapping, where
            # the label "+2h" is backed by the real V4 +6h forecast.
            li = self._lead_index(lh, "live")
            _eff_lh = self._effective_lead_hours(lh, "live")
            if lat is not None and lon is not None:
                i = int(np.argmin(np.abs(self.lats - lat)))
                j = int(np.argmin(np.abs(self.lons - lon)))
                loc_desc = f"{float(self.lats[i]):.2f}N, {float(self.lons[j]):.2f}E"
            else:
                # The default cell must be the highest-risk cell IN WEST BENGAL,
                # for the same reason the hazard cards are WB-masked: the model
                # grid spans 20-28N/84-90E and also covers Odisha, Jharkhand,
                # Bihar, Bangladesh and Nepal. A plain domain-wide argmax made
                # the XAI panel explain a cell in Odisha (measured: 21.00N,
                # 85.25E at 31.0%) while every hazard card described West
                # Bengal's 8% peak -- so the "explanation" belonged to a
                # different place than the thing being explained.
                prob = self.live_pred["severe_weather_prob"][li]
                wb_mask = self._wb_cell_mask()
                if wb_mask is not None and wb_mask.any():
                    masked = np.where(wb_mask, prob, -np.inf)
                    i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
                    scope = "West Bengal"
                else:
                    i, j = np.unravel_index(int(np.argmax(prob)), prob.shape)
                    scope = "model domain"
                i, j = int(i), int(j)
                loc_desc = (
                    f"highest-risk cell in {scope} "
                    f"({float(self.lats[i]):.2f}N, {float(self.lons[j]):.2f}E)"
                )

            s = self.live_surface_state  # real, denormalized t0 GFS analysis state

            # HORIZON-SPECIFICITY.
            #
            # The thermodynamic fields above are the ANALYSIS state: one instant,
            # shared by every lead. Attribution built from them alone is identical
            # at +2h and +6h, which contradicts the contract that the explanation
            # belongs to the SELECTED horizon.
            #
            # The model's own `rain_3h_mm` head is a genuine per-lead output, so
            # the rainfall factor is taken from the forecast AT THIS LEAD rather
            # than from analysis-time rain. That makes the attribution move with
            # the horizon using a real model quantity -- not a reweighting
            # invented to manufacture variation.
            #
            # The remaining factors stay analysis-based and are labelled as such
            # in `factor_basis` below, because the model exposes no per-lead CAPE,
            # humidity or wind head. Inventing per-lead values for them would be
            # fabrication.
            rain_fcst = float(self.live_pred["rain_3h_mm_pred"][li, i, j])
            xai_inputs = {
                # 3h accumulation predicted for this lead; the 1h/6h entries are
                # the same quantity rescaled, matching the factor's formula.
                "rainfall_1h_mm": rain_fcst / 3.0,
                "rainfall_3h_mm": rain_fcst,
                "rainfall_6h_mm": rain_fcst * 2.0,
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
            # The real V4 lead the attribution was computed from. Differs from
            # `lead_hours` whenever the demo horizon mapping is in force.
            out["effective_model_lead_hours"] = int(_eff_lh)
            out["demo_horizon"] = self.demo_horizon_info(lh, "live")
            out["model_identity"] = self.model_identity()
            out["grid_cell"] = {"lat": round(float(self.lats[i]), 2), "lon": round(float(self.lons[j]), 2)}
            out["predicted_severe_prob_pct"] = round(
                float(self.live_pred["severe_weather_prob"][li, i, j]) * 100.0, 1
            )
            out["input_analysis_time"] = self.live_analysis_time
            out["predicted_rain_3h_mm"] = round(rain_fcst, 2)
            # Exactly which quantity each factor was computed from, so a reader
            # can tell what varies with the horizon and what does not.
            out["factor_basis"] = {
                "Recent Rainfall": (
                    f"model rain_3h_mm head at the real V4 +{_eff_lh}h lead "
                    f"(per-lead forecast; shown under the label for +{lh}h)"
                ),
                "Convective Instability (CAPE)": "GFS analysis state at t0 (same for all leads)",
                "Moisture / Humidity": "GFS analysis state at t0 (same for all leads)",
                "Wind Influence": "GFS analysis state at t0 (same for all leads)",
            }
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
    """The process-wide service singleton.

    Normally binds the production checkpoint. Two environment variables allow an
    OFFLINE evaluator (scripts/historical_backtest.py) to point the same
    production code path at a CANDIDATE model instead, so a backtest measures
    the real inference pipeline rather than a reimplementation of it:

        STORMSENSE_CKPT    path to an alternative checkpoint
        STORMSENSE_CONFIG  path to an alternative config (lead times must match
                           the checkpoint's)

    Both are unset in normal operation, so the served model is unchanged. They
    are read here rather than passed as arguments because the backtest reaches
    the service through module-level code it does not own.
    """
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        with _SERVICE_LOCK:
            if _SERVICE_INSTANCE is None:
                ckpt = os.environ.get("STORMSENSE_CKPT") or None
                cfg = os.environ.get("STORMSENSE_CONFIG") or None
                if ckpt is None and cfg is None:
                    ckpt, cfg = _default_active_model()
                _SERVICE_INSTANCE = NowcastService(
                    checkpoint_path=ckpt,
                    config_path=cfg,
                )
    return _SERVICE_INSTANCE


# The active model. Changing these two constants changes which model the
# whole application serves -- the lead set, parameter count, calibration and
# UI model identity are all derived from whatever is loaded here.
#
# V4 (Data/outputs/checkpoints_v4/v2_best.pt) remains on disk and stays
# reachable via STORMSENSE_CKPT/STORMSENSE_CONFIG for backtests and
# comparisons.
ACTIVE_MODEL_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "Data", "outputs", "checkpoints", "v2_calibrated_best.pt")
ACTIVE_MODEL_CONFIG = None  # configs/default.yaml

FALLBACK_MODEL_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "Data", "outputs", "checkpoints", "v2_calibrated_best.pt")
FALLBACK_MODEL_CONFIG = None  # configs/default.yaml


def _default_active_model() -> tuple:
    """(checkpoint, config) for the active model, or the fallback.

    `ACTIVE_MODEL_CONFIG` may legitimately be None (V2 uses the default
    configs/default.yaml, resolved by NowcastService.__init__ itself when
    config_path is None) -- so only the checkpoint path is existence-checked,
    never the config path, which would raise on None.

    Falls back rather than raising so a deployment missing the active
    checkpoint still serves forecasts -- degraded to the fallback model's own
    lead set, which the API and UI then report accurately because both derive
    the lead set from the checkpoint.
    """
    if os.path.exists(ACTIVE_MODEL_CHECKPOINT):
        return ACTIVE_MODEL_CHECKPOINT, ACTIVE_MODEL_CONFIG
    print(
        "[NowcastService] WARNING: active model checkpoint not found at "
        f"{ACTIVE_MODEL_CHECKPOINT}; falling back to {FALLBACK_MODEL_CHECKPOINT}."
    )
    return (FALLBACK_MODEL_CHECKPOINT if os.path.exists(FALLBACK_MODEL_CHECKPOINT)
            else None), FALLBACK_MODEL_CONFIG

"""Live operational atmospheric input pipeline: NOAA GFS -> model-ready tensors.

WHY GFS: SevereWeatherNetV2 is trained exclusively on ERA5 reanalysis, which has
~5-day publication latency and cannot serve a "live" nowcast. GFS (NOAA's global
forecast model) publishes a genuine 0-hour analysis field ("f000") every 6 hours
(00/06/12/18Z) at 0.25 deg resolution, freely available with no API key from the
NOAA Open Data bucket on AWS. GFS f000 is the model's own initial condition -- a
real assimilated atmospheric state, not a forecast -- so it is a legitimate
substitute for ERA5 as a *source of real observations*, subject to the caveats
documented below.

TEMPORAL DESIGN (see plan doc / commit message for full derivation):
  The model was trained on 6 CONSECUTIVE REAL HOURLY ERA5 ANALYSES ending at and
  including t0 ("now"), predicting targets strictly in the future (t0+2h/+4h/+6h).
  GFS has no hourly-observed analysis product -- only f000 analyses every 6h.
  Using GFS forecast hours f001..f005 (from one cycle) as a substitute for the
  "past" would smuggle real forecast information into the model's supposed
  observed-history input, which is scientifically invalid and is explicitly
  forbidden here.

  Instead: we fetch the last 4 real GFS f000 analysis cycles (t0-18h, t0-12h,
  t0-6h, t0 -- all four independently assimilated, real, non-forecast states),
  then LINEARLY INTERPOLATE in time between the two most recent real analyses
  (t0-6h and t0) onto the 6 required hourly slots (t0-5h, t0-4h, t0-3h, t0-2h,
  t0-1h, t0). The final slot (t0) is the real analysis itself -- no interpolation.
  No GFS forecast hour (f001+) is EVER used as an input timestep.

  LIMITATION (disclosed, not hidden): interpolating a 6-hourly analysis onto
  hourly slots smooths out any true sub-6-hour atmospheric variability (e.g. a
  thunderstorm outflow boundary that forms and dissipates within an hour). Every
  value used is still a real GFS-analyzed physical quantity -- nothing is
  invented -- but the temporal resolution of the "recent history" the model sees
  in Live mode is coarser than what it saw in ERA5-based training/backtesting.
  Every one of the 6 slots' timestamp and provenance (real analysis cycle vs.
  interpolated) is tracked and returned for audit.

DISTRIBUTION SHIFT (disclosed, not hidden): GFS and ERA5 are two different NWP
systems (NCEP vs. ECMWF) with different data assimilation, physics packages, and
minor unit/definition differences (harmonized below). This is a legitimate but
out-of-distribution operational substitution for a model trained solely on ERA5.
Live predictions should be read as directionally informative, not as precisely
calibrated as the Historical/backtested metrics (which use real ERA5 test data).

UNIT / VARIABLE HARMONIZATION (must exactly match src/features/normalize.py):
  SINGLE_VARS = ["u10","v10","d2m","t2m","sp","cape","cin","tcwv","tp"]
    u10, v10 : GFS UGRD/VGRD @ 10 m                          -> m/s   (direct)
    t2m      : GFS TMP @ 2 m above ground                    -> K     (direct)
    d2m      : GFS DPT @ 2 m above ground                    -> K     (direct)
    sp       : GFS PRES @ surface                            -> Pa    (direct)
    cape     : GFS CAPE @ surface                             -> J/kg  (direct)
    cin      : GFS CIN @ surface, clipped to >=0 to match how
               nowcast_service.py denormalizes ERA5 cin (np.maximum(0, ...)) -> J/kg
    tcwv     : GFS PWAT (precipitable water, entire atmosphere) -> kg/m^2 (direct;
               physically the same quantity ERA5 calls tcwv)
    tp       : GFS PRATE @ surface (kg/m^2/s) x 3600          -> mm/hour
               (matches era5_loader.py's ERA5 tp-in-mm/hour convention)
  PRESSURE_VARS = ["u","v","z","q","t"] at hPa levels [1000,850,700,500,300,250]
    u, v : GFS UGRD/VGRD @ 1000/850/700 mb only (z/q/t levels are NaN, mirroring
           ERA5's native per-group level coverage; the model only ever reads the
           levels where each variable group actually has real data)
    z    : GFS HGT (geopotential HEIGHT, meters) @ 700/500/300/250 mb, multiplied
           by g=9.80665 to convert to GEOPOTENTIAL (m^2/s^2), matching ERA5 z
    q    : GFS SPFH @ 700/500/300/250 mb                      -> kg/kg (direct)
    t    : GFS TMP @ 700/500/300/250 mb                       -> K     (direct)

Every value produced by this module traces back to a real, byte-range-fetched
GFS GRIB2 message. No variable is ever zero-filled, climatologically guessed, or
fabricated: if a required message is missing, this module raises immediately
with the exact variable/level/cycle that failed.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import xarray as xr

from src.features.normalize import SINGLE_VARS, PRESSURE_VARS

GRAVITY_M_S2 = 9.80665

AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

# hPa levels required by the model, and which of the two ERA5 pressure "groups"
# (wind vs thermo) actually carries real (non-NaN) data at each level -- see
# src/data/era5_loader.py module docstring: u,v @ 1000/850/700; z,q,t @ 700/500/300/250.
#
# ORDER IS LOAD-BEARING and must be ASCENDING (250 -> 1000), matching the level
# axis xarray produces from the outer-join merge in era5_loader.py and consumed
# by index in both src/data/dataset_v2.py:82-83 and src/inference/predictor.py:161-162
# (`pres_norm[:, :2, 3:]` = indices 3,4,5 = 700/850/1000 for wind;
#  `pres_norm[:, 2:, :4]` = indices 0,1,2,3 = 250/300/500/700 for thermo).
# configs/default.yaml documents `pressure_levels_hpa: [1000,850,700,500,300,250]`
# (descending) for human reference only -- it is NOT the array index order the
# model code actually slices by, which is ascending. Do not "fix" this array to
# match the YAML; that would silently swap which levels feed which stream.
PRESSURE_LEVELS_HPA = [250, 300, 500, 700, 850, 1000]
WIND_LEVELS_HPA = [700, 850, 1000]
THERMO_LEVELS_HPA = [250, 300, 500, 700]

# GFS .idx variable:level tokens, matched by EXACT (name, level) equality against
# the colon-split idx line, to avoid ambiguous substring matches (e.g. "TMP:2 m"
# would also match "TMP:2 mb" -- a pressure level -- if matched loosely).
SURFACE_MESSAGES = {
    "u10": ("UGRD", "10 m above ground"),
    "v10": ("VGRD", "10 m above ground"),
    "t2m": ("TMP", "2 m above ground"),
    "d2m": ("DPT", "2 m above ground"),
    "sp": ("PRES", "surface"),
    "cape": ("CAPE", "surface"),
    "cin": ("CIN", "surface"),
    "tcwv": ("PWAT", "entire atmosphere (considered as a single layer)"),
    "tp": ("PRATE", "surface"),
}


def _pressure_message(var_key: str, level_hpa: int) -> Tuple[str, str]:
    gfs_name = {"u": "UGRD", "v": "VGRD", "z": "HGT", "q": "SPFH", "t": "TMP"}[var_key]
    return gfs_name, f"{level_hpa} mb"


@dataclass
class GfsCycle:
    cycle_time: datetime  # UTC, real analysis (f000) timestamp
    surface: Dict[str, np.ndarray] = field(default_factory=dict)     # var -> (H,W)
    pressure: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)  # var -> {level: (H,W)}


class GfsFetchError(RuntimeError):
    """Raised when a required GFS variable/level/cycle cannot be obtained.
    Never caught-and-substituted with a guessed value -- callers must propagate
    this as a genuine live-pipeline failure."""


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=30.0, follow_redirects=True)


def _find_latest_cycle(client: httpx.Client, max_lookback_cycles: int = 8) -> datetime:
    """Return the most recent UTC cycle time whose f000 .idx file is published."""
    now = datetime.now(timezone.utc)
    # GFS cycles run at 00/06/12/18Z; start from the most recent boundary and
    # walk backwards until we find one whose .idx actually exists (production
    # lag is typically 3-5h after the cycle hour).
    cycle_hour = (now.hour // 6) * 6
    candidate = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    for _ in range(max_lookback_cycles):
        url = _idx_url(candidate)
        try:
            r = client.head(url)
            if r.status_code == 200:
                return candidate
        except httpx.HTTPError:
            pass
        candidate = candidate - timedelta(hours=6)
    raise GfsFetchError(
        f"No published GFS f000 cycle found in the last {max_lookback_cycles * 6} hours "
        f"(checked back from {now.isoformat()})."
    )


def _idx_url(cycle_time: datetime, f_hour: int = 0, base: str = AWS_BASE) -> str:
    ymd = cycle_time.strftime("%Y%m%d")
    hh = cycle_time.strftime("%H")
    fh = f"{f_hour:03d}"
    if base == AWS_BASE:
        return f"{AWS_BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fh}.idx"
    return f"{NOMADS_BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fh}.idx"


def _grib_url(cycle_time: datetime, f_hour: int = 0, base: str = AWS_BASE) -> str:
    return _idx_url(cycle_time, f_hour, base)[:-4]  # strip ".idx"


def _parse_idx(idx_text: str) -> List[List[str]]:
    lines = [l for l in idx_text.strip().split("\n") if l]
    return [l.split(":") for l in lines]


def _byte_range_for(parsed_idx: List[List[str]], gfs_name: str, gfs_level: str) -> Tuple[int, int]:
    for i, parts in enumerate(parsed_idx):
        if len(parts) >= 5 and parts[3] == gfs_name and parts[4] == gfs_level:
            start = int(parts[1])
            end = int(parsed_idx[i + 1][1]) - 1 if i + 1 < len(parsed_idx) else start + 20_000_000
            return start, end
    raise GfsFetchError(f"GFS message not found in .idx: {gfs_name}:{gfs_level}")


# Small margin (deg) around the WB domain kept before interpolation, so cfgrib
# only has to materialize a small regional window of the full 721x1440 global
# grid rather than the whole planet for every single-message fetch.
_DOMAIN_MARGIN_DEG = 1.0


def _fetch_message(
    client: httpx.Client, grib_url: str, parsed_idx: List[List[str]], gfs_name: str, gfs_level: str,
    lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Fetch one GRIB2 message via byte-range GET, decode it, immediately crop to
    a small window around the WB domain, and return only that cropped array --
    never holds the full global 721x1440 grid in memory longer than one message."""
    start, end = _byte_range_for(parsed_idx, gfs_name, gfs_level)
    r = client.get(grib_url, headers={"Range": f"bytes={start}-{end}"})
    if r.status_code not in (200, 206):
        raise GfsFetchError(
            f"Failed to fetch GFS message {gfs_name}:{gfs_level} from {grib_url} "
            f"(HTTP {r.status_code})"
        )
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
        tf.write(r.content)
        tmp_path = tf.name
    try:
        ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        varname = list(ds.data_vars)[0]
        da = ds[varname]
        lat_lo, lat_hi = float(lats.min()) - _DOMAIN_MARGIN_DEG, float(lats.max()) + _DOMAIN_MARGIN_DEG
        lon_lo, lon_hi = float(lons.min()) - _DOMAIN_MARGIN_DEG, float(lons.max()) + _DOMAIN_MARGIN_DEG
        cropped = da.sel(latitude=slice(lat_hi, lat_lo), longitude=slice(lon_lo, lon_hi)).load()
        ds.close()
        return cropped
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _subset_to_domain(cropped_da: "xr.DataArray", lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Interpolate an already-cropped GFS message window onto the exact target
    grid vectors used by NowcastService (lat descending 28->20, lon ascending
    84->90). GFS longitude is already 0-360 ascending; West Bengal (84-90E)
    needs no wraparound."""
    out = cropped_da.interp(latitude=lats, longitude=lons, method="linear")
    arr = out.values.astype(np.float32)
    if arr.shape != (len(lats), len(lons)):
        raise GfsFetchError(
            f"Regridded GFS message has shape {arr.shape}, expected {(len(lats), len(lons))}"
        )
    return arr


def fetch_gfs_slice(cycle_time: datetime, f_hour: int, lats: np.ndarray, lons: np.ndarray) -> GfsCycle:
    """Fetch and regrid every required surface + pressure-level message for one
    real GFS f000 analysis cycle. Raises GfsFetchError naming the exact missing
    variable if anything required is unavailable -- never substitutes a guess."""

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "processed", "cache", "gfs")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cycle_time.strftime('%Y%m%d_%H%M%S')}_{f_hour}.pkl")
    if os.path.exists(cache_file):
        try:
            import pickle
            with open(cache_file, 'rb') as f_cache:
                return pickle.load(f_cache)
        except Exception as e:
            pass
    last_err: Optional[Exception] = None
    for base in (AWS_BASE, NOMADS_BASE):
        try:
            with _http_client() as client:
                idx_url = _idx_url(cycle_time, f_hour, base)
                grib_url = _grib_url(cycle_time, f_hour, base)
                idx_resp = client.get(idx_url)
                if idx_resp.status_code != 200:
                    raise GfsFetchError(f"Cycle {cycle_time.isoformat()} .idx unavailable at {base} (HTTP {idx_resp.status_code})")
                parsed = _parse_idx(idx_resp.text)

                cycle = GfsCycle(cycle_time=cycle_time)

                for var, (gfs_name, gfs_level) in SURFACE_MESSAGES.items():
                    cropped = _fetch_message(client, grib_url, parsed, gfs_name, gfs_level, lats, lons)
                    cycle.surface[var] = _subset_to_domain(cropped, lats, lons)

                for var_key in ["u", "v"]:
                    cycle.pressure.setdefault(var_key, {})
                    for lvl in WIND_LEVELS_HPA:
                        gfs_name, gfs_level = _pressure_message(var_key, lvl)
                        cropped = _fetch_message(client, grib_url, parsed, gfs_name, gfs_level, lats, lons)
                        cycle.pressure[var_key][lvl] = _subset_to_domain(cropped, lats, lons)

                for var_key in ["z", "q", "t"]:
                    cycle.pressure.setdefault(var_key, {})
                    for lvl in THERMO_LEVELS_HPA:
                        gfs_name, gfs_level = _pressure_message(var_key, lvl)
                        cropped = _fetch_message(client, grib_url, parsed, gfs_name, gfs_level, lats, lons)
                        arr = _subset_to_domain(cropped, lats, lons)
                        if var_key == "z":
                            arr = arr * GRAVITY_M_S2  # geopotential height (m) -> geopotential (m^2/s^2)
                        cycle.pressure[var_key][lvl] = arr

                try:
                    import pickle
                    with open(cache_file, 'wb') as f_cache:
                        pickle.dump(cycle, f_cache)
                except Exception:
                    pass
                return cycle
        except (GfsFetchError, httpx.HTTPError) as e:
            last_err = e
            continue
    raise GfsFetchError(f"Could not fetch GFS cycle {cycle_time.isoformat()} from AWS or NOMADS: {last_err}")


def _build_pressure_tensor(cycles_pressure: List[Dict[str, Dict[int, np.ndarray]]], H: int, W: int) -> np.ndarray:
    """Stack per-cycle pressure dicts into (n_cycles, n_vars=5, n_levels=6, H, W),
    matching PRESSURE_VARS order and PRESSURE_LEVELS_HPA order. Levels without
    real data for a given variable group are filled with NaN, exactly mirroring
    ERA5's own outer-join NaN pattern (see era5_loader.py) -- the model's slicing
    (predictor.py) only ever reads the levels where each group has real data."""
    n_cycles = len(cycles_pressure)
    out = np.full((n_cycles, len(PRESSURE_VARS), len(PRESSURE_LEVELS_HPA), H, W), np.nan, dtype=np.float32)
    for ci, pdict in enumerate(cycles_pressure):
        for vi, var in enumerate(PRESSURE_VARS):
            for li, lvl in enumerate(PRESSURE_LEVELS_HPA):
                if lvl in pdict.get(var, {}):
                    out[ci, vi, li] = pdict[var][lvl]
    return out


def _build_surface_tensor(cycles_surface: List[Dict[str, np.ndarray]], H: int, W: int) -> np.ndarray:
    n_cycles = len(cycles_surface)
    out = np.full((n_cycles, len(SINGLE_VARS), H, W), np.nan, dtype=np.float32)
    for ci, sdict in enumerate(cycles_surface):
        for vi, var in enumerate(SINGLE_VARS):
            out[ci, vi] = sdict[var]
    # cin must be non-negative, matching how ERA5 cin is treated when denormalized
    # elsewhere in this codebase (nowcast_service.py: np.maximum(0.0, ...)).
    cin_i = SINGLE_VARS.index("cin")
    out[:, cin_i] = np.maximum(0.0, out[:, cin_i])
    # tp (PRATE, kg/m^2/s) -> mm/hour
    tp_i = SINGLE_VARS.index("tp")
    out[:, tp_i] = np.clip(out[:, tp_i] * 3600.0, 0.0, None)
    return out


@dataclass
class HarmonizedLiveInput:
    surface: np.ndarray          # (6, 9, 33, 25) physical units, matching SINGLE_VARS
    pressure: np.ndarray         # (6, 5, 6, 33, 25) physical units, matching PRESSURE_VARS x levels
    t0: datetime                 # EXACT wall-clock reference instant the forecast is issued for
    analysis_t0: datetime        # real GFS f000 analysis time the newest input slot came from
    analysis_cycles: List[datetime]  # the real f000 cycles actually fetched
    slot_timestamps: List[str]   # ISO-8601 timestamp of each of the 6 input slots
    slot_provenance: List[Dict]  # per-slot {"timestamp":..., "source": "analysis"|"interpolated", "bracket": [...]}
    fetched_at: datetime         # wall-clock time this harmonization completed
    wallclock_age_hours: float   # how far behind wall-clock the analysis_t0 state is


# Module-level cache of the last successful harmonization, keyed implicitly by
# its `t0` GFS cycle -- see fetch_and_harmonize().
_HARMONIZED_CACHE: Optional[HarmonizedLiveInput] = None


def _cycles_available_at(wallclock: datetime, production_lag_hours: float) -> List[datetime]:
    """Every GFS cycle time that would GENUINELY have been published by
    `wallclock`, newest first.

    A cycle nominally at 12Z is not on the wire at 12:00Z -- NCEP needs time to
    run and disseminate it. `production_lag_hours` is that publication latency:
    a cycle C is only considered available once wallclock >= C + lag. This is
    what makes the historical backtest honest (no future cycle can ever be
    selected) and is the same rule the live path uses, so backtest and
    production select cycles identically.
    """
    newest_boundary = wallclock.replace(minute=0, second=0, microsecond=0)
    newest_boundary -= timedelta(hours=newest_boundary.hour % 6)
    out: List[datetime] = []
    c = newest_boundary
    # Walk back until we have a generous span of candidates; callers take what
    # they need from the front of this list.
    for _ in range(12):
        if c + timedelta(hours=production_lag_hours) <= wallclock:
            out.append(c)
        c -= timedelta(hours=6)
    return out


# Typical NCEP GFS publication latency. Measured against the live feed: a cycle
# is routinely not fully on the wire until ~3.5-5h after its nominal hour. Used
# as the availability rule for BOTH live selection and the historical backtest so
# neither can ever consume a cycle that did not yet exist.
GFS_PRODUCTION_LAG_HOURS = 5.0


def _interpolate_slot(
    older: GfsCycle, newer: GfsCycle, when: datetime
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, np.ndarray]], float]:
    """Linear time-interpolation of two REAL f000 analyses onto one hourly slot.

    Returns (surface_dict, pressure_dict, weight_on_newer). When `when` coincides
    with one of the two analyses the weight collapses to 0 or 1 and the returned
    fields are that real analysis, unmodified.
    """
    span = (newer.cycle_time - older.cycle_time).total_seconds()
    if span <= 0:
        raise GfsFetchError(
            f"Bracketing analyses are not ordered: {older.cycle_time} -> {newer.cycle_time}"
        )
    w_new = (when - older.cycle_time).total_seconds() / span
    w_new = float(np.clip(w_new, 0.0, 1.0))
    w_old = 1.0 - w_new

    surf = {k: older.surface[k] * w_old + newer.surface[k] * w_new for k in SINGLE_VARS}
    pres: Dict[str, Dict[int, np.ndarray]] = {}
    for k in PRESSURE_VARS:
        pres[k] = {}
        for lvl in older.pressure.get(k, {}):
            if lvl in newer.pressure.get(k, {}):
                pres[k][lvl] = older.pressure[k][lvl] * w_old + newer.pressure[k][lvl] * w_new
    return surf, pres, w_new


def fetch_and_harmonize(
    lats: np.ndarray,
    lons: np.ndarray,
    target_t0: datetime,
    use_cache: bool = True,
    production_lag_hours: float = GFS_PRODUCTION_LAG_HOURS,
) -> HarmonizedLiveInput:
    """Build the model's 6-slot hourly input ending at the newest REAL GFS f000
    analysis available at `target_t0`, using analyses ONLY.

    TEMPORAL CONTRACT (non-negotiable, see module docstring):
      * Only f000 messages are ever fetched. GFS forecast hours f001+ are future
        forecast steps, NOT observed history; feeding them into the model's
        supposed observed-past input would leak forecast information into the
        input the model was trained to read as analysis. This function must
        never request a non-zero forecast hour.
      * No cycle newer than what `production_lag_hours` says was actually
        published by `target_t0` is ever considered -- so a historical backtest
        run at a simulated past `target_t0` selects exactly the cycle an
        operator would have had at that moment.

    The six slots are hourly (t-5h..t0) RELATIVE TO THE ANALYSIS TIME, linearly
    interpolated between the two most recent real analyses. `analysis_t0` is that
    real analysis time; `t0` is the caller's exact wall-clock reference instant.
    These are deliberately distinct: the forecast is ISSUED for the wall-clock
    instant, but the atmospheric STATE it reads is as of the latest published
    analysis, which lags wall-clock by the production lag. Callers must present
    both rather than conflating them.
    """
    global _HARMONIZED_CACHE
    H, W = len(lats), len(lons)

    if target_t0.tzinfo is None:
        target_t0 = target_t0.replace(tzinfo=timezone.utc)

    candidates = _cycles_available_at(target_t0, production_lag_hours)
    if len(candidates) < 2:
        raise GfsFetchError(
            f"Fewer than two GFS analysis cycles were published as of {target_t0.isoformat()} "
            f"(production lag {production_lag_hours}h); cannot build the 6-hour input sequence."
        )

    # Walk the candidate list until we find a NEWEST cycle that is actually
    # fetchable, so a cycle that is late or partially disseminated degrades to
    # the previous real one instead of failing the whole refresh.
    newest: Optional[GfsCycle] = None
    older: Optional[GfsCycle] = None
    last_err: Optional[Exception] = None
    for i in range(len(candidates) - 1):
        try:
            newest = fetch_gfs_slice(candidates[i], 0, lats, lons)
            older = fetch_gfs_slice(candidates[i + 1], 0, lats, lons)
            break
        except (GfsFetchError, httpx.HTTPError) as e:
            last_err = e
            newest = older = None
            continue
    if newest is None or older is None:
        raise GfsFetchError(
            f"No pair of consecutive real GFS f000 analyses could be fetched as of "
            f"{target_t0.isoformat()}: {last_err}"
        )

    analysis_t0 = newest.cycle_time

    if use_cache and _HARMONIZED_CACHE is not None:
        if (
            _HARMONIZED_CACHE.analysis_t0 == analysis_t0
            and _HARMONIZED_CACHE.surface.shape[-2:] == (H, W)
        ):
            # Same real analysis state; only the wall-clock reference moved on.
            cached = _HARMONIZED_CACHE
            return HarmonizedLiveInput(
                surface=cached.surface,
                pressure=cached.pressure,
                t0=target_t0,
                analysis_t0=cached.analysis_t0,
                analysis_cycles=cached.analysis_cycles,
                slot_timestamps=cached.slot_timestamps,
                slot_provenance=cached.slot_provenance,
                fetched_at=cached.fetched_at,
                wallclock_age_hours=(target_t0 - cached.analysis_t0).total_seconds() / 3600.0,
            )

    # Six hourly slots ending exactly at the real analysis time.
    slot_times = [analysis_t0 - timedelta(hours=h) for h in range(5, -1, -1)]

    out_surface: List[Dict[str, np.ndarray]] = []
    out_pressure: List[Dict[str, Dict[int, np.ndarray]]] = []
    slot_provenance: List[Dict] = []

    for st in slot_times:
        surf, pres, w_new = _interpolate_slot(older, newest, st)
        out_surface.append(surf)
        out_pressure.append(pres)
        is_real = abs(w_new - 1.0) < 1e-9 or abs(w_new) < 1e-9
        slot_provenance.append({
            "timestamp": st.isoformat(),
            "source": "analysis" if is_real else "interpolated",
            "forecast_hour_used": 0,
            "bracket": [older.cycle_time.isoformat(), newest.cycle_time.isoformat()],
            "weight_on_newer_analysis": round(w_new, 4),
            "note": (
                "Real GFS f000 analysis, used unmodified."
                if is_real
                else "Linear time-interpolation between two real GFS f000 analyses. "
                     "No forecast hour is used as an input timestep."
            ),
        })

    surf_tensor = _build_surface_tensor(out_surface, H, W)
    pres_tensor = _build_pressure_tensor(out_pressure, H, W)

    res = HarmonizedLiveInput(
        surface=surf_tensor,
        pressure=pres_tensor,
        t0=target_t0,
        analysis_t0=analysis_t0,
        analysis_cycles=[newest.cycle_time, older.cycle_time],
        slot_timestamps=[st.isoformat() for st in slot_times],
        slot_provenance=slot_provenance,
        fetched_at=datetime.now(timezone.utc),
        wallclock_age_hours=(target_t0 - analysis_t0).total_seconds() / 3600.0,
    )

    _HARMONIZED_CACHE = res
    return res

"""Current-conditions observation surface for West Bengal.

This module renders a CURRENT-OBSERVATION field, strictly separate from the
StormSense AI forecast risk field produced by `risk_surface.py`. The two must
never share a code path, a colour ramp, or a cache: one is what is happening
now, the other is what the model predicts will happen later.

SCIENTIFIC BASIS AND ITS LIMITS
-------------------------------
The input is a set of *scattered point observations* from the project's existing
current-weather provider (OpenWeatherMap `/data/2.5/weather`, `base: "stations"`),
sampled at a fixed list of West Bengal settlements. That is genuinely current
data -- observation timestamps (`dt`) run a few minutes behind wall clock -- but
it is NOT a gridded observed product, and this module does not pretend otherwise:

  * Values BETWEEN stations are INTERPOLATED, not measured. The interpolation is
    inverse-distance weighting (IDW), stated in the API response, rendered with a
    visible station overlay, and labelled as interpolated in the UI.
  * Pixels farther than `MAX_INFLUENCE_DEG` from every station are left
    TRANSPARENT rather than extrapolated. Empty space is shown as empty, never
    filled with a plausible-looking guess.
  * A station that fails to report is dropped from the interpolation entirely.
    It is never zero-filled, and a zero is never synthesised to mean "no data" --
    for rainfall especially, absent data and zero rain are different facts.

Nothing here consults the GFS pipeline, the model, or any forecast horizon.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.inference.risk_surface import (
    DEFAULT_H,
    DEFAULT_W,
    WB_MAX_LAT,
    WB_MAX_LON,
    WB_MIN_LAT,
    WB_MIN_LON,
    get_or_create_wb_mask,
)

# Sampling sites spread across the West Bengal domain. These are real
# settlements the provider resolves to distinct observation points (verified:
# 12/12 distinct coordinates returned, including pairs 0.25 deg apart).
# Chosen for geographic spread across all 23 districts, north to south.
OBSERVATION_SITES: Tuple[Tuple[str, float, float], ...] = (
    ("Darjeeling", 27.04, 88.26),
    ("Kalimpong", 27.07, 88.47),
    ("Jalpaiguri", 26.52, 88.72),
    ("Alipurduar", 26.49, 89.53),
    ("Cooch Behar", 26.32, 89.45),
    ("Raiganj", 25.62, 88.13),
    ("Balurghat", 25.22, 88.77),
    ("Malda", 25.01, 88.14),
    ("Berhampore", 24.10, 88.25),
    ("Suri", 23.91, 87.53),
    ("Krishnanagar", 23.40, 88.50),
    ("Asansol", 23.68, 86.98),
    ("Bardhaman", 23.24, 87.86),
    ("Bankura", 23.23, 87.07),
    ("Purulia", 23.33, 86.36),
    ("Barasat", 22.72, 88.48),
    ("Kolkata", 22.57, 88.36),
    ("Howrah", 22.59, 88.26),
    ("Chinsurah", 22.90, 88.39),
    ("Jhargram", 22.45, 86.99),
    ("Midnapore", 22.42, 87.32),
    ("Haldia", 22.06, 88.06),
    ("Diamond Harbour", 22.19, 88.19),
    ("Canning", 22.32, 88.67),
)

# A pixel farther than this (in degrees) from EVERY reporting station is not
# rendered. ~0.9 deg is roughly 100 km: beyond that an interpolated surface
# stops being a defensible statement about local conditions.
MAX_INFLUENCE_DEG = 0.9

# IDW exponent. 2.0 is the conventional choice; higher values make the field
# more "blocky" around stations, lower values over-smooth distant contrasts.
IDW_POWER = 2.0

# Variables this module can render. Each entry carries the display range used to
# normalise the field for colouring, and the unit for the UI.
RENDERABLE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "rain_1h_mm": {"label": "Rainfall (last hour)", "unit": "mm", "vmin": 0.0, "vmax": 10.0},
    "temperature_c": {"label": "Temperature", "unit": "°C", "vmin": 10.0, "vmax": 40.0},
    "humidity_pct": {"label": "Relative humidity", "unit": "%", "vmin": 30.0, "vmax": 100.0},
    "cloud_pct": {"label": "Cloud cover", "unit": "%", "vmin": 0.0, "vmax": 100.0},
}


@dataclass
class StationObservation:
    """One station's current reading, in physical units.

    A field left as None means the provider did not report it. It is carried
    through as None and excluded from that variable's interpolation -- never
    coerced to 0.
    """
    name: str
    lat: float
    lon: float
    observed_at_utc: Optional[str] = None
    observed_unix: Optional[int] = None
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rain_1h_mm: Optional[float] = None
    cloud_pct: Optional[float] = None
    wind_kmh: Optional[float] = None
    pressure_hpa: Optional[float] = None
    condition: Optional[str] = None
    provider_station: Optional[str] = None

    def value_for(self, variable: str) -> Optional[float]:
        return getattr(self, variable, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "observed_at_utc": self.observed_at_utc,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "rain_1h_mm": self.rain_1h_mm,
            "cloud_pct": self.cloud_pct,
            "wind_kmh": self.wind_kmh,
            "pressure_hpa": self.pressure_hpa,
            "condition": self.condition,
            "provider_station": self.provider_station,
        }


def parse_owm_current(
    name: str, lat: float, lon: float, payload: Dict[str, Any]
) -> StationObservation:
    """Convert one OpenWeatherMap current-weather payload into a StationObservation.

    Missing keys stay None. In particular OWM omits the `rain` block entirely
    when there is no precipitation, which genuinely means zero rainfall was
    observed -- distinct from the station failing to report, which surfaces as a
    fetch error and drops the station altogether.
    """
    from datetime import datetime, timezone

    main = payload.get("main") or {}
    wind = payload.get("wind") or {}
    clouds = payload.get("clouds") or {}
    rain = payload.get("rain") or {}
    weather_list = payload.get("weather") or []
    coord = payload.get("coord") or {}

    dt = payload.get("dt")
    observed_iso = (
        datetime.fromtimestamp(dt, tz=timezone.utc).isoformat() if dt else None
    )

    wind_ms = wind.get("speed")
    return StationObservation(
        name=name,
        # Prefer the coordinate the provider actually resolved to.
        lat=float(coord.get("lat", lat)),
        lon=float(coord.get("lon", lon)),
        observed_at_utc=observed_iso,
        observed_unix=dt,
        temperature_c=main.get("temp"),
        humidity_pct=main.get("humidity"),
        # `rain` absent => no rain observed in the last hour (a real 0.0).
        rain_1h_mm=float(rain.get("1h", 0.0)) if payload.get("dt") is not None else None,
        cloud_pct=clouds.get("all"),
        wind_kmh=round(float(wind_ms) * 3.6, 1) if wind_ms is not None else None,
        pressure_hpa=main.get("pressure"),
        condition=(weather_list[0].get("description") if weather_list else None),
        provider_station=payload.get("name"),
    )


def idw_interpolate(
    stations: Sequence[StationObservation],
    variable: str,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    power: float = IDW_POWER,
    max_influence_deg: float = MAX_INFLUENCE_DEG,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Inverse-distance-weight scattered station values onto the display grid.

    Returns (values, coverage, n_used):
      values   : (h, w) interpolated field, NaN where out of influence range
      coverage : (h, w) bool, True where at least one station is close enough
      n_used   : how many stations actually contributed

    Stations that did not report `variable` are excluded rather than zero-filled.
    """
    usable = [s for s in stations if s.value_for(variable) is not None]
    if not usable:
        empty = np.full((h, w), np.nan, dtype=np.float32)
        return empty, np.zeros((h, w), dtype=bool), 0

    lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, h)
    lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, w)
    lat_mesh, lon_mesh = np.meshgrid(lats_fine, lons_fine, indexing="ij")

    weight_sum = np.zeros((h, w), dtype=np.float64)
    weighted_vals = np.zeros((h, w), dtype=np.float64)
    nearest = np.full((h, w), np.inf, dtype=np.float64)

    for st in usable:
        val = float(st.value_for(variable))
        d = np.sqrt((lat_mesh - st.lat) ** 2 + (lon_mesh - st.lon) ** 2)
        nearest = np.minimum(nearest, d)
        # Guard the singularity exactly at a station location.
        d = np.maximum(d, 1e-6)
        wgt = 1.0 / np.power(d, power)
        weight_sum += wgt
        weighted_vals += wgt * val

    values = np.divide(
        weighted_vals, weight_sum,
        out=np.full((h, w), np.nan), where=weight_sum > 0,
    )

    # Refuse to state anything about pixels with no nearby station.
    coverage = nearest <= max_influence_deg
    values = np.where(coverage, values, np.nan)
    return values.astype(np.float32), coverage, len(usable)


def _colormap_observation(
    values: np.ndarray,
    coverage: np.ndarray,
    mask: np.ndarray,
    variable: str,
) -> np.ndarray:
    """Colour the observation field.

    Rainfall uses a precipitation-INTENSITY ramp (yellow -> amber -> orange ->
    red) because intensity is the quantity being shown and the previous single-hue
    blue ramp made 0.1 mm and 10 mm look alike against a dark basemap. It encodes
    measured mm/h, never a probability; the forecast risk surface remains a
    separate layer with its own green -> red ramp, and the UI labels each as
    observation vs forecast.

    Every other variable keeps the BLUE/TEAL ramp: those are atmospheric states
    rather than intensities, so they stay visually distinct from any risk field.
    """
    spec = RENDERABLE_VARIABLES[variable]
    vmin, vmax = float(spec["vmin"]), float(spec["vmax"])

    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = coverage & mask & np.isfinite(values)
    if not valid.any():
        return rgba

    # Replace NaN (uncovered pixels) with 0 BEFORE any integer cast. Casting a
    # NaN to int is undefined behaviour and produced garbage indices; the
    # `valid` mask below is what actually decides which pixels are drawn, so the
    # substituted value is never displayed.
    safe = np.where(np.isfinite(values), values, vmin)
    norm = np.clip((safe - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)

    if variable == "rain_1h_mm":
        # Rainfall INTENSITY must be readable as intensity: a single-hue blue
        # ramp rendered 0.1 mm and 10 mm in near-identical cyan, so heavy rain
        # was indistinguishable from a trace. Run the ramp through to red at the
        # top of the scale, matching how precipitation intensity is conventionally
        # shaded. This is still an OBSERVATION surface -- the colour encodes
        # measured mm/h, never a severe-weather probability -- and the panel's own
        # "Observed now at reporting stations · not an AI forecast" line keeps that
        # distinction explicit.
        stops = np.array([
            [254, 240, 138],  # pale yellow (trace)
            [250, 204, 21],   # yellow (light)
            [245, 158, 11],   # amber (moderate)
            [249, 115, 22],   # orange (heavy)
            [239, 68, 68],    # red (very heavy)
        ], dtype=np.float64)
    else:
        # Non-precipitation fields keep the blue/teal observation ramp: they are
        # states, not intensities, and must stay visually distinct from risk.
        stops = np.array([
            [8, 47, 73],      # deep blue (low)
            [14, 116, 144],   # teal
            [34, 211, 238],   # cyan
            [165, 243, 252],  # pale cyan (high)
        ], dtype=np.float64)

    pos = norm * (len(stops) - 1)
    idx = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - idx)[..., None]
    colours = stops[idx] * (1 - frac) + stops[idx + 1] * frac

    rgba[..., 0] = np.where(valid, colours[..., 0], 0)
    rgba[..., 1] = np.where(valid, colours[..., 1], 0)
    rgba[..., 2] = np.where(valid, colours[..., 2], 0)

    if variable == "rain_1h_mm":
        # Dry areas must still read as "nothing happening", so anything at or
        # near zero stays fully transparent. Above that threshold, however, the
        # old linear ramp left real rain almost invisible: typical observed
        # totals are a fraction of a millimetre against a 10 mm scale, so a
        # genuine 0.9 mm reading drew at ~12% opacity. Give any measurable rain
        # a visible floor and climb from there.
        alpha = np.where(
            norm < 0.01,
            0.0,
            120.0 + np.power(np.clip(norm, 0.0, 1.0), 0.45) * 115.0,
        )
    else:
        alpha = np.full((h, w), 180.0)

    rgba[..., 3] = np.where(valid, alpha.astype(np.uint8), 0)
    return rgba


def generate_observation_surface_png(
    stations: Sequence[StationObservation],
    variable: str = "rain_1h_mm",
    mask: Optional[np.ndarray] = None,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
) -> bytes:
    """Render the current-observation field as a transparent PNG.

    Clipped to the West Bengal boundary (same mask as the forecast surface, so
    both stay geographically contained) and to the station influence radius.
    """
    if variable not in RENDERABLE_VARIABLES:
        raise ValueError(
            f"Unsupported observation variable {variable!r}; "
            f"expected one of {sorted(RENDERABLE_VARIABLES)}"
        )
    if mask is None:
        mask = get_or_create_wb_mask(h=h, w=w)

    values, coverage, _ = idw_interpolate(stations, variable, h=h, w=w)
    rgba = _colormap_observation(values, coverage, mask, variable)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def summarise_observations(
    stations: Sequence[StationObservation], variable: str = "rain_1h_mm"
) -> Dict[str, Any]:
    """Provenance and coverage summary for the API response.

    Everything a caller needs to judge how much the field can be trusted:
    which stations reported, how old the readings are, and the explicit
    statement that between-station values are interpolated.
    """
    reporting = [s for s in stations if s.value_for(variable) is not None]
    times = [s.observed_unix for s in stations if s.observed_unix]

    vals = [float(s.value_for(variable)) for s in reporting]
    observed_min = min(times) if times else None
    observed_max = max(times) if times else None

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).timestamp()

    return {
        "variable": variable,
        "variable_label": RENDERABLE_VARIABLES[variable]["label"],
        "unit": RENDERABLE_VARIABLES[variable]["unit"],
        "value_range": {
            "min": round(min(vals), 2) if vals else None,
            "max": round(max(vals), 2) if vals else None,
        },
        "display_range": {
            "min": RENDERABLE_VARIABLES[variable]["vmin"],
            "max": RENDERABLE_VARIABLES[variable]["vmax"],
        },
        "stations_requested": len(stations),
        "stations_reporting": len(reporting),
        "oldest_observation_utc": (
            datetime.fromtimestamp(observed_min, tz=timezone.utc).isoformat()
            if observed_min else None
        ),
        "newest_observation_utc": (
            datetime.fromtimestamp(observed_max, tz=timezone.utc).isoformat()
            if observed_max else None
        ),
        "max_observation_age_seconds": (
            int(now - observed_min) if observed_min else None
        ),
        # The honesty contract, carried in the payload itself.
        "is_forecast": False,
        "is_interpolated": True,
        "interpolation_method": f"inverse-distance weighting (power {IDW_POWER})",
        "max_influence_deg": MAX_INFLUENCE_DEG,
        "provenance_note": (
            "Current conditions measured at discrete reporting stations. Values "
            "between stations are interpolated, not measured, and areas beyond "
            f"{MAX_INFLUENCE_DEG} degrees of any station are left blank. This is "
            "observed data, not a StormSense model forecast."
        ),
    }

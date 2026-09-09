"""Risk-map generation module.

Converts raw model output arrays into structured geospatial products:
  - GeoJSON FeatureCollections (for the existing frontend/API)
  - NetCDF (optional, for archival)
  - In-memory dict for API responses

All coordinates are WGS-84 (EPSG:4326).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np


def _risk_level(score: float) -> str:
    """Convert a 0-1 risk score to a human-readable level."""
    if score < 0.15:
        return "Low"
    if score < 0.35:
        return "Moderate"
    if score < 0.60:
        return "High"
    return "Very High"


def compute_valid_time(issue_time_str: Optional[str], lead_hours: int) -> Optional[str]:
    """Dynamically compute forecast valid time from issue time and lead hours."""
    if not issue_time_str:
        return None
    try:
        clean = issue_time_str.replace("Z", "+00:00").replace(" ", "T")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        vt = dt + timedelta(hours=lead_hours)
        return vt.isoformat()
    except Exception:
        return f"{issue_time_str} +{lead_hours}h"


def predictions_to_geojson(
    pred: dict,
    valid_time: Optional[str] = None,
    lead_time_hours: Optional[int] = None,
    lead_idx: int = 0,
    as_polygon: bool = False,
) -> dict:
    """Convert spatial prediction arrays for one lead time into a GeoJSON
    FeatureCollection where each Feature is a 0.25 deg grid-cell point or polygon.

    Parameters
    ----------
    pred : dict
        Output of NowcastPredictor.predict().
    valid_time : str | None
        ISO-8601 string for the input window's last timestep.
    lead_time_hours : int | None
        Optional explicit horizon hour.
    lead_idx : int
        Which lead-time slice to export.
    as_polygon : bool
        If True, exports 0.25-deg bounding boxes (Polygon); if False, Point centroids.

    Returns
    -------
    dict
        GeoJSON FeatureCollection ready for JSON serialisation.
    """
    lats = pred["lats"]
    lons = pred["lons"]
    lt = lead_time_hours or pred["lead_times_hours"][lead_idx]
    forecast_vt = compute_valid_time(valid_time, int(lt)) if valid_time else None

    severe_prob = pred["severe_weather_prob"][lead_idx]
    rain_pred   = pred["rain_3h_mm_pred"][lead_idx]
    severe_bin  = pred["severe_weather_binary"][lead_idx]
    ff_risk     = pred["flash_flood_risk"][lead_idx]
    overall     = pred["overall_risk"][lead_idx]

    raw_thresh = pred.get("threshold", 0.5)
    if isinstance(raw_thresh, dict):
        cur_thresh = float(raw_thresh.get(lt, raw_thresh.get(int(lt), raw_thresh.get(str(lt), 0.5))))
    else:
        cur_thresh = float(raw_thresh)

    dlat = 0.25 / 2.0
    dlon = 0.25 / 2.0

    features = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            prob = float(severe_prob[i, j])
            if prob >= 0.75:
                imd_color = "red"
            elif prob >= 0.50:
                imd_color = "orange"
            elif prob >= 0.25:
                imd_color = "yellow"
            else:
                imd_color = "green"

            props = {
                "lat": float(lat),
                "lon": float(lon),
                "thunderstorm_probability": float(round(prob, 4)),
                "thunderstorm_pct": float(round(prob * 100, 1)),
                "heavy_rain_probability": float(round(prob, 4)),  # proxy same source
                "rain_3h_mm_forecast": float(round(float(rain_pred[i, j]), 2)),
                "rainfall_mm": float(round(float(rain_pred[i, j]), 2)),
                "flash_flood_risk": float(round(float(ff_risk[i, j]), 4)),
                "flash_flood_proxy_pct": float(round(float(ff_risk[i, j]) * 100, 1)),
                "overall_risk": float(round(float(overall[i, j]), 4)),
                "risk_level": _risk_level(float(overall[i, j])),
                "alert_level": imd_color.upper(),
                "imd_color": imd_color,
                "severe_weather_alert": bool(int(severe_bin[i, j])),
                "lead_hours": int(lt),
                "lead_time_hours": int(lt),
                "threshold": cur_thresh,
                "grid_resolution": "0.25° (~28 km)",
            }
            if valid_time:
                props["issue_time"] = valid_time
                props["input_valid_time"] = valid_time
            if forecast_vt:
                props["forecast_valid_time"] = forecast_vt

            if as_polygon:
                poly = [
                    [
                        [round(float(lon - dlon), 4), round(float(lat - dlat), 4)],
                        [round(float(lon + dlon), 4), round(float(lat - dlat), 4)],
                        [round(float(lon + dlon), 4), round(float(lat + dlat), 4)],
                        [round(float(lon - dlon), 4), round(float(lat + dlat), 4)],
                        [round(float(lon - dlon), 4), round(float(lat - dlat), 4)],
                    ]
                ]
                geom = {"type": "Polygon", "coordinates": poly}
            else:
                geom = {"type": "Point", "coordinates": [float(lon), float(lat)]}

            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": props,
            }
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "crs": "EPSG:4326",
            "model_grid_resolution": "0.25°",
            "lead_time_hours": int(lt),
            "threshold": cur_thresh,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "issue_time": valid_time,
            "forecast_valid_time": forecast_vt,
            "n_grid_cells": len(features),
            "dem_source": "SRTM 30m DEM (0.25° resampled static elevation)",
            "proxy_disclaimer": (
                "Predictions are derived from SevereWeatherNet V2 Calibrated ML model on ERA5 reanalysis proxies. "
                "All risk values are model estimates, not authoritative forecasts."
            ),
        },
    }


def predictions_to_all_leads_geojson(
    pred: dict,
    valid_time: Optional[str] = None,
) -> dict:
    """Return a dict keyed by lead_time_hours, each value a GeoJSON FC."""
    return {
        f"lead_{lh}h": predictions_to_geojson(pred, valid_time, lh, i)
        for i, lh in enumerate(pred["lead_times_hours"])
    }


def save_geojson(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=None, separators=(",", ":"))  # compact for frontend


def predictions_to_summary(pred: dict, valid_time: Optional[str] = None) -> dict:
    """Compact JSON summary: domain-level stats per lead time."""
    out = {
        "valid_time": valid_time,
        "lead_times_hours": pred["lead_times_hours"],
        "threshold": pred["threshold"],
        "per_lead": {},
    }
    for i, lh in enumerate(pred["lead_times_hours"]):
        overall = pred["overall_risk"][i]
        severe_prob = pred["severe_weather_prob"][i]
        ff = pred["flash_flood_risk"][i]
        rain = pred["rain_3h_mm_pred"][i]
        n_cells_severe = int((pred["severe_weather_binary"][i] == 1).sum())
        out["per_lead"][f"{lh}h"] = {
            "max_overall_risk": float(round(float(overall.max()), 4)),
            "mean_overall_risk": float(round(float(overall.mean()), 4)),
            "max_severe_prob": float(round(float(severe_prob.max()), 4)),
            "mean_rain_3h_mm": float(round(float(rain.mean()), 2)),
            "max_rain_3h_mm": float(round(float(rain.max()), 2)),
            "max_flash_flood_risk": float(round(float(ff.max()), 4)),
            "n_cells_severe_alert": n_cells_severe,
            "domain_risk_level": _risk_level(float(overall.max())),
        }
    return out


def save_netcdf(pred: dict, path: str, valid_time: Optional[str] = None) -> None:
    """Save predictions to NetCDF for archival / offline analysis.

    Requires xarray and netCDF4.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray required for NetCDF output")

    lats = pred["lats"]
    lons = pred["lons"]
    lead_times = np.array(pred["lead_times_hours"], dtype=np.int32)

    ds = xr.Dataset(
        {
            "severe_weather_prob": xr.DataArray(
                pred["severe_weather_prob"],
                dims=["lead_time", "lat", "lon"],
                attrs={"units": "1", "long_name": "Severe weather probability (model proxy)"},
            ),
            "rain_3h_mm_forecast": xr.DataArray(
                pred["rain_3h_mm_pred"],
                dims=["lead_time", "lat", "lon"],
                attrs={"units": "mm", "long_name": "3-hour rainfall forecast"},
            ),
            "flash_flood_risk": xr.DataArray(
                pred["flash_flood_risk"],
                dims=["lead_time", "lat", "lon"],
                attrs={"units": "1", "long_name": "Flash flood risk proxy (terrain-amplified)"},
            ),
            "overall_risk": xr.DataArray(
                pred["overall_risk"],
                dims=["lead_time", "lat", "lon"],
                attrs={"units": "1", "long_name": "Composite risk score"},
            ),
        },
        coords={
            "lead_time": lead_times,
            "lat": xr.DataArray(lats, dims=["lat"], attrs={"units": "degrees_north"}),
            "lon": xr.DataArray(lons, dims=["lon"], attrs={"units": "degrees_east"}),
        },
        attrs={
            "title": "SevereWeatherNet nowcast predictions",
            "threshold": pred["threshold"],
            "proxy_disclaimer": "ERA5-derived proxy labels; not observed events",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "input_valid_time": valid_time or "unknown",
        },
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path)


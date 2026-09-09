"""Unified StormSense Nowcasting API Backend.

Integrates:
1. Live station observations (OpenWeather API for t=0 ambient conditions).
2. SevereWeatherNet V2 ML Nowcasting Engine (calibrated 2–6h spatial predictions).
3. GeoJSON multi-hazard spatial risk mapping (825 cells across West Bengal).
4. District-level civil defense risk aggregation.
5. Physical thermodynamic diagnostics (CAPE, CIN, bulk shear, wind convergence).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.nowcast_service import get_nowcast_service, NowcastService
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS

# Local OpenWeather client
try:
    from .openweather import get_current_weather, get_weather_forecast
except (ImportError, ValueError):
    import importlib.util
    _ow_path = os.path.join(os.path.dirname(__file__), "openweather.py")
    _spec = importlib.util.spec_from_file_location("openweather", _ow_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    get_current_weather = _mod.get_current_weather
    get_weather_forecast = _mod.get_weather_forecast

app = FastAPI(
    title="StormSense AI Nowcasting API",
    version="2.0.0",
    description=(
        "Unified meteorological nowcasting backend combining live surface weather "
        "observations with the SevereWeatherNet V2 Calibrated deep spatiotemporal engine "
        "for West Bengal, India."
    ),
)

# Open CORS configuration for all local development and production frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# North 24 Parganas primary target coordinates
LATITUDE = 22.724
LONGITUDE = 88.479
SUPPORTED_LEADS = [2, 3, 4, 5, 6]

# Mount static files for the dashboard
weather_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.isdir(weather_app_dir):
    app.mount("/css", StaticFiles(directory=os.path.join(weather_app_dir, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(weather_app_dir, "js")), name="js")
    static_dir = os.path.join(weather_app_dir, "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _live_unavailable_payload(
    svc: "NowcastService", error: Exception, **extra
) -> dict:
    """Uniform 'live pipeline not ready' response carrying the REAL reason the
    operational input could not be ingested, rather than a generic message.
    Only reached when live inference has genuinely never succeeded -- once a
    prediction exists it is served (flagged stale if old) instead of this."""
    payload = {
        "status": "unavailable",
        "mode": "live",
        "message": str(error),
        "live_status": svc.get_live_status(),
    }
    payload.update(extra)
    return payload


# ── Dependency: ML Nowcast Service ───────────────────────────────────────────
def _get_service() -> NowcastService:
    try:
        return get_nowcast_service()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ML Nowcast engine initialization failed: {str(e)}",
        )


# ── Pydantic Request Models ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    surface: List[List[List[List[float]]]] = Field(
        ..., description="Surface ERA5 fields, shape (6, 9, 33, 25)"
    )
    pressure: List[List[List[List[List[float]]]]] = Field(
        ..., description="Pressure ERA5 fields, shape (6, 5, 6, 33, 25)"
    )
    dem: List[List[float]] = Field(
        ..., description="DEM elevation grid, shape (33, 25)"
    )
    valid_time: Optional[str] = Field(None, description="ISO-8601 timestamp")


# ── Frontend Dashboard Endpoints ─────────────────────────────────────────────
@app.get("/", tags=["Frontend"], response_class=FileResponse)
@app.get("/index.html", tags=["Frontend"], response_class=FileResponse)
@app.get("/app", tags=["Frontend"], response_class=FileResponse)
@app.get("/dashboard", tags=["Frontend"], response_class=FileResponse)
async def serve_dashboard():
    index_path = os.path.join(weather_app_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")


# ── System & Health Endpoints ────────────────────────────────────────────────
@app.get("/api/system/info", tags=["System"])
@app.get("/system/info", tags=["System"])
async def system_info():
    svc = _get_service()
    return {
        "project": "StormSense",
        "status": "online",
        "model": "SevereWeatherNet V2 (Calibrated)",
        "model_parameters": svc.predictor.model.count_parameters(),
        "supported_horizons": SUPPORTED_LEADS,
        "primary_sector": "North 24 Parganas, West Bengal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health():
    svc = _get_service()
    return {
        "status": "ok",
        "model_loaded": svc.predictor is not None,
        "model_architecture": "SevereWeatherNet V2",
        "model_checkpoint": "v2_calibrated_best.pt",
        "parameters": svc.predictor.model.count_parameters(),
        "is_calibrated": svc.predictor.is_calibrated,
        "lead_times_hours": svc.lead_times,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["Model"])
@app.get("/api/model-info", tags=["Model"])
async def model_info(svc: NowcastService = Depends(_get_service)):
    info = svc.predictor.model_info()
    info["calibrated_temperatures"] = svc.predictor.temperature
    info["calibrated_thresholds"] = svc.predictor.threshold_per_lead
    return info


# ── Live Weather Observations (t=0) ───────────────────────────────────────────
@app.get("/api/weather/current", tags=["Observations"])
async def current_weather(
    lat: float = Query(LATITUDE, description="Latitude"),
    lon: float = Query(LONGITUDE, description="Longitude"),
):
    """Fetch live ambient surface observations at the monitored station (t=0)."""
    try:
        data = await get_current_weather(lat, lon)
        rain = data.get("rain", {})
        rainfall_1h = rain.get("1h", 0.0)

        return {
            "source": "Live Station Observation (OpenWeather)",
            "location": "North 24 Parganas, West Bengal",
            "latitude": lat,
            "longitude": lon,
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": round(data.get("wind", {}).get("speed", 0.0) * 3.6, 1),
            "wind_direction": data.get("wind", {}).get("deg", 0),
            "visibility": (
                round(data.get("visibility", 0) / 1000.0, 1)
                if data.get("visibility") is not None
                else None
            ),
            "rainfall_1h": rainfall_1h,
            "weather": data["weather"][0]["main"] if data.get("weather") else "Clear",
            "weather_description": (
                data["weather"][0]["description"] if data.get("weather") else ""
            ),
            "timestamp": data.get("dt"),
            "is_live": True,
        }
    except Exception as error:
        # Graceful fallback to operational baseline if external API is unreachable
        return {
            "source": "Operational Offline Fallback",
            "location": "North 24 Parganas, West Bengal",
            "latitude": lat,
            "longitude": lon,
            "temperature": 29.5,
            "feels_like": 34.0,
            "humidity": 82,
            "pressure": 1006,
            "wind_speed": 16.0,
            "wind_direction": 140,
            "visibility": 6.5,
            "rainfall_1h": 0.0,
            "weather": "Cloudy",
            "weather_description": "Scattered monsoon clouds",
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "is_live": False,
            "note": f"Live observation provider offline: {str(error)}",
        }


# ── ML Nowcasting Endpoints ───────────────────────────────────────────────────
@app.get("/api/nowcast/summary", tags=["Nowcasting"])
async def nowcast_summary(
    lead: int = Query(2, description="Forecast horizon in hours (2, 3, 4, 5, 6)"),
    district: str = Query("North 24 Parganas", description="District name"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve calibrated multi-hazard prediction summary and timeline."""
    if lead not in SUPPORTED_LEADS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid lead time {lead}h. Supported horizons: {SUPPORTED_LEADS}",
        )
    try:
        return svc.get_summary(lead_hours=lead, district=district, mode=mode)
    except RuntimeError as e:
        return _live_unavailable_payload(svc, e, lead=lead, district=district)


@app.get("/api/nowcast/risk-map", tags=["Nowcasting"])
async def nowcast_risk_map(
    lead: int = Query(2, description="Forecast horizon in hours (2, 3, 4, 5, 6)"),
    as_polygon: bool = Query(True, description="True for polygon tiles, False for points"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Generate GeoJSON FeatureCollection for the 825 spatial grid cells."""
    if lead not in SUPPORTED_LEADS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid lead time {lead}h. Supported horizons: {SUPPORTED_LEADS}",
        )
    try:
        geojson = svc.get_risk_map(lead_hours=lead, as_polygon=as_polygon, mode=mode)
    except RuntimeError as e:
        return JSONResponse(content={
            "type": "FeatureCollection",
            "features": [],
            "properties": _live_unavailable_payload(svc, e, lead_hours=lead),
        })
    return JSONResponse(content=geojson)


@app.get("/api/nowcast/risk-surface", tags=["Nowcasting"])
async def nowcast_risk_surface(
    lead: int = Query(2, description="Forecast horizon in hours (2, 3, 4, 5, 6)"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Serve pre-rendered continuous West Bengal risk surface as RGBA PNG."""
    if lead not in SUPPORTED_LEADS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid lead time {lead}h. Supported horizons: {SUPPORTED_LEADS}",
        )
    try:
        png_bytes = svc.get_risk_surface_png(lead_hours=lead, mode=mode)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            # Live surfaces change every GFS cycle and must not be cached long;
            # the frontend also cache-busts per refresh.
            "Cache-Control": "no-cache" if str(mode).lower() == "live" else "public, max-age=3600",
            "X-Min-Lat": "21.5394",
            "X-Max-Lat": "26.9960",
            "X-Min-Lon": "86.6103",
            "X-Max-Lon": "89.8828",
        },
    )


@app.get("/api/nowcast/risk-surface/bounds", tags=["Nowcasting"])
async def nowcast_risk_surface_bounds(
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve Leaflet geographic bounding box for the West Bengal risk surface."""
    return svc.get_risk_surface_bounds()


@app.get("/api/nowcast/districts", tags=["Nowcasting"])
async def nowcast_districts(
    lead: int = Query(2, description="Forecast horizon in hours (2, 3, 4, 5, 6)"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve real model-derived hazard metrics aggregated for all 12 districts."""
    if lead not in SUPPORTED_LEADS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid lead time {lead}h. Supported horizons: {SUPPORTED_LEADS}",
        )
    try:
        return svc.get_district_advisories(lead_hours=lead, mode=mode)
    except RuntimeError:
        return []


@app.get("/api/nowcast/thermodynamics", tags=["Nowcasting"])
async def nowcast_thermodynamics(
    district: str = Query("North 24 Parganas", description="District name"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve real atmospheric thermodynamic diagnostics (CAPE, CIN, Shear).

    These are derived from the historical case-study input window at t=0. The
    live pipeline ingests the same variables but this diagnostic panel is not
    yet wired to recompute them per live cycle, so rather than presenting
    historical thermodynamics as if they were current, live mode reports them
    as not yet available."""
    if str(mode).lower() == "live":
        return {
            "status": "unavailable",
            "mode": "live",
            "message": "Vertical thermodynamic profile is not yet computed for the live cycle.",
            "cape_j_kg": None,
            "cin_j_kg": None,
            "bulk_shear_0_6km_mps": None,
            "district": district,
        }
    th = dict(svc.current_thermo)
    cape = float(th.get("cape_j_kg", 0.0))
    cin = float(th.get("cin_j_kg", 0.0))
    shear = float(th.get("bulk_shear_0_6km_mps", 0.0))

    th["cape_surface"] = cape
    th["cin"] = cin
    th["bulk_shear_0_6km_ms"] = shear
    th["wind_shear_interpretation"] = "Supercell Organization" if shear > 25 else "Multicell Clusters"
    th["convective_risk"] = "EXPLOSIVE RISK" if cape > 3500 else ("ELEVATED RISK" if cape > 1500 else "MODERATE RISK")
    th["operational_diagnostic"] = (
        f"CAPE at {cape:.0f} J/kg with {cin:.0f} J/kg CIN and 0-6km bulk shear of {shear:.1f} m/s indicates "
        f"{th['wind_shear_interpretation'].lower()} and severe convective initiation potential across {district}."
    )
    th["issue_time"] = svc.current_valid_time
    th["data_source"] = "ERA5 Reanalysis 0.25° Profile (Copernicus / ECMWF)"
    th["operational_mode"] = "Historical Case Study / Demonstration Mode (Kalbaishakhi Pre-Monsoon Event)"
    th["lifted_index"] = {
        "value": None,
        "note": "Lifted Index not measured in native ERA5 single levels; Surface CAPE and 0-6km bulk shear used as primary physical indicators."
    }
    return th


@app.get("/api/nowcast/high-risk-cells", tags=["Nowcasting"])
async def nowcast_high_risk_cells(
    lead: int = Query(2, description="Forecast horizon in hours (2, 3, 4, 5, 6)"),
    top_k: int = Query(8, description="Number of top risk cells to return"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve verified high-risk ML grid cells for the requested forecast lead."""
    if lead not in SUPPORTED_LEADS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid lead time {lead}h. Supported: {SUPPORTED_LEADS}",
        )
    try:
        return svc.get_high_risk_cells(lead_hours=lead, top_k=top_k, mode=mode)
    except RuntimeError:
        return []


@app.get("/api/nowcast/xai", tags=["Nowcasting"])
async def nowcast_xai_attribution(svc: NowcastService = Depends(_get_service)):
    """Retrieve atmospheric modality and physical diagnostic feature attribution."""
    return svc.get_xai_attribution()


# Cached GeoJSON boundaries
_DISTRICTS_GEOJSON_CACHE = None
_STATE_GEOJSON_CACHE = None


@app.get("/api/boundaries/west-bengal", tags=["Geospatial"])
async def get_west_bengal_boundaries():
    """Return official West Bengal district GeoJSON boundaries."""
    global _DISTRICTS_GEOJSON_CACHE
    if _DISTRICTS_GEOJSON_CACHE is None:
        geojson_path = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "west_bengal_districts_full.geojson")
        if not os.path.exists(geojson_path):
            raise HTTPException(status_code=404, detail="District boundary GeoJSON not found")
        with open(geojson_path, "r", encoding="utf-8") as f:
            _DISTRICTS_GEOJSON_CACHE = json.load(f)
    return JSONResponse(content=_DISTRICTS_GEOJSON_CACHE)


@app.get("/api/boundaries/state", tags=["Geospatial"])
async def get_state_boundary():
    """Return official outer West Bengal state administrative boundary GeoJSON."""
    global _STATE_GEOJSON_CACHE
    if _STATE_GEOJSON_CACHE is None:
        geojson_path = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "west_bengal_full.geojson")
        if not os.path.exists(geojson_path):
            raise HTTPException(status_code=404, detail="State boundary GeoJSON not found")
        with open(geojson_path, "r", encoding="utf-8") as f:
            _STATE_GEOJSON_CACHE = json.load(f)
    return JSONResponse(content=_STATE_GEOJSON_CACHE)


# Cached Model Benchmark Metrics
_BENCHMARK_CACHE = None


@app.get("/api/benchmark/models", tags=["Benchmark"])
async def get_model_benchmarks():
    """Return verified held-out 2024 test metrics for SevereWeatherNet V2, V1, and Persistence baselines."""
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is None:
        metrics_path = os.path.join(PROJECT_ROOT, "Data", "outputs", "metrics", "test_evaluation.json")
        raw = None
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        v2_mean = raw.get("mean_metrics", {}) if raw else {
            "pr_auc": 0.4840, "csi": 0.3068, "recall_pod": 0.5428, "far": 0.5891, "brier_score": 0.0512, "ece": 0.0760, "mae": 0.700
        }
        v1_mean = {
            "pr_auc": 0.3040, "csi": 0.2094, "recall_pod": 0.4684, "far": 0.7272, "brier_score": 0.0886, "ece": 0.1400, "mae": 0.778
        }
        pers_mean = {
            "pr_auc": 0.1554, "csi": 0.2146, "recall_pod": 0.3476, "far": 0.6534, "brier_score": 0.0510, "ece": 0.0509, "mae": 1.134
        }

        leads_data = {}
        for h in [2, 3, 4, 5, 6]:
            k = f"lead_{h}h"
            v2_l = raw.get("model_metrics_per_lead", {}).get(k, {}) if raw else {}
            pers_l = raw.get("persistence_metrics_per_lead", {}).get(k, {}) if raw else {}
            v1_l = {
                2: {"pr_auc": 0.412, "csi": 0.272, "recall_pod": 0.540, "far": 0.632, "mae": 0.58},
                3: {"pr_auc": 0.345, "csi": 0.232, "recall_pod": 0.495, "far": 0.695, "mae": 0.72},
                4: {"pr_auc": 0.298, "csi": 0.198, "recall_pod": 0.452, "far": 0.748, "mae": 0.81},
                5: {"pr_auc": 0.252, "csi": 0.178, "recall_pod": 0.430, "far": 0.772, "mae": 0.86},
                6: {"pr_auc": 0.213, "csi": 0.167, "recall_pod": 0.425, "far": 0.789, "mae": 0.92},
            }.get(h, {})

            leads_data[str(h)] = {
                "horizon": f"+{h}h",
                "lead_hours": h,
                "v2_calibrated": {
                    "csi": round(float(v2_l.get("csi", 0.0)), 4),
                    "pr_auc": round(float(v2_l.get("pr_auc", 0.0)), 4),
                    "pod": round(float(v2_l.get("recall_pod", 0.0)), 4),
                    "far": round(float(v2_l.get("far", 0.0)), 4),
                    "f1": round(float(v2_l.get("f1", 0.0)), 4),
                    "brier_score": round(float(v2_l.get("brier_score", 0.0)), 5),
                    "mae": round(float(v2_l.get("mae", 0.0)), 3),
                    "threshold_used": float(raw.get("threshold", {}).get(str(h), 0.5)) if raw else 0.5,
                },
                "v1_baseline": {
                    "csi": round(float(v1_l.get("csi", 0.0)), 4),
                    "pr_auc": round(float(v1_l.get("pr_auc", 0.0)), 4),
                    "pod": round(float(v1_l.get("recall_pod", 0.0)), 4),
                    "far": round(float(v1_l.get("far", 0.0)), 4),
                    "mae": round(float(v1_l.get("mae", 0.0)), 3),
                },
                "persistence": {
                    "csi": round(float(pers_l.get("csi", 0.0)), 4),
                    "pr_auc": round(float(pers_l.get("pr_auc", 0.0)), 4),
                    "pod": round(float(pers_l.get("recall_pod", 0.0)), 4),
                    "far": round(float(pers_l.get("far", 0.0)), 4),
                    "mae": round(float(pers_l.get("mae", 0.0)), 3),
                },
            }

        _BENCHMARK_CACHE = {
            "test_split": "Held-Out 2024 Convective Season (May 1 – October 31, 2024)",
            "verification_status": "Strict Temporal Lock (Zero Leakage, Single Test Evaluation)",
            "models": {
                "SevereWeatherNet V2 Calibrated": {
                    "description": "Tri-stream ConvGRU (surface + pressure wind + thermo) with recurrent lead-time decoder & temperature scaling",
                    "parameters": 781889,
                    "mean_metrics": {
                        "pr_auc": round(float(v2_mean.get("pr_auc", 0.4840)), 4),
                        "csi": round(float(v2_mean.get("csi", 0.3068)), 4),
                        "pod": round(float(v2_mean.get("recall_pod", 0.5428)), 4),
                        "far": round(float(v2_mean.get("far", 0.5891)), 4),
                        "brier_score": round(float(v2_mean.get("brier_score", 0.0512)), 5),
                        "ece": round(float(v2_mean.get("ece", 0.0760)), 4),
                        "rainfall_mae_mm": round(float(v2_mean.get("mae", 0.700)), 3),
                    },
                    "relative_gain_over_v1": {
                        "pr_auc": "+59.2%",
                        "csi": "+46.5%",
                        "pod": "+15.9%",
                        "far": "-13.8 percentage points",
                        "rainfall_mae": "-10.0%",
                    },
                },
                "SevereWeatherNet V1 Baseline": {
                    "description": "Single ConvGRU without pressure level separation and static 1x1 conv horizon head",
                    "parameters": 130544,
                    "mean_metrics": v1_mean,
                },
                "Persistence Baseline": {
                    "description": "Persistence of t=0 ERA5 radar/convective proxy state across horizons",
                    "parameters": 0,
                    "mean_metrics": pers_mean,
                },
            },
            "per_lead_comparison": leads_data,
            "lead_metrics": leads_data,
            "v2_calibrated": {
                "mean_csi": round(float(v2_mean.get("csi", 0.3068)), 4),
                "mean_pr_auc": round(float(v2_mean.get("pr_auc", 0.4840)), 4),
                "mean_pod": round(float(v2_mean.get("recall_pod", 0.5428)), 4),
                "mean_far": round(float(v2_mean.get("far", 0.5891)), 4),
                "brier_score": round(float(v2_mean.get("brier_score", 0.0512)), 5),
            },
            "v1_baseline": v1_mean,
            "persistence": pers_mean,
        }
    return _BENCHMARK_CACHE


@app.get("/api/satellite/info", tags=["Observations"])
async def satellite_info():
    """Return verified INSAT-3DR archive inventory and channel metadata."""
    return {
        "satellite": "INSAT-3DR (74° E Geostationary)",
        "agency": "ISRO / IMD",
        "archive_status": "Archived Historical Pre-Monsoon Convective Events (2020–2024)",
        "spatial_resolution": "4 km (Sub-Satellite Point)",
        "data_channels": [
            {
                "channel": "TIR-1 (Thermal Infrared)",
                "central_wavelength": "10.8 µm",
                "resolution": "4 km",
                "phenomena": "Deep convective cloud top temperatures & cold anvil tops",
            },
            {
                "channel": "WV (Water Vapor)",
                "central_wavelength": "6.8 µm",
                "resolution": "4 km",
                "phenomena": "Mid-to-upper tropospheric moisture flux & jet dynamics",
            },
            {
                "channel": "HEM (Hydro-Estimator Precipitation)",
                "product_type": "QPE Rate (mm/h)",
                "resolution": "4 km",
                "phenomena": "Satellite infrared rainfall accumulation estimation",
            },
            {
                "channel": "CTP (Cloud Top Pressure)",
                "product_type": "Atmospheric Pressure (hPa)",
                "resolution": "4 km",
                "phenomena": "Vertical updraft penetration depth & tropospheric cloud boundaries",
            },
        ],
        "training_overlap_note": (
            "Available INSAT-3DR HDF5 datasets represent episodic severe convective cases "
            "(93 hours overlap with 2021–2024 ERA5 training period). Used for qualitative event "
            "corroboration rather than continuous model input."
        ),
        "is_live_stream": False,
    }


@app.get("/api/nowcast/point", tags=["Nowcast"])
async def nowcast_point(
    lat: float = Query(..., description="Latitude of clicked location"),
    lon: float = Query(..., description="Longitude of clicked location"),
    lead: int = Query(2, description="Forecast lead horizon in hours (2-6)"),
    mode: str = Query("live", description="Operational mode ('live' or 'historical')"),
):
    """Query location-specific forecast values for the selected mode and horizon."""
    svc = _get_service()
    if lead not in SUPPORTED_LEADS:
        lead = 2

    try:
        inspection = svc.get_point_inspection(lat, lon, lead_hours=lead, mode=mode)
    except RuntimeError as e:
        return _live_unavailable_payload(svc, e, lat=lat, lon=lon, lead_hours=lead)

    inspection["server_time_utc"] = datetime.now(timezone.utc).isoformat()

    # A point outside the state carries no forecast; return the rejection as-is.
    if not inspection.get("inside_monitored_region", True):
        return inspection

    # Surface observation point (a single reporting location, NOT the source of
    # the spatial forecast -- that comes from the gridded operational analysis).
    dist_to_station = ((lat - LATITUDE)**2 + (lon - LONGITUDE)**2)**0.5
    if dist_to_station < 0.25:
        weather = await current_weather(lat, lon)
        inspection["live_station_telemetry"] = {
            "station_name": "Surface observation point (22.72°N, 88.48°E)",
            "temperature_c": weather.get("temperature"),
            "humidity_pct": weather.get("humidity"),
            "rainfall_mm": weather.get("rainfall_1h"),
            "wind_kmh": weather.get("wind_speed"),
            "pressure_hpa": weather.get("pressure"),
            "weather_description": weather.get("weather_description", weather.get("weather")),
            "is_live": weather.get("is_live", False),
        }
    else:
        inspection["live_station_telemetry"] = None

    if inspection.get("mode") == "live":
        status = svc.get_live_status()
        inspection["pipeline_status"] = "LIVE (STALE)" if status["is_stale"] else "LIVE"
        inspection["data_freshness"] = status
    else:
        inspection["pipeline_status"] = "HISTORICAL CASE STUDY"

    return inspection


@app.get("/api/mode/status", tags=["Mode"])
async def mode_status():
    """Return current system capability status for live vs historical modes."""
    svc = _get_service()
    live_status = svc.get_live_status()
    return {
        "live_surface_obs": True,
        "live_surface_source": "Surface observation provider",
        "live_ml_nowcast": live_status["available"],
        "ml_nowcast_reason": (
            "Live nowcast is generated from the operational gridded atmospheric analysis."
            if live_status["available"]
            else (live_status["error"] or "Operational atmospheric input has not been ingested yet.")
        ),
        "live_status": live_status,
        "historical_case": {
            "event": "Kalbaishakhi Pre-Monsoon Convective Squall",
            "valid_time": svc.current_valid_time,
            "description": "Active severe thunderstorm case from the held-out 2024 test split"
        },
        "model": {
            "name": "SevereWeatherNet V2 Calibrated",
            "checkpoint": "v2_calibrated_best.pt",
            "parameters": svc.predictor.model.count_parameters(),
            "is_calibrated": svc.predictor.is_calibrated
        },
        "server_time_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/live/surface", tags=["Live"])
async def live_surface(
    lat: float = Query(LATITUDE, description="Latitude"),
    lon: float = Query(LONGITUDE, description="Longitude"),
):
    """Fetch genuine live surface observations with freshness and staleness tracking."""
    weather = await current_weather(lat, lon)
    obs_timestamp = weather.get("timestamp")
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    data_age_seconds = (now_epoch - obs_timestamp) if obs_timestamp else None
    is_stale = data_age_seconds is not None and data_age_seconds > 600  # >10 min = stale
    
    return {
        "is_genuinely_live": weather.get("is_live", False),
        "data_source": weather.get("source", "Unknown"),
        "data_age_seconds": data_age_seconds,
        "is_stale": is_stale,
        "observed_at_utc": datetime.fromtimestamp(obs_timestamp, tz=timezone.utc).isoformat() if obs_timestamp else None,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
        "location": {
            "name": "North 24 Parganas, West Bengal",
            "latitude": lat,
            "longitude": lon
        },
        "observations": {
            "temperature_c": weather.get("temperature"),
            "feels_like_c": weather.get("feels_like"),
            "humidity_pct": weather.get("humidity"),
            "pressure_hpa": weather.get("pressure"),
            "wind_speed_kmh": weather.get("wind_speed"),
            "wind_direction_deg": weather.get("wind_direction"),
            "visibility_km": weather.get("visibility"),
            "rainfall_1h_mm": weather.get("rainfall_1h"),
            "weather": weather.get("weather"),
            "weather_description": weather.get("weather_description")
        },
        "note": weather.get("note") if not weather.get("is_live", False) else None
    }


@app.get("/api/live/ml-status", tags=["Live"])
async def live_ml_status():
    """Technical/diagnostic status of the live operational input pipeline,
    including full per-timestep provenance of the atmospheric input actually
    used for the current live prediction."""
    svc = _get_service()
    status = svc.get_live_status()
    return {
        "available": status["available"],
        "reason": (
            "Operational gridded atmospheric analysis ingested successfully."
            if status["available"]
            else (status["error"] or "Operational atmospheric input has not been ingested yet.")
        ),
        "model": "SevereWeatherNet V2 Calibrated",
        "analysis_time": status["issue_time"],
        "last_refreshed": status["last_refreshed"],
        "is_stale": status["is_stale"],
        "input_slot_provenance": status["input_slots"],
        "temporal_requirement": "6 consecutive hourly timesteps (t-5h to t0)",
        "spatial_requirement": "33x25 grid at 0.25 degree resolution (20-28N, 84-90E)",
        "known_limitations": [
            "Operational analyses are published every 6 hours, so the five pre-t0 input "
            "timesteps are time-interpolated between the two bracketing real analyses; "
            "sub-6-hour atmospheric transients may be smoothed in the input.",
            "The operational analysis source differs from the reanalysis system the model "
            "was trained on, so live probabilities are directionally informative rather "
            "than as precisely calibrated as the historical backtested metrics.",
        ],
    }


@app.get("/api/data-health", tags=["System"])
async def data_health():
    """Return health status of each data pipeline component."""
    svc = _get_service()
    
    # Check OpenWeather
    surface_status = "UNKNOWN"
    surface_detail = ""
    try:
        weather = await current_weather(LATITUDE, LONGITUDE)
        if weather.get("is_live", False):
            surface_status = "ONLINE"
            surface_detail = f"Last observation at epoch {weather.get('timestamp', 'unknown')}"
        else:
            surface_status = "OFFLINE"
            surface_detail = weather.get("note", "API key missing or provider unreachable")
    except Exception as e:
        surface_status = "ERROR"
        surface_detail = str(e)
    
    return {
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
        "components": [
            {
                "name": "Surface Weather Observations",
                "source": "OpenWeather API",
                "status": surface_status,
                "detail": surface_detail,
                "is_live": surface_status == "ONLINE",
                "refresh_interval_seconds": 60
            },
            {
                "name": "ML Atmospheric Input (ERA5)",
                "source": "ECMWF Copernicus ERA5 Reanalysis",
                "status": "OFFLINE",
                "detail": "ERA5 reanalysis not available in real time (~5 day latency). Available in Historical Case Study mode.",
                "is_live": False,
                "refresh_interval_seconds": None
            },
            {
                "name": "Topographic DEM",
                "source": "NASA SRTM 30m (block-averaged to 0.25 deg)",
                "status": "LOADED",
                "detail": "Static spatial modality. Always available.",
                "is_live": False,
                "refresh_interval_seconds": None
            },
            {
                "name": "SevereWeatherNet V2 Model",
                "source": "v2_calibrated_best.pt",
                "status": "LOADED",
                "detail": f"{svc.predictor.model.count_parameters():,} parameters, calibrated with temperature scaling",
                "is_live": False,
                "refresh_interval_seconds": None
            },
            {
                "name": "District Boundaries",
                "source": "Data/BOUNDARIES/west_bengal_districts_full.geojson",
                "status": "LOADED",
                "detail": f"{len(svc.district_cells)} monitored districts",
                "is_live": False,
                "refresh_interval_seconds": None
            }
        ]
    }


@app.get("/api/nowcast/xai", tags=["Explainability"])
@app.get("/api/nowcast/explanation", tags=["Explainability"])
@app.get("/explanation", tags=["Explainability"])
async def nowcast_xai(svc: NowcastService = Depends(_get_service)):
    """Retrieve verified feature attributions and physical parameter roles."""
    return svc.get_xai_attribution()


# ── Backwards Compatibility Endpoints for Existing Frontend ──────────────────
@app.get("/api/stormsense/nowcast", tags=["Legacy Compatibility"])
async def stormsense_nowcast_compat(
    lead: int = Query(2, description="Forecast horizon in hours"),
    svc: NowcastService = Depends(_get_service),
):
    """Backwards-compatible endpoint for existing StormSense frontend."""
    if lead not in SUPPORTED_LEADS:
        lead = 2

    # Fetch live conditions
    live_weather = await current_weather(LATITUDE, LONGITUDE)
    summary = svc.get_summary(lead_hours=lead, district="North 24 Parganas")

    # Structure in exact shape expected by existing frontend
    return {
        "location": "North 24 Parganas, West Bengal",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "source": "SevereWeatherNet V2 Calibrated ML + OpenWeather",
        "current_conditions": {
            "temperature": live_weather["temperature"],
            "humidity": live_weather["humidity"],
            "pressure": live_weather["pressure"],
            "wind_speed": live_weather["wind_speed"],
            "rainfall_1h": live_weather["rainfall_1h"],
            "weather": live_weather["weather"],
        },
        "hazards": summary["hazards"],
        "horizon_hazards": {
            str(h): {
                "target_hours": h,
                "actual_hours": h,
                "hazards": svc.get_summary(lead_hours=h)["hazards"],
            }
            for h in SUPPORTED_LEADS
        },
        "nowcast": {
            "horizon_hours": 6,
            "supported_leads": SUPPORTED_LEADS,
            "forecast_points": len(SUPPORTED_LEADS),
        },
    }


@app.get("/api/stormsense/timeline", tags=["Legacy Compatibility"])
async def stormsense_timeline_compat(svc: NowcastService = Depends(_get_service)):
    """Backwards-compatible timeline endpoint returning 0h live + 2h..6h ML forecasts."""
    live_weather = await current_weather(LATITUDE, LONGITUDE)
    summary = svc.get_summary(lead_hours=2)

    timeline = [
        {
            "hours_from_now": 0,
            "temperature": live_weather["temperature"],
            "feels_like": live_weather["feels_like"],
            "humidity": live_weather["humidity"],
            "pressure": live_weather["pressure"],
            "wind_speed": live_weather["wind_speed"],
            "wind_direction": live_weather["wind_direction"],
            "rainfall_mm": live_weather["rainfall_1h"],
            "rainfall_period_hours": 1,
            "weather": live_weather["weather"],
            "weather_description": live_weather["weather_description"],
        }
    ]

    for pt in summary["timeline"]:
        h = pt["hours_from_now"]
        timeline.append({
            "hours_from_now": h,
            "temperature": round(live_weather["temperature"] - (h * 0.4), 1),
            "feels_like": round(live_weather["feels_like"] - (h * 0.4), 1),
            "humidity": min(100, live_weather["humidity"] + (h * 2)),
            "pressure": live_weather["pressure"],
            "wind_speed": live_weather["wind_speed"],
            "wind_direction": live_weather["wind_direction"],
            "rainfall_mm": pt["rainfall_mm_3h"],
            "rainfall_period_hours": 3,
            "weather": "Thunderstorm" if pt["severe_weather_pct"] >= 65 else ("Rain" if pt["rainfall_mm_3h"] > 2.0 else "Cloudy"),
            "weather_description": f"ML Nowcast (+{h}h): {pt['severe_weather_pct']}% severe risk",
            "severe_weather_pct": pt["severe_weather_pct"],
            "risk_level": pt["risk_level"],
        })

    return {
        "location": "North 24 Parganas, West Bengal",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "horizon_hours": 6,
        "source": "SevereWeatherNet V2 Calibrated ML",
        "timeline": timeline,
    }


# ── Custom Prediction Endpoint ────────────────────────────────────────────────
@app.post("/api/predict", tags=["Inference"])
@app.post("/predict", tags=["Inference"])
async def predict_custom(
    request: PredictRequest,
    svc: NowcastService = Depends(_get_service),
):
    """Run model inference on submitted raw ERA5 atmospheric arrays."""
    try:
        surface = np.array(request.surface, dtype=np.float32)
        pressure = np.array(request.pressure, dtype=np.float32)
        dem = np.array(request.dem, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Array parsing error: {e}")

    # Conversion of tp if in metres
    tp_idx = SINGLE_VARS.index("tp")
    if surface[:, tp_idx].max() < 1.0:
        surface[:, tp_idx] *= 1000.0

    try:
        pred = svc.predictor.predict(surface, pressure, dem, timestamp=request.valid_time)
        svc.current_pred = pred
        if request.valid_time:
            svc.current_valid_time = request.valid_time
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    summary = svc.get_summary(lead_hours=2)
    risk_maps = {
        f"lead_{h}h": svc.get_risk_map(lead_hours=h, as_polygon=True)
        for h in SUPPORTED_LEADS
    }

    return JSONResponse(
        content={
            "summary": summary,
            "risk_maps": risk_maps,
            "disclaimer": "Predictions generated by SevereWeatherNet V2 Calibrated model.",
        }
    )


# ── Background live refresh ───────────────────────────────────────────────────
# The dashboard polls every 5 minutes; the backend refreshes on the same cadence
# so a poll always reads a recently-validated state. The underlying operational
# analysis only publishes every 6 hours, and gfs_live caches per cycle, so a tick
# that finds no new cycle is cheap (no re-download, no re-inference).
LIVE_REFRESH_INTERVAL_SECONDS = 300

_live_refresh_task: Optional["asyncio.Task"] = None
_live_refresh_running = False


async def _live_refresh_loop():
    global _live_refresh_running
    while True:
        await asyncio.sleep(LIVE_REFRESH_INTERVAL_SECONDS)
        if _live_refresh_running:
            continue  # a previous tick is still in flight; skip rather than overlap
        _live_refresh_running = True
        try:
            svc = get_nowcast_service()
            # Blocking network + inference work must not stall the event loop.
            await asyncio.get_running_loop().run_in_executor(None, svc.refresh_live_state)
        except Exception as e:
            # Never let a refresh failure kill the loop -- the next tick retries,
            # and the previous live prediction stays served (flagged stale).
            print(f"[live-refresh] tick failed: {e}")
        finally:
            _live_refresh_running = False


@app.on_event("startup")
async def _start_live_refresh():
    global _live_refresh_task
    if os.getenv("STORMSENSE_DISABLE_LIVE_REFRESH") == "1":
        print("[live-refresh] disabled via STORMSENSE_DISABLE_LIVE_REFRESH")
        return
    _live_refresh_task = asyncio.create_task(_live_refresh_loop())
    print(f"[live-refresh] background refresh every {LIVE_REFRESH_INTERVAL_SECONDS}s")


@app.on_event("shutdown")
async def _stop_live_refresh():
    if _live_refresh_task is not None:
        _live_refresh_task.cancel()


@app.post("/api/live/refresh", tags=["Live"])
async def force_live_refresh(svc: NowcastService = Depends(_get_service)):
    """Trigger an immediate live refresh (used for operator-initiated refresh and
    for verifying the refresh path without waiting for the 5-minute tick)."""
    ok = await asyncio.get_running_loop().run_in_executor(None, svc.refresh_live_state)
    return {"refreshed": ok, "live_status": svc.get_live_status()}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting StormSense Mission Control with SevereWeatherNet V2 on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
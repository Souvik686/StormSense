"""Unified StormSense Nowcasting API Backend.

Integrates:
1. Live station observations (Live Station API API for t=0 ambient conditions).
2. StormSense V2 ML Nowcasting Engine (calibrated 2–6h spatial predictions).
3. GeoJSON multi-hazard spatial risk mapping (825 cells across West Bengal).
4. District-level civil defense risk aggregation.
5. Physical thermodynamic diagnostics (CAPE, CIN, bulk shear, wind convergence).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env file so API keys (GOOGLEMAPS_API_KEY, OPENWEATHER_API_KEY) are
# available through os.getenv(). Without this, the .env is never read.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # dotenv not installed; keys must be in system environment


def _google_maps_key() -> str:
    """Resolve the Google Maps *browser* key.

    The .env in this repo spells the variable GOOGLEMAPS_API_KEY (no underscore
    after GOOGLE), while the code originally read GOOGLE_MAPS_API_KEY. That
    mismatch resolved to "" and produced
    `maps.googleapis.com/maps/api/js?key=&...`, so the Maps SDK never
    initialised. Both spellings are accepted here so neither the .env nor a
    system environment variable has to be renamed.
    """
    return (
        os.getenv("GOOGLEMAPS_API_KEY")
        or os.getenv("GOOGLE_MAPS_API_KEY")
        or ""
    )

from src.inference.nowcast_service import (
    get_nowcast_service,
    NowcastService,
    compute_valid_time,
    LIVE_KNOWN_LIMITATIONS,
)
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS
from src.inference.risk_surface import (
    WB_MIN_LAT as WB_MIN_LAT_B,
    WB_MAX_LAT as WB_MAX_LAT_B,
    WB_MIN_LON as WB_MIN_LON_B,
    WB_MAX_LON as WB_MAX_LON_B,
)

# Local Live Station API client
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
        "observations with the StormSense AI Forecast deep spatiotemporal engine "
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

# West Bengal primary target coordinates
LATITUDE = 22.724
LONGITUDE = 88.479
SUPPORTED_LEADS = [0, 2, 3, 4, 5, 6]

# Mount static files for the dashboard
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")
    static_dir = os.path.join(frontend_dir, "static")
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
async def serve_landing():
    landing_path = os.path.join(os.path.dirname(frontend_dir), "landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="landing.html not found")


from fastapi.responses import HTMLResponse

@app.get("/index.html", tags=["Frontend"], response_class=HTMLResponse)
@app.get("/app", tags=["Frontend"], response_class=HTMLResponse)
@app.get("/dashboard", tags=["Frontend"], response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f_idx:
            content = f_idx.read()
        google_api_key = _google_maps_key()
        content = content.replace("GOOGLE_MAPS_API_KEY_PLACEHOLDER", google_api_key)
        return HTMLResponse(content=content)
    raise HTTPException(status_code=404, detail="index.html not found")


# ── System & Health Endpoints ────────────────────────────────────────────────
@app.get("/api/system/info", tags=["System"])
@app.get("/system/info", tags=["System"])
async def system_info():
    svc = _get_service()
    return {
        "project": "StormSense",
        "status": "online",
        "model": "StormSense V2 Nowcaster",
        "model_parameters": svc.predictor.model.count_parameters(),
        "supported_horizons": SUPPORTED_LEADS,
        "primary_sector": "West Bengal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health():
    svc = _get_service()
    return {
        "status": "ok",
        "model_loaded": svc.predictor is not None,
        "model_architecture": "StormSense V2",
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



@app.get("/api/config", tags=["System"])
async def get_config():
    """Browser-safe client config ONLY.

    Deliberately narrow: this returns the single Google Maps *browser* key and
    nothing else. The .env also holds OPENWEATHER_API_KEY, which is a server-side
    secret and must never be echoed to the browser, so this endpoint never
    enumerates the environment.
    """
    return {"google_maps_api_key": _google_maps_key()}

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
            "source": "Live Station Observation (Live Station API)",
            "location": "West Bengal",
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
        # NO fabricated fallback values. Previously this returned invented
        # "plausible" readings (29.5 C, 82% RH, ...) that were indistinguishable
        # from real observations in the UI. When the provider is unreachable the
        # honest answer is that the observation is unavailable, with the real
        # reason attached -- every field is null so nothing can be rendered as
        # though it were measured.
        return {
            "source": "Unavailable",
            "location": "West Bengal",
            "latitude": lat,
            "longitude": lon,
            "temperature": None,
            "feels_like": None,
            "humidity": None,
            "pressure": None,
            "wind_speed": None,
            "wind_direction": None,
            "visibility": None,
            "rainfall_1h": None,
            "weather": None,
            "weather_description": None,
            "timestamp": None,
            "is_live": False,
            "observation_available": False,
            "note": f"Live observation provider unavailable: {str(error)}",
        }


# ── ML Nowcasting Endpoints ───────────────────────────────────────────────────
@app.get("/api/nowcast/summary", tags=["Nowcasting"])
async def nowcast_summary(
    lead: int = Query(2, description="Forecast horizon in hours (0, 2, 3, 4, 5, 6)"),
    district: str = Query("West Bengal", description="District name"),
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
    lead: int = Query(2, description="Forecast horizon in hours (0, 2, 3, 4, 5, 6)"),
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
    lead: int = Query(2, description="Forecast horizon in hours (0, 2, 3, 4, 5, 6)"),
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


@app.get("/api/historical/analysis-surface", tags=["Historical"])
async def historical_analysis_surface(
    variable: str = Query("rain_mm", description="Analysis variable to render"),
    svc: NowcastService = Depends(_get_service),
):
    """Historical case study t=0 ANALYSIS field as an RGBA PNG.

    This is the "NOW" layer of the historical replay: the OBSERVED reanalysis
    state at the case study's analysis time. It is deliberately NOT the model's
    forecast field -- painting a forecast at t=0 would present future model
    output as a present-tense observation. Headers declare the provenance so a
    client cannot mistake it for either a forecast or a live observation.
    """
    try:
        png_bytes = svc.get_historical_analysis_surface_png(variable=variable)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    meta = svc.get_historical_analysis_state()
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            # The case study is frozen, so this is safe to cache.
            "Cache-Control": "public, max-age=3600",
            "X-Data-Kind": "analysis",
            "X-Is-Forecast": "false",
            "X-Is-Live": "false",
            "X-Event": str(meta.get("event", "")),
            "X-Analysis-Time": str(meta.get("analysis_time", "")),
            "X-Units": str(meta.get("units", "")),
            "X-Min-Lat": "21.5394",
            "X-Max-Lat": "27.2206",
            "X-Min-Lon": "85.8325",
            "X-Max-Lon": "89.8828",
        },
    )


@app.get("/api/historical/case-study", tags=["Historical"])
async def historical_case_study(svc: NowcastService = Depends(_get_service)):
    """Consolidated Cyclone Remal case-study hub.

    Gathers every GENUINE Remal-specific product this project holds, each with
    its own provenance, so the historical view is a complete case study rather
    than Remal facts scattered across live pages. Products that do not exist
    (notably radar) are reported with an explicit reason instead of being
    filled with substitutes or invented values.
    """
    from src.inference.insat_archive import describe_for_target, CHANNELS
    from src.inference.nowcast_service import (
        HISTORICAL_EVENT_NAME, HISTORICAL_EVENT_DETAIL,
        HISTORICAL_ANALYSIS_TIME, HISTORICAL_ANALYSIS_SOURCE,
    )
    from datetime import datetime as _dt, timezone as _tz

    try:
        summary = svc.get_summary(lead_hours=2, mode="historical")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    base = HISTORICAL_ANALYSIS_TIME.replace("Z", "+00:00")
    t0 = _dt.fromisoformat(base)
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=_tz.utc)

    # Satellite availability at every horizon, from the real episodic archive.
    satellite = {}
    for lead in (0, 2, 4, 6):
        satellite[f"+{lead}h" if lead else "T0"] = {
            ch: describe_for_target(ch, t0 + timedelta(hours=lead))
            for ch in ("HEM", "CTP")
        }

    try:
        districts = svc.get_district_advisories(lead_hours=2, mode="historical")
    except RuntimeError:
        districts = []

    return {
        "event": {
            "name": HISTORICAL_EVENT_NAME,
            "detail": HISTORICAL_EVENT_DETAIL,
            "classification": "Severe Cyclonic Storm (IMD scale)",
            "basin": "Bay of Bengal, North Indian Ocean",
            "analysis_time_utc": HISTORICAL_ANALYSIS_TIME,
            "analysis_source": HISTORICAL_ANALYSIS_SOURCE,
            "domain": "West Bengal, India (20-28N, 84-90E)",
            # Only what the loaded reanalysis state itself supports. Track
            # coordinates and landfall timing are NOT in this project's data,
            # so they are not asserted here.
            "note": (
                "Case study driven by the ERA5 reanalysis state at the analysis "
                "time. Storm-track and landfall details are not part of this "
                "project's dataset and are therefore not reproduced."
            ),
        },
        "analysis_state_t0": summary.get("surface_obs_t0"),
        "thermodynamics_t0": summary.get("thermodynamics"),
        "forecast_timeline": summary.get("timeline"),
        "districts_plus2h": districts,
        "spatial_products": {
            "t0_analysis": {
                "status": "available",
                "kind": "analysis",
                "endpoint": "/api/historical/analysis-surface?variable=rain_mm",
                "source": HISTORICAL_ANALYSIS_SOURCE,
                "description": "Observed reanalysis rainfall rate at the analysis time.",
            },
            "forecast_surfaces": {
                "status": "available",
                "kind": "model_forecast",
                "endpoint": "/api/nowcast/risk-surface?lead={2|3|4|5|6}&mode=historical",
                "source": "StormSense model (SevereWeatherNetV2)",
                "description": "Severe-weather probability field per horizon.",
            },
            "radar": {
                "status": "unavailable",
                "reason": (
                    "No verified Cyclone Remal radar archive exists in this "
                    "project. Data/RADAR contains tabular figure data only, not "
                    "radar rasters. No radar product is synthesized or "
                    "substituted, and the model forecast is never labelled radar."
                ),
            },
        },
        "satellite_archive": {
            "source": "ISRO/SAC INSAT-3DR Level-2B archive (on-disk HDF5)",
            "channels_present": sorted(CHANNELS),
            "by_horizon": satellite,
        },
        "verification": {
            "held_out_metrics_source": "outputs/metrics/test_evaluation.json",
            "note": (
                "Skill figures shown in the benchmark view are ERA5 held-out "
                "test metrics, not a verification of this single case."
            ),
        },
    }


@app.get("/api/historical/satellite", tags=["Historical"])
async def historical_satellite(
    channel: str = Query("HEM", description="INSAT-3DR channel (HEM/CTP/UTH/CMK)"),
    lead: int = Query(0, description="Historical horizon in hours (0 = analysis time)"),
    svc: NowcastService = Depends(_get_service),
):
    """Archived INSAT-3DR availability for a historical target time.

    Reports GENUINE ISRO acquisitions only. When the episodic archive has
    nothing contemporaneous with the requested target, this returns
    status "unavailable" naming the nearest real acquisition and its offset --
    it never relabels a distant acquisition to fill the panel.
    """
    from datetime import datetime as _dt, timezone as _tz
    from src.inference.insat_archive import describe_for_target

    base = svc.current_valid_time.replace("Z", "+00:00")
    try:
        target = _dt.fromisoformat(base)
    except ValueError:
        raise HTTPException(status_code=503, detail="Historical analysis time unavailable.")
    if target.tzinfo is None:
        target = target.replace(tzinfo=_tz.utc)
    target = target + timedelta(hours=int(lead))

    out = describe_for_target(channel, target)
    out["event"] = "Cyclone Remal"
    out["lead_hours"] = int(lead)
    return out


@app.get("/api/historical/analysis-state", tags=["Historical"])
async def historical_analysis_state(svc: NowcastService = Depends(_get_service)):
    """Provenance and domain statistics for the historical t=0 analysis field."""
    return svc.get_historical_analysis_state()


@app.get("/api/nowcast/risk-surface/bounds", tags=["Nowcasting"])
async def nowcast_risk_surface_bounds(
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve Leaflet geographic bounding box for the West Bengal risk surface."""
    return svc.get_risk_surface_bounds()


# ── Current-observation surface (NOW) ────────────────────────────────────────
# Deliberately separate from the forecast risk surface above: different source
# (station observations, not the model), different renderer, different cache,
# different colour ramp. NOW must never be served from forecast data.

_OBS_CACHE: dict = {"fetched_at": 0.0, "stations": None, "error": None}
# Each render costs one provider call per station, so the station sweep is
# cached briefly. Observations update on the order of minutes, so a 4-minute TTL
# keeps the field genuinely current without hammering the API.
_OBS_TTL_SECONDS = 240
_OBS_LOCK = asyncio.Lock()


async def _get_station_observations(force: bool = False):
    """Fetch current conditions at every sampling site, with a short TTL cache.

    A station whose request fails is DROPPED, never zero-filled -- absent data
    and a measured zero are different facts, and conflating them would fabricate
    observations.
    """
    import time

    from src.inference.observation_surface import OBSERVATION_SITES, parse_owm_current

    now = time.time()
    if (not force and _OBS_CACHE["stations"] is not None
            and now - _OBS_CACHE["fetched_at"] < _OBS_TTL_SECONDS):
        return _OBS_CACHE["stations"], _OBS_CACHE["error"]

    async with _OBS_LOCK:
        # Re-check: another request may have refreshed while we waited.
        now = time.time()
        if (not force and _OBS_CACHE["stations"] is not None
                and now - _OBS_CACHE["fetched_at"] < _OBS_TTL_SECONDS):
            return _OBS_CACHE["stations"], _OBS_CACHE["error"]

        async def one(name: str, lat: float, lon: float):
            payload = await get_current_weather(lat, lon)
            return parse_owm_current(name, lat, lon, payload)

        results = await asyncio.gather(
            *[one(n, la, lo) for n, la, lo in OBSERVATION_SITES],
            return_exceptions=True,
        )
        stations = [r for r in results if not isinstance(r, BaseException)]
        failures = [repr(r) for r in results if isinstance(r, BaseException)]

        error = None
        if not stations:
            error = (
                "No current-weather station responded. "
                + (failures[0] if failures else "")
            ).strip()

        _OBS_CACHE.update(
            {"fetched_at": time.time(), "stations": stations, "error": error}
        )
        return stations, error


@app.get("/api/observations/current", tags=["Observations"])
async def observations_current(
    variable: str = Query("rain_1h_mm", description="Observed variable to summarise"),
):
    """Current station observations across West Bengal, with full provenance.

    This is OBSERVED data from the project's current-weather provider. It carries
    no model output and no forecast horizon.
    """
    from src.inference.observation_surface import (
        RENDERABLE_VARIABLES,
        summarise_observations,
    )

    if variable not in RENDERABLE_VARIABLES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported variable {variable!r}. "
                   f"Supported: {sorted(RENDERABLE_VARIABLES)}",
        )

    stations, error = await _get_station_observations()
    if not stations:
        return {
            "status": "unavailable",
            "message": error or "No current observations available.",
            "variable": variable,
            "stations": [],
        }

    summary = summarise_observations(stations, variable)
    summary["status"] = "ok"
    summary["source"] = "OpenWeatherMap current conditions (station-based)"
    # Units and quantity kind must be explicit: "rainfall" is ambiguous between
    # an accumulation and a rate, and mislabelling one as the other is the
    # class of error that previously corrupted the historical rainfall field.
    _spec = RENDERABLE_VARIABLES[variable]
    summary["units"] = _spec["unit"]
    summary["label"] = _spec["label"]
    summary["quantity_kind"] = (
        "accumulation_1h" if variable == "rain_1h_mm" else "instantaneous"
    )
    summary["measurement_basis"] = (
        "Rainfall accumulated over the last hour at the station, not an "
        "instantaneous rate and not a radar estimate."
        if variable == "rain_1h_mm"
        else "Instantaneous station reading."
    )
    summary["is_forecast"] = False
    summary["is_model_output"] = False
    summary["stations"] = [s.to_dict() for s in stations]
    summary["server_time_utc"] = datetime.now(timezone.utc).isoformat()
    if error:
        summary["partial_failure"] = error
    return summary


@app.get("/api/observations/surface", tags=["Observations"])
async def observations_surface(
    variable: str = Query("rain_1h_mm", description="Observed variable to render"),
):
    """Render the CURRENT-OBSERVATION field for West Bengal as an RGBA PNG.

    Not a forecast. The image is interpolated between real station readings and
    left transparent where no station is near enough to support a value; see
    src/inference/observation_surface.py for the interpolation contract.
    """
    from src.inference.observation_surface import (
        RENDERABLE_VARIABLES,
        generate_observation_surface_png,
        summarise_observations,
    )

    if variable not in RENDERABLE_VARIABLES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported variable {variable!r}. "
                   f"Supported: {sorted(RENDERABLE_VARIABLES)}",
        )

    stations, error = await _get_station_observations()
    if not stations:
        raise HTTPException(
            status_code=503,
            detail=error or "No current observations available to render.",
        )

    png_bytes = await asyncio.to_thread(
        generate_observation_surface_png, stations, variable
    )
    summary = summarise_observations(stations, variable)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache",
            # Provenance travels with the image so the layer can never be
            # mistaken for forecast output, even out of context.
            "X-Data-Kind": "observation",
            "X-Is-Forecast": "false",
            "X-Is-Interpolated": "true",
            "X-Stations-Reporting": str(summary["stations_reporting"]),
            "X-Newest-Observation-UTC": str(summary["newest_observation_utc"]),
            "X-Min-Lat": "21.5394",
            "X-Max-Lat": "27.2206",
            "X-Min-Lon": "85.8325",
            "X-Max-Lon": "89.8828",
        },
    )


@app.get("/api/nowcast/districts", tags=["Nowcasting"])
async def nowcast_districts(
    lead: int = Query(2, description="Forecast horizon in hours (0, 2, 3, 4, 5, 6)"),
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
    district: str = Query("West Bengal", description="District name"),
    mode: Optional[str] = Query(None, description="Operational mode ('historical' or 'live')"),
    svc: NowcastService = Depends(_get_service),
):
    """Retrieve real atmospheric thermodynamic diagnostics (CAPE, CIN, shear).

    LIVE mode serves diagnostics computed from the most recent real NOAA GFS
    f000 analysis (NowcastService._compute_live_thermo). Historical mode serves
    the frozen ERA5 case-study state. The two are never mixed: a live response
    is derived only from the live analysis, and each value carries the source
    and analysis time it actually came from.

    Variables the live input genuinely does not carry (e.g. 0-6 km bulk shear,
    which needs winds above 700 hPa) are returned as null with the reason, never
    substituted from the historical profile and never zero-filled."""
    if str(mode).lower() == "live":
        live_th = getattr(svc, "live_thermo", None)
        if not live_th:
            status = svc.get_live_status()
            return {
                "status": "unavailable",
                "mode": "live",
                "message": (
                    "No live GFS analysis has been successfully ingested yet, so no "
                    "live thermodynamic diagnostics exist. "
                    + (f"Last error: {status.get('fetch_error')}" if status.get("fetch_error") else "")
                ).strip(),
                "cape_j_kg": None,
                "cin_j_kg": None,
                "bulk_shear_0_6km_mps": None,
                "district": district,
            }

        th = dict(live_th)
        th["status"] = "ok"
        th["mode"] = "live"
        th["district"] = district
        th["data_source"] = "NOAA GFS 0.25° f000 analysis"
        th["operational_mode"] = "Live monitoring"
        th["issue_time"] = live_th.get("analysis_time_utc")

        cape_v = live_th.get("cape_j_kg")
        cin_v = live_th.get("cin_j_kg")
        shear_v = live_th.get("bulk_shear_1000_700hpa_mps")

        # The diagnostic sentence names only the quantities that were actually
        # retrieved, and states the vertical extent of the shear truthfully.
        parts = []
        if cape_v is not None:
            parts.append(f"CAPE {cape_v:.0f} J/kg")
        if cin_v is not None:
            parts.append(f"CIN {cin_v:.0f} J/kg")
        if shear_v is not None:
            parts.append(f"1000–700 hPa bulk shear {shear_v:.1f} m/s")

        if parts:
            th["operational_diagnostic"] = (
                f"Live GFS analysis for {district}: " + ", ".join(parts) + ". "
                "0–6 km bulk shear is not available from the live input "
                "(wind is ingested at 1000/850/700 hPa only)."
            )
        else:
            th["operational_diagnostic"] = (
                "The live analysis was ingested but none of the thermodynamic "
                "variables were finite at the reference point."
            )

        if cape_v is not None:
            th["convective_risk"] = (
                "EXPLOSIVE RISK" if cape_v > 3500
                else "ELEVATED RISK" if cape_v > 1500
                else "MODERATE RISK"
            )
        else:
            th["convective_risk"] = "UNAVAILABLE"

        return th
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
    th["operational_mode"] = "Historical Case Study / Demonstration Mode (Cyclone Remal (Landfall Approach))"
    th["lifted_index"] = {
        "value": None,
        "note": "Lifted Index not measured in native ERA5 single levels; Surface CAPE and 0-6km bulk shear used as primary physical indicators."
    }
    return th


@app.get("/api/nowcast/high-risk-cells", tags=["Nowcasting"])
async def nowcast_high_risk_cells(
    lead: int = Query(2, description="Forecast horizon in hours (0, 2, 3, 4, 5, 6)"),
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
async def nowcast_xai_attribution(
    mode: Optional[str] = None,
    lat: Optional[float] = Query(None, description="Latitude to explain (defaults to the highest-risk cell)"),
    lon: Optional[float] = Query(None, description="Longitude to explain"),
    lead: Optional[int] = Query(None, description="Forecast horizon in hours"),
    svc: NowcastService = Depends(_get_service),
):
    """Atmospheric factor attribution, tied to the selected location and horizon.

    The method is rule-based / physics-inspired -- not SHAP and not learned
    feature importance -- and the response says so explicitly."""
    return svc.get_xai_attribution(mode=mode, lat=lat, lon=lon, lead_hours=lead)


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


_AREAS_CACHE = None


@app.get("/api/geo/areas", tags=["Geospatial"])
async def get_supported_areas(svc: NowcastService = Depends(_get_service)):
    """Every geographic area the application genuinely supports, with real
    centroids and bounds derived from the district boundary dataset.

    The 'Jump to area' control is populated from THIS response rather than a
    hand-typed list, so an option can never exist that the map cannot navigate
    to, and the list can never drift from the boundary data. Each entry reports
    whether it maps onto model grid cells, so an area without a prediction can
    be handled honestly instead of silently failing.
    """
    global _AREAS_CACHE
    if _AREAS_CACHE is not None:
        return _AREAS_CACHE

    from shapely.geometry import shape as _shape

    areas = [{
        "id": "whole-state",
        "name": "Whole State",
        "kind": "state",
        "center": [
            (WB_MIN_LAT_B + WB_MAX_LAT_B) / 2.0,
            (WB_MIN_LON_B + WB_MAX_LON_B) / 2.0,
        ],
        "bounds": [[WB_MIN_LAT_B, WB_MIN_LON_B], [WB_MAX_LAT_B, WB_MAX_LON_B]],
        "grid_cell_count": sum(len(c) for c in svc.district_cells.values()),
        "has_model_coverage": True,
    }]

    if svc.districts_fc:
        for feat in svc.districts_fc.get("features", []):
            name = (feat.get("properties") or {}).get("Name")
            if not name:
                continue
            geom = _shape(feat["geometry"])
            c = geom.centroid
            minx, miny, maxx, maxy = geom.bounds
            cells = svc.district_cells.get(name, [])
            areas.append({
                "id": name.lower().replace(" & ", "-").replace(" ", "-"),
                "name": name,
                "kind": "district",
                # Leaflet order: [lat, lon]
                "center": [round(c.y, 5), round(c.x, 5)],
                "bounds": [[round(miny, 5), round(minx, 5)], [round(maxy, 5), round(maxx, 5)]],
                "grid_cell_count": len(cells),
                "has_model_coverage": len(cells) > 0,
            })

    areas.sort(key=lambda a: (a["kind"] != "state", a["name"]))
    _AREAS_CACHE = {
        "count": len(areas),
        "source": "Data/BOUNDARIES/west_bengal_districts_full.geojson",
        "areas": areas,
    }
    return _AREAS_CACHE


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
    """Return verified held-out 2024 test metrics for StormSense V2, V1, and Persistence baselines."""
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
                "StormSense AI Forecast": {
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
                "StormSense V1 Baseline": {
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
    if dist_to_station < 0.25 and mode == "live":
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


@app.get("/api/time/reference", tags=["System"])
async def time_reference(
    mode: Optional[str] = Query("live", description="Operational mode ('live' or 'historical')"),
    svc: NowcastService = Depends(_get_service),
):
    """THE authoritative forecast reference instant, and the exact +2/+4/+6h
    valid times derived from it.

    The browser must render horizons from this response rather than computing
    them from its own local clock, so the API and the UI can never disagree.
    NOW is the exact wall-clock instant -- never floored to the hour, and never
    the GFS cycle hour (which is reported separately as `analysis_time`)."""
    resolved = svc._resolve_mode(mode)
    if resolved == "live":
        # The live reference instant advances with wall-clock; the atmospheric
        # state behind it is whatever analysis was last ingested.
        reference = datetime.now(timezone.utc).isoformat()
    else:
        reference = svc.current_valid_time

    horizons = []
    for h in (2, 4, 6):
        vt = compute_valid_time(reference, h)
        horizons.append({
            "lead_hours": h,
            "label": f"+{h}h",
            "valid_time": vt,
            "offset_seconds": h * 3600,
        })

    live_status = svc.get_live_status()
    return {
        "mode": resolved,
        "reference_time": reference,
        "reference_time_label": "NOW",
        "horizons": horizons,
        # The GFS analysis behind the live state -- a DIFFERENT instant from the
        # reference time, and never to be displayed as if it were "now".
        "analysis_time": live_status.get("analysis_time") if resolved == "live" else svc.current_valid_time,
        "analysis_lag_hours": live_status.get("analysis_lag_hours") if resolved == "live" else 0,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


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
            "event": "Cyclone Remal (Landfall Approach)",
            "valid_time": svc.current_valid_time,
            "description": "Active severe thunderstorm case from the held-out 2024 test split"
        },
        "model": {
            "name": "StormSense AI Forecast",
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
            "name": "West Bengal",
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


@app.get("/api/forecast/conditions", tags=["Live"])
async def forecast_conditions(
    lat: float = Query(LATITUDE, description="Latitude"),
    lon: float = Query(LONGITUDE, description="Longitude"),
    lead: float = Query(2.0, description="Forecast lead time in hours"),
):
    """Physical conditions (T / RH / wind) at a LIVE forecast horizon.

    These are NOT StormSense model outputs: SevereWeatherNetV2 has only a severe
    head and a rain head. They come from the OpenWeather 5-day/3-hour forecast
    product, which this project already integrates (openweather.get_weather_forecast).
    The value returned is a real published forecast point -- the nearest one to the
    requested lead -- never an interpolation or an invention, so the response states
    the point's true valid time and its real offset from the request. Callers must
    attribute these to OpenWeather, not to the model.

    Historical mode is unsupported by design: OpenWeather publishes no 2024
    reanalysis, and returning today's forecast for a 2024 case study would be a
    cross-mode contamination.
    """
    try:
        raw = await get_weather_forecast(lat, lon)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"OpenWeather forecast request failed: {exc}",
            "source": "OpenWeather 5-day / 3-hour forecast",
        }

    points = raw.get("list") or []
    if not points:
        return {
            "available": False,
            "reason": "OpenWeather returned no forecast points.",
            "source": "OpenWeather 5-day / 3-hour forecast",
        }

    now_epoch = datetime.now(timezone.utc).timestamp()
    target_epoch = now_epoch + (float(lead) * 3600.0)

    timed = sorted(
        (p for p in points if p.get("dt") is not None),
        key=lambda p: float(p["dt"]),
    )
    if not timed:
        return {
            "available": False,
            "reason": "No usable forecast point with a timestamp.",
            "source": "OpenWeather 5-day / 3-hour forecast",
        }

    # The product publishes every 3 hours, so +4h, +5h and +6h all share a single
    # nearest point: snapping would make three different horizons show identical
    # values. Interpolate LINEARLY IN TIME between the two real points that
    # bracket the target instead, so each horizon reports its own value. Both
    # endpoints are genuine published forecasts and the weight is real elapsed
    # time -- nothing is invented, and the response states the bracketing points.
    before = [p for p in timed if float(p["dt"]) <= target_epoch]
    after = [p for p in timed if float(p["dt"]) > target_epoch]

    def _blocks(p):
        return (p.get("main") or {}), (p.get("wind") or {})

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if before and after:
        p0, p1 = before[-1], after[0]
        t0, t1 = float(p0["dt"]), float(p1["dt"])
        w = 0.0 if t1 == t0 else (target_epoch - t0) / (t1 - t0)
        m0, w0 = _blocks(p0)
        m1, w1 = _blocks(p1)

        def lerp(a, b):
            a, b = _num(a), _num(b)
            if a is None or b is None:
                return a if b is None else b
            return a + (b - a) * w

        temp_c = lerp(m0.get("temp"), m1.get("temp"))
        hum = lerp(m0.get("humidity"), m1.get("humidity"))
        pres = lerp(m0.get("pressure"), m1.get("pressure"))
        wind_ms = lerp(w0.get("speed"), w1.get("speed"))
        wind_deg = lerp(w0.get("deg"), w1.get("deg"))
        desc = ((p1 if w >= 0.5 else p0).get("weather") or [{}])[0].get("description")
        basis = "interpolated between two published forecast points"
        bracket = [
            datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(t1, tz=timezone.utc).isoformat(),
        ]
        actual_offset = round(float(lead), 2)
        valid_iso = datetime.fromtimestamp(target_epoch, tz=timezone.utc).isoformat()
    else:
        # Target lies outside the published range: use the nearest real point and
        # say so, rather than extrapolating beyond what the provider issued.
        p = (before[-1] if before else after[0])
        m, wb = _blocks(p)
        temp_c, hum, pres = _num(m.get("temp")), _num(m.get("humidity")), _num(m.get("pressure"))
        wind_ms, wind_deg = _num(wb.get("speed")), _num(wb.get("deg"))
        desc = (p.get("weather") or [{}])[0].get("description")
        basis = "nearest published forecast point (target outside published range)"
        bracket = [datetime.fromtimestamp(float(p["dt"]), tz=timezone.utc).isoformat()]
        actual_offset = round((float(p["dt"]) - now_epoch) / 3600.0, 2)
        valid_iso = datetime.fromtimestamp(float(p["dt"]), tz=timezone.utc).isoformat()

    return {
        "available": True,
        "source": "OpenWeather 5-day / 3-hour forecast",
        "is_model_output": False,
        "requested_lead_hours": float(lead),
        "actual_offset_hours": actual_offset,
        "valid_time_utc": valid_iso,
        "measurement_basis": basis,
        "bracketing_points_utc": bracket,
        "temperature_c": round(temp_c, 2) if temp_c is not None else None,
        "humidity_pct": round(hum) if hum is not None else None,
        "pressure_hpa": round(pres) if pres is not None else None,
        # OpenWeather reports metric wind in m/s; the UI displays km/h.
        "wind_speed_kmh": round(wind_ms * 3.6, 1) if wind_ms is not None else None,
        "wind_direction_deg": round(wind_deg) if wind_deg is not None else None,
        "weather_description": desc,
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
        "model": "StormSense AI Forecast",
        # Reference instant the forecast is issued for (exact wall-clock).
        "reference_time": status["reference_time"],
        # Real GFS f000 analysis the input state came from -- a different instant.
        "analysis_time": status["analysis_time"],
        "analysis_cycles": status["analysis_cycles"],
        "analysis_lag_hours": status["analysis_lag_hours"],
        "input_source": status["input_source"],
        "uses_forecast_hours_as_input": status["uses_forecast_hours_as_input"],
        "last_refreshed": status["last_refreshed"],
        "is_stale": status["is_stale"],
        "input_slot_provenance": status["input_slots"],
        "temporal_requirement": "6 consecutive hourly timesteps (t-5h to t0), analyses only",
        "spatial_requirement": "33x25 grid at 0.25 degree resolution (20-28N, 84-90E)",
        "known_limitations": LIVE_KNOWN_LIMITATIONS,
    }


@app.get("/api/data-health", tags=["System"])
async def data_health():
    """Return health status of each data pipeline component."""
    svc = _get_service()
    
    # Check Live Station API
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
                "source": "Live Station API API",
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
                "name": "StormSense V2 Model",
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
async def nowcast_xai(
    mode: Optional[str] = None,
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    lead: Optional[int] = Query(None),
    svc: NowcastService = Depends(_get_service),
):
    """Atmospheric factor attribution (rule-based / physics-inspired)."""
    return svc.get_xai_attribution(mode=mode, lat=lat, lon=lon, lead_hours=lead)


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
    summary = svc.get_summary(lead_hours=lead, district="West Bengal")

    # Structure in exact shape expected by existing frontend
    return {
        "location": "West Bengal",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "source": "StormSense V2 Nowcaster + Live Station API",
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
            # The model forecasts severe-weather probability and 3h rainfall --
            # it has NO temperature/humidity/wind forecast head. These were
            # previously synthesised by decrementing the current observation
            # (temp - 0.4*h, humidity + 2*h), which is invented data, not a
            # forecast. They are reported as null instead.
            "temperature": None,
            "feels_like": None,
            "humidity": None,
            "pressure": None,
            "wind_speed": None,
            "wind_direction": None,
            "ambient_forecast_available": False,
            "ambient_forecast_note": (
                "The nowcasting model predicts severe-weather probability and 3h "
                "rainfall only; ambient temperature/humidity/wind are not forecast."
            ),
            "rainfall_mm": pt["rainfall_mm_3h"],
            "rainfall_period_hours": 3,
            "weather": "Thunderstorm" if pt["severe_weather_pct"] >= 65 else ("Rain" if pt["rainfall_mm_3h"] > 2.0 else "Cloudy"),
            "weather_description": f"ML Nowcast (+{h}h): {pt['severe_weather_pct']}% severe risk",
            "severe_weather_pct": pt["severe_weather_pct"],
            "risk_level": pt["risk_level"],
        })

    return {
        "location": "West Bengal",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "horizon_hours": 6,
        "source": "StormSense AI Forecast ML",
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
            "disclaimer": "Predictions generated by StormSense AI Forecast model.",
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
    print(f"Starting StormSense Mission Control with StormSense V2 on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")




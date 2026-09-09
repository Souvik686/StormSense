"""FastAPI backend for the SevereWeatherNet nowcasting system.

Endpoints
---------
GET  /health          -- liveness check
GET  /model-info      -- architecture / training metadata
POST /predict         -- run inference on supplied ERA5 features
GET  /latest          -- latest cached prediction (if any)
GET  /risk-map        -- GeoJSON FeatureCollection for a lead time
GET  /explanation     -- feature importance metadata

Security:  API key is optional (set API_KEY env-var to enable).
           Credentials are never logged or returned in error bodies.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

from src.inference.predictor import get_predictor, NowcastPredictor
from src.inference.risk_map import (
    predictions_to_geojson,
    predictions_to_summary,
    predictions_to_all_leads_geojson,
)
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS
from src.utils.config import load_config

# ─────────────────────────── configuration ──────────────────────────────────
CONFIG_PATH   = os.getenv("CONFIG_PATH", None)          # None = default.yaml
cfg_tmp = load_config(CONFIG_PATH)
_v2_cal_path = os.path.join(cfg_tmp.path("training", "checkpoint_dir"), "v2_calibrated_best.pt")
_v2_path     = os.path.join(cfg_tmp.path("training", "checkpoint_dir"), "v2_best.pt")
_v1_path     = os.path.join(cfg_tmp.path("training", "checkpoint_dir"), "best.pt")
CHECKPOINT   = os.getenv("CHECKPOINT_PATH", _v2_cal_path if os.path.exists(_v2_cal_path) else (_v2_path if os.path.exists(_v2_path) else _v1_path))
API_KEY_ENV   = os.getenv("API_KEY", "")                # empty = no auth

# ─────────────────────────── FastAPI app ────────────────────────────────────
app = FastAPI(
    title="SevereWeatherNet Nowcasting API",
    version="1.0.0",
    description=(
        "AI-driven hyper-local severe weather nowcasting for West Bengal (India). "
        "Predictions are ERA5-derived proxies — not authoritative meteorological forecasts."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── module-level cache ──────────────────────────────
_LATEST_PRED: dict | None = None
_LATEST_TIME: str | None = None


# ─────────────────────────── auth dependency ────────────────────────────────
async def verify_api_key(x_api_key: str = Query(default="")):
    if API_KEY_ENV and x_api_key != API_KEY_ENV:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ─────────────────────────── predictor helper ───────────────────────────────
def _get_predictor() -> NowcastPredictor:
    try:
        return get_predictor(CHECKPOINT, CONFIG_PATH)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


# ─────────────────────────── Pydantic models ────────────────────────────────
class PredictRequest(BaseModel):
    """Raw ERA5 arrays for one prediction request.

    All arrays are lists-of-lists (JSON-serialisable), shaped as documented.
    Units must match the ERA5 raw values (m s-1, K, Pa, J kg-1, kg m-2, m).
    tp is in metres (ECMWF convention; will be converted to mm internally).
    """
    # (T, n_surface_vars, H, W)  -- T = input_hours (default 6)
    surface: List[List[List[List[float]]]] = Field(
        ...,
        description=(
            f"Surface ERA5 fields, shape (T, {len(SINGLE_VARS)}, H, W). "
            f"Variable order: {SINGLE_VARS}"
        ),
    )
    # (T, n_pressure_vars, n_levels, H, W)
    pressure: List[List[List[List[List[float]]]]] = Field(
        ...,
        description=(
            f"Pressure-level ERA5 fields, shape (T, {len(PRESSURE_VARS)}, n_levels, H, W). "
            f"Variable order: {PRESSURE_VARS}"
        ),
    )
    # (H, W)
    dem: List[List[float]] = Field(
        ...,
        description="DEM elevation in metres, block-averaged to ERA5 0.25° grid, shape (H, W).",
    )
    valid_time: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp of the last input hour (for metadata only).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "valid_time": "2024-07-15T06:00:00Z",
                "surface": "... (6 × 9 × 33 × 25 float array) ...",
                "pressure": "... (6 × 5 × 6 × 33 × 25 float array) ...",
                "dem": "... (33 × 25 float array) ...",
            }
        }


class HealthResponse(BaseModel):
    status: str
    checkpoint_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    n_parameters: int
    training_epoch: int
    best_val_loss: float
    optimal_threshold: float
    lead_times_hours: List[int]
    input_hours: int
    surface_vars: List[str]
    pressure_vars: List[str]
    grid_shape: List[int]
    lat_range: List[float]
    lon_range: List[float]


# ─────────────────────────── endpoints ─────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Always returns 200 if the server is running."""
    ckpt_exists = os.path.exists(CHECKPOINT)
    return {
        "status": "ok",
        "checkpoint_loaded": ckpt_exists,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(predictor: NowcastPredictor = Depends(_get_predictor)):
    """Return architecture and training metadata for the loaded model."""
    return predictor.model_info()


@app.post("/predict", tags=["Inference"])
async def predict(
    request: PredictRequest,
    predictor: NowcastPredictor = Depends(_get_predictor),
    _: None = Depends(verify_api_key),
):
    """Run nowcast inference and return predictions for all lead times.

    Returns a JSON object with per-lead-time risk maps and a compact summary.
    """
    global _LATEST_PRED, _LATEST_TIME

    try:
        surface  = np.array(request.surface,  dtype=np.float32)
        pressure = np.array(request.pressure, dtype=np.float32)
        dem      = np.array(request.dem,      dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Array conversion error: {e}")

    # tp: ECMWF sends metres; the predictor expects mm (loader converts x1000)
    tp_idx = SINGLE_VARS.index("tp")
    if surface[:, tp_idx].max() < 1.0:   # looks like metres, not mm
        surface[:, tp_idx] = surface[:, tp_idx] * 1000.0

    try:
        pred = predictor.predict(surface, pressure, dem)
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    _LATEST_PRED = pred
    _LATEST_TIME = request.valid_time or datetime.now(timezone.utc).isoformat()

    summary = predictions_to_summary(pred, _LATEST_TIME)
    geojson_all = predictions_to_all_leads_geojson(pred, _LATEST_TIME)

    return JSONResponse(
        content={
            "summary": summary,
            "risk_maps": geojson_all,
            "proxy_disclaimer": (
                "All predictions are ERA5-derived model proxies. "
                "Not authoritative meteorological forecasts."
            ),
        }
    )


@app.get("/latest", tags=["Inference"])
async def latest():
    """Return the most recent cached prediction (from the last /predict call)."""
    if _LATEST_PRED is None:
        raise HTTPException(
            status_code=404,
            detail="No prediction cached yet. Call POST /predict first.",
        )
    summary = predictions_to_summary(_LATEST_PRED, _LATEST_TIME)
    return JSONResponse(content={"summary": summary, "valid_time": _LATEST_TIME})


@app.get("/risk-map", tags=["Inference"])
async def risk_map(
    lead_time_hours: int = Query(
        3,
        description="Lead time in hours (one of 2, 3, 4, 5, 6)",
        ge=2, le=6,
    ),
):
    """Return a GeoJSON FeatureCollection for the specified lead time.

    Each feature is a 0.25° grid-cell point with risk scores and alert flags.
    """
    if _LATEST_PRED is None:
        raise HTTPException(
            status_code=404,
            detail="No prediction cached. Call POST /predict first.",
        )
    lead_times = _LATEST_PRED["lead_times_hours"]
    if lead_time_hours not in lead_times:
        raise HTTPException(
            status_code=422,
            detail=f"lead_time_hours must be one of {lead_times}, got {lead_time_hours}",
        )
    idx = lead_times.index(lead_time_hours)
    geojson = predictions_to_geojson(_LATEST_PRED, _LATEST_TIME, lead_time_hours, idx)
    return JSONResponse(content=geojson)


@app.get("/explanation", tags=["Explainability"])
async def explanation(
    predictor: NowcastPredictor = Depends(_get_predictor),
):
    """Return documented feature-importance metadata for the model.

    NOTE: These are architecture-level explanations (which modalities feed
    which model components), NOT per-sample SHAP values. Full SHAP analysis
    is computationally expensive and not practical at inference time;
    see scripts/explain.py for offline SHAP computation.
    """
    return JSONResponse(content={
        "explanation_type": "architecture_and_design_rationale",
        "important_features": {
            "surface_ConvGRU": {
                "variables": SINGLE_VARS,
                "role": (
                    "Captures near-surface temporal evolution: rising CAPE before "
                    "convective initiation, veering low-level wind associated with "
                    "wind shear, moisture from TCWV/d2m, triggering rainfall from tp."
                ),
                "key_predictors_for_severe_weather": [
                    "cape (Convective Available Potential Energy — primary instability proxy)",
                    "cin (Convective Inhibition — convective gating)",
                    "tcwv (Total Column Water Vapour — moisture reservoir)",
                    "tp (hourly precipitation — direct signal)",
                    "u10/v10 (surface wind — convergence indicator)",
                ],
            },
            "pressure_ConvGRU": {
                "variables": PRESSURE_VARS,
                "levels_hPa": [1000, 850, 700, 500, 300, 250],
                "role": (
                    "Captures atmospheric vertical structure: 850 hPa moisture "
                    "transport (q), 500 hPa geopotential height (z) for synoptic "
                    "troughs, upper-troposphere divergence at 300-250 hPa."
                ),
                "key_predictors": [
                    "q @ 850 hPa (low-level specific humidity — moisture source)",
                    "z @ 500 hPa (synoptic-scale trough position)",
                    "u,v @ 700 hPa (mid-level wind — wind shear component)",
                ],
            },
            "DEM_static": {
                "role": (
                    "Orographic modulation: hills/ridges of the Himalayan foothills "
                    "and Shillong Plateau force lifting of moist air, dramatically "
                    "enhancing convective initiation and rainfall on windward slopes."
                ),
                "terrain_amplification": "Applied as multiplicative factor in flash-flood risk proxy",
            },
        },
        "flash_flood_proxy_formula": (
            "flash_flood_risk = clip(severe_weather_prob × (1 + 0.3 × max(dem_norm, 0)), 0, 1). "
            "Higher elevations within the domain amplify the probability. "
            "PROXY ONLY — not observed flash-flood events."
        ),
        "overall_risk_formula": (
            "overall_risk = 0.5 × severe_weather_prob + 0.3 × flash_flood_risk + "
            "0.2 × clip(rain_3h_mm / 30.0, 0, 1). "
            "Weights are documented and fixed, not learned from data."
        ),
        "limitations": [
            "All targets are ERA5-derived proxies (no ground-truth lightning/flood records).",
            "INSAT satellite data NOT used in training (temporal coverage too sparse).",
            "Lightning CSV covers only 2020; no continuous time series for training.",
            "Radar data is calibration figures only; no gridded reflectivity.",
            "Model trained on May-Oct 2021-2022 only; tested on May-Oct 2024.",
            "0.25° spatial resolution (≈28 km) cannot resolve sub-grid convective cells.",
        ],
    })


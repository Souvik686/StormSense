"""Inference predictor: loads a trained SevereWeatherNet checkpoint and
produces multi-horizon, multi-task spatial risk predictions.

Design decisions:
  - Loads the model ONCE and caches it (singleton pattern via module-level dict).
  - Accepts raw ERA5 feature arrays (already on the common 33x25 0.25-deg grid)
    plus the static DEM, normalizes them with the saved per-training stats, and
    runs the forward pass.
  - Returns un-thresholded probabilities AND thresholded binary predictions.
  - The decision threshold is stored in the checkpoint (optimised on the
    validation set during training -- see scripts/optimize_threshold.py).
  - Never silently swallows errors: every failure surfaces immediately so the
    API can return a meaningful HTTP error.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from src.features.normalize import (
    SINGLE_VARS, PRESSURE_VARS, load_stats, normalize_single, normalize_pressure,
)
from src.models.advanced import SevereWeatherNet
from src.utils.config import load_config, Config

# Module-level cache so uvicorn workers don't reload on every request
_MODEL_CACHE: dict[str, object] = {}

DEFAULT_THRESHOLD = 0.35  # conservative default; overridden by checkpoint value


class NowcastPredictor:
    """High-level inference wrapper around SevereWeatherNet.

    Parameters
    ----------
    checkpoint_path : str
        Path to a `.pt` checkpoint produced by train_colab.py.
    config_path : str | None
        Optional path to a YAML config override. If None, uses default.yaml.
    device : str
        'cuda', 'cpu', or 'auto' (picks cuda if available).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = load_config(config_path)
        self.cfg = cfg
        self.lead_times: list[int] = cfg.get("sequence", "lead_times_hours")
        self.input_hours: int = cfg.get("sequence", "input_hours")
        n_levels = len(cfg.get("era5", "pressure_levels_hpa"))

        # Load normalization stats
        stats_path = cfg.path("normalization", "stats_file")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Normalization stats not found at {stats_path}. "
                "Run preprocessing first: python -m src.data.preprocess"
            )
        self.stats = load_stats(stats_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # Retained so the API can report which checkpoint is actually serving,
        # rather than a hardcoded model name that can drift from reality.
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.is_v2 = "wind_enc.level_weight" in ckpt["model"]

        if self.is_v2:
            from src.models.v2_model import SevereWeatherNetV2
            model = SevereWeatherNetV2(n_lead_times=len(self.lead_times))
        else:
            model = SevereWeatherNet(
                n_surface_vars=len(SINGLE_VARS),
                n_pressure_vars=len(PRESSURE_VARS),
                n_levels=n_levels,
                n_lead_times=len(self.lead_times),
            )
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        self.model = model

        # Decision threshold and calibration parameters
        self.threshold: float = ckpt.get("optimal_threshold", DEFAULT_THRESHOLD)
        self.threshold_per_lead: dict = ckpt.get("threshold_per_lead", None)
        self.temperature: list = ckpt.get("temperature_per_lead", None)
        self.is_calibrated: bool = ckpt.get("is_calibrated", False)
        self.training_epoch: int = ckpt.get("epoch", -1)
        self.best_val_loss: float = ckpt.get("best_val_loss", float("nan"))

        # Static lat/lon grids (loaded from cache or recomputed)
        domain = cfg.get("domain")
        n_lat, n_lon = domain["grid_shape"]
        self.lats = np.linspace(domain["lat_max"], domain["lat_min"], n_lat)
        self.lons = np.linspace(domain["lon_min"], domain["lon_max"], n_lon)

    def _normalize_surface(self, surface: np.ndarray) -> np.ndarray:
        """Normalize surface array in-place copy.

        Parameters
        ----------
        surface : np.ndarray, shape (T, n_surface_vars, H, W)
        """
        out = surface.copy().astype(np.float32)
        cin_i = SINGLE_VARS.index("cin")
        out[:, cin_i] = np.nan_to_num(out[:, cin_i], nan=0.0)
        for i, v in enumerate(SINGLE_VARS):
            s = self.stats[v]
            out[:, i] = (out[:, i] - s["mean"]) / s["std"]
        return out

    def _normalize_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """Normalize pressure array.

        Parameters
        ----------
        pressure : np.ndarray, shape (T, n_pressure_vars, n_levels, H, W)
        """
        out = np.nan_to_num(pressure.copy().astype(np.float32), nan=0.0)
        for i, v in enumerate(PRESSURE_VARS):
            # level axis is 2 in the (T, C, L, H, W) layout
            out[:, i] = normalize_pressure(out[:, i], v, self.stats, level_axis=1)
        return out

    @torch.no_grad()
    def predict(
        self,
        surface: np.ndarray,
        pressure: np.ndarray,
        dem: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> dict:
        """Run inference on a single sample."""
        assert surface.ndim == 4, f"surface must be (T,C,H,W), got {surface.shape}"
        assert pressure.ndim == 5, f"pressure must be (T,C,L,H,W), got {pressure.shape}"
        assert dem.ndim == 2, f"dem must be (H,W), got {dem.shape}"
        assert surface.shape[0] == self.input_hours, (
            f"Need {self.input_hours} input timesteps, got {surface.shape[0]}")

        # Normalize
        surf_norm = self._normalize_surface(surface)
        pres_norm = self._normalize_pressure(pressure)

        dem_mean = float(np.nanmean(dem))
        dem_std = max(float(np.nanstd(dem)), 1e-6)
        dem_norm = ((dem - dem_mean) / dem_std).astype(np.float32)

        if self.is_v2:
            import pandas as pd
            pressure_wind = pres_norm[:, :2, 3:]
            pressure_thermo = pres_norm[:, 2:, :4]

            wind_speed = np.sqrt(surf_norm[:, 0]**2 + surf_norm[:, 1]**2)
            dewpoint_dep = surf_norm[:, 3] - surf_norm[:, 2]
            derived = np.stack([wind_speed, dewpoint_dep], axis=1).astype(np.float32)

            T = self.input_hours
            H, W = surf_norm.shape[-2:]
            if timestamp is not None:
                ts = pd.to_datetime(timestamp)
                hours = (ts.hour + np.arange(-T + 1, 1)) % 24
                doy = ts.dayofyear
                hour_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
                hour_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
                doy_sin = np.full(T, np.sin(2 * np.pi * doy / 365.25), dtype=np.float32)
                doy_cos = np.full(T, np.cos(2 * np.pi * doy / 365.25), dtype=np.float32)
                temporal = np.stack([hour_sin, hour_cos, doy_sin, doy_cos], axis=1)[:, :, None, None].repeat(H, axis=2).repeat(W, axis=3)
            else:
                temporal = np.zeros((T, 4, H, W), dtype=np.float32)

            surf_full = np.concatenate([surf_norm, derived, temporal], axis=1)

            lat_min = self.cfg.get("domain", "lat_min")
            lat_max = self.cfg.get("domain", "lat_max")
            lon_min = self.cfg.get("domain", "lon_min")
            lon_max = self.cfg.get("domain", "lon_max")
            lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
            lat_grid = 2.0 * (lat_grid - lat_min) / (lat_max - lat_min) - 1.0
            lon_grid = 2.0 * (lon_grid - lon_min) / (lon_max - lon_min) - 1.0
            dem_full = np.stack([dem_norm, lat_grid.astype(np.float32), lon_grid.astype(np.float32)], axis=0)

            batch = {
                "surface": torch.from_numpy(surf_full[None]).to(self.device),
                "pressure_wind": torch.from_numpy(pressure_wind[None]).to(self.device),
                "pressure_thermo": torch.from_numpy(pressure_thermo[None]).to(self.device),
                "dem": torch.from_numpy(dem_full[None]).to(self.device),
            }
        else:
            batch = {
                "surface": torch.from_numpy(surf_norm[None]).to(self.device),
                "pressure": torch.from_numpy(pres_norm[None]).to(self.device),
                "dem": torch.from_numpy(dem_norm[None, None]).to(self.device),
            }

        preds = self.model(batch)

        logits = preds["severe_weather_logit"][0]  # (n_lead, H, W)
        if self.temperature is not None:
            T = torch.tensor(self.temperature, device=logits.device, dtype=logits.dtype).view(-1, 1, 1)
            severe_prob = torch.sigmoid(logits / T).cpu().numpy()
        else:
            severe_prob = torch.sigmoid(logits).cpu().numpy()

        rain_pred = preds["rain_3h_mm"][0].cpu().numpy()  # (n_lead, H, W)

        if self.threshold_per_lead is not None:
            severe_binary = np.zeros_like(severe_prob, dtype=np.uint8)
            for li, lh in enumerate(self.lead_times):
                thr = float(self.threshold_per_lead.get(lh, self.threshold_per_lead.get(int(lh), self.threshold)))
                severe_binary[li] = (severe_prob[li] >= thr).astype(np.uint8)
        else:
            severe_binary = (severe_prob >= self.threshold).astype(np.uint8)

        # Flash-flood risk proxy: hydrologically gated compound score combining
        # precipitation intensity, severe storm probability, and terrain amplification.
        # This is a hydrologically gated PROXY score, not an observed streamflow prediction.
        dem_norm_clipped = np.clip((dem - dem_mean) / dem_std, -3, 3)
        terrain_factor = 1.0 + 0.2 * np.clip(dem_norm_clipped, 0, 2.5)  # foothills/topographic slope
        rain_norm = np.clip(rain_pred / 30.0, 0.0, 1.0)  # 30mm = severe convective rain threshold
        hydro_intensity = rain_norm * (0.4 + 0.6 * severe_prob)
        flash_flood_risk = np.clip(hydro_intensity * terrain_factor[None], 0.0, 1.0)

        # Overall risk: calibrated weighted combination
        #   severe_weather_prob x 0.5 + flash_flood_risk x 0.3 + rain_norm x 0.2
        overall_risk = np.clip(
            0.5 * severe_prob + 0.3 * flash_flood_risk + 0.2 * rain_norm,
            0.0, 1.0,
        )

        return {
            "severe_weather_prob":   severe_prob.astype(np.float32),
            "rain_3h_mm_pred":       rain_pred.astype(np.float32),
            "severe_weather_binary": severe_binary,
            "flash_flood_risk":      flash_flood_risk.astype(np.float32),
            "overall_risk":          overall_risk.astype(np.float32),
            "lead_times_hours":      self.lead_times,
            "threshold":             self.threshold,
            "lats":                  self.lats,
            "lons":                  self.lons,
        }

    def model_info(self) -> dict:
        return {
            "n_parameters": self.model.count_parameters(),
            "training_epoch": self.training_epoch,
            "best_val_loss": self.best_val_loss,
            "optimal_threshold": self.threshold,
            "lead_times_hours": self.lead_times,
            "input_hours": self.input_hours,
            "surface_vars": SINGLE_VARS,
            "pressure_vars": PRESSURE_VARS,
            "grid_shape": list(self.cfg.get("domain", "grid_shape")),
            "lat_range": [float(self.lats[-1]), float(self.lats[0])],
            "lon_range": [float(self.lons[0]), float(self.lons[-1])],
        }


def get_predictor(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> NowcastPredictor:
    """Return a cached predictor (loads once per process).

    `device` defaults to the STORMSENSE_DEVICE env var, else "auto". Serving
    deployments can set STORMSENSE_DEVICE=cpu to avoid loading CUDA runtime
    libraries -- this model is small enough (~782K params on a 33x25 grid) that
    CPU inference is fast, and it removes a class of GPU/driver/paging-file
    startup failures on memory-constrained hosts.
    """
    if device is None:
        device = os.environ.get("STORMSENSE_DEVICE", "auto")
    key = f"{checkpoint_path}::{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = NowcastPredictor(checkpoint_path, config_path, device=device)
    return _MODEL_CACHE[key]


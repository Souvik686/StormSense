"""Feature normalization, fit strictly on the training split to avoid
normalization leakage (val/test statistics never touch the fitted mean/std).
"""
from __future__ import annotations

import json
import os

import numpy as np
import xarray as xr

from src.utils.config import Config

SINGLE_VARS = ["u10", "v10", "d2m", "t2m", "sp", "cape", "cin", "tcwv", "tp"]
PRESSURE_VARS = ["u", "v", "z", "q", "t"]


def fit_normalization_stats(ds: xr.Dataset, train_time_indices: np.ndarray) -> dict:
    """Compute per-variable (and per-pressure-level, for 3D vars) mean/std
    using ONLY the timesteps in `train_time_indices`."""
    stats = {}
    train_ds = ds.isel(valid_time=train_time_indices)

    for v in SINGLE_VARS:
        arr = train_ds[v].values
        finite = arr[np.isfinite(arr)]
        stats[v] = {"mean": float(finite.mean()), "std": float(max(finite.std(), 1e-6))}

    for v in PRESSURE_VARS:
        arr = train_ds[v].values  # (time, level, lat, lon)
        n_levels = arr.shape[1]
        per_level = []
        for li in range(n_levels):
            a = arr[:, li]
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                per_level.append({"mean": 0.0, "std": 1.0})
            else:
                per_level.append({"mean": float(finite.mean()), "std": float(max(finite.std(), 1e-6))})
        stats[v] = {"per_level": per_level,
                    "levels_hpa": [float(x) for x in ds.pressure_level.values]}
    return stats


def save_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_single(arr: np.ndarray, var: str, stats: dict) -> np.ndarray:
    s = stats[var]
    return (arr - s["mean"]) / s["std"]


def normalize_pressure(arr: np.ndarray, var: str, stats: dict, level_axis: int = 0) -> np.ndarray:
    """arr shape (..., n_levels, ...) with n_levels on `level_axis`."""
    s = stats[var]["per_level"]
    means = np.array([x["mean"] for x in s], dtype=np.float32)
    stds = np.array([x["std"] for x in s], dtype=np.float32)
    shape = [1] * arr.ndim
    shape[level_axis] = -1
    return (arr - means.reshape(shape)) / stds.reshape(shape)


def validate_stats(stats: dict) -> None:
    for v in SINGLE_VARS:
        assert v in stats, f"missing stats for {v}"
        assert stats[v]["std"] > 0
        assert np.isfinite(stats[v]["mean"])
    for v in PRESSURE_VARS:
        assert v in stats, f"missing stats for {v}"
        for lvl in stats[v]["per_level"]:
            assert lvl["std"] > 0
            assert np.isfinite(lvl["mean"])

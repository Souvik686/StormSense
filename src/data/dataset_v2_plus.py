"""PyTorch Dataset with Enhanced Atmospheric Physics Features (V2+).

Adds:
  1. Low-level bulk wind shear (1000 hPa to 700 hPa)
  2. Mid-tropospheric lapse rate (700 hPa vs 500 hPa temperature difference)
  3. Orographic terrain slope magnitude (2D spatial gradient of DEM)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import Config
from src.data.windowing import build_windows, WindowSample
from src.features.normalize import (
    SINGLE_VARS, PRESSURE_VARS, load_stats, normalize_single, normalize_pressure,
)


class NowcastDatasetV2Plus(Dataset):
    def __init__(self, cache_path: str, cfg: Config, split: str):
        self.cache_path = cache_path
        self.cfg = cfg
        self.split = split
        self.input_hours = cfg.get("sequence", "input_hours")
        self.lead_times = cfg.get("sequence", "lead_times_hours")

        self.surface = np.load(os.path.join(cache_path, "surface.npy"), mmap_mode="r")
        self.pressure = np.load(os.path.join(cache_path, "pressure.npy"), mmap_mode="r")
        self.dem = np.load(os.path.join(cache_path, "dem_elevation_m.npy"), mmap_mode="r")
        self.times = np.load(os.path.join(cache_path, "valid_time.npy"), mmap_mode="r").astype("datetime64[ns]")
        self.severe = np.load(os.path.join(cache_path, "target_severe_weather.npy"), mmap_mode="r")
        self.rain3h = np.load(os.path.join(cache_path, "target_rain_3h_mm.npy"), mmap_mode="r")
        self.label_valid = np.load(os.path.join(cache_path, "target_label_valid.npy"), mmap_mode="r").astype(bool)

        self.stats = load_stats(cfg.path("normalization", "stats_file"))

        dem_mean, dem_std = float(np.mean(self.dem)), float(max(np.std(self.dem), 1e-6))
        self.dem_norm = ((self.dem - dem_mean) / dem_std).astype(np.float32)

        # Compute Terrain Slope via 2D spatial gradients (Sobel filter approximation)
        grad_y, grad_x = np.gradient(self.dem)
        slope = np.sqrt(grad_y ** 2 + grad_x ** 2)
        slope_mean, slope_std = float(np.mean(slope)), float(max(np.std(slope), 1e-6))
        self.slope_norm = ((slope - slope_mean) / slope_std).astype(np.float32)

        # Spatial coordinate grids
        domain = cfg.get("domain")
        lat_min, lat_max = domain["lat_min"], domain["lat_max"]
        lon_min, lon_max = domain["lon_min"], domain["lon_max"]
        n_lat, n_lon = domain["grid_shape"]

        lats = np.linspace(lat_max, lat_min, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        self.lat_grid = (2.0 * (lat_grid - lat_min) / (lat_max - lat_min) - 1.0).astype(np.float32)
        self.lon_grid = (2.0 * (lon_grid - lon_min) / (lon_max - lon_min) - 1.0).astype(np.float32)

        self.windows: list[WindowSample] = build_windows(self.times, cfg, split)
        self.windows = [w for w in self.windows
                        if all(self.label_valid[idx].all() for idx in w.lead_target_idx.values())]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        start = w.input_end_idx - self.input_hours + 1
        end = w.input_end_idx + 1

        surface = self.surface[start:end].copy()      # (input_hours, 9, H, W)
        pressure = self.pressure[start:end].copy()    # (input_hours, 5, 6, H, W)

        # Normalize surface
        for j, v in enumerate(SINGLE_VARS):
            surface[:, j] = normalize_single(surface[:, j], v, self.stats)

        # Normalize pressure
        for j, v in enumerate(PRESSURE_VARS):
            pressure[:, j] = normalize_pressure(pressure[:, j], v, self.stats, level_axis=1)

        # Slices:
        # u,v at 700/850/1000 hPa (indices 3,4,5)
        pressure_wind = pressure[:, :2, 3:]    # (T, 2, 3, H, W)
        # z,q,t at 250/300/500/700 hPa (indices 0,1,2,3)
        pressure_thermo = pressure[:, 2:, :4]  # (T, 3, 4, H, W)

        # 1. Derived surface features
        wind_speed = np.sqrt(surface[:, 0]**2 + surface[:, 1]**2)
        dewpoint_dep = surface[:, 3] - surface[:, 2]

        # 2. Atmospheric Physical Features:
        # Low-level shear: u,v at 700 hPa (level idx 0 in pressure_wind) vs 1000 hPa (level idx 2 in pressure_wind)
        u_shear = pressure_wind[:, 0, 0] - pressure_wind[:, 0, 2]
        v_shear = pressure_wind[:, 1, 0] - pressure_wind[:, 1, 2]
        bulk_shear = np.sqrt(u_shear**2 + v_shear**2)

        # Mid-level lapse rate: t at 700 hPa (var idx 2, level idx 3 in pressure_thermo) vs t at 500 hPa (level idx 2)
        lapse_rate = pressure_thermo[:, 2, 3] - pressure_thermo[:, 2, 2]

        # 3. Temporal features
        times_window = pd.DatetimeIndex(self.times[start:end])
        hours = times_window.hour.values
        doys = times_window.dayofyear.values

        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        doy_sin = np.sin(2 * np.pi * doys / 365.25)
        doy_cos = np.cos(2 * np.pi * doys / 365.25)

        H, W = surface.shape[-2:]
        T = self.input_hours

        temporal = np.stack([hour_sin, hour_cos, doy_sin, doy_cos], axis=1).astype(np.float32)
        temporal = temporal[:, :, None, None].repeat(H, axis=2).repeat(W, axis=3)

        derived_surf = np.stack([wind_speed, dewpoint_dep, bulk_shear, lapse_rate], axis=1).astype(np.float32)

        # Surface full: 9 raw + 4 derived physics + 4 temporal = 17 channels
        surface_full = np.concatenate([surface, derived_surf, temporal], axis=1)

        # Terrain & Spatial: DEM elevation + DEM slope + Lat grid + Lon grid = 4 channels
        dem_full = np.stack([self.dem_norm, self.slope_norm, self.lat_grid, self.lon_grid], axis=0)

        lead_idx = [w.lead_target_idx[lh] for lh in self.lead_times]
        severe = self.severe[lead_idx].copy()
        rain3h = self.rain3h[lead_idx].copy()

        return {
            "surface": torch.from_numpy(surface_full),
            "pressure_wind": torch.from_numpy(pressure_wind),
            "pressure_thermo": torch.from_numpy(pressure_thermo),
            "dem": torch.from_numpy(dem_full),
            "severe_weather": torch.from_numpy(severe),
            "rain_3h_mm": torch.from_numpy(rain3h),
            "persistence_rain_3h": torch.from_numpy(self.rain3h[w.input_end_idx].copy()),
            "persistence_severe": torch.from_numpy(self.severe[w.input_end_idx].copy()),
            "valid_time": str(self.times[w.input_end_idx]),
        }


def make_dataloaders_v2_plus(cache_path: str, cfg: Config):
    from torch.utils.data import DataLoader
    dl_cfg = cfg.get("dataloader") or {}
    loaders = {}
    for split in ("train", "val", "test"):
        ds = NowcastDatasetV2Plus(cache_path, cfg, split)
        loaders[split] = DataLoader(
            ds, batch_size=dl_cfg.get("batch_size", 32),
            shuffle=(split == "train"),
            num_workers=dl_cfg.get("num_workers", 0),
            pin_memory=dl_cfg.get("pin_memory", False),
            drop_last=(split == "train"),
        )
    return loaders


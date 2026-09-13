"""PyTorch Dataset over the preprocessed ERA5 cache for V2 model.
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


class NowcastDatasetV2(Dataset):
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
        
        # Spatial features (Lat/Lon)
        lat_min = cfg.get("domain", "lat_min")
        lat_max = cfg.get("domain", "lat_max")
        lon_min = cfg.get("domain", "lon_min")
        lon_max = cfg.get("domain", "lon_max")
        n_lat, n_lon = cfg.get("domain", "grid_shape")
        
        lats = np.linspace(lat_max, lat_min, n_lat)  # N -> S normally
        lons = np.linspace(lon_min, lon_max, n_lon)  # W -> E
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Normalize to [-1, 1]
        lat_grid = 2.0 * (lat_grid - lat_min) / (lat_max - lat_min) - 1.0
        lon_grid = 2.0 * (lon_grid - lon_min) / (lon_max - lon_min) - 1.0
        self.lat_grid = lat_grid.astype(np.float32)
        self.lon_grid = lon_grid.astype(np.float32)

        self.windows: list[WindowSample] = build_windows(self.times, cfg, split)
        self.windows = [w for w in self.windows
                        if all(self.label_valid[idx].all() for idx in w.lead_target_idx.values())]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        start = w.input_end_idx - self.input_hours + 1
        end = w.input_end_idx + 1
        
        # Slice lazily from the mmap objects and copy to memory
        surface = self.surface[start:end].copy()      # (input_hours, C, H, W)
        pressure = self.pressure[start:end].copy()    # (input_hours, C, L, H, W)

        # Apply normalization dynamically on this tiny chunk
        for j, v in enumerate(SINGLE_VARS):
            surface[:, j] = normalize_single(surface[:, j], v, self.stats)
        for j, v in enumerate(PRESSURE_VARS):
            pressure[:, j] = normalize_pressure(pressure[:, j], v, self.stats, level_axis=1)
            
        # Split pressure — stored level order is [250,300,500,700,850,1000]
        # u,v (var indices 0,1) have real data at level indices 3,4,5 (700/850/1000 hPa)
        # z,q,t (var indices 2,3,4) have real data at level indices 0,1,2,3 (250/300/500/700 hPa)
        pressure_wind = pressure[:, :2, 3:]    # u,v at 700/850/1000 → (T, 2, 3, H, W)
        pressure_thermo = pressure[:, 2:, :4]  # z,q,t at 250/300/500/700 → (T, 3, 4, H, W)
        
        # Compute derived surface features from normalized features
        # 0: u10, 1: v10, 2: d2m, 3: t2m
        wind_speed = np.sqrt(surface[:, 0]**2 + surface[:, 1]**2)
        dewpoint_dep = surface[:, 3] - surface[:, 2]
        
        # Temporal features
        times_window = pd.DatetimeIndex(self.times[start:end])
        hours = times_window.hour.values
        doys = times_window.dayofyear.values
        
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        doy_sin = np.sin(2 * np.pi * doys / 365.25)
        doy_cos = np.cos(2 * np.pi * doys / 365.25)
        
        H, W = surface.shape[-2:]
        T = self.input_hours
        
        # Broadcast temporal to grid
        temporal = np.stack([hour_sin, hour_cos, doy_sin, doy_cos], axis=1).astype(np.float32) # (T, 4)
        temporal = temporal[:, :, None, None].repeat(H, axis=2).repeat(W, axis=3) # (T, 4, H, W)
        
        derived = np.stack([wind_speed, dewpoint_dep], axis=1).astype(np.float32) # (T, 2, H, W)
        
        surface_full = np.concatenate([surface, derived, temporal], axis=1) # (T, 9+2+4=15, H, W)
        
        # DEM + spatial
        dem_full = np.stack([self.dem_norm, self.lat_grid, self.lon_grid], axis=0) # (3, H, W)

        lead_idx = [w.lead_target_idx[lh] for lh in self.lead_times]
        severe = self.severe[lead_idx].copy()         # (n_lead, H, W)
        rain3h = self.rain3h[lead_idx].copy()         # (n_lead, H, W)

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


def make_dataloaders_v2(cache_path: str, cfg: Config):
    from torch.utils.data import DataLoader
    dl_cfg = cfg.get("dataloader") or {}
    loaders = {}
    for split in ("train", "val", "test"):
        ds = NowcastDatasetV2(cache_path, cfg, split)
        loaders[split] = DataLoader(
            ds, batch_size=dl_cfg.get("batch_size", 32),
            shuffle=(split == "train"),
            num_workers=dl_cfg.get("num_workers", 0),
            pin_memory=dl_cfg.get("pin_memory", False),
            drop_last=(split == "train"),
        )
    return loaders

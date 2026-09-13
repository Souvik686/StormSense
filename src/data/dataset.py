"""PyTorch Dataset over the preprocessed ERA5 cache.

Each sample is:
  inputs:
    surface:   (input_hours, n_surface_vars, H, W)       float32, normalized
    pressure:  (input_hours, n_pressure_vars, n_levels, H, W) float32, normalized
    dem:       (1, H, W)                                  float32, normalized (static, broadcast per sample)
  targets:
    severe_weather: (n_lead_times,)  binary, one value per grid cell... but
      since this is a *spatial* nowcasting problem we keep the full grid:
    severe_weather_grid: (n_lead_times, H, W) binary
    rain_3h_mm_grid:     (n_lead_times, H, W) regression target (mm)
  meta:
    valid_time (input window end), lead_times_hours
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import Config
from src.data.windowing import build_windows, WindowSample
from src.features.normalize import (
    SINGLE_VARS, PRESSURE_VARS, load_stats, normalize_single, normalize_pressure,
)


class NowcastDataset(Dataset):
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

        self.windows: list[WindowSample] = build_windows(self.times, cfg, split)
        # Drop windows whose target grid has any warm-up NaN target cell, so
        # every returned sample has fully-defined supervision.
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

        lead_idx = [w.lead_target_idx[lh] for lh in self.lead_times]
        severe = self.severe[lead_idx].copy()         # (n_lead, H, W)
        rain3h = self.rain3h[lead_idx].copy()         # (n_lead, H, W)

        return {
            "surface": torch.from_numpy(surface),
            "pressure": torch.from_numpy(pressure),
            "dem": torch.from_numpy(self.dem_norm[None].copy()),
            "severe_weather": torch.from_numpy(severe),
            "rain_3h_mm": torch.from_numpy(rain3h),
            "persistence_rain_3h": torch.from_numpy(self.rain3h[w.input_end_idx].copy()),
            "persistence_severe": torch.from_numpy(self.severe[w.input_end_idx].copy()),
            "valid_time": str(self.times[w.input_end_idx]),
        }


def make_dataloaders(cache_path: str, cfg: Config):
    from torch.utils.data import DataLoader
    dl_cfg = cfg.get("dataloader") or {}
    loaders = {}
    for split in ("train", "val", "test"):
        ds = NowcastDataset(cache_path, cfg, split)
        loaders[split] = DataLoader(
            ds, batch_size=dl_cfg.get("batch_size", 32),
            shuffle=(split == "train"),
            num_workers=dl_cfg.get("num_workers", 0),
            pin_memory=dl_cfg.get("pin_memory", False),
            drop_last=(split == "train"),
        )
    return loaders

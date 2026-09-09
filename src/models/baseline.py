"""Baseline models to establish whether the advanced model earns its
complexity. Two baselines, cheapest first:

  1. PersistenceBaseline (no learning): "severe weather at t+lead equals
     severe weather at the last input timestep" and "rain_3h_mm at t+lead
     equals rain_3h_mm at the last input timestep". Standard nowcasting
     sanity baseline -- if a learned model cannot beat this it has learned
     nothing useful about atmospheric evolution.

  2. LogisticConvBaseline: a single 1x1-conv (== per-pixel logistic
     regression over channels) applied to the time-flattened, channel-
     concatenated input, with one linear head per lead time. This is the
     simplest *learned* spatial model: it uses instantaneous multivariate
     information per grid cell but no spatial context and no temporal
     dynamics beyond flattening the lookback window into channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PersistenceBaseline:
    """Not an nn.Module -- a rule, evaluated directly on a batch dict."""

    def __init__(self, n_lead_times: int):
        self.n_lead_times = n_lead_times

    def predict(self, batch: dict) -> dict:
        last_surface = batch["surface"][:, -1]          # (B, C, H, W)
        tp_idx = 8  # SINGLE_VARS order: ... , "tp" is last (see normalize.py)
        last_tp = last_surface[:, tp_idx]                # normalized tp at last input hour, unused directly
        # Persistence must act on the *label-scale* quantity, not the model
        # input; the caller supplies the raw rain_3h at the input-window end
        # in batch["persistence_rain_3h"] and batch["persistence_severe"].
        rain = batch["persistence_rain_3h"].unsqueeze(1).repeat(1, self.n_lead_times, 1, 1)
        severe = batch["persistence_severe"].unsqueeze(1).repeat(1, self.n_lead_times, 1, 1)
        return {"severe_weather_logit": torch.logit(severe.clamp(1e-4, 1 - 1e-4)),
                "rain_3h_mm": rain}


class LogisticConvBaseline(nn.Module):
    """Per-pixel linear/logistic model: flattens (input_hours x surface_vars)
    into channels, 1x1 conv to a shared hidden width, then one 1x1-conv head
    per output (severe-weather logit and rain regression) across lead times.
    Ignores pressure-level fields and any spatial neighbourhood beyond the
    single pixel -- deliberately weak, to bound how much the advanced model's
    spatial/vertical/temporal machinery actually buys.
    """

    def __init__(self, n_surface_vars: int, input_hours: int, n_lead_times: int, dem_channels: int = 1):
        super().__init__()
        in_ch = n_surface_vars * input_hours + dem_channels
        hidden = 32
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.severe_head = nn.Conv2d(hidden, n_lead_times, kernel_size=1)
        self.rain_head = nn.Conv2d(hidden, n_lead_times, kernel_size=1)

    def forward(self, batch: dict) -> dict:
        surface = batch["surface"]                       # (B, T, C, H, W)
        b, t, c, h, w = surface.shape
        flat = surface.reshape(b, t * c, h, w)
        x = torch.cat([flat, batch["dem"]], dim=1)
        feat = self.proj(x)
        severe_logit = self.severe_head(feat)             # (B, n_lead, H, W)
        rain = torch.relu(self.rain_head(feat))            # rainfall is non-negative
        return {"severe_weather_logit": severe_logit, "rain_3h_mm": rain}

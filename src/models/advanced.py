"""Advanced multimodal spatiotemporal model for severe-weather nowcasting.

Architecture (deliberately sized for Colab single-GPU training on ~33x25
grids, not scaled up just to look impressive):

  Surface encoder   : ConvGRU over the (input_hours) sequence of surface
                       fields (u10,v10,d2m,t2m,sp,cape,cin,tcwv,tp) ->
                       captures near-surface temporal evolution (e.g. rising
                       CAPE, veering wind) at each grid cell with a 3x3
                       spatial receptive field per step.
  Pressure encoder  : a small 3D conv block over (level, H, W) applied at
                       each input timestep independently, pooled over level,
                       then fed through the same ConvGRU as an extra
                       feature stream -- captures vertical-profile structure
                       (e.g. mid-level dry intrusion, upper divergence) that
                       is invisible to single-level fields alone.
  Static encoder    : DEM elevation passed through a 1x1 conv (orographic
                       modulation of the fused features, e.g. windward
                       enhancement) and concatenated at the fusion step.
  Fusion            : channel-concat of surface-GRU state, pressure-GRU
                       state, and DEM features, then a 3x3 conv fusion
                       block.
  Multi-task heads  : one shared trunk, two heads per lead time --
                       (a) severe-weather binary logit (classification)
                       (b) 3h-rainfall regression (mm) -- trained jointly
                       since both derive from the same underlying moisture/
                       instability state and rainfall regression gives the
                       classification head a denser gradient signal.

This is intentionally a single compact model (not one encoder per lead
time): all lead times share the trunk and only the final 1x1-conv heads are
lead-time-specific, which is both more parameter-efficient and enforces a
smooth diurnal/lead-time relationship in the shared representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        reset, update = torch.chunk(torch.sigmoid(gates), 2, dim=1)
        combined_r = torch.cat([x, reset * h], dim=1)
        cand = torch.tanh(self.candidate(combined_r))
        return (1 - update) * h + update * cand


class ConvGRU(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvGRUCell(in_ch, hidden_ch, kernel_size)
        self.hidden_ch = hidden_ch

    def forward(self, seq):
        # seq: (B, T, C, H, W)
        b, t, c, h, w = seq.shape
        state = torch.zeros(b, self.hidden_ch, h, w, device=seq.device, dtype=seq.dtype)
        for i in range(t):
            state = self.cell(seq[:, i], state)
        return state  # (B, hidden_ch, H, W)


class PressureEncoder(nn.Module):
    """Per-timestep 3D-ish encoder: 2D conv applied independently to each
    pressure level, then a weighted pool over levels (levels differ in
    physical meaning, so a learned per-level weight is more appropriate than
    a plain mean)."""

    def __init__(self, n_vars: int, n_levels: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_vars, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_weight = nn.Parameter(torch.ones(n_levels) / n_levels)
        self.proj = nn.Conv2d(16, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, T, n_vars, n_levels, H, W)
        b, t, v, l, h, w = x.shape
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * t * l, v, h, w)
        feat = self.conv(x)                                   # (B*T*L, 16, H, W)
        feat = feat.reshape(b, t, l, 16, h, w)
        weights = torch.softmax(self.level_weight, dim=0).view(1, 1, l, 1, 1, 1)
        pooled = (feat * weights).sum(dim=2)                   # (B, T, 16, H, W)
        pooled = self.proj(pooled.reshape(b * t, 16, h, w)).reshape(b, t, -1, h, w)
        return pooled


class SevereWeatherNet(nn.Module):
    def __init__(self, n_surface_vars: int, n_pressure_vars: int, n_levels: int,
                 n_lead_times: int, surface_hidden: int = 32, pressure_out_ch: int = 16,
                 fusion_ch: int = 48):
        super().__init__()
        self.n_lead_times = n_lead_times
        self.pressure_encoder = PressureEncoder(n_pressure_vars, n_levels, pressure_out_ch)
        self.surface_gru = ConvGRU(n_surface_vars, surface_hidden)
        self.pressure_gru = ConvGRU(pressure_out_ch, surface_hidden)
        self.dem_proj = nn.Conv2d(1, 8, kernel_size=1)

        fusion_in = surface_hidden * 2 + 8
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, fusion_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_ch, fusion_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.severe_head = nn.Conv2d(fusion_ch, n_lead_times, kernel_size=1)
        self.rain_head = nn.Conv2d(fusion_ch, n_lead_times, kernel_size=1)

    def forward(self, batch: dict) -> dict:
        surface = batch["surface"]        # (B, T, Cs, H, W)
        pressure = batch["pressure"]       # (B, T, Cp, L, H, W)
        dem = batch["dem"]                 # (B, 1, H, W)

        surf_state = self.surface_gru(surface)
        pressure_feat = self.pressure_encoder(pressure)   # (B, T, pressure_out_ch, H, W)
        pres_state = self.pressure_gru(pressure_feat)
        dem_feat = torch.relu(self.dem_proj(dem))

        fused_in = torch.cat([surf_state, pres_state, dem_feat], dim=1)
        fused = self.fusion(fused_in)

        severe_logit = self.severe_head(fused)             # (B, n_lead, H, W)
        rain = torch.relu(self.rain_head(fused))
        return {"severe_weather_logit": severe_logit, "rain_3h_mm": rain}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

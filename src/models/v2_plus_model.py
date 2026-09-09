"""SevereWeatherNet V2+ Model with Enhanced Atmospheric Physics Inputs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.v2_model import ConvGRU, ConvGRUCell, PressureEncoder


class SevereWeatherNetV2Plus(nn.Module):
    def __init__(
        self,
        n_surface_vars: int = 17,
        n_wind_vars: int = 2,
        n_wind_levels: int = 3,
        n_thermo_vars: int = 3,
        n_thermo_levels: int = 4,
        n_dem_vars: int = 4,
        n_lead_times: int = 5,
    ):
        super().__init__()
        self.n_lead_times = n_lead_times

        # Encoders
        self.surface_gru = ConvGRU(n_surface_vars, 64)

        self.wind_enc = PressureEncoder(n_wind_vars, n_wind_levels, 16)
        self.wind_gru = ConvGRU(16, 32)

        self.thermo_enc = PressureEncoder(n_thermo_vars, n_thermo_levels, 16)
        self.thermo_gru = ConvGRU(16, 32)

        # 4 DEM/Spatial channels (elevation, slope, lat, lon) -> 16
        self.dem_proj = nn.Conv2d(n_dem_vars, 16, kernel_size=1)

        # Fusion: 64 + 32 + 32 + 16 = 144
        fusion_in = 64 + 32 + 32 + 16
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Multi-scale pyramid
        self.down1 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)

        # Autoregressive Lead-time Decoder
        self.lead_embed = nn.Embedding(n_lead_times, 8)
        self.decoder_cell = ConvGRUCell(8, 96)
        self.severe_head = nn.Conv2d(96, 1, kernel_size=1)
        self.rain_head = nn.Conv2d(96, 1, kernel_size=1)

    def forward(self, batch: dict) -> dict:
        surface = batch["surface"]
        pressure_wind = batch["pressure_wind"]
        pressure_thermo = batch["pressure_thermo"]
        dem = batch["dem"]

        # Encode
        surf_state = self.surface_gru(surface)
        wind_state = self.wind_gru(self.wind_enc(pressure_wind))
        thermo_state = self.thermo_gru(self.thermo_enc(pressure_thermo))
        dem_feat = torch.relu(self.dem_proj(dem))

        # Fuse
        fused_in = torch.cat([surf_state, wind_state, thermo_state, dem_feat], dim=1)
        x0 = self.fusion(fused_in)

        # Multi-scale
        x1 = torch.relu(self.down1(x0))
        x2 = torch.relu(self.down2(x1))

        x1_up = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x0_up = F.interpolate(x1 + x1_up, size=x0.shape[2:], mode="bilinear", align_corners=False)
        state = x0 + x0_up

        # Decode
        b, _, h, w = state.shape
        severe_preds, rain_preds = [], []

        for i in range(self.n_lead_times):
            idx = torch.full((b,), i, dtype=torch.long, device=state.device)
            emb = self.lead_embed(idx)
            emb = emb.view(b, 8, 1, 1).expand(-1, -1, h, w)
            state = self.decoder_cell(emb, state)
            severe_preds.append(self.severe_head(state))
            rain_preds.append(torch.relu(self.rain_head(state)))

        severe_logit = torch.cat(severe_preds, dim=1)
        rain_log = torch.cat(rain_preds, dim=1)
        rain_mm = torch.expm1(rain_log)

        return {
            "severe_weather_logit": severe_logit,
            "rain_log": rain_log,
            "rain_3h_mm": rain_mm,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


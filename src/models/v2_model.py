"""V2 Model for SevereWeatherNet"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        b, t, c, h, w = seq.shape
        state = torch.zeros(b, self.hidden_ch, h, w, device=seq.device, dtype=seq.dtype)
        for i in range(t):
            state = self.cell(seq[:, i], state)
        return state

class PressureEncoder(nn.Module):
    def __init__(self, n_vars: int, n_levels: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_vars, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_weight = nn.Parameter(torch.ones(n_levels) / n_levels)
        self.proj = nn.Conv2d(16, out_ch, kernel_size=1)

    def forward(self, x):
        b, t, v, l, h, w = x.shape
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * t * l, v, h, w)
        feat = self.conv(x)
        feat = feat.reshape(b, t, l, 16, h, w)
        weights = torch.softmax(self.level_weight, dim=0).view(1, 1, l, 1, 1, 1)
        pooled = (feat * weights).sum(dim=2)
        pooled = self.proj(pooled.reshape(b * t, 16, h, w)).reshape(b, t, -1, h, w)
        return pooled

class SevereWeatherNetV2(nn.Module):
    def __init__(self, n_surface_vars: int = 15, 
                 n_wind_vars: int = 2, n_wind_levels: int = 3,
                 n_thermo_vars: int = 3, n_thermo_levels: int = 4,
                 n_lead_times: int = 5):
        super().__init__()
        self.n_lead_times = n_lead_times
        
        # Encoders
        self.surface_gru = ConvGRU(n_surface_vars, 64)
        
        self.wind_enc = PressureEncoder(n_wind_vars, n_wind_levels, 16)
        self.wind_gru = ConvGRU(16, 32)
        
        self.thermo_enc = PressureEncoder(n_thermo_vars, n_thermo_levels, 16)
        self.thermo_gru = ConvGRU(16, 32)
        
        self.dem_proj = nn.Conv2d(3, 16, kernel_size=1)
        
        # Fusion
        fusion_in = 64 + 32 + 32 + 16 # 144
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale
        self.down1 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        
        # Decoder
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
        
        x1_up = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x0_up = F.interpolate(x1 + x1_up, size=x0.shape[2:], mode='bilinear', align_corners=False)
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
        
        return {"severe_weather_logit": severe_logit, "rain_log": rain_log, "rain_3h_mm": rain_mm}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

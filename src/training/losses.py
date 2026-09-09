"""Multi-task loss: focal loss for the imbalanced severe-weather label
(base rate ~3%, see docs/ML_DESIGN.md) + MSE for the rainfall regression,
combined per lead time with equal weight by default (configurable).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                            alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017). alpha up-weights the positive
    (severe-weather) class to counter its ~3% base rate."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, rain_weight: float = 0.5, focal_alpha: float = 0.75, focal_gamma: float = 2.0):
        super().__init__()
        self.rain_weight = rain_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, preds: dict, batch: dict) -> dict:
        severe_loss = focal_loss_with_logits(
            preds["severe_weather_logit"], batch["severe_weather"],
            alpha=self.focal_alpha, gamma=self.focal_gamma)
        rain_loss = F.mse_loss(preds["rain_3h_mm"], batch["rain_3h_mm"])
        total = severe_loss + self.rain_weight * rain_loss
        return {"total": total, "severe_loss": severe_loss, "rain_loss": rain_loss}


class MultiTaskLossV2(nn.Module):
    def __init__(self, rain_weight: float = 0.05, focal_alpha: float = 0.90, focal_gamma: float = 2.5):
        super().__init__()
        self.rain_weight = rain_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, preds: dict, batch: dict) -> dict:
        severe_loss = focal_loss_with_logits(
            preds["severe_weather_logit"], batch["severe_weather"],
            alpha=self.focal_alpha, gamma=self.focal_gamma)
        
        target_log = torch.log1p(batch["rain_3h_mm"])
        pred_log = preds["rain_log"] if "rain_log" in preds else torch.log1p(preds["rain_3h_mm"])
        rain_loss = F.smooth_l1_loss(pred_log, target_log)
        
        total = severe_loss + self.rain_weight * rain_loss
        return {"total": total, "severe_loss": severe_loss, "rain_loss": rain_loss}

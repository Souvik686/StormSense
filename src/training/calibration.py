"""Meteorological probability calibration and reliability verification.

Implements:
  1. Brier Score (BS) and Brier Skill Score (BSS)
  2. Expected Calibration Error (ECE)
  3. Reliability Diagram binning
  4. Temperature Scaling (Guo et al. 2017)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute Brier Score, Brier Skill Score, and Expected Calibration Error."""
    y_true = y_true.ravel().astype(np.float64)
    y_prob = y_prob.ravel().astype(np.float64)
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    # 1. Brier Score
    bs = float(np.mean((y_prob - y_true) ** 2))
    base_rate = float(np.mean(y_true))
    bs_ref = base_rate * (1.0 - base_rate)
    bss = float(1.0 - (bs / bs_ref)) if bs_ref > 0 else float("nan")

    # 2. Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []
    n = len(y_true)

    for m in range(n_bins):
        bin_lower, bin_upper = bins[m], bins[m + 1]
        if m < n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        else:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin": m,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })
        else:
            bin_data.append({
                "bin": m,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": 0,
                "accuracy": None,
                "confidence": None,
            })

    return {
        "brier_score": round(bs, 6),
        "brier_skill_score": round(bss, 4),
        "base_rate": round(base_rate, 4),
        "ece": round(float(ece), 6),
        "bins": bin_data,
    }


class TemperatureScaler(nn.Module):
    """Post-processing temperature scaling for multi-horizon binary logits."""
    def __init__(self, n_lead_times: int = 5):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(n_lead_times))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        shape = [1] * logits.ndim
        shape[1] = -1
        return logits / T.view(*shape)

    def fit(self, logits_np: np.ndarray, targets_np: np.ndarray, max_iter: int = 50, lr: float = 0.05) -> list[float]:
        n_lead = self.log_temp.shape[0]
        temps = []
        criterion = nn.BCEWithLogitsLoss()

        for li in range(n_lead):
            z = torch.from_numpy(logits_np[:, li].ravel()).float()
            y = torch.from_numpy(targets_np[:, li].ravel()).float()

            log_t = nn.Parameter(torch.zeros(1))
            opt = optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

            def eval_loss():
                opt.zero_grad()
                loss = criterion(z / torch.exp(log_t), y)
                loss.backward()
                return loss

            opt.step(eval_loss)
            t_val = float(torch.exp(log_t).item())
            temps.append(round(t_val, 4))

        self.log_temp.data = torch.log(torch.tensor(temps, dtype=torch.float32))
        return temps


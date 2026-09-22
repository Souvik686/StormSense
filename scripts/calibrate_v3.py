"""Fit per-lead temperature scaling for a candidate checkpoint.

Raw sigmoid output is NOT a calibrated probability. The v3 long-lead checkpoint
came out of training with no temperature scaling at all, and measured badly:
expected calibration error 0.22-0.23 on the test split, i.e. a cell shown as
"60%" was not occurring 60% of the time. Production's v2 checkpoint carries
temperature_per_lead ~0.38-0.42 for exactly this reason.

Temperatures are fitted on the VALIDATION split only (never test), by minimising
BCE on the logits, then written into the checkpoint as `temperature_per_lead` --
the same field src/inference/predictor.py already reads and applies.

Reliability is reported BEFORE and AFTER so the improvement is measured rather
than assumed. Decision thresholds are deliberately NOT refitted here: they are a
separate quantity, optimised by scripts/optimize_threshold.py, and re-running
that after calibration is a conscious follow-up step.

Usage:
    python scripts/calibrate_v3.py --config configs/v3_longlead.yaml \
        --checkpoint Data/outputs/checkpoints_v3/v2_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("STORMSENSE_DEVICE", "cpu")

from src.data.dataset_v2 import make_dataloaders_v2
from src.models.v2_model import SevereWeatherNetV2
from src.training.calibration import TemperatureScaler
from src.utils.config import load_config


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_longlead.yaml")
    ap.add_argument("--checkpoint", default="Data/outputs/checkpoints_v3/v2_best.pt")
    ap.add_argument("--out", default="reports/v3_calibration.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cpu")
    leads = list(cfg.get("sequence", "lead_times_hours"))

    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = SevereWeatherNetV2(n_lead_times=len(leads)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[calib] {a.checkpoint}  epoch={ck.get('epoch')}  leads={leads}")
    print(f"[calib] existing temperature_per_lead={ck.get('temperature_per_lead')}")

    cache = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    loaders = make_dataloaders_v2(cache, cfg)

    # Fit on VALIDATION only. Using test here would leak the evaluation set
    # into the model's calibration.
    dl = loaders["val"]
    print(f"[calib] fitting on val: {len(dl.dataset)} samples, {len(dl)} batches")

    logits, targets = [], []
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            out = model({k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()})
            logits.append(out["severe_weather_logit"].cpu().numpy())
            targets.append(batch["severe_weather"].numpy())
            if (bi + 1) % 40 == 0:
                print(f"    ...{bi+1}/{len(dl)}")

    Z = np.concatenate(logits)
    Y = np.concatenate(targets)
    print(f"[calib] logits {Z.shape}")

    before = {}
    for li, L in enumerate(leads):
        p = 1.0 / (1.0 + np.exp(-Z[:, li].ravel().astype(np.float64)))
        y = Y[:, li].ravel().astype(np.float64)
        before[L] = {"ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)),
                     "mean_pred": float(p.mean()), "base_rate": float(y.mean())}

    scaler = TemperatureScaler(n_lead_times=len(leads))
    temps = scaler.fit(Z, Y)
    print(f"[calib] fitted temperature_per_lead = {temps}")

    after = {}
    for li, L in enumerate(leads):
        z = Z[:, li].ravel().astype(np.float64) / temps[li]
        p = 1.0 / (1.0 + np.exp(-z))
        y = Y[:, li].ravel().astype(np.float64)
        after[L] = {"ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)),
                    "mean_pred": float(p.mean()), "base_rate": float(y.mean())}

    print()
    print(f"{'lead':>6} {'T':>8} {'ECE before':>11} {'ECE after':>10} "
          f"{'Brier before':>13} {'Brier after':>12}")
    improved = 0
    for li, L in enumerate(leads):
        b, af = before[L], after[L]
        if af["ece"] < b["ece"]:
            improved += 1
        print(f"{'+'+str(L)+'h':>6} {temps[li]:8.4f} {b['ece']:11.4f} "
              f"{af['ece']:10.4f} {b['brier']:13.5f} {af['brier']:12.5f}")

    if improved < len(leads):
        print(f"\n[calib] WARNING: calibration improved only {improved}/{len(leads)} leads")

    ck["temperature_per_lead"] = temps
    ck["is_calibrated"] = True
    ck["calibration_method"] = "per-lead temperature scaling (LBFGS on val BCE)"
    torch.save(ck, a.checkpoint)
    print(f"\n[calib] wrote temperature_per_lead into {a.checkpoint}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": a.checkpoint,
            "fitted_on": "val",
            "leads": leads,
            "temperature_per_lead": temps,
            "before": {str(k): v for k, v in before.items()},
            "after": {str(k): v for k, v in after.items()},
        }, f, indent=2)
    print(f"[calib] wrote {a.out}")


if __name__ == "__main__":
    main()

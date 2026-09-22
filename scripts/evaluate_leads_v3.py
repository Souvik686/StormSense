"""Per-lead skill evaluation for a candidate checkpoint, on the TEST split.

Answers one question per forecast lead: does this lead carry enough skill to be
exposed in production, or should it stay hidden?

Reports, for every lead:
  - PR-AUC and the no-skill baseline (event base rate). PR-AUC is the headline
    because the target is heavily imbalanced (~2.2% positive), which makes
    accuracy meaningless.
  - Confusion matrix, precision, recall, F1, CSI, FAR at the checkpoint's own
    per-lead decision threshold.
  - Brier score and expected calibration error, so a well-ranked but badly
    calibrated lead is not mistaken for a usable one.
  - Persistence-baseline CSI, the same reference the training script prints.

Writes JSON so the numbers in any later report are traceable to a real run.

Usage:
    python scripts/evaluate_leads_v3.py --config configs/v3_longlead.yaml \
        --checkpoint Data/outputs/checkpoints_v3/v2_best.pt \
        --out reports/v3_lead_evaluation.json
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
from src.utils.config import load_config


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision. Computed directly so sklearn is not required."""
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1e-12)
    total_pos = max(float(y.sum()), 1e-12)
    rec = tp / total_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def ece(y_true: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.mean()) * abs(p[m].mean() - y_true[m].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_longlead.yaml")
    ap.add_argument("--checkpoint", default="Data/outputs/checkpoints_v3/v2_best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="reports/v3_lead_evaluation.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cpu")
    leads = list(cfg.get("sequence", "lead_times_hours"))

    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = SevereWeatherNetV2(n_lead_times=len(leads)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    temp = ck.get("temperature_per_lead")
    thr_per_lead = ck.get("threshold_per_lead") or {}
    print(f"[eval] checkpoint {a.checkpoint}")
    print(f"[eval] epoch={ck.get('epoch')}  leads={leads}")
    print(f"[eval] temperature_per_lead={temp}")
    print(f"[eval] threshold_per_lead={thr_per_lead}")

    cache = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    loaders = make_dataloaders_v2(cache, cfg)
    dl = loaders[a.split]
    print(f"[eval] split={a.split}  samples={len(dl.dataset)}  batches={len(dl)}")

    probs, truths, persist = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            # SevereWeatherNetV2.forward takes the whole batch dict and reads
            # surface / pressure_wind / pressure_thermo / dem from it.
            out = model({k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()})
            logit = out["severe_weather_logit"]
            if temp is not None:
                t = torch.tensor(temp, dtype=logit.dtype).view(1, -1, 1, 1)
                logit = logit / t
            probs.append(torch.sigmoid(logit).cpu().numpy())
            truths.append(batch["severe_weather"].numpy())
            if "persistence_severe" in batch:
                persist.append(batch["persistence_severe"].numpy())
            if (bi + 1) % 40 == 0:
                print(f"    ...{bi+1}/{len(dl)} batches")

    P = np.concatenate(probs)        # (N, n_lead, H, W)
    Y = np.concatenate(truths)
    PS = np.concatenate(persist) if persist else None
    print(f"[eval] prob array {P.shape}")

    results = {
        "checkpoint": a.checkpoint,
        "epoch": int(ck.get("epoch", -1)),
        "split": a.split,
        "leads": leads,
        "temperature_per_lead": temp,
        "threshold_per_lead": {str(k): v for k, v in thr_per_lead.items()},
        "per_lead": {},
    }

    for li, L in enumerate(leads):
        p = P[:, li].ravel().astype(np.float64)
        y = Y[:, li].ravel().astype(np.float64)
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        base = float(y.mean())

        thr = float(thr_per_lead.get(L, thr_per_lead.get(str(L), 0.5)))
        pred = (p >= thr).astype(np.float64)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())

        prec = tp / max(tp + fp, 1e-12)
        rec = tp / max(tp + fn, 1e-12)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        csi = tp / max(tp + fp + fn, 1e-12)
        far = fp / max(tp + fp, 1e-12)
        ap_ = pr_auc(y, p)

        entry = {
            "lead_hours": L,
            "n_cells": int(y.size),
            "event_base_rate": base,
            "threshold_used": thr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall_pod": rec, "f1": f1,
            "csi": csi, "far": far,
            "pr_auc": ap_,
            "pr_auc_no_skill": base,
            "skill_vs_no_skill": ap_ / max(base, 1e-12),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": ece(y, p),
        }

        # Persistence baseline: "the severe field at the last input timestep
        # persists to the target time". It is per-SAMPLE (N, H, W), not
        # per-lead, so the same field is scored against each lead's target --
        # which is exactly what makes it a baseline.
        if PS is not None and PS.ndim == 3:
            ps = PS.reshape(PS.shape[0], -1).ravel().astype(np.float64)[ok]
            ptp = float(((ps == 1) & (y == 1)).sum())
            pfp = float(((ps == 1) & (y == 0)).sum())
            pfn = float(((ps == 0) & (y == 1)).sum())
            entry["persistence_csi"] = ptp / max(ptp + pfp + pfn, 1e-12)

        results["per_lead"][str(L)] = entry
        print(f"  +{L:2d}h  PR-AUC {ap_:.4f} (no-skill {base:.4f}, "
              f"{ap_/max(base,1e-12):.2f}x)  CSI {csi:.4f}  POD {rec:.4f}  "
              f"FAR {far:.4f}  Brier {entry['brier']:.5f}  ECE {entry['ece']:.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] wrote {a.out}")


if __name__ == "__main__":
    main()

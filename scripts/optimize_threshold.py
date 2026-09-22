"""Threshold optimization script.

After training, run this against the validation set to find the optimal
decision threshold for the severe-weather classification head.

The threshold is then:
  1. Saved into the checkpoint (so the API uses it automatically)
  2. Reported per lead time (threshold may differ by lead time)

Objective: maximize CSI (Critical Success Index), which is the most
operationally meaningful single metric for a warning system because it
simultaneously penalizes false alarms AND misses:

    CSI = TP / (TP + FP + FN)

Usage:
    python scripts/optimize_threshold.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data.dataset import make_dataloaders
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS
from src.models.advanced import SevereWeatherNet
from src.training.metrics import classification_metrics


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def collect_predictions(model, loader, device, lead_times: list,
                        temperature: list | None = None) -> dict:
    """Accumulate all val predictions into arrays. Returns per-lead arrays.

    Applies the checkpoint's per-lead TEMPERATURE SCALING before the sigmoid,
    matching what src/inference/predictor.py serves. Without this the optimiser
    tunes thresholds against RAW sigmoid output while production compares them
    to CALIBRATED probabilities -- two different distributions, so the operating
    point chosen here would not be the operating point actually applied. (With
    T ~0.29 the calibrated probabilities are markedly sharper, so the mismatch
    is not a rounding detail.)
    """
    model.eval()
    all_probs = [[] for _ in lead_times]
    all_true  = [[] for _ in lead_times]
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        logit = out["severe_weather_logit"]
        if temperature is not None:
            t = torch.tensor(temperature, dtype=logit.dtype, device=logit.device)
            logit = logit / t.view(1, -1, 1, 1)
        probs = torch.sigmoid(logit).cpu().numpy()  # (B, L, H, W)
        true  = batch["severe_weather"].cpu().numpy()
        for li in range(len(lead_times)):
            all_probs[li].append(probs[:, li].ravel())
            all_true[li].append(true[:, li].ravel())
    return {
        "probs": [np.concatenate(p) for p in all_probs],
        "true":  [np.concatenate(t) for t in all_true],
    }


def find_best_threshold(y_prob: np.ndarray, y_true: np.ndarray,
                         thresholds: np.ndarray,
                         objective: str = "csi") -> tuple[float, dict]:
    """Grid-search threshold for a given objective."""
    best_val, best_thr, best_metrics = -np.inf, 0.5, {}
    for thr in thresholds:
        m = classification_metrics(y_true, y_prob, threshold=float(thr), fast=True)
        score = m[objective]
        if score > best_val:
            best_val, best_thr, best_metrics = score, float(thr), m
    
    # Compute the full metrics with AUC only once for the optimal threshold
    best_metrics = classification_metrics(y_true, y_prob, threshold=best_thr, fast=False)
    return best_thr, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint. Defaults to outputs/checkpoints/best.pt")
    parser.add_argument("--objective", default="csi",
                        choices=["csi", "f1", "recall_pod"])
    parser.add_argument("--n-thresholds", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[threshold] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    if not os.path.exists(cache_path):
        print(f"ERROR: cache not found at {cache_path}. Run preprocessing first.")
        sys.exit(1)

    ckpt_path = args.checkpoint or os.path.join(
        cfg.path("training", "checkpoint_dir"), "v2_best.pt" if os.path.exists(os.path.join(cfg.path("training", "checkpoint_dir"), "v2_best.pt")) else "best.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    lead_times = cfg.get("sequence", "lead_times_hours")
    n_levels = len(cfg.get("era5", "pressure_levels_hpa"))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "dem_proj.weight" in ckpt["model"] and ckpt["model"]["dem_proj.weight"].shape[1] == 4:
        from src.models.v2_plus_model import SevereWeatherNetV2Plus
        from src.data.dataset_v2_plus import make_dataloaders_v2_plus
        loaders = make_dataloaders_v2_plus(cache_path, cfg)
        model = SevereWeatherNetV2Plus(n_lead_times=len(lead_times))
        print(f"[threshold] detected V2+ architecture in {ckpt_path}")
    elif "wind_enc.level_weight" in ckpt["model"]:
        from src.models.v2_model import SevereWeatherNetV2
        from src.data.dataset_v2 import make_dataloaders_v2
        loaders = make_dataloaders_v2(cache_path, cfg)
        model = SevereWeatherNetV2(n_lead_times=len(lead_times))
        print(f"[threshold] detected V2 architecture in {ckpt_path}")
    else:
        loaders = make_dataloaders(cache_path, cfg)
        model = SevereWeatherNet(
            n_surface_vars=len(SINGLE_VARS),
            n_pressure_vars=len(PRESSURE_VARS),
            n_levels=n_levels,
            n_lead_times=len(lead_times),
        )
        print(f"[threshold] detected V1 architecture in {ckpt_path}")

    model.load_state_dict(ckpt["model"])
    model.to(device)

    print("[threshold] collecting validation predictions ...")
    # Use the checkpoint's calibration, so thresholds are chosen against the
    # same probabilities inference will produce.
    _temp = ckpt.get("temperature_per_lead")
    if _temp is not None:
        print(f"[threshold] applying temperature_per_lead={_temp}")
    else:
        print("[threshold] checkpoint carries no temperature; using raw sigmoid")
    preds = collect_predictions(model, loaders["val"], device, lead_times,
                                temperature=_temp)

    thresholds = np.linspace(0.05, 0.95, args.n_thresholds)
    results = {}
    best_per_lead = {}
    print(f"\n[threshold] optimizing for '{args.objective}' per lead time:")
    for li, lh in enumerate(lead_times):
        y_prob = preds["probs"][li]
        y_true = preds["true"][li]
        best_thr, best_m = find_best_threshold(y_prob, y_true, thresholds, args.objective)
        best_per_lead[lh] = best_thr
        results[f"lead_{lh}h"] = {
            "optimal_threshold": float(round(best_thr, 3)),
            "csi": float(round(best_m["csi"], 4)),
            "recall_pod": float(round(best_m["recall_pod"], 4)),
            "far": float(round(best_m["far"], 4)),
            "f1": float(round(best_m["f1"], 4)),
            "pr_auc": float(round(best_m.get("pr_auc", float("nan")), 4)),
            "roc_auc": float(round(best_m.get("roc_auc", float("nan")), 4)),
            "n_positive": int(best_m["n_positive"]),
            "n_total": int(best_m["n"]),
        }
        print(f"  lead_{lh}h: thr={best_thr:.3f}  csi={best_m['csi']:.4f}  "
              f"pod={best_m['recall_pod']:.4f}  far={best_m['far']:.4f}")

    # Use median threshold across lead times as the single operational threshold
    # (simpler for the API to store one number)
    global_threshold = float(np.median(list(best_per_lead.values())))
    print(f"\n[threshold] global threshold (median across leads): {global_threshold:.3f}")

    # Patch the checkpoint
    ckpt["optimal_threshold"] = global_threshold
    ckpt["threshold_per_lead"] = {int(lh): float(thr) for lh, thr in best_per_lead.items()}
    ckpt["threshold_results"] = results
    torch.save(ckpt, ckpt_path)
    print(f"[threshold] saved updated checkpoint -> {ckpt_path}")

    # Also save a JSON report
    out_dir = cfg.path("paths", "outputs_root")
    report_path = os.path.join(out_dir, "metrics", "threshold_optimization.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "objective": args.objective,
            "global_threshold": global_threshold,
            "per_lead": results,
        }, f, indent=2)
    print(f"[threshold] report -> {report_path}")


if __name__ == "__main__":
    main()


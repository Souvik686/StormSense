"""Validation evaluation script.

Runs strictly on the 2023 Validation split (never touches test split).
Computes all classification, regression, calibration, and reliability metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data.dataset_v2 import make_dataloaders_v2
from src.models.v2_model import SevereWeatherNetV2
from src.training.losses import MultiTaskLossV2
from src.training.metrics import classification_metrics, regression_metrics
from src.training.calibration import compute_calibration_metrics


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def run_validation_evaluation(model, loader, loss_fn, lead_times, device, thresholds: dict | float = 0.5) -> dict:
    model.eval()
    total_loss, total_severe, total_rain, n_batches = 0.0, 0.0, 0.0, 0
    all_severe_true, all_severe_logits, all_severe_prob = [], [], []
    all_rain_true, all_rain_pred = [], []

    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)
        losses = loss_fn(preds, batch)

        total_loss += losses["total"].item()
        total_severe += losses["severe_loss"].item()
        total_rain += losses["rain_loss"].item()
        n_batches += 1

        all_severe_true.append(batch["severe_weather"].cpu().numpy())
        all_severe_logits.append(preds["severe_weather_logit"].cpu().numpy())
        all_severe_prob.append(torch.sigmoid(preds["severe_weather_logit"]).cpu().numpy())
        all_rain_true.append(batch["rain_3h_mm"].cpu().numpy())
        all_rain_pred.append(preds["rain_3h_mm"].cpu().numpy())

    sev_true = np.concatenate(all_severe_true)       # (N, n_lead, H, W)
    sev_logits = np.concatenate(all_severe_logits)   # (N, n_lead, H, W)
    sev_prob = np.concatenate(all_severe_prob)       # (N, n_lead, H, W)
    rain_true = np.concatenate(all_rain_true)        # (N, n_lead, H, W)
    rain_pred = np.concatenate(all_rain_pred)        # (N, n_lead, H, W)

    results_per_lead = {}

    for i, lh in enumerate(lead_times):
        y_t = sev_true[:, i].ravel()
        y_p = sev_prob[:, i].ravel()
        r_t = rain_true[:, i].ravel()
        r_p = rain_pred[:, i].ravel()

        # Threshold to use
        if isinstance(thresholds, dict):
            thr = float(thresholds.get(lh, thresholds.get(int(lh), 0.5)))
        else:
            thr = float(thresholds)

        cls_m = classification_metrics(y_t, y_p, threshold=thr, fast=False)
        reg_m = regression_metrics(r_t, r_p)
        cal_m = compute_calibration_metrics(y_t, y_p, n_bins=10)

        results_per_lead[f"lead_{lh}h"] = {
            "threshold_used": thr,
            "csi": round(cls_m["csi"], 4),
            "pod": round(cls_m["recall_pod"], 4),
            "far": round(cls_m["far"], 4),
            "f1": round(cls_m["f1"], 4),
            "precision": round(cls_m["precision"], 4),
            "pr_auc": round(cls_m["pr_auc"], 4),
            "roc_auc": round(cls_m["roc_auc"], 4),
            "brier_score": cal_m["brier_score"],
            "brier_skill_score": cal_m["brier_skill_score"],
            "ece": cal_m["ece"],
            "rain_mae": round(reg_m["mae"], 3),
            "rain_rmse": round(reg_m["rmse"], 3),
            "calibration_bins": cal_m["bins"],
        }

    mean_metrics = {
        "mean_csi": round(float(np.mean([m["csi"] for m in results_per_lead.values()])), 4),
        "mean_pod": round(float(np.mean([m["pod"] for m in results_per_lead.values()])), 4),
        "mean_far": round(float(np.mean([m["far"] for m in results_per_lead.values()])), 4),
        "mean_f1": round(float(np.mean([m["f1"] for m in results_per_lead.values()])), 4),
        "mean_pr_auc": round(float(np.mean([m["pr_auc"] for m in results_per_lead.values()])), 4),
        "mean_roc_auc": round(float(np.mean([m["roc_auc"] for m in results_per_lead.values()])), 4),
        "mean_brier_score": round(float(np.mean([m["brier_score"] for m in results_per_lead.values()])), 6),
        "mean_ece": round(float(np.mean([m["ece"] for m in results_per_lead.values()])), 6),
        "mean_rain_mae": round(float(np.mean([m["rain_mae"] for m in results_per_lead.values()])), 3),
        "total_loss": round(total_loss / max(n_batches, 1), 4),
        "severe_loss": round(total_severe / max(n_batches, 1), 4),
        "rain_loss": round(total_rain / max(n_batches, 1), 4),
    }

    return {
        "mean_metrics": mean_metrics,
        "per_lead": results_per_lead,
        "raw_arrays": {
            "sev_true": sev_true,
            "sev_logits": sev_logits,
            "sev_prob": sev_prob,
            "rain_true": rain_true,
            "rain_pred": rain_pred,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="Data/outputs/checkpoints/v2_best_baseline.pt")
    parser.add_argument("--out", default="Data/outputs/metrics/v2_validation_baseline.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[val_eval] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    lead_times = cfg.get("sequence", "lead_times_hours")
    n_lead = len(lead_times)

    print(f"[val_eval] loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if ckpt["model"]["dem_proj.weight"].shape[1] == 4:
        from src.models.v2_plus_model import SevereWeatherNetV2Plus
        from src.data.dataset_v2_plus import make_dataloaders_v2_plus
        loaders = make_dataloaders_v2_plus(cache_path, cfg)
        model = SevereWeatherNetV2Plus(n_lead_times=n_lead)
        print("[val_eval] detected V2+ architecture")
    else:
        loaders = make_dataloaders_v2(cache_path, cfg)
        model = SevereWeatherNetV2(n_lead_times=n_lead)
        print("[val_eval] detected V2 baseline architecture")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    loss_fn = MultiTaskLossV2()

    thresholds_per_lead = ckpt.get("threshold_per_lead", {lh: 0.5 for lh in lead_times})
    global_thr = ckpt.get("optimal_threshold", 0.5)

    print("\n" + "="*75)
    print("  EVALUATING ON LOCKED 2023 VALIDATION SET (Per-Lead Optimal Thresholds)")
    print("="*75)
    res_per_lead = run_validation_evaluation(model, loaders["val"], loss_fn, lead_times, device, thresholds=thresholds_per_lead)

    print(f"  {'Lead':>8}  {'Thr':>6}  {'CSI':>6}  {'POD':>6}  {'FAR':>6}  {'PR-AUC':>7}  {'Brier':>8}  {'ECE':>7}  {'MAE':>6}")
    print("  " + "-"*73)
    for k, m in res_per_lead["per_lead"].items():
        print(f"  {k:>8}  {m['threshold_used']:>6.3f}  {m['csi']:>6.4f}  {m['pod']:>6.4f}  {m['far']:>6.4f}  "
              f"{m['pr_auc']:>7.4f}  {m['brier_score']:>8.5f}  {m['ece']:>7.4f}  {m['rain_mae']:>6.3f}")

    mm = res_per_lead["mean_metrics"]
    print("  " + "-"*73)
    print(f"  {'MEAN':>8}  {'--':>6}  {mm['mean_csi']:>6.4f}  {mm['mean_pod']:>6.4f}  {mm['mean_far']:>6.4f}  "
          f"{mm['mean_pr_auc']:>7.4f}  {mm['mean_brier_score']:>8.5f}  {mm['mean_ece']:>7.4f}  {mm['mean_rain_mae']:>6.3f}")

    print("\n" + "="*75)
    print(f"  EVALUATING ON LOCKED 2023 VALIDATION SET (Global Threshold = {global_thr:.3f})")
    print("="*75)
    res_global = run_validation_evaluation(model, loaders["val"], loss_fn, lead_times, device, thresholds=global_thr)

    print(f"  {'Lead':>8}  {'Thr':>6}  {'CSI':>6}  {'POD':>6}  {'FAR':>6}  {'PR-AUC':>7}  {'Brier':>8}  {'ECE':>7}")
    print("  " + "-"*65)
    for k, m in res_global["per_lead"].items():
        print(f"  {k:>8}  {m['threshold_used']:>6.3f}  {m['csi']:>6.4f}  {m['pod']:>6.4f}  {m['far']:>6.4f}  "
              f"{m['pr_auc']:>7.4f}  {m['brier_score']:>8.5f}  {m['ece']:>7.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", -1),
        "split": "validation_2023",
        "timestamp_eval": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_lead_threshold_results": {
            "mean_metrics": res_per_lead["mean_metrics"],
            "per_lead": res_per_lead["per_lead"],
        },
        "global_threshold_results": {
            "global_threshold": global_thr,
            "mean_metrics": res_global["mean_metrics"],
            "per_lead": res_global["per_lead"],
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[val_eval] validation baseline report saved -> {args.out}")


if __name__ == "__main__":
    main()


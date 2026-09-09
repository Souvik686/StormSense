"""Experiment 2: False-Alarm Reduction & Operational Threshold Analysis.

Evaluates Pareto frontiers between POD and FAR on the locked 2023 validation set.
Tests multiple operational objectives:
  1. Maximum CSI (standard baseline)
  2. Maximum F1 score
  3. Maximum F0.5 score (precision-weighted, heavy FAR penalty)
  4. FAR-constrained CSI optimization: max CSI s.t. FAR <= 0.50
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import load_config
from src.data.dataset_v2 import make_dataloaders_v2
from src.models.v2_model import SevereWeatherNetV2
from src.training.metrics import classification_metrics


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def collect_val_probs(model, loader, device, lead_times):
    model.eval()
    all_true, all_probs = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)
        all_true.append(batch["severe_weather"].cpu().numpy())
        all_probs.append(torch.sigmoid(preds["severe_weather_logit"]).cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_probs)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1.0 + b2) * (precision * recall) / denom if denom > 0 else 0.0


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp2_far] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    loaders = make_dataloaders_v2(cache_path, cfg)
    lead_times = cfg.get("sequence", "lead_times_hours")
    n_lead = len(lead_times)

    ckpt_path = "Data/outputs/checkpoints/v2_best_baseline.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SevereWeatherNetV2(n_lead_times=n_lead)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    print("[exp2_far] collecting validation probabilities ...")
    y_true, y_prob = collect_val_probs(model, loaders["val"], device, lead_times)

    thresholds = np.linspace(0.10, 0.90, 200)

    objectives = ["max_csi", "max_f1", "max_f05", "far_le_50"]
    objective_results = {obj: {} for obj in objectives}

    print("\n" + "="*85)
    print("  OPERATIONAL FALSE-ALARM REDUCTION OBJECTIVES (2023 Validation Set)")
    print("="*85)

    for li, lh in enumerate(lead_times):
        yt = y_true[:, li].ravel()
        yp = y_prob[:, li].ravel()

        sweep_records = []
        for thr in thresholds:
            m = classification_metrics(yt, yp, threshold=float(thr), fast=True)
            p, r, far, csi, f1 = m["precision"], m["recall_pod"], m["far"], m["csi"], m["f1"]
            f05 = compute_f_beta(p, r, beta=0.5)
            sweep_records.append({
                "threshold": float(thr),
                "precision": p,
                "recall_pod": r,
                "far": far,
                "csi": csi,
                "f1": f1,
                "f05": f05,
            })

        # 1. Max CSI
        best_csi_rec = max(sweep_records, key=lambda x: x["csi"])
        objective_results["max_csi"][f"lead_{lh}h"] = best_csi_rec

        # 2. Max F1
        best_f1_rec = max(sweep_records, key=lambda x: x["f1"])
        objective_results["max_f1"][f"lead_{lh}h"] = best_f1_rec

        # 3. Max F0.5 (heavy precision/low-FAR weight)
        best_f05_rec = max(sweep_records, key=lambda x: x["f05"])
        objective_results["max_f05"][f"lead_{lh}h"] = best_f05_rec

        # 4. FAR <= 0.50 constrained CSI
        valid_far = [x for x in sweep_records if x["far"] <= 0.50 and x["recall_pod"] > 0.05]
        best_far_rec = max(valid_far, key=lambda x: x["csi"]) if valid_far else best_f05_rec
        objective_results["far_le_50"][f"lead_{lh}h"] = best_far_rec

    print(f"\n{'Horizon':>8}  {'Objective':>14}  {'Threshold':>10}  {'CSI':>8}  {'POD (Recall)':>13}  {'FAR':>8}  {'Precision':>10}")
    print("-" * 80)
    for lh in lead_times:
        k = f"lead_{lh}h"
        for obj in objectives:
            rec = objective_results[obj][k]
            print(f"{k:>8}  {obj:>14}  {rec['threshold']:>10.3f}  {rec['csi']:>8.4f}  {rec['recall_pod']:>13.4f}  {rec['far']:>8.4f}  {rec['precision']:>10.4f}")
        print("-" * 80)

    # Compute mean summary per objective
    summary = {}
    for obj in objectives:
        mean_csi = np.mean([objective_results[obj][f"lead_{lh}h"]["csi"] for lh in lead_times])
        mean_pod = np.mean([objective_results[obj][f"lead_{lh}h"]["recall_pod"] for lh in lead_times])
        mean_far = np.mean([objective_results[obj][f"lead_{lh}h"]["far"] for lh in lead_times])
        mean_thr = np.mean([objective_results[obj][f"lead_{lh}h"]["threshold"] for lh in lead_times])
        summary[obj] = {
            "mean_threshold": round(float(mean_thr), 3),
            "mean_csi": round(float(mean_csi), 4),
            "mean_pod": round(float(mean_pod), 4),
            "mean_far": round(float(mean_far), 4),
        }

    print("\nSUMMARY ACROSS ALL LEAD TIMES (+2h to +6h):")
    print(f"{'Objective':>14}  {'Mean Thr':>10}  {'Mean CSI':>10}  {'Mean POD':>10}  {'Mean FAR':>10}")
    print("-" * 60)
    for obj, s in summary.items():
        print(f"{obj:>14}  {s['mean_threshold']:>10.3f}  {s['mean_csi']:>10.4f}  {s['mean_pod']:>10.4f}  {s['mean_far']:>10.4f}")

    out_file = "Data/outputs/metrics/exp2_false_alarm_analysis.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Exp2_False_Alarm_Analysis",
            "summary": summary,
            "objectives": objective_results,
        }, f, indent=2)
    print(f"\n[exp2_far] results saved -> {out_file}")


if __name__ == "__main__":
    main()


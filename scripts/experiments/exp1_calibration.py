"""Experiment 1: Post-processing Probability Calibration & Temperature Scaling.

Fits per-lead temperature scaling strictly on the 2023 validation set.
Measures Brier Score, Brier Skill Score, and ECE before and after calibration.
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
from src.training.calibration import compute_calibration_metrics, TemperatureScaler
from src.training.metrics import classification_metrics


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def collect_val_predictions(model, loader, device, lead_times):
    model.eval()
    all_true, all_logits = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)
        all_true.append(batch["severe_weather"].cpu().numpy())
        all_logits.append(preds["severe_weather_logit"].cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_logits)


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_calib] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    loaders = make_dataloaders_v2(cache_path, cfg)
    lead_times = cfg.get("sequence", "lead_times_hours")
    n_lead = len(lead_times)

    ckpt_path = "Data/outputs/checkpoints/v2_best_baseline.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SevereWeatherNetV2(n_lead_times=n_lead)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    print("[exp1_calib] collecting validation logits and targets ...")
    y_true, logits = collect_val_predictions(model, loaders["val"], device, lead_times)
    # y_true shape: (N, n_lead, H, W)
    # logits shape: (N, n_lead, H, W)

    scaler = TemperatureScaler(n_lead_times=n_lead)
    print("\n[exp1_calib] fitting temperature scaler on validation set ...")
    temps = scaler.fit(logits, y_true, max_iter=100, lr=0.05)
    print(f"[exp1_calib] learned temperatures per lead: {dict(zip(lead_times, temps))}")

    uncal_probs = 1.0 / (1.0 + np.exp(-logits))
    # Reshape temperatures for broadcasting: (1, n_lead, 1, 1)
    T_arr = np.array(temps, dtype=np.float32).reshape(1, n_lead, 1, 1)
    cal_probs = 1.0 / (1.0 + np.exp(-logits / T_arr))

    results = {}
    print("\n" + "="*80)
    print("  CALIBRATION METRICS COMPARISON (Locked 2023 Validation Split)")
    print("="*80)
    print(f"  {'Lead':>8}  {'Temp':>6}  {'Uncal Brier':>12}  {'Cal Brier':>10}  {'Uncal ECE':>10}  {'Cal ECE':>8}  {'Brier Red%':>10}")
    print("  " + "-"*78)

    brier_uncal_list, brier_cal_list = [], []
    ece_uncal_list, ece_cal_list = [], []

    thresholds = np.linspace(0.01, 0.90, 150)
    cal_thresholds_per_lead = {}

    for i, lh in enumerate(lead_times):
        yt = y_true[:, i].ravel()
        p_uncal = uncal_probs[:, i].ravel()
        p_cal = cal_probs[:, i].ravel()

        m_uncal = compute_calibration_metrics(yt, p_uncal)
        m_cal = compute_calibration_metrics(yt, p_cal)

        b_u = m_uncal["brier_score"]
        b_c = m_cal["brier_score"]
        e_u = m_uncal["ece"]
        e_c = m_cal["ece"]
        red_pct = (b_u - b_c) / b_u * 100.0

        brier_uncal_list.append(b_u)
        brier_cal_list.append(b_c)
        ece_uncal_list.append(e_u)
        ece_cal_list.append(e_c)

        # Optimize threshold on calibrated probabilities
        best_csi, best_thr, best_cls = -1.0, 0.5, {}
        for thr in thresholds:
            m = classification_metrics(yt, p_cal, threshold=float(thr), fast=True)
            if m["csi"] > best_csi:
                best_csi, best_thr, best_cls = m["csi"], float(thr), m
        cal_thresholds_per_lead[int(lh)] = round(best_thr, 3)

        print(f"  lead_{lh}h  {temps[i]:>6.3f}  {b_u:>12.6f}  {b_c:>10.6f}  {e_u:>10.4f}  {e_c:>8.4f}  {red_pct:>9.2f}%")

        results[f"lead_{lh}h"] = {
            "temperature": temps[i],
            "uncalibrated": {
                "brier_score": b_u,
                "brier_skill_score": m_uncal["brier_skill_score"],
                "ece": e_u,
                "bins": m_uncal["bins"],
            },
            "calibrated": {
                "brier_score": b_c,
                "brier_skill_score": m_cal["brier_skill_score"],
                "ece": e_c,
                "optimal_threshold": best_thr,
                "csi_at_optimal_thr": round(best_cls["csi"], 4),
                "pod_at_optimal_thr": round(best_cls["recall_pod"], 4),
                "far_at_optimal_thr": round(best_cls["far"], 4),
                "bins": m_cal["bins"],
            }
        }

    mean_b_u = float(np.mean(brier_uncal_list))
    mean_b_c = float(np.mean(brier_cal_list))
    mean_e_u = float(np.mean(ece_uncal_list))
    mean_e_c = float(np.mean(ece_cal_list))
    mean_red = (mean_b_u - mean_b_c) / mean_b_u * 100.0

    print("  " + "-"*78)
    print(f"  {'MEAN':>8}  {'--':>6}  {mean_b_u:>12.6f}  {mean_b_c:>10.6f}  {mean_e_u:>10.4f}  {mean_e_c:>8.4f}  {mean_red:>9.2f}%")

    out_file = "Data/outputs/metrics/exp1_calibration_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    summary = {
        "experiment": "Exp1_Temperature_Scaling",
        "checkpoint": ckpt_path,
        "split": "validation_2023",
        "learned_temperatures": {int(lh): temps[i] for i, lh in enumerate(lead_times)},
        "calibrated_thresholds_per_lead": cal_thresholds_per_lead,
        "mean_brier_uncalibrated": round(mean_b_u, 6),
        "mean_brier_calibrated": round(mean_b_c, 6),
        "mean_ece_uncalibrated": round(mean_e_u, 4),
        "mean_ece_calibrated": round(mean_e_c, 4),
        "brier_reduction_pct": round(mean_red, 2),
        "ece_reduction_pct": round((mean_e_u - mean_e_c) / mean_e_u * 100.0, 2),
        "details_per_lead": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[exp1_calib] results saved -> {out_file}")


if __name__ == "__main__":
    main()


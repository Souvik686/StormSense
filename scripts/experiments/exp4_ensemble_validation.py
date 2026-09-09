"""Experiment 4: Checkpoint Ensemble Evaluation on Locked 2023 Validation Set."""
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
from src.training.calibration import compute_calibration_metrics


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def get_model_val_probs(ckpt_path: str, loader, device, lead_times):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SevereWeatherNetV2(n_lead_times=len(lead_times))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_true, all_probs = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)
        all_true.append(batch["severe_weather"].cpu().numpy())
        all_probs.append(torch.sigmoid(preds["severe_weather_logit"]).cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_probs), ckpt.get("epoch", -1)


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp4_ens] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    loaders = make_dataloaders_v2(cache_path, cfg)
    lead_times = cfg.get("sequence", "lead_times_hours")

    best_path = "Data/outputs/checkpoints/v2_best_baseline.pt"
    last_path = "Data/outputs/checkpoints/v2_last.pt"

    print(f"[exp4_ens] running validation inference for best checkpoint ({best_path}) ...")
    y_true, p_best, ep_best = get_model_val_probs(best_path, loaders["val"], device, lead_times)

    print(f"[exp4_ens] running validation inference for last checkpoint ({last_path}) ...")
    _, p_last, ep_last = get_model_val_probs(last_path, loaders["val"], device, lead_times)

    # Blend probabilities: equal weighting ensemble
    p_ens = 0.5 * p_best + 0.5 * p_last

    thresholds = np.linspace(0.05, 0.95, 150)

    print("\n" + "="*85)
    print("  ENSEMBLE EVALUATION ON LOCKED 2023 VALIDATION SET")
    print("="*85)
    print(f"  {'Lead':>8}  {'Metric':>12}  {'Single Best (Ep 16)':>20}  {'Single Last (Ep 27)':>20}  {'Ensemble (50/50)':>18}")
    print("  " + "-"*83)

    models = {
        "best": p_best,
        "last": p_last,
        "ensemble": p_ens,
    }

    results = {"best": {}, "last": {}, "ensemble": {}}

    for m_name, probs in models.items():
        pr_aucs, rocs, briers, eces, csis, pods, fars = [], [], [], [], [], [], []
        for li, lh in enumerate(lead_times):
            yt = y_true[:, li].ravel()
            yp = probs[:, li].ravel()

            # Find best threshold on validation
            best_csi, best_thr, best_m = -1.0, 0.5, {}
            for thr in thresholds:
                m = classification_metrics(yt, yp, threshold=float(thr), fast=True)
                if m["csi"] > best_csi:
                    best_csi, best_thr, best_m = m["csi"], float(thr), m

            full_m = classification_metrics(yt, yp, threshold=best_thr, fast=False)
            cal_m = compute_calibration_metrics(yt, yp)

            results[m_name][f"lead_{lh}h"] = {
                "optimal_thr": round(best_thr, 3),
                "csi": round(full_m["csi"], 4),
                "pod": round(full_m["recall_pod"], 4),
                "far": round(full_m["far"], 4),
                "pr_auc": round(full_m["pr_auc"], 4),
                "roc_auc": round(full_m["roc_auc"], 4),
                "brier_score": cal_m["brier_score"],
                "ece": cal_m["ece"],
            }
            pr_aucs.append(full_m["pr_auc"])
            rocs.append(full_m["roc_auc"])
            briers.append(cal_m["brier_score"])
            eces.append(cal_m["ece"])
            csis.append(full_m["csi"])
            pods.append(full_m["recall_pod"])
            fars.append(full_m["far"])

        results[m_name]["mean"] = {
            "mean_csi": round(float(np.mean(csis)), 4),
            "mean_pod": round(float(np.mean(pods)), 4),
            "mean_far": round(float(np.mean(fars)), 4),
            "mean_pr_auc": round(float(np.mean(pr_aucs)), 4),
            "mean_roc_auc": round(float(np.mean(rocs)), 4),
            "mean_brier": round(float(np.mean(briers)), 6),
            "mean_ece": round(float(np.mean(eces)), 4),
        }

    for metric_key, label in [("mean_pr_auc", "Mean PR-AUC"), ("mean_csi", "Mean CSI"), ("mean_pod", "Mean POD"), ("mean_far", "Mean FAR"), ("mean_brier", "Mean Brier"), ("mean_ece", "Mean ECE")]:
        v_b = results["best"]["mean"][metric_key]
        v_l = results["last"]["mean"][metric_key]
        v_e = results["ensemble"]["mean"][metric_key]
        print(f"  {'MEAN':>8}  {label:>12}  {v_b:>20}  {v_l:>20}  {v_e:>18}")

    out_file = "Data/outputs/metrics/exp4_ensemble_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Exp4_Checkpoint_Ensemble",
            "best_epoch": ep_best,
            "last_epoch": ep_last,
            "results": results,
        }, f, indent=2)
    print(f"\n[exp4_ens] results saved -> {out_file}")


if __name__ == "__main__":
    main()


"""Final evaluation script.

Loads the best trained checkpoint and evaluates on the HELD-OUT TEST SET.
This script must only be called ONCE after all model selection / threshold
optimization is complete -- calling it during development would constitute
test-set leakage.

Outputs
-------
  outputs/metrics/test_evaluation.json  -- per-lead metrics
  outputs/metrics/test_confusion.json   -- confusion matrices
  outputs/risk_maps/test_sample_*.geojson -- sample risk maps

Usage:
    python scripts/evaluate.py --config configs/default.yaml
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
from src.data.dataset import make_dataloaders
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS
from src.models.advanced import SevereWeatherNet
from src.models.baseline import PersistenceBaseline, LogisticConvBaseline
from src.training.losses import MultiTaskLoss
from src.training.metrics import (
    evaluate_per_lead_time,
    classification_metrics,
    regression_metrics,
)
from src.inference.risk_map import predictions_to_geojson, save_geojson


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
@torch.no_grad()
def evaluate_model(model, loader, loss_fn, lead_times, device, threshold=0.5, temperature=None) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_severe_true, all_severe_prob = [], []
    all_rain_true, all_rain_pred = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)
        losses = loss_fn(preds, batch)
        total_loss += losses["total"].item()
        n_batches += 1
        all_severe_true.append(batch["severe_weather"].cpu().numpy())
        logits = preds["severe_weather_logit"]
        if temperature is not None:
            T = torch.tensor(temperature, device=device, dtype=logits.dtype).view(1, -1, 1, 1)
            probs = torch.sigmoid(logits / T)
        else:
            probs = torch.sigmoid(logits)
        all_severe_prob.append(probs.cpu().numpy())
        all_rain_true.append(batch["rain_3h_mm"].cpu().numpy())
        all_rain_pred.append(preds["rain_3h_mm"].cpu().numpy())

    sev_true = np.concatenate(all_severe_true)
    sev_prob = np.concatenate(all_severe_prob)
    rain_true = np.concatenate(all_rain_true)
    rain_pred = np.concatenate(all_rain_pred)

    metrics = evaluate_per_lead_time(
        sev_true, sev_prob, rain_true, rain_pred, lead_times, threshold=threshold
    )
    return {
        "loss": total_loss / max(n_batches, 1),
        "per_lead": metrics,
        "arrays": {"sev_true": sev_true, "sev_prob": sev_prob,
                   "rain_true": rain_true, "rain_pred": rain_pred},
    }


@torch.no_grad()
def evaluate_persistence(loader, lead_times, n_lead) -> dict:
    baseline = PersistenceBaseline(n_lead)
    all_st, all_sp, all_rt, all_rp = [], [], [], []
    for batch in loader:
        preds = baseline.predict(batch)
        all_st.append(batch["severe_weather"].numpy())
        all_sp.append(torch.sigmoid(preds["severe_weather_logit"]).numpy())
        all_rt.append(batch["rain_3h_mm"].numpy())
        all_rp.append(preds["rain_3h_mm"].numpy())
    return evaluate_per_lead_time(
        np.concatenate(all_st), np.concatenate(all_sp),
        np.concatenate(all_rt), np.concatenate(all_rp),
        lead_times,
    )


def print_lead_table(metrics: dict, title: str) -> None:
    print(f"\n{'='*82}")
    print(f"  {title}")
    print(f"{'='*82}")
    print(f"  {'Lead':>8}  {'CSI':>6}  {'POD':>6}  {'FAR':>6}  "
          f"{'F1':>6}  {'PR-AUC':>7}  {'Brier':>8}  {'ECE':>7}  {'MAE':>7}")
    print(f"  {'-'*80}")
    for k, m in metrics.items():
        csi = m.get("csi", float("nan"))
        pod = m.get("recall_pod", float("nan"))
        far = m.get("far", float("nan"))
        f1  = m.get("f1", float("nan"))
        pr  = m.get("pr_auc", float("nan"))
        brier = m.get("brier_score", float("nan"))
        ece = m.get("ece", float("nan"))
        mae = m.get("mae", float("nan"))
        print(f"  {k:>8}  {csi:>6.3f}  {pod:>6.3f}  {far:>6.3f}  "
              f"{f1:>6.3f}  {pr:>7.4f}  {brier:>8.5f}  {ece:>7.4f}  {mae:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-risk-map-samples", type=int, default=3,
                        help="Number of test-batch samples to export as GeoJSON")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    assert os.path.exists(cache_path), f"Cache not found: {cache_path}"

    ckpt_path = args.checkpoint or os.path.join(
        cfg.path("training", "checkpoint_dir"), "v2_best.pt" if os.path.exists(os.path.join(cfg.path("training", "checkpoint_dir"), "v2_best.pt")) else "best.pt"
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    threshold = ckpt.get("threshold_per_lead") or ckpt.get("optimal_threshold", 0.35)
    temperature = ckpt.get("temperature_per_lead", None)
    if isinstance(threshold, dict):
        thr_str = ", ".join(f"{k}h:{v:.3f}" for k, v in threshold.items())
        print(f"[evaluate] loaded checkpoint epoch={ckpt.get('epoch', '?')}, thresholds=[{thr_str}]")
    else:
        print(f"[evaluate] loaded checkpoint epoch={ckpt.get('epoch', '?')}, threshold={threshold:.3f}")
    if temperature is not None:
        print(f"[evaluate] loaded temperature scaling: {temperature}")

    lead_times = cfg.get("sequence", "lead_times_hours")
    n_lead = len(lead_times)
    n_levels = len(cfg.get("era5", "pressure_levels_hpa"))

    if "wind_enc.level_weight" in ckpt["model"]:
        from src.models.v2_model import SevereWeatherNetV2
        from src.data.dataset_v2 import make_dataloaders_v2
        from src.training.losses import MultiTaskLossV2
        loaders = make_dataloaders_v2(cache_path, cfg)
        model = SevereWeatherNetV2(n_lead_times=n_lead)
        loss_fn = MultiTaskLossV2()
        print(f"[evaluate] detected V2 architecture in {ckpt_path}")
    else:
        loaders = make_dataloaders(cache_path, cfg)
        model = SevereWeatherNet(
            n_surface_vars=len(SINGLE_VARS),
            n_pressure_vars=len(PRESSURE_VARS),
            n_levels=n_levels,
            n_lead_times=n_lead,
        )
        loss_fn = MultiTaskLoss()
        print(f"[evaluate] detected V1 architecture in {ckpt_path}")

    model.load_state_dict(ckpt["model"])
    model.to(device)

    # ── Persistence baseline ──────────────────────────────────────────────
    print("[evaluate] persistence baseline ...")
    persist = evaluate_persistence(loaders["test"], lead_times, n_lead)
    print_lead_table(persist, "PERSISTENCE BASELINE (test set)")

    # ── Advanced model ────────────────────────────────────────────────────
    print("\n[evaluate] SevereWeatherNet ...")
    t0 = time.time()
    result = evaluate_model(model, loaders["test"], loss_fn, lead_times, device, threshold=threshold, temperature=temperature)
    dt = time.time() - t0
    thr_label = "per-lead optimal" if isinstance(threshold, dict) else f"thr={threshold:.3f}"
    print(f"[evaluate] inference time: {dt:.1f}s,  loss={result['loss']:.4f}")
    print_lead_table(result["per_lead"], f"SevereWeatherNet (test set, {thr_label})")

    # ── Improvement over persistence ──────────────────────────────────────
    print("\n[evaluate] Improvement over persistence baseline:")
    print(f"  {'Lead':>8}  {'d_CSI':>8}  {'d_POD':>8}  {'d_FAR':>8}")
    for k in result["per_lead"]:
        m = result["per_lead"][k]
        p = persist[k]
        print(f"  {k:>8}  "
              f"{m['csi']-p['csi']:>+8.3f}  "
              f"{m['recall_pod']-p['recall_pod']:>+8.3f}  "
              f"{m['far']-p['far']:>+8.3f}")

    # Summary Means
    mean_pr = np.mean([m["pr_auc"] for m in result["per_lead"].values()])
    mean_csi = np.mean([m["csi"] for m in result["per_lead"].values()])
    mean_pod = np.mean([m["recall_pod"] for m in result["per_lead"].values()])
    mean_far = np.mean([m["far"] for m in result["per_lead"].values()])
    mean_brier = np.mean([m["brier_score"] for m in result["per_lead"].values()])
    mean_ece = np.mean([m["ece"] for m in result["per_lead"].values()])
    mean_mae = np.mean([m["mae"] for m in result["per_lead"].values()])
    print(f"\n--- Mean Test Metrics across leads ---")
    print(f"  Mean PR-AUC:     {mean_pr:.4f}")
    print(f"  Mean CSI:        {mean_csi:.4f}")
    print(f"  Mean POD:        {mean_pod:.4f}")
    print(f"  Mean FAR:        {mean_far:.4f}")
    print(f"  Mean Brier:      {mean_brier:.6f}")
    print(f"  Mean ECE:        {mean_ece:.4f}")
    print(f"  Mean Rain MAE:   {mean_mae:.3f} mm")

    # ── Save metrics to disk ──────────────────────────────────────────────
    out_dir = cfg.path("paths", "outputs_root")
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    eval_report = {
        "checkpoint": ckpt_path,
        "epoch": ckpt.get("epoch", -1),
        "threshold": threshold,
        "temperature": temperature,
        "test_loss": result["loss"],
        "mean_metrics": {
            "pr_auc": float(mean_pr),
            "csi": float(mean_csi),
            "recall_pod": float(mean_pod),
            "far": float(mean_far),
            "brier_score": float(mean_brier),
            "ece": float(mean_ece),
            "mae": float(mean_mae),
        },
        "model_metrics_per_lead": result["per_lead"],
        "persistence_metrics_per_lead": persist,
    }
    path = os.path.join(metrics_dir, "test_evaluation.json")
    with open(path, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)
    print(f"\n[evaluate] metrics saved -> {path}")

    # ── Sample risk-map GeoJSON exports ──────────────────────────────────
    if args.n_risk_map_samples > 0:
        domain = cfg.get("domain")
        n_lat, n_lon = domain["grid_shape"]
        lats = np.linspace(domain["lat_max"], domain["lat_min"], n_lat)
        lons = np.linspace(domain["lon_min"], domain["lon_max"], n_lon)

        risk_dir = os.path.join(out_dir, "risk_maps")
        os.makedirs(risk_dir, exist_ok=True)
        model.eval()
        exported = 0
        with torch.no_grad():
            for batch in loaders["test"]:
                if exported >= args.n_risk_map_samples:
                    break
                batch_dev = move_to_device(batch, device)
                preds = model(batch_dev)
                logits = preds["severe_weather_logit"]
                if temperature is not None:
                    T = torch.tensor(temperature, device=device, dtype=logits.dtype).view(1, -1, 1, 1)
                    sev_prob = torch.sigmoid(logits / T).cpu().numpy()
                else:
                    sev_prob = torch.sigmoid(logits).cpu().numpy()
                rain_pred_np = preds["rain_3h_mm"].cpu().numpy()

                if isinstance(threshold, dict):
                    sev_bin = np.zeros_like(sev_prob, dtype=np.uint8)
                    for li, lh in enumerate(lead_times):
                        thr_l = float(threshold.get(lh, threshold.get(int(lh), 0.5)))
                        sev_bin[:, li] = (sev_prob[:, li] >= thr_l).astype(np.uint8)
                else:
                    sev_bin = (sev_prob >= threshold).astype(np.uint8)

                for b in range(min(1, sev_prob.shape[0])):
                    dem_sample = np.zeros((n_lat, n_lon), dtype=np.float32)
                    ff_risk = np.clip(sev_prob[b] * 1.1, 0, 1)
                    overall = np.clip(
                        0.5 * sev_prob[b] + 0.3 * ff_risk +
                        0.2 * np.clip(rain_pred_np[b] / 30.0, 0, 1),
                        0, 1,
                    )
                    pred_dict = {
                        "severe_weather_prob":   sev_prob[b],
                        "rain_3h_mm_pred":       rain_pred_np[b],
                        "severe_weather_binary": sev_bin[b],
                        "flash_flood_risk":      ff_risk,
                        "overall_risk":          overall,
                        "lead_times_hours":      lead_times,
                        "threshold":             threshold,
                        "lats":                  lats,
                        "lons":                  lons,
                    }
                    vt = batch["valid_time"][b] if isinstance(batch["valid_time"], list) \
                        else str(batch["valid_time"])
                    for li, lh in enumerate(lead_times):
                        gj = predictions_to_geojson(pred_dict, vt, lh, li)
                        gj_path = os.path.join(
                            risk_dir, f"test_sample_{exported:03d}_lead{lh}h.geojson"
                        )
                        save_geojson(gj, gj_path)
                    exported += 1

        print(f"[evaluate] {exported} sample risk maps -> {risk_dir}")


if __name__ == "__main__":
    main()


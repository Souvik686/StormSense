"""Training entry point for SevereWeatherNet V2."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data.dataset_v2 import make_dataloaders_v2
from src.models.v2_model import SevereWeatherNetV2
from src.models.baseline import PersistenceBaseline
from src.training.losses import MultiTaskLossV2
from src.training.metrics import evaluate_per_lead_time


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def run_eval(model, loader, loss_fn, lead_times, device) -> dict:
    model.eval()
    total_loss, total_severe, total_rain, n_batches = 0.0, 0.0, 0.0, 0
    all_severe_true, all_severe_prob = [], []
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
        all_severe_prob.append(torch.sigmoid(preds["severe_weather_logit"]).cpu().numpy())
        all_rain_true.append(batch["rain_3h_mm"].cpu().numpy())
        all_rain_pred.append(preds["rain_3h_mm"].cpu().numpy())

    metrics = evaluate_per_lead_time(
        np.concatenate(all_severe_true), np.concatenate(all_severe_prob),
        np.concatenate(all_rain_true), np.concatenate(all_rain_pred),
        lead_times,
    )
    mean_csi = float(np.nanmean([m["csi"] for m in metrics.values()]))
    mean_pr_auc = float(np.nanmean([m["pr_auc"] for m in metrics.values()]))
    return {
        "loss": total_loss / max(n_batches, 1),
        "severe_loss": total_severe / max(n_batches, 1),
        "rain_loss": total_rain / max(n_batches, 1),
        "mean_csi": mean_csi,
        "mean_pr_auc": mean_pr_auc,
        "per_lead": metrics
    }


@torch.no_grad()
def run_persistence_eval(loader, lead_times, n_lead) -> dict:
    baseline = PersistenceBaseline(n_lead)
    all_severe_true, all_severe_prob = [], []
    all_rain_true, all_rain_pred = [], []
    for batch in loader:
        preds = baseline.predict(batch)
        all_severe_true.append(batch["severe_weather"].numpy())
        all_severe_prob.append(torch.sigmoid(preds["severe_weather_logit"]).numpy())
        all_rain_true.append(batch["rain_3h_mm"].numpy())
        all_rain_pred.append(preds["rain_3h_mm"].numpy())
    return evaluate_per_lead_time(
        np.concatenate(all_severe_true), np.concatenate(all_severe_prob),
        np.concatenate(all_rain_true), np.concatenate(all_rain_pred),
        lead_times,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--resume-from", default=None, help="checkpoint path to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 batches smoke test")
    parser.add_argument("--eval-test", action="store_true", help="Run test evaluation after training")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    set_seed(cfg.get("project", "seed", default=42))

    is_smoke = args.smoke_test or os.environ.get("SMOKE_TEST") == "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    
    print("[train] building V2 dataloaders ...")
    loaders = make_dataloaders_v2(cache_path, cfg)
    for split, dl in loaders.items():
        print(f"  {split}: {len(dl.dataset)} samples, {len(dl)} batches")

    lead_times = cfg.get("sequence", "lead_times_hours")
    n_lead = len(lead_times)

    model = SevereWeatherNetV2(n_lead_times=n_lead).to(device)
    print(f"[train] V2 model params: {model.count_parameters():,}")

    loss_fn = MultiTaskLossV2()

    train_cfg = cfg.get("training")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    epochs = train_cfg["epochs"] if not is_smoke else 1
    warmup_epochs = train_cfg["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"] and device.type == "cuda")

    ckpt_dir = cfg.path("training", "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "v2_best.pt")
    last_path = os.path.join(ckpt_dir, "v2_last.pt")

    start_epoch = 0
    best_val_score = -float("inf")
    patience_ctr = 0

    resume_path = args.resume_from or (last_path if train_cfg.get("resume") and os.path.exists(last_path) else None)
    if resume_path:
        print(f"[train] resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_score = ckpt.get("best_val_score", -float("inf"))
        patience_ctr = ckpt.get("patience_ctr", 0)

    print("[train] persistence baseline (sanity check) ...")
    persist_metrics = run_persistence_eval(loaders["val"], lead_times, n_lead)
    for lh, m in persist_metrics.items():
        print(f"  {lh}: csi={m['csi']:.3f} pod={m['recall_pod']:.3f} far={m['far']:.3f} mae={m['mae']:.3f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running_loss, running_sev, running_rain = 0.0, 0.0, 0.0
        for i, batch in enumerate(loaders["train"]):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                preds = model(batch)
                losses = loss_fn(preds, batch)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += losses["total"].item()
            running_sev += losses["severe_loss"].item()
            running_rain += losses["rain_loss"].item()
            
            if is_smoke and i >= 1:
                break

        scheduler.step()
        n_train = 2 if is_smoke else len(loaders["train"])
        train_loss = running_loss / n_train

        val_result = run_eval(model, loaders["val"], loss_fn, lead_times, device)
        val_loss = val_result["loss"]
        val_csi = val_result["mean_csi"]
        val_pr_auc = val_result["mean_pr_auc"]
        val_score = val_pr_auc  # primary ranking metric for rare-event severe weather
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_pr_auc={val_pr_auc:.4f} val_csi={val_csi:.4f} "
              f"sev_loss={val_result['severe_loss']:.4f} rain_loss={val_result['rain_loss']:.4f} "
              f"lr={lr:.2e} time={dt:.1f}s")
        for lh, m in val_result["per_lead"].items():
            print(f"    {lh}: csi={m['csi']:.3f} pod={m['recall_pod']:.3f} far={m['far']:.3f} "
                  f"pr_auc={m['pr_auc']:.3f} mae={m['mae']:.3f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_score": best_val_score,
            "best_val_pr_auc": val_pr_auc,
            "best_val_csi": val_csi,
            "patience_ctr": patience_ctr,
            "config": cfg.raw(),
        }
        torch.save(ckpt, last_path)

        if val_score > best_val_score:
            best_val_score = val_score
            patience_ctr = 0
            ckpt["best_val_score"] = best_val_score
            torch.save(ckpt, best_path)
            print(f"    -> new best (mean_pr_auc={best_val_score:.4f}), saved {best_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= train_cfg["early_stopping_patience"]:
                print(f"[train] early stopping at epoch {epoch+1} (patience={patience_ctr})")
                break
                
        if is_smoke:
            print("[smoke-test] Smoke test finished successfully!")
            break

    if args.eval_test:
        print("[train] loading best checkpoint for final test evaluation ...")
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        test_result = run_eval(model, loaders["test"], loss_fn, lead_times, device)
        print(f"[test] loss={test_result['loss']:.4f} pr_auc={test_result['mean_pr_auc']:.4f} csi={test_result['mean_csi']:.4f}")
    else:
        print(f"[train] Model training complete! Best checkpoint saved to {best_path}.")
        print("[train] Test set locked. Next step: run threshold optimization on validation set.")


if __name__ == "__main__":
    main()

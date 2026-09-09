"""Training script for SevereWeatherNet V2+ (Physics-Enhanced).

Trains exclusively on 2021-2022 and validates exclusively on 2023.
Locked test set is NOT touched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import load_config
from src.data.dataset_v2_plus import make_dataloaders_v2_plus
from src.models.v2_plus_model import SevereWeatherNetV2Plus
from src.training.losses import MultiTaskLossV2
from src.training.metrics import evaluate_per_lead_time


def set_seed(seed: int = 42):
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
        "per_lead": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    set_seed(cfg.get("project", "seed", default=42))

    is_smoke = args.smoke_test or os.environ.get("SMOKE_TEST") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_v2_plus] device = {device}")

    cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    loaders = make_dataloaders_v2_plus(cache_path, cfg)
    lead_times = cfg.get("sequence", "lead_times_hours")
    n_lead = len(lead_times)

    model = SevereWeatherNetV2Plus(n_lead_times=n_lead).to(device)
    print(f"[train_v2_plus] model params: {model.count_parameters():,}")

    loss_fn = MultiTaskLossV2()
    train_cfg = cfg.get("training")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    epochs = 1 if is_smoke else args.epochs
    warmup_epochs = train_cfg["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg["mixed_precision"] and device.type == "cuda")

    ckpt_dir = cfg.path("training", "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "v2_plus_best.pt")
    last_path = os.path.join(ckpt_dir, "v2_plus_last.pt")

    best_val_score = -float("inf")
    patience_ctr = 0

    print("[train_v2_plus] starting training on 2021-2022 (val on 2023) ...")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0

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
            if is_smoke and i >= 1:
                break

        scheduler.step()
        n_train = 2 if is_smoke else len(loaders["train"])
        train_loss = running_loss / n_train

        val_result = run_eval(model, loaders["val"], loss_fn, lead_times, device)
        val_score = val_result["mean_pr_auc"]
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} val_loss={val_result['loss']:.4f} "
              f"val_pr_auc={val_score:.4f} val_csi={val_result['mean_csi']:.4f} "
              f"lr={lr:.2e} time={dt:.1f}s")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_score": best_val_score,
            "best_val_pr_auc": val_score,
            "best_val_csi": val_result["mean_csi"],
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
                print(f"[train_v2_plus] early stopping at epoch {epoch+1} (patience={patience_ctr})")
                break

        if is_smoke:
            print("[smoke-test] smoke test passed!")
            break

    print(f"\n[train_v2_plus] Training complete. Best checkpoint: {best_path} with val_pr_auc={best_val_score:.4f}")
    print("[train_v2_plus] 2024 test split remains completely locked.")


if __name__ == "__main__":
    main()


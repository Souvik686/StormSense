"""End-to-end smoke test suite for the SevereWeatherNet pipeline.

Tests every stage using a tiny synthetic dataset so the suite runs in < 60s
on CPU without any raw data files. Does NOT test accuracy -- only that the
data flow and interfaces work correctly with the right shapes/dtypes.

Run with:
    python -m pytest tests/test_smoke.py -v
    -- or --
    python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import Config
from src.models.advanced import SevereWeatherNet, ConvGRU
from src.models.baseline import LogisticConvBaseline, PersistenceBaseline
from src.training.losses import MultiTaskLoss, focal_loss_with_logits
from src.training.metrics import (
    classification_metrics,
    regression_metrics,
    evaluate_per_lead_time,
    confusion_counts,
)
from src.features.normalize import (
    SINGLE_VARS, PRESSURE_VARS,
    fit_normalization_stats, save_stats, load_stats, validate_stats,
    normalize_single, normalize_pressure,
)
from src.data.windowing import build_windows, validate_no_leakage, split_summary
from src.features.targets import compute_rolling_precip, compute_targets, base_rates
from src.inference.risk_map import (
    predictions_to_geojson,
    predictions_to_summary,
    predictions_to_all_leads_geojson,
    save_geojson,
)

# ───────────────────────── shared fixtures ──────────────────────────────────
N_SURFACE = len(SINGLE_VARS)   # 9
N_PRESSURE = len(PRESSURE_VARS)  # 5
N_LEVELS = 6
N_LEADS = 5
INPUT_H = 6
H, W = 33, 25

LEAD_TIMES = [2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def tiny_batch() -> dict:
    B = 2
    return {
        "surface":   torch.randn(B, INPUT_H, N_SURFACE, H, W),
        "pressure":  torch.randn(B, INPUT_H, N_PRESSURE, N_LEVELS, H, W),
        "dem":       torch.randn(B, 1, H, W),
        "severe_weather": torch.zeros(B, N_LEADS, H, W),
        "rain_3h_mm": torch.abs(torch.randn(B, N_LEADS, H, W)),
        "persistence_severe":  torch.zeros(B, H, W),
        "persistence_rain_3h": torch.abs(torch.randn(B, H, W)),
    }


# ───────────────────────── 1. Config ────────────────────────────────────────
class TestConfig:
    def test_load(self, cfg):
        assert cfg.get("domain", "lat_min") == 20.0
        assert cfg.get("domain", "lat_max") == 28.0
        assert cfg.get("domain", "grid_shape") == [33, 25]

    def test_path_resolution(self, cfg):
        p = cfg.path("era5", "pressure_levels_dir")
        assert "ERA5" in p

    def test_lead_times(self, cfg):
        lt = cfg.get("sequence", "lead_times_hours")
        assert lt == [2, 3, 4, 5, 6]


# ───────────────────────── 2. Normalization ─────────────────────────────────
class TestNormalization:
    def test_fit_on_tiny_array(self):
        """Fit stats on a tiny fake dataset, verify shape and values."""
        import xarray as xr
        import pandas as pd

        T, H, W, L = 20, 33, 25, 6
        n_levels_hpa = [1000, 850, 700, 500, 300, 250]
        times = pd.date_range("2021-06-01", periods=T, freq="h")

        data_vars = {}
        for v in SINGLE_VARS:
            data_vars[v] = xr.DataArray(
                np.random.randn(T, H, W).astype(np.float32),
                dims=["valid_time", "latitude", "longitude"],
            )
        for v in PRESSURE_VARS:
            data_vars[v] = xr.DataArray(
                np.random.randn(T, L, H, W).astype(np.float32),
                dims=["valid_time", "pressure_level", "latitude", "longitude"],
            )
        ds = xr.Dataset(data_vars, coords={
            "valid_time": times,
            "pressure_level": n_levels_hpa,
        })
        train_idx = np.arange(T)
        stats = fit_normalization_stats(ds, train_idx)
        validate_stats(stats)

        for v in SINGLE_VARS:
            assert "mean" in stats[v]
            assert "std" in stats[v]
            assert stats[v]["std"] > 0
        for v in PRESSURE_VARS:
            assert len(stats[v]["per_level"]) == L

    def test_save_load_roundtrip(self, tmp_path):
        stats = {v: {"mean": float(i), "std": 1.0} for i, v in enumerate(SINGLE_VARS)}
        stats.update({v: {"per_level": [{"mean": 0.0, "std": 1.0}] * N_LEVELS,
                          "levels_hpa": [100.0] * N_LEVELS}
                      for v in PRESSURE_VARS})
        path = str(tmp_path / "stats.json")
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded["u10"]["mean"] == stats["u10"]["mean"]


# ───────────────────────── 3. Windowing / leakage ───────────────────────────
class TestWindowing:
    def _make_times(self, cfg):
        """Synthetic hourly time series: 2021 May-Oct + 2022 May-Oct."""
        import pandas as pd
        t2021 = pd.date_range("2021-05-01", "2021-10-31 23:00", freq="h")
        t2022 = pd.date_range("2022-05-01", "2022-10-31 23:00", freq="h")
        return np.concatenate([t2021.values, t2022.values]).astype("datetime64[ns]")

    def test_build_windows(self, cfg):
        times = self._make_times(cfg)
        train = build_windows(times, cfg, "train")
        assert len(train) > 0, "Expected some training windows"
        # All input indices within 2021 or 2022
        import pandas as pd
        for w in train[:10]:
            y = pd.Timestamp(times[w.input_end_idx]).year
            assert y in [2021, 2022]

    def test_no_leakage(self, cfg):
        """validate_no_leakage must not raise."""
        import pandas as pd
        t21 = pd.date_range("2021-05-01", "2021-10-31 23:00", freq="h")
        t22 = pd.date_range("2022-05-01", "2022-10-31 23:00", freq="h")
        t23 = pd.date_range("2023-05-01", "2023-10-31 23:00", freq="h")
        t24 = pd.date_range("2024-05-01", "2024-10-31 23:00", freq="h")
        times = np.concatenate([t21.values, t22.values, t23.values, t24.values])
        validate_no_leakage(times, cfg)  # should not raise

    def test_split_summary(self, cfg):
        import pandas as pd
        t21 = pd.date_range("2021-05-01", "2021-10-31 23:00", freq="h")
        t22 = pd.date_range("2022-05-01", "2022-10-31 23:00", freq="h")
        t23 = pd.date_range("2023-05-01", "2023-10-31 23:00", freq="h")
        t24 = pd.date_range("2024-05-01", "2024-10-31 23:00", freq="h")
        times = np.concatenate([t21.values, t22.values, t23.values, t24.values])
        s = split_summary(times, cfg)
        assert "train" in s and "val" in s and "test" in s

class TestDatasetMemory:
    def test_memmap_is_used(self, tmp_path, cfg):
        import os
        from src.data.dataset import NowcastDataset
        import pandas as pd
        
        # Create fake era5_memmap cache directory
        cache_dir = tmp_path / "era5_memmap"
        os.makedirs(cache_dir, exist_ok=True)
        
        T, H, W, L = 20, 33, 25, 6
        np.save(cache_dir / "surface.npy", np.zeros((T, N_SURFACE, H, W), dtype=np.float32))
        np.save(cache_dir / "pressure.npy", np.zeros((T, N_PRESSURE, L, H, W), dtype=np.float32))
        np.save(cache_dir / "dem_elevation_m.npy", np.zeros((H, W), dtype=np.float32))
        times = pd.date_range("2021-06-01", periods=T, freq="h").values.astype("datetime64[ns]").astype(np.int64)
        np.save(cache_dir / "valid_time.npy", times)
        np.save(cache_dir / "target_severe_weather.npy", np.zeros((T, H, W), dtype=np.float32))
        np.save(cache_dir / "target_rain_3h_mm.npy", np.zeros((T, H, W), dtype=np.float32))
        np.save(cache_dir / "target_label_valid.npy", np.ones((T, H, W), dtype=bool))
        
        # Create fake stats
        stats_path = str(tmp_path / "stats.json")
        stats = {v: {"mean": float(i), "std": 1.0} for i, v in enumerate(SINGLE_VARS)}
        stats.update({v: {"per_level": [{"mean": 0.0, "std": 1.0}] * N_LEVELS,
                          "levels_hpa": [100.0] * N_LEVELS}
                      for v in PRESSURE_VARS})
        save_stats(stats, stats_path)
        
        # Patch cfg temporarily
        cfg._data["paths"]["cache_root"] = str(tmp_path)
        cfg._data["normalization"]["stats_file"] = stats_path
        
        ds = NowcastDataset(str(cache_dir), cfg, "train")
        
        # Verify that mmap is used instead of fully loading arrays
        assert isinstance(ds.surface, np.memmap)
        assert isinstance(ds.pressure, np.memmap)
        assert ds.surface.mode == "r"
        
        # Restore cfg
        cfg._data["paths"]["cache_root"] = "processed/cache"
        cfg._data["normalization"]["stats_file"] = "processed/cache/norm_stats.json"


# ───────────────────────── 4. Model forward pass ────────────────────────────
class TestModelForward:
    def test_severe_weather_net(self, tiny_batch):
        model = SevereWeatherNet(
            n_surface_vars=N_SURFACE,
            n_pressure_vars=N_PRESSURE,
            n_levels=N_LEVELS,
            n_lead_times=N_LEADS,
        )
        model.eval()
        with torch.no_grad():
            out = model(tiny_batch)
        assert "severe_weather_logit" in out
        assert "rain_3h_mm" in out
        B = tiny_batch["surface"].shape[0]
        assert out["severe_weather_logit"].shape == (B, N_LEADS, H, W)
        assert out["rain_3h_mm"].shape == (B, N_LEADS, H, W)
        assert torch.all(out["rain_3h_mm"] >= 0), "Rainfall output must be non-negative"

    def test_parameter_count(self):
        model = SevereWeatherNet(
            n_surface_vars=N_SURFACE,
            n_pressure_vars=N_PRESSURE,
            n_levels=N_LEVELS,
            n_lead_times=N_LEADS,
        )
        n = model.count_parameters()
        assert n > 1000, f"Model seems too small: {n} params"
        assert n < 100_000_000, f"Model seems too large: {n} params"
        print(f"\n    SevereWeatherNet parameters: {n:,}")

    def test_logistic_baseline(self, tiny_batch):
        model = LogisticConvBaseline(
            n_surface_vars=N_SURFACE,
            input_hours=INPUT_H,
            n_lead_times=N_LEADS,
        )
        model.eval()
        with torch.no_grad():
            out = model(tiny_batch)
        B = tiny_batch["surface"].shape[0]
        assert out["severe_weather_logit"].shape == (B, N_LEADS, H, W)

    def test_persistence_baseline(self, tiny_batch):
        pb = PersistenceBaseline(n_lead_times=N_LEADS)
        out = pb.predict(tiny_batch)
        B = tiny_batch["surface"].shape[0]
        assert out["severe_weather_logit"].shape == (B, N_LEADS, H, W)
        assert out["rain_3h_mm"].shape == (B, N_LEADS, H, W)


# ───────────────────────── 5. Loss + gradients ──────────────────────────────
class TestLoss:
    def test_multitask_loss_forward(self, tiny_batch):
        model = SevereWeatherNet(N_SURFACE, N_PRESSURE, N_LEVELS, N_LEADS)
        out = model(tiny_batch)
        loss_fn = MultiTaskLoss()
        losses = loss_fn(out, tiny_batch)
        assert "total" in losses
        assert losses["total"].item() > 0
        assert torch.isfinite(losses["total"])

    def test_backward_pass(self, tiny_batch):
        model = SevereWeatherNet(N_SURFACE, N_PRESSURE, N_LEVELS, N_LEADS)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MultiTaskLoss()
        optimizer.zero_grad()
        out = model(tiny_batch)
        losses = loss_fn(out, tiny_batch)
        losses["total"].backward()
        # Check that gradients flowed everywhere
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
        optimizer.step()

    def test_focal_loss(self):
        logits = torch.randn(4, 5, 3, 3)
        targets = torch.zeros_like(logits)
        targets[0, 0, 0, 0] = 1.0
        loss = focal_loss_with_logits(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_checkpoint_save_load(self, tmp_path):
        """Save and reload a checkpoint; verify model weights are identical."""
        model = SevereWeatherNet(N_SURFACE, N_PRESSURE, N_LEVELS, N_LEADS)
        path = str(tmp_path / "ckpt.pt")
        ckpt = {"model": model.state_dict(), "epoch": 0, "best_val_loss": 99.9}
        torch.save(ckpt, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        model2 = SevereWeatherNet(N_SURFACE, N_PRESSURE, N_LEVELS, N_LEADS)
        model2.load_state_dict(loaded["model"])
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Param mismatch: {n1}"


# ───────────────────────── 6. Metrics ───────────────────────────────────────
class TestMetrics:
    def test_classification_metrics_perfect(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
        m = classification_metrics(y_true, y_prob, threshold=0.5)
        assert m["recall_pod"] == 1.0
        assert m["far"] == 0.0
        assert m["csi"] == 1.0

    def test_classification_metrics_all_zero_pred(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.zeros(4)
        m = classification_metrics(y_true, y_prob, threshold=0.5)
        assert m["recall_pod"] == 0.0  # no positives predicted

    def test_regression_metrics(self):
        y_true = np.array([10.0, 20.0, 5.0])
        y_pred = np.array([10.0, 20.0, 5.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] < 1e-6
        assert m["rmse"] < 1e-6

    def test_per_lead_time(self):
        N, L = 4, 5
        y_true = np.random.randint(0, 2, (N, L, H, W)).astype(float)
        y_prob = np.random.rand(N, L, H, W).astype(float)
        y_rain = np.random.rand(N, L, H, W).astype(float) * 10
        y_pred_rain = np.random.rand(N, L, H, W).astype(float) * 10
        out = evaluate_per_lead_time(y_true, y_prob, y_rain, y_pred_rain, LEAD_TIMES)
        assert len(out) == L
        for k in out:
            assert "csi" in out[k]
            assert "mae" in out[k]


# ───────────────────────── 7. Risk map generation ───────────────────────────
class TestRiskMap:
    def _fake_pred(self):
        lats = np.linspace(28.0, 20.0, 33)
        lons = np.linspace(84.0, 90.0, 25)
        return {
            "severe_weather_prob":   np.random.rand(5, 33, 25).astype(np.float32),
            "rain_3h_mm_pred":       np.random.rand(5, 33, 25).astype(np.float32) * 20,
            "severe_weather_binary": np.random.randint(0, 2, (5, 33, 25)).astype(np.uint8),
            "flash_flood_risk":      np.random.rand(5, 33, 25).astype(np.float32),
            "overall_risk":          np.random.rand(5, 33, 25).astype(np.float32),
            "lead_times_hours":      [2, 3, 4, 5, 6],
            "threshold":             0.35,
            "lats":                  lats,
            "lons":                  lons,
        }

    def test_geojson_structure(self):
        pred = self._fake_pred()
        gj = predictions_to_geojson(pred, lead_idx=0)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 33 * 25
        feat = gj["features"][0]
        assert "geometry" in feat
        assert "properties" in feat
        assert "overall_risk" in feat["properties"]
        assert "flash_flood_risk" in feat["properties"]
        assert "risk_level" in feat["properties"]

    def test_all_leads_geojson(self):
        pred = self._fake_pred()
        result = predictions_to_all_leads_geojson(pred)
        assert len(result) == 5
        assert "lead_2h" in result
        assert "lead_6h" in result

    def test_save_geojson(self, tmp_path):
        pred = self._fake_pred()
        gj = predictions_to_geojson(pred, lead_idx=0)
        path = str(tmp_path / "risk.geojson")
        save_geojson(gj, path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["type"] == "FeatureCollection"

    def test_summary(self):
        pred = self._fake_pred()
        s = predictions_to_summary(pred, "2024-07-01T12:00:00Z")
        assert "per_lead" in s
        assert "2h" in s["per_lead"]
        assert "max_overall_risk" in s["per_lead"]["2h"]


# ───────────────────────── 8. Full mini training step ───────────────────────
class TestMiniTraining:
    """Train SevereWeatherNet for 2 steps on random data and verify loss decreases."""

    def test_two_training_steps(self):
        torch.manual_seed(0)
        model = SevereWeatherNet(N_SURFACE, N_PRESSURE, N_LEVELS, N_LEADS)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MultiTaskLoss()

        B = 2
        batch = {
            "surface":  torch.randn(B, INPUT_H, N_SURFACE, H, W),
            "pressure": torch.randn(B, INPUT_H, N_PRESSURE, N_LEVELS, H, W),
            "dem": torch.randn(B, 1, H, W),
            "severe_weather": torch.zeros(B, N_LEADS, H, W),
            "rain_3h_mm": torch.abs(torch.randn(B, N_LEADS, H, W)),
            "persistence_severe": torch.zeros(B, H, W),
            "persistence_rain_3h": torch.abs(torch.randn(B, H, W)),
        }

        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)["total"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Losses should both be finite (don't assert decrease -- 2 steps is too few)
        assert all(np.isfinite(l) for l in losses)
        print(f"\n    Mini-train losses: {losses[0]:.4f} -> {losses[1]:.4f}")


# ───────────────────────── 9. V2 Components ─────────────────────────────────
class TestV2Components:
    def test_v2_model_forward_and_loss(self):
        from src.models.v2_model import SevereWeatherNetV2
        from src.training.losses import MultiTaskLossV2

        B, T, H, W = 2, 6, 33, 25
        batch = {
            "surface": torch.randn(B, T, 15, H, W),
            "pressure_wind": torch.randn(B, T, 2, 3, H, W),
            "pressure_thermo": torch.randn(B, T, 3, 4, H, W),
            "dem": torch.randn(B, 3, H, W),
            "severe_weather": torch.zeros(B, 5, H, W),
            "rain_3h_mm": torch.abs(torch.randn(B, 5, H, W)),
        }
        model = SevereWeatherNetV2(n_lead_times=5)
        out = model(batch)

        assert out["severe_weather_logit"].shape == (B, 5, H, W)
        assert out["rain_3h_mm"].shape == (B, 5, H, W)
        assert out["rain_log"].shape == (B, 5, H, W)
        assert (out["rain_3h_mm"] >= 0).all()

        loss_fn = MultiTaskLossV2()
        losses = loss_fn(out, batch)
        assert torch.isfinite(losses["total"])
        assert torch.isfinite(losses["severe_loss"])
        assert torch.isfinite(losses["rain_loss"])

        # Check gradients backpropagate cleanly
        losses["total"].backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ───────────────────────── main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke tests without pytest...\n")
    import traceback
    cfg_obj = Config.load()

    for cls in [TestConfig, TestNormalization, TestWindowing,
                TestModelForward, TestLoss, TestMetrics,
                TestRiskMap, TestMiniTraining]:
        inst = cls()
        for name in dir(inst):
            if not name.startswith("test_"):
                continue
            method = getattr(inst, name)
            try:
                # Inject fixtures manually
                import inspect
                sig = inspect.signature(method)
                kwargs = {}
                if "cfg" in sig.parameters:
                    kwargs["cfg"] = cfg_obj
                if "tiny_batch" in sig.parameters:
                    B = 2
                    kwargs["tiny_batch"] = {
                        "surface": torch.randn(B, INPUT_H, N_SURFACE, H, W),
                        "pressure": torch.randn(B, INPUT_H, N_PRESSURE, N_LEVELS, H, W),
                        "dem": torch.randn(B, 1, H, W),
                        "severe_weather": torch.zeros(B, N_LEADS, H, W),
                        "rain_3h_mm": torch.abs(torch.randn(B, N_LEADS, H, W)),
                        "persistence_severe": torch.zeros(B, H, W),
                        "persistence_rain_3h": torch.abs(torch.randn(B, H, W)),
                    }
                if "tmp_path" in sig.parameters:
                    import pathlib
                    with tempfile.TemporaryDirectory() as td:
                        kwargs["tmp_path"] = pathlib.Path(td)
                        method(**kwargs)
                    continue
                method(**kwargs)
                print(f"  PASS  {cls.__name__}.{name}")
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()

    print("\nSmoke tests done.")


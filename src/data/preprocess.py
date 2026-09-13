"""Materialize the merged ERA5 dataset + targets + static DEM feature into a
single memmap-friendly directory cache under processed/cache/. This avoids re-parsing
zips/HDF5/NetCDF on every training run and gives every downstream stage
(dataset, baseline, model, tests) one fast, deterministic source of arrays.

Raw data under Data/ is never modified -- this only *reads* it and writes
new files under processed/.
"""
from __future__ import annotations

import os
import time

import numpy as np

from src.utils.config import Config
from src.data.era5_loader import load_era5, subset_domain, validate_era5
from src.data.dem_loader import load_dem_on_era5_grid, fill_ocean_cells, validate_dem
from src.features.targets import compute_targets, validate_targets
from src.features.normalize import (
    SINGLE_VARS, PRESSURE_VARS, fit_normalization_stats, save_stats, validate_stats,
)
from src.data.windowing import build_windows, validate_no_leakage


def run_preprocessing(cfg: Config, out_path: str | None = None, verbose: bool = True) -> str:
    t0 = time.time()
    out_path = out_path or os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
    os.makedirs(out_path, exist_ok=True)

    if verbose:
        print("[preprocess] loading ERA5 ...")
    ds = load_era5(cfg)
    ds = subset_domain(ds, cfg)
    era5_report = validate_era5(ds, cfg)
    if verbose:
        print(f"[preprocess] ERA5 validated: {era5_report}")

    if verbose:
        print("[preprocess] computing targets ...")
    targets = compute_targets(ds, cfg)
    target_report = validate_targets(targets, cfg)
    if verbose:
        print(f"[preprocess] target base rates: {target_report}")

    if verbose:
        print("[preprocess] loading DEM ...")
    elev, dem_lats, dem_lons = load_dem_on_era5_grid(cfg)
    elev = fill_ocean_cells(elev)
    dem_report = validate_dem(elev)
    if verbose:
        print(f"[preprocess] DEM validated: {dem_report}")
    assert np.allclose(dem_lats, ds.latitude.values) and np.allclose(dem_lons, ds.longitude.values), \
        "DEM grid does not align with ERA5 grid"

    times = ds.valid_time.values
    validate_no_leakage(times, cfg)
    if verbose:
        print("[preprocess] leakage check passed")

    train_windows = build_windows(times, cfg, "train")
    train_idx = sorted({i for w in train_windows
                        for i in range(w.input_end_idx - cfg.get("sequence", "input_hours") + 1,
                                       w.input_end_idx + 1)})
    stats = fit_normalization_stats(ds, np.array(train_idx))
    validate_stats(stats)
    save_stats(stats, cfg.path("normalization", "stats_file"))
    if verbose:
        print(f"[preprocess] normalization stats fit on {len(train_idx)} train timesteps, saved")

    if verbose:
        print("[preprocess] stacking and saving surface variables (memmap friendly) ...")
    try:
        surface = np.stack([ds[v].values.astype(np.float32) for v in SINGLE_VARS], axis=1)
    except MemoryError as e:
        print(f"\nFATAL: Insufficient RAM to stack surface variables during preprocessing. "
              f"Requires ~500MB free RAM. Close other apps or run on Colab.\nOriginal error: {e}")
        import sys
        sys.exit(1)
    cin_i = SINGLE_VARS.index("cin")
    surface[:, cin_i] = np.nan_to_num(surface[:, cin_i], nan=0.0)
    np.save(os.path.join(out_path, "surface.npy"), surface)
    del surface

    if verbose:
        print("[preprocess] stacking and saving pressure variables (memmap friendly) ...")
    try:
        pressure = np.stack([ds[v].values.astype(np.float32) for v in PRESSURE_VARS], axis=1)
    except MemoryError as e:
        print(f"\nFATAL: Insufficient RAM to stack pressure variables during preprocessing. "
              f"Requires ~1.8GB free RAM. Close other apps or run on Colab.\nOriginal error: {e}")
        import sys
        sys.exit(1)
    np.nan_to_num(pressure, nan=0.0, copy=False)
    np.save(os.path.join(out_path, "pressure.npy"), pressure)
    del pressure

    if verbose:
        print("[preprocess] saving targets and meta ...")
    np.save(os.path.join(out_path, "pressure_levels_hpa.npy"), ds.pressure_level.values.astype(np.float32))
    np.save(os.path.join(out_path, "latitude.npy"), ds.latitude.values.astype(np.float32))
    np.save(os.path.join(out_path, "longitude.npy"), ds.longitude.values.astype(np.float32))
    np.save(os.path.join(out_path, "valid_time.npy"), times.astype("datetime64[ns]").astype(np.int64))
    np.save(os.path.join(out_path, "dem_elevation_m.npy"), elev.astype(np.float32))

    for name in ["heavy_rain", "extreme_rain", "severe_convective", "severe_weather", "label_valid", "rain_3h_mm"]:
        np.save(os.path.join(out_path, f"target_{name}.npy"), targets[name].values.astype(np.float32 if name != "label_valid" else bool))

    if verbose:
        # Sum size of files in directory
        size_mb = sum(os.path.getsize(os.path.join(out_path, f)) for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))) / 1e6
        print(f"[preprocess] wrote {out_path}/ ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")

    return out_path


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.utils.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, cli_overrides=args.overrides)
    run_preprocessing(cfg)

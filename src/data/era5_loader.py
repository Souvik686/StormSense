"""Load and merge raw ERA5 pressure-level and single-level files into one
tidy xarray.Dataset on the native 0.25 deg / hourly grid.

Design notes (see docs/DATA_INVENTORY.md for the full forensic inspection):
  - Pressure-level files are split across two variable groups per month:
      u,v            @ 1000/850/700 hPa
      z,q,t          @ 700/500/300/250 hPa
    They must be merged on the *union* of pressure levels (NaN where a
    variable was not requested at a given level) rather than intersected,
    since 700 hPa is the only level common to both groups.
  - Single-level files ship as two members per zip: an "instant" file
    (u10,v10,d2m,t2m,sp,cape,cin,tcwv) and an "accum" file (tp) using
    ECMWF's stepType split. Both share the same hourly valid_time axis.
  - Coverage is May-Oct only, for 2021-2024 (no downloads exist for the
    Nov-Apr off-season) -- this is a deliberate download choice, not
    missing data, and is treated as the modelling season throughout.
  - Grid: latitude descending 28.0->20.0, longitude ascending 84.0->90.0,
    33x25 points, exactly matching configs/default.yaml::domain.
"""
from __future__ import annotations

import glob
import os
import tempfile
import zipfile
from functools import lru_cache

import numpy as np
import xarray as xr

from src.utils.config import Config


def _open_pressure_level_files(pl_dir: str) -> xr.Dataset:
    files = sorted(glob.glob(os.path.join(pl_dir, "*", "*", "*.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 pressure-level files under {pl_dir}")
    by_group: dict[str, list[xr.Dataset]] = {}
    for p in files:
        ds = xr.open_dataset(p, decode_timedelta=False)
        key = "+".join(sorted(ds.data_vars))
        by_group.setdefault(key, []).append(ds)
    merged_groups = []
    for key, dsets in by_group.items():
        dsets = sorted(dsets, key=lambda d: d.valid_time.values[0])
        cat = xr.concat(dsets, dim="valid_time")
        cat = cat.sortby("valid_time").drop_duplicates("valid_time")
        merged_groups.append(cat)
    # Outer-join on pressure_level so 700 hPa (shared) lines up and the
    # levels unique to each group get NaN outside their native group.
    out = xr.merge(merged_groups, join="outer", compat="no_conflicts")
    for v in ("expver", "number"):
        if v in out.coords:
            out = out.drop_vars(v)
    return out


def _open_single_level_zips(sl_dir: str) -> xr.Dataset:
    zips = sorted(glob.glob(os.path.join(sl_dir, "*.zip")))
    if not zips:
        raise FileNotFoundError(f"No ERA5 single-level zips under {sl_dir}")
    instants, accums = [], []
    tmp = tempfile.mkdtemp(prefix="era5_sl_")
    for z in zips:
        dest = os.path.join(tmp, os.path.splitext(os.path.basename(z))[0])
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        for n in os.listdir(dest):
            ds = xr.open_dataset(os.path.join(dest, n), decode_timedelta=False)
            (accums if "accum" in n else instants).append(ds)
    inst = xr.concat(sorted(instants, key=lambda d: d.valid_time.values[0]), dim="valid_time")
    inst = inst.sortby("valid_time").drop_duplicates("valid_time")
    acc = xr.concat(sorted(accums, key=lambda d: d.valid_time.values[0]), dim="valid_time")
    acc = acc.sortby("valid_time").drop_duplicates("valid_time")
    out = xr.merge([inst, acc], join="inner", compat="no_conflicts")
    for v in ("expver", "number"):
        if v in out.coords:
            out = out.drop_vars(v)
    return out


@lru_cache(maxsize=1)
def _cached_load(pl_dir: str, sl_dir: str) -> xr.Dataset:
    pl = _open_pressure_level_files(pl_dir)
    sl = _open_single_level_zips(sl_dir)
    merged = xr.merge([pl, sl], join="inner", compat="no_conflicts")
    merged = merged.sortby("latitude", ascending=False).sortby("longitude")
    return merged


def load_era5(cfg: Config, use_cache: bool = True) -> xr.Dataset:
    """Return the merged ERA5 dataset (pressure + single levels).

    tp is converted from meters/hour (raw ECMWF units) to mm/hour in-place.
    """
    pl_dir = cfg.path("era5", "pressure_levels_dir")
    sl_dir = cfg.path("era5", "single_levels_dir")
    ds = _cached_load(pl_dir, sl_dir) if use_cache else _open_and_merge(pl_dir, sl_dir)
    ds = ds.copy()
    ds["tp"] = ds["tp"] * 1000.0
    ds["tp"].attrs["units"] = "mm"
    return ds


def _open_and_merge(pl_dir, sl_dir):
    pl = _open_pressure_level_files(pl_dir)
    sl = _open_single_level_zips(sl_dir)
    merged = xr.merge([pl, sl], join="inner", compat="no_conflicts")
    return merged.sortby("latitude", ascending=False).sortby("longitude")


def subset_domain(ds: xr.Dataset, cfg: Config) -> xr.Dataset:
    lat_min, lat_max = cfg.get("domain", "lat_min"), cfg.get("domain", "lat_max")
    lon_min, lon_max = cfg.get("domain", "lon_min"), cfg.get("domain", "lon_max")
    return ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))


def validate_era5(ds: xr.Dataset, cfg: Config) -> dict:
    """Run sanity checks and return a small report dict (raises on hard failures)."""
    report = {}
    expected_shape = tuple(cfg.get("domain", "grid_shape"))
    actual_shape = (ds.sizes["latitude"], ds.sizes["longitude"])
    assert actual_shape == expected_shape, f"grid shape {actual_shape} != expected {expected_shape}"
    report["grid_shape"] = actual_shape

    t = ds.valid_time.values
    assert np.all(np.diff(t).astype("timedelta64[h]").astype(int) >= 0), "valid_time not sorted"
    report["n_hours"] = len(t)
    report["t_min"], report["t_max"] = str(t.min()), str(t.max())

    for v in ["u10", "v10", "t2m", "d2m", "sp", "cape", "cin", "tcwv", "tp"]:
        assert v in ds.data_vars, f"missing single-level var {v}"
    for v in ["u", "v", "z", "q", "t"]:
        assert v in ds.data_vars, f"missing pressure-level var {v}"

    tp = ds["tp"].values
    assert np.nanmin(tp) >= -1e-6, "negative precipitation found"
    report["tp_max_mmph"] = float(np.nanmax(tp))
    report["tp_nan_frac"] = float(np.isnan(tp).mean())

    t2m = ds["t2m"].values
    assert np.nanmin(t2m) > 250 and np.nanmax(t2m) < 330, f"t2m out of physical range: [{np.nanmin(t2m)}, {np.nanmax(t2m)}]"
    report["t2m_range_K"] = [float(np.nanmin(t2m)), float(np.nanmax(t2m))]

    return report

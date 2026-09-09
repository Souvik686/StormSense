"""Target (label) generation for the severe-weather nowcasting problem.

IMPORTANT -- proxy labels, not observed ground truth
------------------------------------------------------
No direct, gridded, sub-daily flash-flood or "severe weather occurred" label
exists anywhere in the workspace (see docs/DATA_INVENTORY.md and
docs/DATA_COMPATIBILITY.md for the full inspection). IMD gridded rainfall is
daily-only; INSAT satellite QPE (HEM) barely overlaps the ERA5 training period
(93 hours, two cyclones only -- see docs/DATA_COMPATIBILITY.md) and cannot
support supervised learning. The only field with hourly, gap-free, 4-year
coverage over the full domain is ERA5 itself.

We therefore define all severe-weather targets as physically-motivated
proxies computed directly from ERA5 fields, using thresholds calibrated to
the empirical distribution of this specific domain/period (see
docs/ML_DESIGN.md "Label thresholds" for the percentile analysis that
produced the default config values). This is a deliberate, documented
methodological choice -- NOT a substitute for observed severe-weather
reports, and every place that consumes these labels must say "proxy".

Targets produced (all binary, computed on the same 0.25deg/hourly ERA5 grid
the inputs live on -- no downsampling/upsampling that could smear labels
across cells and leak neighbourhood information into a "spatial" label):

  heavy_rain    : rolling 3h accumulated tp > cfg.labels.heavy_rain_3h_mm
  extreme_rain  : rolling 3h accumulated tp > cfg.labels.extreme_rain_3h_mm
  severe_convective : (CAPE > cape_severe_jkg) & (CIN < cin_weak_jkg), i.e.
                       strong instability with weak inhibition -- a standard
                       synoptic proxy for thunderstorm/convective potential
  severe_weather (primary label) : heavy_rain OR severe_convective

Labels are computed on the FULL time axis before any windowing, using only
values at or before each label's own timestamp (the rolling window looks
backward), so label generation itself introduces no future leakage. Leakage
across the train/val/test boundary is prevented at the sequence-building
stage (src/data/dataset.py), which drops any window whose lookback or
lead-time horizon crosses a split boundary.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from src.utils.config import Config


def compute_rolling_precip(ds: xr.Dataset, window_hours: int) -> xr.DataArray:
    """Backward-looking rolling sum of tp (mm) over `window_hours`, ending at
    each timestamp (inclusive). First (window_hours - 1) timestamps per
    contiguous block are NaN (insufficient history) -- callers must mask
    these out rather than treating them as "zero rain"."""
    return ds["tp"].rolling(valid_time=window_hours, min_periods=window_hours).sum()


def compute_targets(ds: xr.Dataset, cfg: Config) -> xr.Dataset:
    window = cfg.get("labels", "rolling_window_hours")
    heavy_thr = cfg.get("labels", "heavy_rain_3h_mm")
    extreme_thr = cfg.get("labels", "extreme_rain_3h_mm")
    cape_thr = cfg.get("labels", "cape_severe_jkg")
    cin_thr = cfg.get("labels", "cin_weak_jkg")

    rain3h = compute_rolling_precip(ds, window)
    heavy_rain = (rain3h > heavy_thr)
    extreme_rain = (rain3h > extreme_thr)

    cin_filled = ds["cin"].fillna(0.0)  # ECMWF leaves CIN unset when CAPE==0; treat as "no inhibition"
    severe_conv = (ds["cape"] > cape_thr) & (cin_filled < cin_thr)

    severe_weather = (heavy_rain.fillna(False) | severe_conv).astype(np.float32)
    valid_mask = rain3h.notnull()  # False for the warm-up hours of each rolling window

    out = xr.Dataset({
        "rain_3h_mm": rain3h,
        "heavy_rain": heavy_rain.astype(np.float32).where(valid_mask),
        "extreme_rain": extreme_rain.astype(np.float32).where(valid_mask),
        "severe_convective": severe_conv.astype(np.float32),
        "severe_weather": severe_weather.where(valid_mask),
        "label_valid": valid_mask,
    })
    out.attrs["proxy_disclaimer"] = (
        "All labels are ERA5-derived proxies, not observed severe-weather reports. "
        "See docs/ML_DESIGN.md and docs/LIMITATIONS.md."
    )
    return out


def base_rates(targets: xr.Dataset) -> dict:
    """Report class balance for each label, restricted to valid (non-warm-up) timestamps."""
    valid = targets["label_valid"].values
    out = {}
    for name in ["heavy_rain", "extreme_rain", "severe_weather"]:
        v = targets[name].values[valid]
        out[name] = {"positive_rate": float(np.nanmean(v)), "n_valid": int(valid.sum())}
    conv = targets["severe_convective"].values
    out["severe_convective"] = {"positive_rate": float(np.mean(conv)), "n_valid": int(conv.size)}
    return out


def validate_targets(targets: xr.Dataset, cfg: Config) -> dict:
    rates = base_rates(targets)
    for name, r in rates.items():
        assert 0.0 <= r["positive_rate"] <= 1.0
        assert r["positive_rate"] < 0.5, (
            f"{name} positive rate {r['positive_rate']:.3f} suspiciously high for a 'severe' label "
            "-- check threshold calibration")
    return rates

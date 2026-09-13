"""Build (input window -> multi-horizon target) sample indices with a
chronological, event-aware, leakage-safe train/val/test split.

Leakage safeguards implemented here:
  1. Splits are by *year* (whole months), never by shuffling individual
     hours -- this keeps entire synoptic events (which last days) inside a
     single split.
  2. A configurable buffer (`split.buffer_hours`) of samples is dropped at
     each split boundary so that no input window (which looks `input_hours`
     back) or target horizon (which looks up to `max(lead_times)` forward)
     can span two different splits.
  3. Normalization statistics (mean/std per variable) are fit on the TRAIN
     split only (see src/features/normalize.py) and applied unchanged to
     val/test -- this file only produces the index lists that make that
     possible; it does not compute statistics itself.
  4. A sample's input window must be fully "valid" (see targets.label_valid)
     and the base rolling-precip warm-up region is excluded automatically
     because those timestamps carry NaN targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import Config


@dataclass
class WindowSample:
    """One training sample: the index of the last input timestep, plus the
    indices (into the full time axis) of each requested lead time's target."""
    input_end_idx: int
    lead_target_idx: dict  # {lead_hours: idx_into_time_axis}


def _year_of(times: np.ndarray) -> np.ndarray:
    return pd.DatetimeIndex(times).year.values


def build_windows(times: np.ndarray, cfg: Config, split: str) -> list[WindowSample]:
    """Return valid WindowSamples whose input window and ALL requested lead
    times fall within the given split's year range, with a buffer dropped
    at both ends of that range."""
    input_hours = cfg.get("sequence", "input_hours")
    lead_times = cfg.get("sequence", "lead_times_hours")
    stride = cfg.get("sequence", "stride_hours")
    buffer_h = cfg.get("split", "buffer_hours")

    years = _year_of(times)
    if split == "train":
        wanted_years = set(cfg.get("split", "train_years"))
    elif split == "val":
        wanted_years = {cfg.get("split", "val_year")}
    elif split == "test":
        wanted_years = {cfg.get("split", "test_year")}
    else:
        raise ValueError(split)

    n = len(times)
    max_lead = max(lead_times)
    samples = []
    # `times` is not necessarily contiguous across the Nov-Apr gap; treat any
    # step > 1 hour as a season boundary that a window may not cross.
    dt_hours = np.empty(n)
    dt_hours[0] = 0
    dt_hours[1:] = np.diff(times).astype("timedelta64[h]").astype(int)
    season_id = np.cumsum(dt_hours > 1)

    for end_idx in range(input_hours - 1, n - max_lead):
        start_idx = end_idx - input_hours + 1
        if season_id[start_idx] != season_id[end_idx]:
            continue  # window crosses an off-season gap
        if years[end_idx] not in wanted_years:
            continue
        # buffer: require the whole window+horizon to be >= buffer_h away
        # from a year that is NOT in wanted_years, so no leakage across
        # the split boundary through input lookback or target lookahead.
        lo = end_idx - input_hours + 1 - buffer_h
        hi = end_idx + max_lead + buffer_h
        if lo < 0 or hi >= n:
            continue
        if season_id[max(lo, 0)] != season_id[end_idx] or season_id[min(hi, n - 1)] != season_id[end_idx]:
            continue
        boundary_years = years[max(lo, 0):min(hi, n - 1) + 1]
        if not set(boundary_years).issubset(wanted_years):
            continue

        lead_idx = {}
        ok = True
        for lh in lead_times:
            ti = end_idx + lh
            if ti >= n or season_id[ti] != season_id[end_idx]:
                ok = False
                break
            lead_idx[lh] = ti
        if not ok:
            continue
        if (end_idx - (input_hours - 1)) % stride != 0:
            continue
        samples.append(WindowSample(input_end_idx=end_idx, lead_target_idx=lead_idx))
    return samples


def split_summary(times: np.ndarray, cfg: Config) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        s = build_windows(times, cfg, split)
        if s:
            starts = [times[w.input_end_idx] for w in s]
            out[split] = {"n_samples": len(s), "t_min": str(min(starts)), "t_max": str(max(starts))}
        else:
            out[split] = {"n_samples": 0}
    return out


def validate_no_leakage(times: np.ndarray, cfg: Config) -> None:
    """Hard assertion: no timestamp used as an input or target index in one
    split may also be used (as input or target) in another split."""
    input_hours = cfg.get("sequence", "input_hours")

    def touched_indices(samples):
        idx = set()
        for s in samples:
            idx.update(range(s.input_end_idx - input_hours + 1, s.input_end_idx + 1))
            idx.update(s.lead_target_idx.values())
        return idx

    train = touched_indices(build_windows(times, cfg, "train"))
    val = touched_indices(build_windows(times, cfg, "val"))
    test = touched_indices(build_windows(times, cfg, "test"))

    assert not (train & val), f"train/val index overlap: {len(train & val)} timestamps"
    assert not (train & test), f"train/test index overlap: {len(train & test)} timestamps"
    assert not (val & test), f"val/test index overlap: {len(val & test)} timestamps"

"""Build a GFS-input training cache for domain fine-tuning.

WHY

The model is trained on ERA5 and served on GFS. Measured over 54 cycles paired
at identical valid times, three inputs are badly shifted:

    cin   -1.717 training sd   (corr +0.20 even after the sign fix)
    d2m   -0.839 sd
    cape  -0.827 sd            (ERA5 mean 1450 J/kg vs GFS 760 -- literally half)

so at inference the model reads "weak instability" where the atmosphere it was
trained to recognise would have said "unstable". That is the largest identified
cause of live skill being roughly half the ERA5-input figure.

Fine-tuning on GFS INPUT with the UNCHANGED ERA5 LABELS lets the network learn
GFS's conventions without redefining the prediction target.

WHAT THIS WRITES

A directory with the same file layout `NowcastDatasetV2` already reads, so no
dataset or model code changes:

    surface.npy            (T, 9, H, W)    from GFS analyses
    pressure.npy           (T, 5, 6, H, W) from GFS analyses
    valid_time.npy         (T,)            int64 ns, the analysis times
    dem_elevation_m.npy                    copied from the ERA5 cache
    latitude.npy longitude.npy pressure_levels_hpa.npy   copied
    target_*.npy           (T, H, W)       ERA5 labels SLICED to those times

Labels are taken from the ERA5 cache at the *same valid times*, never recomputed
from GFS fields -- deliberately, so the target keeps its production definition.

TIME AXIS -- USE `--hourly`

GFS analyses exist only at 00/06/12/18Z, so the raw cache is 6-hourly. That is a
trap, because `windowing.build_windows` resolves a lead as a ROW OFFSET
(`ti = end_idx + lead_hours`), not as hours: on a 6-hourly axis "lead 8" means
48 hours ahead, and a 6-row input window spans 36 real hours. Training that way
with leads [4..16] would silently learn 24-96 hour forecasts while every config,
log line and served label still said 4-16 hours.

`--hourly` linearly interpolates between the real analyses so a row offset is an
hour again. This is also what the LIVE path already does
(`gfs_live._interpolate_slot`), so it makes training-time and inference-time
inputs identically constructed instead of differing in temporal resolution.
Interpolation never bridges a gap wider than one 6h bracket, and labels are
re-looked-up at the new times rather than interpolated (the target is
threshold-derived; interpolating it would invent fractional events).

Disclosed: interpolated rows are not independent observations. They add no
information beyond the bracketing analyses and smooth away sub-6-hourly
variability -- the same limitation the live pipeline documents. This cache is for
FINE-TUNING an ERA5-trained model, not for training one from scratch.

Resumable: every cycle is cached as a pickle by `fetch_gfs_slice`, so re-running
after an interruption re-downloads nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.normalize import SINGLE_VARS, PRESSURE_VARS
from src.inference import gfs_live

ERA5_CACHE = "processed/cache/era5_memmap"
COPY_FILES = ["dem_elevation_m.npy", "latitude.npy", "longitude.npy",
              "pressure_levels_hpa.npy"]
TARGETS = ["target_severe_weather.npy", "target_rain_3h_mm.npy",
           "target_heavy_rain.npy", "target_extreme_rain.npy",
           "target_severe_convective.npy", "target_label_valid.npy"]


def cycles_in(start: datetime, end: datetime, months):
    out, c = [], start
    while c <= end:
        if c.month in months:
            out.append(c)
        c += timedelta(hours=6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-05-01")
    ap.add_argument("--end", default="2022-10-31")
    ap.add_argument("--months", default="5,6,7,8,9,10")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="Stop after this many successful cycles (subset runs).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every Nth cycle; spreads a subset over the season "
                         "instead of clustering it at the start.")
    ap.add_argument("--out", default="processed/cache/gfs_memmap")
    ap.add_argument("--hourly", action="store_true",
                    help="Interpolate the 6-hourly analyses onto an HOURLY row "
                         "axis (strongly recommended -- see the note below).")
    a = ap.parse_args()

    months = {int(x) for x in a.months.split(",")}
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)

    lats = np.load(os.path.join(ERA5_CACHE, "latitude.npy"))
    lons = np.load(os.path.join(ERA5_CACHE, "longitude.npy"))
    levels = [int(x) for x in np.load(os.path.join(ERA5_CACHE, "pressure_levels_hpa.npy"))]
    H, W = len(lats), len(lons)

    e_times = np.load(os.path.join(ERA5_CACHE, "valid_time.npy")).astype("datetime64[ns]")
    e_index = {}
    for i, t in enumerate(e_times):
        e_index[datetime.fromtimestamp(t.astype("datetime64[s]").astype(int), tz=timezone.utc)] = i

    wanted = cycles_in(start, end, months)[::max(1, a.stride)]
    print(f"[build] {len(wanted)} candidate cycles in {a.start}..{a.end} "
          f"months={sorted(months)} stride={a.stride}")

    surf_rows, pres_rows, keep_times, keep_eidx = [], [], [], []
    failed = 0
    for n, cyc in enumerate(wanted, 1):
        if a.max_cycles and len(keep_times) >= a.max_cycles:
            break
        if cyc not in e_index:
            continue  # no ERA5 label at this instant; skip rather than invent one
        try:
            g = gfs_live.fetch_gfs_slice(cyc, 0, lats, lons)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [skip] {cyc:%Y-%m-%d %HZ}: {type(e).__name__}: {str(e)[:80]}")
            continue

        s = np.full((len(SINGLE_VARS), H, W), np.nan, dtype=np.float32)
        ok = True
        for vi, v in enumerate(SINGLE_VARS):
            if v not in g.surface:
                ok = False
                break
            s[vi] = g.surface[v]
        if not ok:
            failed += 1
            continue
        # Same harmonization the live path applies (CIN sign, tp units).
        ci = SINGLE_VARS.index("cin")
        s[ci] = np.abs(s[ci])
        ti = SINGLE_VARS.index("tp")
        s[ti] = np.clip(s[ti] * 3600.0, 0.0, None)

        p = np.full((len(PRESSURE_VARS), len(levels), H, W), np.nan, dtype=np.float32)
        for vi, v in enumerate(PRESSURE_VARS):
            lv = g.pressure.get(v, {})
            for li, L in enumerate(levels):
                if L in lv:
                    p[vi, li] = lv[L]

        surf_rows.append(s)
        pres_rows.append(p)
        keep_times.append(np.datetime64(cyc.replace(tzinfo=None), "ns"))
        keep_eidx.append(e_index[cyc])
        if len(keep_times) % 25 == 0:
            print(f"  [{n}/{len(wanted)}] kept={len(keep_times)} failed={failed}", flush=True)

    if not keep_times:
        raise SystemExit("no cycles were fetched; nothing to write")

    surf = np.stack(surf_rows)
    pres = np.stack(pres_rows)
    times64 = np.array(keep_times).astype("datetime64[ns]")

    if a.hourly:
        # WHY THIS IS NOT OPTIONAL IN PRACTICE.
        #
        # `windowing.build_windows` resolves a lead as a ROW OFFSET
        # (`ti = end_idx + lead_hours`), not as a number of hours. On a 6-hourly
        # row axis, "lead 8" would therefore mean 48 hours ahead, and a 6-row
        # input window would span 36 real hours. Training on that axis with the
        # config's [4..16] leads would silently learn 24-96 hour forecasts while
        # the code, the config and the served model all call them 4-16 hours.
        #
        # Interpolating onto an HOURLY axis makes a row offset equal an hour
        # again, so leads mean what they say and match both ERA5 training and
        # what the model is asked for at inference.
        #
        # It also removes a second mismatch: the LIVE path already linearly
        # interpolates two real analyses onto hourly slots
        # (gfs_live._interpolate_slot). Building the cache the same way means
        # training-time and inference-time inputs are constructed identically
        # rather than differing in temporal resolution.
        #
        # DISCLOSED: interpolated rows are not independent observations. They
        # carry no information beyond the bracketing analyses and smooth away
        # sub-6-hourly variability -- exactly the limitation the live pipeline
        # already documents. Only rows landing on a real analysis time are real.
        step_h = 1
        t0, t1 = times64.min(), times64.max()
        grid = np.arange(t0, t1 + np.timedelta64(1, "h"), np.timedelta64(step_h, "h"))
        src = times64.astype("datetime64[s]").astype(np.int64).astype(np.float64)
        dst = grid.astype("datetime64[s]").astype(np.int64).astype(np.float64)

        # Only interpolate INSIDE a 6h bracket; a gap (missing cycle, or the
        # Oct->May season break) must not be bridged by a fabricated ramp.
        keep_mask = np.zeros(len(dst), dtype=bool)
        seg = np.searchsorted(src, dst, side="right") - 1
        for k, (d, s_i) in enumerate(zip(dst, seg)):
            if 0 <= s_i < len(src) - 1 and (src[s_i + 1] - src[s_i]) <= 6 * 3600 + 1:
                keep_mask[k] = True
            elif s_i >= 0 and abs(d - src[s_i]) < 1.0:
                keep_mask[k] = True  # exactly on a real analysis
        dst_k = dst[keep_mask]
        grid_k = grid[keep_mask]

        def interp_axis(arr):
            """Vectorised linear interpolation along axis 0.

            `np.interp` is 1-D, so the obvious implementation loops over every
            spatial column -- 825 surface cells plus 24,750 pressure cells per
            variable, which is minutes per build. This computes the bracketing
            indices and blend weight once and applies them to the whole array,
            which is the same arithmetic in one pass.
            """
            flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
            hi = np.clip(np.searchsorted(src, dst_k, side="left"), 1, len(src) - 1)
            lo = hi - 1
            span = (src[hi] - src[lo])
            w = np.where(span > 0, (dst_k - src[lo]) / np.where(span > 0, span, 1.0), 0.0)
            w = np.clip(w, 0.0, 1.0).astype(np.float32)[:, None]
            out = flat[lo] * (1.0 - w) + flat[hi] * w
            return out.reshape((len(dst_k),) + arr.shape[1:])

        print(f"[build] interpolating {len(src)} six-hourly rows -> "
              f"{len(dst_k)} hourly rows (dropped {int((~keep_mask).sum())} "
              f"outside a 6h bracket)")
        surf = interp_axis(surf)
        pres = interp_axis(pres)
        times64 = grid_k

        # Labels must be re-looked-up at the NEW hourly times, never interpolated:
        # the target is binary/threshold-derived and interpolating it would invent
        # fractional events.
        new_eidx, keep_rows = [], []
        for r, t in enumerate(times64):
            dtu = datetime.fromtimestamp(t.astype("datetime64[s]").astype(int), tz=timezone.utc)
            if dtu in e_index:
                new_eidx.append(e_index[dtu])
                keep_rows.append(r)
        if not keep_rows:
            raise SystemExit("no interpolated row has an ERA5 label; aborting")
        keep_rows = np.array(keep_rows)
        surf, pres, times64 = surf[keep_rows], pres[keep_rows], times64[keep_rows]
        keep_eidx = new_eidx
        print(f"[build] {len(keep_rows)} hourly rows have an ERA5 label")

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "surface.npy"), surf)
    np.save(os.path.join(a.out, "pressure.npy"), pres)
    np.save(os.path.join(a.out, "valid_time.npy"), times64.astype(np.int64))
    for f in COPY_FILES:
        np.save(os.path.join(a.out, f), np.load(os.path.join(ERA5_CACHE, f)))
    idx = np.array(keep_eidx)
    for f in TARGETS:
        src = os.path.join(ERA5_CACHE, f)
        if os.path.exists(src):
            np.save(os.path.join(a.out, f), np.load(src, mmap_mode="r")[idx])

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(keep_times), "failed_cycles": failed,
        "window": [a.start, a.end], "months": sorted(months), "stride": a.stride,
        "time_axis": ("hourly, linearly interpolated between real 6-hourly analyses "
                      "(same construction as the live path)" if a.hourly else
                      "6-hourly GFS analysis times (00/06/12/18Z), NOT hourly"),
        "hourly_interpolated": bool(a.hourly),
        "inputs": "real GFS f000 analyses, same harmonization as the live path "
                  "(CIN abs(), tp = PRATE*3600)",
        "labels": "ERA5-derived targets sliced at the SAME valid times; the "
                  "production label definition is unchanged",
        "caveat": ("interpolated rows are not independent observations: they carry "
                   "no information beyond the bracketing analyses and smooth away "
                   "sub-6-hourly variability. Only rows on a real analysis time are "
                   "real. This cache is for FINE-TUNING an ERA5-trained model."
                   if a.hourly else
                   "input_hours consecutive rows here span 6x longer in real time "
                   "than on the hourly ERA5 axis, so a lead of N rows is N*6 hours; "
                   "use --hourly unless you know you want this"),
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\n[build] wrote {len(keep_times)} rows to {a.out} (failed {failed})")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

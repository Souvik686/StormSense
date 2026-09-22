"""FAR/POD trade-off menu, measured from a finished GFS backtest.

The model over-predicts: on the corrected v3 backtest it raises roughly six
false alarms for every correct one (+8h: 196 TP vs 1,278 FP). That is a
THRESHOLD choice, not a model limit -- the operating points in the checkpoint
were optimised for CSI, which penalises a miss and a false alarm equally.

Raising the threshold cuts false alarms immediately, at the cost of catching
fewer real events. There is no setting that improves both. This script measures
the actual curve so the operating point is a decision made on numbers rather
than a guess.

It reads the per-case probability/truth arrays the backtest already wrote, so it
adds no inference and cannot disturb a running job.

Usage:
    python scripts/threshold_sweep.py \
        --in reports/gfs_production_backtest_corrected_v3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.abspath("."))

CACHE = "processed/cache/era5_memmap"


def load_truth():
    times_ns = np.load(os.path.join(CACHE, "valid_time.npy"))
    times = [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in times_ns]
    return {t: i for i, t in enumerate(times)}, np.load(
        os.path.join(CACHE, "target_severe_weather.npy"), mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path",
                    default="reports/gfs_production_backtest_corrected_v3.json")
    ap.add_argument("--ckpt", default="Data/outputs/checkpoints_v3/v2_best.pt")
    ap.add_argument("--config", default="configs/v3_longlead.yaml")
    ap.add_argument("--out", default="reports/v3_threshold_sweep.json")
    a = ap.parse_args()

    bt = json.load(open(a.path, encoding="utf-8"))
    if bt.get("status") == "blocked":
        print("source backtest is blocked; nothing to sweep")
        return 1

    # Re-run the same cases the backtest used, capturing probabilities so the
    # sweep is over IDENTICAL data. (The backtest JSON stores summary stats, not
    # the raw arrays.)
    os.environ.setdefault("STORMSENSE_DEVICE", "cpu")
    os.environ["STORMSENSE_CKPT"] = a.ckpt
    os.environ["STORMSENSE_CONFIG"] = a.config
    from src.inference.nowcast_service import get_nowcast_service

    truth_index, severe = load_truth()
    svc = get_nowcast_service()
    leads = [L for L in svc.lead_times if L != 0]

    cases = [c for c in bt.get("cases", []) if c.get("status") == "ok"]
    print(f"[sweep] replaying {len(cases)} backtest cases for leads {leads}")

    acc = {L: {"p": [], "y": []} for L in leads}
    for i, c in enumerate(cases, 1):
        T = datetime.fromisoformat(c["simulated_now"])
        ok = svc.refresh_live_state(target_t0=T)
        if not ok:
            print(f"  case {i}: refresh failed, skipped")
            continue
        for li, L in enumerate(leads):
            vt = T + timedelta(hours=L)
            if vt not in truth_index:
                continue
            p = np.asarray(svc.live_pred["severe_weather_prob"][
                svc.lead_times.index(L)]).ravel()
            y = (np.asarray(severe[truth_index[vt]]) > 0).astype(int).ravel()
            if y.shape != p.shape:
                continue
            acc[L]["p"].append(p)
            acc[L]["y"].append(y)
        print(f"  case {i}/{len(cases)}  {T.isoformat()}")

    current = svc.predictor.threshold_per_lead or {}
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    out = {"source": a.path, "checkpoint": a.ckpt,
           "current_thresholds": {str(k): v for k, v in current.items()},
           "per_lead": {}}

    for L in leads:
        if not acc[L]["p"]:
            continue
        p = np.concatenate(acc[L]["p"]).astype(np.float64)
        y = np.concatenate(acc[L]["y"]).astype(np.float64)
        base = float(y.mean())
        cur = float(current.get(L, current.get(str(L), 0.5)))

        print()
        print(f"=== +{L}h   n={y.size:,}  base rate={base:.4f}  "
              f"current threshold={cur:.3f} ===")
        print(f"{'thr':>6} {'POD':>7} {'FAR':>7} {'CSI':>7} {'prec':>7} "
              f"{'F1':>7} {'alerts':>8} {'TP':>6} {'FP':>6}")
        rows = []
        for t in grid:
            pred = (p >= t)
            tp = float((pred & (y == 1)).sum())
            fp = float((pred & (y == 0)).sum())
            fn = float((~pred & (y == 1)).sum())
            pod = tp / max(tp + fn, 1e-12)
            far = fp / max(tp + fp, 1e-12)
            csi = tp / max(tp + fp + fn, 1e-12)
            prec = tp / max(tp + fp, 1e-12)
            f1 = 2 * prec * pod / max(prec + pod, 1e-12)
            rows.append({"threshold": float(t), "pod": pod, "far": far,
                         "csi": csi, "precision": prec, "f1": f1,
                         "alerts": int(tp + fp), "tp": int(tp), "fp": int(fp)})
            mark = "  <- current" if abs(t - cur) < 0.025 else ""
            print(f"{t:>6.2f} {pod:>7.3f} {far:>7.3f} {csi:>7.3f} {prec:>7.3f} "
                  f"{f1:>7.3f} {int(tp+fp):>8,} {int(tp):>6,} {int(fp):>6,}{mark}")

        best_csi = max(rows, key=lambda r: r["csi"])
        best_f1 = max(rows, key=lambda r: r["f1"])
        far70 = [r for r in rows if r["far"] <= 0.70]
        far50 = [r for r in rows if r["far"] <= 0.50]
        out["per_lead"][str(L)] = {
            "base_rate": base, "current_threshold": cur, "sweep": rows,
            "best_csi": best_csi, "best_f1": best_f1,
            "lowest_threshold_with_far_le_0.70": (
                min(far70, key=lambda r: r["threshold"]) if far70 else None),
            "lowest_threshold_with_far_le_0.50": (
                min(far50, key=lambda r: r["threshold"]) if far50 else None),
        }
        print(f"  best CSI  : thr={best_csi['threshold']:.2f} "
              f"CSI={best_csi['csi']:.3f} POD={best_csi['pod']:.3f} FAR={best_csi['far']:.3f}")
        if far70:
            r = min(far70, key=lambda x: x["threshold"])
            print(f"  FAR<=0.70 : thr={r['threshold']:.2f} "
                  f"POD={r['pod']:.3f} FAR={r['far']:.3f} (catches "
                  f"{r['pod']*100:.0f}% of events)")
        else:
            print("  FAR<=0.70 : NOT REACHABLE at any threshold")
        if far50:
            r = min(far50, key=lambda x: x["threshold"])
            print(f"  FAR<=0.50 : thr={r['threshold']:.2f} "
                  f"POD={r['pod']:.3f} FAR={r['far']:.3f}")
        else:
            print("  FAR<=0.50 : NOT REACHABLE at any threshold")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[sweep] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

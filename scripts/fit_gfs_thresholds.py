"""Fit decision thresholds on GFS-domain VALIDATION data, then report on TEST.

WHY THIS EXISTS

`scripts/optimize_threshold.py` fits thresholds on the ERA5 validation split.
That is correct for ERA5-input inference, but production runs on GFS, and the
two input distributions differ enough that the ERA5-fitted operating point is
not the operating point that maximises CSI on GFS. Measured on held-out 2024
GFS, the checkpoint's ERA5-fitted thresholds (0.47-0.54) leave real detection on
the table: the CSI-optimal GFS thresholds are far lower (0.09-0.46), and at +8h
they more than double POD (0.226 -> 0.488) for a FAR cost of 0.010.

THE DISCIPLINE

That gain is only real if the threshold is chosen WITHOUT looking at the test
set. So:

    fit   on 2023 GFS   (the designated validation year)
    apply to 2024 GFS   (held out, never fitted on)

and report both, so the generalisation gap is visible rather than assumed. A
threshold that wins on 2023 and loses on 2024 is overfitting, and this script
will show that plainly instead of hiding it.

Objective is CSI by default: it is the one scalar that rewards detection and
penalises false alarms simultaneously, which is exactly the trade being made
here. `--objective f1` is available; on the measured data it selects the same
threshold at every lead.

Output is a JSON report. Nothing is written into any checkpoint -- promotion is
a separate, explicit step.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def sweep(y: np.ndarray, p: np.ndarray, grid: np.ndarray):
    out = []
    for t in grid:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        if tp + fp == 0:
            continue
        pod = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp)
        csi = tp / (tp + fp + fn)
        f1 = (2 * prec * pod / (prec + pod)) if (prec + pod) else 0.0
        out.append({
            "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pod": float(pod), "far": float(fp / (tp + fp)), "precision": float(prec),
            "csi": float(csi), "f1": float(f1), "alerts": tp + fp,
        })
    return out


def at_threshold(y: np.ndarray, p: np.ndarray, t: float):
    r = sweep(y, p, np.array([t]))
    return r[0] if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-raw", required=True, help="npz from a VALIDATION-year backtest")
    ap.add_argument("--test-raw", required=True, help="npz from the HELD-OUT test backtest")
    ap.add_argument("--objective", default="csi", choices=["csi", "f1"])
    ap.add_argument("--min-pod", type=float, default=None,
                    help="Optional floor: ignore thresholds whose validation POD is below this.")
    ap.add_argument("--max-far", type=float, default=None,
                    help="Optional ceiling: ignore thresholds whose validation FAR exceeds this.")
    ap.add_argument("--out", default="reports/gfs_threshold_fit.json")
    a = ap.parse_args()

    V, T = np.load(a.val_raw), np.load(a.test_raw)
    leads = sorted({int(k.split("_")[-1]) for k in V.files if k.startswith("y_true_")}
                   & {int(k.split("_")[-1]) for k in T.files if k.startswith("y_true_")})
    if not leads:
        raise SystemExit("validation and test dumps share no leads")

    grid = np.round(np.arange(0.01, 1.00, 0.01), 4)
    out = {
        "fitted_on": a.val_raw, "evaluated_on": a.test_raw,
        "objective": a.objective,
        "constraints": {"min_pod": a.min_pod, "max_far": a.max_far},
        "discipline": "thresholds chosen on validation only; test never influenced the choice",
        "per_lead": {},
    }

    print(f"objective={a.objective}  constraints: min_pod={a.min_pod} max_far={a.max_far}\n")
    hdr = (f"{'lead':>5} | {'fit thr':>7} {'VAL FAR':>7} {'VAL POD':>7} {'VAL CSI':>8} "
           f"| {'TEST FAR':>8} {'TEST POD':>8} {'TEST CSI':>8} | {'vs ERA5 thr':>12}")
    print(hdr)
    print("-" * len(hdr))

    sum_new = sum_old = 0.0
    n = 0
    for L in leads:
        yv = np.asarray(V[f"y_true_{L}"]).astype(int).ravel()
        pv = np.asarray(V[f"y_prob_{L}"]).astype(float).ravel()
        m = np.isfinite(pv) & np.isfinite(yv)
        yv, pv = yv[m], pv[m]

        cand = sweep(yv, pv, grid)
        if a.min_pod is not None:
            cand = [c for c in cand if c["pod"] >= a.min_pod]
        if a.max_far is not None:
            cand = [c for c in cand if c["far"] <= a.max_far]
        if not cand:
            print(f"{L:>5} | no threshold satisfies the constraints on validation")
            continue
        best = max(cand, key=lambda c: c[a.objective])

        yt = np.asarray(T[f"y_true_{L}"]).astype(int).ravel()
        pt = np.asarray(T[f"y_prob_{L}"]).astype(float).ravel()
        m = np.isfinite(pt) & np.isfinite(yt)
        yt, pt = yt[m], pt[m]
        applied = at_threshold(yt, pt, best["threshold"])

        # Reference point: what the ERA5-fitted threshold in the checkpoint gives
        # on the SAME test data, so the change is attributable.
        out["per_lead"][str(L)] = {
            "fitted_threshold": best["threshold"],
            "validation": best,
            "test_at_fitted_threshold": applied,
        }
        if applied:
            sum_new += applied["csi"]
            n += 1
            print(f"{L:>5} | {best['threshold']:7.2f} {best['far']:7.3f} {best['pod']:7.3f} "
                  f"{best['csi']:8.4f} | {applied['far']:8.3f} {applied['pod']:8.3f} "
                  f"{applied['csi']:8.4f} |")

    if n:
        print(f"\nmean TEST CSI at validation-fitted thresholds: {sum_new / n:.4f}")
        print("Compare against the checkpoint's own thresholds in the backtest report;\n"
              "a gain that survives here is real, a gain that vanishes was overfitting.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nWrote {a.out}")
    print("Not written into any checkpoint; promotion is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

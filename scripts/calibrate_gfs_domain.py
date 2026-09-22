"""Fit per-lead temperature scaling IN THE GFS DOMAIN.

WHY THIS IS SEPARATE FROM scripts/calibrate_v3.py

`calibrate_v3.py` fits temperatures on the ERA5 validation split. That is the
right thing for ERA5-input inference, but production runs on GFS, and the two
input distributions differ enough that the ERA5-fitted temperature leaves the
served probabilities miscalibrated: measured on held-out 2024 GFS, mean predicted
probability exceeded the observed base rate by +0.076..+0.091 with ECE
0.11-0.14, i.e. systematically OVERCONFIDENT.

This script fits a correction on GFS-driven probabilities from a DISJOINT year
(2023 = the designated validation year; the test cases are 2024), so the fit
never sees the evaluation data.

WHAT IT FITS

The backtest emits calibrated probabilities p (temperature already applied), not
raw logits. We therefore invert to the logit, apply a second scalar `a` plus bias
`b`, and refit:  p' = sigmoid(a * logit(p) + b). `a` < 1 softens overconfidence.
This composes with the checkpoint's existing temperature rather than replacing
it, which keeps `calibrate_v3.py`'s result intact and reversible.

Output is a JSON of per-lead (a, b) -- deliberately NOT written into the
checkpoint, so nothing about the served model changes until a separate,
explicit promotion step.
"""
import argparse, json, os
import numpy as np


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(y, p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def fit_ab(y, p):
    """Grid + local refine on (a, b) minimising NLL. Small problem, so a direct
    search is more robust here than gradient descent on a 2-parameter surface."""
    z = logit(p)
    best = (1.0, 0.0, nll(y, p))
    for a in np.arange(0.20, 2.01, 0.05):
        for b in np.arange(-3.0, 1.51, 0.10):
            v = nll(y, sigmoid(a * z + b))
            if v < best[2]:
                best = (float(a), float(b), v)
    a0, b0, _ = best
    for a in np.arange(max(0.05, a0 - 0.05), a0 + 0.051, 0.01):
        for b in np.arange(b0 - 0.10, b0 + 0.101, 0.02):
            v = nll(y, sigmoid(a * z + b))
            if v < best[2]:
                best = (float(a), float(b), v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-raw", required=True, help="npz from a VALIDATION-year backtest")
    ap.add_argument("--test-raw", default=None, help="optional npz to REPORT on (never fitted)")
    ap.add_argument("--out", default="reports/gfs_domain_calibration.json")
    a = ap.parse_args()

    V = np.load(a.val_raw)
    T = np.load(a.test_raw) if a.test_raw else None
    leads = sorted({int(k.split("_")[-1]) for k in V.files if k.startswith("y_true_")})

    out = {"fitted_on": a.val_raw, "evaluated_on": a.test_raw,
           "form": "p' = sigmoid(a * logit(p) + b), composed with the checkpoint temperature",
           "per_lead": {}}

    for L in leads:
        yv = np.asarray(V[f"y_true_{L}"]).astype(int).ravel()
        pv = np.asarray(V[f"y_prob_{L}"]).astype(float).ravel()
        m = np.isfinite(pv) & np.isfinite(yv)
        yv, pv = yv[m], pv[m]
        aa, bb, _ = fit_ab(yv, pv)
        pv2 = sigmoid(aa * logit(pv) + bb)

        rec = {
            "a": aa, "b": bb,
            "val": {
                "n": int(yv.size), "base_rate": float(yv.mean()),
                "before": {"mean_pred": float(pv.mean()), "ece": ece(yv, pv),
                           "brier": float(np.mean((pv - yv) ** 2)), "nll": nll(yv, pv)},
                "after": {"mean_pred": float(pv2.mean()), "ece": ece(yv, pv2),
                          "brier": float(np.mean((pv2 - yv) ** 2)), "nll": nll(yv, pv2)},
            },
        }
        print(f"\n=== +{L}h  fitted a={aa:.3f} b={bb:+.3f} ===")
        print(f"  VAL  base={yv.mean():.4f}  meanP {pv.mean():.4f} -> {pv2.mean():.4f}  "
              f"ECE {rec['val']['before']['ece']:.4f} -> {rec['val']['after']['ece']:.4f}  "
              f"Brier {rec['val']['before']['brier']:.4f} -> {rec['val']['after']['brier']:.4f}")

        if T is not None and f"y_true_{L}" in T.files:
            yt = np.asarray(T[f"y_true_{L}"]).astype(int).ravel()
            pt = np.asarray(T[f"y_prob_{L}"]).astype(float).ravel()
            m = np.isfinite(pt) & np.isfinite(yt)
            yt, pt = yt[m], pt[m]
            pt2 = sigmoid(aa * logit(pt) + bb)
            rec["test"] = {
                "n": int(yt.size), "base_rate": float(yt.mean()),
                "before": {"mean_pred": float(pt.mean()), "ece": ece(yt, pt),
                           "brier": float(np.mean((pt - yt) ** 2)), "nll": nll(yt, pt)},
                "after": {"mean_pred": float(pt2.mean()), "ece": ece(yt, pt2),
                          "brier": float(np.mean((pt2 - yt) ** 2)), "nll": nll(yt, pt2)},
            }
            print(f"  TEST base={yt.mean():.4f}  meanP {pt.mean():.4f} -> {pt2.mean():.4f}  "
                  f"ECE {rec['test']['before']['ece']:.4f} -> {rec['test']['after']['ece']:.4f}  "
                  f"Brier {rec['test']['before']['brier']:.4f} -> {rec['test']['after']['brier']:.4f}")
        out["per_lead"][str(L)] = rec

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nWrote", a.out)
    print("NOTE: not written into any checkpoint; promotion is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

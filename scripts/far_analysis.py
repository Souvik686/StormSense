"""Dense threshold sweep + calibration audit on held-out GFS probabilities.

Consumes the .npz written by `historical_backtest.py --dump-raw`, i.e. the SAME
arrays the headline metrics were computed from -- no re-inference, no
re-sampling, no separate protocol.

Sections B and C of the FAR-reduction phase:
  * FAR / POD / precision / F1 / CSI / TP-FP-FN-TN at thresholds 0.01..0.99
  * best operating point under POD >= 0.30 / 0.40 / 0.50 constraints
  * whether FAR <= 0.70 / 0.75 / 0.80 is reachable at useful POD
  * reliability diagram, ECE, Brier, and a test of whether the model is
    systematically overconfident on GFS input
"""
import argparse, json, os
import numpy as np


def contingency(y, p, t):
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def metrics_at(y, p, t):
    tp, fp, fn, tn = contingency(y, p, t)
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    f1 = (2 * prec * pod / (prec + pod)) if (prec + pod) and np.isfinite(prec) and np.isfinite(pod) and (prec + pod) > 0 else float("nan")
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pod": float(pod), "far": float(far), "precision": float(prec),
            "csi": float(csi), "f1": float(f1), "alerts": tp + fp}


def reliability(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = ids == b
        if not np.any(m):
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        w = float(m.sum() / len(p))
        ece += w * abs(conf - acc)
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": int(m.sum()), "mean_predicted": conf,
                     "observed_frequency": acc, "gap": conf - acc})
    return rows, float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="tag for this candidate")
    a = ap.parse_args()

    d = np.load(a.raw)
    leads = sorted({int(k.split("_")[-1]) for k in d.files if k.startswith("y_true_")})
    grid = np.round(np.arange(0.01, 1.00, 0.01), 4)
    out = {"candidate": a.label, "source": a.raw, "per_lead": {}}

    for L in leads:
        y = np.asarray(d[f"y_true_{L}"]).astype(int).ravel()
        p = np.asarray(d[f"y_prob_{L}"]).astype(float).ravel()
        m = np.isfinite(p) & np.isfinite(y)
        y, p = y[m], p[m]
        base = float(y.mean())

        sweep = [metrics_at(y, p, t) for t in grid]
        valid = [s for s in sweep if s["alerts"] > 0 and np.isfinite(s["far"])]

        def best_under(min_pod):
            c = [s for s in valid if np.isfinite(s["pod"]) and s["pod"] >= min_pod]
            return min(c, key=lambda s: s["far"]) if c else None

        def reachable(max_far):
            c = [s for s in valid if np.isfinite(s["far"]) and s["far"] <= max_far]
            return max(c, key=lambda s: s["pod"]) if c else None

        best_csi = max(valid, key=lambda s: (s["csi"] if np.isfinite(s["csi"]) else -1)) if valid else None
        best_f1 = max(valid, key=lambda s: (s["f1"] if np.isfinite(s["f1"]) else -1)) if valid else None

        rows, ece = reliability(y, p)
        brier = float(np.mean((p - y) ** 2))
        # Overconfidence: mean predicted probability vs observed base rate.
        overconf = float(p.mean() - base)

        out["per_lead"][str(L)] = {
            "samples": int(y.size), "event_prevalence": base,
            "mean_predicted_probability": float(p.mean()),
            "overconfidence_mean_pred_minus_base": overconf,
            "brier": brier, "ece": ece,
            "reliability_bins": rows,
            "best_csi": best_csi, "best_f1": best_f1,
            "min_far_at_pod": {f"{k:.2f}": best_under(k) for k in (0.30, 0.40, 0.50)},
            "max_pod_at_far": {f"{k:.2f}": reachable(k) for k in (0.70, 0.75, 0.80, 0.85)},
            "sweep": sweep,
        }

        print(f"\n=== +{L}h  n={y.size}  base={base:.4f}  meanP={p.mean():.4f} "
              f"(overconfidence {overconf:+.4f})  Brier={brier:.4f} ECE={ece:.4f} ===")
        if best_csi:
            print(f"  best CSI : t={best_csi['threshold']:.2f} CSI={best_csi['csi']:.4f} "
                  f"POD={best_csi['pod']:.3f} FAR={best_csi['far']:.3f} prec={best_csi['precision']:.3f}")
        for k in (0.30, 0.40, 0.50):
            s = best_under(k)
            print(f"  POD>={k:.2f} -> " + (f"min FAR={s['far']:.3f} at t={s['threshold']:.2f} "
                  f"(POD={s['pod']:.3f} prec={s['precision']:.3f} CSI={s['csi']:.4f} alerts={s['alerts']})"
                  if s else "UNREACHABLE"))
        for k in (0.70, 0.75, 0.80):
            s = reachable(k)
            print(f"  FAR<={k:.2f} -> " + (f"max POD={s['pod']:.3f} at t={s['threshold']:.2f} "
                  f"(FAR={s['far']:.3f} CSI={s['csi']:.4f} alerts={s['alerts']})"
                  if s else "UNREACHABLE"))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nWrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

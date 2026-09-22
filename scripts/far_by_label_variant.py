"""How much of the measured FAR is the LABEL rather than the model?

Re-scores the SAME held-out GFS predictions against two truth definitions:

  production : (3h rain > 15mm) OR (CAPE > 2000 AND CIN < 50)   [unchanged]
  impact     : (3h rain > 15mm)                                  [separate experiment]

The production label is NOT modified anywhere; this only re-scores. Section H
requires alternative label definitions to be evaluated as separate experiments,
and this is that experiment.

Motivation (reports/label_quality_audit.json): 77.7% of production positives come
from the CAPE/CIN branch alone, and 56.9% of all positives carry < 1mm of rain.
So a model that reproduces the label perfectly still looks like it false-alarms
to a user who reads an alert as "heavy rain will occur".
"""
import argparse, json, os, sys
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath("."))
CACHE = "processed/cache/era5_memmap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", required=True, help="backtest json (for case/valid times)")
    ap.add_argument("--raw", required=True, help="npz of y_true/y_prob per lead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bt = json.load(open(a.backtest))
    d = np.load(a.raw)
    leads = sorted({int(k.split("_")[-1]) for k in d.files if k.startswith("y_true_")})

    times = np.load(os.path.join(CACHE, "valid_time.npy")).astype("datetime64[ns]")
    tindex = {}
    for i, t in enumerate(times):
        tindex[datetime.fromtimestamp(t.astype("datetime64[s]").astype(int), tz=timezone.utc)] = i
    heavy = np.load(os.path.join(CACHE, "target_heavy_rain.npy"), mmap_mode="r")

    # Rebuild the impact-label truth in the SAME case/lead order the backtest used.
    ok_cases = [c for c in bt["cases"] if c["status"] == "ok"]
    out = {"note": "production label unchanged; 'impact' is a separate scoring experiment",
           "per_lead": {}}

    for L in leads:
        y_prod = np.asarray(d[f"y_true_{L}"]).astype(int).ravel()
        p = np.asarray(d[f"y_prob_{L}"]).astype(float).ravel()
        imp = []
        for c in ok_cases:
            T = datetime.fromisoformat(c["simulated_now"])
            vt = T + timedelta(hours=L)
            if vt not in tindex:
                imp.append(None); continue
            arr = np.asarray(heavy[tindex[vt]], dtype=np.float32).ravel()
            imp.append(arr)
        if any(x is None for x in imp):
            print(f"+{L}h: missing truth for some cases; skipped")
            continue
        y_imp = (np.concatenate(imp) > 0).astype(int)
        if y_imp.shape != y_prod.shape:
            print(f"+{L}h: shape mismatch {y_imp.shape} vs {y_prod.shape}; skipped")
            continue
        m = np.isfinite(p) & np.isfinite(y_imp)
        p_, yp_, yi_ = p[m], y_prod[m], y_imp[m]

        rec = {"samples": int(p_.size)}
        for name, y in (("production", yp_), ("impact_rain_only", yi_)):
            base = float(y.mean())
            # best-CSI operating point, chosen on THIS scoring's own sweep
            best = None
            for t in np.round(np.arange(0.01, 1.0, 0.01), 4):
                pred = p_ >= t
                tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
                fn = int(np.sum(~pred & (y == 1)))
                if tp + fp == 0:
                    continue
                csi = tp / (tp + fp + fn)
                if best is None or csi > best["csi"]:
                    best = {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn,
                            "csi": float(csi), "pod": float(tp / (tp + fn)) if (tp + fn) else None,
                            "far": float(fp / (tp + fp)), "precision": float(tp / (tp + fp))}
            try:
                from sklearn.metrics import precision_recall_curve, auc
                pr, rc, _ = precision_recall_curve(y, p_)
                pra = float(auc(rc, pr))
            except Exception:
                pra = None
            rec[name] = {"base_rate": base, "pr_auc": pra,
                         "skill_vs_no_skill": (pra / base) if (pra and base) else None,
                         "best_csi_point": best}
        out["per_lead"][str(L)] = rec
        pp, ii = rec["production"], rec["impact_rain_only"]
        print(f"\n=== +{L}h ===")
        print(f"  production label : base={pp['base_rate']:.4f} PR-AUC={pp['pr_auc']:.4f} "
              f"skill={pp['skill_vs_no_skill']:.2f}x  bestCSI FAR={pp['best_csi_point']['far']:.3f} "
              f"POD={pp['best_csi_point']['pod']:.3f}")
        print(f"  impact (rain>=15): base={ii['base_rate']:.4f} PR-AUC={ii['pr_auc']:.4f} "
              f"skill={ii['skill_vs_no_skill']:.2f}x  bestCSI FAR={ii['best_csi_point']['far']:.3f} "
              f"POD={ii['best_csi_point']['pod']:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nWrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

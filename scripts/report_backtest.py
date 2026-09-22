"""Per-lead production-readiness report from a GFS backtest JSON.

Adds the metrics the raw backtest does not emit (precision, F1, skill ratio,
confusion matrix in one place) and, critically, checks the SAMPLING for the
diurnal alias that invalidated the earlier runs: if observed prevalence climbs
monotonically with lead, the schedule -- not the model -- is driving the scores.

Production criteria are fixed BEFORE looking at the numbers, so no lead can be
talked into passing:

  1. PR-AUC >= 2.0x its own no-skill baseline   (must beat climatology clearly)
  2. FAR <= 0.85                                 (at most ~5 in 6 alerts false)
  3. POD >= 0.30                                 (catches at least ~1 in 3)
  4. Sampling not diurnally aliased              (prevalence not monotonic in lead)

A lead must satisfy ALL of them.

Usage:
    python scripts/report_backtest.py --in reports/gfs_production_backtest_corrected_v3.json
"""
from __future__ import annotations

import argparse
import json


CRITERIA = {
    "min_skill_ratio": 2.0,
    "max_far": 0.85,
    "min_pod": 0.30,
}


def derive(h: dict) -> dict:
    tp, fp, fn, tn = h["tp"], h["fp"], h["fn"], h["tn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))
    ns = h.get("pr_auc_baseline_no_skill")
    pa = h.get("pr_auc")
    skill = (pa / ns) if (pa is not None and ns) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "skill_ratio": skill}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path",
                    default="reports/gfs_production_backtest_corrected_v3.json")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    a = ap.parse_args()

    d = json.load(open(a.path, encoding="utf-8"))
    if d.get("status") == "blocked":
        print(f"BLOCKED: {d.get('reason')}")
        return 1

    ph = d["per_horizon"]
    leads = sorted(int(k) for k in ph)

    print("=" * 78)
    print("CORRECTED GFS PRODUCTION BACKTEST -- PER-LEAD REPORT")
    print("=" * 78)
    print(f"source        : {a.path}")
    print(f"cases         : {d.get('cases_completed')} / {d.get('cases_attempted')} attempted")
    meth = d.get("methodology", {})
    for k in ("cycle_selection", "input", "inference_path", "truth", "threshold"):
        if k in meth:
            print(f"{k:14}: {meth[k]}")

    # ---- sampling validity -------------------------------------------------
    prev = [ph[str(L)]["event_prevalence"] for L in leads]
    print()
    print("SAMPLING CHECK  (diurnal alias detector)")
    print(f"  prevalence by lead: " +
          "  ".join(f"+{L}h={p:.4f}" for L, p in zip(leads, prev)))

    # Test the TREND, not strict monotonicity. The first aliased run dipped
    # between +8h and +10h (0.0461 -> 0.0386) and then climbed hard to 0.1043,
    # so a strict all-pairs-increasing check reported "OK" on data already
    # proven to be aliased. What matters is whether prevalence is systematically
    # higher at longer leads, which is what a fixed sampling hour produces.
    n = len(prev)
    mean_p = sum(prev) / n
    mean_l = sum(leads) / n
    cov = sum((leads[i] - mean_l) * (prev[i] - mean_p) for i in range(n))
    var_l = sum((L - mean_l) ** 2 for L in leads)
    slope = cov / var_l if var_l else 0.0
    # Correlation, so the verdict does not depend on prevalence units.
    var_p = sum((p - mean_p) ** 2 for p in prev)
    corr = (cov / (var_l ** 0.5 * var_p ** 0.5)) if (var_l and var_p) else 0.0
    ratio = (max(prev) / min(prev)) if min(prev) > 0 else float("inf")

    # Aliased when prevalence trends strongly upward with lead AND the spread is
    # large enough to move the scores. Both must hold, so ordinary sampling
    # noise does not trip it.
    monotonic = (corr >= 0.80) and (ratio >= 1.5)

    print(f"  trend: slope {slope:+.5f}/h   corr(lead, prevalence) {corr:+.3f}   "
          f"max/min {ratio:.2f}x")
    if monotonic:
        print("  >> ALIASED: prevalence trends strongly upward with lead. The "
              "sampling\n     schedule is driving the scores; cross-horizon "
              "comparison is NOT valid.")
    else:
        print("  >> OK: prevalence shows no strong upward trend with lead, so "
              "the schedule\n     is not systematically feeding stormier hours "
              "to longer leads.")

    # ---- per lead ----------------------------------------------------------
    print()
    hdr = (f"{'lead':>5} {'samples':>8} {'base':>7} {'PR-AUC':>8} {'skill':>7} "
           f"{'CSI':>7} {'POD':>7} {'FAR':>7} {'prec':>7} {'F1':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for L in leads:
        h = ph[str(L)]
        e = derive(h)
        rows[L] = {**h, **e}
        print(f"{'+'+str(L)+'h':>5} {h['samples']:>8,} {h['event_prevalence']:>7.4f} "
              f"{(h['pr_auc'] or float('nan')):>8.4f} {e['skill_ratio']:>6.2f}x "
              f"{h['csi']:>7.4f} {h['pod']:>7.4f} {h['far']:>7.4f} "
              f"{e['precision']:>7.4f} {e['f1']:>7.4f}")

    print()
    print("CONFUSION MATRICES")
    for L in leads:
        h = ph[str(L)]
        print(f"  +{L:>2}h   TP={h['tp']:>6}  FP={h['fp']:>6}  "
              f"FN={h['fn']:>6}  TN={h['tn']:>6}   thr={h['threshold_used']:.3f}")

    # ---- decision ----------------------------------------------------------
    print()
    print("PRODUCTION DECISION  (criteria fixed in advance)")
    print(f"  skill >= {CRITERIA['min_skill_ratio']}x no-skill, "
          f"FAR <= {CRITERIA['max_far']}, POD >= {CRITERIA['min_pod']}, "
          f"sampling not aliased")
    print()
    decisions = {}
    for L in leads:
        r = rows[L]
        checks = {
            "skill": (r["skill_ratio"] >= CRITERIA["min_skill_ratio"],
                      f"{r['skill_ratio']:.2f}x"),
            "far": (r["far"] <= CRITERIA["max_far"], f"{r['far']:.3f}"),
            "pod": (r["recall"] >= CRITERIA["min_pod"], f"{r['recall']:.3f}"),
            "sampling": (not monotonic, "aliased" if monotonic else "ok"),
        }
        ok = all(v[0] for v in checks.values())
        decisions[L] = {
            "production_worthy": bool(ok),
            "checks": {k: {"pass": bool(v[0]), "value": v[1]}
                       for k, v in checks.items()},
            "metrics": {k: r[k] for k in
                        ("pr_auc", "csi", "pod", "far", "precision", "f1",
                         "skill_ratio", "event_prevalence", "samples",
                         "tp", "fp", "fn", "tn", "threshold_used")},
        }
        fails = [k for k, v in checks.items() if not v[0]]
        verdict = "PRODUCTION-WORTHY" if ok else "NOT production-worthy"
        detail = "" if ok else "  (fails: " + ", ".join(
            f"{k}={checks[k][1]}" for k in fails) + ")"
        print(f"  +{L:>2}h  {verdict}{detail}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({
                "source": a.path,
                "criteria": CRITERIA,
                "sampling_aliased": bool(monotonic),
                "prevalence_by_lead": {str(L): p for L, p in zip(leads, prev)},
                "decisions": {str(k): v for k, v in decisions.items()},
            }, f, indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

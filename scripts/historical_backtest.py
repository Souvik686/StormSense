"""Historical GFS PRODUCTION backtest.

WHAT THIS DOES (and, just as importantly, what it does not)
-----------------------------------------------------------
For each simulated historical "NOW" instant T:

  1. Determine which GFS operational cycle would GENUINELY have been published
     by T, honouring real production latency (src.inference.gfs_live's
     `_cycles_available_at` -- the SAME selection rule the live path uses). A
     cycle from T's future can never be selected.
  2. Fetch that cycle's real f000 ANALYSES (never forecast hours) and run the
     actual production preprocessing + inference path
     (NowcastService.refresh_live_state), not a reimplementation.
  3. Produce +2/+4/+6h predictions.
  4. Verify them against INDEPENDENT ground truth at the corresponding valid
     times -- the ERA5-derived target fields, which are a completely different
     data source from the GFS inputs the model consumed.
  5. Report sample counts, event prevalence, CSI, POD, FAR, PR-AUC, Brier and
     calibration (ECE), per horizon.

WHAT IS BEING VERIFIED
  Whether the model, driven by GFS analyses exactly as it is in production,
  produces skilful severe-weather probabilities at +2/+4/+6h, scored against
  ERA5-derived proxy labels.

WHAT IS *NOT* BEING VERIFIED
  * Not verified against observed severe-weather reports. The labels are ERA5
    proxies (see src/features/targets.py); no gridded sub-daily observed
    severe-weather truth exists for this domain/period.
  * The verification labels come from ERA5 while the inputs come from GFS.
    That is deliberate (it keeps truth independent of input) but it means
    part of any measured error is GFS-vs-ERA5 representation difference, not
    purely model error.
  * Sample size is bounded by how far back the public GFS archive and the
    local ERA5 target cache overlap. N is reported; treat wide intervals as
    wide.

The scored variable is `severe_weather_prob` -- the calibrated probability head
whose training label is exactly the target field used as truth here. The
compound `overall_risk` is deliberately NOT scored against that label: it mixes
in rainfall and terrain terms and is not the quantity the label defines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.inference import gfs_live  # noqa: E402
from src.inference.nowcast_service import get_nowcast_service  # noqa: E402

CACHE = "processed/cache/era5_memmap"


def load_truth():
    """ERA5-derived verification targets, independent of the GFS model input."""
    times_ns = np.load(os.path.join(CACHE, "valid_time.npy"))
    times = [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in times_ns]
    index = {t: i for i, t in enumerate(times)}
    severe = np.load(os.path.join(CACHE, "target_severe_weather.npy"), mmap_mode="r")
    return index, severe, times


def calc_ece(y_true, y_prob, n_bins=10):
    """Expected calibration error: mean |confidence - accuracy| over bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if np.any(m):
            ece += (m.sum() / len(y_prob)) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def contingency(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def score(y_true, y_prob, threshold):
    from sklearn.metrics import auc, brier_score_loss, precision_recall_curve

    y_pred = (y_prob >= threshold).astype(int)
    tp, fp, fn, tn = contingency(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")

    out = {
        "samples": int(len(y_true)),
        "event_prevalence": float(np.mean(y_true)),
        "threshold_used": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "csi": float(csi), "pod": float(pod), "far": float(far),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": calc_ece(y_true, y_prob),
    }
    # PR-AUC is undefined without both classes present.
    if len(np.unique(y_true)) > 1:
        pr, rc, _ = precision_recall_curve(y_true, y_prob)
        out["pr_auc"] = float(auc(rc, pr))
        out["pr_auc_baseline_no_skill"] = float(np.mean(y_true))
    else:
        out["pr_auc"] = None
        out["pr_auc_baseline_no_skill"] = None
        out["note"] = "PR-AUC undefined: only one class present in the truth sample."
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=24,
                    help="Maximum simulated NOW instants to attempt.")
    ap.add_argument("--start", default="2024-05-01",
                    help="Earliest simulated NOW date (YYYY-MM-DD).")
    ap.add_argument("--end", default="2024-10-30",
                    help="Latest simulated NOW date (YYYY-MM-DD).")
    # 91h, not 72h: a stride that is a MULTIPLE OF 24 pins every simulated NOW
    # to the same hour of day, so each lead always verifies at the same local
    # time. Convection over West Bengal has a strong diurnal cycle, so that
    # makes later leads sample stormier hours and produces the physically
    # impossible result that skill RISES with lead time (measured with a 96h
    # stride: 1.84x at +8h climbing to 3.38x at +16h, with observed prevalence
    # climbing 0.046 -> 0.104 in lockstep). A stride coprime with 24 rotates
    # NOW through the diurnal cycle and removes the alias.
    ap.add_argument("--step-hours", type=int, default=91,
                    help="Spacing between simulated NOW instants. Should be "
                         "coprime with 24, or every case lands at the same "
                         "hour of day and the diurnal cycle aliases into the "
                         "per-lead scores.")
    ap.add_argument("--out", default="reports/gfs_production_backtest.json")
    # Opt-in raw dump. The aggregate JSON stores scored metrics at ONE threshold,
    # which cannot support a dense threshold sweep, a reliability diagram, or a
    # recalibration audit. This writes the concatenated held-out (y_true, y_prob)
    # per horizon so those analyses run on the SAME arrays the metrics above were
    # computed from -- no re-inference, no re-sampling, no separate protocol.
    ap.add_argument("--dump-raw", default=None,
                    help="Optional .npz path: per-horizon y_true/y_prob arrays.")
    args = ap.parse_args()

    if args.step_hours % 24 == 0:
        print(f"\nWARNING: --step-hours {args.step_hours} is a multiple of 24. "
              "Every simulated NOW will fall at the same hour of day, so each "
              "forecast lead verifies at a FIXED local time. With a diurnal "
              "convection cycle this aliases into the per-horizon metrics and "
              "can make skill appear to increase with lead time. Use a stride "
              "coprime with 24 (e.g. 91) for a comparable measurement.\n")

    truth_index, severe, truth_times = load_truth()
    truth_set = set(truth_index)
    print(f"[truth] ERA5 target cache: {len(truth_times)} timestamps, "
          f"{min(truth_times).isoformat()} .. {max(truth_times).isoformat()}")

    svc = get_nowcast_service()
    leads = svc.lead_times
    thresholds = svc.predictor.threshold_per_lead or {}

    # Candidate simulated NOW instants across the window where the retained GFS
    # archive and the ERA5 verification cache actually overlap. Only instants
    # whose +2/+4/+6 valid times ALL exist in the verification cache qualify.
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    candidates = []
    T = start
    while T <= end and len(candidates) < args.max_cases:
        # Anchor to a 6-hourly boundary + a deliberately non-zero offset so the
        # backtest also exercises the "NOW is not a cycle hour" path.
        # Verify against the horizons THIS model actually produces, not a fixed
        # (2, 4, 6). A long-lead checkpoint emits 8-16h, whose targets sit
        # further ahead in the verification cache; requiring 2/4/6 would have
        # selected instants that cannot score those leads at all.
        if all((T + timedelta(hours=L)) in truth_set
               for L in leads if L != 0):
            candidates.append(T)
        T += timedelta(hours=args.step_hours)

    if not candidates:
        print("\nBLOCKER: no simulated NOW instant has BOTH a fetchable GFS cycle "
              "window and ERA5 verification targets at +2/+4/+6h.")
        print("The local ERA5 target cache covers "
              f"{min(truth_times).date()} .. {max(truth_times).date()}, "
              "while the public GFS archive only retains recent cycles. "
              "They do not overlap, so a real GFS production backtest cannot be "
              "run from this machine's data. Reporting this rather than "
              "substituting a weaker experiment.")
        result = {
            "status": "blocked",
            "reason": "no overlap between retained GFS archive and ERA5 verification cache",
            "era5_truth_range": [min(truth_times).isoformat(), max(truth_times).isoformat()],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")
        return 0

    print(f"[plan] {len(candidates)} simulated NOW instants to attempt "
          f"({candidates[-1].isoformat()} .. {candidates[0].isoformat()})")

    # Score the horizons THIS checkpoint emits. Hardcoding (2, 4, 6) meant a
    # long-lead model (8-16h) raised "ValueError: 2 is not in list" on the
    # first case, because leads.index(2) does not exist for it.
    scored_leads = [L for L in leads if L != 0]
    acc = {L: {"y_true": [], "y_prob": []} for L in scored_leads}
    cases = []

    for T in candidates:
        # The selection rule is the production one: no cycle newer than what
        # production latency says existed at T.
        avail = gfs_live._cycles_available_at(T, gfs_live.GFS_PRODUCTION_LAG_HOURS)
        print(f"\n[case] simulated NOW = {T.isoformat()}  "
              f"(newest genuinely-available cycle: {avail[0].isoformat() if avail else 'none'})")

        gfs_live._HARMONIZED_CACHE = None  # force a real fetch per case
        try:
            ok = svc.refresh_live_state(target_t0=T)
        except Exception as e:  # noqa: BLE001
            print(f"   -> pipeline raised: {e}")
            cases.append({"simulated_now": T.isoformat(), "status": "error", "error": str(e)})
            continue
        if not ok or svc.live_pred is None:
            print(f"   -> pipeline failed: {svc.live_fetch_error}")
            cases.append({"simulated_now": T.isoformat(), "status": "failed",
                          "error": svc.live_fetch_error})
            continue

        analysis_t = svc.live_analysis_time
        # Hard guarantee: the analysis used must not be from T's future.
        assert datetime.fromisoformat(analysis_t) <= T, (
            f"LEAKAGE: analysis {analysis_t} is after simulated NOW {T.isoformat()}"
        )
        print(f"   -> analysis used: {analysis_t} (lag {(T - datetime.fromisoformat(analysis_t)).total_seconds()/3600:.1f}h)")

        case = {"simulated_now": T.isoformat(), "status": "ok",
                "analysis_time": analysis_t, "horizons": {}}

        for L in scored_leads:
            vt = T + timedelta(hours=L)
            li = leads.index(L)
            y_prob = np.asarray(svc.live_pred["severe_weather_prob"][li]).ravel()
            y_true = (np.asarray(severe[truth_index[vt]]) > 0).astype(int).ravel()
            if y_true.shape != y_prob.shape:
                print(f"   -> +{L}h shape mismatch {y_true.shape} vs {y_prob.shape}; skipped")
                continue
            acc[L]["y_true"].append(y_true)
            acc[L]["y_prob"].append(y_prob)
            case["horizons"][str(L)] = {
                "valid_time": vt.isoformat(),
                "cells": int(y_true.size),
                "observed_event_rate": float(y_true.mean()),
                "mean_predicted_prob": float(y_prob.mean()),
            }
            print(f"   -> +{L}h valid {vt.isoformat()}  "
                  f"truth rate={y_true.mean():.4f}  mean p={y_prob.mean():.4f}")
        cases.append(case)

    print("\n" + "=" * 64)
    print("HISTORICAL GFS PRODUCTION BACKTEST RESULTS")
    print("=" * 64)

    n_ok = sum(1 for c in cases if c["status"] == "ok")
    print(f"\nSimulated NOW instants attempted : {len(cases)}")
    print(f"Completed through production path: {n_ok}")

    metrics = {}
    for L in scored_leads:
        if not acc[L]["y_true"]:
            continue
        y_true = np.concatenate(acc[L]["y_true"])
        y_prob = np.concatenate(acc[L]["y_prob"])
        thr = float(thresholds.get(L, thresholds.get(str(L), 0.5)))
        m = score(y_true, y_prob, thr)
        metrics[str(L)] = m
        print(f"\n--- HORIZON +{L}h ---")
        print(f"  Simulated cases    : {n_ok}")
        print(f"  Samples (cells)    : {m['samples']}")
        print(f"  Event prevalence   : {m['event_prevalence']:.4f}")
        print(f"  Threshold (calib.) : {m['threshold_used']:.4f}")
        print(f"  CSI                : {m['csi']:.4f}")
        print(f"  POD                : {m['pod']:.4f}")
        print(f"  FAR                : {m['far']:.4f}")
        pa = m["pr_auc"]
        print(f"  PR-AUC             : {pa:.4f}" if pa is not None else "  PR-AUC             : undefined")
        if m.get("pr_auc_baseline_no_skill") is not None:
            print(f"    (no-skill baseline: {m['pr_auc_baseline_no_skill']:.4f})")
        print(f"  Brier score        : {m['brier_score']:.4f}")
        print(f"  ECE (calibration)  : {m['ece']:.4f}")

    result = {
        "status": "ok" if n_ok else "no_successful_cases",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "cycle_selection": "production rule: only cycles published by the simulated NOW, "
                               f"assuming {gfs_live.GFS_PRODUCTION_LAG_HOURS}h production latency",
            "input": "real GFS f000 analyses only; no forecast hour used as an input timestep",
            "inference_path": "NowcastService.refresh_live_state (the production path)",
            "scored_variable": "severe_weather_prob (calibrated severe-weather probability)",
            "truth": "ERA5-derived severe_weather proxy target at the valid time",
            "threshold": "per-lead calibrated threshold from the checkpoint",
        },
        "caveats": [
            "Truth labels are ERA5-derived PROXIES, not observed severe-weather reports.",
            "Inputs are GFS while truth is ERA5, so measured error includes GFS-vs-ERA5 "
            "representation differences as well as model error.",
            "Sample size is limited by GFS archive retention; N is reported per horizon "
            "and confidence intervals are correspondingly wide.",
            "This is NOT a claim of operational accuracy against observed severe weather.",
        ],
        "cases_attempted": len(cases),
        "cases_completed": n_ok,
        "per_horizon": metrics,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")

    if args.dump_raw:
        raw = {}
        for L in scored_leads:
            if not acc[L]["y_true"]:
                continue
            raw[f"y_true_{L}"] = np.concatenate(acc[L]["y_true"])
            raw[f"y_prob_{L}"] = np.concatenate(acc[L]["y_prob"])
        if raw:
            d = os.path.dirname(args.dump_raw)
            if d:
                os.makedirs(d, exist_ok=True)
            np.savez_compressed(args.dump_raw, **raw)
            print(f"Wrote raw arrays -> {args.dump_raw} ({len(raw) // 2} horizons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

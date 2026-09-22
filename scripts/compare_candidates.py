"""Fair side-by-side comparison of candidate checkpoints on held-out GFS.

Section L of the FAR-reduction phase. Every candidate is scored through the SAME
protocol -- same simulated NOW instants, same 91h stride, same ERA5-derived truth
at the valid time, same production inference path -- so the numbers are
comparable. Anything that differs between candidates is the model, not the
measurement.

A candidate is one (name, checkpoint, config) triple. For each, this drives
`historical_backtest.py` with STORMSENSE_CKPT/STORMSENSE_CONFIG so inference goes
through `NowcastService`, not a reimplementation, then reports per lead:

    base rate, PR-AUC, skill vs no-skill, CSI, POD, FAR, precision, F1,
    Brier, ECE, TP/FP/FN/TN

Comparing models with DIFFERENT lead sets is handled honestly: leads are reported
on their own terms and only leads present in both candidates are put side by side.
A v2 (2-6h) and a v4 (4-16h) model overlap only at 4-6h, and pretending otherwise
would compare a 2h forecast against a 12h one.

Usage:
    python scripts/compare_candidates.py \
        --candidate "v3:Data/outputs/checkpoints_v3/v2_best.pt:configs/v3_longlead.yaml" \
        --candidate "v4:Data/outputs/checkpoints_v4/v2_best.pt:configs/v4_wallclock.yaml" \
        --max-cases 14 --out reports/candidate_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_backtest(name, ckpt, cfg, max_cases, step_hours, start, end, out_json, raw_npz):
    env = dict(os.environ)
    env["STORMSENSE_DEVICE"] = "cpu"
    env["STORMSENSE_CKPT"] = ckpt
    env["STORMSENSE_CONFIG"] = cfg
    cmd = [
        sys.executable, "-u", os.path.join("scripts", "historical_backtest.py"),
        "--max-cases", str(max_cases), "--step-hours", str(step_hours),
        "--start", start, "--end", end, "--out", out_json,
    ]
    if raw_npz:
        cmd += ["--dump-raw", raw_npz]
    print(f"\n[{name}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] FAILED rc={r.returncode}")
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return None
    # surface the per-case lines so a reader can see the cases were real
    tail = [l for l in r.stdout.splitlines() if "analysis used" in l or "HORIZON" in l]
    print(f"[{name}] completed; {len(tail)} progress lines")
    with open(os.path.join(REPO, out_json), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True,
                    help="name:checkpoint:config (repeatable)")
    ap.add_argument("--max-cases", type=int, default=14)
    ap.add_argument("--step-hours", type=int, default=91,
                    help="Must NOT be a multiple of 24 (diurnal aliasing).")
    ap.add_argument("--start", default="2024-05-01")
    ap.add_argument("--end", default="2024-10-30")
    ap.add_argument("--raw-dir", default=None,
                    help="If set, dump per-candidate raw arrays here.")
    ap.add_argument("--out", default="reports/candidate_comparison.json")
    a = ap.parse_args()

    if a.step_hours % 24 == 0:
        raise SystemExit(
            f"--step-hours {a.step_hours} is a multiple of 24, which pins every "
            "simulated NOW to the same hour of day and aliases the diurnal cycle. "
            "Use 91."
        )

    results = {}
    for spec in a.candidate:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad --candidate {spec!r}; want name:checkpoint:config")
        name, ckpt, cfg = parts
        for p in (ckpt, cfg):
            if not os.path.exists(os.path.join(REPO, p)):
                raise SystemExit(f"[{name}] missing: {p}")
        out_json = f"reports/_cmp_{name}.json"
        raw = os.path.join(a.raw_dir, f"raw_{name}.npz") if a.raw_dir else None
        res = run_backtest(name, ckpt, cfg, a.max_cases, a.step_hours,
                           a.start, a.end, out_json, raw)
        if res:
            results[name] = {"checkpoint": ckpt, "config": cfg,
                             "per_horizon": res.get("per_horizon", {}),
                             "cases_completed": res.get("cases_completed"),
                             "backtest_json": out_json}

    if not results:
        raise SystemExit("no candidate completed")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("CANDIDATE COMPARISON -- identical held-out GFS protocol")
    print("=" * 78)
    for name, r in results.items():
        print(f"\n### {name}  ({r['checkpoint']})  cases={r['cases_completed']}")
        print(f"{'lead':>5} {'base':>7} {'PR-AUC':>8} {'skill':>7} {'CSI':>7} "
              f"{'POD':>6} {'FAR':>6} {'prec':>6} {'F1':>6} {'Brier':>7} {'ECE':>7}")
        for L in sorted(r["per_horizon"], key=lambda x: int(x)):
            m = r["per_horizon"][L]
            pa, nb = m.get("pr_auc"), m.get("pr_auc_baseline_no_skill")
            skill = (pa / nb) if (pa and nb) else float("nan")
            tp, fp = m.get("tp", 0), m.get("fp", 0)
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            pod = m.get("pod", float("nan"))
            f1 = (2 * prec * pod / (prec + pod)) if (prec == prec and pod == pod and (prec + pod)) else float("nan")
            print(f"{L:>5} {m.get('event_prevalence', 0):7.4f} {(pa or 0):8.4f} {skill:6.2f}x "
                  f"{m.get('csi', 0):7.4f} {pod:6.3f} {m.get('far', 0):6.3f} {prec:6.3f} "
                  f"{f1:6.3f} {m.get('brier_score', 0):7.4f} {m.get('ece', 0):7.4f}")

    names = list(results)
    if len(names) >= 2:
        shared = set(results[names[0]]["per_horizon"])
        for n in names[1:]:
            shared &= set(results[n]["per_horizon"])
        print("\n--- leads present in EVERY candidate (the only fair head-to-head) ---")
        if shared:
            for L in sorted(shared, key=lambda x: int(x)):
                row = f"  +{L}h  "
                for n in names:
                    m = results[n]["per_horizon"][L]
                    pa, nb = m.get("pr_auc"), m.get("pr_auc_baseline_no_skill")
                    sk = (pa / nb) if (pa and nb) else float("nan")
                    row += f"{n}: skill={sk:.2f}x FAR={m.get('far', 0):.3f} CSI={m.get('csi', 0):.4f}   "
                print(row)
        else:
            print("  none -- the candidates' lead sets do not overlap, so no lead can be "
                  "compared directly. Report them separately.")
        for n in names:
            print(f"  {n} leads: {sorted(results[n]['per_horizon'], key=lambda x: int(x))}")

    payload = {
        "protocol": {
            "step_hours": a.step_hours, "max_cases": a.max_cases,
            "window": [a.start, a.end],
            "truth": "ERA5-derived severe_weather proxy at the valid time",
            "inference": "NowcastService.refresh_live_state (production path)",
            "note": "Identical cases/stride/truth for every candidate.",
        },
        "candidates": results,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(os.path.join(REPO, a.out), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

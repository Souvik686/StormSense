#!/usr/bin/env bash
# Post-training pipeline for the v4 wall-clock candidate.
#
# Runs, in the only order that keeps the evaluation honest:
#   1. ERA5-validation temperature scaling      (val split only, never test)
#   2. decision thresholds on CALIBRATED probs  (val split only, never test)
#   3. ERA5 test-split per-lead evaluation      (held out)
#   4. GFS backtest on 2023                     (the GFS-domain VALIDATION year)
#   5. GFS-domain recalibration fitted on 2023, reported on 2024
#   6. GFS backtest on 2024                     (held out, never fitted on)
#   7. dense threshold sweep + reliability on the 2024 arrays
#
# Nothing here touches Data/outputs/checkpoints/v2_calibrated_best.pt.
# Step 1 and 2 DO write into the v4 candidate checkpoint, which is the intended
# place for its own calibration; production is a different file.
#
# Usage:  bash scripts/finalize_v4.sh [scratch_dir]
set -u

CKPT="Data/outputs/checkpoints_v4/v2_best.pt"
CFG="configs/v4_wallclock.yaml"
SCRATCH="${1:-${TMPDIR:-/tmp}}"
mkdir -p "$SCRATCH" reports

if [ ! -f "$CKPT" ]; then
  echo "FATAL: $CKPT not found -- has training produced a checkpoint yet?" >&2
  exit 1
fi

export STORMSENSE_DEVICE=cpu

step() { echo; echo "============ $* ============"; }

step "1/7 temperature scaling on the ERA5 validation split"
python scripts/calibrate_v3.py --config "$CFG" --checkpoint "$CKPT" \
  --out reports/v4_calibration.json || echo "WARN: calibration step failed"

step "2/7 decision thresholds on calibrated probabilities (val split)"
python scripts/optimize_threshold.py --config "$CFG" --checkpoint "$CKPT" \
  --objective csi || echo "WARN: threshold step failed"

step "3/7 per-lead evaluation on the held-out ERA5 test split"
python scripts/evaluate_leads_v3.py --config "$CFG" --checkpoint "$CKPT" \
  --split test --out reports/v4_lead_evaluation.json || echo "WARN: eval step failed"

step "4/7 GFS backtest on 2023 (GFS-domain validation year)"
STORMSENSE_CKPT="$CKPT" STORMSENSE_CONFIG="$CFG" \
python -u scripts/historical_backtest.py --max-cases 16 --step-hours 91 \
  --start 2023-05-01 --end 2023-10-25 \
  --out reports/v4_gfs_VAL2023.json \
  --dump-raw "$SCRATCH/raw_v4_val2023.npz" || echo "WARN: 2023 backtest failed"

step "6/7 GFS backtest on 2024 (held out)"
STORMSENSE_CKPT="$CKPT" STORMSENSE_CONFIG="$CFG" \
python -u scripts/historical_backtest.py --max-cases 14 --step-hours 91 \
  --start 2024-05-01 --end 2024-10-30 \
  --out reports/v4_gfs_TEST2024.json \
  --dump-raw "$SCRATCH/raw_v4_test2024.npz" || echo "WARN: 2024 backtest failed"

step "5/7 GFS-domain recalibration (fit 2023, report 2024)"
python scripts/calibrate_gfs_domain.py \
  --val-raw "$SCRATCH/raw_v4_val2023.npz" \
  --test-raw "$SCRATCH/raw_v4_test2024.npz" \
  --out reports/v4_gfs_domain_calibration.json || echo "WARN: GFS recalibration failed"

step "7/7 dense threshold sweep + reliability on the held-out 2024 arrays"
python scripts/far_analysis.py --raw "$SCRATCH/raw_v4_test2024.npz" \
  --out reports/v4_far_sweep.json --label "v4 hourly 4-16h" \
  || echo "WARN: sweep failed"

echo
echo "Done. Artifacts:"
ls -1 reports/v4_*.json 2>/dev/null
echo
echo "Next: scripts/compare_candidates.py for the v2/v3/v4 head-to-head."

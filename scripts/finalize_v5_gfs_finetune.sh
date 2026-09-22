#!/usr/bin/env bash
# GFS-domain fine-tuning: build the hourly cache, fine-tune from v4, evaluate.
#
# WHAT THIS TESTS
#
# The model is trained on ERA5 and served on GFS. Paired over 54 cycles at
# identical valid times, three inputs are badly shifted -- cin -1.717 training sd,
# d2m -0.839, cape -0.827 (ERA5 mean 1450 J/kg vs GFS 760). This is the largest
# identified reason live skill is roughly half the ERA5-input figure.
#
# Hypothesis: fine-tuning on GFS INPUT with the UNCHANGED ERA5 LABELS recovers
# part of that gap. It is a hypothesis, not an assumption -- the held-out 2024 GFS
# backtest at the end decides it.
#
# SAFETY
#   * production checkpoint untouched (separate checkpoint_dir)
#   * v4 untouched (--init-from copies weights, never writes back)
#   * labels are the production ERA5 targets, never recomputed from GFS
#   * the 2024 GFS cases used for the final comparison are never trained on
#
# Usage:  bash scripts/finalize_v5_gfs_finetune.sh
set -u

CFG="configs/v5_gfs_finetune.yaml"
INIT="Data/outputs/checkpoints_v4/v2_best.pt"
CKPT="Data/outputs/checkpoints_v5_gfsft/v2_best.pt"
RAW="reports/raw"
mkdir -p "$RAW" reports

export STORMSENSE_DEVICE=cpu
step() { echo; echo "======== $* ========"; }

if [ ! -f "$INIT" ]; then
  echo "FATAL: $INIT missing -- v4 must finish training first." >&2
  exit 1
fi

step "1/6 build the HOURLY GFS cache (reuses cached pickles, no re-download)"
# --hourly is load-bearing: build_windows treats a lead as a ROW OFFSET, so on a
# 6-hourly axis "lead 8" would mean 48 hours. See the script's docstring.
python -u scripts/build_gfs_training_cache.py \
  --start 2021-05-01 --end 2022-10-31 --stride 2 --hourly \
  --out processed/cache/gfs_train_hourly || { echo "FATAL: cache build failed"; exit 1; }

step "2/6 fine-tune from v4 (weights only, fresh optimizer at the config LR)"
python -u scripts/train_v2.py --config "$CFG" --init-from "$INIT" \
  || echo "WARN: fine-tuning exited non-zero"

if [ ! -f "$CKPT" ]; then
  echo "FATAL: fine-tuning produced no checkpoint at $CKPT" >&2
  exit 1
fi

step "3/6 temperature scaling on the fine-tune VALIDATION split"
python scripts/calibrate_v3.py --config "$CFG" --checkpoint "$CKPT" \
  --out reports/v5_calibration.json || echo "WARN: calibration failed"

step "4/6 GFS backtest on 2023 (threshold/calibration fitting year)"
STORMSENSE_CKPT="$CKPT" STORMSENSE_CONFIG="$CFG" \
python -u scripts/historical_backtest.py --max-cases 16 --step-hours 91 \
  --start 2023-05-01 --end 2023-10-25 \
  --out reports/v5_gfs_VAL2023.json \
  --dump-raw "$RAW/v5_val2023.npz" || echo "WARN: 2023 backtest failed"

step "5/6 GFS backtest on 2024 (HELD OUT -- never fitted on)"
STORMSENSE_CKPT="$CKPT" STORMSENSE_CONFIG="$CFG" \
python -u scripts/historical_backtest.py --max-cases 14 --step-hours 91 \
  --start 2024-05-01 --end 2024-10-30 \
  --out reports/v5_gfs_TEST2024.json \
  --dump-raw "$RAW/v5_test2024.npz" || echo "WARN: 2024 backtest failed"

step "6/6 fit thresholds on 2023, apply to 2024; then the dense sweep"
python scripts/fit_gfs_thresholds.py \
  --val-raw "$RAW/v5_val2023.npz" --test-raw "$RAW/v5_test2024.npz" \
  --objective csi --out reports/v5_gfs_threshold_fit.json \
  || echo "WARN: threshold fit failed"
python scripts/far_analysis.py --raw "$RAW/v5_test2024.npz" \
  --out reports/v5_far_sweep.json --label "v5 GFS-finetuned" \
  || echo "WARN: sweep failed"

echo
echo "Done. Compare against v2/v3/v4 on the same protocol:"
echo "  python scripts/compare_candidates.py \\"
echo "    --candidate \"v4:Data/outputs/checkpoints_v4/v2_best.pt:configs/v4_wallclock.yaml\" \\"
echo "    --candidate \"v5:$CKPT:$CFG\" --max-cases 14"
ls -1 reports/v5_*.json 2>/dev/null

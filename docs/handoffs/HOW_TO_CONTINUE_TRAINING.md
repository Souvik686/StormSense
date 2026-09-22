# How to continue StormSense training yourself

Everything below is copy-pasteable from the repo root (`D:\Weather-Hackathon`).
Run commands in **Git Bash** (or use PowerShell equivalents where noted).

---

## 0. Where things stand

| thing | value |
|---|---|
| Candidate model | `Data/outputs/checkpoints_v4/v2_best.pt` — 13 hourly leads (4–16 h) |
| Best so far | epoch 9, validation PR-AUC **0.2172**, CSI 0.1632 |
| Production (do not overwrite) | `Data/outputs/checkpoints/v2_calibrated_best.pt` |
| Config | `configs/v4_wallclock.yaml` |
| Speed | ~20 min/epoch, CPU only |

`v2_best.pt` always holds the **best-validated** weights. `v2_last.pt` holds the
**most recent** epoch plus the optimizer/scheduler state needed to resume.

---

## 1. Is it still running?

```bash
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*train_v2*'} | Select-Object ProcessId,CommandLine | Format-List"
```

Check progress at any time:

```bash
python -c "
import torch
b=torch.load('Data/outputs/checkpoints_v4/v2_best.pt',map_location='cpu',weights_only=False)
l=torch.load('Data/outputs/checkpoints_v4/v2_last.pt',map_location='cpu',weights_only=False)
print('best epoch',b['epoch']+1,'pr_auc=%.4f'%b.get('best_val_pr_auc',0),'| last epoch',l['epoch']+1)
"
```

## 2. Stop it

```bash
powershell -NoProfile -Command "Stop-Process -Id <PID> -Force"
```

Nothing is lost: checkpoints are written every epoch, so you lose at most the
epoch in progress.

## 3. Resume later — days later is fine

```bash
python scripts/train_v2.py --config configs/v4_wallclock.yaml \
  --resume-from Data/outputs/checkpoints_v4/v2_last.pt --eval-test
```

This continues the **same** run: the optimizer state, the cosine LR schedule
position, the epoch counter, the best score and the early-stopping patience
counter all come back from the checkpoint. It is not a fresh start.

Run it in the background so closing the terminal does not kill it:

```bash
nohup python -u scripts/train_v2.py --config configs/v4_wallclock.yaml \
  --resume-from Data/outputs/checkpoints_v4/v2_last.pt --eval-test \
  > train_v4_resume.log 2>&1 &
tail -f train_v4_resume.log
```

### When does it stop on its own?
`epochs: 60` is a ceiling; `early_stopping_patience: 8` is what actually stops
it — 8 consecutive epochs with no new best. Gains are already small
(0.2032 → 0.2066 → 0.2088 → 0.2172), so expect it to stop around epoch 14–20.

---

## 4. **Do this before using the model anywhere**

`v2_best.pt` comes out of training with **no calibration**, so its outputs are raw
sigmoid values, not probabilities. They drift upward with lead time (0.078 at
+5 h to 0.186 at +16 h), which is an artefact, not weather. A dashboard built on
them would show wrong percentages.

```bash
# 1) temperature scaling, fitted on the VALIDATION split only
STORMSENSE_DEVICE=cpu python scripts/calibrate_v3.py \
  --config configs/v4_wallclock.yaml \
  --checkpoint Data/outputs/checkpoints_v4/v2_best.pt \
  --out reports/v4_calibration.json

# 2) decision thresholds, on the calibrated probabilities, validation only
STORMSENSE_DEVICE=cpu python scripts/optimize_threshold.py \
  --config configs/v4_wallclock.yaml \
  --checkpoint Data/outputs/checkpoints_v4/v2_best.pt \
  --objective csi
```

Verify it took:

```bash
python -c "
import torch
c=torch.load('Data/outputs/checkpoints_v4/v2_best.pt',map_location='cpu',weights_only=False)
print('calibrated:',c.get('is_calibrated'))
print('temperatures:',c.get('temperature_per_lead'))
print('thresholds:',c.get('threshold_per_lead'))
"
```

Or just run the whole post-training pipeline in one go:

```bash
bash scripts/finalize_v4.sh
```

which does calibration → thresholds → ERA5 test evaluation → 2023 GFS backtest →
2024 held-out GFS backtest → GFS threshold refit → dense threshold sweep.

---

## 5. Measure it honestly on GFS

ERA5-input numbers are laboratory numbers. Production runs on GFS, where skill is
roughly half. The real measurement:

```bash
mkdir -p reports/raw

# validation year (for fitting thresholds)
STORMSENSE_DEVICE=cpu \
STORMSENSE_CKPT=Data/outputs/checkpoints_v4/v2_best.pt \
STORMSENSE_CONFIG=configs/v4_wallclock.yaml \
python -u scripts/historical_backtest.py --max-cases 16 --step-hours 91 \
  --start 2023-05-01 --end 2023-10-25 \
  --out reports/v4_gfs_VAL2023.json --dump-raw reports/raw/v4_val2023.npz

# held-out test year -- never fit anything on this
STORMSENSE_DEVICE=cpu \
STORMSENSE_CKPT=Data/outputs/checkpoints_v4/v2_best.pt \
STORMSENSE_CONFIG=configs/v4_wallclock.yaml \
python -u scripts/historical_backtest.py --max-cases 14 --step-hours 91 \
  --start 2024-05-01 --end 2024-10-30 \
  --out reports/v4_gfs_TEST2024.json --dump-raw reports/raw/v4_test2024.npz

# fit thresholds on 2023, report on 2024
python scripts/fit_gfs_thresholds.py \
  --val-raw reports/raw/v4_val2023.npz \
  --test-raw reports/raw/v4_test2024.npz \
  --objective csi --out reports/v4_gfs_threshold_fit.json
```

**`--step-hours` must not be a multiple of 24.** A stride of 72 or 96 pins every
simulated "now" to the same hour of day, aliases the diurnal convection cycle, and
**inverts** the per-lead skill ranking. Use 91.

---

## 6. Compare against the current models

```bash
python scripts/compare_candidates.py \
  --candidate "v2:Data/outputs/checkpoints/v2_calibrated_best.pt:configs/default.yaml" \
  --candidate "v3:Data/outputs/checkpoints_v3/v2_best.pt:configs/v3_longlead.yaml" \
  --candidate "v4:Data/outputs/checkpoints_v4/v2_best.pt:configs/v4_wallclock.yaml" \
  --max-cases 14 --out reports/candidate_comparison.json
```

Promotion rules are written down in `reports/V4_DECISION_CRITERIA.md` — they were
fixed **before** any result was known, which is the only way they cannot be bent
to fit the outcome. Ties go to the incumbent.

---

## 7. Optional: the GFS-domain fine-tune (v5)

The largest untapped improvement. The model trains on ERA5 but runs on GFS, where
CAPE is roughly half (ERA5 mean 1450 J/kg vs GFS 760) and CIN is shifted by 1.7
training standard deviations.

```bash
# needs the GFS cache first; --hourly is NOT optional
python -u scripts/build_gfs_training_cache.py \
  --start 2021-05-01 --end 2022-10-31 --stride 2 --hourly \
  --out processed/cache/gfs_train_hourly

bash scripts/finalize_v5_gfs_finetune.sh
```

Already-downloaded cycles are cached as pickles under `processed/cache/gfs/`, so
re-running costs no extra downloads.

**Why `--hourly` matters:** `build_windows` treats a lead as a *row offset*. On a
6-hourly axis "lead 8" would mean 48 hours, so the model would silently learn
24–96 hour forecasts while everything still labelled them 4–16 hours.

---

## 8. Rules that protect the project

* **Never overwrite** `Data/outputs/checkpoints/v2_calibrated_best.pt`
  (md5 `6afdc4d6c1c1dc7ae1901bee794eb543`). Experiments use their own directory.
* **Never fit calibration or thresholds on the test split.** Measured here: doing
  so suggested +8.8 % CSI when the honest gain was +4.5 %. Half of it was noise.
* **Never use `--step-hours` divisible by 24.**
* Run the tests after any change: `python -m pytest tests/ -q --ignore=tests/test_browser_e2e.py`
  (413 should pass).
* Do not edit `.gitignore`.

---

## 9. If you get a GPU

Everything above is CPU-bound at ~20 min/epoch. On a CUDA GPU it is roughly
30 s/epoch — the same run finishes in minutes instead of hours, and
`mixed_precision: true` in the config starts working. No code change needed;
`train_v2.py` already selects CUDA when available.

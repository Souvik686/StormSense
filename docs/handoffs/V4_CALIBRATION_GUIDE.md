# Calibrating and evaluating the v4 candidate

## Why

v4 (`Data/outputs/checkpoints_v4/v2_best.pt`, config `configs/v4_wallclock.yaml`)
already beats v2 on CSI and FAR at every lead they both cover (4h/5h/6h), per
`reports/candidate_comparison.json`. But its raw output is **uncalibrated**:
probabilities rise from 0.078 at +5h to 0.186 at +16h regardless of actual
risk, which is physically backwards and would show misleading numbers if
deployed as-is. Calibration fixes that reliability problem. It does **not**
change detection skill (CSI/FAR) — those are already what they are.

This does not touch the live site. Production stays on
`Data/outputs/checkpoints/v2_calibrated_best.pt` throughout every step below.

## What you need

- Your existing venv — nothing new to install
- CPU only — the script forces `STORMSENSE_DEVICE=cpu`, no GPU required
- Internet access for two steps (GFS backtests fetch real NOAA data)
- Git Bash (already your shell) — the script is bash, not PowerShell

## Run it

```bash
cd /d/Weather-Hackathon
bash scripts/finalize_v4.sh
```

Optionally pass a scratch directory for intermediate `.npz` dumps:

```bash
bash scripts/finalize_v4.sh /path/to/scratch
```

Each of the 7 steps is wrapped so one failure prints a warning and the script
keeps going rather than stopping cold. At the end it lists whatever
`reports/v4_*.json` files it actually produced.

## What each step does

| # | Step | Reads | Writes | Needs internet |
|---|------|-------|--------|-----------------|
| 1 | Temperature scaling on ERA5 validation split | v4 checkpoint | into the v4 checkpoint itself | No |
| 2 | Decision thresholds on calibrated probabilities | v4 checkpoint (now calibrated) | into the v4 checkpoint itself | No |
| 3 | Per-lead evaluation, held-out ERA5 test split | v4 checkpoint | `reports/v4_lead_evaluation.json` | No |
| 4 | GFS backtest on 2023 (GFS-domain validation year) | live NOAA GFS archive | `reports/v4_gfs_VAL2023.json` | Yes |
| 5 | GFS-domain recalibration, fit 2023 / report 2024 | the two backtests' raw arrays | `reports/v4_gfs_domain_calibration.json` | No |
| 6 | GFS backtest on 2024 (held out, never fitted on) | live NOAA GFS archive | `reports/v4_gfs_TEST2024.json` | Yes |
| 7 | Dense threshold sweep + reliability diagram | 2024 raw arrays | `reports/v4_far_sweep.json` | No |

Steps 1 and 2 write directly into the v4 checkpoint file — that's the
intended, one-time place for a candidate to carry its own calibration,
exactly how `v2_calibrated_best.pt` already works.

## Timing

- Steps 1–3, 5, 7: fast — CPU inference on a small (~782K param) model.
  Minutes, not hours.
- Steps 4 and 6: depend on your network speed fetching GRIB files from NOAA's
  archive (16 and 14 cases respectively, ~91h stride). Could be several
  minutes to longer — this is the same kind of fetch you already saw take a
  while earlier in this project.

## After it finishes

Read the printed file list, then look at:

- `reports/v4_lead_evaluation.json` — sanity-check ERA5-side per-lead metrics
  didn't regress from calibration (they shouldn't — calibration is monotonic).
- `reports/v4_gfs_TEST2024.json` — the real, held-out, live-GFS-domain
  backtest. This is the number that matters for "is v4 actually good."
- `reports/v4_far_sweep.json` — reliability diagram data, confirms the
  calibration fix actually worked (probabilities should no longer drift
  upward with lead for no reason).

Then run the same head-to-head comparison already used for v2/v3/v4 earlier:

```bash
STORMSENSE_DEVICE=cpu \
python scripts/compare_candidates.py \
  --candidate "v2:Data/outputs/checkpoints/v2_calibrated_best.pt:configs/default.yaml" \
  --candidate "v4:Data/outputs/checkpoints_v4/v2_best.pt:configs/v4_wallclock.yaml" \
  --max-cases 14 --out reports/candidate_comparison_v4_calibrated.json
```

`--step-hours` defaults inside that script to something that is **not** a
multiple of 24 already — don't override it to 72/96/etc., that aliases the
diurnal cycle and inverts the per-lead ranking (documented in
`reports/STARTUP_AND_VERIFICATION.md`).

## What this does NOT resolve

- v4 still has **no coverage at 2h or 3h** — its config starts at 4h. Calibration
  doesn't add leads; only retraining with those leads in `lead_times_hours`
  would.
- FAR stays high (currently ~0.84–0.86 at 4–6h) — calibration fixes the
  *meaning* of the number, not the false-alarm rate itself.
- This is still only a 14–16 case backtest sample. A win here is real evidence,
  not exhaustive proof across every storm type/season.

## Before deploying

Do **not** point the live server at the v4 checkpoint (via `STORMSENSE_CKPT`/
`STORMSENSE_CONFIG`) until you've reviewed the numbers above and are satisfied
with them. If you do want to test it live temporarily:

```bash
STORMSENSE_CKPT=Data/outputs/checkpoints_v4/v2_best.pt \
STORMSENSE_CONFIG=configs/v4_wallclock.yaml \
python run_server.py
```

This overrides the checkpoint for that one server process only — it never
modifies `configs/default.yaml` or touches the actual production checkpoint
file.

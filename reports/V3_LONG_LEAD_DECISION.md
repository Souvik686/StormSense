# StormSense v3 long-lead model — evaluation and deployment decision

Date: 2026-09-21
Candidate: `Data/outputs/checkpoints_v3/v2_best.pt`
Production (unchanged): `Data/outputs/checkpoints/v2_calibrated_best.pt` (md5 `6afdc4d6c1c1dc7ae1901bee794eb543`)

## What was trained

Same architecture as production — `SevereWeatherNetV2`, 781,889 parameters,
tri-stream ConvGRU, 33x25 grid at 0.25 deg. **Only the forecast leads changed**:
`[2,3,4,5,6]` -> `[8,10,12,14,16]` hours. Config: `configs/v3_longlead.yaml`.

Splits are unchanged and leakage-checked: train 2021-2022, val 2023, test 2024,
6 h buffer at each boundary (`validate_no_leakage` passes for the new leads).
Samples: 8,604 train / 4,302 val / 4,302 test. Event base rate ~2.25% at every
lead, so no class collapse at longer horizons.

Training early-stopped at epoch 17 (patience 10); best epoch 7.

## Calibration

The checkpoint came out of training with **no** temperature scaling and was badly
miscalibrated. Per-lead temperatures were fitted by LBFGS on validation BCE
(`scripts/calibrate_v3.py`), never on test.

| lead | T | ECE before | ECE after | Brier before | Brier after |
|---|---|---|---|---|---|
| +8h | 0.2855 | 0.2172 | 0.0816 | 0.0928 | 0.0592 |
| +10h | 0.2891 | 0.2120 | 0.0725 | 0.0895 | 0.0559 |
| +12h | 0.2981 | 0.2099 | 0.0714 | 0.0886 | 0.0559 |
| +14h | 0.3019 | 0.2130 | 0.0757 | 0.0902 | 0.0579 |
| +16h | 0.2828 | 0.2199 | 0.0742 | 0.0929 | 0.0572 |

Improved on 5/5 leads, and the improvement held out-of-sample on test
(ECE 0.077-0.088, Brier 0.059-0.062).

### Bug found and fixed while doing this
`scripts/optimize_threshold.py` applied `sigmoid` to **raw** logits, ignoring the
checkpoint's temperature. It therefore tuned operating points against a
different distribution from the one `predictor.py` serves. Fixed; thresholds
were refitted on calibrated probabilities.

## Test-split results (2024, held out)

| lead | PR-AUC | no-skill | skill | CSI | POD | FAR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| +8h | 0.2482 | 0.0389 | **6.38x** | 0.1761 | 0.4244 | 0.7687 | 0.0618 | 0.0876 |
| +10h | 0.2181 | 0.0388 | 5.62x | 0.1612 | 0.4316 | 0.7953 | 0.0586 | 0.0781 |
| +12h | 0.2010 | 0.0387 | 5.19x | 0.1539 | 0.4017 | 0.8003 | 0.0590 | 0.0773 |
| +14h | 0.1940 | 0.0386 | 5.02x | 0.1477 | 0.4025 | 0.8109 | 0.0615 | 0.0815 |
| +16h | 0.1979 | 0.0385 | 5.14x | 0.1487 | 0.3990 | 0.8083 | 0.0615 | 0.0810 |

Persistence baseline CSI for reference: 0.073 / 0.062 / 0.057 / 0.056 / 0.059.
Every lead beats persistence by roughly 2.4-2.6x.

## Comparison with production (same harness, same test split)

| lead | PR-AUC | skill vs no-skill | CSI |
|---|---|---|---|
| v2 +2h | 0.6417 | 16.40x | 0.4054 |
| v2 +4h | 0.4655 | 11.92x | 0.2925 |
| v2 +6h | 0.3640 | 9.35x | 0.2413 |
| **v3 +8h** | 0.2482 | 6.38x | 0.1761 |
| **v3 +16h** | 0.1979 | 5.14x | 0.1487 |

The two models are **not interchangeable**. v3 is markedly weaker than v2 at
every metric, which is expected: it forecasts 8-16 h ahead where v2 forecasts
2-6 h ahead, and forecast skill decays with lead time. v3 does not replace v2 —
it extends the horizon beyond where v2 can predict at all.

> **CORRECTION (2026-09-22) — the paragraph above does not hold in production.**
>
> Every number in the table above is **ERA5-input**. That is a valid comparison
> of the two models *on reanalysis*, but production runs on **GFS**, and v2 had
> never been backtested on GFS. It has been now
> (`reports/gfs_backtest_v2_TEST2024.json`, same 14 cases and 91 h stride as v3):
>
> | lead | FAR ERA5 → GFS | POD ERA5 → GFS | CSI ERA5 → GFS | GFS skill |
> |---|---|---|---|---|
> | v2 +2h | 0.457 → **0.924** | 0.616 → **0.122** | 0.405 → **0.049** | 1.79× |
> | v2 +4h | 0.602 → **0.904** | 0.525 → **0.157** | 0.293 → **0.063** | 1.86× |
> | v2 +6h | 0.674 → **0.860** | 0.481 → **0.163** | 0.241 → **0.081** | 1.80× |
>
> v2 is ~1.8× no-skill on GFS, not the 9–16× the ERA5 table shows. Comparing
> both models on **GFS**, with thresholds refit on 2023 and applied to
> untouched 2024 (`scripts/fit_gfs_thresholds.py`):
>
> | model | leads | mean FAR | mean POD | mean CSI |
> |---|---|---|---|---|
> | v2 | 2–6 h | 0.896 | 0.439 | 0.0897 |
> | **v3** | 8–16 h | **0.868** | 0.391 | **0.1080** |
>
> **v3 has lower FAR and higher CSI than v2 while forecasting four times further
> ahead.** The claim that v3 is "markedly weaker at every metric" was an artefact
> of comparing v3-on-GFS against v2-on-ERA5. Measured on the same input, the
> ordering reverses.
>
> The *decision* below — do not promote v3 — still stands, but for a different
> reason than the one given: v3's five coarse leads cannot serve wall-clock
> NOW/+2/+4/+6 (they cover under half the instants in a GFS cycle), which is what
> the v4 hourly-lead candidate exists to fix. The reason is no longer "v3 is
> weaker than v2 in production", because it is not.

## Decision

**Do NOT promote v3 to the production checkpoint.** Keep
`v2_calibrated_best.pt` active.

Reasons:

1. **It answers a different question.** v2 owns 2-6 h; v3 owns 8-16 h. Replacing
   v2 with v3 would delete the strongest horizons the product has (+2h at 16.4x
   no-skill) to gain weaker ones.
2. **FAR is 0.77-0.81 at every v3 lead** — roughly four in five alerts are false.
   That is real skill (5-6.4x no-skill, 2.4-2.6x persistence) but it is not
   alert-grade on its own, and must not be presented as if it were.
3. Serving both models simultaneously is a genuine architecture change
   (two checkpoints loaded, leads merged across them) that has not been built or
   regression-tested. Shipping it untested would risk the working system.

### On exposing the long leads

Per the requirement that only validated leads reach production: all five clear
no-skill and persistence, so none is statistically worthless. But none is
deployed in this pass, because the dual-model serving path does not exist yet.
`SUPPORTED_LEADS` is now derived from the loaded checkpoint
(`backend/main.py::_supported_leads`) and the frontend validates against the
same list, so whichever model is active determines the horizons offered — no
hardcoded set to fall out of sync.

If the long leads are later exposed, +8h and +10h are the defensible ones
(6.38x / 5.62x, FAR 0.77/0.80). +14h and +16h should carry an explicit
low-confidence treatment given FAR > 0.81.

## Reproducing

```
python scripts/train_v2.py       --config configs/v3_longlead.yaml --eval-test
python scripts/calibrate_v3.py   --config configs/v3_longlead.yaml --checkpoint Data/outputs/checkpoints_v3/v2_best.pt
python scripts/optimize_threshold.py --config configs/v3_longlead.yaml --checkpoint Data/outputs/checkpoints_v3/v2_best.pt --objective csi
python scripts/evaluate_leads_v3.py  --config configs/v3_longlead.yaml --checkpoint Data/outputs/checkpoints_v3/v2_best.pt --split test
```

Artifacts: `reports/v3_lead_evaluation.json`, `reports/v3_calibration.json`,
`reports/v2_baseline_lead_evaluation.json`,
`outputs/metrics/threshold_optimization.json`.

## Improvement work (2026-09-21, after the first evaluation)

### Diagnosis: the model overfits almost immediately

The v3 training curve is unambiguous. `train_loss` falls monotonically
(0.0155 -> 0.0082 over 17 epochs) while `val_loss` bottoms at **epoch 2**
(0.0159) and then climbs to 0.0214; validation `sev_loss` rises 0.0107 ->
0.0160. Best val PR-AUC was epoch 7. Everything after that was memorisation.

Cause: the network has **no regularisation** beyond `weight_decay=1e-4` -- no
dropout, no normalisation layers anywhere in `SevereWeatherNetV2`. The focal
loss (`alpha=0.90, gamma=2.5`) already handles class imbalance well, so that was
not the gap.

### Change: opt-in spatial dropout

`nn.Dropout2d` on the fused representation, before the multi-scale pyramid and
the shared decoder. Dropout2d rather than plain Dropout because the features are
spatial maps -- dropping whole channels regularises, dropping individual pixels
leaks through neighbours.

**Default is 0.0, i.e. a true `nn.Identity` no-op.** Verified: the production
checkpoint still loads with `strict=True`, parameter count is unchanged at
781,889, the `state_dict` keys are identical, and outputs are bit-identical in
both `train()` and `eval()`. Pinned by `tests/test_model_dropout.py` (5 tests).

A candidate run, **v3.1**, opts in via `configs/v31_longlead_reg.yaml`:
dropout 0.15, `weight_decay` 1e-4 -> 5e-4, patience 10 -> 8, writing to a
separate `outputs/checkpoints_v31/`. v3 is left untouched for comparison.

### Bugs found and fixed while doing this

1. **`optimize_threshold.py` ignored calibration.** It applied `sigmoid` to raw
   logits, so operating points were tuned against a different distribution from
   the one `predictor.py` serves. Fixed; v3's thresholds were refitted on
   calibrated probabilities.
2. **`historical_backtest.py` hardcoded `(2, 4, 6)`** in four places (candidate
   selection, accumulator, scoring loop, reporting loop). A long-lead model
   raised `ValueError: 2 is not in list` on the first case. Now derives the
   scored horizons from the loaded checkpoint.
3. **Frontend clamped every horizon into `{2,4,6}`** --
   `lead < 3 ? 2 : (lead < 5 ? 4 : 6)`. Requesting +3h painted the +4h field and
   +5h painted +6h, so the UI showed one valid time over another horizon's data.
   Verified stable (not a render race) before the fix, and verified corrected
   after: all five horizons now paint their own slice. It would also have
   collapsed every long lead to 6.
4. **`SUPPORTED_LEADS` was hardcoded** in the backend and frontend, so a model
   with different leads would have had all of them rejected or coerced. Both now
   derive from the loaded checkpoint.

### Offline evaluation of a candidate model

`get_nowcast_service()` now honours `STORMSENSE_CKPT` / `STORMSENSE_CONFIG`, so
a backtest can drive a **candidate** checkpoint through the real production
inference path rather than a reimplementation. Unset in normal operation; the
served model is unchanged. Pinned by `tests/test_service_checkpoint_override.py`.

## GFS production backtest — and a measurement artifact it exposed

The first GFS-driven backtest of v3 (14 cases, `--step-hours 96`) produced a
result that is physically impossible:

| lead | prevalence | PR-AUC | no-skill | skill | CSI | FAR |
|---|---|---|---|---|---|---|
| +8h | 0.0461 | 0.0848 | 0.0461 | 1.84x | 0.075 | 0.916 |
| +10h | 0.0386 | 0.0754 | 0.0386 | 1.95x | 0.076 | 0.918 |
| +12h | 0.0625 | 0.1558 | 0.0625 | 2.49x | 0.130 | 0.848 |
| +14h | 0.0865 | 0.2819 | 0.0865 | 3.26x | 0.211 | 0.719 |
| +16h | 0.1043 | 0.3523 | 0.1043 | 3.38x | 0.246 | 0.573 |

Skill **rises** with lead time (1.84x -> 3.38x). Forecast skill must decay with
lead; a +16h forecast cannot be better than a +8h one from the same analysis.

**Cause: the sampling stride aliased the diurnal cycle.** `--step-hours 96` is a
multiple of 24, so every simulated NOW landed at 00:00 UTC. The leads then map
to fixed local times -- +8h is 13:30 IST, +16h is 21:30 IST. Convection over
West Bengal peaks in the afternoon and evening, so later leads systematically
verified against stormier hours. Observed prevalence climbing monotonically
0.046 -> 0.104 across the leads is the fingerprint: that is the diurnal cycle,
not model behaviour.

**This affects the existing v2 backtest too.** `reports/gfs_production_backtest.json`
was produced with `--step-hours 72`, also a multiple of 24, so its per-horizon
numbers carry the same bias. They should not be compared across horizons until
re-run.

### Corrected backtest (91h stride) — AUTHORITATIVE

`reports/gfs_production_backtest_corrected_v3.json`, 14/14 cases, sampling
**10+ distinct UTC hours** (00, 03, 04, 08, 09, 13, 14, 18, 19, 23).

Reproduce:

```
STORMSENSE_DEVICE=cpu \
STORMSENSE_CKPT=Data/outputs/checkpoints_v3/v2_best.pt \
STORMSENSE_CONFIG=configs/v3_longlead.yaml \
python scripts/historical_backtest.py --max-cases 14 --step-hours 91 \
  --out reports/gfs_production_backtest_corrected_v3.json
python scripts/report_backtest.py \
  --in reports/gfs_production_backtest_corrected_v3.json \
  --out reports/v3_production_decision.json
```

**Sampling check passes.** Prevalence is flat across leads
(0.0700 / 0.0651 / 0.0653 / 0.0629 / 0.0696), corr(lead, prevalence) = **-0.158**,
max/min = 1.11x. Compare the invalid run: +0.944 and 2.70x.

| lead | samples | base | PR-AUC | skill | CSI | POD | FAR | precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| +8h | 11,550 | 0.0700 | 0.1145 | 1.64x | 0.0939 | 0.2423 | 0.8670 | 0.1330 | 0.1717 |
| +10h | 11,550 | 0.0651 | 0.1187 | 1.82x | 0.1012 | 0.3098 | 0.8693 | 0.1307 | 0.1838 |
| +12h | 11,550 | 0.0653 | 0.1051 | 1.61x | 0.0814 | 0.2586 | 0.8938 | 0.1062 | 0.1505 |
| +14h | 11,550 | 0.0629 | 0.0928 | 1.48x | 0.0919 | 0.3127 | 0.8848 | 0.1152 | 0.1683 |
| +16h | 11,550 | 0.0696 | 0.1263 | 1.81x | 0.1269 | 0.3843 | 0.8407 | 0.1593 | 0.2252 |

Confusion matrices:

| lead | TP | FP | FN | TN | threshold |
|---|---|---|---|---|---|
| +8h | 196 | 1,278 | 613 | 9,463 | 0.541 |
| +10h | 233 | 1,550 | 519 | 9,248 | 0.468 |
| +12h | 195 | 1,642 | 559 | 9,154 | 0.477 |
| +14h | 227 | 1,744 | 499 | 9,080 | 0.477 |
| +16h | 309 | 1,631 | 495 | 9,115 | 0.486 |

### Production decision — criteria fixed BEFORE the corrected results

1. PR-AUC >= **2.0x** its own no-skill baseline
2. FAR <= **0.85**
3. POD >= **0.30**
4. Sampling not diurnally aliased

All four must hold.

| lead | verdict | failing criteria |
|---|---|---|
| +8h | **NOT production-worthy** | skill 1.64x, FAR 0.867, POD 0.242 |
| +10h | **NOT production-worthy** | skill 1.82x, FAR 0.869 |
| +12h | **NOT production-worthy** | skill 1.61x, FAR 0.894, POD 0.259 |
| +14h | **NOT production-worthy** | skill 1.48x, FAR 0.885 |
| +16h | **NOT production-worthy** | skill 1.81x |

**No lead qualifies.** Every one fails the skill threshold; four of five also
exceed the false-alarm ceiling.

Note what the alias had hidden: on the invalid run the same criteria would have
passed +12h, +14h and +16h, and +16h would have looked like the BEST lead
(3.38x). Corrected, +16h is 1.81x. The artifact did not merely add noise -- it
inverted the ranking.

### Live vs laboratory

| lead | PR-AUC on ERA5 input (test split) | PR-AUC on real GFS input | ratio |
|---|---|---|---|
| +8h | 0.2482 (6.38x) | 0.1145 (1.64x) | 0.46x |
| +10h | 0.2181 (5.62x) | 0.1187 (1.82x) | 0.54x |
| +12h | 0.2010 (5.19x) | 0.1051 (1.61x) | 0.52x |
| +14h | 0.1940 (5.02x) | 0.0928 (1.48x) | 0.48x |
| +16h | 0.1979 (5.14x) | 0.1263 (1.81x) | 0.64x |

Live skill is roughly **half** the ERA5-input figure at every lead. This is the
GFS-vs-ERA5 distribution shift, and it is why the earlier ERA5-only table could
not support a deployment decision on its own.

## Limitations

- Metrics above are **ERA5-input** metrics. Live inference uses GFS, and the
  measured GFS/ERA5 distribution shift costs roughly 4x (see
  `reports/gfs_production_backtest.json`). Live v3 skill will be lower than the
  table shows; a GFS production backtest for these leads has NOT been run.
- The target is a proxy label (3 h rain > 15 mm, or CAPE > 2000 with CIN < 50),
  not human-verified severe-weather reports.
- Spatial resolution is unchanged at 0.25 deg (~28 km) and no claim beyond that
  is made; map smoothing is visualisation only.

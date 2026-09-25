# StormSense — false-alarm reduction phase

Date: 2026-09-21
Production checkpoint: `Data/outputs/checkpoints/v2_calibrated_best.pt` — **unchanged**

All numbers below come from the corrected, non-aliased GFS backtest protocol
(91-hour stride, real GFS f000 analyses, ERA5-derived truth at the valid time,
inference through the production path). The 72h/96h aliased runs are not used.

---

## Headline answer

**FAR cannot be reduced below ~0.80 at any useful detection rate, and the reason
is arithmetic rather than a tuning failure.**

`FAR = 1 − precision`. The event base rate on held-out GFS is ≈ 0.065. The model
achieves a genuine precision lift of **2.2–3.0× over chance**, which puts
precision at 0.14–0.20 and therefore floors FAR at **0.80–0.86**. Reaching
FAR ≤ 0.70 would require precision ≥ 0.30, a **4.6× lift** — more than double
the discrimination the model demonstrably has.

A dense sweep of every threshold from 0.01 to 0.99 confirms it empirically:
**FAR ≤ 0.80 is UNREACHABLE at every lead.**

---

## A. Baseline (v3 long-lead, held-out 2024 GFS, 14 cases, 11,550 cells/lead)

| lead | base | PR-AUC | skill | CSI | POD | FAR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| +8h | 0.0700 | 0.1145 | 1.64× | 0.0939 | 0.2423 | 0.8670 | 0.1195 | — |
| +10h | 0.0651 | 0.1187 | 1.82× | 0.1012 | 0.3098 | 0.8693 | 0.1132 | — |
| +12h | 0.0653 | 0.1051 | 1.61× | 0.0814 | 0.2586 | 0.8938 | 0.1202 | — |
| +14h | 0.0629 | 0.0928 | 1.48× | 0.0919 | 0.3127 | 0.8848 | 0.1268 | — |
| +16h | 0.0696 | 0.1263 | 1.81× | 0.1269 | 0.3843 | 0.8407 | 0.1210 | — |

---

## D/E. Cause of the false alarms — two real input bugs, found by measurement

54 cached GFS f000 cycles were paired with ERA5 at **identical valid times**
(`scripts/gfs_domain_shift_paired.py`), so the differences are model-vs-model
bias rather than seasonal or sampling artefacts.

| var | bias (GFS−ERA5) | in training σ | corr |
|---|---|---|---|
| cin | −289.45 | **−1.717** | **−0.204** |
| d2m | −4.48 | −0.839 | 0.837 |
| cape | −689.88 | −0.827 | 0.759 |
| tp | −0.259 | −0.266 | 0.457 |
| tcwv | −3.99 | −0.249 | 0.930 |
| t2m | +0.86 | +0.144 | 0.963 |
| sp | +36.98 | +0.004 | 0.997 |

### Bug 1 — CIN sign convention (the negative correlation gave it away)

* ERA5 `cin`: 0.0 % negative, 28.3 % exactly zero → **positive magnitude**
* GFS `cin`: **91.7 % negative**, 0.0 % exactly zero → **negative quantity**

The harmonizer applied `np.maximum(0.0, cin)`, which mapped **91.7 % of GFS CIN
to exactly zero** — telling a model trained on positive-magnitude inhibition that
nothing was capping convection anywhere. That is a direct false-alarm generator.

Correlation with ERA5: **−0.133 under the clip**, −0.204 raw, **+0.204 with
`abs()`**. Fixed to `np.abs()`.

*Residual limitation:* corr is still only +0.20 and magnitudes differ ~3×
(ERA5 223 vs |GFS| 66 J/kg). This fixes the **convention**, not the underlying
representation difference.

### Bug 2 — a GFS f000 analysis contains no precipitation at all

The `.idx` for f000 lists CAPE/CIN at `anl` but has **no PRATE and no APCP
message**. The `PRATE:surface` lookup therefore returned a structurally zero
field: measured GFS `tp` mean **0.0000 mm/h** against ERA5's 0.2590.

Replaced with **APCP accumulated over the 6 h window ending at the analysis
time**, divided by 6 → mm/hour, taken from the prior cycle (`analysis_t0 − 6h`,
f006).

**Why this is not leakage.** The accumulation window has *already elapsed* at the
analysis instant, and the source cycle was published hours before issue time.
This is distinct from the practice the pipeline forbids — using f001–f005 of the
*current* cycle as stand-ins for unobserved hourly history.

Validated against ERA5 at identical valid times:

| valid time | GFS (new) | ERA5 | corr |
|---|---|---|---|
| 2024-05-26 12Z | 1.036 mm/h | 1.111 mm/h | **+0.703** |
| 2024-05-27 00Z | 1.884 mm/h | 1.929 mm/h | **+0.827** |

### Measured effect of both fixes (same 14 cases, same protocol)

| lead | skill before → after | FAR before → after | Brier before → after |
|---|---|---|---|
| +8h | 1.64× → **1.78×** (+8.8 %) | 0.867 → **0.850** | 0.1195 → 0.1088 |
| +10h | 1.82× → **1.94×** (+6.5 %) | 0.869 → **0.853** | 0.1132 → 0.1025 |
| +12h | 1.61× → **1.69×** (+4.8 %) | 0.894 → **0.879** | 0.1202 → 0.1089 |
| +14h | 1.48× → **1.56×** (+5.9 %) | 0.885 → **0.878** | 0.1268 → 0.1136 |
| +16h | 1.81× → **1.97×** (+8.4 %) | 0.841 → **0.828** | 0.1210 → 0.1076 |

Improvement at **every lead on every metric**, with a small honest POD cost
(+8h 0.242 → 0.226).

---

## B. Dense threshold sweep (0.01 → 0.99)

Minimum achievable FAR subject to a detection floor:

| lead | POD ≥ 0.30 | POD ≥ 0.40 | POD ≥ 0.50 | FAR ≤ 0.80 |
|---|---|---|---|---|
| +8h | 0.850 | 0.855 | 0.863 | **unreachable** |
| +10h | 0.854 | 0.871 | 0.883 | **unreachable** |
| +12h | 0.880 | 0.886 | 0.887 | **unreachable** |
| +14h | 0.885 | 0.899 | 0.907 | **unreachable** |
| +16h | 0.821 | 0.842 | 0.861 | **unreachable** |

Best achievable precision (and hence the FAR floor):

| lead | base | best precision | lift | ⇒ min FAR |
|---|---|---|---|---|
| +8h | 0.0700 | 0.1541 | 2.20× | 0.846 |
| +10h | 0.0651 | 0.1975 | 3.03× | 0.803 |
| +12h | 0.0653 | 0.1432 | 2.19× | 0.857 |
| +14h | 0.0629 | 0.1388 | 2.21× | 0.861 |
| +16h | 0.0696 | 0.1952 | 2.80× | 0.805 |

---

## C. Calibration audit — the one clearly fixable defect

The model is **systematically overconfident on GFS input**: mean predicted
probability exceeds the observed base rate by **+0.076 … +0.091**, ECE 0.11–0.14.
Cause: v3's temperatures were fitted on **ERA5** validation data, but production
runs on **GFS**.

Fixed by fitting `p' = sigmoid(a·logit(p) + b)` on a **disjoint validation year
(2023 GFS, 16 cases)** and evaluating on the untouched 2024 test cases
(`scripts/calibrate_gfs_domain.py`).

| lead | a | b | ECE test | Brier test |
|---|---|---|---|---|
| +8h | 0.200 | −1.980 | 0.1146 → **0.0073** | 0.1088 → **0.0632** |
| +10h | 0.260 | −1.800 | 0.1136 → **0.0049** | 0.1025 → **0.0591** |
| +12h | 0.340 | −1.740 | 0.1205 → **0.0198** | 0.1089 → **0.0605** |
| +14h | 0.280 | −2.100 | 0.1368 → **0.0183** | 0.1136 → **0.0583** |
| +16h | 0.290 | −2.140 | 0.1225 → **0.0208** | 0.1076 → **0.0629** |

**ECE improves 6–23×, Brier drops ≈ 45 %, out-of-sample.**

**Stated plainly: this buys reliability, not skill.** The transform is monotonic,
so PR-AUC is unchanged to six decimal places and FAR at matched POD is unchanged
(0.821–0.884 after vs 0.821–0.885 before). A displayed "30 %" now genuinely means
30 %; it does not mean fewer false alarms at the same detection rate.

---

## B2. Thresholds were fitted in the wrong domain — the one real detection win

`scripts/optimize_threshold.py` fits decision thresholds on the **ERA5**
validation split. Production runs on **GFS**, and the ERA5-optimal operating
point is not the GFS-optimal one. The checkpoint's thresholds (0.47–0.54) turn
out to sit far above the GFS-optimal ones (0.16–0.49), leaving real detection
unused.

Refitted honestly — **fit on 2023 GFS, apply to untouched 2024 GFS**
(`scripts/fit_gfs_thresholds.py`):

| lead | POD | CSI | FAR |
|---|---|---|---|
| +8h | 0.226 → **0.513** (+127 %) | 0.0989 → **0.1205** (+22 %) | 0.850 → 0.864 |
| +10h | 0.283 → **0.468** (+65 %) | 0.1073 → 0.1077 | 0.853 → 0.877 |
| +12h | 0.245 → 0.260 | 0.0881 → 0.0907 | 0.879 → 0.878 |
| +14h | 0.274 → **0.365** | 0.0920 → 0.0907 | 0.878 → 0.892 |
| +16h | 0.350 → 0.347 | 0.1305 → 0.1305 | 0.828 → 0.827 |
| **mean** | 0.276 → **0.391 (+42 %)** | 0.1034 → **0.1080 (+4.5 %)** | 0.858 → 0.868 |

At +8h this reached CSI **0.1205 against a test-set oracle ceiling of 0.1218** —
99 % of the attainable gain, with the choice made entirely on validation data.

**The overfitting this avoided is measurable.** Choosing thresholds on the test
set suggested +8.8 % mean CSI; the honest validation-fitted choice delivered
**+4.5 %**. Half of the apparent gain was fitting noise.

### A FAR ceiling was tested and rejected

Re-running the fit under a hard `FAR ≤ 0.85` constraint:

| strategy | FAR | POD | CSI |
|---|---|---|---|
| current (ERA5-fitted) | 0.858 | 0.276 | 0.1034 |
| **2023-fitted, CSI, unconstrained** | 0.868 | **0.391** | **0.1080** |
| 2023-fitted, CSI, FAR ≤ 0.85 | 0.861 | 0.229 | 0.0900 |

Forcing FAR down buys 0.007 of FAR and costs 42 % of detection — mean CSI falls
*below* the status quo, and +8h collapses to POD 0.043. This is the concrete
demonstration that chasing a FAR target is counterproductive here. The
unconstrained CSI fit is the operating point the evidence supports.

CSI and F1 select the **identical** threshold at every lead, so no further gain
is available from the choice of objective.

---

## H. Label quality — a large part of "FAR" is definitional

Full-array audit (14,412,750 finite cells, `scripts/label_quality_audit.py`):

* overall severe prevalence **0.0303**
* **77.7 % of positives come from the CAPE/CIN branch alone**
* 21.8 % from the heavy-rain branch alone, 0.5 % from both
* of **all** positives: **22.3 %** have rain ≥ 15 mm; **56.9 % have < 1 mm**
* a rain-only (≥ 15 mm) label would have prevalence 0.0068 — the production label
  is **4.5× more prevalent** than an impact label

So the target mostly flags *an environment supportive of convection*, not severe
weather that occurred. A model reproducing the label perfectly still looks like
it false-alarms to a user who reads an alert as "severe weather will happen".

The label is **not** noisy: positives are spatially coherent (only 3.9 % isolated
single cells, median 18 positive cells per active frame). Events are short —
median duration 2 h, 45 % single-hour.

### Impact-label experiment (separate scoring, production label untouched)

Re-scoring the *same* predictions against `rain ≥ 15 mm`:

| lead | production FAR | impact FAR | production skill | impact skill |
|---|---|---|---|---|
| +8h | 0.860 | **0.985** | 1.78× | 1.80× |
| +10h | 0.862 | **0.958** | 1.94× | 2.79× |
| +12h | 0.887 | **0.980** | 1.69× | 2.08× |
| +14h | 0.885 | **0.965** | 1.56× | 2.06× |
| +16h | 0.829 | **0.910** | 1.97× | 3.18× |

**Negative result, reported as such.** Switching to an impact label makes FAR
*worse*, because the base rate collapses to 0.006–0.009. Relative skill does rise
(up to 3.18×), so the rain signal is real — but **low base rate is the binding
constraint under both labels.** A label change cannot buy FAR.

---

## Freshest available live data (measured, not assumed)

| source | publication lag | covers West Bengal? | verdict |
|---|---|---|---|
| **NOAA GFS 0.25°** | **3.60 h** (n=12: min 3.54, max 3.81) | yes | **selected** |
| NCEP GDAS | ~6.8 h | yes | rejected — slower |
| ECMWF IFS open data | 6.45–7.57 h | yes | rejected — slower |
| NOAA HRRR | ~1 h | **no** (CONUS/Alaska) | unusable |
| NCEP RAP | ~1 h | **no** (North America) | unusable |

GFS is the freshest source that actually covers the domain.

`GFS_PRODUCTION_LAG_HOURS` is deliberately **kept at 5.0 h** rather than lowered
to the measured 3.60 h: `Last-Modified` marks when the `.idx` *appears*, not when
the cycle is dependably retrievable, and this constant gates the backtest's
notion of what an operator could have had. Tightening it would bias measured
skill **upward** — the one direction that must never be optimistic.

---

## Files changed in this phase

| file | change |
|---|---|
| `src/inference/gfs_live.py` | CIN `abs()` fix; `fetch_elapsed_precip_mm_per_hour`; precip provenance; lag documentation |
| `src/inference/nowcast_service.py` | `resolve_wallclock_horizons`; precip provenance; horizon-specific XAI |
| `src/inference/target_time.py` | **new** — wall-clock target-time resolver |
| `src/inference/observation_surface.py` | rain palette returned to a non-risk (blue→violet) ramp |
| `scripts/historical_backtest.py` | `--dump-raw` for sweep/calibration analysis |
| `scripts/gfs_domain_shift_paired.py` | **new** — paired GFS/ERA5 analysis |
| `scripts/label_quality_audit.py` | **new** — label composition audit |
| `scripts/far_analysis.py` | **new** — dense sweep + reliability |
| `scripts/far_by_label_variant.py` | **new** — impact-label experiment |
| `scripts/calibrate_gfs_domain.py` | **new** — GFS-domain recalibration |
| `scripts/train_v2.py` | flush epoch logs |
| `tests/test_target_time.py` | **new** — 13 tests |

No checkpoint was overwritten. The GFS-domain calibration is written to a report,
**not** into any checkpoint; promotion remains a separate, explicit step.

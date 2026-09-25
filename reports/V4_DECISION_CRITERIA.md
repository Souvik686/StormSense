# v4 promotion criteria — fixed BEFORE the results are known

Written 2026-09-22, while v4 was still training (best epoch 2, val PR-AUC 0.1697).
Recording the rules first is the only way they cannot be bent to fit whatever
comes out.

Candidate: `Data/outputs/checkpoints_v4/v2_best.pt` — hourly leads 4–16 h.
Incumbent: `Data/outputs/checkpoints/v2_calibrated_best.pt` — leads 2–6 h,
md5 `6afdc4d6c1c1dc7ae1901bee794eb543`, **must not be overwritten**.

## What v4 is actually for

v4 exists to make the four user-facing horizons *literally true*. Reaching
"+2 h from now" from an analysis that is 3.6–9.6 h old needs a 5.6–11.6 h
forecast lead, and the required lead moves continuously as the analysis ages.
v2's five leads (2–6 h) cannot reach those targets at all; v3's coarse
{8,10,12,14,16} misses most of them. Only an hourly set spanning ~4–16 h can.

So v4 is **not** competing with v2 on v2's own ground. Measured coverage of
NOW/+2/+4/+6 across a full GFS cycle:

| model | leads | cycle coverage |
|---|---|---|
| v2 | 2–6 h | NOW only (≈ 25 %) |
| v3 | 8,10,12,14,16 | < 50 % |
| v4 | 4–16 h hourly | **100 %** at the measured 3.60 h lag |

## Lead sets barely overlap — so compare only where comparison is real

| pair | shared leads |
|---|---|
| v2 ∩ v4 | 4, 5, 6 |
| v3 ∩ v4 | 8, 10, 12, 14, 16 |
| v2 ∩ v3 | **none** |

v4 is the only candidate comparable with both. Any table putting a v2 +2 h
forecast beside a v3 +12 h forecast is comparing different questions, and
`scripts/compare_candidates.py` refuses to do it.

## Decision rules

All evaluation on the **same** held-out protocol: 2024 GFS, 91 h stride (never a
multiple of 24), ERA5-derived truth at the valid time, inference through
`NowcastService`. Calibration and thresholds fitted on validation only.

### Rule 1 — production leads 2–6 h

**Keep v2 unless v4 is at least as good at every shared lead (4, 5, 6).**
v2 is the strongest thing the product has at short range (+2 h at 16.4×
no-skill on ERA5). A candidate that is merely "close" does not justify replacing
it. Ties go to the incumbent.

### Rule 2 — the wall-clock horizons

v4 may serve NOW/+2/+4/+6 **only** for leads that clear, on held-out GFS:

1. PR-AUC ≥ **2.0×** its own no-skill baseline
2. FAR ≤ **0.85**
3. POD ≥ **0.30**
4. sampling not diurnally aliased (corr(lead, prevalence) small, max/min < 1.5)

These are the same four thresholds v3 was judged against, unchanged, so the two
verdicts are commensurable. A lead that fails is **not exposed**; the horizon
reports unavailable with its reason rather than being filled by a neighbour.

### Rule 3 — FAR is not the deciding metric on its own

FAR ≤ 0.80 is arithmetically unreachable here: `FAR = 1 − precision`, the base
rate is ≈ 0.065, and a 2.2–3.0× precision lift caps precision at 0.14–0.20.
A candidate is therefore **not** rejected for FAR ≈ 0.85, nor accepted for a low
FAR bought by collapsing POD. FAR 0.60 at POD 0.05 loses to FAR 0.75 at POD 0.40.

### Rule 4 — calibration is reported, never used to claim skill

GFS-domain recalibration is monotonic: it improves ECE and Brier and leaves
PR-AUC unchanged to six decimals. It may be promoted on its own merits
(reliability), and must never be presented as a false-alarm reduction.

### Rule 5 — no partial credit for the UI

If no v4 lead clears Rule 2, the wall-clock horizons are **not shipped**, the UI
keeps reporting them unavailable, and production stays on v2. "The dashboard has
four buttons" is not a reason to promote a model.

## Possible outcomes, all acceptable

| outcome | action |
|---|---|
| v4 clears Rule 2 at some leads | expose exactly those horizons; keep v2 for 2–6 h unless Rule 1 is also met |
| v4 clears Rule 2 at no lead | keep v2; horizons stay unavailable; report the limitation |
| v4 beats v2 at 4/5/6 **and** clears Rule 2 | promote v4 as the single served model |
| results inconclusive | keep v2 — the safer incumbent — and say so |

## What gets written down either way

Per candidate, per lead: base rate, PR-AUC, skill vs no-skill, CSI, POD, FAR,
precision, F1, Brier, ECE, TP/FP/FN/TN, and the threshold used. Plus which
horizons each model can honestly serve across a GFS cycle.

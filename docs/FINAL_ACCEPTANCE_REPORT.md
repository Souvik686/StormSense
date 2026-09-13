# StormSense — Final Acceptance Report

**Date:** 2026-09-11
**Verdict:** **GO**, with the documented limitations in §15 (all of which are
disclosure/scope limits, not broken functionality).

---

## 1. What was actually changed

### 1.1 Scientific correctness (highest-severity findings)

**A GFS temporal-leakage regression was found and fixed.**
`src/inference/gfs_live.fetch_and_harmonize` had been rewritten to pick a single
"anchor cycle" and interpolate the model's six *past* input timesteps from
**forecast hours f003/f006/f009** of that cycle. Those are future forecast steps,
not observed history. Feeding them into the input the model was trained to read
as analysis leaks forecast information into the observed past. This contradicted
the module's own docstring ("No GFS forecast hour (f001+) is EVER used as an
input timestep").

The pipeline now fetches **GFS ANALYSIS fields/states produced through data assimilation only** and interpolates the hourly slots
between two real bracketing analyses. A negative-control check confirmed the new
test suite flags the old behaviour as a leak (forecast hours 3/6/9 detected).

**Cycle availability is now modelled honestly.** `_cycles_available_at()` will
only consider a GFS cycle once real production latency (5 h) says it was actually
published. Empirically confirmed: at 14:48 Z the 12 Z cycle returned HTTP 404,
exactly as the rule predicts. The same rule drives both live and backtest, so a
backtest can never consume a cycle from its own future.

**Exact wall-clock reference time separated from GFS analysis time.** These were
conflated: the UI displayed the issue time under a "GFS Run:" label, implying
GFS had issued an operational cycle at the current instant. They are now distinct
fields end-to-end (`live_reference_time` vs `live_analysis_time`), and the UI
labels the latter "Input analysis".

**A latent import bug was silently degrading every timestamp.**
`src/inference/risk_map.py` used `timedelta` without importing it, so
`compute_valid_time()` always fell into its `except` branch and returned the
string `"<issue> +Nh"` instead of a real timestamp for every GeoJSON grid cell.

### 1.2 Fabricated data removed (no replacements invented)

| Location | Was | Now |
|---|---|---|
| `main.py` current_weather fallback | Invented readings (29.5 °C, 82 % RH, 1006 hPa…) served as if measured | All fields `null`, `observation_available: false`, real error attached |
| `main.py` timeline compat | Ambient "forecast" synthesised as `temp − 0.4·h`, `humidity + 2·h` | `null` + explicit note that the model has no ambient forecast head |
| `nowcast_service.get_xai_attribution` (live) | Hardcoded placeholders (humidity 85, CAPE 1500, wind 15) with a `pass` where real values belonged | Real per-cell values from the live GFS analysis state |
| `app.js` view model timeline | `temperature ?? 28.0`, `humidity ?? 80`, `wind ?? 15.0` — the cause of the identical repeated rows | Real per-horizon model output (severe %, mm/3h, flood proxy) |
| `app.js` district cards | `confidencePct ?? 92` shown as "Model Skill: 92%" | Only rendered when the backend supplies a real value |

### 1.3 UI / UX

- **Map risk field (E, F, Y):** the "PREDICTED HIGH-RISK ML CELL #n" dot markers
  are gone; the AI forecast is communicated by the continuous 0.25° risk field.
  Colour bands applied as specified (uncoloured → green → orange → red).
- **Grid On/Off (G):** control removed entirely (it only ever worked in
  historical mode). The underlying model grid is untouched and still drives every
  value. No dead button remains.
- **Refresh (O):** "REFRESH LIVE DATA" removed; exactly one control remains
  (`AUTO-REFRESH: 4m 43s | ↻ Refresh`). One deadline (`nextRefreshAt`) drives both
  the countdown and the firing, so they cannot drift apart. Verified: one
  `/api/live/refresh` and one summary request per manual refresh.
  **Note:** a previously dead `window.forceLiveRefresh` was defined *outside* the
  IIFE, so `API_BASE`/`parseJson` were out of scope — it would have thrown.
- **Layout (S):** map left, Current Location right, bulletins full-width below.
  Fixing this surfaced **a pre-existing HTML nesting bug**: views 2–8 were nested
  *inside* `#view-dashboard`, so every sidebar section rendered at 0×0. Sidebar
  navigation had been broken for all seven non-dashboard views.
- **Current Location (J):** panel is location-specific and shares coordinates with
  the map marker. With geolocation denied it reports "Location unavailable" rather
  than presenting West Bengal as the user's position.
- **Live location marker (I):** blue/cyan, never green.
- **Bulletins (K):** real higher-risk districts from the same district
  aggregation the map uses; clicking focuses the map. Says so honestly when no
  district exceeds the watch level.
- **Jump to area (R):** populated from `/api/geo/areas` (19 real districts +
  Whole State) with real centroids/bounds. No hand-typed or dead entries.
- **Search (Q):** removed.
- **Historical button (P):** exactly one, in the top-right action area.
- **Public language (T):** `SevereWeatherNet`, `ConvGRU`, `v2_calibrated_best.pt`,
  `CALIBRATED V2`, `V2 ENGINE` and the numeric threshold values removed from the
  public UI. The legend shows categories only.
- **NOW-mode 422s (found during final log review).** While the NOW horizon was
  selected, `window.currentLeadHours` is the string `'now'`, and several code
  paths passed it straight into `lead=`. The backend correctly rejected those
  with HTTP 422, so the background refresh silently failed whenever NOW was the
  active horizon. A single `apiLeadHours()` resolver now guards every
  lead-carrying request (NOW resolves to the shortest real horizon for the
  forecast products shown underneath it). Verified in-browser: **71 API
  responses across NOW/+2/+4/+6 plus auto- and manual refresh, all HTTP 200,
  zero 4xx/5xx.** Regression test added.

---

## 2. Files changed

**Backend / inference**
- `src/inference/gfs_live.py` — analysis-only harmonization, production-lag cycle
  selection, per-slot provenance
- `src/inference/nowcast_service.py` — reference/analysis time separation, real
  live surface state, honest XAI, shared limitations constant
- `src/inference/risk_surface.py` — specified colour bands, variable traced in code
- `src/inference/risk_map.py` — `timedelta` import fix
- `weather-app/backend/main.py` — `/api/time/reference`, `/api/geo/areas`,
  fabricated-fallback removal, XAI location params, honest provenance

**Frontend**
- `weather-app/index.html` — layout, control removal, HTML nesting fix, legend
- `weather-app/js/app.js` — refresh unification, live location, bulletins,
  horizon semantics, real values, jump-to-area

**Tests / scripts**
- `tests/test_exact_wall_clock.py` (rewritten), `tests/test_temporal_leakage.py`
  (rewritten), `tests/test_map_domain.py` (new), `tests/test_browser_e2e.py` (new)
- `scripts/historical_backtest.py` (rewritten)

---

## 3. Tests executed & 4. Exact results

Command: `pytest tests/ -q` (CPU, server running on 127.0.0.1:8000)

| File | Tests | Result |
|---|---|---|
| `test_api.py` | 28 | passed |
| `test_smoke.py` | 27 | passed |
| `test_geo_mask.py` | 5 | passed |
| `test_exact_wall_clock.py` | 48 | passed |
| `test_temporal_leakage.py` | 9 | passed |
| `test_map_domain.py` | 97 | passed |
| `test_browser_e2e.py` | 37 | passed |
| **Total** | **251** | **251 passed, 0 failed** |

Baseline before this work was 62 tests. `tests/test_browser_live.py` is a manual
script with no test functions (collects 0); superseded by `test_browser_e2e.py`.

**No test was weakened.** Two E2E assertions were corrected *against the app's
correct behaviour*: a case-sensitive string match, and a countdown measurement
that raced the 1 s ticker. Both were made stricter afterwards (the refresh test
now also asserts no duplicate API requests).

---

## 5. Historical GFS backtest — methodology

For each simulated historical NOW instant `T`:
1. Select only GFS cycles genuinely published by `T` (5 h production latency).
2. Fetch that cycle's real **GFS ANALYSIS fields/states produced through data assimilation** (never forecast hours).
3. Run the **actual production path** (`NowcastService.refresh_live_state`) — not
   a reimplementation.
4. Produce +2/+4/+6 h predictions.
5. Verify against **independent** ERA5-derived targets at the valid times.

A hard assertion in the harness fails the run if the selected analysis is ever
later than the simulated NOW. It never fired across 30 cases.

**Scored variable:** `severe_weather_prob` — the calibrated probability head whose
training label (`src/features/targets.py`) is exactly the field used as truth.
The compound `overall_risk` is deliberately *not* scored against that label.
**Threshold:** the per-lead calibrated threshold from the checkpoint.

---

## 6. Historical GFS backtest — results

30/30 cases completed. Window **2024-05-02 → 2024-10-14**. N = 24,750 grid-cell
samples per horizon.

| Horizon | N | Prevalence | Thr | CSI | POD | FAR | PR-AUC | (no-skill) | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| +2h | 24,750 | 0.0411 | 0.727 | 0.1045 | 0.1900 | 0.8115 | 0.0996 | 0.0411 | 0.0740 | 0.0859 |
| +4h | 24,750 | 0.0419 | 0.673 | 0.0832 | 0.1455 | 0.8371 | 0.1165 | 0.0419 | 0.0669 | 0.0743 |
| +6h | 24,750 | 0.0312 | 0.625 | 0.0867 | 0.1816 | 0.8576 | 0.0981 | 0.0312 | 0.0574 | 0.0694 |

**Honest reading:** PR-AUC is roughly **2.4–2.8× the no-skill baseline**, so there
is real but **weak** skill. It is **far below** the 0.484 PR-AUC reported for
ERA5-driven held-out testing. That gap is the GFS↔ERA5 distribution shift, now
*measured* rather than asserted. FAR is high (0.81–0.86) at the calibrated
thresholds, which were optimised on ERA5 inputs and are **not** well calibrated
for GFS inputs.

This is **not** a claim of operational accuracy, and ERA5-only validation is
**not** presented as GFS validation. Distribution-shift Z-scores are **not** used
as a substitute for this backtest.

---

## 7. Temporal-leakage methodology

`tests/test_temporal_leakage.py` (9 tests) parses every requested GFS URL and
asserts on the actual **forecast hour** and **cycle time**:
- every product request is `f000`;
- no cycle is requested that production latency says was unpublished at the
  reference instant (including a simulated *past* instant);
- cycle-availability boundaries (incl. ±1 s around the lag boundary and midnight);
- the six input slots are consecutive hourly steps ending at the analysis time,
  the final slot being the real analysis at weight 1.0, with interpolation
  strictly bounded by the two real analyses.

The previous version of this file wrapped the call in `except: pass` and only
checked URL substrings — it passed while the pipeline was fetching f003/f006/f009.

---

## 8. Exact NOW / +2 / +4 / +6 proof

Live from `GET /api/time/reference`:

```
reference_time : 2026-09-11T14:57:20.127111+00:00     <- NOT floored
+2h            : 2026-09-11T16:57:20.127111+00:00     offset 7200 s
+4h            : 2026-09-11T18:57:20.127111+00:00     offset 14400 s
+6h            : 2026-09-11T20:57:20.127111+00:00     offset 21600 s
analysis_time  : 2026-09-11T06:00:00+00:00  (lag 8.95 h)  <- separate field
```

The specified acceptance instant, verified on parsed datetimes:

```
NOW 2026-09-11T13:06:10.449910Z
 +2 2026-09-11T15:06:10.449910Z   +4 2026-09-11T17:06:10.449910Z   +6 2026-09-11T19:06:10.449910Z
```

48 assertions cover non-zero minutes/seconds, fractional seconds, midnight/month/
year rollover, IST conversion, and mutual consistency — all on parsed datetimes,
never string inspection. A **negative control** proves the tests detect flooring.
The browser renders horizons from this endpoint, so UI and API share one reference.

---

## 9. GFS provenance

- Input is **real NOAA GFS GFS ANALYSIS fields/states produced through data assimilation only**, at 0.25°.
- The operational cycle (00/06/12/18 Z), the analysis time, the wall-clock
  reference instant and each forecast valid time are four distinct, separately
  reported values.
- Hourly slots between two analyses are labelled `interpolated`; the final slot
  is labelled `analysis` and used unmodified. Every slot carries
  `forecast_hour_used: 0`.
- `/api/live/ml-status` exposes `uses_forecast_hours_as_input: false`,
  `analysis_cycles`, `analysis_lag_hours`, and per-slot provenance.
- The UI never labels an interpolated or derived state as a GFS operational run.

---

## 10. Map / domain validation

97 assertions in `tests/test_map_domain.py`. 18 real West Bengal locations across
**all five regions** (south/central/north/east/west) are accepted at +2/+4/+6 h
and map to a nearest grid cell within half a grid step. Four genuinely outside
points (Patna, Dhaka, Kathmandu, Bay of Bengal) are rejected with no forecast.

Traced explicitly: latitude vector is **descending** (28→20), longitude ascending
(84→90), grid 33×25 at 0.25°. Tests assert a northern point maps to a lower row
index than a southern one (catches a flipped axis), and that a swapped lat/lon
pair is rejected.

**The backend was never the cause of the reported "Location outside model
domain".** All 21 probe points passed containment. The defect was in the
frontend: a falsy/failed response rendered the *geographic* rejection message.
Failures are now reported as service failures, and only a genuine
`inside_monitored_region: false` produces an out-of-region message.

---

## 11. Current-location validation

The map marker and the panel read one shared state (`StormSenseUserLocation`).
With geolocation denied the panel shows "Location unavailable" and no marker is
drawn — verified in-browser. Values come from `/api/nowcast/point` at the live
coordinates. West Bengal remains the target region in the banner.

---

## 12. Refresh validation

One control, one deadline, one timer. Verified in-browser: countdown advances
each second; `manualRefresh()` issues exactly **one** `/api/live/refresh` and
**one** summary request, repaints, and restarts the countdown at the full window.
No duplicate timers or requests. In historical mode the countdown reads
"paused (historical)" instead of the control vanishing.

---

## 13. Sidebar validation

All eight sections (dashboard, radar, advisories, wrf, xai, gis, threshold,
streams) are visible after navigation and render substantive content; navigation
preserves the selected horizon and raises no JS errors.

This is where the HTML nesting bug was caught — before the fix, seven of eight
sections rendered at 0×0 despite containing markup.

---

## 14. Jump-to-area validation

All 20 options (19 districts + Whole State) are exercised in-browser: each
resolves to a real area, moves the map, and lands within the domain
(20–28.5 °N, 84–90.5 °E). Options are generated from the boundary dataset, and
areas without model grid coverage would be labelled as such (none currently are).

---

## 15. Known limitations

1. **Live skill is weak and not well calibrated.** Measured PR-AUC 0.10–0.12 vs
   no-skill 0.03–0.04; FAR 0.81–0.86 at ERA5-derived thresholds. Live output is
   directionally informative, not operationally calibrated.
2. **Verification labels are ERA5-derived proxies**, not observed severe-weather
   reports. No gridded sub-daily observed truth exists for this domain/period.
3. **Truth is ERA5 while inputs are GFS**, so measured error includes
   representation differences as well as model error.
4. **Backtest N is modest** (30 cases, 24,750 cells/horizon, 2024 convective
   season only). Confidence intervals are correspondingly wide. Not seasonally
   or inter-annually representative.
5. **Sub-6-hour transients are smoothed** — GFS publishes analyses 6-hourly and
   the hourly slots are interpolated between them.
6. **The live atmospheric state lags wall-clock** by the GFS production lag
   (typically 6–10 h). NOW/+2/+4/+6 labels are exact wall-clock, but the state
   behind them is as of the last published analysis. Both are shown.
7. **XAI is rule-based / physics-inspired**, not SHAP and not learned feature
   importance. The attribution weighting itself is **not** separately validated.
8. **Historical mode is one frozen prediction** from a fixed 2024 case.
9. **Thermodynamic diagnostics are not recomputed for the live cycle** and report
   as unavailable in live mode rather than borrowing historical values.
10. **Geolocation depends on browser permission.** Denied → honest unavailable
    state, not a substituted coordinate.
11. `weather-app/.venv` is broken (references another machine's Python). Use the
    project `venv`. Not fixed, as it is outside this task's scope.

---

## 16. Explicit non-claims

- No claim of "scientifically perfect", "zero error", or "flawless fluid dynamics".
- ERA5-only validation is **not** called GFS validation.
- Distribution-shift analysis is **not** called a predictive accuracy backtest.
- No claim of perfect forecasting, and no claim that the XAI attribution was
  validated.

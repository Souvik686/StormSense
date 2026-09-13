# StormSense — Complete Handoff for a New Claude Session

**Written:** 2026-09-12
**Repository:** `D:\Weather-Hackathon`
**Branch:** `master` (main branch for PRs is `main`)
**Audience:** A Claude session with **no access** to prior conversation history.

> **Read this file completely before touching anything.** Then verify against the
> real repository — this document was written by inspecting the code, running the
> tests, and driving the app in a browser, but code changes after it was written.
> Where something could not be verified, it is marked **UNVERIFIED** or **UNKNOWN**.

---

## 1. PROJECT OVERVIEW

**Name:** StormSense — AI Severe Weather Early Warning / Nowcasting platform.

**Purpose:** Short-range (0–6 h) severe-weather nowcasting for **West Bengal, India**.
It predicts severe-weather probability, 3-hour rainfall, and a flash-flood proxy on a
0.25° grid, and presents them alongside genuine current observations.

**Hackathon context:** A hackathon project that has been through several hardening
passes. Judges are expected to probe it adversarially for fabricated data, fake
timestamps, and overstated accuracy. **Scientific honesty is the top priority** —
above UI polish, above impressive-looking numbers.

### Architecture (data flow)

```
NOAA GFS GFS ANALYSIS fields/states produced through data assimilation (AWS Open Data S3, byte-range GRIB2)
        │
        ▼
src/inference/gfs_live.py ──► harmonize to ERA5-shaped tensors (6 hourly slots)
        │
        ▼
src/inference/nowcast_service.py ──► NowcastService (singleton)
        │      ├── predictor.py → PyTorch tri-stream ConvGRU (v2_calibrated_best.pt)
        │      ├── risk_map.py → GeoJSON per-cell risk
        │      ├── risk_surface.py → continuous FORECAST risk PNG (green→red ramp)
        │      ├── observation_surface.py → current OBSERVATION PNG (blue/teal ramp)
        │      └── xai.py → rule-based factor attribution
        ▼
weather-app/backend/main.py (FastAPI) ──► 40 route decorators + serves the frontend
        ▼
weather-app/index.html + weather-app/js/app.js (vanilla JS + Leaflet + Tailwind CDN)
```

**Separately**, for the NOW view only:
```
OpenWeatherMap /data/2.5/weather (24 West Bengal sites)
        │
        ▼
weather-app/backend/openweather.py → main.py `_get_station_observations()` (4-min TTL)
        ▼
src/inference/observation_surface.py → IDW interpolation → blue/teal PNG
```

### Technologies
- **Backend:** Python 3.13, FastAPI, uvicorn, httpx (async), Pydantic
- **ML:** PyTorch 2.6.0+cu124 (runs CPU), NumPy, SciPy
- **Geo:** xarray + cfgrib/eccodes (GRIB2), shapely, matplotlib.path, Pillow
- **Frontend:** Vanilla JS (no build step), Leaflet 1.9.4, Tailwind CDN, Material Symbols
- **Tests:** pytest, Playwright (Chromium)

### Main directories
| Path | Contents |
|---|---|
| `src/inference/` | Live pipeline, service, rendering, XAI |
| `src/data/`, `src/features/`, `src/models/`, `src/training/` | Training-time code |
| `weather-app/backend/` | FastAPI app |
| `weather-app/js/`, `weather-app/css/` | Frontend |
| `tests/` | pytest suites |
| `Data/BOUNDARIES/` | West Bengal GeoJSON + cached raster mask |
| `outputs/metrics/` | Held-out ERA5 test metrics |
| `reports/` | GFS backtest results |
| `configs/default.yaml` | Domain/grid/path config |

---

## 2. USER'S CORE REQUIREMENTS / NON-NEGOTIABLE RULES

These are **absolute**. Breaking any of them is a project-level failure.

### Temporal contract
- **NOW** = actual current wall-clock reference time / current-observation state.
- **+2H** = exactly NOW + 2 hours (**7200 s**).
- **+4H** = exactly NOW + 4 hours (**14400 s**).
- **+6H** = exactly NOW + 6 hours (**21600 s**).
- Valid times are **never** floored to the hour, snapped to a GFS cycle, hardcoded,
  or shifted to look better. Sub-hour precision (minutes, seconds, microseconds)
  must survive.
- **GFS analysis time and wall-clock reference time are intentionally separate.**
  They must never be conflated or displayed as one another.

### Anti-fabrication
- **Never** fabricate weather values. No fallback constants like `32.0 °C`, `1006 hPa`,
  `70 %`, `16.0 km/h`. Missing data renders as `—` / `Unavailable`.
- **Never** zero-fill a missing meteorological variable. Absent data ≠ measured zero
  (critical for rainfall).
- **Never** use future GFS forecast hours (f003/f006/f009) as "current" or "past" input.
- **Never** relabel a stale forecast, or the +2h field, as NOW.
- **Never** present historical case-study data as live.
- **Never** copy MSN, or any weather website's values or rendered maps. Do not
  "optimize against MSN".
- **Never** manufacture a continuous spatial observation field from insufficient data.
  If interpolating, say so explicitly in code, API, and UI.

### Scientific honesty
- **Never** claim perfect forecasting, zero error, or flawless prediction.
- **Never** claim a model-skill figure that measurement does not support (a hardcoded
  "92% model skill" was removed for exactly this reason).
- XAI is **rule-based / physics-inspired**, never "validated ML explainability".
- Verification targets are **ERA5-derived proxies**, never "observed severe-weather reports".
- Distribution-shift analysis is **not** predictive backtesting.
- Disclose GFS analysis latency honestly rather than hiding it.

### Structural
- Preserve the **8-view shell** (see §18) unless a fix explicitly requires otherwise.
- **Historical mode stays separate from Live mode.** They never share state.
- **RainViewer must NOT appear on the Interactive Nowcasting Map.** It belongs only
  to the Radar & Satellite Feeds view.
- Forecast valid time must remain explicit and visible.
- Preserve geographic containment to West Bengal.

---

## 3. CURRENT ARCHITECTURE (file by file)

### `src/inference/gfs_live.py` (529 lines) — live GFS ingestion
**Does:** Fetches real NOAA GFS **GFS ANALYSIS fields/states produced through data assimilation** and harmonizes them into the
ERA5-shaped tensors the model expects. See §4 for full detail.

Key items:
- `GfsCycle` dataclass — `cycle_time` (UTC GFS ANALYSIS field/state produced through data assimilation timestamp)
- `HarmonizedInput` dataclass — `surface`, `pressure`, `t0`, `analysis_t0`,
  `analysis_cycles`, `slot_provenance`, `fetched_at`, `wallclock_age_hours`
- `fetch_and_harmonize(lats, lons, target_t0)` — main entry point
- `PRESSURE_LEVELS_HPA = [250, 300, 500, 700, 850, 1000]` — **ASCENDING, load-bearing**

> **DO NOT "fix" `PRESSURE_LEVELS_HPA` to match `configs/default.yaml`.** The YAML
> lists levels descending for human reference only. The model slices by *index*
> (`pres_norm[:, :2, 3:]` = 700/850/1000 wind; `pres_norm[:, 2:, :4]` =
> 250/300/500/700 thermo). Reordering silently swaps which levels feed which stream.
> This trap is documented in the file's own comments — leave them there.

**Assumption:** Raises `GfsFetchError` naming the exact missing variable/level/cycle
rather than filling gaps. Preserve this fail-loud behaviour.

### `src/inference/nowcast_service.py` (1372 lines) — the core service
`NowcastService` is a singleton (`get_nowcast_service()`), guarded by `_SERVICE_LOCK`.

**Two completely separate states — never mix them:**

| Historical (frozen case study) | Live (GFS-driven) |
|---|---|
| `current_pred` | `live_pred` |
| `current_valid_time` (`2024-05-05T15:00:00Z`) | `live_reference_time` (exact wall clock) |
| `current_thermo` (ERA5 sounding) | `live_thermo` (GFS analysis) |
| `risk_surface_png_cache` | `live_risk_surface_png_cache` |
| — | `live_analysis_time` (real GFS f000 time) |
| — | `live_surface_state` (physical units at t0) |

Important methods:
- `_init_operational_state()` — loads ERA5 memmap sample #100 (a real 2024-05-05
  Kalbaishakhi squall), runs inference, computes `current_thermo`. **Historical only.**
- `refresh_live_state(target_t0=None)` — the live path. Fetches GFS, runs
  `predictor.predict(...)`, derives compound risks, regenerates risk PNGs, sets
  `live_thermo`. **On failure it leaves `live_pred` untouched** (stale-data protection)
  and records `live_fetch_error`. Preserve this.
- `_denormalize_live_surface(surface)` — extracts real t0 physical values
  (temp, dewpoint, RH via Magnus, wind, pressure, **cape_j_kg**, **cin_j_kg**, tcwv, rain).
- `_compute_live_thermo(surface_state, pressure, analysis_t0)` — live CAPE/CIN/shear/wind.
  See §12. Does **not** read `current_thermo` (test-enforced).
- `_level_for_prob(p)` — the authoritative risk classifier (§10).
- `get_point_inspection(lat, lon, lead_hours, mode)` — per-coordinate forecast.
- `derive_compound_risks(severe_prob, rain_pred, dem_norm)` — module-level, **shared by
  both modes** so live and historical differ only in atmospheric input, never in how
  risk is derived.

### `weather-app/backend/main.py` (1387 lines) — FastAPI
Serves the frontend at `/` and 40 route decorators (some paths are aliased, e.g. `/health` and `/api/health`). Full list in §3.1.
- `_get_service()` — dependency; 503 on init failure.
- `_live_unavailable_payload(...)` — uniform "not ready" response carrying the **real** reason.
- `_get_station_observations(force=False)` — OWM sweep, `_OBS_TTL_SECONDS = 240`,
  guarded by `_OBS_LOCK`. Failed stations are **dropped, not zero-filled**.
- Background scheduler: `@app.on_event("startup")` runs live refresh every **300 s**.

#### 3.1 API endpoints (verified present)
```
/  /app  /dashboard  /index.html          → frontend
/api/health  /api/system/info  /api/model-info  /api/data-health
/api/time/reference                        → AUTHORITATIVE reference + horizons
/api/nowcast/summary        ?lead&mode
/api/nowcast/districts      ?lead&mode
/api/nowcast/point          ?lat&lon&lead&mode
/api/nowcast/risk-map       ?lead&mode
/api/nowcast/risk-surface   ?lead&mode     → FORECAST PNG
/api/nowcast/risk-surface/bounds
/api/nowcast/high-risk-cells ?lead&top_k&mode
/api/nowcast/thermodynamics ?district&mode
/api/nowcast/xai            ?mode
/api/nowcast/explanation
/api/observations/current   ?variable      → OBSERVED station data + provenance
/api/observations/surface   ?variable      → OBSERVATION PNG (NOW layer)
/api/live/surface  /api/live/refresh  /api/live/ml-status
/api/weather/current        ?lat&lon
/api/benchmark/models  /api/mode/status  /api/geo/areas
/api/boundaries/state  /api/boundaries/west-bengal
/api/satellite/info  /api/predict  /api/stormsense/nowcast  /api/stormsense/timeline
```

### `weather-app/js/app.js` (3390 lines) — frontend engine
Vanilla IIFE, no build step. Critical globals:
- `window.currentLeadHours` — `'now'` (string) **or** `2|3|4|5|6` (number). The
  string/number distinction is load-bearing; `apiLeadHours()` coerces `'now'` → `2`
  **for API calls only**, never for display or surface selection.
- `window.currentBenchmarkLead`, `window.stormSenseMode` (`'live'|'historical'`)
- `window.stormSenseMap` (nowcasting map), `window.stormSenseRadarMap` (radar view)
- `window.StormSenseUserLocation`, `window.StormSenseUserPoint`, `window.StormSenseObservations`

Key functions (see §5, §6, §9):
`updateDynamicTimes()`, `formatIstStamp()`, `formatIstClock()`,
`renderContinuousRiskSurface()`, `renderObservationSurface()`,
`removeObservationSurface()`, `removeRiskSurface()`, `applyHorizonLegend()`,
`setForecastHorizon()`, `renderBenchmarkCards()`, `applyPointRiskMeters()`,
`renderThermodynamics()`, `renderThermodynamicsLive()`, `isRadarFeedsMap()`.

### `weather-app/index.html` (1544 lines)
UTF-8 **with BOM** (verified). Tailwind via CDN, dark theme. Contains all 8 views as
`<div id="view-*">`, hidden/shown by `switchNowcastView()`.

### Other inference files
- `predictor.py` (288) — checkpoint load, normalization, derived features, calibration
  (temperature scaling + per-lead thresholds).
- `risk_map.py` (271) — `compute_valid_time(issue_time, lead_hours)` (**the temporal
  helper**), `predictions_to_geojson()`, `_risk_level()` (Low/Moderate/High/Very High —
  a *different, internal* scale from the public one; see §10).
- `risk_surface.py` (219) — `get_or_create_wb_mask()`, `generate_risk_surface_png()`
  (cubic interpolation of the 33×25 model grid → 300×200 RGBA, WB-clipped).
  Bounds: `WB_BOUNDS_LEAFLET = [[21.5394, 85.8325], [27.2206, 89.8828]]`.
- `observation_surface.py` (374) — **NEW this session**; see §6.
- `xai.py` (136) — rule-based attribution; see §11.

---

## 4. GFS LIVE DATA PIPELINE

**Why GFS, not ERA5:** ERA5 has ~5-day publication latency and cannot serve a live
nowcast. GFS publishes a genuine 0-hour analysis (**f000**) every 6 hours
(00/06/12/18 UTC).

**Source:** AWS Open Data — `https://noaa-gfs-bdp-pds.s3.amazonaws.com`
**Fallback:** NOMADS — `https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod`

**Byte-range .idx approach:** Each GRIB2 file has a `.idx` index. The pipeline parses
it, matches messages by **exact `(name, level)` equality** (loose substring matching
would let `TMP:2 m` also match `TMP:2 mb`, a pressure level), and issues HTTP
byte-range requests for only the needed messages.

### f000-only policy (critical)
**Only f000 messages are ever fetched.** GFS forecast hours f001+ are future
predictions; using them as input timesteps would be temporal leakage — the model
would see the future and appear more skilful than it is. This was a real bug
(f003/f006/f009 used as "past" states) and is now structurally prevented.

### Cycle selection
- `_latest_published_cycle()` returns the most recent cycle whose f000 `.idx` is
  **actually published** (verified by HTTP), not merely whose nominal time has passed.
- Typical NCEP publication latency ≈ **5 hours** (`production_lag_hours`).
- The 6 input slots are hourly: `analysis_t0 - 5h … analysis_t0`.
- GFS has no hourly analysis product, so intermediate slots are **linearly
  interpolated between two REAL GFS ANALYSIS fields/states produced through data assimilation**. Every slot's timestamp and
  provenance (real vs interpolated) is tracked in `slot_provenance` and returned
  for audit. This smooths genuine sub-6h transients — a disclosed limitation.

### The three distinct timestamps — never conflate
| Name | Meaning | Example (observed) |
|---|---|---|
| `live_analysis_time` / `analysis_t0` | Real GFS ANALYSIS field/state produced through data assimilation (f000) the input came from | `2026-09-11T12:00:00+00:00` |
| `live_reference_time` | Exact wall-clock instant the forecast is issued for | `2026-09-11T17:08:43.453099+00:00` |
| forecast valid time | `reference_time + lead_hours` | `2026-09-11T19:08:43.453099+00:00` |

Observed lag in testing: **5.1–5.5 h** between analysis and wall clock. This is
**displayed honestly** as "Input analysis", never as "now".

### Grid & variables
- Domain: **lat 20.0–28.0 N, lon 84.0–90.0 E**, 0.25°, `grid_shape: [33, 25]`
- **Latitude DESCENDING (28 → 20); longitude ASCENDING (84 → 90).** Load-bearing.
- `SINGLE_VARS = ["u10","v10","d2m","t2m","sp","cape","cin","tcwv","tp"]`
  - `tcwv` ← GFS **PWAT**; `tp` ← GFS **PRATE × 3600** (mm/hour);
    `cin` clipped to ≥ 0 to match ERA5 denormalization.
- `PRESSURE_VARS = ["u","v","z","q","t"]`; **u,v only at 1000/850/700 mb**;
  **z,q,t only at 700/500/300/250 mb** (mirrors ERA5 group coverage; other levels NaN).
  `z` ← GFS **HGT × 9.80665** (height → geopotential).
- **DEM:** static SRTM elevation, loaded from the ERA5 memmap cache
  (`dem_elevation_m.npy`) and reused for live. DEM never changes and is not ingested
  from GFS. If absent it is zero-filled **with a printed warning** — the one
  documented zero-fill, and it is a static terrain field, not a meteorological variable.

### No-zero-fill rule
No meteorological variable is ever zero-filled, climatologically guessed, or
fabricated. A missing message raises immediately.

---

## 5. TEMPORAL SEMANTICS

### Backend
`/api/time/reference?mode=live` is **authoritative**:
```json
{
  "mode": "live",
  "reference_time": "2026-09-11T17:13:06.424093+00:00",
  "reference_time_label": "NOW",
  "horizons": [
    {"lead_hours": 2, "label": "+2h", "valid_time": "2026-09-11T19:13:06.424093+00:00", "offset_seconds": 7200},
    {"lead_hours": 4, "label": "+4h", "valid_time": "2026-09-11T21:13:06.424093+00:00", "offset_seconds": 14400},
    {"lead_hours": 6, "label": "+6h", "valid_time": "2026-09-11T23:13:06.424093+00:00", "offset_seconds": 21600}
  ],
  "analysis_time": "2026-09-11T12:00:00+00:00",
  "analysis_lag_hours": 5.21
}
```
(Verified live during this session. Microseconds are preserved — never floored.)

In live mode `reference_time = datetime.now(timezone.utc).isoformat()`. Valid times
come from `compute_valid_time(reference, lead)` in `risk_map.py` = `dt + timedelta(hours=lead)`.

### Frontend
- `updateDynamicTimes(issueTimeStr, leadHours, analysisTimeStr)` writes every timing
  label from the **same** authoritative instant the API supplied.
- `formatIstStamp(date)` → `"12 SEP · 02:02 IST"`. Uses **UTC+5:30 arithmetic on the
  epoch** (`getUTCHours()` after adding 5.5 h), so date/month/year rollovers are
  correct regardless of the viewer's own timezone.
- **IST is primary** (monitored region is West Bengal); UTC is secondary provenance.

### Anchoring subtlety (important — do not "fix" this)
The displayed valid time is anchored to the **`issue_time` of the forecast actually
painted on the map**, not to a freshly-read wall clock. `/api/time/reference` returns a
new instant on every call, so a page showing a forecast issued at 18:32 will display
`18:32 + lead`, which may be ~3 min behind a clock read at 18:35.

**This is correct.** Re-anchoring the label to live wall-clock would make it claim a
validity the displayed field does not have. A browser test initially "failed" on this
and the *test* was wrong, not the code.

### What was fixed to prevent temporal leakage
1. f003/f006/f009 no longer usable as past inputs (f000-only).
2. Publication-aware cycle selection (only genuinely published cycles).
3. `live_reference_time` separated from `live_analysis_time`.
4. A `timedelta` bug in `compute_valid_time` fixed.
5. NOW no longer renders the +2h forecast field (see §6).
6. `updateDynamicTimes()` no longer stamps `"+2h"` over the legend while NOW is selected.

### Tests proving timing
- `tests/test_exact_wall_clock.py` (8 tests)
- `tests/test_temporal_leakage.py` (5 tests)
- `tests/test_audit_invariants.py` — parametrized exact-offset tests across non-zero
  minutes/seconds, hour/date/year rollover
- `tests/test_observation_surface.py` — exact 7200/14400/21600 s + microsecond preservation

---

## 6. CURRENT DATA / NOW MODE ⚠️ MOST IMPORTANT SECTION

### Status: **IMPLEMENTED and browser-verified** (this session)

### What genuine current observations are available
**Provider:** OpenWeatherMap Current Weather — `https://api.openweathermap.org/data/2.5/weather`
(the project's pre-existing integration in `weather-app/backend/openweather.py`;
API key in `weather-app/.env` as `OPENWEATHER_API_KEY`).

**Investigation performed before implementing** (do not skip this reasoning):
- Probed 12 points across the domain → **12/12 distinct coordinates returned**,
  including pairs only 0.25° apart resolving to genuinely different stations
  (Dum Dum 28.0 °C / 63 % cloud vs Bhātpāra 27.2 °C / 44 %).
- `base: "stations"`, `dt` timestamps **0–8 minutes old** → genuinely current
  observation data, **not** forecast.
- Real spatial variation: 18.3–28.7 °C, cloud 30–94 %, live rain in the north.

**Conclusion:** sufficient for a defensible *interpolated* current field — but **not**
a gridded observed product.

### What NOW currently displays
- **A blue/teal observed-rainfall heatmap** from `/api/observations/surface?variable=rain_1h_mm`
- **24 station markers** (small cyan circles) showing where values are actually measured
- The violet live-location marker (if geolocation granted)
- Legend title: `CURRENT OBSERVED RAINFALL`
- Legend subtext: *"Observed now at reporting stations · values between stations are
  interpolated, not measured · not an AI forecast"*
- Provenance line (`#obs-provenance`), e.g.:
  `24/24 stations · Rainfall (last hour) up to 3.01 mm · observed ≤6 min ago · interpolated between stations`

### What NOW intentionally does NOT display
- **No forecast risk field.** `renderContinuousRiskSurface()` short-circuits on
  `leadHours === 'now'`, calls `removeRiskSurface()` then `renderObservationSurface()`.
- **No RainViewer.**
- **No extrapolation** beyond `MAX_INFLUENCE_DEG = 0.9` (~100 km) from any station —
  those pixels stay fully transparent.

### The three honesty safeguards (in `src/inference/observation_surface.py`)
1. Pixels >0.9° from every station → **NaN → transparent**, never guessed.
2. A station not reporting a variable is **dropped from that variable's interpolation**,
   never zero-filled. Test-enforced: 3 of 4 stations used when one lacks rainfall.
3. The 24 real stations are **drawn on the map**, so measurement vs interpolation is visible.

Interpolation: **IDW, power 2.0**. Declared in the API payload
(`is_interpolated: true`, `is_forecast: false`, `interpolation_method`,
`provenance_note`) and in PNG headers (`X-Data-Kind: observation`, `X-Is-Forecast: false`).

### Why a continuous NOW **AI** heatmap must never be fabricated
The AI model outputs *forecast* risk. Painting it at NOW would present future model
output as a present-tense observation. This was a **real bug** that existed in this
codebase and was fixed. Do not reintroduce it in any form.

### Limitations of the current NOW field (state these, don't hide them)
- 24 stations for ~89,000 km² ≈ one per 3,700 km². Convective rain is patchy far below
  this scale: a real cell between stations can be **missed entirely**, and a cell at one
  station is **smeared** over its neighbourhood.
- IDW contains **no meteorology** — no terrain, no advection. It cannot know rain stops at a ridge.
- `rain_1h_mm` is a 1-hour **accumulation**, not an instantaneous rate — it lags fast cells.
- OWM values are model-assimilated station data, not raw gauges. Upstream network and
  latency are outside our control.
- 4-minute server cache → field can be ~4 min staler than the stated observation time.

### Rules for the next Claude
- A future forecast risk field is **NOT** allowed to be displayed as NOW.
- RainViewer must **NOT** be used as a hidden substitute on the Interactive Nowcasting Map.
- If observations are insufficient, **do not manufacture** a continuous field — leave it honest.

---

## 7. FORECAST MODES (+2H / +4H / +6H)

| Aspect | Value |
|---|---|
| Valid timestamp | `reference_time + lead` (exact seconds; IST primary, UTC secondary) |
| Model input timestamp | `analysis_t0` — real GFS f000, typically 5 h earlier — shown as "Input analysis" |
| Risk field | `/api/nowcast/risk-surface?lead=N&mode=live` — continuous PNG, green→amber→orange→red |
| Forecast strip | 5 cards (+2h…+6h): severe-risk %, mm/3h, flood proxy, **IST valid stamp**, **"+N:00 from now"**, UTC |
| Strip badge | `AI FORECAST` (live) / `HISTORICAL FORCING` (historical) — **never** `LIVE OBSERVATION` |
| Map header | `+NH FORECAST` / `12 SEP · 02:02 IST` / `+N:00 FROM NOW` / `Valid … · Input analysis …` |
| Station markers | **Absent** (they belong to NOW only) |

**Forecast vs observed:** severe-weather probability, 3 h rainfall, flash-flood proxy
and overall risk are **model predictions**. The only observed quantities in live mode
are the OWM station readings (NOW layer, Current Location panel) and the GFS analysis
state used as *input*.

Supported leads: `SUPPORTED_LEADS = [2, 3, 4, 5, 6]`. The map exposes 2/4/6; the strip
shows all five; the Benchmark Desk exposes 2/4/6.

---

## 8. CURRENT LOCATION

**Flow:** `acquireLiveLocation()` → `navigator.geolocation.getCurrentPosition`
(timeout 8 s, `enableHighAccuracy`, `maximumAge` 60 s) → `window.applyLiveLocation(loc)`
→ `GET /api/nowcast/point?lat&lon&lead&mode` → `applyPointRiskMeters(d)`.

**Displays:**
- District name resolved from the real boundary GeoJSON, plus the user's coordinates.
- **Section 1 — Current Observations** (emerald header) with observed age
  (`#location-obs-age`, e.g. "observed 22:47 IST · just now"): temperature, rainfall,
  humidity, wind. Missing values → `—` **with no unit appended**.
- **Section 2 — AI Forecast Risk · This Location** (cyan header) with a horizon tag
  (`#location-risk-horizon`, e.g. "AI forecast · +2h"): thunderstorm %, rainfall mm,
  flash-flood proxy — bound to the **point forecast at the user's own coordinates**.

**The bug that was fixed (do not regress):** the risk meters were painted from the
**state-wide** summary, so a user in South 24 Parganas saw the statewide peak district's
numbers as their own. Verified fixed: with geolocation at South 24 Parganas, statewide
read `Cooch Behar (39%)` while the local meters read `0%`, matching `/api/nowcast/point`.

`paintForecastCards()` must **not** write `#meter-*` — that is the location card's job.

**Statewide values** live in the Bulletins card, labelled *"Active High-Risk Area
(state-wide)"*. KPI cards are labelled *"… · WEST BENGAL PEAK"*. The dashboard subtitle
states the distinction.

**Geolocation denied:** heading shows **"Location unavailable"**, meta explains
state-wide monitoring is shown instead, **no marker is drawn**, and no location is invented.

**Live-location marker colour: violet `#a855f7`** — deliberately outside the risk ramp
(`#10b981` / `#f59e0b` / `#f97316` / `#ef4444`) so it can never read as a severity level,
and distinct from the cyan AI-forecast accent. Test-enforced.

**Never show as local current weather:** another district's values, statewide aggregates,
forecast values without a forecast label, or any hardcoded constant.

---

## 9. MAP (Interactive Nowcasting Map)

- **Container:** `#map-radar-placeholder`; **Leaflet map:** `window.stormSenseMap`
- **Base tiles:** Esri World Dark Gray Base (no API key, no watermark)
- **Model domain:** lat 20–28 N, lon 84–90 E, 0.25°, 33×25 = 825 cells
- **Render bounds:** `WB_BOUNDS_LEAFLET = [[21.5394, 85.8325], [27.2206, 89.8828]]`
- **Mask:** `Data/BOUNDARIES/west_bengal_full.geojson` (all **23** districts — an earlier
  file had only 13 and silently clipped the map), rasterized to 300×200 and cached at
  `Data/BOUNDARIES/wb_mask_full_300x200.npy`
- `fitBounds(wbBounds, padding 20)`, `setMaxBounds(wbBounds.pad(0.35))`,
  `setMinZoom(getBoundsZoom(wbBounds))` — keeps the map framed on West Bengal
- **Layers:** state boundary, district boundaries, **exactly one** of
  {forecast risk overlay, observation overlay}, optional grid-inspection layer,
  station markers (NOW only), live-location marker

### Allowed on this map
Forecast risk surface (+2/+4/+6h), observation surface (NOW), station markers (NOW),
one live-location marker, boundaries, legend HUD, map tools.

### Forbidden on this map
- ❌ RainViewer radar (tiles or attribution)
- ❌ Fabricated markers — a red "IMD Kolkata DWR" marker and a green "Live Telemetry
  Station" marker (with invented 32.0 °C / 1006 hPa / 70 % / 16.0 km/h fallbacks)
  were **removed at source**. Do not restore them.
- ❌ "PREDICTED HIGH-RISK ML CELL #n" beacons (`renderHotspotBeacons()` now only clears)
- ❌ Any arbitrary coloured dot not backed by real data

### RainViewer enforcement (exact implementation)
```js
function isRadarFeedsMap(map) {
  return !!map && !!map.getContainer &&
         map.getContainer().id === "radar-map-container";
}
async function loadRainViewerRadar(map) {
  if (!isRadarFeedsMap(map)) {
    console.warn("loadRainViewerRadar: refused …");
    return;                      // structural guard
  }
  …
}
```
Plus: `initMap()` contains **no** `loadRainViewerRadar` call, and `initRadarMap()` is
**lazily** created on first visit to the radar view (it used to run at page load,
fetching radar tiles while the user was on the Dashboard).

**Verified:** 0 RainViewer requests while on Dashboard; `tileFound: false`; no
RainViewer attribution on the nowcasting map; 1 RainViewer tile layer on the radar map.

---

## 10. RISK CLASSIFICATION

### Authoritative classifier — `NowcastService._level_for_prob(p)` (verified in code)
```python
if p >= 0.75:   return "red"       # WARNING
elif p >= 0.50: return "orange"    # ALERT
elif p >= 0.25: return "yellow"    # WATCH
else:           return "green"     # NORMAL
```

### Public legend (verified — matches the classifier exactly)
| Band | Label | Colour |
|---|---|---|
| `<25%` | Normal | emerald `#10b981` |
| `25–50%` | Watch | amber `#f59e0b` |
| `50–75%` | Alert | orange `#f97316` |
| `≥75%` | Warning | red `#ef4444` |

The user's intended ranges (`<25 / 25–50 / 50–75 / ≥75`) were **verified against the
code and are correct**. They were restored from git history (commit `da41ff4`) after a
previous pass removed them. They appear in the map legend HUD, the header severity
pills, and `applyHorizonLegend()` in `app.js` — all three kept in sync. Test-enforced
by `test_legend_bands_match_classifier_thresholds`.

**Values are probabilities** (0–1 from the model, ×100 for display), calibrated via
temperature scaling with per-lead thresholds
(`{2: 0.727, 3: 0.679, 4: 0.673, 5: 0.637, 6: 0.625}`).

**It is severe-weather risk, not simple rainfall.** Rainfall (mm/3h) is a separate
predicted quantity. Do not conflate them.

⚠️ **Second, different scale exists:** `risk_map.py::_risk_level()` returns
`Low/Moderate/High/Very High` at thresholds `0.15 / 0.35 / 0.60`. This is an *internal*
GeoJSON field, not the public legend. **Do not "harmonize" them without checking every
consumer** — they serve different purposes.

Shared helpers: `derive_compound_risks()` (module-level, used by both modes),
`_stage_for_level()`, `_action_for_level()`, `levelClass()` (frontend).

---

## 11. XAI / ATMOSPHERIC ATTRIBUTION

**File:** `src/inference/xai.py` — `calculate_xai_factors(xai_inputs: dict)`

> **This is rule-based, physics-inspired factor attribution — NOT a validated ML
> explanation method.** Not SHAP, not LIME, not integrated gradients. It does not
> inspect model weights or gradients at all.

**Inputs:** `rainfall_1h_mm`, `rainfall_3h_mm`, `rainfall_6h_mm`, `humidity_percent`,
`dew_point_c`, `cape_jkg`, `wind_speed_kmh`, optional `radar_dbz`.
(CIN and shear are *not* currently inputs — **UNVERIFIED** whether that is intentional.)

**Scoring:** weighted arithmetic, e.g. `rainfall_score = rainfall_1h*5 + rainfall_3h*2 + rainfall_6h`,
clamped 0–100, then normalized into percentage contributions.

**Output:** four factors — *Convective Instability (CAPE)*, *Moisture / Humidity*,
*Wind Influence*, *Recent Rainfall* — each with `name`, `value`, `physical_role`,
`importance_rank`, `impact` (`Dominant` / `High` / `Moderate` / `Secondary`),
sorted by contribution. Envelope: `attribution_method: "Physics-Inspired Factor Attribution"`,
`parameters: "Rule-Based Physics Proxy"`.

**UI:** XAI card sits on the **right** of the Bulletins/XAI grid. Method line reads
**"Physics-based rules (not learned attribution)"**; subtitle **"Rule-based atmospheric
factor attribution"**.

**Claims NOT allowed:** "validated ML explainability", "SHAP/LIME", "the model's actual
reasoning", or any implication the attribution was independently validated.

---

## 12. ATMOSPHERIC INSTABILITY (live diagnostics)

**Endpoint:** `/api/nowcast/thermodynamics?mode=live` → `NowcastService.live_thermo`,
computed by `_compute_live_thermo()` from the **real GFS analysis only**.

**Verified live output during this session:**
```
cape_j_kg: 916.0                       (real GFS CAPE)
cin_j_kg: 0.0                          (real GFS CIN)
bulk_shear_1000_700hpa_mps: 2.4
bulk_shear_label: "1000–700 hPa bulk shear (~0–3 km)"
surface_wind_kmh: 16.2                 (GFS 10 m u10/v10)
bulk_shear_0_6km_mps: null
bulk_shear_0_6km_status: "Not computed: live GFS ingestion carries wind at
                          1000/850/700 hPa only, so winds near 6 km are not available."
analysis_time_utc: "2026-09-11T12:00:00+00:00"
data_source: "NOAA GFS 0.25° GFS ANALYSIS field/state produced through data assimilation"
```

### ⚠️ Why true 0–6 km shear is unavailable
Live GFS ingestion carries wind at **1000/850/700 hPa only** (`WIND_LEVELS_HPA`).
700 hPa ≈ **3 km**. Winds near 6 km are simply not in the input tensor.

**The shear shown is a 1000→700 hPa (~0–3 km) bulk shear and is labelled as such.**
**Do NOT relabel it as 0–6 km shear.** `bulk_shear_0_6km_mps` stays `null` with a
stated reason. Test-enforced (`test_live_thermo_reports_shear_layer_truthfully`).

Historical mode serves a genuine 0–6 km shear from the ERA5 profile via `current_thermo`.
`renderThermodynamics()` branches on `mode`/`source` and reads the matching field —
reading the wrong field for the wrong mode would blank live values or mislabel their extent.

### The bug that was fixed
The UI hardcoded `"Unavailable (Live)"` for CAPE/CIN and printed **"Station Anemometer"**
for shear — claiming a station instrument StormSense does not read — *while the live GFS
analysis already carried real CAPE and CIN*. It also fell back to an invented 1006 hPa.
`live_thermo` is kept **strictly separate** from `current_thermo` so historical soundings
can never be served as live (test-enforced).

The `Source:` line is now written at runtime (`#thermo-source-label`) from the payload
that supplied the values — never hardcoded.

---

## 13. RADAR / SATELLITE

**RainViewer** (`https://api.rainviewer.com/public/weather-maps.json` → tile URLs)
is used **only** in the **Radar & Satellite Feeds** view (`#view-radar`,
map container `#radar-map-container`, `window.stormSenseRadarMap`).

- **Lazy loading:** created on first visit via `switchNowcastView('radar')`, not at page load.
- **Layer:** `L.tileLayer(tileUrl, {opacity: 0.65, zIndex: 400, maxNativeZoom: 7, errorTileUrl: <1px gif>})`
- **Status HUD:** `#radar-status` (`RADAR LOADING` → `RADAR READY` / `RADAR OFFLINE`),
  `#radar-info`, `#radar-last-update` (last scan time from the frame timestamp).
- **Error handling:** try/catch → `RADAR OFFLINE` + "Radar feed unavailable"; never fabricates frames.
- **Why not on the nowcasting map:** overlaying a third-party precipitation mosaic made
  external radar imagery look like StormSense output and conflated *precipitation* with
  *severe-weather risk*. Enforced by `isRadarFeedsMap()` (§9).

Also in this view: an AI Spatial Forecast panel and an **INSAT-3DR** archive panel
(episodic HDF5 imagery, 14 storm events 2020–2024, used for qualitative validation —
explicitly **not** a live satellite feed).

---

## 14. HISTORICAL MODE

Toggled by `window.toggleOperationalMode()` (button top-right of the header).

- **Frozen case study:** ERA5 memmap sample #100 — a real **Kalbaishakhi pre-monsoon
  squall, 2024-05-05 15:00 UTC** (`current_valid_time = "2024-05-05T15:00:00Z"`).
- Uses `current_pred` / `current_thermo` / `risk_surface_png_cache` — never live state.
- **No live-location marker** (removed on entering historical; verified).
- **No observation surface** (`renderObservationSurface()` returns early in historical mode).
- Labels: `HISTORICAL CASE STUDY · KALBAISHAKHI`; strip badge `HISTORICAL FORCING`.
- Thermodynamics: genuine ERA5 0–6 km shear.

### Held-out ERA5 test metrics (`outputs/metrics/test_evaluation.json`)
Mean across leads: **CSI 0.3068, PR-AUC 0.4840, POD 0.5428, FAR 0.5891, Brier 0.0512, ECE 0.0760**
Per-lead CSI: +2h **0.4054**, +3h 0.3341, +4h 0.2925, +5h 0.2605, +6h 0.2413
Test set: May–Oct 2024, 4,322 sequences, 3,565,650 grid evaluations.

### GFS production backtest (`reports/gfs_production_backtest.json`, generated 2026-09-11)
Methodology: *"production rule: only cycles published by the simulated NOW, assuming 5.0 h
production latency"*; input *"real GFS GFS ANALYSIS fields/states produced through data assimilation only; no forecast hour used as an
input timestep"*; inference path = `NowcastService.refresh_live_state` (the production path);
truth = **ERA5-derived severe-weather proxy at the valid time**.

30/30 cases completed, 24,750 samples per horizon:

| Horizon | Prevalence | CSI | POD | FAR | PR-AUC | No-skill PR-AUC |
|---|---|---|---|---|---|---|
| +2h | 0.0411 | **0.1045** | 0.1900 | 0.8115 | 0.0996 | 0.0411 |
| +4h | 0.0419 | **0.0832** | 0.1455 | 0.8371 | 0.1165 | 0.0419 |
| +6h | 0.0312 | **0.0867** | 0.1816 | 0.8576 | 0.0981 | 0.0312 |

### ⚠️ Read this honestly
**Live GFS skill is ~4× weaker than the ERA5 held-out metrics** (CSI 0.104 vs 0.405 at +2h),
FAR is ~0.81–0.86, and PR-AUC is only ~2.4× the no-skill baseline. Causes: GFS↔ERA5
distribution shift, 5 h analysis latency, and sub-6h interpolation smoothing.

**The benchmark desk shows the ERA5 test metrics.** Those are *not* live skill.
Do not present them as such.

### Why ERA5-derived targets ≠ observed reports
The verification target is an **ERA5-derived proxy** for severe weather, not IMD storm
reports, lightning strikes, or damage surveys. A proxy can be systematically biased and
correlated with the model's own inputs. Never call these "observed severe-weather reports".

**Sample size caveat:** 30 cases, convective season only — modest and seasonally biased.

---

## 15. DATA LEAKAGE / SCIENTIFIC VALIDITY

### Fixed
1. **f003/f006/f009 misused as "past" input** → **f000-only policy**. The single most
   serious leakage bug: using forecast hours as input timesteps let the model see the future.
2. **Cycle selection ignored publication latency** → publication-aware selection
   (`_latest_published_cycle()` verifies the `.idx` exists) + `production_lag_hours ≈ 5 h`
   in backtesting, so a backtest cannot use a cycle that was not yet published at the simulated NOW.
3. **Model input time conflated with forecast valid time** → `analysis_t0`,
   `live_reference_time`, and valid time are three separate, separately-displayed fields.
4. **Verification against current time instead of forecast valid time** → targets are now
   constructed at the **valid time**.
5. **Independent target construction** → ERA5-derived proxy built independently of the
   GFS inputs (different NWP system entirely).
6. **NOW displaying the +2h forecast field** → NOW now renders only observations.

### Remaining scientific limitations (disclosed, unfixed)
- GFS↔ERA5 distribution shift (NCEP vs ECMWF; different assimilation and physics).
- Sub-6h transients smoothed by interpolation between 6-hourly analyses.
- ERA5-derived proxy targets, not observed reports.
- 30-case, convective-season-only backtest.
- Live skill weak and miscalibrated (§14).

---

## 16. TESTING — **CURRENT VERIFIED STATUS**

### pytest (run 2026-09-12, this machine)
```bash
cd D:\Weather-Hackathon
python -m pytest tests/ -q -p no:cacheprovider \
  --ignore=tests/test_browser_e2e.py --ignore=tests/test_browser_live.py
```
**Result: `287 passed, 10 warnings` — 0 failures.** (Warnings are FastAPI `on_event`
and Pydantic deprecations, pre-existing and harmless.)

| Suite | Test defs | Covers |
|---|---|---|
| `tests/test_api.py` | 28 | API contracts |
| `tests/test_audit_invariants.py` | 30 | Encoding, temporal, map, markers, legend, layout, provenance |
| `tests/test_observation_surface.py` | 20 | NOW observation surface + temporal contract |
| `tests/test_smoke.py` | 27 | End-to-end smoke |
| `tests/test_exact_wall_clock.py` | 8 | Exact wall-clock semantics |
| `tests/test_map_domain.py` | 7 | Map domain/bounds |
| `tests/test_geo_mask.py` | 5 | Geographic containment |
| `tests/test_temporal_leakage.py` | 5 | Leakage guards |
| `tests/test_browser_e2e.py` | 24 | **Excluded** — needs a running server |
| `tests/test_browser_live.py` | 0 | No test defs |

> The two browser suites are **excluded** from the standard run because they require a
> live server on port 8000. **UNVERIFIED:** whether `test_browser_e2e.py` currently passes.

### Browser validation (Playwright, this session) — **132/132 checks passed**
Scripts lived in the session scratchpad (`…/scratchpad/`), which is **temporary and may
be gone**. Recreate as needed:

| Suite | Checks | Covers |
|---|---|---|
| core audit | 29 | Encoding, legend, RainViewer, markers, benchmark binding, thermo, layout, mobile |
| geolocation granted | 23 | Live marker, per-location meters, exact timing, historical round-trip |
| NOW semantics | 12 | NOW layer vs forecast layer, refresh persistence |
| all 8 views | 29 | Every view renders, no mojibake, radar isolation |
| strip labelling | 5 | Forecast cards badged `AI FORECAST` |
| NOW heatmap + timing | 34 | Observation source, provenance, IST stamps, offsets |

Key verified facts: 24/24 stations reporting, max observation age 313–466 s,
0 RainViewer requests on Dashboard, exact 7200/14400/21600 s offsets,
no U+FFFD in rendered DOM, no horizontal overflow at 420 px.

---

## 17. RECENT FIXES (this session and the audit before it)

### Temporal / leakage
- f000-only GFS policy; f003/f006/f009 no longer usable as past inputs
- Publication-aware cycle selection
- `live_reference_time` separated from `live_analysis_time`
- `compute_valid_time` `timedelta` bug fixed
- **NOW no longer paints the +2h forecast field** (was future data under a "CURRENT
  OBSERVATIONS" legend) — `renderContinuousRiskSurface()` short-circuits on `'now'`
- NOW **removes** the overlay rather than setting `opacity: 0` (an attached invisible
  layer could be repainted)
- `updateDynamicTimes()` no longer stamps `"+2h"` over the legend while NOW is selected
- Live refresh passes the **raw** horizon, not `apiLeadHours()` (which coerces `'now'`→2)

### Fabrication removal
- Fake green "Live Telemetry Station" marker (hardcoded 22.724/88.479, invented
  32.0 °C / 16.0 km/h / 1006 hPa / 70 %) — **deleted at source**
- Fake red "IMD Kolkata DWR" marker — **deleted at source**
- `renderNowcastLive` fallbacks (32.0 / 36.0 / 70 / 1006 / 16.0 / 6.5) → `—`
- Rainfall no longer defaults to `"0.0"` (absent ≠ zero rain)
- Unit suffixes no longer appended to `—` (was `"—°C"`)
- Hardcoded **"92% model skill"** removed
- "PREDICTED HIGH-RISK ML CELL #n" beacons removed
- Hardcoded live XAI removed; routed through `xai.py`

### Atmospheric diagnostics
- Live CAPE/CIN/shear/wind wired from the real GFS analysis (`_compute_live_thermo`)
- **"Station Anemometer"** label removed (claimed a non-existent instrument)
- Shear labelled `1000–700 hPa (~0–3 km)`; `bulk_shear_0_6km_mps: null` + reason
- `Source:` line bound at runtime instead of hardcoded

### Encoding
- **90 U+FFFD characters** in `app.js` reconstructed (a prior script read the file as
  cp1252 and rewrote it as UTF-8, destroying `°`, `–`, `—`, `·`). This corrupted
  *data-bearing* strings: `"°C"` → `"\uFFFDC"`, and the `—` no-data placeholder.
- 3 mojibake `?` restored to `→`
- Project-wide U+FFFD count now **0**

### UI / semantics
- Risk percentage ranges restored (`<25 / 25–50 / 50–75 / ≥75`) from git history
- Benchmark horizon labels bound dynamically (were frozen at "+2h Horizon" /
  "Lead Time: +2 Hours" while metrics changed); dead `bm-lead-display-tag` write removed
- Invented phrase "Critical Initiation" → factual "120-minute lead"
- Current Location meters bound to the **point** forecast, not statewide
- Current Location split into labelled Observations / Forecast sections
- Statewide "Active High-Risk Area" moved to Bulletins, labelled state-wide
- Bulletins **left** / XAI **right** side-by-side grid (was a tall empty gap)
- Historical Case Study button moved to true top-right (x 352 → 1158)
- Live-location marker → violet `#a855f7`
- Forecast strip badge corrected to `AI FORECAST` (was `LIVE OBSERVATION` on forecast cards)
- Pipeline badge `AI FORECAST ACTIVE` → `LIVE PIPELINE ACTIVE`
- Internal `V1`/`V2`/`V2 (Calibrated)` labels removed from public UI (8 sites)
- KPI cards labelled `· WEST BENGAL PEAK`
- Baseline Brier cells given real IDs + `(mean)` label
- 7 stale initial-markup horizon states neutralized
- RainViewer removed from nowcasting map + `isRadarFeedsMap()` guard + lazy radar init

### NOW heatmap (newest work)
- New `src/inference/observation_surface.py` (IDW, coverage mask, blue/teal ramp)
- New endpoints `/api/observations/current` and `/api/observations/surface`
- Frontend `renderObservationSurface()` / `removeObservationSurface()` /
  `drawObservationStations()` / `updateObservationProvenance()`
- NaN-before-integer-cast bug fixed in the colormap (undefined behaviour)

### Infrastructure
- `tests/test_geo_mask.py` native GEOS **segfault** fixed: replaced `buffer()` on a
  ~1.4 M-char polygon with an exact `distance()` check. Mutation-verified equivalent
  (accepts all in-state, rejects all out-of-state points, 0.0102° tolerance).

---

## 18. CURRENT UI STATE — the 8-view shell

Navigation is a left sidebar (`xl:flex`, hidden below `xl`); `switchNowcastView(id)`
toggles `#view-*` containers.

| # | View id | Nav label | Contents |
|---|---|---|---|
| 1 | `dashboard` | Dashboard | KPI cards, 0–6 h forecast strip, Interactive Nowcasting Map, Current Location, Bulletins + XAI |
| 2 | `radar` | Radar & Satellite Feeds | AI Spatial Forecast + **RainViewer**, INSAT-3DR archive, high-risk grid cells |
| 3 | `advisories` | District Advisories | Per-district advisory cards |
| 4 | `wrf` | AI Nowcast & Benchmark | Horizon selector (2/4/6), 3-way model comparison, evolution timeline, **Atmospheric Instability State**, verification table |
| 5 | `xai` | XAI Feature Attribution | Rule-based factor attribution |
| 6 | `gis` | GIS Spatial Layers | Spatial layer controls |
| 7 | `threshold` | Calibrated Thresholds | Per-lead calibrated thresholds |
| 8 | `streams` | Data Ingestion Streams | Ingestion status |

**Header:** IST clock (1 s tick) + observed time; mode badge; severity pills
(`<25% Normal` … `≥75% Warning`); Export Report; Broadcast Advisory;
**View Historical Case Study →** (top-right, last element).

**Banner:** `Live monitoring · AI-derived severe weather risk · Not an official
government warning` + target region.

**Horizon controls:** `NOW | +2h | +4h | +6h` beside the prominent temporal banner
(`+2H FORECAST` / `12 SEP · 02:02 IST` / `+2:00 FROM NOW` / `Valid … · Input analysis …`)
and the pipeline badge.

**Typography:** Plus Jakarta Sans (UI) + JetBrains Mono (data), Material Symbols icons,
dark theme `#0B0F19`. **Responsive:** verified no horizontal overflow at 420 px;
Bulletins/XAI stack on narrow screens.

---

## 19. PUBLIC UI / TERMINOLOGY RULES

### MUST NOT appear in public UI
- `V1`, `V2`, `V2 (Calibrated)`, `SevereWeatherNet V2`
- `ERA5`, `ECMWF`, `Copernicus` (as live provenance)
- `OpenWeather` / `OpenWeatherMap` as a **headline provider label** —
  *(note: it IS named in the `/api/observations/current` `source` field for provenance,
  which is appropriate; keep it out of prominent UI chrome)*
- `Open-Meteo`, raw tensor/dataset terminology (`memmap`, `tensor`, `dataloader`)
- Any "demo"/"mock" branding

### Approved public terminology
- **AI Forecast Model** (not V2), **Earlier Model Baseline** (not V1), **Persistence Baseline**
- **NOAA GFS 0.25° GFS ANALYSIS field/state produced through data assimilation** — correct and approved for live provenance
- **Physics-based rules (not learned attribution)** for XAI
- **Not an official government warning** on risk output
- Risk legend: `<25% Normal` / `25–50% Watch` / `50–75% Alert` / `≥75% Warning`

Current verified counts in `index.html`: `V1`/`V2` = **0**, `SevereWeatherNet` = 0.

---

## 20. KNOWN LIMITATIONS (be brutally honest)

1. **Live forecast skill is weak and miscalibrated** — CSI 0.104 at +2h vs 0.405 on the
   ERA5 held-out test (~4× weaker); FAR 0.81–0.86; PR-AUC only ~2.4× no-skill.
2. **GFS analysis latency 5.1–5.5 h** — the atmospheric input state is hours behind
   wall-clock. Disclosed, not hidden.
3. **Verification targets are ERA5-derived proxies**, not observed severe-weather reports.
4. **Modest backtest sample** — 30 cases, 24,750 samples/horizon, convective season only.
5. **Sub-6h transient smoothing** — hourly slots interpolated between 6-hourly analyses.
6. **GFS↔ERA5 distribution shift** — model trained on ERA5, served GFS.
7. **XAI is rule-based**, not validated ML explainability.
8. **NOW observation coverage is sparse** — 24 stations for ~89,000 km²; interpolated,
   no terrain/advection; 1-hour accumulation lags fast cells; 4-min cache.
9. **Geolocation depends on browser permission**; denied → "Location unavailable".
10. **`weather-app/.venv` is BROKEN** (§22).
11. **GEOS/PyTorch native instability** — the geo test segfaulted under full-suite memory
    pressure. Worked around, but the underlying native fragility remains.
12. **Repo hygiene** — **62** stray `.py` scripts (`fix_*.py`, `patch_*.py`, `test_*.py`, `diagnostic*.py`) in the
    project root from earlier sessions. Harmless but noisy. `CLAUDE_HANDOFF.md` is a
    Python *script*, not a document (§23).
13. **No CI**; tests are run manually.

---

## 21. UNRESOLVED / TODO

### ✅ DONE (verified this session)
- [x] **NOW browser verification after forecast-overlay removal** — 12/12 NOW-semantics
      checks; NOW paints 0 risk layers, 1 observation layer
- [x] **Genuine current NOW spatial field** — implemented from OpenWeatherMap stations,
      IDW-interpolated, disclosed; 34/34 browser checks
- [x] Forecast timing prominent everywhere (IST stamp + offset on banner and all 5 cards)
- [x] Stale horizon wording eliminated (`Critical Initiation`, `Lead Time: +2 Hours`)
- [x] RainViewer isolated to the radar view (structural guard + lazy init)
- [x] Encoding cleanup (0 U+FFFD project-wide)
- [x] Fabricated markers/values removed
- [x] Live CAPE/CIN wired; shear labelled honestly
- [x] Current Location bound to the user's own coordinates
- [x] Risk percentage ranges restored and verified against the classifier
- [x] Bulletins/XAI side-by-side; Historical button top-right
- [x] pytest 287/287

### ❓ UNVERIFIED
- [?] `tests/test_browser_e2e.py` (24 tests) — excluded from standard runs; current status unknown
- [?] Whether the 4-min observation cache TTL is right for the OWM plan's rate limit
      (24 calls per refresh; **plan tier UNKNOWN**)
- [?] Whether XAI *should* consume CIN and shear (currently it does not)
- [?] Long-run stability of the 300 s live-refresh scheduler (only observed for ~1 h)
- [?] Behaviour when OpenWeatherMap is unreachable — code path exists
      (`status: "unavailable"`, 503 on the PNG) but was **not** exercised end-to-end in a browser
- [?] Historical mode + NOW horizon interaction — `renderObservationSurface()` returns
      early in historical, but this specific combination was not explicitly browser-tested

### 🔲 NOT STARTED
- [ ] Repo cleanup of stray root-level scripts
- [ ] Restore `CLAUDE_HANDOFF.md` to being a document rather than a script
- [ ] Fix or delete `weather-app/.venv`
- [ ] CI pipeline
- [ ] Improve live GFS skill / recalibrate for the GFS distribution (the real scientific gap)
- [ ] Replace ERA5-proxy targets with observed severe-weather reports (would need IMD data)
- [ ] Additional observation variables on the NOW map (temperature/humidity/cloud are
      supported by the backend but only rainfall is exposed in the UI)

---

## 22. ENVIRONMENT / HOW TO RUN

### ⚠️ Interpreter — read before running anything
- **`weather-app/.venv` is BROKEN.** Its `pyvenv.cfg` points at
  `C:\Users\SUBHOJIT\AppData\Local\Python\pythoncore-3.14-64\python.exe`, which does not
  exist on this machine. **Do not use it.**
- `venv/` works (Python 3.13.7) but has **CPU-only torch 2.14.0+cpu**.
- **Everything in this session used system Python 3.13.7**
  (`C:\Program Files\Python313\python.exe`, torch 2.6.0+cu124). Use plain `python`.

### Start the backend (also serves the frontend)
```bash
cd D:\Weather-Hackathon
python run_server.py                 # http://127.0.0.1:8000
```
`PYTHONUTF8=1` is recommended — the Windows console is cp1252 and will crash on `≥`/`–`
when printing. Startup takes **~45–60 s** (model load + first GFS fetch). Wait with:
```bash
until curl -s -m 5 http://127.0.0.1:8000/api/health >/dev/null; do sleep 3; done
```

> **The frontend hardcodes port 8000** (`API_BASE` in `app.js` line ~21). Running on any
> other port breaks all API calls. Do not "helpfully" change the port.

**No separate frontend server and no build step** — `index.html` and `js/app.js` are
served as static files by FastAPI.

### Tests
```bash
python -m pytest tests/ -q -p no:cacheprovider \
  --ignore=tests/test_browser_e2e.py --ignore=tests/test_browser_live.py
```
(`--timeout` is **not** available — pytest-timeout is not installed.)

### Browser validation
Playwright + Chromium are installed. Write scripts to a scratch directory (not the repo),
and start them with:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console is cp1252
```
Point them at `http://127.0.0.1:8000`.

### Logs / stopping
Redirect server output to a log file and `tail` it. To stop:
```bash
for pid in $(netstat -ano | grep ":8000" | grep LISTENING | awk '{print $5}' | sort -u); do
  taskkill //F //PID $pid
done
```

### Secrets
`weather-app/.env` holds `OPENWEATHER_API_KEY` (32 chars, present and working).
`weather-app/backend/config.py` raises at import if it is missing.

---

## 23. IMPORTANT FILES

| Path | Purpose | Safe to modify? | Warnings |
|---|---|---|---|
| `src/inference/gfs_live.py` | GFS f000 ingestion + harmonization | ⚠️ With extreme care | **Never reorder `PRESSURE_LEVELS_HPA`.** Never allow f001+. Never zero-fill. |
| `src/inference/nowcast_service.py` | Core service, both modes | ⚠️ Care | Never mix live/historical state. Preserve stale-data protection. |
| `src/inference/observation_surface.py` | NOW observation surface | ✅ Yes | Preserve the 3 honesty safeguards (§6). Mask NaN before any int cast. |
| `src/inference/risk_surface.py` | Forecast risk PNG + WB mask | ✅ Careful | Mask cache at `Data/BOUNDARIES/wb_mask_full_300x200.npy`. |
| `src/inference/risk_map.py` | `compute_valid_time`, GeoJSON | ⚠️ Care | `_risk_level()` is a *different* internal scale from the public legend. |
| `src/inference/predictor.py` | Model load/inference/calibration | ⚠️ Care | Slices pressure levels **by index**. |
| `src/inference/xai.py` | Rule-based attribution | ✅ Yes | Never call it validated ML explainability. |
| `weather-app/backend/main.py` | FastAPI app | ✅ Careful | `_OBS_TTL_SECONDS` rate-limits OWM. Keep provenance headers. |
| `weather-app/backend/openweather.py` | OWM client | ✅ Yes | Needs `OPENWEATHER_API_KEY`. |
| `weather-app/js/app.js` | Frontend engine (3390 lines) | ✅ Careful | UTF-8, **no BOM**. Never bulk-rewrite encoding. `currentLeadHours` is `'now'` or a number. |
| `weather-app/index.html` | All 8 views | ✅ Careful | UTF-8 **with BOM**. `div` balance currently **390/390** (verified). |
| `configs/default.yaml` | Domain/grid/paths | ⚠️ Care | Its descending level list is **human reference only**. |
| `Data/BOUNDARIES/west_bengal_full.geojson` | 23-district outline | ❌ No | An earlier file had only 13 districts. |
| `tests/test_audit_invariants.py` | Anti-regression guards | ✅ Add to | Each test encodes a real bug. Don't weaken to make things pass. |
| `tests/test_observation_surface.py` | NOW surface guards | ✅ Add to | Enforces no-fabrication rules. |
| `CLAUDE_HANDOFF.md` | **Actually a Python script**, not a doc | ⚠️ Leave | Overwritten by a patch script in an earlier session. User asked to keep it. |
| `outputs/checkpoints/v2_calibrated_best.pt` | Trained model | ❌ No | 781,889 params. |
| `outputs/metrics/test_evaluation.json` | ERA5 held-out metrics | ❌ No | Source of benchmark numbers. |
| `reports/gfs_production_backtest.json` | Live-skill backtest | ❌ No | Source of §14 live metrics. |

---

## 24. GIT / CHANGE HISTORY

**Branch:** `master`. **Nothing was committed this session** — all work is uncommitted
in the working tree.

### Recent commits (newest first)
```
c5458e7  Final verification pass: Public UI cleanup and dynamic KPI resolution
eaf7902  Fix NowcastService concurrency MEMFS crash and live timestamp isolation
1f0d6ce  Refactor hardcoded GeoJSON paths to resolve dynamically from module location
3dd8763  Fix observation age formatting and backend 500 errors
f00c602  Fix JS parse crash, refresh btn layout, buffering states, newline artifact
f9a72db  Fix UI map breakage, honest data states, missing data, refresh button, dynamic timestamps
eebc83a  Add geographic validation tests; update API tests for working live mode
4f60d9d  Fix West Bengal boundary: previous file covered only 13 of 23 districts
```

### Uncommitted MODIFIED (tracked) — **all intentional, do not revert**
```
 M requirements.txt                    +6
 M src/inference/gfs_live.py           +281/-…   f000 policy, publication-aware cycles
 M src/inference/nowcast_service.py    +370/-…   live_thermo, live/historical separation
 M src/inference/risk_map.py           +6        compute_valid_time fix
 M src/inference/risk_surface.py       +93/-…    full-state mask, bounds
 M tests/test_api.py                   +19
 M tests/test_geo_mask.py              +17       GEOS segfault workaround
 M weather-app/backend/main.py         +479/-…   observation endpoints, live thermo
 M weather-app/index.html              +304/-…   layout, legend, timing banner
 M weather-app/js/app.js               +1849/-…  encoding repair, NOW layer, timing, markers
```
Total ≈ **+2561 / −863** across 10 files.

### Untracked NEW files to KEEP
```
src/inference/observation_surface.py   NOW observation surface
src/inference/xai.py                   rule-based attribution
tests/test_audit_invariants.py         30 anti-regression tests
tests/test_observation_surface.py      20 NOW-surface tests
reports/gfs_production_backtest.json   live-skill backtest
reports/DISTRIBUTION_SHIFT.txt
scripts/historical_backtest.py, distribution_shift.py, stress_test.py, print_metrics.py
FINAL_ACCEPTANCE_REPORT.md
CLAUDE_NEW_SESSION_HANDOFF.md          (this file)
```

### Untracked noise (safe to ignore or clean)
62 root-level `.py` scripts (`fix_*`, `patch_*`, `test_*`, `diagnostic*`), `screenshot.png`,
`test_*.png`, `scratch/`, `friend-work/`.

### ⚠️ Do NOT accidentally revert
The encoding repair in `app.js` (90 characters) is **not recoverable by re-running any
script** — the original bytes were destroyed. A `git checkout` of `app.js` would restore
the corrupted version and undo every frontend fix. **Never `git checkout`/`reset` these
10 modified files without explicit instruction.**

---

## 25. HANDOFF INSTRUCTIONS FOR THE NEXT CLAUDE

1. **Read this file completely before acting.**
2. **Then inspect the repository rather than trusting this handoff blindly.** Run the
   tests yourself; re-read any file before editing it.
3. **Do not undo existing fixes.** Every item in §17 addresses a real, verified defect.
4. **Do not change data provenance without explicit justification.** Every displayed
   value must trace to a real source with a real timestamp.
5. **Do not fabricate observations.** No fallback constants. Missing → `—` / `Unavailable`.
6. **Do not optimize against MSN** or any external weather site. Do not copy their values
   or visualizations.
7. **Do not use RainViewer to fake NOW** on the Interactive Nowcasting Map.
8. **Preserve exact NOW/+2/+4/+6 temporal semantics** (7200/14400/21600 s; never floored).
9. **Run the full pytest suite after every change** and fix what you break. Do not say
   "unrelated to my changes".
10. **Use skeptical scientific validation.** Assume a hostile judge is trying to disprove
    the app. Ask: where did this number come from? What timestamp? Measured or inferred?
11. **Clearly distinguish** observations / model inputs / forecasts / verification targets.
    These four are different things and the UI must never blur them.
12. **Never claim accuracy that has not been demonstrated.** Live skill is weak (§14) —
    say so.
13. **When a test fails after a legitimate behaviour change**, update the test *only* if
    the new behaviour is scientifically correct, and explain why. (Example: a test asserted
    NOW had no spatial field; once a genuine observation source was wired, the correct
    invariant became "no *forecast* at NOW", not "nothing at NOW".)
14. **Never bulk-rewrite file encodings.** That is what destroyed 90 characters in `app.js`.

---

## 26. FINAL STATE CHECKLIST

### ✅ Completed & verified
- [x] pytest **287/287 passing**, 0 failures
- [x] Browser validation **132/132 checks** across 6 suites
- [x] NOW shows a genuine current-observation heatmap (24 OWM stations, IDW, disclosed)
- [x] NOW shows **no** forecast field; forecast horizons show **no** observation field
- [x] Exact +2/+4/+6 = 7200/14400/21600 s, sub-hour precision preserved
- [x] Prominent IST valid time + "+N:00 FROM NOW" on banner and all 5 forecast cards
- [x] GFS analysis time displayed separately from wall-clock reference
- [x] RainViewer only on Radar & Satellite Feeds (structural guard + lazy init)
- [x] 0 U+FFFD project-wide
- [x] Risk bands `<25 / 25–50 / 50–75 / ≥75` match the classifier exactly
- [x] Live CAPE/CIN from real GFS; shear honestly labelled 1000–700 hPa
- [x] Current Location bound to the user's own coordinates
- [x] No fabricated markers or fallback weather values
- [x] Public UI free of `V1`/`V2`/`SevereWeatherNet`

### 🔲 Unresolved
- [ ] `tests/test_browser_e2e.py` status unknown (excluded from runs)
- [ ] OWM-unreachable path not browser-tested
- [ ] Historical + NOW horizon combination not explicitly browser-tested
- [ ] Repo cleanup; `CLAUDE_HANDOFF.md` is a script; `weather-app/.venv` broken
- [ ] No CI

### ❗ Known limitations (do not hide)
- [!] Live GFS skill ~4× weaker than ERA5 test metrics (CSI 0.104 vs 0.405 at +2h)
- [!] GFS analysis lags wall-clock by ~5 h
- [!] Verification uses ERA5-derived proxies, not observed reports
- [!] 30-case, convective-season-only backtest
- [!] NOW field is interpolated from 24 stations — patchy convection can be missed
- [!] XAI is rule-based, not validated explainability
- [!] GEOS/PyTorch native fragility under memory pressure

### ❓ Requires verification
- [?] Run `tests/test_browser_e2e.py` against a live server
- [?] OpenWeatherMap plan tier / rate limit vs 24 calls per 4 min
- [?] Live-refresh scheduler stability over many hours
- [?] Whether XAI should consume CIN and shear

---

### 👉 WHAT TO DO FIRST

1. `python run_server.py`, wait for health, and **open the dashboard in a browser.**
   Click **NOW**, then **+2h / +4h / +6h**. Confirm NOW shows the blue/teal observation
   field with station dots and the forecast horizons show the orange risk field with none.
2. Run the pytest suite and confirm **287 passed**.
3. Only then start new work — and ask the user what they want next, because the two
   most recent requirements (NOW heatmap, prominent forecast timing) are **complete**.

**If you change anything: re-run the tests, re-verify in a browser, and keep the
data-provenance guarantees intact.**

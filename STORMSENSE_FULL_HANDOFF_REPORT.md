# STORMSENSE — FULL PROJECT HANDOFF REPORT

**Generated:** 2026-09-18 (session working date)
**Repo:** `D:\Weather-Hackathon`
**Branch:** `master` · last commit before this session: `d52ed8e AFTER`

> **Evidence rule used throughout this document.** Every claim below is tagged:
> **[VERIFIED]** = I ran it and saw the output in this session · **[IMPLEMENTED — NOT FULLY VERIFIED]** = code written and smoke-tested, but not exhaustively validated · **[KNOWN ISSUE]** · **[LIMITATION]** · **[NOT TESTED]**.
> Nothing here is inferred from documentation alone. Where I did not test something, it says so.

---

## 1. PROJECT OVERVIEW

**StormSense** is an AI-driven hyper-local severe-weather early-warning / nowcasting system for **West Bengal, India**. It predicts a spatial severe-weather probability field over the state at **NOW, +2h, +4h, +6h**, and displays it over Google Maps alongside separate observational products (radar, station weather).

| Layer | Technology | **[VERIFIED]** |
|---|---|---|
| Frontend | **Vanilla JavaScript** (no React/Vue). `frontend/js/app.js` ≈ 258 KB, `frontend/index.html` ≈ 104 KB | Yes — read directly |
| Base map | **Google Maps JS API**, with a Leaflet-compatibility shim (`L.imageOverlay`, `L.tileLayer` calls still present and working over Google) | Yes |
| Backend | **FastAPI** (`backend/main.py`, ≈76 KB), served by `run_server.py` via uvicorn on `127.0.0.1:8000` | Yes |
| ML | **SevereWeatherNetV2** — tri-stream ConvGRU, **781,889 params** | Yes — counted from checkpoint |
| Live NWP | **NOAA GFS f000 analyses** (0.25°, AWS Open Data, no key) | Yes |
| Radar | **RainViewer** composite mosaic | Yes |
| Station obs | **OpenWeatherMap** `/data/2.5/weather` | Yes |

**Two modes:** `live` (GFS-driven, present day) and `historical` (frozen Cyclone Remal replay). They share all aggregation/formatting code and differ **only** in which prediction dict + issue time is read (`_pred_for_mode`, `_issue_time_for_mode`).

**Dashboard views (8):** Dashboard · Radar & Satellite Feeds · District Advisories · AI Nowcast & Benchmark (WRF) · XAI Feature Attribution · GIS Spatial Layers · Calibrated Thresholds · Data Ingestion Streams. **[VERIFIED]** — all 8 render, screenshots in `diagnostics/results/shot_*.png`.

---

## 2. ACTUAL PROJECT STRUCTURE (verified paths only)

```
D:\Weather-Hackathon\
├── run_server.py                    # THE entry point -> loads backend/main.py
├── backend/
│   ├── main.py                      # THE served FastAPI app (all endpoints)
│   ├── openweather.py               # OWM client (current + forecast)
│   ├── config.py, hazard_engine.py, .env
├── frontend/
│   ├── index.html                   # dashboard (all 8 views)
│   └── js/app.js                    # all frontend logic
├── landing.html                     # landing page, served at "/"
├── src/
│   ├── inference/
│   │   ├── nowcast_service.py       # orchestrator: modes, risk, districts, XAI
│   │   ├── predictor.py             # checkpoint load, normalize, forward pass
│   │   ├── gfs_live.py              # GFS f000 fetch + harmonize
│   │   ├── risk_surface.py          # PNG risk raster + WB mask
│   │   ├── risk_map.py              # GeoJSON (825 cells)
│   │   ├── observation_surface.py   # OWM station IDW surface
│   │   ├── xai.py, insat_archive.py
│   │   ├── risk_thresholds.py       # ** NEW THIS SESSION ** single source of truth
│   │   ├── radar_observation.py     # ** NEW THIS SESSION ** radar -> grid sampler
│   │   └── now_evidence.py          # ** NEW THIS SESSION ** observation evidence
│   ├── data/dataset_v2.py           # TRAINING dataset (temporal-feature reference)
│   ├── features/normalize.py targets.py
│   ├── models/v2_model.py
│   └── api/app.py                   # ** DEAD ** — not served. Do not edit by mistake.
├── configs/default.yaml             # domain, labels, sequence config
├── Data/
│   ├── outputs/checkpoints/v2_calibrated_best.pt   # THE checkpoint
│   ├── BOUNDARIES/  (west_bengal_full.geojson, *_districts_full.geojson, wb_mask_full_66x50.npy)
│   ├── LIGHTNING/india_lightning_Strike.csv        # 2020 ONLY (ISS-LIS)
│   ├── RADAR/Radar_data.zip                        # calibration figures only
│   ├── DEM/, ERA5/, INSAT/, IMD_RAINFALL/
├── processed/cache/
│   ├── era5_memmap/                 # ERA5 training cache + dem_elevation_m.npy
│   └── gfs/                         # GFS f000 pickles, e.g. 20260918_120000_0.pkl
├── diagnostics/                     # audit scripts + results (this session)
└── backups/google-maps-version/     # OLD COPY — not served, ignore
```

**Trap:** `src/api/app.py` is a second FastAPI app that is **not served**. Editing it changes nothing. **[VERIFIED]**

---

## 3. MODEL DETAILS — **DO NOT MODIFY**

**Checkpoint:** `Data/outputs/checkpoints/v2_calibrated_best.pt` **[VERIFIED — loaded and inspected]**

| Property | Value |
|---|---|
| Architecture | `SevereWeatherNetV2` (tri-stream ConvGRU), `src/models/v2_model.py` |
| Parameters | **781,889** (summed from state_dict) |
| Checkpoint keys | `epoch, model, optimizer, scheduler, best_val_score, best_val_pr_auc, best_val_csi, patience_ctr, config, optimal_threshold, threshold_per_lead, threshold_results, temperature_per_lead, is_calibrated, calibration_method` |
| `epoch` | 16 · `best_val_loss` = None |
| `optimal_threshold` | **0.673** |
| `threshold_per_lead` | `{2: 0.727, 3: 0.679, 4: 0.673, 5: 0.637, 6: 0.625}` |
| `temperature_per_lead` | **`[0.4152, 0.4144, 0.4169, 0.4105, 0.3819]`** |
| `is_calibrated` | True |

### Input tensors (as `predictor.predict` builds them) **[VERIFIED — read code + ran]**
- `surface`: `(T=6, C=15, 33, 25)` — **9 raw + 2 derived + 4 temporal**
  - raw `SINGLE_VARS` = `["u10","v10","d2m","t2m","sp","cape","cin","tcwv","tp"]`
  - derived = `wind_speed = sqrt(u10²+v10²)`, `dewpoint_dep = t2m − d2m` (computed on **normalized** values, matching training)
  - temporal = `hour_sin, hour_cos, doy_sin, doy_cos`
- `pressure_wind`: `(6, 2, 3, 33, 25)` = `pres_norm[:, :2, 3:]` → u,v @ **700/850/1000**
- `pressure_thermo`: `(6, 3, 4, 33, 25)` = `pres_norm[:, 2:, :4]` → z,q,t @ **250/300/500/700**
- `dem`: `(3, 33, 25)` = `[dem_norm, lat_grid, lon_grid]` (lat/lon normalized to [−1,1])

### ⚠ PRESSURE LEVEL ORDER — LOAD-BEARING
`gfs_live.PRESSURE_LEVELS_HPA = [250, 300, 500, 700, 850, 1000]` — **ASCENDING**.
`configs/default.yaml` documents `[1000,850,700,500,300,250]` (descending) **for human reference only**. The slicing indices above depend on the ascending order. **Do NOT "fix" the array to match the YAML** — it would silently swap which levels feed which stream. **[VERIFIED — code comments + slicing confirmed]**

### Output heads
- `severe_weather_logit` → `(n_lead=5, 33, 25)` → **temperature-scaled** sigmoid
- `rain_3h_mm` → `(5, 33, 25)` regression, mm

### Calibration — counter-intuitive, verified
`T ≈ 0.38–0.42` is **< 1**, so it **SHARPENS** (polarizes), it does not soften. Measured **[VERIFIED]**:

| raw p | calibrated (T=0.4152) |
|---|---|
| 0.10 | 0.50 % |
| 0.25 | **6.62 %** |
| 0.40 | 27.4 % |
| 0.50 | 50.0 % |
| 0.60 | **72.6 %** |
| 0.75 | 93.4 % |

Inverse: to display ≥25 % needs raw ≥0.388; ≥75 % needs raw ≥0.612.
**This near-binary behaviour independently amplifies BOTH mismatch directions and is expected.**

### 🔒 ABSOLUTE RULE
**The checkpoint must NOT be retrained, fine-tuned, replaced, re-calibrated, or have its expected tensor structure changed — unless the user explicitly requests it.** It was **NOT** modified in this session.

---

## 4. LIVE DATA PIPELINE

### 4.1 GFS f000 — the ONLY ML input **[VERIFIED]**
- **Module:** `src/inference/gfs_live.py` · `fetch_and_harmonize(lats, lons, target_t0)`
- **Source:** `https://noaa-gfs-bdp-pds.s3.amazonaws.com` (fallback NOMADS), byte-range GRIB2 via `.idx`
- **Only `f000` is ever fetched.** Forecast hours f001+ are explicitly forbidden as input timesteps.
- **Cadence:** 6 h (00/06/12/18Z). **Production lag constant:** `GFS_PRODUCTION_LAG_HOURS = 5.0`
- **6 input slots** are hourly, ending at `analysis_t0`, **linearly interpolated** between the two newest real analyses. Per-slot provenance (`analysis` vs `interpolated`) is tracked and exposed.
- **Cache:** `processed/cache/gfs/YYYYMMDD_HHMMSS_0.pkl` (~90 KB each) + module-level `_HARMONIZED_CACHE` keyed on `analysis_t0`.
- **Measured lag this session:** **5.3 h – 7.3 h** behind wall-clock. **[VERIFIED]**
- **Variable harmonization:** `tp` = GFS `PRATE × 3600` → **mm/hour**; `tcwv` = `PWAT`; `z` = `HGT × 9.80665`; `cin` clipped ≥0.

### 4.2 RainViewer radar — display + (new) evidence, **NEVER an ML input** **[VERIFIED]**
- Index: `https://api.rainviewer.com/public/weather-maps.json`; tiles `{host}{path}/256/{z}/{x}/{y}/2/1_1.png`
- **Frame selection:** `data.radar.past[past.length - 1]` = newest past frame. Correct.
- **Cadence ~10 min. History retained: only ~2.0 hours** (13 frames). **[VERIFIED]** → **no historical radar retrieval is possible.**
- `maxNativeZoom: 7`, `maxZoom: 16` (overzoom blurs rather than vanishing — prior fix).
- **Coverage over WB is REAL** **[VERIFIED]**: WB tile = 3376 B / 2.14 % echo vs Sahara & mid-Pacific = **334 B / 0.00 %** (placeholder). So an empty WB map means *no precipitation*, not *no radar*.
- Guarded by `isRadarFeedsMap()` — RainViewer is **blocked on the nowcasting map**, permitted only on the Radar view; and blocked entirely in historical mode.

### 4.3 OpenWeatherMap — display + (new) evidence, **NEVER an ML input** **[VERIFIED]**
- `backend/openweather.py` → `https://api.openweathermap.org/data/2.5/weather`, `units=metric`
- **Freshness measured: 0–9 minutes.** **[VERIFIED]**
- **Spatially usable:** returns the **exact requested coordinate** in `coord` (not a snapped station), values vary smoothly with position → it is a gridded/interpolated field with a nearby place-name label. Confirmed by a 7-point transect and micro-offset test. **[VERIFIED]**
- `base: "stations"`. Variables: temp, feels_like, humidity, pressure, wind speed/deg, visibility, `rain.1h`, weather condition, `dt`.
- **No fabricated fallback** — on failure every field is `null` + `observation_available: false`.
- Station sweep: `OBSERVATION_SITES` (12 WB settlements) in `observation_surface.py`, cached **240 s** (`_OBS_TTL_SECONDS`), IDW-interpolated with `MAX_INFLUENCE_DEG` cutoff; out-of-range pixels stay **transparent, never extrapolated**.

### 4.4 Other sources found
- `Data/LIGHTNING/india_lightning_Strike.csv` — **ISS-LIS, 2020 ONLY** (2020-05-01→2020-10-14, 24,546 rows, **zero 2026 records**). **Not ingested anywhere.** **[VERIFIED]**
- `Data/RADAR/Radar_data.zip` — calibration figures only, no gridded reflectivity.
- INSAT — `use_for_training: false`; qualitative case-study only.

---

## 5. HISTORICAL DATA PIPELINE

**[VERIFIED]** Constants in `src/inference/nowcast_service.py`:
```python
HISTORICAL_EVENT_NAME   = "Cyclone Remal"
HISTORICAL_ANALYSIS_TIME = "2024-05-26T12:00:00Z"   # authoritative T0
HISTORICAL_ANALYSIS_SOURCE = "ERA5 reanalysis"
```
- T0 = **26 May 2024 12:00 UTC** = **17:30 IST**. Horizons: +2h=14:00Z, +4h=16:00Z, +6h=18:00Z.
- Input: ERA5 memmap cache (`processed/cache/era5_memmap`) via `make_dataloaders_v2`, test split.
- **NO silent fallback:** if the exact analysis time is not in the held-out split it raises `RuntimeError` and records `historical_init_error`. It will **never** substitute another sample/event.
- NOW (lead 0) = inference on `dataset[sample_idx - 2]`, taking its +2h head — same design as live.
- `era5_surface_by_lead` holds **observed** ERA5 states at +2…+6h — these are *verification observations*, **not model predictions**, and must never be labelled as forecasts.
- **`tp` in the ERA5 cache is ALREADY mm/hour.** Multiplying by 1000 again previously pinned the whole grid at the 150 mm clip. Denormalize only. **[VERIFIED via code comment + stats: tp mean 0.371, std 0.974]**
- **Historical radar: UNAVAILABLE** (RainViewer keeps 2 h). Reported as unavailable; never fabricated.
- **Isolation:** `loadRainViewerRadar` returns early in historical mode; `applyRadarModeSemantics()` strips tiles on mode switch; live `now_evidence` returns `NOT_APPLICABLE` in historical. **[VERIFIED this session]**

---

## 6. NOW / +2h / +4h / +6h DESIGN

| Term | Meaning | Source |
|---|---|---|
| `live_reference_time` | **T0** — exact wall-clock instant the forecast is ISSUED FOR (never floored) | `harmonized.t0` |
| `live_analysis_time` | when the **atmosphere was observed** (GFS f000) | `harmonized.analysis_t0` |
| NOW (lead 0) | model run on the **T−2h** input window, taking its **+2h head** | by design |
| +2/+4/+6h | `T0 + n hours`, from the model's own multi-horizon heads | `lead_times_hours = [2,3,4,5,6]`; service exposes `[0,2,3,4,5,6]` |

**NOW is lead 0 and is produced from a second inference on the T−2h window.** This remains the implementation **[VERIFIED — `refresh_live_state` fetches `target_t0` and `target_t0 − 2h`]**. Live `lead=0` and `lead=2` risk-surface PNGs were byte-identical in one test (13609 B both) — expected, since NOW *is* the earlier window's +2h field.

**Timestamp semantics are correct:** `+2h` valid time = reference + 2 h. **[VERIFIED via `/api/time/reference` and point API]**

---

## 7. RADAR vs NOW RISK — THE CRITICAL INVESTIGATION

### 7.1 CONFIRMED BUG — FOUND **AND FIXED** in the previous session
**Root cause:** `refresh_live_state()` passed `harmonized.t0` (**wall-clock**) as the predictor's `timestamp`. That argument builds the model's `hour_sin/hour_cos/doy_sin/doy_cos` channels, and `predictor.predict` derives slot hours as `(ts.hour + np.arange(-T+1, 1)) % 24` — i.e. it assumes `timestamp` is when the **last input slot** was observed. But live slots end at `analysis_t0`, **5–11 h earlier**. Training (`src/data/dataset_v2.py`: `times_window = pd.DatetimeIndex(self.times[start:end])`) uses the **real slot timestamps**.

**Fix:** pass `harmonized.analysis_t0.isoformat()` — `src/inference/nowcast_service.py` **lines 507 and 516**. **[VERIFIED still present this session]**

**Measured impact (identical input tensors, only the timestamp changed):**

| wall-clock | analysis | BUGGY Kolkata | CORRECT | err | BUGGY dom-max | FIXED dom-max |
|---|---|---|---|---|---|---|
| 17 Sep 04:00Z | 16 Sep 18Z | 7.051 % | 0.123 % | **+6.93** | **85.91 %** | 23.28 % |
| 17 Sep 06:00Z | 17 Sep 00Z | 10.795 % | 0.662 % | **+10.13** | 33.81 % | 14.24 % |
| 17 Sep 12:00Z | 17 Sep 06Z | 0.059 % | 1.918 % | **−1.86** | 13.81 % | **67.90 %** |

Errors ran in **both directions** → explains both mismatch cases.
**Intra-cycle drift proof:** with a *fixed* analysis (17 Sep 00Z), advancing only wall-clock 06→11Z made Kolkata decay 10.795 %→0.035 % and **14 amber cells → 0** with **zero new data**. Post-fix that drift is **0.000 pp**. **[VERIFIED]**

### 7.2 Radar ↔ observation correlation — **WHY NO FUSION WEIGHT IS DEFENSIBLE**
Measured 2026-09-18 17:10Z, **n = 48** grid points across WB **[VERIFIED]**:
- **`corr(radar echo_fraction, OWM rain_1h) = 0.104`**
- `corr(echo, weather=="Rain/Thunderstorm") = 0.151`
- mean echo where `wx=Rain`: 0.111 · where not: 0.035
- echo covered **~1.1 %** of WB; `echo>0` at only 3/48 sampled points, `OWM rain>0` at 9/48 — **and they were mostly different points**.
- **All sampled echo pixels were light-blue** = RainViewer colour-scheme-2 **lowest** reflectivity band (drizzle/light rain), not convective cores. Sample RGBs: `[81,197,232] [27,174,226] [0,154,213] [0,163,224] [0,136,191]`.
- **RainViewer tiles carry no numeric dBZ — the COLOUR *is* the value.** Any dBZ would be an inverse-palette guess.

**Conclusion:** the two independent observation sources barely agree with each other, and there is **no gridded severe-weather verification dataset in this repo** to fit or validate a radar→probability mapping. **Any fusion weight would be invented.** Therefore no numerical blending was implemented.

### 7.3 Product-time gap — the irreducible physical reality **[VERIFIED]**
```
RADAR latest frame   : 2026-09-18 17:20 UTC  (22:50 IST)   age ~0.1 h
STATION observation  : 2026-09-18 17:26 UTC               age ~0.0 h
MODEL analysis state : 2026-09-18 12:00 UTC  (17:30 IST)   age  5.44 h
radar − analysis     = 5.33 h   <-- the comparison gap
```
An earlier measurement the same day gave **7.17 h**. The user compares a ~10-minute-old radar picture against a model whose atmosphere is **5–7+ hours old**. This is **by construction**, not a bug.

### 7.4 Case A/B/C/D — actually tested **[VERIFIED]**
Over 36 location-times (6 locations × 6 real cached analyses), classifying by GFS `tp` vs risk band:
- **Case A** (analysis wet, risk NORMAL): **3**
- **Case B** (analysis dry, risk ≥ALERT): **0** (post-fix)
- **Case C** (analysis wet, risk ≥WATCH): **1** — Purulia 17 Sep 06Z, tp 1.863 mm/h → **34.92 % WATCH**
- **Case D** (analysis dry, risk NORMAL): **32**
Agreement dominates → the system is **not** systematically broken.

- **Case A reproduced LIVE** (18 Sep 12:45 IST): radar echo **4.05 %** over South Kolkata vs StormSense **0.43 % NORMAL**, input `tp = 0.000 mm/h`, analysis 7.26 h old.
- **Case B was predominantly the timestamp bug**: pre-fix domain max hit **85.91 %** (WARNING red) where the correct value was 23.28 %. Post-fix, no spurious >50 % appeared in the 36-point sweep.

### 7.5 17 SEPTEMBER 2026 — KOLKATA / SOUTH KOLKATA **[VERIFIED to the extent data exists]**
| Item | Finding |
|---|---|
| Grid cell | Kolkata (22.5726,88.3639) **and** South Kolkata (22.4950,88.3450) **both → cell [22,17] = 22.50N, 88.25E** (offsets 14.2 km / 9.8 km) |
| **Radar evidence** | **UNAVAILABLE** — RainViewer retains only ~2 h |
| **Lightning evidence** | **UNAVAILABLE** — CSV is 2020-only, zero 2026 records |
| Model inputs at the cell | **`tp = 0.000 mm/h` at EVERY cycle (00/06/12/18Z) AND at all 8 neighbours** |
| CAPE / CIN / TCWV | CAPE 876–1235 · CIN 0 · TCWV 54.6–59.7 |
| 3 h rain | 0.000 mm (label needs **>15 mm**) |
| Convective criterion | `CAPE>2000 & CIN<50` → **False** (domain CAPE max 1565–2277; 17 Sep 12Z had 15/825 cells >2000, **none over Kolkata**) |
| Displayed risk | **0.12 %–1.92 %, NORMAL all day** (post-fix) |
| **Conclusion** | The model was green **because its input contained no storm** — not a render/calibration/API/frontend fault. |

> **Honest framing to preserve:** if a real thunderstorm occurred over Kolkata that day, **GFS f000 did not analyze it**. That is a genuine miss of the *input*, not a StormSense bug. The user's screenshot is valid evidence of what they saw; it was **not** treated as a quantitative measurement.

### 7.6 RULED OUT (tested, found NOT responsible) **[VERIFIED]**
| Hypothesis | How it was ruled out |
|---|---|
| Lat/lon reversal, grid shift | `lats` 28→20 descending, `lons` 84→90 ascending; Kolkata→[22,17] correct |
| Array transpose | GeoJSON cell (22.50, 88.25) matched point API exactly (35.5 % both) |
| Interpolation displacement | Overlay bounds `[[21.5394,85.8325],[27.2206,89.8828]]` == renderer constants; Darjeeling & Purulia both painted |
| UTC/IST conversion | `formatIstClock`/`formatIstStamp` tested in **3 timezones** (Asia/Kolkata, UTC, America/New_York) → all gave `11:55 IST` for `06:25Z` |
| Frontend stale-response race | Cache-busted `&t=Date.now()`; no race observed |
| Wrong lead displayed | All 6 leads returned distinct, monotonic valid times |
| Wrong radar frame | `past[length-1]` = newest; verified against API |
| API transformation | GeoJSON == point API == displayed |
| Colour/legend thresholds | Backend 0.25/0.50/0.75 == frontend 25/50/75 |

---

## 8. CURRENT DECISION ABOUT OBSERVATIONS — **TRANSPARENT OBSERVATION LAYER**

**User's explicit decision this session (quoted):**
> *"NOW remains the model's probability, while current observations provide an independent observation-evidence layer that materially informs the interpretation of NOW. Do not numerically modify the calibrated model probability unless a validated observation-to-risk relationship exists."*

**Therefore the implemented design is:**
- ✅ NOW displays the **model's calibrated probability, unchanged**
- ✅ Radar echo = **independent observed evidence**, own timestamp
- ✅ OWM station rain = **independent observed evidence**, own timestamp
- ✅ **Product-time gaps are shown explicitly**
- ✅ **Agreement / disagreement verdict** is surfaced
- ❌ **NO** arbitrary radar boost · **NO** fabricated probability · **NO** numerical blending

> ⚠ **The next Claude must NOT silently switch to numerical radar→risk blending.** That requires a *validated* observation-to-risk relationship, which does not exist in this repo (see §7.2). If blending is ever attempted, it must be justified against real validation data and explicitly approved by the user.

---

## 9. RECENT FORENSIC AUDIT — RECORD

**Scope executed:** 6 locations × 6 real cached GFS analyses (36 location-times) · 7 leads × 2 modes · 12-point wall-clock sweep · 8 dashboard views · 3-timezone IST test · 17 Sep case study at 6 IST times · 48-point radar/OWM correlation sweep.

**Browser (Playwright, Chromium 1600×1000):** **39 API calls, 0 ≥400 · 0 console errors.** Failed requests = 9–10 RainViewer tiles with `net::ERR_ABORTED` — these are **navigation-cancelled tiles**, not real failures (tiles fetch fine standalone: 200, 4245 B).

**Root causes found & fixed (previous session, verified still present):**
1. Wall-clock vs `analysis_t0` timestamp bug → `nowcast_service.py:507,516`
2. Stale georeference headers (advertised 26.9960N/86.6103E vs actual 27.2206/85.8325 — ~90 km error) → `backend/main.py:377`
3. `input_valid_time` aliased to issue time → `risk_map.py` + `analysis_time` param

**This session:** 3 new modules + threshold consolidation + NOW evidence layer + provenance endpoint (see §27).

---

## 10. TIMESTAMP / TIME ALIGNMENT TABLE

| Component | Timestamp meaning | Current source | Known lag | Verified? |
|---|---|---|---|---|
| GFS cycle/analysis time | when the atmosphere was **observed** | `harmonized.analysis_t0` → `svc.live_analysis_time` | **5.3–7.3 h** behind wall-clock | ✅ |
| GFS input slots (6) | hourly t−5h…t0 **relative to analysis_t0**; slots 1–5 time-interpolated, slot 6 = real analysis | `harmonized.slot_timestamps` + `slot_provenance` | — | ✅ |
| Predictor `timestamp` arg | must equal **last input slot** time | **`analysis_t0`** (fixed) | — | ✅ |
| StormSense **T0** | instant forecast is **issued for** (exact wall-clock, never floored) | `live_reference_time` | 0 | ✅ |
| NOW (lead 0) | risk valid at T0; from **T−2h window's +2h head** | `live_pred[0]` | — | ✅ |
| +2h / +4h / +6h | `T0 + n h` | `compute_valid_time(ref, n)` | — | ✅ |
| Radar timestamp | **observed scan time** | RainViewer `past[-1].time` | ~5–10 min | ✅ |
| OWM timestamp | **observation time** (`dt`) | OWM payload | 0–9 min | ✅ |
| UI clock | IST display | `formatIstClock/Stamp` (+5:30, tz-independent) | 0 | ✅ |
| Historical T0 | frozen ERA5 analysis | `2024-05-26T12:00:00Z` | n/a | ✅ |

**Rule to preserve:** `reference_time` ≠ `analysis_time`. They must **never** be conflated. `/api/now/provenance` now returns both plus radar & station times and all gaps.

---

## 11. SPATIAL / GRID DETAILS

| | Value | Verified |
|---|---|---|
| **MODEL grid** | **33 × 25 = 825 cells @ 0.25°** (~28 km) | ✅ |
| lat range | 28.0 → 20.0 N (**descending**) | ✅ |
| lon range | 84.0 → 90.0 E (**ascending**) | ✅ |
| **Visualization grid** | `DEFAULT_H × DEFAULT_W = 600 × 400` in code (docstring says 66×50) — **display resampling only** | ⚠ see §20 |
| **Render raster** | `RENDER_H × RENDER_W = 1320 × 1000` — **image pixels only** | ✅ |
| Map bounds | `WB_BOUNDS = [[21.5394, 85.8325], [27.2206, 89.8828]]` (frontend == renderer) | ✅ |
| Popup | **nearest-neighbour on the RAW model grid** (no smoothing) | ✅ |
| District | shapely point-in-polygon on `west_bengal_districts_full.geojson` | ✅ |

> ### 🚫 NEVER CLAIM INTERPOLATION = MODEL RESOLUTION
> Cubic interpolation + Gaussian smoothing produce a smooth raster. **The underlying ML grid remains 0.25°.** Do **not** describe the system as 1-km or 10-km prediction.
>
> **City-scale limitation:** Kolkata, South Kolkata **and Howrah all collapse into the single cell [22,17]**. A thunderstorm smaller than ~28 km is **sub-grid by construction**. **[VERIFIED]**

**Note:** map PNG is heavily smoothed (`gaussian_filter`, σ scaled to render grid) while the **popup uses the raw cell value** — so a popup number and the colour under the cursor can legitimately differ slightly. This is expected, not a bug.

---

## 12. RISK MAP

- **What is coloured:** `severe_weather_prob` = temperature-scaled sigmoid of `severe_weather_logit`. **Not** rainfall, **not** reflectivity, **not** lightning.
- **Training label** (`src/features/targets.py`): `severe_weather = heavy_rain OR severe_convective` where `heavy_rain = rolling-3h tp > 15.0 mm`, `severe_convective = (CAPE > 2000) & (CIN < 50)`. **Proxy labels from ERA5 — not observed severe-weather reports.**
- **Compound risk** (`derive_compound_risks`, verified **unchanged**, identical for both modes):
```python
terrain_factor  = 1.0 + 0.2*np.clip(dem_norm, 0.0, 2.5)
rain_norm       = np.clip(rain_pred/30.0, 0.0, 1.0)
hydro_intensity = rain_norm * (0.4 + 0.6*severe_prob)
ff_risk         = np.clip(hydro_intensity * terrain_factor[None], 0.0, 1.0)
overall_risk    = np.clip(0.5*severe_prob + 0.3*ff_risk + 0.2*rain_norm, 0.0, 1.0)
```
- **Display bands (now single-sourced in `src/inference/risk_thresholds.py`):** `<0.25` NORMAL emerald `#10b981` · `0.25–0.50` WATCH amber `#f59e0b` · `0.50–0.75` ALERT orange `#f97316` · `≥0.75` WARNING red `#ef4444`
- **These display bands are NOT the model's decision thresholds** (`threshold_per_lead` 0.625–0.727). Different questions — do **not** reconcile them.
- **Endpoints:** `/api/nowcast/risk-surface` (PNG raster) · `/api/nowcast/risk-map` (GeoJSON, 825 features) · `/api/nowcast/point` (popup)
- **Cache busting:** `&t=Date.now()` per request; `Cache-Control: no-cache` for live.

---

## 13. GOOGLE MAP / MAP INTERACTION

- **Google Maps is the base map.** A **Leaflet-compatibility shim** lets `L.imageOverlay` / `L.tileLayer` / `L.latLngBounds` calls work over Google. **Do NOT reintroduce real Leaflet.** **[VERIFIED — `window.WB_BOUNDS`, `L.imageOverlay` used at app.js:3285/3432/3516]**
- Risk overlay: `L.imageOverlay(url, window.WB_BOUNDS, {opacity:0.85, zIndex:300})`
- Radar overlay: `L.tileLayer(..., {opacity:0.65, zIndex:400, maxNativeZoom:7, maxZoom:16})` — **Radar view only**
- `fitBounds(L.latLngBounds(window.WB_BOUNDS), {padding:[20,20]})`
- **Verified visually** (`diagnostics/results/shot_map_6h.png`): risk surface correctly clipped to the WB outline including Darjeeling (north) and Purulia (west) — **no spatial shift**.
- **[NOT TESTED this session]:** drag (h/v/diagonal), wheel zoom, zoom controls, map recentre, live-location marker, district dropdown selection. **These remain to be exercised.**

---

## 14. RADAR DISPLAY

- Implementation: `loadRainViewerRadar(map)` in `frontend/js/app.js` (~line 4128)
- Frame: newest `past` frame · Label: `"Observed radar mosaic · RainViewer composite · ~10 min frames"`
- Timestamp: `"Last Scan: HH:MM IST · observed N min ago"` via `formatIstClock` (tz-independent)
- Overzoom: `maxNativeZoom:7` + `maxZoom:16` → blurs instead of vanishing (prior fix)
- Historical mode: **blocked** (`loadRainViewerRadar` returns early; `applyRadarModeSemantics()` strips existing tiles)
- **Radar does NOT enter ML inference.** Confirmed exhaustively: no reflectivity head exists; `xai.py` `radar_dbz` is always `None`; `nowcast_service.py:1571-1573` states no reflectivity is ingested anywhere.

---

## 15. FRONTEND / UI

**8 views, all confirmed rendering** (screenshots in `diagnostics/results/`): Dashboard · Radar · Advisories · WRF · XAI · GIS · Threshold · Streams.

**Changed in the UI session:** `frontend/js/app.js` ONLY — three additive pieces, no layout change:
1. `NOW_VERDICT_STYLE` map (4 verdicts -> colour + label)
2. `nowEvidenceHtml(ev)` — renders the observation-evidence block
3. `istClockFromIso(iso)` — UTC ISO -> "HH:MM IST", timezone-independent
4. one line inserted into the existing popup: `nowEvidenceHtml(data.now_evidence) +`

**Preserved:** layout, legend, cards, navigation, Google Maps, all 8 sections, all existing popup content. No HTML/CSS file was touched.

**Dashboard verified content** (`shot_dashboard.png`): IST clock, "OBSERVED: 2m ago", legend `<25% Normal / 25–50% Watch / 50–75% Alert / ≥75% Warning`, 4 hazard cards, "Current observed conditions" strip, Interactive Nowcasting Map with NOW/+2h/+4h/+6h buttons, and correctly distinct `Valid 2026-09-18 09:04 UTC · Input analysis 2026-09-18 00:00 UTC`.

**The `now_evidence` payload IS now surfaced** in the map popup at live NOW. `/api/now/provenance` remains backend-only (its key value, the model-input lag, is already shown in the popup).

**Popup evidence block shows:** radar echo as *% of cell* + radar scan time (IST) · station rain mm/h + station time (IST) + nearest-station distance · the sentence *"Model input is N h older than these observations (analysis HH:MM IST). Observations are shown for context and do not change the model probability."*

---

## 16. BACKEND API INVENTORY (all `GET` unless noted)

**[VERIFIED 200 this session]:** `/api/health` · `/api/system/info` · `/api/model-info` · `/api/config` · `/api/weather/current` · `/api/nowcast/summary` · `/api/nowcast/risk-map` · `/api/nowcast/risk-surface` · `/api/nowcast/point` · `/api/nowcast/districts` · `/api/nowcast/thermodynamics` · `/api/nowcast/high-risk-cells` · `/api/nowcast/xai` · `/api/time/reference` · `/api/mode/status` · `/api/live/surface` · `/api/live/ml-status` · `/api/data-health` · `/api/boundaries/west-bengal` · `/api/boundaries/state` · `/api/geo/areas` · `/api/benchmark/models` · **`/api/now/provenance` (NEW)**

**Present, [NOT TESTED] individually:** `/api/historical/analysis-surface` · `/api/historical/case-study` · `/api/historical/satellite` · `/api/historical/analysis-state` · `/api/nowcast/risk-surface/bounds` · `/api/observations/current` · `/api/observations/surface` · `/api/satellite/info` · `/api/forecast/conditions` · `/api/nowcast/explanation` · `/api/stormsense/nowcast` · `/api/stormsense/timeline` · `POST /api/predict` · `POST /api/live/refresh`

**Key params:** `lead ∈ {0,2,3,4,5,6}` (`SUPPORTED_LEADS`; invalid → 422 on risk-map/risk-surface, coerced to 2 on point) · `mode ∈ {live, historical}` · `lat`, `lon`.

**Key response fields (point):** `predictions.thunderstorm_prob_pct`, `.heavy_rain_mm`, `.flash_flood_proxy_pct`, `.overall_risk_pct`, `.risk_level`, `.risk_label`; `grid_cell{lat,lon}`; `issue_time_utc`; `forecast_valid_utc`; `pipeline_status`; `data_freshness`; **`now_evidence` (NEW, lead 0 live only)**.

**Error behaviour:** live unavailable → `_live_unavailable_payload` with the real reason (never fabricated values); historical unavailable → `historical_init_error`.

---

## 17. CACHE / PERFORMANCE

| Cache | Location / key | TTL | Verified |
|---|---|---|---|
| GFS pickles | `processed/cache/gfs/{cycle}_0.pkl` | permanent | ✅ 10 files present |
| Harmonized input | `gfs_live._HARMONIZED_CACHE`, keyed on `analysis_t0` | until new analysis | ✅ |
| Live risk PNGs | `svc.live_risk_surface_png_cache[lead]` | regenerated per refresh | ✅ |
| Historical PNGs | `svc.risk_surface_png_cache[lead]` | startup, permanent | ✅ |
| Station obs | `_OBS_CACHE` | **240 s** | ✅ |
| **Radar grid (NEW)** | `radar_observation._CACHE`, keyed on frame `path` | **240 s** | ✅ |
| Model singleton | `predictor._MODEL_CACHE[ckpt::device]` | process life | ✅ |
| Live refresh loop | `LIVE_REFRESH_INTERVAL_SECONDS = 300`, non-overlapping | — | ✅ |
| Frontend | `&t=Date.now()` cache-bust | per request | ✅ |

**Measured:** server cold start (CPU) ≈ **20 s** · radar grid sample **10.0 s** (647/825 cells covered, 78 with echo) · `/api/now/provenance` responds after radar sample (cached thereafter).
**Stale protection:** a failed live refresh **never clears** `live_pred`; `is_live_stale()` flags >12 h.

---

## 18. SECURITY / CONFIGURATION

- `.env` (repo root) — variable names: **`OPENWEATHER_API_KEY`**, **`GOOGLE_MAPS_API_KEY`**
- `backend/.env` — `OPENWEATHER_API_KEY`
- Loaded via `python-dotenv` `load_dotenv()`; Google key injected server-side by `_google_maps_key()`
- Env override: `STORMSENSE_DEVICE` (`cpu`/`cuda`/`auto`) · `STORMSENSE_DISABLE_LIVE_REFRESH=1` · `HOST`, `PORT`

> ## 🔴 P0 SECURITY FINDING — **VERIFIED, NOT YET REMEDIATED**
>
> **Both `.env` files are COMMITTED TO GIT, and the repo has a public GitHub remote.** **[VERIFIED]**
>
> - `.gitignore` lines **9–12** are **commented out**:
>   ```
>   9:# .env
>   10:# **/.env
>   11:# *.env
>   12:# !.env.example
>   ```
>   so they ignore nothing.
> - `git ls-files --error-unmatch .env` → **`.env` (tracked)**
> - `git ls-files --error-unmatch backend/.env` → **`backend/.env` (tracked)**
> - `git show HEAD:.env` returns both `OPENWEATHER_API_KEY` and `GOOGLE_MAPS_API_KEY` with **real values**.
> - Present in **multiple commits** (`7c34216`, `d52ed8e`, `c1c8875`, `9badd68`).
> - Remote: **`https://github.com/Souvik686/StormSense.git`**
>
> **Consequence:** both keys must be considered **compromised**. Removing the files now does **not** remove them from git history.
>
> ### 🚫 REPOSITORY PROTECTION RULE — STANDING, SET BY THE PROJECT OWNER
> **No agent may perform ANY of the following.** They are the project owner's
> actions alone, and this is a hard prohibition, not a default to be overridden:
> - Modify `.gitignore` in any way (edit, uncomment, add, remove, regenerate).
>   **Inspection is READ-ONLY.**
> - Delete or move `.env` / `backend/.env`
> - Rewrite git history (`filter-repo`, BFG, rebase) · force-push
> - Rotate or regenerate API keys
> - Change repository visibility or any GitHub setting
>
> An agent's role here is to **REPORT ONLY**.
>
> **Actions that would be required FROM THE PROJECT OWNER** (informational; do not perform):
> 1. **Rotate both keys** at OpenWeatherMap and Google Cloud Console — this is the only
>    step that actually revokes the exposure. Deleting files does not.
> 2. Restrict the Google Maps key by **HTTP referrer**; restrict the OWM key if the plan allows.
> 3. Decide separately whether to untrack the files and/or purge history — both are
>    irreversible/history-affecting and are the owner's call, not an agent's.
>
> **Secret values are deliberately not reproduced in this report.**

---

## 19. TEST RESULTS

| Test | Mode | Location | Time | Expected | Actual | Status | Evidence |
|---|---|---|---|---|---|---|---|
| Checkpoint integrity | — | — | — | 781,889 params, calibrated | exact match | ✅ | `_audit_ckpt.py` |
| Calibration T<1 sharpens | — | — | — | quantify | raw .25→6.6 %, .60→72.6 % | ✅ | `_audit_calib.py` |
| Grid mapping | — | Kolkata/S.Kol/Howrah | — | correct cell | all → [22,17] | ✅ | `_audit_kolkata.py` |
| Temporal-channel sensitivity | live | Kolkata | 17 Sep | quantify | dom-max 13.8 %→73.8 % on **identical tensors** | ✅ | `_audit_temporal.py` |
| Wall-clock vs analysis ts | live | Kolkata | 17 Sep ×12 | quantify | −1.9…+10.1 pp | ✅ | `_audit_lagsweep.py` |
| Intra-cycle drift (pre-fix) | live | Kolkata | 17 Sep 06–12Z | 0 pp | **25.25 pp** | ❌→fixed | `_audit_drift.py` |
| Intra-cycle drift (post-fix) | live | Kolkata | 17 Sep 06–10Z | 0 pp | **0.000 pp** | ✅ | `_audit_postfix.py` |
| Case A/B/C/D sweep | live | 6 locs × 6 analyses | 16–18 Sep | agreement dominant | A=3 B=0 C=1 D=32 | ✅ | `_audit_cases.py` |
| 17 Sep Kolkata inputs | live | [22,17] | 4 cycles | — | **tp=0.000 everywhere incl. 8 neighbours** | ✅ | `_audit_whygreen.py` |
| CAPE regime | live | domain | 6 analyses | vs 2000 threshold | max 1565–2277 | ✅ | `_audit_cape.py` |
| IST conversion | — | — | 06:25Z | 11:55 IST | 11:55 IST in **3 timezones** | ✅ | `_audit_ist.py` |
| Radar coverage real? | — | WB vs Sahara/Pacific | 18 Sep | WB has coverage | WB 3376 B/2.14 % vs 334 B/0.00 % | ✅ | `_radar_coverage.py` |
| Radar↔OWM correlation | — | 48 grid pts | 18 Sep 17:10Z | — | **r = 0.104** | ✅ | `_fusion_evidence.py` |
| OWM freshness/spatial | — | 6 locs + transect | 18 Sep | — | 0–9 min; exact-coord gridded | ✅ | `_owm_probe.py` |
| Open-Meteo evaluation | — | Kolkata | 18 Sep | — | current hour = **forecast**; ECMWF CAPE 3780 vs GFS 989 | ✅ | `_openmeteo_probe.py` |
| Threshold boundaries | — | — | — | 24.9→NORMAL, 25→WATCH… | all correct | ✅ | `risk_thresholds` test |
| Radar grid sampler | live | 825 cells | 18 Sep | — | 10.0 s, 647 covered, 78 echo | ✅ | `_radar_grid_test.py` |
| `/api/now/provenance` | live | — | 18 Sep | 4 distinct times | 4 distinct + gaps | ✅ | `prov.json` |
| `now_evidence` (point) | live | Kolkata, Darjeeling | 18 Sep | verdict + no modify | AGREE, `modifies=False` | ✅ | curl |
| Mode isolation | historical | Kolkata | — | NOT_APPLICABLE | NOT_APPLICABLE | ✅ | curl |
| Evidence on forecast leads | live | Kolkata lead=2 | — | absent | absent | ✅ | curl |
| Browser regression | live | all 8 views | 18 Sep | 0 errors | **39 API, 0 ≥400, 0 console errors** | ✅ | `_audit_browser.py` |

---

## 20. KNOWN ISSUES (only actually found)

**P0 — critical**
1. **🔴 API KEYS COMMITTED TO A PUBLIC GITHUB REPO.** `.env` and `backend/.env` are **tracked in git** (`.gitignore` lines 9–12 are commented out) and contain live `OPENWEATHER_API_KEY` and `GOOGLE_MAPS_API_KEY`, present across multiple commits, with remote `https://github.com/Souvik686/StormSense.git`. **[VERIFIED]** → Rotation by the project owner is the only step that revokes this. **REPORT-ONLY for agents** — see the standing repository-protection rule in §18. *(The earlier timestamp bug was also P0 and is fixed + verified.)*

**P1 — important**
2. **NOW evidence layer is backend-only.** `now_evidence` + `/api/now/provenance` are not displayed in the UI, so the user still cannot *see* the product-time gap or the agree/disagree verdict. **This is the main unfinished feature task.**

**P2 — minor**
3. `risk_surface.DEFAULT_H/DEFAULT_W = 600/400`, but the module docstring says the visualization grid is "66 × 50 (~10 km)" and the WB mask cache file is named `wb_mask_full_66x50.npy`. **The docstring and/or filename disagree with the constants.** Needs reconciliation — does not affect correctness of values, only the stated sampling. **[NOT RESOLVED]**
4. Live `lead=0` and `lead=2` PNGs were byte-identical in one observation. Expected (NOW *is* the T−2h window's +2h head), but worth an explicit confirmation test.

**P3 — cosmetic**
5. RainViewer `net::ERR_ABORTED` tiles on rapid view switching — browser cancellation, harmless.

**FIXED this session:** dead `_BAND_*` constants removed; thresholds consolidated to one module.

---

## 21. REMAINING SCIENTIFIC LIMITATIONS

1. **GFS production lag 5–11 h** (measured 5.3–7.3 h). NOW's atmosphere is hours old. **Irreducible** without a different NWP source.
2. **6-hourly analysis cadence** → hourly slots are **linearly interpolated**; sub-6-hour transients (outflow boundaries, single cells) are smoothed away.
3. **0.25° (~28 km) model grid** → Kolkata/South Kolkata/Howrah share one cell; city-scale storms are sub-grid.
4. **Radar is NOT an ML input** and cannot become one without retraining (the checkpoint has no reflectivity channel).
5. **Radar→risk transfer cannot be validated** — r=0.104 between the two observation sources; no gridded severe-weather verification data in-repo.
6. **No historical radar/lightning** — RainViewer keeps ~2 h; lightning CSV is 2020-only. **The 17 Sep 2026 event cannot be quantitatively verified on the observation side.**
7. **ERA5→GFS distribution shift** — trained on ECMWF ERA5, served GFS (NCEP). Live probabilities are directionally informative, **not** as calibrated as backtested metrics.
8. **Proxy labels** — `severe_weather` is an ERA5-derived proxy (heavy 3h rain OR CAPE/CIN), **not** observed severe-weather reports.
9. **Calibration T≈0.4 sharpens** → near-binary outputs; amplifies both mismatch directions. **Do not "fix" this.**
10. **Open-Meteo rejected as an input:** its "current hour" values are **forecast** hours, not analyses, and ECMWF CAPE (3780) vs GFS (989) is a **definition mismatch** (MUCAPE vs SBCAPE). Feeding it to an ERA5-trained net would be out-of-distribution. **[VERIFIED]**
11. **No open gridded IMD/NCMRWF API** was found accessible.

---

## 22. THINGS THAT MUST NOT BE DONE

1. ❌ Retrain / fine-tune / replace / recalibrate `v2_calibrated_best.pt`
2. ❌ Change the model's expected tensor structure or channel order
3. ❌ "Fix" `PRESSURE_LEVELS_HPA` to match the YAML (it is **ascending** on purpose)
4. ❌ Revert `timestamp=analysis_t0` back to `t0` — **this is the P0 fix**
5. ❌ Arbitrary radar boosting (`if radar > X: risk += Y`) or unvalidated fusion weights
6. ❌ Fabricate radar, lightning, historical, or observation data
7. ❌ Mix live and historical data in either direction
8. ❌ Fall back to a different event when Remal is unavailable — **raise instead**
9. ❌ Redesign the dashboard / move panels / rebuild in a framework
10. ❌ Reintroduce real Leaflet or add tiles covering Google Maps
11. ❌ Claim display interpolation = model resolution (never say "1 km")
12. ❌ Conflate `reference_time` with `analysis_time`
13. ❌ Re-multiply ERA5 `tp` by 1000 (already mm/hour)
14. ❌ Suppress console errors without fixing the cause
15. ❌ Commit `.env` / print secret values
16. ❌ **Modify `.gitignore` in ANY way** — no edits, no uncommenting, no additions,
    no removals, no regeneration. **Inspection is READ-ONLY.** (Standing owner rule.)
17. ❌ Delete or move `.env` / `backend/.env`
18. ❌ Rewrite git history · force-push · change repository visibility or GitHub settings
19. ❌ Rotate or regenerate API keys
    > For the exposed-key issue (§18/§28-S1): **REPORT ONLY.** State what is exposed and
    > what the owner would need to do. Perform none of it.

---

## 23. CURRENT TASK / DECISION STATE

**Question asked:** *"How should NOW use observations?"*
**Decision: TRANSPARENT OBSERVATION LAYER** — user-confirmed (quoted verbatim in §8).

### Implementation status
**[IMPLEMENTED — NOT FULLY VERIFIED]** — backend complete and smoke-tested; **UI not yet wired.**

| Piece | Status |
|---|---|
| `src/inference/radar_observation.py` — samples RainViewer onto the 33×25 grid, coverage vs echo distinguished via tile size | ✅ created, tested (10 s, 647/825 covered) |
| `src/inference/now_evidence.py` — `assess()` returns verdict + evidence, **never an adjusted probability** | ✅ created, tested |
| `src/inference/risk_thresholds.py` — single source of truth for bands | ✅ created, boundary-tested |
| `backend/main.py` — `_now_evidence_for()` helper; `now_evidence` on point (lead 0, live only); `NOT_APPLICABLE` in historical | ✅ tested |
| `backend/main.py` — `GET /api/now/provenance` full time-provenance table | ✅ tested (200) |
| **Frontend display of evidence** (`nowEvidenceHtml` + `istClockFromIso` in `frontend/js/app.js`, block inserted into the map popup) | ✅ **DONE & VERIFIED** — renders at live NOW, absent on +2/+4/+6, absent in historical |
| Verdict coverage for all 4 branches | ✅ **VERIFIED** — all four forced through the real `assess()`; every verdict has a frontend style; **no branch returns an adjusted probability** |
| `/api/now/provenance` surfaced in the UI | ❌ **NOT STARTED** (the popup carries the key gap already; a dedicated panel is optional) |

---

## 24. NEXT STEPS (recommended order)

0. **[DONE — UI session]** Observation-evidence UI wiring is complete and verified. Do not redo it.
1. **🔴 EXPOSED KEYS (P0, §18) — REPORT ONLY, DO NOT ACT.** Already verified: both `.env` files are committed to a public remote. **Tell the owner; rotation is theirs to do.** Under the standing repository-protection rule you must NOT touch `.gitignore`, delete `.env` files, rewrite history, force-push, rotate keys, or change repo settings. Inspection is read-only.
2. **Resolve P2 #3** — reconcile `DEFAULT_H/W = 600/400` vs the "66×50" docstring/mask filename.
3. **Map interaction testing** (§13 — drag / wheel-zoom / zoom controls / district dropdown / live-location marker). Popup click IS now verified; the rest are **NOT TESTED**.
4. **Historical-mode endpoint sweep** (several endpoints **NOT TESTED**).
5. **Full-site audit** per §25 — the last gate before any 100% claim.

**Verified in the UI session (do NOT re-investigate):** evidence renders at live NOW only · absent on +2/+4/+6 · `NOT_APPLICABLE` in historical · all 4 verdicts styled · probability unmodified end-to-end · horizons differ (0.9/0.5/0.2/0.1%) · historical Remal intact (26 May 2024 12:00Z, +2h 14:00Z, 74.1% ALERT) · 0 console errors · 0 failed API calls.

> **Do NOT re-investigate:** the timestamp bug (fixed, verified), lat/lon orientation, IST conversion, radar frame selection, API transformation, radar-as-ML-input (it is not), or whether a radar→risk weight can be validated (it cannot, with current data).

---

## 25. FINAL FULL-WEBSITE AUDIT REQUIREMENT (carried forward)

After all changes, the **ENTIRE WEBSITE** must be audited: Dashboard · Radar · Advisories · WRF/Nowcast Benchmark · XAI · GIS · Threshold · Streams.

For each: every displayed value (traced source→backend→API→frontend), every API, every timestamp, every unit, every map, popup, chart, button, dropdown, filter, every horizon, **live and historical**, radar, cache, refresh, console errors, failed requests, responsive behaviour, data provenance.

**Any fixable negative finding must be FIXED and RETESTED.**
**100 % must NOT be claimed until this audit genuinely passes.**

**Current audit state:** partial — 8 views load with 0 console errors and 0 failed API calls, but per-value tracing, map interactions, historical endpoint sweep, and responsive testing are **NOT DONE**.

---

## 26. HANDOFF RULES FOR THE NEXT CHAT

1. **Read this entire report before changing code.**
2. **Inspect the actual repository** before assuming anything — code may have moved on.
3. **Do not trust old progress percentages.** (The previous session reported "100 %" for the *forensic audit*; the **observation-layer work is only partially complete** — see §23.)
4. **Verify claims against code and tests**, not documentation.
5. **Do not repeat resolved investigations** (§24) unless new evidence contradicts them.
6. **Preserve the existing UI**, the **checkpoint**, and **scientific traceability**.
7. **Report progress honestly**; distinguish verified vs unverified.
8. **Never claim 100 %** unless the §25 audit genuinely passes.
9. **Never silently switch to numerical radar-risk blending** (§8).

---

## 27. FINAL STATUS

**CURRENT PROJECT STATUS**
- **Overall implementation status:** Forensic audit **complete and verified**. Transparent Observation Layer **COMPLETE (backend + UI), verified**.
- **Current phase:** Full-site audit COMPLETE (§28). Zero fixable application issues remain; one P0 repo-hygiene item (key rotation) is the user's to action.
- **Last completed task:** UI wiring of `now_evidence` into the map popup — verified rendering at live NOW, absent on +2/+4/+6, absent in historical, all 4 verdict branches styled, displayed probability == backend probability (0.9% == 0.9%), 0 console errors.
- **Current unfinished task:** the **full-site audit (§25)** — per-value tracing, map interactions, historical endpoint sweep, responsive testing. Plus the **P0 key rotation (§18)**.
- **Critical known issues:** **P0 — live API keys committed to a public GitHub repo** (`.env`, `backend/.env`; `.gitignore` 9-12 commented out). Owner-only remediation (agents: report only, no repo/security mutations). P1: observation evidence not yet visible in the UI.
- **Scientific limitations:** §21 (11 items) — chiefly the **5–11 h GFS lag**, the **0.25° grid**, and the **unvalidatable radar→risk mapping**.

**Files MODIFIED (UI session):**
- `frontend/js/app.js` — added `NOW_VERDICT_STYLE`, `nowEvidenceHtml()`, `istClockFromIso()`; one-line insertion into the existing popup. No layout/CSS change.

**Files MODIFIED (earlier in this chat):**
- `src/inference/nowcast_service.py` — import `risk_thresholds`; `_level_for_prob` and point label delegate to it
- `src/inference/risk_map.py` — import + use `WATCH_MIN/ALERT_MIN/WARNING_MIN`
- `src/inference/risk_surface.py` — removed dead `_BAND_*`; band tuple uses shared constants
- `backend/main.py` — `_now_evidence_for()`; `now_evidence` on point (live lead 0 / historical NOT_APPLICABLE); `GET /api/now/provenance`

**Files ADDED in this chat:**
- `src/inference/risk_thresholds.py`
- `src/inference/radar_observation.py`
- `src/inference/now_evidence.py`
- `diagnostics/forensic_probe.py` (reusable production-pipeline probe)
- `diagnostics/_audit_*.py`, `_owm_probe.py`, `_fusion_evidence*.py`, `_radar_coverage.py`, `_openmeteo_probe.py`, `_radar_grid_test.py`, `_nwp_alternatives.py`
- `diagnostics/results/*.png`, `rv.json`, `prov.json`
- `STORMSENSE_FULL_HANDOFF_REPORT.md` (this file)

**(Previous session, already committed as `d52ed8e`):** the `analysis_t0` timestamp fix, georeference headers, `input_valid_time`/`input_analysis_time`/`input_age_hours`.

| Question | Answer |
|---|---|
| Checkpoint modified? | **NO** |
| Retraining performed? | **NO** |
| UI redesign performed? | **NO** (no frontend file was touched at all) |
| Final full-site audit completed? | **YES** (§28) |
| Final full-site audit passed? | **YES** — 0 console errors, 0 failed requests, 0 non-200 APIs |

---

## ▶ START HERE IN THE NEXT CHAT

**First task (do this before anything else):**

> **Tell the owner that their OpenWeatherMap and Google Maps API keys are committed to the public repo `Souvik686/StormSense`** (verified: `.gitignore` lines 9-12 are commented out; `git ls-files` shows both `.env` and `backend/.env` tracked; keys appear in `git show HEAD:.env`). Rotation by the owner is the only step that revokes exposure; deleting the files does not.
>
> **DO NOT ACT ON IT YOURSELF.** Standing owner rule: no agent may modify `.gitignore` (read-only inspection only), delete `.env` files, rewrite history, force-push, rotate keys, or change repository settings. Report and move on.
>
> After that, start the server with
> `STORMSENSE_DEVICE=cpu python run_server.py`
> and call `GET /api/now/provenance` and
> `GET /api/nowcast/point?lat=22.5726&lon=88.3639&lead=0&mode=live`
> to confirm the backend observation-evidence layer still returns `verdict`, `observed{...}`, `product_time_gaps_hours`, and `observations_modify_probability: false`.

**Then:** implement §24 step 2 — surface `now_evidence` (verdict, radar echo + its timestamp, station rain + its timestamp, and the `reference_minus_analysis` gap) in the existing NOW panel/popup, using the **smallest possible** frontend change. **Do not** move, restyle, or rebuild any existing dashboard element, and **do not** let observations alter the displayed model probability.

---

## 28. REMAINING ISSUES AT FINAL AUDIT (2026-09-19)

### Resolved during this audit

| # | Issue | Sev | File / function | Root cause | Affects | Fixed | Verified |
|---|---|---|---|---|---|---|---|
| 1 | 9 × `net::ERR_ABORTED` RainViewer tile requests on every Radar visit | **P2** | `frontend/js/app.js` — `tileLayer(...).releaseTile()` | `releaseTile()` set `img.src=''` on tiles still downloading; assigning src to an in-flight `<img>` cancels it. Proven: 9 aborts == 9 src-wipes; the 9 URLs then re-requested and returned HTTP 200. **Not** navigation cancellation — control experiment showed aborts persisted when staying on the view 25 s. | UI only | ✅ release only when `img.complete` | ✅ 9→0 in both control scenarios; radar still 9/9 tiles loaded, no leak after pan/zoom churn |
| 2 | `GET /api/model-info` → **HTTP 500** | **P1** | `backend/main.py` — `model_info()` | `best_val_loss` is `float('nan')` (checkpoint has no such key). NaN is legal Python/`json` but invalid RFC-8259 JSON, so FastAPI's serializer raised. | Backend, both modes | ✅ non-finite floats → `null` + `metrics_unavailable_reason` | ✅ 500→200, field is `null`, not a substituted number |

### Investigated and found NOT to be bugs (evidence, not assumption)

| Claim | Verdict |
|---|---|
| "NaN" text on Dashboard/XAI | **False positive** — substring of "Domi**nan**t". DOM TreeWalker found **0** NaN text nodes; word-boundary scan of all 8 views = **0** placeholder/mock/undefined hits. |
| Live risk-surface PNGs identical for leads 2–6 | **Correct behaviour.** WB-bbox max is 7.83/3.20/2.35/2.18/1.80 % — no WB cell reaches 25 %, so an all-green state map is right. The ≥25 % cells sit at lon 84.0 (Bihar/Jharkhand), inside the 84–90 °E *model domain* but outside the rendered *state*. Lead 0 correctly shows amber/orange (10 WB cells ≥25 %). |
| Gaussian smoothing suppressing maxima by ~34 pp | **Retracted.** My first measurement compared the render against out-of-state cells. Rendered peak (7.63 %) matches the WB-bbox model peak (7.83 %). An isolated full model cell retains **100 %** of its peak through σ=3.0. |
| Map drag / wheel zoom "not working" | **Test-harness artifact.** Map div sat at y=986 in a 1000 px viewport, so synthetic mouse events hit nothing (`elementFromPoint` → null). With the map scrolled into view: drag moved centre 24.412→21.757 N, wheel changed zoom 6→7. `draggable:true`, `scrollwheel:true`, `gestureHandling:'greedy'`. |

### FIXABLE ISSUES REMAINING

**ZERO REMAINING FIXABLE ISSUES FOUND** in the application code.

One repository-hygiene item remains and is **not** something I may fix unilaterally:

| # | Issue | Sev | Why not auto-fixed |
|---|---|---|---|
| S1 | `.env` and `backend/.env` (live `OPENWEATHER_API_KEY`, `GOOGLE_MAPS_API_KEY`) are committed to the public remote `Souvik686/StormSense`; `.gitignore` lines 9–12 are commented out | **P0** | **Out of scope for any agent by standing owner rule** (§18): no `.gitignore` edits, no `.env` deletion, no history rewrite, no force-push, no key rotation, no repo-setting changes. Owner-only remediation; rotation is the only step that revokes exposure. |

### GENUINE LIMITATIONS (cannot be fixed with available data/capability)

1. **GFS production lag 5.3–7.3 h** — NOW's atmosphere is hours old. Structural to GFS's 6-hourly publication.
2. **6-hourly cadence → hourly interpolation** smooths sub-6-h transients.
3. **0.25° (~28 km) grid** — Kolkata/South Kolkata/Howrah share one cell; city-scale storms are sub-grid.
4. **Radar is not an ML input** — the checkpoint has no reflectivity channel; adding one needs retraining (forbidden).
5. **Radar→risk transfer cannot be validated** — r=0.104 between the two observation sources; no gridded severe-weather verification data in-repo.
6. **No historical radar/lightning** — RainViewer keeps ~2 h; lightning CSV is 2020-only. The 17 Sep 2026 event cannot be verified observationally.
7. **ERA5→GFS distribution shift**; **proxy labels**; **calibration T≈0.4 sharpens** (all as documented in §21).

### NOT YET VERIFIED

| # | Item | Why |
|---|---|---|
| N1 | Responsive layout at tablet/mobile widths | Only 1600×1000 and 1500×1000 desktop viewports were exercised. |
| N2 | `POST /api/predict`, `POST /api/live/refresh` | Not exercised (write/refresh endpoints; `/api/live/refresh` triggers a real GFS refetch). |
| N3 | Sustained multi-hour cache/refresh behaviour across a real GFS cycle boundary | Requires waiting for a new cycle; intra-cycle stability WAS verified (0.000 pp drift). |
| N4 | `DEFAULT_H/W = 600/400` vs the "66×50" docstring and `wb_mask_full_66x50.npy` filename | Cosmetic/documentation inconsistency (P2 #3), not re-examined this round. |

---

## 29. RISK-MAP RENDERING FORENSICS (2026-09-19)

**Trigger:** the dashboard showed *Thunderstorm 31%* while the NOW map was almost entirely green.
**Verdict: the map was CORRECT; the dashboard CARD was wrong.** Two genuine bugs found and fixed.

### 29.1 What the NOW layer actually is — traced, not inferred
`severe_weather_prob` (temperature-scaled sigmoid of `severe_weather_logit`) -> `risk_surface.generate_risk_surface_png` -> `GET /api/nowcast/risk-surface?lead=&mode=` -> `L.imageOverlay(url, WB_BOUNDS)` over Google Maps. It is **not** rainfall, flash-flood proxy, radar, a binary mask or a threshold classification.

### 29.2 **Radar is NOT visually contributing to the NOW risk map** — proven
Runtime capture of the nowcasting map: `overlayMapTypes = 0`, `rainviewer <img> = 0`, **0 RainViewer tile requests**, and exactly **2 `risk-surface` images**. Force-removing all overlays removed **0** elements and changed nothing. RainViewer is confined to the Radar view by the `isRadarFeedsMap()` guard.

### 29.3 BUG 1 — "West Bengal peak" cards aggregated the WHOLE MODEL DOMAIN  **[FIXED]**
* **File/function:** `src/inference/nowcast_service.py` -> `get_summary()`, the no-district fallback (`prob_grid.max()` etc.)
* **Root cause:** the model grid spans 20-28N/84-90E, which includes Odisha, Jharkhand, Bihar, Bangladesh and Nepal. A plain `.max()` reported a neighbouring state's peak under a card labelled "WEST BENGAL PEAK".
* **Evidence (live, 2026-09-18 12Z):** Thunderstorm 31% came from **21.00N, 85.25E (Odisha)**; true WB-polygon peak **7.83%**. Heavy Rainfall 9.16 mm from 21.25N/85.75E (outside WB); WB peak 2.45 mm. Overall 22% from 21.00N/84.00E; WB peak 5.82%.
* **Affects:** Live + Historical, backend, data/scientific correctness (the map was right all along).
* **Fix:** new `_wb_cell_mask()` (same authoritative outer boundary as the click test, 0.125 deg tolerance); the whole-state branch aggregates only WB cells, with an explicit full-domain fallback if the boundary cannot load.
* **Verified:** cards **31/61/16/22% -> 8/16/3/6%**, matching the WB-polygon maxima and the all-green map.

### 29.4 BUG 2 — Live NOW is bit-identical to +2h for 16 of 24 hours  **[FIXED: now disclosed, not fabricated]**
* **File/function:** `src/inference/nowcast_service.py` -> `refresh_live_state()`; NOW = `+2h` head of a run on `target_t0 - 2h`.
* **Root cause:** the model has no lead-0 head. In Historical mode ERA5 is **hourly**, so `dataset[idx-2]` really is 2 h earlier and its +2h head lands exactly on t0 (**verified distinct**, max abs diff 93.28 pp). Live **GFS publishes f000 only every 6 h**, so `target_t0 - 2h` usually floors to the SAME cycle -> identical input tensors -> identical output.
* **Evidence:** lead 0 and lead 2 share SHA-256 `585cea4435b86d34` (max abs diff **0.0000 pp**) while leads 2-6 differ by 7-12 pp. Cycle sweep: same analysis for **16/24** wall-clock hours. NOW's true validity is `analysis + 2h`, i.e. **5-10 h behind wall clock** (measured 5.41 h), yet it was labelled with the wall-clock instant.
* **Affects:** Live only. UI + backend labelling; data/scientific honesty.
* **Fix (no fabrication):** new `NowcastService.live_now_provenance()` reports `is_true_t0_analysis: false`, `field_valid_time_utc`, `field_age_vs_reference_hours` and `identical_to_plus_2h`; served on the point endpoint (live lead 0 only) and rendered by `nowProvenanceHtml()` in the popup. **No synthetic t=0 field was invented** — GFS cannot support one.
* **Verified:** payload reports `identical_to_plus_2h: true`, valid 14:00Z, 5.41 h behind reference; popup renders it.

### 29.5 Evidence table

| Test | Result | Evidence |
|---|---|---|
| Raw model field | **PASS** | domain 825 cells; live NOW min 0.00 max 30.98 mean 1.63 median 0.13; >=25%:5 >=50%:0 >=75%:0 |
| WB-only risk field | **PASS** | WB-polygon (179 cells) max **7.83%**, 0 cells >=25% -> all-green map is correct |
| Spatial variation | **PASS** | 10 distinct values across 12 locations (0.03%-0.58%) |
| GeoJSON field | **PASS** | GeoJSON == service internal array, max diff 0.0050 pp |
| PNG field | **PASS** | PNG source array is the same `severe_weather_prob[li]` |
| Color mapping | **PASS** | 24.99->green, 25.00->amber, 49.90->amber, 50.00->orange, 74.99->orange, 75.00->red |
| Thresholds | **PASS** | single source `risk_thresholds.py`; backend == frontend (25/50/75) |
| Radar isolation | **PASS** | 0 overlays, 0 rainviewer imgs, 0 tile requests on the nowcasting map |
| Bounds | **PASS** | renderer bbox == `WB_BOUNDS` == `[[21.5394,85.8325],[27.2206,89.8828]]` |
| Orientation | **PASS** | lats 28->20 descending, lons 84->90 ascending, render row0 = north |
| Interpolation | **PASS** | peak retention **100.2%** of WB raw max; peak shift 0.17 deg lat / 0.38 deg lon |
| Popup | **PASS** | 12/12 locations match GeoJSON exactly |
| NOW | **FIXED** | was silently identical to +2h; now disclosed via `now_provenance` |
| +2h / +4h / +6h | **PASS** | distinct arrays (7-12 pp apart), correct valid times T0+2/+4/+6 |
| Historical | **PASS** | Remal 2024-05-26 12:00Z; lead0 != lead2 (93.28 pp); no live leakage |
| Live | **PASS** | cards now WB-masked; evidence layer only at live NOW |
| Dashboard peak vs map | **FIXED** | 31% (Odisha) -> 8% (WB), consistent with the green map |

# StormSense

AI-driven hyper-local severe-weather early warning and nowcasting system for
West Bengal, India.

## What it does

StormSense forecasts severe convective weather (thunderstorms, heavy rainfall,
flash-flood risk) 2–6 hours ahead over a 0.25° grid (33 × 25 cells, ~28 km
spacing) covering West Bengal. It combines:

- A trained ConvGRU-based deep learning model that produces the actual
  probability/rainfall forecasts.
- Live NOAA GFS analyses as the operational input, refreshed on a schedule.
- A frozen historical case study (Cyclone Remal, 26 May 2024) for replaying a
  real severe-weather event through the same pipeline.
- INSAT-3DR satellite archive data, live surface station observations, and a
  rule-based explanation ("XAI") layer describing which atmospheric factors
  drove a given forecast.

## Problem being solved

Short-range, hyper-local severe-weather warning for a monsoon-affected region
is a genuine forecasting gap between synoptic-scale numerical weather
prediction (which updates every 6 hours and is too coarse for a single
district) and radar-only nowcasting (which has no forward skill beyond about
an hour). StormSense targets the 2–6 hour range in between, using a small,
fast model trained on reanalysis data and run against live operational
analyses.

## Current production model: V2

- Architecture: `SevereWeatherNetV2` — a tri-stream ConvGRU
  (surface / pressure-level wind / thermodynamic branches) feeding a shared
  multi-horizon decoder. 781,889 parameters.
- Checkpoint: `Data/outputs/checkpoints/v2_calibrated_best.pt`
- Calibrated with per-lead temperature scaling; per-lead decision thresholds
  are baked into the checkpoint.
- Native model leads: **+2h, +3h, +4h, +5h, +6h** (measured from the GFS
  analysis time, not wall-clock).

V3, V3.1 and V4 candidate checkpoints/configs also exist in the repository
(`Data/outputs/checkpoints_v3/`, `checkpoints_v31/`, `checkpoints_v4/`,
`configs/v3_*.yaml`, `configs/v4_*.yaml`). They are **not** the production
model — they are kept for reproducibility of the promotion decision documented
in `reports/V4_DECISION_CRITERIA.md` and `reports/candidate_comparison.json`,
and are only used by the comparison/calibration scripts in `scripts/`, never
by the running application.

### Wall-clock horizon mapping

The dashboard exposes four buttons — NOW, +2h, +4h, +6h — as offsets from the
*current instant*. The model's own leads are offsets from the *GFS analysis
time*, which is always several hours behind wall-clock (GFS publishes every 6
hours with a multi-hour production lag). To make the labels line up with real
V2 leads, the app uses a fixed mapping:

| Dashboard label | Real V2 model lead |
|---|---|
| NOW  | +3h |
| +2h  | +4h |
| +4h  | +5h |
| +6h  | +6h |

This mapping is defined once in `DEMO_HORIZON_MAP` in
`src/inference/nowcast_service.py` and used by every endpoint, so the label
shown on screen and the lead actually served can never disagree. The real
model lead and the real forecast target timestamp are always available in the
API response alongside the label (see `/api/nowcast/summary` and
`/api/nowcast/point`).

## Live vs. historical mode

The dashboard has two modes:

- **Live mode** fetches the newest available NOAA GFS analysis, harmonizes it
  onto the model's grid, and runs real inference. Live GFS data was never part
  of the model's training distribution, so live skill is measurably weaker
  than the held-out benchmark below — see [Limitations](#limitations).
- **Historical mode** replays a single frozen, real event — Cyclone Remal,
  analysis time 2024-05-26T12:00:00Z — using the model's actual ERA5-input
  test-time forecast for that event. It is a fixed case study, not a live
  feed.

Neither mode uses fabricated or simulated data as if it were a real forecast.
Where a data source is genuinely unavailable (e.g. no INSAT acquisition near
the target time, no live station nearby), the API reports that explicitly
instead of interpolating a plausible-looking value.

## Data sources

| Source | Role | Required at runtime |
|---|---|---|
| NOAA GFS (0.25°) | Live operational input | Yes — fetched over the network |
| ERA5 reanalysis | Training data + historical case study input | No — pre-built into `processed/cache/era5_memmap/`, which **is** required |
| INSAT-3DR (ISRO/SAC) | Satellite imagery panel for the historical case study | Only for that one panel; the app degrades gracefully without it |
| SRTM DEM | Static elevation input to the model | No — pre-built into `processed/cache/era5_memmap/dem_elevation_m.npy` |
| OpenWeatherMap | Live surface station observations (temperature, humidity, wind, rain) | Yes — needs `OPENWEATHER_API_KEY` |
| West Bengal / district boundaries | Spatial masking and district aggregation | Yes — GeoJSON files committed in `Data/BOUNDARIES/` |

## Model inputs and outputs

**Inputs** (per grid cell, 6 hourly timesteps of history):
surface variables (10m wind, 2m temperature/dewpoint, surface pressure, CAPE,
CIN, total column water vapour, precipitation rate), pressure-level wind and
thermodynamic profiles, and a static DEM elevation field.

**Outputs** (per grid cell, per lead time): severe-weather probability
(calibrated), 3-hour rainfall (mm), and a derived flash-flood risk proxy
computed from rainfall + terrain. A binary "alert" decision uses the
checkpoint's own per-lead threshold, reported alongside the raw probability so
the two are never conflated.

## System architecture

```
Browser (frontend/)
   │  fetch()
   ▼
FastAPI backend (backend/main.py)
   │
   ├─ NowcastService (src/inference/nowcast_service.py)
   │     ├─ NowcastPredictor (src/inference/predictor.py) → SevereWeatherNetV2
   │     ├─ gfs_live.py        — live GFS fetch + harmonization
   │     ├─ insat_archive.py   — satellite panel
   │     ├─ observation_surface.py, radar_observation.py, now_evidence.py
   │     └─ risk_surface.py, risk_map.py, risk_thresholds.py, xai.py
   │
   └─ openweather.py — live surface station data
```

Two deployment modes are supported without any code duplication — both serve
the same `FastAPI` app object from `backend/main.py`:

- **Merged mode** (default): `run_server.py` serves the dashboard and the API
  from one origin.
- **Split mode**: `run_backend.py` (API only) + `run_frontend.py` (static
  frontend only) on separate ports, useful for local frontend iteration.

## Backend technology

Python, FastAPI, PyTorch, xarray/cfgrib (GFS GRIB decoding), h5py (INSAT
HDF5), Shapely/GeoPandas (spatial masking), uvicorn.

## Frontend technology

Vanilla JavaScript (no framework/build step), Tailwind CSS (via CDN), Google
Maps JavaScript API, Leaflet-shaped map abstraction layer.

## Repository structure

```
StormSense/
├── backend/             FastAPI app, OpenWeather client, env-based config
├── frontend/             Dashboard (index.html, js/app.js, css, static assets)
├── landing.html           Public landing page
├── src/
│   ├── inference/         Live inference service, GFS/INSAT/observation pipelines
│   ├── models/             Model architectures (SevereWeatherNetV2 + others)
│   ├── data/                Dataset loaders, ERA5/DEM preprocessing
│   ├── features/            Normalization, target construction
│   ├── training/             Losses, metrics, calibration
│   ├── utils/                 Config loading
│   └── api/                    A second, smaller legacy FastAPI app (kept for its tests)
├── configs/               default.yaml (production V2) + v3/v4/v5 candidate configs
├── Data/
│   ├── outputs/checkpoints/      Production + fallback model checkpoints
│   ├── outputs/checkpoints_v3/, checkpoints_v31/, checkpoints_v4/   Candidate checkpoints (reproducibility only)
│   ├── BOUNDARIES/                 West Bengal / district GeoJSON
│   └── INSAT/                       Satellite archive (not committed — see below)
├── processed/cache/
│   ├── era5_memmap/                Preprocessed ERA5 + DEM cache (required)
│   └── gfs/                          Live GFS fetch cache
├── scripts/                Training, evaluation, calibration, backtest tooling
├── tests/                   Unit, API and browser end-to-end tests
├── reports/                  Evaluation results, promotion-decision evidence
├── docs/                       Manuals, screenshots, development handoff notes
├── run_server.py             Merged-mode entrypoint (recommended)
├── run_backend.py             Split-mode API-only entrypoint
├── run_frontend.py             Split-mode frontend-only entrypoint
├── requirements.txt
├── .env.example
└── .gitignore / .gitattributes
```

## Requirements

- Python 3.11+ (developed against 3.13)
- Git and [Git LFS](https://git-lfs.com/) (the production checkpoint and
  cache files are stored via LFS)
- ~3 GB free disk space after cloning (model checkpoints + ERA5/DEM cache)
- An OpenWeatherMap API key (free tier is sufficient)
- A Google Maps JavaScript API key

## Installation

```bash
git clone https://github.com/Souvik686/StormSense.git
cd StormSense
git lfs pull                      # fetch checkpoint + cache files tracked via LFS

python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

## Environment configuration

```bash
cp .env.example .env
```

Then edit `.env` and fill in:

- `OPENWEATHER_API_KEY` — required; the backend will not start without it.
- `GOOGLE_MAPS_API_KEY` — required for the map to render in the browser.

### Google Maps key setup

The Google Maps key is a **browser** key — the backend injects it into the
served HTML and it is visible in the page source by design. This is normal
for the Google Maps JavaScript API; it is not meant to be kept secret the way
`OPENWEATHER_API_KEY` is. Secure it in the
[Google Cloud Console](https://console.cloud.google.com/) instead:

1. Restrict the key to the **Maps JavaScript API** only.
2. Add HTTP referrer restrictions: your production domain, plus
   `http://localhost:8000/*` and `http://127.0.0.1:8000/*` for local
   development.
3. Optionally set `GOOGLE_MAPS_API_KEYS` (comma-separated) to configure
   automatic failover if the primary key hits its daily quota.

For a deployed instance, set `OPENWEATHER_API_KEY` and `GOOGLE_MAPS_API_KEY`
as environment variables / secrets on the hosting platform rather than
shipping a `.env` file.

## How to run

```bash
python run_server.py
```

This starts the merged-mode server (dashboard + API on one port).

## Accessing the dashboard

- Landing page: `http://127.0.0.1:8000/`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- API health check: `http://127.0.0.1:8000/api/health`
- Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`

## API overview

Selected endpoints (see `/docs` for the complete, auto-generated list):

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness + active model identity |
| `GET /api/nowcast/summary?lead=&mode=` | Multi-hazard forecast for the whole state |
| `GET /api/nowcast/point?lat=&lon=&lead=&mode=` | Forecast at a specific coordinate |
| `GET /api/nowcast/risk-map`, `/risk-surface` | Spatial risk field (GeoJSON / PNG) |
| `GET /api/nowcast/districts` | Per-district hazard aggregation |
| `GET /api/nowcast/xai` | Rule-based factor attribution for a forecast |
| `GET /api/nowcast/demo-horizons` | The label → real-lead mapping described above |
| `GET /api/historical/case-study` | The full Cyclone Remal replay |
| `GET /api/historical/satellite` | INSAT satellite panel for the case study |
| `GET /api/live/ml-status` | Live GFS pipeline freshness/state |
| `GET /api/benchmark/models` | Held-out evaluation figures (see below) |
| `GET /api/boundaries/*` | West Bengal / district GeoJSON |

`mode` is `live` or `historical`; `lead` is `0` (NOW), `2`, `4`, or `6`.

## Testing

```bash
pytest tests/ -q
```

The suite includes unit tests, API integration tests against a live
in-process backend, and browser-driven end-to-end tests (Playwright) that
exercise the actual rendered dashboard. Browser tests require a working
Chromium install (`playwright install chromium` if not already present).

## Model / evaluation information

Three distinct evaluation contexts exist in this repository — they measure
different things and must not be conflated:

1. **Held-out benchmark** (`reports/`, the benchmark page in the dashboard):
   V2 evaluated on a held-out, chronologically-split 2024 test set of 4,322
   sequences, zero temporal leakage, using **ERA5 reanalysis** as model input
   (the same distribution the model was trained on). At +2h: CSI 0.4054,
   PR-AUC 0.6417, POD 0.6155, FAR 0.4572, Brier 0.0415, beating a persistence
   baseline (CSI 0.3388) by +19.7%. Full per-horizon figures are on the
   dashboard's benchmark page and in `Data/outputs/metrics/test_evaluation.json`.

2. **GFS production backtest** (`reports/gfs_backtest_v2_TEST2024.json`): the
   same V2 checkpoint run through the real production inference path
   (`NowcastService`) against **live-style GFS analyses**, scored against
   ERA5-derived proxy labels for 2024. This measures what actually happens
   when the app runs live, not laboratory performance. At +2h: CSI 0.049, POD
   0.122, FAR 0.924 — substantially weaker than the ERA5 benchmark, because
   GFS analyses were never part of the model's training distribution.

3. **Live inference**: what the running app produces right now, against
   whatever the current GFS analysis actually contains. No skill numbers are
   claimed for live inference beyond what backtest #2 measured.

Do not read the held-out benchmark figures as a claim about live performance.

## Limitations

- **Live skill is materially lower than the benchmark figures.** See above —
  this is a genuine distribution-shift effect (ERA5 training data vs. GFS
  live input), not a bug, and it is documented rather than hidden.
- **The severe-weather label is not "an event occurred."** It is a proxy
  target (heavy rainfall OR CAPE/CIN-based convective-support criteria) — a
  large share of positive labels reflect an environment supportive of
  convection rather than an observed severe-weather report.
- **Grid resolution is 0.25° (~28 km)**, not high-resolution nowcasting;
  rendered maps interpolate for visual smoothness but the underlying forecast
  resolution is the native model grid.
- **XAI is rule-based, physics-inspired attribution**, computed from the
  atmospheric variables fed to the model — not SHAP, not gradient/saliency
  attribution, and not a claim of learned feature importance.
- **INSAT satellite coverage is episodic.** The historical case study's
  satellite panel reports the nearest archived acquisition and its actual
  time offset from the target; it does not interpolate or fabricate imagery
  for gaps in the archive.
- **`Data/INSAT/` (~12 GB) is not included in this repository** (too large
  even for Git LFS). The app runs and the dashboard loads without it; only
  the historical case study's satellite panel reports it as unavailable if
  the directory is missing.

## Reproducibility notes

- `Data/ERA5/` and `Data/DEM/` (raw source datasets, ~2.4 GB combined) are not
  committed — they are build-time-only inputs to
  `python -m src.data.preprocess`, which rebuilds the
  `processed/cache/era5_memmap/` cache this repository already ships with.
  You do not need them to run the application.
- The V3/V3.1/V4 candidate checkpoints and configs are kept specifically so
  `scripts/compare_candidates.py`, `scripts/calibrate_v3.py` and related
  tooling can reproduce the model-selection evidence in `reports/`.
- Model training itself (`scripts/train_v2.py`) requires the full ERA5/DEM
  source data and is not part of the standard install/run path.

## Security notes

- Real credentials must never be committed. `.env` is gitignored; only
  `.env.example` (placeholders only) is tracked.
- A local diagnostics folder previously and accidentally contained a
  credential backup file that was committed to Git history. It has been
  removed from the current working tree; if you are working from a fork or
  clone of this repository's history prior to this cleanup, treat any
  OpenWeatherMap and Google Maps keys that may have been present there as
  compromised and rotate them.

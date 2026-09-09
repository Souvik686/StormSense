# AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting

## Smart India Hackathon 2024 | West Bengal Region

---

## Quick Start

### 1. Preprocess (once, ~20-30 min)
```bash
python scripts/preprocess.py --config configs/default.yaml
```

### 2. Train (GPU recommended — use Google Colab)
```bash
python scripts/colab/train_colab.py --config configs/default.yaml
```

### 3. Optimize threshold
```bash
python scripts/optimize_threshold.py
```

### 4. Evaluate on test set
```bash
python scripts/evaluate.py
```

### 5. Start API
```bash
python -m src.api.run_server
# Or: uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 6. Run tests
```bash
python -m pytest tests/ -v
```

---

## Project Overview

This system predicts severe weather risk 2–6 hours ahead over a 33×25 point grid (0.25° ≈ 28 km resolution) covering West Bengal and surrounding states (20–28°N, 84–90°E).

**Three risk outputs per grid cell per forecast horizon:**
- **Severe weather probability** — thunderstorm / heavy rainfall proxy
- **Flash flood risk** — terrain-amplified proxy score
- **Overall composite risk** — weighted combination

---

## Dataset Summary (from `reports/data_inventory.json`)

| Dataset | Period | Resolution | n_hours | Notes |
|---------|--------|-----------|---------|-------|
| ERA5 pressure levels | May–Oct 2021–2024 | 0.25°, 1h | 17,472 | u,v@1000/850/700; z,q,t@700/500/300/250 hPa |
| ERA5 single levels | May–Oct 2021–2024 | 0.25°, 1h | 17,664 | u10,v10,t2m,d2m,sp,cape,cin,tcwv,tp |
| IMD Rainfall | 2020–2025 | 0.25°, daily | 6 files | Daily only; not used for hourly labels |
| SRTM DEM | static | ~30m (1″) | — | 68 tiles covering 20–28°N, 84–92°E |
| INSAT HEM/UTH/CTP/CMK | 14 event days | ~4km, 30min | 369–483 | NOT used for training (see Limitations) |
| Lightning (ISS-LIS) | May–Oct 2020 | point | 24,546 | Not continuous; 2020 only → not used |
| Radar | — | — | — | Calibration figures only; no gridded data |

**Common training period:** May 2021 – October 2022 (train) | May–Oct 2023 (val) | May–Oct 2024 (test)

---

## Architecture

```
ERA5 Surface Fields (6h × 9 vars × 33 × 25)
    → ConvGRU (T=6, hidden=32, kernel=3)  → surface state (32, 33, 25)

ERA5 Pressure Levels (6h × 5 vars × 6 levels × 33 × 25)
    → PressureEncoder (per-level 2D conv + learned level weights)  → (6h, 16, 33, 25)
    → ConvGRU (hidden=32)  → pressure state (32, 33, 25)

SRTM DEM (static, 33 × 25)
    → 1×1 Conv  → dem features (8, 33, 25)

Fusion: concat [surface_state, pressure_state, dem_feat]  → (72, 33, 25)
    → 3×3 Conv + ReLU + 3×3 Conv + ReLU  → fused (48, 33, 25)

Multi-task heads (5 lead times × 2 tasks):
    → Severe Weather Head: 1×1 Conv → (5, 33, 25) logits
    → Rainfall Head:       1×1 Conv → (5, 33, 25) mm (ReLU)

Total parameters: 130,544
```

**Architecture choices rationale:**
- ConvGRU over plain GRU: preserves spatial structure (important for hyper-local prediction)
- Small model size: justified by dataset size (~17k training hours)
- Multi-task learning: dense rainfall regression gradient helps sparse binary classification
- Focal loss (α=0.75, γ=2): handles ~3% severe-weather event rate

---

## Target / Label Definition

**All labels are ERA5-derived proxies** (no direct observation ground truth available):

| Target | Definition | Threshold |
|--------|-----------|----------|
| `heavy_rain` | 3h rolling ERA5 tp | > 15 mm (~p99.3) |
| `extreme_rain` | 3h rolling ERA5 tp | > 30 mm (~p99.9) |
| `severe_convective` | ERA5 CAPE AND CIN | CAPE > 2000 J/kg AND CIN < 50 J/kg |
| `severe_weather` (primary) | `heavy_rain OR severe_convective` | — |

**Flash flood risk** = terrain-amplified proxy (not observed events):
```
flash_flood_risk = clip(severe_weather_prob × (1 + 0.3 × max(dem_z, 0)), 0, 1)
```

---

## Train / Val / Test Split

| Split | Years | Months | n_hours |
|-------|-------|--------|---------|
| Train | 2021, 2022 | May–Oct | ~8,760 |
| Val | 2023 | May–Oct | ~4,380 |
| Test | 2024 | May–Oct | ~4,380 |

**Leakage safeguards:**
1. Whole-year chronological split (no shuffling)
2. 6-hour buffer at split boundaries
3. No window crosses off-season gap (Nov–Apr)
4. Normalization stats fit on train-only timesteps

---

## Evaluation Metrics

Reported per forecast horizon (2h / 3h / 4h / 5h / 6h):

**Classification:** Precision, Recall/POD, F1, FAR, CSI, ROC-AUC, PR-AUC, Brier Score  
**Regression:** MAE, RMSE (for 3h rainfall)

> **Note:** Training results cannot be reported here because GPU training has not been run locally. Run `scripts/evaluate.py` after training to get real numbers. The system is designed to run on Google Colab T4/A100.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Architecture + training metadata |
| POST | `/predict` | Run inference on ERA5 input arrays |
| GET | `/latest` | Latest cached prediction summary |
| GET | `/risk-map?lead_time_hours=3` | GeoJSON FeatureCollection |
| GET | `/explanation` | Feature importance metadata |

Interactive docs: `http://localhost:8000/docs`

---

## Project Structure

```
Weather-Hackathon/
├── configs/
│   └── default.yaml              # All configuration (paths, thresholds, hyperparams)
├── Data/                         # Raw data (READ-ONLY)
│   ├── ERA5/pressure_levels/     # .nc files (2021-2024, May-Oct)
│   ├── ERA5/single_levels/       # .zip files (8 zips)
│   ├── INSAT/{HEM,UTH,CTP,CMK}/ # .h5 event files (not used for training)
│   ├── IMD_RAINFALL/             # .nc files (2020-2025, daily)
│   ├── DEM/SRTM/                 # .tif tiles (68 files)
│   ├── LIGHTNING/                # CSV (2020 only)
│   ├── RADAR/                    # ZIP (calibration data only)
│   └── BOUNDARIES/               # GeoJSON (West Bengal)
├── processed/cache/
│   ├── era5_memmap           # Preprocessed feature + label cache
│   └── norm_stats.json           # Training-set normalization statistics
├── src/
│   ├── utils/config.py           # YAML config loader
│   ├── data/
│   │   ├── era5_loader.py        # ERA5 zip/nc reader
│   │   ├── dem_loader.py         # SRTM DEM mosaic + regrid
│   │   ├── preprocess.py         # Full preprocessing pipeline
│   │   ├── dataset.py            # PyTorch Dataset
│   │   └── windowing.py          # Leakage-safe temporal windowing
│   ├── features/
│   │   ├── normalize.py          # Z-score normalization (train-only fit)
│   │   └── targets.py            # Proxy label generation
│   ├── models/
│   │   ├── baseline.py           # Persistence + LogisticConv baselines
│   │   └── advanced.py           # SevereWeatherNet (ConvGRU + multi-task)
│   ├── training/
│   │   ├── losses.py             # Focal loss + multi-task loss
│   │   └── metrics.py            # CSI, POD, FAR, PR-AUC, etc.
│   ├── inference/
│   │   ├── predictor.py          # Checkpoint loader + inference runner
│   │   └── risk_map.py           # GeoJSON / NetCDF risk-map generation
│   └── api/
│       ├── app.py                # FastAPI application
│       └── run_server.py         # Uvicorn server entry point
├── scripts/
│   ├── 00_inventory.py           # Dataset inventory (writes reports/)
│   ├── preprocess.py             # Preprocessing runner
│   ├── optimize_threshold.py     # Val-set threshold optimization
│   ├── evaluate.py               # Final test-set evaluation
│   └── colab/
│       ├── train_colab.py        # Full training loop (GPU compatible)
│       └── colab_workflow.py     # Step-by-step Colab notebook cells
├── outputs/
│   ├── checkpoints/best.pt       # Best model checkpoint
│   ├── metrics/                  # Evaluation reports (JSON)
│   └── risk_maps/                # Sample GeoJSON risk maps
├── reports/
│   └── data_inventory.json       # Machine-readable dataset inventory
├── tests/
│   └── test_smoke.py             # 25-test smoke suite (6s on CPU)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Google Colab Training Guide

See `scripts/colab/colab_workflow.py` for copy-paste Colab cells.

**Steps:**
1. Upload repo to Google Drive
2. Open new Colab notebook, connect to GPU runtime (T4 or better)
3. Mount drive: `drive.mount('/content/drive')`
4. `%cd /content/drive/MyDrive/Weather-Hackathon`
5. `!pip install -q netCDF4 rasterio h5py xarray`
6. Run preprocessing: `!python scripts/preprocess.py` (~20-30 min, CPU ok)
7. Train: `!python scripts/colab/train_colab.py` (~30-45 min on T4)
8. Optimize threshold: `!python scripts/optimize_threshold.py`
9. Evaluate: `!python scripts/evaluate.py`

---

## Known Limitations

1. **Proxy targets**: All severe-weather labels are ERA5-derived. No observed lightning/flood ground truth was available in the workspace.
2. **INSAT not used for training**: The satellite data (HEM, UTH, CTP, CMK) covers only 14 event days across 2020–2024, with no overlap with the training period (2021–2022 monsoon season). Using it would require event-based learning that's outside the scope of this pipeline.
3. **Lightning CSV**: ISS-LIS orbital sampling — not a continuous ground-based network. Available only for 2020. Cannot be used as a training signal for 2021–2024.
4. **Radar data**: The ZIP contains calibration scatter plots (GPM vs ground radar), not gridded reflectivity fields.
5. **0.25° resolution**: Cannot resolve individual convective cells (~1–10 km). The model predicts grid-cell-level risk, not storm tracks.
6. **IMD rainfall daily only**: Can validate monthly/seasonal climatology but not hourly targets.
7. **Test metrics**: Require actual GPU training to produce. Use Google Colab as described above.

---

## Contact

Smart India Hackathon 2024 submission.


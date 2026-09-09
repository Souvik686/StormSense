"""
SevereWeatherNet – Complete Google Colab GPU Training Workflow
=============================================================

USAGE in Colab (paste each cell):
----------------------------------
Step 0: Mount Drive + Setup
Step 1: Preprocess (ERA5 -> memmap cache)
Step 2: Train
Step 3: Optimize decision threshold
Step 4: Evaluate on test set
Step 5: Export risk maps
Step 6: Start API server (ngrok optional)

Assumes the Weather-Hackathon repo (with Data/) is in your Google Drive.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 – Mount drive + set working directory
# ═══════════════════════════════════════════════════════════════════════════════
CELL_1 = """
from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO = '/content/drive/MyDrive/Weather-Hackathon'   # <-- change if needed
os.chdir(REPO)
sys.path.insert(0, REPO)
print(f"Working directory: {os.getcwd()}")
print(f"GPU: ", end="")
import subprocess; print(subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader',
                                                  shell=True).decode().strip())
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 – Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
CELL_2 = """
# Most packages are pre-installed in Colab; only install what's missing
!pip install -q netCDF4 rasterio h5py xarray geopandas fastapi uvicorn python-dotenv pydantic
print("Dependencies ready.")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 – Preprocess raw ERA5 → memmap cache (run ONCE, ~20-30 min on CPU)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_3 = """
import os
cache = 'processed/cache/era5_memmap'
if os.path.isdir(cache) and len(os.listdir(cache)) > 5:
    size_mb = sum(os.path.getsize(os.path.join(cache, f)) for f in os.listdir(cache) if os.path.isfile(os.path.join(cache, f))) / 1e6
    print(f"Cache already exists ({size_mb:.0f} MB). Skipping preprocessing.")
else:
    print("Running preprocessing (this takes ~20-30 min)...")
    !python scripts/preprocess.py --config configs/default.yaml
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 – Verify cache and normalization stats
# ═══════════════════════════════════════════════════════════════════════════════
CELL_4 = """
import numpy as np, json, os

cache = 'processed/cache/era5_memmap'
print("Cache arrays:", sorted(os.listdir(cache)))
surface = np.load(os.path.join(cache, 'surface.npy'), mmap_mode='r')
valid_time = np.load(os.path.join(cache, 'valid_time.npy'), mmap_mode='r')
severe = np.load(os.path.join(cache, 'target_severe_weather.npy'), mmap_mode='r')
heavy = np.load(os.path.join(cache, 'target_heavy_rain.npy'), mmap_mode='r')

print(f"Surface shape: {surface.shape}")  # (T, C, H, W)
print(f"Time range: {np.datetime64(valid_time[0], 'ns')} -> {np.datetime64(valid_time[-1], 'ns')}")
print(f"Severe weather positive rate: {severe.mean():.3%}")
print(f"Heavy rain positive rate:     {heavy.mean():.3%}")

with open('processed/cache/norm_stats.json') as f:
    stats = json.load(f)
print(f"\\nNormalization stats for CAPE: mean={stats['cape']['mean']:.1f} J/kg, std={stats['cape']['std']:.1f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 – Run leakage validation
# ═══════════════════════════════════════════════════════════════════════════════
CELL_5 = """
import numpy as np, os
from src.utils.config import load_config
from src.data.windowing import validate_no_leakage, split_summary

cfg = load_config()
times = np.load('processed/cache/era5_memmap/valid_time.npy', mmap_mode='r').astype('datetime64[ns]')

validate_no_leakage(times, cfg)
print("✅ Leakage check passed (no train/val/test overlap)")

summary = split_summary(times, cfg)
for split, info in summary.items():
    print(f"  {split}: {info}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 – Train baseline models (fast, ~2-3 min)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_6 = """
# Train the logistic regression baseline first (no GPU needed, fast)
# This gives us a meaningful lower-bound to compare against

import torch, numpy as np
from src.utils.config import load_config
from src.data.dataset import make_dataloaders
from src.models.baseline import LogisticConvBaseline, PersistenceBaseline
from src.training.losses import MultiTaskLoss
from src.training.metrics import evaluate_per_lead_time
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS

cfg = load_config()
loaders = make_dataloaders('processed/cache/era5_memmap', cfg)
lead_times = cfg.get('sequence', 'lead_times_hours')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Train: {len(loaders['train'].dataset)} samples")
print(f"Val:   {len(loaders['val'].dataset)} samples")
print(f"Test:  {len(loaders['test'].dataset)} samples")

# Persistence baseline
pb = PersistenceBaseline(len(lead_times))
all_st, all_sp, all_rt, all_rp = [], [], [], []
for batch in loaders['val']:
    preds = pb.predict(batch)
    all_st.append(batch['severe_weather'].numpy())
    all_sp.append(torch.sigmoid(preds['severe_weather_logit']).numpy())
    all_rt.append(batch['rain_3h_mm'].numpy())
    all_rp.append(preds['rain_3h_mm'].numpy())

persist_metrics = evaluate_per_lead_time(
    np.concatenate(all_st), np.concatenate(all_sp),
    np.concatenate(all_rt), np.concatenate(all_rp), lead_times)
print("\\nPersistence baseline (val):")
for k, m in persist_metrics.items():
    print(f"  {k}: csi={m['csi']:.3f} pod={m['recall_pod']:.3f} far={m['far']:.3f} mae={m['mae']:.2f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 – Train SevereWeatherNet (GPU, ~60 epochs, ~30-45 min on T4)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_7 = """
# Full training run. Checkpoints are saved to outputs/checkpoints/
# If interrupted, re-run this cell -- it will resume from last.pt

!python scripts/colab/train_colab.py \\
    --config configs/default.yaml \\
    --set dataloader.num_workers=2 \\
    --set training.epochs=60 \\
    --set training.mixed_precision=true
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 – Optimize decision threshold on validation set
# ═══════════════════════════════════════════════════════════════════════════════
CELL_8 = """
!python scripts/optimize_threshold.py \\
    --config configs/default.yaml \\
    --objective csi \\
    --n-thresholds 100
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 – Final test evaluation (run ONLY ONCE)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_9 = """
!python scripts/evaluate.py \\
    --config configs/default.yaml \\
    --n-risk-map-samples 5

# View the metrics
import json
with open('outputs/metrics/test_evaluation.json') as f:
    report = json.load(f)

print(f"\\nTest loss: {report['test_loss']:.4f}")
print(f"Threshold: {report['threshold']:.3f}")
print("\\nPer-lead metrics:")
for k, m in report['model_metrics_per_lead'].items():
    print(f"  {k}: csi={m['csi']:.3f} pod={m['recall_pod']:.3f} "
          f"far={m['far']:.3f} pr_auc={m['pr_auc']:.4f} mae={m['mae']:.2f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 – Quick inference demo
# ═══════════════════════════════════════════════════════════════════════════════
CELL_10 = """
import numpy as np, torch
from src.inference.predictor import NowcastPredictor
from src.inference.risk_map import predictions_to_summary
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS

predictor = NowcastPredictor('outputs/checkpoints/best.pt')
info = predictor.model_info()
print(f"Model parameters: {info['n_parameters']:,}")
print(f"Threshold: {info['optimal_threshold']:.3f}")
print(f"Lead times: {info['lead_times_hours']}h")

# Synthetic test input (replace with real ERA5 data for actual predictions)
T, H, W, L = 6, 33, 25, 6
surface  = np.random.randn(T, len(SINGLE_VARS), H, W).astype(np.float32)
pressure = np.random.randn(T, len(PRESSURE_VARS), L, H, W).astype(np.float32)
dem      = np.random.rand(H, W).astype(np.float32) * 500

pred = predictor.predict(surface, pressure, dem)
summary = predictions_to_summary(pred, '2024-07-15T06:00:00Z')
print("\\nSummary (synthetic input):")
for lh, m in summary['per_lead'].items():
    print(f"  {lh}: max_risk={m['max_overall_risk']:.3f} "
          f"n_alerts={m['n_cells_severe_alert']} level={m['domain_risk_level']}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 – Start API (with ngrok tunnel for external access)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_11 = """
# Option A: direct uvicorn (accessible only within Colab)
# !uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

# Option B: ngrok tunnel (install ngrok, set auth token)
# !pip install -q pyngrok
# from pyngrok import ngrok
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # from ngrok.com/signup
# public_url = ngrok.connect(8000)
# print(f"API docs: {public_url}/docs")
# !uvicorn src.api.app:app --host 0.0.0.0 --port 8000

print("To start API: run `!uvicorn src.api.app:app --host 0.0.0.0 --port 8000` in a cell")
print("Docs at: http://localhost:8000/docs")
"""

if __name__ == "__main__":
    print("Colab training workflow script.")
    print("Paste each CELL_N string into a separate Colab code cell and run in order.")
    print("\nCells:")
    for i in range(1, 12):
        name = f"CELL_{i}"
        val = globals()[name]
        print(f"\n{'='*60}")
        print(f"CELL {i}:")
        print(val.strip())


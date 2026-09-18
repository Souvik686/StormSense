"""Intermittency proof: within ONE GFS analysis window (6h), the served risk
map changes purely because wall-clock advances. No new data arrives."""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone
from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor

cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
pred=get_predictor("Data/outputs/checkpoints/v2_calibrated_best.pt",None,device="cpu")
dem=np.load(os.path.join(cfg.path("paths","cache_root"),"era5_memmap","dem_elevation_m.npy")).astype(np.float32)

# Fix ONE analysis: 2026-09-17T00Z. It is the newest available for wallclock 06:00Z..11:59Z
h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=datetime(2026,9,17,6,0,tzinfo=timezone.utc),use_cache=False)
print("FIXED analysis_t0 =",h.analysis_t0.isoformat(),"(identical tensors throughout)\n")
I,J=22,17
print(f"{'wall-clock (served NOW)':>26} {'Kolkata +2h':>12} {'domain max':>11} {'cells>=25%':>11} {'cells>=50%':>11}")
prev=None
for hh,mm in [(6,0),(7,0),(8,0),(9,0),(10,0),(11,0),(11,59)]:
    ts=datetime(2026,9,17,hh,mm,tzinfo=timezone.utc)
    sp=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=ts.isoformat())["severe_weather_prob"]
    g=sp[0]
    print(f"{ts.strftime('%Y-%m-%d %H:%MZ'):>26} {g[I,J]*100:11.3f}% {g.max()*100:10.2f}% {int((g>=.25).sum()):11d} {int((g>=.50).sum()):11d}")
print("\nSame atmosphere, same cycle -> the displayed risk field still moves.")

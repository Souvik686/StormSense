"""Quantify the production bug: temporal channel says wall-clock, tensors are
analysis-time. Sweep the resulting error across a full day."""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor

cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
pred=get_predictor("Data/outputs/checkpoints/v2_calibrated_best.pt",None,device="cpu")
dem=np.load(os.path.join(cfg.path("paths","cache_root"),"era5_memmap","dem_elevation_m.npy")).astype(np.float32)
I,J=22,17
print(f"{'wallclock UTC':>18} {'analysis':>12} {'lag_h':>6} | {'BUGGY p%':>9} {'CORRECT p%':>11} {'err_pp':>8} | {'BUGmax%':>8} {'FIXmax%':>8}")
rows=[]
for hh in range(0,24,2):
    t0=datetime(2026,9,17,hh,0,tzinfo=timezone.utc)
    try: h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=t0,use_cache=False)
    except Exception as e: print(f"{t0.strftime('%m-%d %H:%MZ'):>18}  fetch fail"); continue
    bug=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=h.t0.isoformat())["severe_weather_prob"]
    fix=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=h.analysis_t0.isoformat())["severe_weather_prob"]
    lag=h.wallclock_age_hours
    print(f"{t0.strftime('%m-%d %H:%MZ'):>18} {h.analysis_t0.strftime('%d %HZ'):>12} {lag:6.1f} | "
          f"{bug[0,I,J]*100:8.3f}% {fix[0,I,J]*100:10.3f}% {(bug[0,I,J]-fix[0,I,J])*100:+7.2f} | "
          f"{bug[0].max()*100:7.2f}% {fix[0].max()*100:7.2f}%")
    rows.append((bug[0].max(),fix[0].max()))
r=np.array(rows)
print(f"\nDomain-max severe prob across the day:")
print(f"  BUGGY (wall-clock ts): min={r[:,0].min()*100:.2f}% max={r[:,0].max()*100:.2f}%  spread={np.ptp(r[:,0])*100:.2f} pp")
print(f"  FIXED (analysis ts)  : min={r[:,1].min()*100:.2f}% max={r[:,1].max()*100:.2f}%  spread={np.ptp(r[:,1])*100:.2f} pp")

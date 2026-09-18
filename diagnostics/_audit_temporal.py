"""Same analysis state, different wall-clock t0 -> how much does the temporal
encoding (hour-of-day) alone move the prediction?"""
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

# ONE fixed harmonized input (analysis 2026-09-17T06Z)
h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=datetime(2026,9,17,12,30,tzinfo=timezone.utc),use_cache=False)
print("analysis_t0 =",h.analysis_t0.isoformat(),"| slot timestamps:",h.slot_timestamps)
print("NOTE: predictor timestamp is passed as harmonized.t0 in production? -> check nowcast_service")
print()
print("Sweep the `timestamp` arg ONLY (same tensors) -> hour-of-day encoding effect:")
print(f"{'ts passed':>22} {'+2h Kolkata':>12} {'domain max':>11}")
for hh in range(0,24,3):
    ts=datetime(2026,9,17,hh,0,tzinfo=timezone.utc)
    out=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=ts.isoformat())
    sp=out["severe_weather_prob"]
    print(f"{ts.strftime('%Y-%m-%d %H:%MZ'):>22} {sp[0,I,J]*100:11.3f}% {sp[0].max()*100:10.2f}%")

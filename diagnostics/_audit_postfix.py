"""Post-fix verification: drift must vanish within a cycle, and the fixed
prediction must be reproducible from the same analysis regardless of clock."""
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
I,J=22,17
print("POST-FIX: timestamp=analysis_t0. Sweep wall-clock inside one cycle.\n")
print(f"{'wall-clock':>20} {'analysis':>12} {'Kolkata+2h':>11} {'domain max':>11}")
vals=[]
for hh in [6,7,8,9,10,11]:
    t0=datetime(2026,9,17,hh,0,tzinfo=timezone.utc)
    h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=t0,use_cache=False)
    sp=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,
                    timestamp=h.analysis_t0.isoformat())["severe_weather_prob"]
    print(f"{t0.strftime('%m-%d %H:%MZ'):>20} {h.analysis_t0.strftime('%d %HZ'):>12} {sp[0,I,J]*100:10.3f}% {sp[0].max()*100:10.2f}%")
    vals.append(sp[0].max())
print(f"\nDrift within cycle now = {np.ptp(vals)*100:.4f} pp  (was 25.25 pp)")

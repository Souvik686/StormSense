"""17 Sep 2026 Kolkata case study, POST-FIX, on real cached GFS f000 analyses."""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor
from src.features.normalize import SINGLE_VARS

cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
pred=get_predictor("Data/outputs/checkpoints/v2_calibrated_best.pt",None,device="cpu")
dem=np.load(os.path.join(cfg.path("paths","cache_root"),"era5_memmap","dem_elevation_m.npy")).astype(np.float32)
idx={v:SINGLE_VARS.index(v) for v in SINGLE_VARS}
I,J=22,17
def band(p): return "WARNING" if p>=.75 else "ALERT" if p>=.50 else "WATCH" if p>=.25 else "NORMAL"

print("Kolkata/South Kolkata both map to model cell [22,17] = 22.50N, 88.25E\n")
print(f"{'IST wall-clock':>16} {'GFS analysis':>16} {'lag':>5} {'CAPE':>6} {'CIN':>4} {'TCWV':>6} {'tp':>6} | {'NOW':>7} {'+2h':>7} {'+4h':>7} {'+6h':>7} {'band(NOW)':>10}")
for ist_h in [9,12,15,18,21,23]:
    t0=datetime(2026,9,17,ist_h,0,tzinfo=timezone.utc)-timedelta(hours=5,minutes=30)
    try:
        h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=t0,use_cache=False)
    except Exception as e:
        print(f"{ist_h:02d}:00 IST  fetch fail: {e}"); continue
    # production NOW = model run on T-2h window, take its +2h head
    h2=gfs_live.fetch_and_harmonize(lats,lons,target_t0=t0-timedelta(hours=2),use_cache=False)
    now=pred.predict(surface=h2.surface,pressure=h2.pressure,dem=dem,
                     timestamp=h2.analysis_t0.isoformat())["severe_weather_prob"][0]
    fc=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,
                    timestamp=h.analysis_t0.isoformat())["severe_weather_prob"]
    s0=h.surface[-1]
    print(f"{ist_h:02d}:00 IST ({t0.strftime('%H:%MZ')}) {h.analysis_t0.strftime('%d %b %HZ'):>16} "
          f"{h.wallclock_age_hours:4.1f}h {s0[idx['cape']][I,J]:6.0f} {s0[idx['cin']][I,J]:4.0f} "
          f"{s0[idx['tcwv']][I,J]:6.1f} {s0[idx['tp']][I,J]:6.2f} | "
          f"{now[I,J]*100:6.2f}% {fc[0,I,J]*100:6.2f}% {fc[2,I,J]*100:6.2f}% {fc[4,I,J]*100:6.2f}% {band(float(now[I,J])):>10}")
print("\nDomain-wide peak severe prob (any cell, any lead) per analysis, post-fix:")
for cy in ["20260917_000000","20260917_060000","20260917_120000","20260917_180000"]:
    an=datetime.strptime(cy,"%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=an+timedelta(hours=6),use_cache=False)
    sp=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=h.analysis_t0.isoformat())["severe_weather_prob"]
    c=h.surface[-1][idx['cape']]
    print(f"  {an.strftime('%d %b %HZ')}: domain max {sp.max()*100:6.2f}%  | Kolkata max over leads {sp[:,I,J].max()*100:6.2f}%  | domain CAPE max {np.nanmax(c):.0f}")

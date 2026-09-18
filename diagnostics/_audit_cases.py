"""Case A/B/C/D + control sweep, post-fix, on real cached GFS analyses."""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone
from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor
from src.features.normalize import SINGLE_VARS

cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
pred=get_predictor("Data/outputs/checkpoints/v2_calibrated_best.pt",None,device="cpu")
dem=np.load(os.path.join(cfg.path("paths","cache_root"),"era5_memmap","dem_elevation_m.npy")).astype(np.float32)
idx={v:SINGLE_VARS.index(v) for v in SINGLE_VARS}

LOCS={"Kolkata":(22.5726,88.3639),"S.Kolkata":(22.4950,88.3450),"Darjeeling":(27.0410,88.2663),
      "Purulia":(23.3322,86.3616),"Digha":(21.6270,87.5090),"Malda":(25.0000,88.1500)}
CYCLES=["20260916_180000","20260917_000000","20260917_060000","20260917_120000","20260917_180000","20260918_000000"]
print(f"{'analysis':>14} {'loc':>10} {'CAPE':>6} {'CIN':>5} {'TCWV':>6} {'tp mm/h':>8} | {'NOW/+2h':>8} {'band':>8}")
def band(p): return "WARNING" if p>=.75 else "ALERT" if p>=.50 else "WATCH" if p>=.25 else "NORMAL"
nA=nB=nC=nD=0
for cy in CYCLES:
    an=datetime.strptime(cy,"%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    wall=an.replace(hour=an.hour)+ __import__("datetime").timedelta(hours=6)
    try: h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=wall,use_cache=False)
    except Exception as e: print(cy,"skip",e); continue
    if h.analysis_t0!=an: pass
    sp=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=h.analysis_t0.isoformat())["severe_weather_prob"]
    s0=h.surface[-1]
    for nm,(la,lo) in LOCS.items():
        i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
        p=float(sp[0,i,j]); tp=float(s0[idx['tp']][i,j]); cape=float(s0[idx['cape']][i,j])
        print(f"{h.analysis_t0.strftime('%d %b %HZ'):>14} {nm:>10} {cape:6.0f} {float(s0[idx['cin']][i,j]):5.0f} "
              f"{float(s0[idx['tcwv']][i,j]):6.1f} {tp:8.3f} | {p*100:7.2f}% {band(p):>8}")
        wet = tp>=0.5
        if wet and p<0.25: nA+=1
        elif (not wet) and p>=0.50: nB+=1
        elif wet and p>=0.25: nC+=1
        else: nD+=1
print(f"\nGFS-tp vs risk agreement tally over {len(CYCLES)*len(LOCS)} location-times:")
print(f"  Case A (analysis wet, risk NORMAL)  : {nA}")
print(f"  Case B (analysis dry, risk >=ALERT) : {nB}")
print(f"  Case C (analysis wet, risk >=WATCH) : {nC}")
print(f"  Case D (analysis dry, risk NORMAL)  : {nD}")

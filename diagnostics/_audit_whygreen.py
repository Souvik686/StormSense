"""Did GFS f000 contain ANY evidence of the Kolkata storm on 17 Sep 2026?"""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.config import load_config
from src.inference import gfs_live
from src.features.normalize import SINGLE_VARS
cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
idx={v:SINGLE_VARS.index(v) for v in SINGLE_VARS}
I,J=22,17
hv=cfg.get("labels","heavy_rain_3h_mm"); cp=cfg.get("labels","cape_severe_jkg")
print(f"Label needs: 3h rain > {hv} mm  OR  (CAPE > {cp} AND CIN < 50)\n")
print(f"{'analysis':>14} {'tp mm/h':>8} {'3h~mm':>7} {'CAPE':>6} {'CIN':>4} | {'heavy?':>7} {'conv?':>6} {'LABEL':>6}")
for cy in ["20260917_000000","20260917_060000","20260917_120000","20260917_180000"]:
    an=datetime.strptime(cy,"%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=an+timedelta(hours=6),use_cache=False)
    s=h.surface  # (6,C,H,W) hourly slots ending at analysis
    tp=s[:,idx['tp'],I,J]; three=float(tp[-3:].sum())
    cape=float(s[-1,idx['cape'],I,J]); cin=float(s[-1,idx['cin'],I,J])
    heavy=three>hv; conv=(cape>cp and cin<50)
    print(f"{an.strftime('%d %b %HZ'):>14} {float(tp[-1]):8.3f} {three:7.3f} {cape:6.0f} {cin:4.0f} | "
          f"{str(heavy):>7} {str(conv):>6} {str(heavy or conv):>6}")
# neighbourhood: was ANY nearby cell wet?
print("\n3x3 neighbourhood around Kolkata, tp mm/h at each analysis:")
for cy in ["20260917_060000","20260917_120000"]:
    an=datetime.strptime(cy,"%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=an+timedelta(hours=6),use_cache=False)
    blk=h.surface[-1,idx['tp'],I-1:I+2,J-1:J+2]
    print(f"  {an.strftime('%d %b %HZ')}:"); 
    for row in blk: print("    "+"  ".join(f"{v:6.3f}" for v in row))

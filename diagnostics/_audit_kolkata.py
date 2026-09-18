"""Forensic: run the PRODUCTION pipeline at simulated wall-clock times across
17 Sep 2026 and inspect the Kolkata / South Kolkata grid cells."""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor

cfg=load_config(None)
d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat)
lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
print("lats[0],lats[-1] =",lats[0],lats[-1],"step",lats[1]-lats[0])
print("lons[0],lons[-1] =",lons[0],lons[-1],"step",lons[1]-lons[0])

# Kolkata / South Kolkata
TARGETS={"Kolkata (city)":(22.5726,88.3639),"South Kolkata (Tollygunge)":(22.4950,88.3450),
         "Howrah":(22.5958,88.2636)}
for name,(la,lo) in TARGETS.items():
    i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
    dlat=(lats[i]-la)*111.0; dlon=(lons[j]-lo)*111.0*np.cos(np.radians(la))
    print(f"{name}: req({la},{lo}) -> cell[{i},{j}] = ({lats[i]:.2f},{lons[j]:.2f})  offset {np.hypot(dlat,dlon):.1f} km")

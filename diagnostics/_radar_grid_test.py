import os,sys,time
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from src.utils.config import load_config
from src.inference.radar_observation import sample_echo_to_grid
cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
t=time.time()
f=sample_echo_to_grid(lats,lons)
print(f"elapsed {time.time()-t:.1f}s")
import json; print(json.dumps(f.to_summary(),indent=1))
print("\nKolkata cell [22,17] echo:",f.echo_fraction[22,17],"coverage:",f.coverage[22,17])
e=f.echo_fraction
print("grid echo>0 cells:",int(np.nansum(e>0)),"/",e.size)

"""Why is live NOW identical to +2h? Reproduce refresh_live_state's two fetches."""
import os,sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime,timezone,timedelta
from src.utils.config import load_config
from src.inference import gfs_live
cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
now=datetime.now(timezone.utc)
print("wall clock:",now.isoformat())
h  = gfs_live.fetch_and_harmonize(lats,lons,target_t0=now)
h2 = gfs_live.fetch_and_harmonize(lats,lons,target_t0=now-timedelta(hours=2))
print(f"  forecast fetch  target_t0={now.strftime('%H:%M')}  -> analysis_t0={h.analysis_t0}")
print(f"  NOW      fetch  target_t0={(now-timedelta(hours=2)).strftime('%H:%M')} -> analysis_t0={h2.analysis_t0}")
print(f"  SAME ANALYSIS? {h.analysis_t0==h2.analysis_t0}")
print(f"  surface tensors identical? {np.array_equal(h.surface,h2.surface)}")
print(f"  pressure tensors identical? {np.array_equal(h.pressure,h2.pressure)}")
print()
print("  slot timestamps (forecast):",h.slot_timestamps[-2:])
print("  slot timestamps (NOW)     :",h2.slot_timestamps[-2:])
print()
print("CONSEQUENCE: if both harmonizations resolve to the SAME GFS analysis,")
print("the 'T-2h' run reads the SAME six input slots, so its +2h head equals")
print("the forecast run's +2h head -> NOW == +2h exactly.")
print()
# when does it NOT collapse? sweep wall-clock across a cycle
print("Sweep: does target_t0-2h select an EARLIER analysis?")
for hh in range(0,24,1):
    t=datetime(now.year,now.month,now.day,hh,0,tzinfo=timezone.utc)
    c1=gfs_live._cycles_available_at(t,gfs_live.GFS_PRODUCTION_LAG_HOURS)
    c2=gfs_live._cycles_available_at(t-timedelta(hours=2),gfs_live.GFS_PRODUCTION_LAG_HOURS)
    if not c1 or not c2: continue
    diff = c1[0]!=c2[0]
    print(f"   {hh:02d}:00Z -> fc_analysis={c1[0].strftime('%d %HZ')} now_analysis={c2[0].strftime('%d %HZ')} DIFFERENT={diff}")

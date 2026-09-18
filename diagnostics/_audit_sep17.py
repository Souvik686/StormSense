"""Run the REAL production live pipeline at simulated wall-clock instants on
17 Sep 2026, using the cached GFS f000 analyses already on disk."""
import os, sys
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor
from src.features.normalize import SINGLE_VARS

cfg=load_config(None)
d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat)
lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
pred=get_predictor("Data/outputs/checkpoints/v2_calibrated_best.pt",None,device="cpu")
print("lead_times:",pred.lead_times,"temp:",pred.temperature,"thr_per_lead:",pred.threshold_per_lead)

# static DEM in meters, same as production
dem_path=os.path.join(cfg.path("paths","cache_root"),"era5_memmap","dem_elevation_m.npy")
dem=np.load(dem_path).astype(np.float32)
print("DEM shape",dem.shape,"min/max",float(dem.min()),float(dem.max()))

I,J=22,17   # Kolkata cell
# simulate IST afternoon/evening of 17 Sep 2026 -> UTC
for ist_hour in [12,15,18,21]:
    t0=datetime(2026,9,17,ist_hour,0,tzinfo=timezone.utc)-timedelta(hours=5,minutes=30)
    try:
        h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=t0,use_cache=False)
    except Exception as e:
        print(f"IST {ist_hour:02d}:00 -> FETCH FAIL: {e}"); continue
    out=pred.predict(surface=h.surface,pressure=h.pressure,dem=dem,timestamp=h.t0.isoformat())
    sp=out["severe_weather_prob"]
    idx={v:SINGLE_VARS.index(v) for v in SINGLE_VARS}
    s0=h.surface[-1]
    print(f"\n=== IST {ist_hour:02d}:00 (UTC {t0.strftime('%H:%M')}) ===")
    print(f" analysis_t0={h.analysis_t0.isoformat()}  wallclock_age={h.wallclock_age_hours:.1f}h")
    print(f" cycles={[c.isoformat() for c in h.analysis_cycles]}")
    print(f" Kolkata cell inputs: CAPE={s0[idx['cape']][I,J]:.0f} CIN={s0[idx['cin']][I,J]:.0f} "
          f"TCWV={s0[idx['tcwv']][I,J]:.1f} tp={s0[idx['tp']][I,J]:.3f}mm/h t2m={s0[idx['t2m']][I,J]-273.15:.1f}C")
    print(f" domain CAPE max={np.nanmax(s0[idx['cape']]):.0f}  domain tp max={np.nanmax(s0[idx['tp']]):.2f}mm/h")
    for li,lh in enumerate(pred.lead_times):
        print(f"   lead+{lh}h  calibrated p@Kolkata={sp[li,I,J]*100:6.2f}%   domain_max={sp[li].max()*100:6.2f}%")

"""Is Open-Meteo genuinely fresher than GFS f000 for the CURRENT hour?
And does its CAPE agree with what our GFS pipeline reads?"""
import os,sys
sys.path.insert(0,os.path.abspath("."))
os.environ.setdefault("STORMSENSE_DEVICE","cpu")
import numpy as np, httpx
from datetime import datetime,timezone,timedelta
from src.utils.config import load_config
from src.inference import gfs_live
from src.features.normalize import SINGLE_VARS

now=datetime.now(timezone.utc)
cfg=load_config(None); d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
idx={v:SINGLE_VARS.index(v) for v in SINGLE_VARS}
h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=now,use_cache=False)
print(f"our GFS analysis: {h.analysis_t0}  lag {h.wallclock_age_hours:.2f}h")
I,J=22,17
print(f"GFS@Kolkata cell: CAPE={h.surface[-1,idx['cape'],I,J]:.0f} CIN={h.surface[-1,idx['cin'],I,J]:.0f} "
      f"TCWV={h.surface[-1,idx['tcwv'],I,J]:.1f} tp={h.surface[-1,idx['tp'],I,J]:.3f}mm/h")
print()
hh=now.replace(minute=0,second=0,microsecond=0)
with httpx.Client(timeout=40) as c:
    for model in ["gfs_seamless","ecmwf_ifs025","icon_seamless"]:
        r=c.get("https://api.open-meteo.com/v1/forecast",params={
            "latitude":22.50,"longitude":88.25,
            "hourly":"cape,convective_inhibition,total_column_integrated_water_vapour,precipitation",
            "past_days":1,"forecast_days":1,"models":model})
        d2=r.json().get("hourly",{})
        if not d2: print(model,"no data"); continue
        times=d2["time"]
        tgt=hh.strftime("%Y-%m-%dT%H:00")
        if tgt in times:
            k=times.index(tgt)
            print(f"{model:15s} @ {tgt}Z  CAPE={d2['cape'][k]} CIN={d2['convective_inhibition'][k]} "
                  f"TCWV={d2['total_column_integrated_water_vapour'][k]} precip={d2['precipitation'][k]}mm")
        # how far into the future does it go, and is the CURRENT hour available?
        print(f"                 range {times[0]} .. {times[-1]}  (current hour present: {tgt in times})")

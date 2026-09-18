"""Evaluate rapid-refresh NWP alternatives for reducing the 5-11h GFS analysis lag.
Tests ACCESSIBILITY only -- no downloads of full fields."""
import httpx
from datetime import datetime,timezone,timedelta
now=datetime.now(timezone.utc)
print("wall-clock:",now.strftime("%Y-%m-%d %H:%M UTC"),"\n")
tests=[]
# 1. GFS current lag (what we use today)
def gfs_latest():
    with httpx.Client(timeout=25,follow_redirects=True) as c:
        cyc=now.replace(minute=0,second=0,microsecond=0)
        cyc-=timedelta(hours=cyc.hour%6)
        for k in range(6):
            cc=cyc-timedelta(hours=6*k)
            u=f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{cc:%Y%m%d}/{cc:%H}/atmos/gfs.t{cc:%H}z.pgrb2.0p25.f000.idx"
            try:
                r=c.get(u)
                if r.status_code==200:
                    return cc,(now-cc).total_seconds()/3600
            except Exception: pass
    return None,None
c,l=gfs_latest()
print(f"GFS f000      : latest published {c}  lag {l:.1f}h  res 0.25deg  cadence 6h  [CURRENTLY USED]")

# 2. HRRR - CONUS only, irrelevant for India but check
# 3. Open-Meteo (ECMWF IFS + others, free, no key)
try:
    with httpx.Client(timeout=30) as cl:
        r=cl.get("https://api.open-meteo.com/v1/forecast",params={
            "latitude":22.5726,"longitude":88.3639,
            "current":"temperature_2m,precipitation,surface_pressure,relative_humidity_2m",
            "hourly":"cape,precipitation","forecast_days":1,"models":"ecmwf_ifs025"})
        d=r.json()
        print(f"Open-Meteo ECMWF IFS025: HTTP {r.status_code}")
        if r.status_code==200:
            cur=d.get("current",{})
            print(f"   current time={cur.get('time')} T={cur.get('temperature_2m')} precip={cur.get('precipitation')}")
            print(f"   hourly vars available: {list(d.get('hourly',{}).keys())}")
except Exception as e: print("Open-Meteo ECMWF ERR",e)
# 4. Open-Meteo GFS/ICON with CAPE
for model in ["gfs_seamless","icon_seamless","ecmwf_ifs025"]:
    try:
        with httpx.Client(timeout=30) as cl:
            r=cl.get("https://api.open-meteo.com/v1/forecast",params={
                "latitude":22.5726,"longitude":88.3639,"hourly":"cape,convective_inhibition,total_column_integrated_water_vapour,precipitation",
                "forecast_days":1,"models":model})
            d=r.json()
            h=d.get("hourly",{})
            print(f"   {model:16s} HTTP {r.status_code} vars={[k for k in h.keys() if k!='time']}")
    except Exception as e: print(f"   {model} ERR {e}")

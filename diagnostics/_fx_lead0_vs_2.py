"""Is live lead=0 genuinely identical to lead=2, or a rendering/serving bug?"""
import json,urllib.request,numpy as np,hashlib
lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
def grid(lead,mode="live"):
    g=json.load(urllib.request.urlopen(
      f"http://127.0.0.1:8000/api/nowcast/risk-map?lead={lead}&mode={mode}&as_polygon=false",timeout=90))
    a=np.full((33,25),np.nan)
    rain=np.full((33,25),np.nan)
    for f in g["features"]:
        lo,la=f["geometry"]["coordinates"]
        i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
        a[i,j]=f["properties"]["thunderstorm_probability"]
        rain[i,j]=f["properties"]["rain_3h_mm_forecast"]
    return a,rain,g["features"][0]["properties"]
print("LIVE: are the severe-prob arrays identical between leads?")
arrs={}
for lead in [0,2,3,4,5,6]:
    a,r,p0=grid(lead)
    arrs[lead]=a
    print(f"  lead={lead}: sha={hashlib.sha256(a.tobytes()).hexdigest()[:16]} max={np.nanmax(a)*100:6.2f}% "
          f"rain_max={np.nanmax(r):6.2f}mm  lead_hours_prop={p0['lead_hours']}")
print()
for a_,b_ in [(0,2),(2,3),(3,4),(4,5),(5,6)]:
    same=np.allclose(arrs[a_],arrs[b_],equal_nan=True)
    d=np.nanmax(np.abs(arrs[a_]-arrs[b_]))
    print(f"  lead {a_} vs {b_}: identical={same}  max_abs_diff={d*100:.4f} pp")
print()
print("POINT API (independent path) at Kolkata:")
for lead in [0,2,3,4,5,6]:
    d=json.load(urllib.request.urlopen(
      f"http://127.0.0.1:8000/api/nowcast/point?lat=22.5726&lon=88.3639&lead={lead}&mode=live",timeout=60))
    print(f"  lead={lead}: {d['predictions']['thunderstorm_prob_pct']}%  valid={d['forecast_valid_utc']}")

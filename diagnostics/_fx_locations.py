"""SECTION 3+11: per-location raw values + popup agreement."""
import json,urllib.request,numpy as np
lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
LOCS={"Kolkata":(22.5726,88.3639),"Darjeeling":(27.0410,88.2663),"Siliguri":(26.7271,88.3953),
      "Digha":(21.6270,87.5090),"Purulia":(23.3322,86.3616),"Bankura":(23.2324,87.0750),
      "Murshidabad":(24.1750,88.2800),"North 24 Parganas":(22.7240,88.4790),
      "South 24 Parganas":(22.1600,88.4300),"North WB (Cooch Behar)":(26.3200,89.4500),
      "West WB (Jhargram)":(22.4500,86.9900),"South WB (Sagar Is.)":(21.6500,88.0800)}
def band(p): return "WARNING" if p>=.75 else "ALERT" if p>=.50 else "WATCH" if p>=.25 else "NORMAL"
for lead in [0,2]:
    g=json.load(urllib.request.urlopen(f"http://127.0.0.1:8000/api/nowcast/risk-map?lead={lead}&mode=live&as_polygon=false",timeout=90))
    grid=np.full((33,25),np.nan)
    for f in g["features"]:
        lo,la=f["geometry"]["coordinates"]
        grid[int(np.argmin(np.abs(lats-la))),int(np.argmin(np.abs(lons-lo)))]=f["properties"]["thunderstorm_probability"]
    print(f"\n===== LEAD={lead}h (live) =====")
    print(f"{'location':<24}{'lat':>8}{'lon':>8} {'cell':>16} {'geojson':>9} {'popup':>8} {'band':>8} {'match':>6}")
    vals=[]
    for nm,(la,lo) in LOCS.items():
        i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
        gv=grid[i,j]
        d=json.load(urllib.request.urlopen(f"http://127.0.0.1:8000/api/nowcast/point?lat={la}&lon={lo}&lead={lead}&mode=live",timeout=60))
        pv=d["predictions"]["thunderstorm_prob_pct"]
        cell=f"{lats[i]:.2f},{lons[j]:.2f}"
        ok=abs(gv*100-pv)<0.06
        print(f"{nm:<24}{la:>8.3f}{lo:>8.3f} {cell:>16} {gv*100:>8.2f}% {pv:>7.1f}% {band(gv):>8} {str(ok):>6}")
        vals.append(round(gv*100,2))
    print(f"  distinct values: {len(set(vals))}/{len(vals)}  -> spatial variation {'PRESENT' if len(set(vals))>1 else 'ABSENT (BUG)'}")

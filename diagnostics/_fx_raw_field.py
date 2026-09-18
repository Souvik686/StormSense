"""SECTION 2+3+10: raw 33x25 model field stats, domain vs WB, per-location."""
import sys,os,json,urllib.request
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from shapely.geometry import shape, Point
from src.inference.risk_surface import WB_MIN_LAT,WB_MAX_LAT,WB_MIN_LON,WB_MAX_LON,WB_STATE_GEOJSON

lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
wb=shape(json.load(open(WB_STATE_GEOJSON))["features"][0]["geometry"])

def grid_for(lead,mode):
    g=json.load(urllib.request.urlopen(
      f"http://127.0.0.1:8000/api/nowcast/risk-map?lead={lead}&mode={mode}&as_polygon=false",timeout=90))
    a=np.full((33,25),np.nan,dtype=np.float64)
    for f in g["features"]:
        lo,la=f["geometry"]["coordinates"]
        i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
        a[i,j]=f["properties"]["thunderstorm_probability"]
    return a

# masks
bbox=np.zeros((33,25),bool); poly=np.zeros((33,25),bool)
for i,la in enumerate(lats):
    for j,lo in enumerate(lons):
        if WB_MIN_LAT<=la<=WB_MAX_LAT and WB_MIN_LON<=lo<=WB_MAX_LON: bbox[i,j]=True
        if wb.contains(Point(lo,la)) or wb.distance(Point(lo,la))<0.125: poly[i,j]=True
print(f"cells: domain=825  WB-bbox={bbox.sum()}  WB-polygon(+0.125deg)={poly.sum()}\n")

def stats(a,m,label):
    v=a[m]; v=v[np.isfinite(v)]
    print(f"  {label:<26} n={v.size:>4} min={v.min()*100:6.2f} max={v.max()*100:6.2f} "
          f"mean={v.mean()*100:6.2f} median={np.median(v)*100:6.2f} "
          f">=25%:{int((v>=.25).sum()):>3} >=50%:{int((v>=.50).sum()):>3} >=75%:{int((v>=.75).sum()):>3}")

for mode in ["live","historical"]:
    print(f"===== MODE={mode} =====")
    for lead in [0,2,4,6]:
        a=grid_for(lead,mode)
        print(f" lead={lead}h")
        stats(a,np.ones_like(bbox),"FULL DOMAIN (33x25)")
        stats(a,bbox,"WB bbox")
        stats(a,poly,"WB polygon")
        mi=np.unravel_index(np.nanargmax(a),a.shape)
        inpoly=poly[mi]
        print(f"     domain argmax -> cell[{mi[0]},{mi[1]}] = {lats[mi[0]]:.2f}N,{lons[mi[1]]:.2f}E "
              f"= {a[mi]*100:.2f}%  inside_WB_polygon={inpoly}")
        if poly.any():
            b=np.where(poly,a,-1); mj=np.unravel_index(np.argmax(b),b.shape)
            print(f"     WB-poly argmax-> cell[{mj[0]},{mj[1]}] = {lats[mj[0]]:.2f}N,{lons[mj[1]]:.2f}E = {a[mj]*100:.2f}%")
    print()
np.save("diagnostics/results/fx_live_now.npy", grid_for(0,"live"))

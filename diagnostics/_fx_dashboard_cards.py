"""SECTION 10+14: which exact cell produces each dashboard card value?"""
import json,urllib.request,numpy as np,sys,os
sys.path.insert(0,os.path.abspath("."))
from shapely.geometry import shape,Point
from src.inference.risk_surface import WB_STATE_GEOJSON,WB_MIN_LAT,WB_MAX_LAT,WB_MIN_LON,WB_MAX_LON
lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
wb=shape(json.load(open(WB_STATE_GEOJSON))["features"][0]["geometry"])
s=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/nowcast/summary?lead=2&mode=live",timeout=90))
print("=== /api/nowcast/summary hazards (what the cards show) ===")
hz=s.get("hazards",{})
for k,v in hz.items():
    if isinstance(v,dict): print(f"   {k:<14} prob={v.get('probability')} level={v.get('level')} label={v.get('label')}")
g=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/nowcast/risk-map?lead=2&mode=live&as_polygon=false",timeout=90))
sev=np.full((33,25),np.nan); rain=np.full((33,25),np.nan); ff=np.full((33,25),np.nan); ov=np.full((33,25),np.nan)
for f in g["features"]:
    lo,la=f["geometry"]["coordinates"]; i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
    p=f["properties"]
    sev[i,j]=p["thunderstorm_probability"]; rain[i,j]=p["rain_3h_mm_forecast"]
    ff[i,j]=p["flash_flood_risk"]; ov[i,j]=p["overall_risk"]
def rep(name,a,unit="%",scale=100):
    mi=np.unravel_index(np.nanargmax(a),a.shape)
    la,lo=lats[mi[0]],lons[mi[1]]
    inpoly=wb.contains(Point(lo,la)) or wb.distance(Point(lo,la))<0.125
    inbbox=WB_MIN_LAT<=la<=WB_MAX_LAT and WB_MIN_LON<=lo<=WB_MAX_LON
    # WB-poly max
    best=-1;bc=None
    for i in range(33):
        for j in range(25):
            p=Point(lons[j],lats[i])
            if (wb.contains(p) or wb.distance(p)<0.125) and np.isfinite(a[i,j]) and a[i,j]>best:
                best=a[i,j];bc=(lats[i],lons[j])
    print(f"   {name:<20} DOMAIN max={a[mi]*scale:7.2f}{unit} @ {la:.2f}N,{lo:.2f}E inWB_poly={inpoly} inWB_bbox={inbbox}")
    print(f"   {'':<20} WB-POLY max={best*scale:7.2f}{unit} @ {bc[0]:.2f}N,{bc[1]:.2f}E")
print("\n=== which cell produces each card ===")
rep("Thunderstorm",sev); rep("Heavy Rainfall",rain,"mm",1); rep("Flash Flood",ff); rep("Overall",ov)

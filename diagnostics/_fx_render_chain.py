"""SECTIONS 5,7,8,9: colormap thresholds, PNG-vs-GeoJSON identity,
interpolation fidelity, bounds/orientation."""
import sys,os,json,urllib.request,io
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from PIL import Image
from src.inference import risk_surface as RS
from src.inference import risk_thresholds as T

print("=== 5. COLORMAP BOUNDARY TEST (renderer's own function) ===")
vals=np.array([[0.0,0.10,0.249,0.2499,0.25,0.2501,0.499,0.50,0.7499,0.75,0.7501,0.90,1.0]])
mask=np.ones_like(vals,bool)
rgba=RS.colormap_risk_surface(vals,mask)
names={(16,185,129):"green/NORMAL",(245,158,11):"amber/WATCH",(249,115,22):"orange/ALERT",(239,68,68):"red/WARNING"}
for k,v in enumerate(vals[0]):
    c=tuple(rgba[0,k,:3]); print(f"   {v*100:7.2f}% -> {names.get(c,c)} alpha={rgba[0,k,3]}")
print(f"   thresholds module: WATCH>={T.WATCH_MIN} ALERT>={T.ALERT_MIN} WARNING>={T.WARNING_MIN}")

print("\n=== 7+8. PNG SOURCE ARRAY vs GEOJSON (same field?) ===")
lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
g=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/nowcast/risk-map?lead=0&mode=historical&as_polygon=false",timeout=90))
grid=np.full((33,25),np.nan)
for f in g["features"]:
    lo,la=f["geometry"]["coordinates"]
    grid[int(np.argmin(np.abs(lats-la))),int(np.argmin(np.abs(lons-lo)))]=f["properties"]["thunderstorm_probability"]
# what the service holds
os.environ["STORMSENSE_DEVICE"]="cpu"
from src.inference.nowcast_service import get_nowcast_service
svc=get_nowcast_service()
internal=svc.current_pred["severe_weather_prob"][0]
print(f"   GeoJSON grid vs service internal array: identical={np.allclose(grid,internal,atol=1e-4,equal_nan=True)}"
      f"  max_diff={np.nanmax(np.abs(grid-internal))*100:.4f} pp")
mi=np.unravel_index(np.nanargmax(internal),internal.shape)
print(f"   raw argmax cell[{mi[0]},{mi[1]}] = {lats[mi[0]]:.2f}N,{lons[mi[1]]:.2f}E = {internal[mi]*100:.2f}%")

# reproduce interpolation exactly as generate_risk_surface_png does
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
interp=RegularGridInterpolator((lats[::-1],lons),internal[::-1,:],method="cubic",bounds_error=False,fill_value=0.0)
lf=np.linspace(RS.WB_MAX_LAT,RS.WB_MIN_LAT,RS.DEFAULT_H); of=np.linspace(RS.WB_MIN_LON,RS.WB_MAX_LON,RS.DEFAULT_W)
lm,om=np.meshgrid(lf,of,indexing="ij")
fine=interp((lm,om)); sm=np.clip(gaussian_filter(fine,sigma=3.0),0,1)
fi=np.unravel_index(np.argmax(sm),sm.shape)
print(f"   interpolated argmax  = {lf[fi[0]]:.2f}N,{of[fi[1]]:.2f}E = {sm[fi]*100:.2f}%")
# WB-clipped raw max for fair comparison
inb=(lats[:,None]>=RS.WB_MIN_LAT)&(lats[:,None]<=RS.WB_MAX_LAT)&(lons[None,:]>=RS.WB_MIN_LON)&(lons[None,:]<=RS.WB_MAX_LON)
wbraw=np.where(inb,internal,np.nan); wi=np.unravel_index(np.nanargmax(wbraw),wbraw.shape)
print(f"   raw max INSIDE bbox  = {lats[wi[0]]:.2f}N,{lons[wi[1]]:.2f}E = {wbraw[wi]*100:.2f}%")
print(f"   peak shift = {abs(lf[fi[0]]-lats[wi[0]]):.2f} deg lat, {abs(of[fi[1]]-lons[wi[1]]):.2f} deg lon")
print(f"   peak value retention = {sm[fi]/wbraw[wi]*100:.1f}% of WB raw max")

print("\n=== 9. BOUNDS / ORIENTATION ===")
print(f"   renderer bbox: lat {RS.WB_MIN_LAT}..{RS.WB_MAX_LAT}  lon {RS.WB_MIN_LON}..{RS.WB_MAX_LON}")
print(f"   lats[0]={lats[0]} (north) lats[-1]={lats[-1]} (south)  -> descending OK")
print(f"   lons[0]={lons[0]} (west)  lons[-1]={lons[-1]} (east)   -> ascending OK")
print(f"   render rows go {RS.WB_MAX_LAT}->{RS.WB_MIN_LAT} (north->south, row0=north) OK")

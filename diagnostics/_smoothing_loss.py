"""How much does the render pipeline suppress isolated maxima?"""
import os,sys
sys.path.insert(0,os.path.abspath("."))
import numpy as np, json, urllib.request
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from src.inference.risk_surface import (WB_MAX_LAT,WB_MIN_LAT,WB_MIN_LON,WB_MAX_LON,
                                        DEFAULT_H,DEFAULT_W,RENDER_H,RENDER_W)
lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
for lead in [2,4,6]:
    g=json.load(urllib.request.urlopen(f"http://127.0.0.1:8000/api/nowcast/risk-map?lead={lead}&mode=live&as_polygon=false",timeout=60))
    grid=np.zeros((33,25),dtype=np.float32)
    for f in g["features"]:
        la,lo=f["geometry"]["coordinates"][1],f["geometry"]["coordinates"][0]
        i=int(np.argmin(np.abs(lats-la))); j=int(np.argmin(np.abs(lons-lo)))
        grid[i,j]=f["properties"]["thunderstorm_probability"]
    # replicate generate_risk_surface_png exactly
    lat_asc=lats[::-1]; ga=grid[::-1,:]
    interp=RegularGridInterpolator((lat_asc,lons),ga,method="cubic",bounds_error=False,fill_value=0.0)
    lf=np.linspace(WB_MAX_LAT,WB_MIN_LAT,DEFAULT_H); of=np.linspace(WB_MIN_LON,WB_MAX_LON,DEFAULT_W)
    lm,om=np.meshgrid(lf,of,indexing="ij")
    fine=interp((lm,om))
    sm=np.clip(gaussian_filter(fine,sigma=3.0),0,1)
    bi=RegularGridInterpolator((lf[::-1],of),sm[::-1,:],method="cubic",bounds_error=False,fill_value=0.0)
    lr=np.linspace(WB_MAX_LAT,WB_MIN_LAT,RENDER_H); orr=np.linspace(WB_MIN_LON,WB_MAX_LON,RENDER_W)
    lrm,orm=np.meshgrid(lr,orr,indexing="ij")
    rv=np.clip(bi((lrm,orm)),0,1)
    rv2=np.clip(gaussian_filter(rv,sigma=max(4.0,RENDER_H/float(DEFAULT_H)*1.5)),0,1)
    print(f"lead={lead}: model max={grid.max()*100:6.2f}%  cells>=25%={int((grid>=.25).sum()):3d}")
    print(f"          after viz-interp+sigma3 : max={sm.max()*100:6.2f}%")
    print(f"          after render+final blur : max={rv2.max()*100:6.2f}%  <-- what the PNG colours")
    print(f"          SUPPRESSION: {(grid.max()-rv2.max())*100:.2f} pp lost\n")

"""Why do radar echo and OWM rain disagree? Check zoom/geolocation of tiles
and whether the echo pixels are where we think they are."""
import os,sys,io,math,json,urllib.request
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime,timezone
from PIL import Image
rv=json.load(urllib.request.urlopen("https://api.rainviewer.com/public/weather-maps.json",timeout=30))
host=rv["host"]; fr=rv["radar"]["past"][-1]
print("frame:",datetime.fromtimestamp(fr["time"],timezone.utc))
def get(z,x,y):
    b=urllib.request.urlopen(f"{host}{fr['path']}/256/{z}/{x}/{y}/2/1_1.png",timeout=30).read()
    return np.array(Image.open(io.BytesIO(b)).convert("RGBA")),len(b)
def t(z,lat,lon):
    n=2**z
    return ((lon+180)/360*n,(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n)
# Whole-WB coverage at several zooms: what fraction of WB has ANY echo?
print("\nWB-wide echo fraction by zoom (bbox 21.5-27.2N, 85.8-89.9E):")
for z in [5,6,7]:
    xs=set(); ys=set()
    for la in np.linspace(21.5,27.2,12):
        for lo in np.linspace(85.8,89.9,12):
            fx,fy=t(z,la,lo); xs.add(int(fx)); ys.add(int(fy))
    tot=0; nz=0; nb=0
    for x in sorted(xs):
        for y in sorted(ys):
            try:
                a,b=get(z,x,y); nb+=b
                tot+=a.shape[0]*a.shape[1]; nz+=int((a[...,3]>0).sum())
            except Exception as e: pass
    print(f"  z={z}: tiles={len(xs)*len(ys)} bytes={nb} echo px {nz}/{tot} = {100*nz/max(tot,1):.2f}%")
# What do the coloured pixels look like (are they light or heavy)?
z=7
fx,fy=t(z,23.5,84.5)
a,_=get(z,int(fx),int(fy))
m=a[...,3]>0
print(f"\nSample tile z7 at (23.5,84.5): {m.sum()} echo px")
if m.sum():
    print("  unique RGB of echo px (top 6):")
    cols,counts=np.unique(a[m][:,:3],axis=0,return_counts=True)
    for c,n in sorted(zip(cols.tolist(),counts.tolist()),key=lambda k:-k[1])[:6]:
        print("   ",c,n)

"""Do RainViewer echo and OWM rain1h agree? Measure over the WB grid.
This decides whether an observation-aware NOW is defensible at all."""
import os,sys,io,math,json,urllib.request
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime,timezone
from dotenv import load_dotenv; load_dotenv(".env")
import httpx
from PIL import Image
KEY=os.getenv("OPENWEATHER_API_KEY")

rv=json.load(urllib.request.urlopen("https://api.rainviewer.com/public/weather-maps.json",timeout=30))
host=rv["host"]; fr=rv["radar"]["past"][-1]
rt=datetime.fromtimestamp(fr["time"],timezone.utc)
print("radar frame:",rt.strftime("%Y-%m-%d %H:%M UTC"))

Z=7
_tiles={}
def tile(z,x,y):
    k=(z,x,y)
    if k in _tiles: return _tiles[k]
    try:
        b=urllib.request.urlopen(f"{host}{fr['path']}/256/{z}/{x}/{y}/2/1_1.png",timeout=30).read()
        a=np.array(Image.open(io.BytesIO(b)).convert("RGBA"))
    except Exception: a=None
    _tiles[k]=a; return a

def echo_at(lat,lon,z=Z):
    n=2**z
    fx=(lon+180)/360*n
    fy=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n
    a=tile(z,int(fx),int(fy))
    if a is None: return None
    px=int((fx-int(fx))*256); py=int((fy-int(fy))*256)
    # 5x5 window around the point
    x0,x1=max(0,px-2),min(256,px+3); y0,y1=max(0,py-2),min(256,py+3)
    w=a[y0:y1,x0:x1]
    return float((w[...,3]>0).mean())

# sample a coarse subset of the model grid
lats=np.linspace(28.0,20.0,33); lons=np.linspace(84.0,90.0,25)
pts=[(float(lats[i]),float(lons[j])) for i in range(2,33,4) for j in range(2,25,4)]
print(f"sampling {len(pts)} grid points\n")
rows=[]
with httpx.Client(timeout=25) as c:
    for la,lo in pts:
        e=echo_at(la,lo)
        try:
            d=c.get("https://api.openweathermap.org/data/2.5/weather",
                    params={"lat":la,"lon":lo,"appid":KEY,"units":"metric"}).json()
            r1=float(d.get("rain",{}).get("1h",0.0)); wx=d["weather"][0]["main"]
            rh=d["main"]["humidity"]; cl=d.get("clouds",{}).get("all",0)
        except Exception: r1,wx,rh,cl=None,"?",None,None
        rows.append((la,lo,e,r1,wx,rh,cl))
import collections
print(f"{'lat':>6}{'lon':>7}{'echo%':>7}{'owm_r1h':>9}{'wx':>10}{'RH':>4}{'cld':>5}")
for la,lo,e,r1,wx,rh,cl in rows:
    print(f"{la:6.2f}{lo:7.2f}{(e*100 if e is not None else float('nan')):7.1f}{(r1 if r1 is not None else float('nan')):9.2f}{wx:>10}{rh if rh else 0:4d}{cl if cl is not None else 0:5d}")
ev=[(e,r1,wx) for _,_,e,r1,wx in [(a,b,c,d,f) for a,b,c,d,f,_,_ in rows] if e is not None and r1 is not None]
E=np.array([x[0] for x in ev]); R=np.array([x[1] for x in ev])
print(f"\nn={len(ev)}  echo>0: {(E>0).sum()}  owm rain>0: {(R>0).sum()}")
if len(ev)>3 and E.std()>0 and R.std()>0:
    print("corr(echo, owm_rain1h) =",round(float(np.corrcoef(E,R)[0,1]),3))
rainwx=np.array([1.0 if x[2] in ("Rain","Thunderstorm","Drizzle") else 0.0 for x in ev])
print("corr(echo, wx==Rain/Thunder) =",round(float(np.corrcoef(E,rainwx)[0,1]),3) if rainwx.std()>0 else "n/a")
print("mean echo where wx=Rain:",round(float(E[rainwx==1].mean()),4) if (rainwx==1).any() else "n/a",
      "| where not:",round(float(E[rainwx==0].mean()),4) if (rainwx==0).any() else "n/a")

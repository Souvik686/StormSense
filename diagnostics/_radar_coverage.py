"""Does RainViewer have REAL radar coverage over West Bengal, or is the
apparent 'quiet' just absence of radar? Compare WB against a region with
known dense radar (Europe) in the same frame."""
import io,math,json,urllib.request
import numpy as np
from PIL import Image
rv=json.load(urllib.request.urlopen("https://api.rainviewer.com/public/weather-maps.json",timeout=30))
host=rv["host"]; fr=rv["radar"]["past"][-1]
def t(z,lat,lon):
    n=2**z
    return (int((lon+180)/360*n),int((1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n))
def stat(z,lat,lon):
    x,y=t(z,lat,lon)
    try:
        b=urllib.request.urlopen(f"{host}{fr['path']}/256/{z}/{x}/{y}/2/1_1.png",timeout=30).read()
        a=np.array(Image.open(io.BytesIO(b)).convert("RGBA"))
        return len(b), 100.0*(a[...,3]>0).mean()
    except Exception as e: return None,str(e)
print(f"{'region':<22}{'z':>3}{'bytes':>8}{'echo%':>8}")
for nm,(la,lo) in {"Kolkata WB":(22.57,88.36),"N Bengal":(26.5,88.5),"Bay of Bengal":(20.5,88.0),
                   "Germany (dense radar)":(50.1,8.7),"USA Midwest":(41.9,-87.6),
                   "Central Sahara":(23.0,12.0),"Mid Pacific":(0.0,-160.0)}.items():
    for z in [6]:
        b,e=stat(z,la,lo)
        print(f"{nm:<22}{z:>3}{(b if b else 0):>8}{(e if isinstance(e,float) else -1):>8.2f}")

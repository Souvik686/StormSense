"""Are live leads 2-6 the SAME image, or just same-size? Compare bytes+pixels."""
import urllib.request,hashlib,io
import numpy as np
from PIL import Image
B="http://127.0.0.1:8000"
def png(lead,mode):
    return urllib.request.urlopen(f"{B}/api/nowcast/risk-surface?lead={lead}&mode={mode}",timeout=60).read()
for mode in ["live","historical"]:
    print(f"\n=== {mode} ===")
    H={}
    for lead in [0,2,3,4,5,6]:
        raw=png(lead,mode); h=hashlib.sha256(raw).hexdigest()[:16]
        a=np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
        painted=(a[...,3]>0)
        # count pixels per band colour
        def cnt(rgb): return int(((a[...,0]==rgb[0])&(a[...,1]==rgb[1])&(a[...,2]==rgb[2])&painted).sum())
        print(f"  lead={lead} {len(raw):>7}B sha={h} painted={int(painted.sum()):>7} "
              f"green={cnt((16,185,129)):>7} amber={cnt((245,158,11)):>6} orange={cnt((249,115,22)):>6} red={cnt((239,68,68)):>6}")
        H[lead]=h
    dup={}
    for k,v in H.items(): dup.setdefault(v,[]).append(k)
    same=[v for v in dup.values() if len(v)>1]
    print("  identical-image groups:", same if same else "none (all distinct)")

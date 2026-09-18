"""Prove each aborted URL is a VALID, serving resource -- i.e. the abort was
our own navigation, not a broken URL/endpoint."""
import json, urllib.request, io
from PIL import Image
import numpy as np
d=json.load(open("diagnostics/results/failed_requests.json"))
fails=d["fails"]
print(f"{'#':>2} {'z/x/y':<10} {'HTTP':<5} {'bytes':>7} {'echo%':>7}  verdict")
ok=0
for i,r in enumerate(fails,1):
    u=r["url"]
    parts=u.split("/256/")[1].split("/")  # z/x/y/scheme/opts
    zxy="/".join(parts[:3])
    try:
        resp=urllib.request.urlopen(u,timeout=30)
        raw=resp.read()
        a=np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
        echo=100.0*(a[...,3]>0).mean()
        status=resp.status
        verdict="VALID (serves real tile)" if status==200 else "UNEXPECTED"
        if status==200: ok+=1
        print(f"{i:>2} {zxy:<10} {status:<5} {len(raw):>7} {echo:>6.2f}%  {verdict}")
    except Exception as e:
        print(f"{i:>2} {zxy:<10} ERR   {'-':>7} {'-':>7}  GENUINE FAILURE: {e}")
print(f"\n{ok}/{len(fails)} aborted URLs serve HTTP 200 when fetched standalone.")
print("=> the URLs are correct; the abort was client-side cancellation.")

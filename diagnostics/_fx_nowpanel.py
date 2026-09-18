"""Does the Current Location panel populate at NOW vs forecast horizons?"""
from playwright.sync_api import sync_playwright
import time
IDS=['bind-temp','bind-rain','bind-humidity','bind-wind','location-obs-age','location-obs-title']
def read(pg):
    return pg.evaluate("""(ids)=>{const o={};for(const i of ids){const e=document.getElementById(i);o[i]=e?(e.innerText||'').trim():'(missing)';}return o;}""",IDS)
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page()
    errs=[]; reqs=[]
    pg.on("console", lambda m: errs.append((m.type,m.text)) if m.type in("error","warning") else None)
    pg.on("request", lambda r: reqs.append(r.url.split("/api/")[-1]) if "forecast/conditions" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(12)
    print("ON LOAD (default horizon):")
    for k,v in read(pg).items(): print(f"   {k:<20}: {v!r}")
    print("   forecast/conditions requests:", reqs)
    for label in ["NOW","+2h","+4h"]:
        n=len(reqs)
        pg.evaluate("""(lb)=>{const b=[...document.querySelectorAll('button')].find(x=>((x.textContent||'').trim()===lb)); if(b)b.click();}""", label)
        time.sleep(8)
        r=read(pg)
        print(f"\nAfter clicking '{label}':  (new cond reqs: {reqs[n:]})")
        for k,v in r.items(): print(f"   {k:<20}: {v!r}")
    b.close()
print("\nconsole msgs:")
for t,m in errs[:8]: print(f"   [{t}] {m[:140]}")

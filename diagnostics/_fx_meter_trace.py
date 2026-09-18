"""Which 3 are still stuck, and does applyLiveLocation's null branch even run?"""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1200})
    logs=[]
    pg.on("console", lambda m: logs.append(m.text))
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(14)
    r=pg.evaluate("""() => {
        const ids=['meter-ts-label','meter-rain-label','meter-flood-label',
                   'bind-temp','bind-rain','bind-humidity','bind-wind','location-obs-age'];
        const o={};
        for(const i of ids){const e=document.getElementById(i); o[i]=e?(e.innerText||'').trim():'(missing)';}
        o._userLoc = JSON.stringify(window.StormSenseUserLocation);
        return o;
    }""")
    for k,v in r.items(): print(f"   {k:<20}: {v!r}")
    print("\n   'Live location unavailable' logged?:",
          any("Live location unavailable" in m for m in logs))
    b.close()

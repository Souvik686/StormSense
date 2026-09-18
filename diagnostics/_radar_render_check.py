"""The fix must not break radar rendering or leak memory via retained tiles."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    fails=[];errs=[]
    pg.on("requestfailed", lambda r: fails.append(r.url))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(5)
    el=pg.query_selector("[data-view='radar']") or pg.query_selector("#nav-radar")
    if el: el.click()
    time.sleep(12)
    st=pg.evaluate("""() => {
        const imgs=[...document.querySelectorAll('img')].filter(i=>(i.getAttribute('src')||'').includes('rainviewer'));
        return {
          tiles: imgs.length,
          loaded: imgs.filter(i=>i.complete && i.naturalWidth>0).length,
          status: (document.getElementById('radar-status')||{}).textContent,
          info: (document.getElementById('radar-info')||{}).textContent,
          update: (document.getElementById('radar-last-update')||{}).textContent
        };
    }""")
    print("radar tiles in DOM      :", st["tiles"])
    print("tiles fully loaded      :", st["loaded"])
    print("radar-status            :", st["status"])
    print("radar-info              :", st["info"])
    print("radar-last-update       :", st["update"])
    # pan + zoom to force tile churn (exercises releaseTile heavily)
    pg.evaluate("""() => { const m=window.stormSenseRadarMap && window.stormSenseRadarMap._gmap; if(m){m.setZoom(8); m.panTo(new google.maps.LatLng(23.5,87.5));} }""")
    time.sleep(8)
    pg.evaluate("""() => { const m=window.stormSenseRadarMap && window.stormSenseRadarMap._gmap; if(m){m.setZoom(6);} }""")
    time.sleep(8)
    after=pg.evaluate("""() => [...document.querySelectorAll('img')].filter(i=>(i.getAttribute('src')||'').includes('rainviewer')).length""")
    pg.screenshot(path="diagnostics/results/shot_radar_after_fix.png")
    b.close()
rv=[f for f in fails if "rainviewer" in f.lower()]
print("tiles after pan/zoom churn:", after)
print("rainviewer failures       :", len(rv))
print("console errors            :", len(errs))

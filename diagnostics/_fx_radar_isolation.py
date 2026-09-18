"""SECTION 6: Does radar contribute ANY pixel to the NOW risk map?
Experiment: capture the nowcasting map with radar forcibly removed, and
compare to the normal capture. Runtime-only; no app code changed."""
from playwright.sync_api import sync_playwright
import time,hashlib
def cap(tag, kill_radar):
    with sync_playwright() as p:
        b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
        rv=[]
        pg.on("request", lambda r: rv.append(r.url) if "rainviewer" in r.url.lower() else None)
        pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
        time.sleep(7)
        pg.evaluate("window.currentLeadHours='now'")
        pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
        time.sleep(3)
        if kill_radar:
            n=pg.evaluate("""() => {
                let removed=0;
                const m=window.stormSenseMap && window.stormSenseMap._gmap;
                if(m && m.overlayMapTypes){
                    // remove EVERY overlay map type (radar tiles live here)
                    removed = m.overlayMapTypes.getLength();
                    m.overlayMapTypes.clear();
                }
                // also strip any rainviewer <img>
                [...document.querySelectorAll('img')].forEach(i=>{if((i.src||'').includes('rainviewer')){i.remove();removed++;}});
                return removed;
            }""")
            print(f"   [{tag}] overlays/imgs removed: {n}")
            time.sleep(3)
        el=pg.query_selector("#map-radar-placeholder")
        el.screenshot(path=f"diagnostics/results/fx_map_{tag}.png")
        info=pg.evaluate("""() => {
            const m=window.stormSenseMap && window.stormSenseMap._gmap;
            const imgs=[...document.querySelectorAll('img')];
            return {
              overlayCount: m && m.overlayMapTypes ? m.overlayMapTypes.getLength() : -1,
              rainviewerImgs: imgs.filter(i=>(i.src||'').includes('rainviewer')).length,
              riskSurfaceImgs: imgs.filter(i=>(i.src||'').includes('risk-surface')).length,
              riskUrls: imgs.filter(i=>(i.src||'').includes('risk-surface')).map(i=>i.src.split('?')[1]).slice(0,3)
            };
        }""")
        b.close()
    rvreq=[u for u in rv if "tilecache" in u]
    print(f"   [{tag}] {info}  rainviewer TILE requests during load: {len(rvreq)}")
    return info
print("A) NORMAL nowcasting map:")
a=cap("normal", False)
print("\nB) Radar/overlays forcibly REMOVED at runtime:")
b_=cap("noradar", True)
import hashlib,os
for t in ["normal","noradar"]:
    p=f"diagnostics/results/fx_map_{t}.png"
    print(f"   {t}: {os.path.getsize(p)}B sha={hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}")

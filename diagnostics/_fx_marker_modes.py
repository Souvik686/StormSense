"""Is the violet location marker present in BOTH modes?"""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1100},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(14)
    def probe(tag):
        r=pg.evaluate("""() => {
            // The Google shim renders L.divIcon as an SVG data-URI <img>,
            // so look for the marker OBJECT the app stores, plus the violet
            // data-URI image actually in the DOM.
            const stored = !!(window.stormSenseMap && window.stormSenseMap._liveLocationMarker);
            const imgs=[...document.querySelectorAll('img')]
                .filter(i=>(i.src||'').startsWith('data:image/svg+xml')
                        && decodeURIComponent(i.src).includes('a855f7')).length;
            return {storedMarker:stored, violetImgs:imgs, mode:window.stormSenseMode,
                    hasLoc: !!window.StormSenseUserLocation};
        }""")
        print(f"  {tag:<12} storedMarker={r['storedMarker']}  violetImgs={r['violetImgs']}  mode={r['mode']}  hasLocation={r['hasLoc']}")
        return r
    pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
    time.sleep(2)
    probe("LIVE")
    pg.screenshot(path="diagnostics/results/fx_marker_live.png", clip={"x":340,"y":200,"width":560,"height":380})
    pg.evaluate("switchMode('historical')"); time.sleep(16)
    pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
    time.sleep(2)
    probe("HISTORICAL")
    pg.screenshot(path="diagnostics/results/fx_marker_hist.png", clip={"x":340,"y":200,"width":560,"height":380})
    pg.evaluate("switchMode('live')"); time.sleep(14)
    probe("BACK->LIVE")
    b.close()
print("console errors:",len(errs))

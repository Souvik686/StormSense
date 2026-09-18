"""Instrument loadRainViewerRadar to see HOW MANY times it runs and from where."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    logs=[]
    pg.on("console", lambda m: logs.append(m.text))
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(5)
    # wrap the function BEFORE opening the radar view
    pg.evaluate("""() => {
        window.__rvCalls=[];
        const orig = window.loadRainViewerRadar;
        if (typeof orig !== 'function') { window.__rvWrapped='NOT GLOBAL'; return; }
        window.loadRainViewerRadar = function(map){
            window.__rvCalls.push(new Error().stack);
            return orig.apply(this, arguments);
        };
        window.__rvWrapped='ok';
    }""")
    print("wrap:", pg.evaluate("window.__rvWrapped"))
    el=pg.query_selector("[data-view='radar']") or pg.query_selector("#nav-radar")
    if el: el.click()
    time.sleep(18)
    n=pg.evaluate("window.__rvCalls ? window.__rvCalls.length : -1")
    print("loadRainViewerRadar call count:", n)
    for i,s in enumerate(pg.evaluate("window.__rvCalls||[]")):
        print(f"\n--- call {i+1} stack ---")
        print("\n".join(s.split("\n")[1:6]))
    # also check how many tileLayers are attached
    print("\nradarLayer identity checks:")
    print(pg.evaluate("""() => {
        const imgs=[...document.querySelectorAll('img')].filter(i=>(i.src||'').includes('rainviewer'));
        return {tilesInDom: imgs.length};
    }"""))
    b.close()

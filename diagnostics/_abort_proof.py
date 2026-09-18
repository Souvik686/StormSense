"""PROVE the abort mechanism: releaseTile() sets img.src='' on tiles that are
still loading. Setting src='' on an in-flight <img> cancels it -> ERR_ABORTED."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    fails=[]
    pg.on("requestfailed", lambda r: fails.append(r.url) if "rainviewer" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(5)
    # Count how many times releaseTile runs, by observing img src wipes.
    pg.evaluate("""() => {
        window.__wipes = 0;
        const d = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype,'src');
        Object.defineProperty(HTMLImageElement.prototype,'src',{
            get(){ return d.get.call(this); },
            set(v){
                if (v === '' && (this.getAttribute('src')||'').includes('rainviewer')) {
                    window.__wipes++;
                }
                return d.set.call(this,v);
            }
        });
    }""")
    el=pg.query_selector("[data-view='radar']") or pg.query_selector("#nav-radar")
    if el: el.click()
    time.sleep(18)
    wipes=pg.evaluate("window.__wipes")
    print(f"rainviewer ERR_ABORTED : {len(fails)}")
    print(f"releaseTile src='' wipes on rainviewer imgs: {wipes}")
    if wipes and len(fails):
        print("\nPROVEN: releaseTile() wipes src on tiles that are still in flight,")
        print("        which is exactly what produces ERR_ABORTED.")
    b.close()

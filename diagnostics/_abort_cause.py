"""Find the REAL cause. Track request lifecycle + what the app does to the
radar layer around the abort time."""
from playwright.sync_api import sync_playwright
import time
events=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    T0=time.time()
    def t(): return round(time.time()-T0,2)
    pg.on("request", lambda r: events.append((t(),"REQ ",r.url)) if "rainviewer" in r.url else None)
    pg.on("response", lambda r: events.append((t(),"RESP",f"{r.status} {r.url}")) if "rainviewer" in r.url else None)
    pg.on("requestfailed", lambda r: events.append((t(),"FAIL",f"{r.failure} {r.url}")) if "rainviewer" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(5)
    el=pg.query_selector("[data-view='radar']") or pg.query_selector("#nav-radar")
    if el: el.click()
    time.sleep(20)
    # How many radar layers exist? Was the layer swapped?
    info=pg.evaluate("""() => {
        const out={};
        out.hasRadarMap = !!window.radarMap;
        // count img tiles referencing rainviewer currently in the DOM
        const imgs=[...document.querySelectorAll('img')].filter(i=>(i.src||'').includes('rainviewer'));
        out.tilesInDom = imgs.length;
        out.distinctFrames = [...new Set(imgs.map(i=>(i.src.match(/radar\/([0-9a-f]+)\//)||[])[1]))];
        return out;
    }""")
    b.close()
print("RAINVIEWER REQUEST LIFECYCLE:")
for tt,kind,txt in events:
    short=txt.replace("https://tilecache.rainviewer.com/v2/radar/","")
    print(f"  t={tt:6.2f}  {kind}  {short[:96]}")
print("\nDOM STATE:", info)
# group by tile id
from collections import defaultdict
g=defaultdict(list)
for tt,kind,txt in events:
    key=txt.split("rainviewer.com")[-1].split(" ")[-1]
    g[key].append((tt,kind))
print("\nPER-URL LIFECYCLE (did any URL get BOTH a response and a failure?):")
for k,v in g.items():
    kinds=[x[1] for x in v]
    if "FAIL" in kinds:
        print(f"  {k[-40:]}: {kinds}")

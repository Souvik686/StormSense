"""Capture the EXACT Google Maps failure reason."""
from playwright.sync_api import sync_playwright
import time
msgs=[]; reqs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("console", lambda m: msgs.append((m.type,m.text)))
    pg.on("response", lambda r: reqs.append((r.status,r.url)) if "googleapis" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(10)
    print("=== GOOGLE MAPS RESPONSES ===")
    for s,u in reqs[:14]:
        print(f"  {s}  {u[:130]}")
    print("\n=== CONSOLE (errors/warnings) ===")
    for t,m in msgs:
        if t in ("error","warning"):
            print(f"  [{t}] {m[:300]}")
    # Does the map div contain Google's error panel?
    st=pg.evaluate("""() => {
        const d=document.getElementById('map-radar-placeholder');
        const txt=d?d.innerText.slice(0,200):'(no div)';
        return {text:txt, hasGmap: !!(window.stormSenseMap && window.stormSenseMap._gmap),
                gmObj: typeof google!=='undefined' && !!google.maps};
    }""")
    print("\n=== MAP DIV STATE ===")
    for k,v in st.items(): print(f"  {k}: {v}")
    b.close()

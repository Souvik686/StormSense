from playwright.sync_api import sync_playwright
import time
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
    time.sleep(2)
    zm=lambda: pg.evaluate("()=>window.stormSenseMap._gmap.getZoom()")
    print("start zoom:",zm())
    for label in ["Zoom In","Zoom In","Zoom Out"]:
        ok=pg.evaluate("""(lb)=>{const b=[...document.querySelectorAll('button')]
            .find(x=>((x.title||x.ariaLabel||x.textContent||'').trim()===lb));
            if(b){b.click();return true;} return false;}""", label)
        time.sleep(2.5)
        print(f"  clicked '{label}': found={ok} zoom={zm()}")
    # WB Focus / toggle view
    ok=pg.evaluate("""()=>{const b=[...document.querySelectorAll('button')]
        .find(x=>/Toggle View/i.test((x.title||x.ariaLabel||x.textContent||'')));
        if(b){b.click();return (b.title||b.textContent||'').trim().slice(0,40);} return null;}""")
    time.sleep(3)
    print("  toggle view clicked:",ok,"-> zoom",zm(),
          pg.evaluate("()=>{const c=window.stormSenseMap._gmap.getCenter();return[+c.lat().toFixed(3),+c.lng().toFixed(3)];}"))
    b.close()
print("console errors:",len(errs))

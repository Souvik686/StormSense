"""Is the live-location button visible/clickable in both modes?"""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(13)
    def probe():
        return pg.evaluate("""() => {
            const b=document.getElementById('btn-my-location');
            if(!b) return {exists:false};
            const r=b.getBoundingClientRect();
            const cs=getComputedStyle(b);
            return {exists:true, w:Math.round(r.width), h:Math.round(r.height),
                    x:Math.round(r.x), y:Math.round(r.y),
                    display:cs.display, visibility:cs.visibility, opacity:cs.opacity,
                    hasHandler: typeof window.handleMyLocationClick,
                    parentHidden: b.offsetParent===null,
                    mode: window.stormSenseMode};
        }""")
    print("LIVE      :", probe())
    pg.evaluate("switchMode('historical')"); time.sleep(14)
    print("HISTORICAL:", probe())
    # try clicking it in historical
    pg.evaluate("()=>{const b=document.getElementById('btn-my-location'); if(b)b.click();}")
    time.sleep(8)
    print("after click (historical):", pg.evaluate("""() => ({
        district:(document.getElementById('profile-district')||{}).innerText,
        coords:(document.getElementById('profile-coords')||{}).innerText,
        ts:(document.getElementById('meter-ts-label')||{}).innerText
    })"""))
    b.close()
print("console errors:",len(errs))
for e in errs[:5]: print("  ",e[:160])

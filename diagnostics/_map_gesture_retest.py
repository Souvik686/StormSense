"""Retest drag/wheel with the map ACTUALLY in the viewport."""
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
    r=pg.evaluate("""()=>{const d=document.getElementById('map-radar-placeholder').getBoundingClientRect();
        const top=document.elementFromPoint(d.x+d.width/2,d.y+d.height/2);
        return {x:d.x,y:d.y,w:d.width,h:d.height,top:top?top.tagName+'.'+String(top.className).slice(0,40):null};}""")
    print("map rect after scroll:",r)
    cx,cy=r["x"]+r["w"]/2, r["y"]+r["h"]/2
    ctr=lambda: pg.evaluate("()=>{const c=window.stormSenseMap._gmap.getCenter();return[+c.lat().toFixed(4),+c.lng().toFixed(4)];}")
    zm =lambda: pg.evaluate("()=>window.stormSenseMap._gmap.getZoom()")
    b0,z0=ctr(),zm(); print("before:",b0,"zoom",z0)
    pg.mouse.move(cx,cy); pg.mouse.down()
    for i in range(1,15): pg.mouse.move(cx-14*i, cy-9*i); time.sleep(0.03)
    pg.mouse.up(); time.sleep(4)
    b1=ctr(); print("after DRAG:",b1,"changed:",b1!=b0)
    pg.mouse.move(cx,cy); pg.mouse.wheel(0,-500); time.sleep(4)
    z1=zm(); print("after WHEEL:",ctr(),"zoom",z1,"changed:",z1!=z0)
    # zoom buttons in the UI
    btn=pg.query_selector("button[title*='Zoom in'], button[aria-label*='Zoom in']")
    if btn:
        btn.click(); time.sleep(3); print("after ZOOM BUTTON:",zm())
    else:
        print("custom zoom controls: querying app buttons")
        z2=pg.evaluate("""()=>{const bs=[...document.querySelectorAll('button')].filter(b=>/zoom|\+|\-/i.test(b.title+b.ariaLabel+b.textContent));
            return bs.slice(0,6).map(b=>(b.title||b.ariaLabel||b.textContent||'').trim().slice(0,22));}""")
        print("  candidate zoom buttons:",z2)
    b.close()
print("console errors:",len(errs))

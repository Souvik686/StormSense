"""Is drag/wheel actually disabled, or did synthetic input miss the map?
Check Google Map option flags directly -- that is authoritative."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    info=pg.evaluate("""() => {
        const L=window.stormSenseMap, m=L && L._gmap;
        if(!m) return {err:'no map'};
        const d=m.getDiv();
        const r=d.getBoundingClientRect();
        // what element is actually on top at the map centre?
        const top=document.elementFromPoint(r.x+r.width/2, r.y+r.height/2);
        return {
            draggable: m.get('draggable'),
            scrollwheel: m.get('scrollwheel'),
            gestureHandling: m.get('gestureHandling'),
            disableDoubleClickZoom: m.get('disableDoubleClickZoom'),
            zoomControl: m.get('zoomControl'),
            rect: {x:r.x,y:r.y,w:r.width,h:r.height},
            topElAtCenter: top ? (top.tagName+'#'+(top.id||'')+'.'+String(top.className).slice(0,60)) : null,
            mapDivId: d.id
        };
    }""")
    for k,v in info.items(): print(f"  {k}: {v}")
    # try dragging using Google's own event pipeline on the correct target
    before=pg.evaluate("()=>{const c=window.stormSenseMap._gmap.getCenter();return[+c.lat().toFixed(4),+c.lng().toFixed(4)];}")
    r=info["rect"]; cx,cy=r["x"]+r["w"]/2, r["y"]+r["h"]/2
    pg.mouse.move(cx,cy); pg.mouse.down()
    for i in range(1,13): pg.mouse.move(cx-12*i, cy-8*i); time.sleep(0.03)
    pg.mouse.up(); time.sleep(4)
    after=pg.evaluate("()=>{const c=window.stormSenseMap._gmap.getCenter();return[+c.lat().toFixed(4),+c.lng().toFixed(4)];}")
    print(f"\n  drag: before={before} after={after} changed={before!=after}")
    b.close()

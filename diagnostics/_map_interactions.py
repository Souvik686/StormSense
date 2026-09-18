"""Map interaction audit: drag, wheel zoom, zoom controls, popup at multiple
locations, district dropdown. Verifies values CHANGE with location."""
from playwright.sync_api import sync_playwright
import time,re
errs=[];fails=[]
def popup(pg):
    return pg.evaluate("""() => {const e=document.querySelector('.leaflet-popup-content, .gm-style-iw'); return e?e.innerText:'';}""")
def click_at(pg,la,lo):
    pg.evaluate("() => { try{window.stormSenseMap.closePopup();}catch(e){} }")
    time.sleep(0.8)
    pg.evaluate("""([a,o])=>{const m=window.stormSenseMap._gmap;
        google.maps.event.trigger(m,'click',{latLng:new google.maps.LatLng(a,o)});}""",[la,lo])
    for _ in range(20):
        time.sleep(1); t=popup(pg)
        if t and "Querying" not in t: return t
    return popup(pg)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.on("requestfailed", lambda r: fails.append(r.url))
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    g=lambda: pg.evaluate("()=>{const m=window.stormSenseMap._gmap;const c=m.getCenter();return{z:m.getZoom(),lat:+c.lat().toFixed(3),lng:+c.lng().toFixed(3)};}")
    start=g(); print("initial:",start)
    # zoom in / out programmatically (zoom controls path)
    pg.evaluate("()=>window.stormSenseMap._gmap.setZoom(9)"); time.sleep(3); z9=g()
    pg.evaluate("()=>window.stormSenseMap._gmap.setZoom(6)"); time.sleep(3); z6=g()
    print("after setZoom(9):",z9," after setZoom(6):",z6)
    # pan
    pg.evaluate("()=>window.stormSenseMap._gmap.panTo(new google.maps.LatLng(26.5,88.5))"); time.sleep(3)
    panned=g(); print("after panTo(26.5,88.5):",panned)
    # real mouse drag over the map
    box=pg.evaluate("""()=>{const d=window.stormSenseMap._gmap.getDiv().getBoundingClientRect();return{x:d.x,y:d.y,w:d.width,h:d.height};}""")
    cx,cy=box["x"]+box["w"]/2, box["y"]+box["h"]/2
    pg.mouse.move(cx,cy); pg.mouse.down(); pg.mouse.move(cx-140,cy-90,steps=12); pg.mouse.up()
    time.sleep(3); dragged=g(); print("after mouse drag:",dragged)
    # wheel zoom
    pg.mouse.move(cx,cy); pg.mouse.wheel(0,-400); time.sleep(3); wheeled=g()
    print("after wheel zoom:",wheeled)
    print("\nDRAG changed center:", dragged!=panned, "| WHEEL changed zoom:", wheeled["z"]!=dragged["z"])
    # popups at distinct locations
    pg.evaluate("window.currentLeadHours='now'")
    print("\npopup values at different locations:")
    seen={}
    for nm,(la,lo) in {"Kolkata":(22.5726,88.3639),"Darjeeling":(27.0410,88.2663),
                       "Purulia":(23.3322,86.3616),"Digha":(21.6270,87.5090)}.items():
        t=click_at(pg,la,lo)
        d=re.search(r"^(.+?)\n",t.split("Grid Cell")[0].split("LIVE FORECAST")[0].strip().split("\n",2)[-1]) if t else None
        cell=re.search(r"Grid Cell:\s*([\d.]+)°N,\s*([\d.]+)°E",t)
        prob=re.search(r"Thunderstorm Risk:\s*\n?([\d.]+)%",t)
        dist=re.search(r"\n([A-Za-z0-9 ()]+)\nLIVE FORECAST",t)
        print(f"  {nm:<11} cell={cell.group(0) if cell else '?':<34} prob={prob.group(1)+'%' if prob else '?':<8} district={dist.group(1).strip() if dist else '?'}")
        seen[nm]=(cell.group(0) if cell else None, prob.group(1) if prob else None)
    cells={v[0] for v in seen.values()}; probs={v[1] for v in seen.values()}
    print(f"\ndistinct grid cells: {len(cells)}/4  distinct probs: {len(probs)}/4")
    # district dropdown
    sel=pg.query_selector("select")
    if sel:
        opts=pg.evaluate("()=>[...document.querySelector('select').options].map(o=>o.text).slice(0,5)")
        pg.select_option("select", index=2); time.sleep(4)
        print("\ndropdown options(first5):",opts)
        print("after select index2, map center:",g())
    b.close()
print("\nconsole errors:",len(errs),"| failed requests:",len(fails))

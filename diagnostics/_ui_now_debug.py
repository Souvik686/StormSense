from playwright.sync_api import sync_playwright
import time,json
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
    reqs=[]
    pg.on("request", lambda r: reqs.append(r.url) if "nowcast/point" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(6)
    pg.evaluate("window.currentLeadHours = 'now'")
    print("currentLeadHours =", pg.evaluate("window.currentLeadHours"))
    print("apiLeadHours()   =", pg.evaluate("typeof apiLeadHours==='function' ? apiLeadHours() : 'not global'"))
    reqs.clear()
    pg.evaluate("""() => {
        const m=window.stormSenseMap && window.stormSenseMap._gmap;
        google.maps.event.trigger(m,'click',{latLng:new google.maps.LatLng(22.5726,88.3639)});
    }""")
    time.sleep(8)
    print("point requests fired:", reqs)
    t=pg.evaluate("""() => {const el=document.querySelector('.leaflet-popup-content, .gm-style-iw'); return el?el.innerText:'(none)';}""")
    print("--- popup ---"); print(t[:700])
    b.close()

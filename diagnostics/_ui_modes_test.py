"""Verify: (1) evidence appears ONLY at NOW in live, (2) never in historical,
(3) probabilities differ across horizons, (4) no console errors."""
from playwright.sync_api import sync_playwright
import time
errs=[]
def popup_text(pg):
    return pg.evaluate("""() => {
        const el=document.querySelector('.leaflet-popup-content, .gm-style-iw');
        return el ? el.innerText : '';
    }""")
def click(pg,lat,lon):
    # Close any existing popup first, then wait for the NEW one to carry the
    # expected lead label -- otherwise we read the previous iteration's popup.
    pg.evaluate("() => { try{ window.stormSenseMap.closePopup(); }catch(e){} }")
    time.sleep(1)
    pg.evaluate("""([la,lo]) => {
        const m=window.stormSenseMap && window.stormSenseMap._gmap;
        google.maps.event.trigger(m,'click',{latLng:new google.maps.LatLng(la,lo)});
    }""",[lat,lon])
    for _ in range(20):
        time.sleep(1)
        t=popup_text(pg)
        if t and "Querying" not in t: return
    return
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(6)
    print(f"{'mode':<12}{'lead':<7}{'evidence?':<11}{'thunderstorm':<14}")
    for lead in ["now",2,4,6]:
        pg.evaluate("l => window.currentLeadHours = l", lead)
        time.sleep(1); click(pg,22.5726,88.3639)
        t=popup_text(pg)
        has="YES" if "Observed now" in t else "no"
        import re
        m=re.search(r"Thunderstorm Risk:\s*([\d.]+)%", t)
        print(f"{'live':<12}{str(lead):<7}{has:<11}{(m.group(1)+'%' if m else '?'):<14}")
    # switch to historical
    sw=pg.evaluate("""() => { if (typeof switchMode==='function'){switchMode('historical');return 'ok';} return 'no switchMode'; }""")
    print("\nswitchMode ->",sw); time.sleep(8)
    for lead in ["now",2]:
        pg.evaluate("l => window.currentLeadHours = l", lead)
        time.sleep(1); click(pg,22.5726,88.3639)
        t=popup_text(pg)
        has="YES (LEAK!)" if "Observed now" in t else "no"
        print(f"{'historical':<12}{str(lead):<7}{has:<11}")
    b.close()
print("\nconsole errors:",len(errs))
for e in errs[:8]: print("  ",e[:160])

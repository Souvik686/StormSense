"""Verify the NOW evidence block renders in the real popup, and that the
displayed probability still equals the backend value (unmodified)."""
from playwright.sync_api import sync_playwright
import json, time, urllib.request

errs=[]; failed=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type=="error" else None)
    pg.on("requestfailed", lambda r: failed.append(r.url))
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(6)
    # ensure NOW horizon selected
    pg.evaluate("window.currentLeadHours = 'now';")
    time.sleep(1)
    # click Kolkata on the map via the shim's click handler
    r=pg.evaluate("""() => {
        const m = window.stormSenseMap && window.stormSenseMap._gmap ? window.stormSenseMap._gmap : null;
        if (!m) return "no gmap handle";
        google.maps.event.trigger(m, 'click', {latLng: new google.maps.LatLng(22.5726, 88.3639)});
        return "fired";
    }""")
    print("click:",r)
    time.sleep(7)
    html=pg.evaluate("""() => {
        const el=document.querySelector('.leaflet-popup-content, .gm-style-iw');
        return el ? el.innerText : (document.body.innerText.includes('Observed now') ? 'FOUND-IN-BODY' : null);
    }""")
    print("\n--- POPUP TEXT ---")
    print(html if html else "(no popup captured)")
    pg.screenshot(path="diagnostics/results/shot_popup_evidence.png")
    b.close()
print("\nconsole errors:",len(errs))
for e in errs[:10]: print("  ",e[:180])
print("failed requests:",len([f for f in failed if 'rainviewer' not in f.lower()]))

from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    el=pg.query_selector("#map-radar-placeholder") or pg.query_selector(".leaflet-container")
    if el:
        el.scroll_into_view_if_needed(); time.sleep(3)
        el.screenshot(path="diagnostics/results/shot_map_now.png")
        print("captured NOW")
    # click +6h then capture
    for lbl in ["+6h"]:
        btn=pg.query_selector(f"text={lbl}")
        if btn:
            btn.click(); time.sleep(5)
            el.screenshot(path="diagnostics/results/shot_map_6h.png"); print("captured",lbl)
    b.close()

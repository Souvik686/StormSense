"""What do the 4 hazard cards actually display now?"""
from playwright.sync_api import sync_playwright
import time,re
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(8)
    t=pg.evaluate("() => document.body.innerText")
    i=t.find("ATMOSPHERIC HAZARD")
    print(t[i:i+700] if i>=0 else "(cards not found)")
    pg.screenshot(path="diagnostics/results/fx_cards_now.png", clip={"x":0,"y":140,"width":1560,"height":360})
    b.close()

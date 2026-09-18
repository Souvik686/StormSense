"""Does the 'Current observed conditions' strip change with mode?"""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page()
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(14)
    def snap():
        return pg.evaluate("""() => ({
            mode: window.stormSenseMode,
            title: (document.getElementById('nowcast-timeline-title')||{}).innerText,
            badge: (document.getElementById('nowcast-timeline-badge')||{}).innerText,
            strip: (document.getElementById('nowcast-steps')||{innerText:''}).innerText.replace(/\s+/g,' ').trim().slice(0,170)
        })""")
    r=snap(); print(f"LIVE      : [{r['title']} / {r['badge']}]\n            {r['strip']}\n")
    pg.evaluate("switchMode('historical')"); time.sleep(15)
    r=snap(); print(f"HISTORICAL: [{r['title']} / {r['badge']}]\n            {r['strip']}\n")
    b.close()

"""Is the first area change stale, or just slow? Re-test with longer settle
and by revisiting Bankura after other areas."""
from playwright.sync_api import sync_playwright
import time
def obs(pg):
    return pg.evaluate("""() => {
        const t=document.getElementById('nowcast-steps');
        return t?t.innerText.replace(/\s+/g,' ').trim().slice(0,120):'';
    }""")
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page()
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(12)
    seq=["whole-state","bankura","cooch-behar","bankura","whole-state","bankura"]
    for v in seq:
        pg.select_option("#sector-selector", v)
        time.sleep(14)          # generous settle
        print(f"  {v:<14} -> {obs(pg)[:110]}")
    b.close()

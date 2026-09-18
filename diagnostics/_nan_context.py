from playwright.sync_api import sync_playwright
import re,time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    for v in ["dashboard","xai"]:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el: el.click()
        time.sleep(4)
        t=pg.evaluate("() => document.body.innerText")
        for m in re.finditer(r"nan", t, re.I):
            a=max(0,m.start()-70); z=min(len(t),m.end()+70)
            print(f"[{v}] ...{t[a:z]!r}...")
        print()
    b.close()

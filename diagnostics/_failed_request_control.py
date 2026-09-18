"""CONTROL EXPERIMENT: if the 9 aborts are caused by navigating AWAY from the
Radar view mid-load, then staying on Radar must produce ZERO aborts."""
from playwright.sync_api import sync_playwright
import time

def run(label, navigate_away):
    fails=[]; cerrs=[]
    with sync_playwright() as p:
        b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
        pg.on("requestfailed", lambda r: fails.append((r.url, str(r.failure))))
        pg.on("console", lambda m: cerrs.append(m.text) if m.type=="error" else None)
        pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
        time.sleep(5)
        el=pg.query_selector("[data-view='radar']") or pg.query_selector("#nav-radar")
        if el: el.click()
        if navigate_away:
            time.sleep(2.5)                       # cut tiles off mid-flight
            el2=pg.query_selector("[data-view='advisories']") or pg.query_selector("#nav-advisories")
            if el2: el2.click()
            time.sleep(6)
        else:
            time.sleep(25)                        # let every tile finish
        b.close()
    rv=[f for f in fails if "rainviewer" in f[0].lower()]
    other=[f for f in fails if "rainviewer" not in f[0].lower()]
    print(f"{label:<42} rainviewer_aborts={len(rv):<3} other_failures={len(other):<3} console_errors={len(cerrs)}")
    for u,f in other[:5]: print("     OTHER FAILURE:",u[:110],f)
    return len(rv), len(other), len(cerrs)

print("CONTROL EXPERIMENT — same page, only the navigation behaviour differs\n")
a=run("A: navigate AWAY from Radar mid-load", True)
b_=run("B: STAY on Radar until tiles finish", False)
print()
if b_[0]==0 and a[0]>0:
    print("PROVEN: aborts occur ONLY when navigating away mid-load.")
    print("        Staying on the view -> 0 aborts. Cause = client-side cancellation.")
elif b_[0]>0:
    print("NOT PROVEN: aborts persist even without navigation -> investigate as GENUINE.")

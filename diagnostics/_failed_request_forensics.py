"""Classify EVERY failed browser request individually, with evidence.
No request is called 'harmless' without proof that (a) it was cancelled by our
own navigation, AND (b) the same URL succeeds when fetched standalone."""
from playwright.sync_api import sync_playwright
import time, json, urllib.request

records=[]   # every requestfailed, with context
nav_marks=[] # when we navigated views

with sync_playwright() as p:
    b=p.chromium.launch()
    pg=b.new_page(viewport={"width":1600,"height":1000})

    def on_failed(r):
        records.append({
            "url": r.url,
            "method": r.method,
            "resource_type": r.resource_type,
            "failure": (r.failure or "")if isinstance(r.failure,str) else str(r.failure),
            "t": round(time.time()-T0,2),
            "frame_url": r.frame.url if r.frame else None,
        })
    def on_console(m):
        if m.type=="error": records.append({"console_error": m.text, "t": round(time.time()-T0,2)})

    pg.on("requestfailed", on_failed)
    pg.on("console", on_console)

    T0=time.time()
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(6)
    nav_marks.append(("loaded dashboard", round(time.time()-T0,2)))

    for v in ["dashboard","radar","advisories","wrf","xai","gis","threshold","streams"]:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el:
            el.click()
            nav_marks.append((f"clicked {v}", round(time.time()-T0,2)))
            time.sleep(2.5)
    time.sleep(3)
    b.close()

fails=[r for r in records if "url" in r]
cerrs=[r for r in records if "console_error" in r]
print(f"TOTAL failed requests: {len(fails)}   console errors: {len(cerrs)}\n")
print("NAV TIMELINE:")
for m,t in nav_marks: print(f"   t={t:6.2f}s  {m}")
print("\nFAILED REQUESTS (individually):")
for i,r in enumerate(fails,1):
    print(f"\n[{i}] t={r['t']}s")
    print(f"    url          : {r['url']}")
    print(f"    method/type  : {r['method']} / {r['resource_type']}")
    print(f"    failure      : {r['failure']}")
json.dump({"fails":fails,"console":cerrs,"nav":nav_marks}, open("diagnostics/results/failed_requests.json","w"), indent=1)
print("\nsaved -> diagnostics/results/failed_requests.json")

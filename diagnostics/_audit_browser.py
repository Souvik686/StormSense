from playwright.sync_api import sync_playwright
import time, json
errs=[]; failed=[]; reqs=[]
with sync_playwright() as p:
    b=p.chromium.launch()
    pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error",) else None)
    pg.on("requestfailed", lambda r: failed.append(f"{r.url} :: {r.failure}"))
    pg.on("response", lambda r: reqs.append((r.status, r.url)) if "/api/" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(6)
    pg.screenshot(path="diagnostics/results/shot_dashboard.png", full_page=False)
    # exercise each dashboard view
    views=["dashboard","radar","advisories","wrf","xai","gis","threshold","streams"]
    for v in views:
        try:
            el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
            if el: el.click(); time.sleep(2.5)
            pg.screenshot(path=f"diagnostics/results/shot_{v}.png")
        except Exception as e: errs.append(f"view {v}: {e}")
    b.close()
print("API responses:")
bad=[r for r in reqs if r[0]>=400]
for s,u in sorted(set(reqs))[:40]: print(f"  {s} {u.split('127.0.0.1:8000')[-1][:95]}")
print(f"\nTotal API calls: {len(reqs)}  | >=400: {len(bad)}")
for s,u in bad: print("  BAD:",s,u)
print(f"\nConsole errors: {len(errs)}")
for e in errs[:20]: print("  ",e[:200])
print(f"\nFailed requests: {len(failed)}")
for f in failed[:20]: print("  ",f[:200])

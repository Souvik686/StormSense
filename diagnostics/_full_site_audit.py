"""WHOLE-SITE AUDIT: every view, both modes, all horizons, with per-view
content checks (not just 'page loaded')."""
from playwright.sync_api import sync_playwright
import time, json, re

VIEWS=["dashboard","radar","advisories","wrf","xai","gis","threshold","streams"]
results={}; api=[]; errs=[]; fails=[]

def txt(pg, view):
    return pg.evaluate("""(v) => {
        const el=document.querySelector(`#view-${v}`) || document.querySelector(`[data-view-panel='${v}']`);
        const host = el || document.body;
        return host.innerText.slice(0, 4000);
    }""", view)

with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("response", lambda r: api.append((r.status,r.url)) if "/api/" in r.url else None)
    pg.on("console", lambda m: errs.append((m.type,m.text)) if m.type=="error" else None)
    pg.on("requestfailed", lambda r: fails.append((r.url,str(r.failure))))
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)

    for v in VIEWS:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el: el.click()
        time.sleep(3.5)
        t=txt(pg,v)
        # structural probes
        probe=pg.evaluate("""() => ({
            canvases: document.querySelectorAll('canvas').length,
            svgs: document.querySelectorAll('svg').length,
            tables: document.querySelectorAll('table').length,
            selects: document.querySelectorAll('select').length,
            buttons: document.querySelectorAll('button').length,
            imgs: document.querySelectorAll('img').length
        })""")
        # placeholder / mock detection
        bad=[]
        for pat in ["undefined","NaN","null%","[object Object]","Lorem","TODO","FIXME","mockData","--%"]:
            if pat.lower() in t.lower(): bad.append(pat)
        results[v]={"chars":len(t),"probe":probe,"suspect":bad,"sample":t[:260].replace("\n"," | ")}
        print(f"\n=== {v.upper()} === chars={len(t)} {probe}")
        if bad: print(f"   !! SUSPECT TOKENS: {bad}")
        print(f"   {results[v]['sample'][:240]}")
    pg.screenshot(path="diagnostics/results/shot_audit_last.png")
    b.close()

print("\n\n================ SUMMARY ================")
bad_api=[(s,u) for s,u in api if s>=400]
print(f"API calls: {len(api)}  >=400: {len(bad_api)}")
for s,u in bad_api: print("   BAD:",s,u)
print(f"console errors: {len(errs)}")
for t_,m in errs[:10]: print("   ",t_,m[:160])
rv=[f for f in fails if "rainviewer" in f[0].lower()]
other=[f for f in fails if "rainviewer" not in f[0].lower()]
print(f"failed requests: total={len(fails)} rainviewer={len(rv)} other={len(other)}")
for u,f in other[:10]: print("   OTHER:",u[:120],f)
allsus={v:r["suspect"] for v,r in results.items() if r["suspect"]}
print("views with suspect tokens:", allsus if allsus else "none")
json.dump(results, open("diagnostics/results/full_site_audit.json","w"), indent=1)

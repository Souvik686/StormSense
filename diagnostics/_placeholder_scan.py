"""Word-boundary scan for REAL placeholder/mock artifacts across all 8 views."""
from playwright.sync_api import sync_playwright
import re,time
PATS=[r"\bNaN\b",r"\bundefined\b",r"\bnull\b",r"\[object Object\]",r"\bLorem\b",
      r"\bTODO\b",r"\bFIXME\b",r"\bmock\b",r"\bdummy\b",r"\bplaceholder\b",
      r"\bsample data\b",r"\bfake\b",r"--%",r"\bInfinity\b"]
VIEWS=["dashboard","radar","advisories","wrf","xai","gis","threshold","streams"]
tot=0
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    for v in VIEWS:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el: el.click()
        time.sleep(3)
        t=pg.evaluate("() => document.body.innerText")
        hits=[]
        for pat in PATS:
            for m in re.finditer(pat,t,re.I):
                a=max(0,m.start()-55); z=min(len(t),m.end()+55)
                hits.append((pat, t[a:z].replace("\n"," ")))
        tot+=len(hits)
        print(f"{v:<12} {'CLEAN' if not hits else str(len(hits))+' HIT(S)'}")
        for pat,ctx in hits[:6]: print(f"     {pat}: ...{ctx}...")
    b.close()
print(f"\nTOTAL genuine placeholder hits across 8 views: {tot}")

"""Does the NOW button actually repaint the surface with lead=0?"""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
    reqs=[]
    pg.on("request", lambda r: reqs.append(r.url.split("/api/")[-1]) if "risk-surface" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
    time.sleep(2)
    print("initial surface requests:", reqs[-2:])
    def srcs():
        return pg.evaluate("""()=>[...document.querySelectorAll('img')].filter(i=>(i.src||'').includes('risk-surface')).map(i=>i.src.split('?')[1])""")
    print("initial overlay src:", srcs())
    for label in ["NOW","+2h","+4h","+6h"]:
        n=len(reqs)
        ok=pg.evaluate("""(lb)=>{const b=[...document.querySelectorAll('button')]
            .find(x=>((x.textContent||'').trim()===lb));
            if(b){b.click();return true;} return false;}""", label)
        time.sleep(5)
        new=reqs[n:]
        print(f"  click '{label}': found={ok}  new surface reqs={new}  overlay_src={srcs()}")
    b.close()

"""Are the '...' placeholders a transient loading state, or permanently stuck?
Sample the same elements repeatedly over 40s."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1200})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    def snap():
        return pg.evaluate("""() => {
            const out=[];
            const els=[...document.querySelectorAll('div,span,p,strong,b')];
            for(const e of els){
                if(e.children.length>0) continue;
                const r=e.getBoundingClientRect();
                if(r.width===0||r.height===0) continue;
                const t=(e.innerText||'').trim();
                if(t==='...'||t==='\u2014'){
                    const par=e.parentElement;
                    out.push({t:t,id:e.id||'',cls:String(e.className).slice(0,48),
                              ctx:(par?(par.innerText||''):'').replace(/\s+/g,' ').trim().slice(0,70)});
                }
            }
            return out;
        }""")
    for s in [5,12,20,30,45]:
        while time.time()-t0 < s if (t0:=globals().setdefault('t0',time.time())) else False: pass
        time.sleep(0)  # no-op
    t0=time.time()
    for target in [6,15,25,40]:
        while time.time()-t0 < target: time.sleep(1)
        cur=snap()
        print(f"\n--- t={target}s : {len(cur)} placeholder element(s) ---")
        for c in cur: print(f"    '{c['t']}' id={c['id']!r} cls={c['cls']!r} ctx={c['ctx']!r}")
    b.close()

"""Find the exact elements displaying NaN."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(7)
    for v in ["dashboard","xai"]:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el: el.click()
        time.sleep(4)
        hits=pg.evaluate("""() => {
            const out=[];
            const w=document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let n;
            while(n=w.nextNode()){
                const t=(n.nodeValue||'').trim();
                if(!t || !/NaN/.test(t)) continue;
                const e=n.parentElement;
                if(!e) continue;
                const r=e.getBoundingClientRect();
                out.push({text:t.slice(0,90), id:e.id||'', cls:(e.className||'').toString().slice(0,70),
                          tag:e.tagName, visible:!!(r.width&&r.height), parentId:(e.parentElement||{}).id||''});
            }
            return out;
        }""")
        print(f"\n=== {v.upper()} : {len(hits)} NaN text nodes ===")
        for h in hits: print("  ",h)
    b.close()

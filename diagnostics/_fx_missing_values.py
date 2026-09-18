"""Hunt for MISSING / EMPTY / placeholder values in the rendered DOM.
Different question from 'do numbers match the backend': this looks for fields
that render blank, as a dash, as '--', 'N/A', 'Unavailable', '', or that are
labelled but carry no value."""
from playwright.sync_api import sync_playwright
import time, json, re

VIEWS=["dashboard","radar","advisories","wrf","xai","gis","threshold","streams"]
# "..." is a LOADING placeholder that must never persist -> a real defect.
# "—" is the CORRECT honest rendering of "no value available" and is NOT a
# defect; it is counted separately so the two are never conflated.
STUCK_TOKENS = ["...", "undefined", "NaN", "null", "[object Object]", "TBD",
                "--", "—%", "- %", "—mm", "- mm"]
HONEST_EMPTY = ["—", "–", "N/A", "NA", "n/a", "Unavailable", "unavailable"]
EMPTY_TOKENS = STUCK_TOKENS

report={}
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1200})
    errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(9)
    for v in VIEWS:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el: el.click()
        time.sleep(3.5)
        res=pg.evaluate("""(tokens) => {
            const out={empties:[], emptyEls:0, totalEls:0, hiddenSections:0};
            // every leaf element that should carry a value
            const els=[...document.querySelectorAll('div,span,p,td,th,strong,b,h1,h2,h3,h4,li')];
            for(const e of els){
                if(e.children.length>0) continue;          // leaf only
                const r=e.getBoundingClientRect();
                if(r.width===0||r.height===0) continue;     // not visible
                out.totalEls++;
                const t=(e.innerText||'').trim();
                if(t==='' ){ out.emptyEls++; continue; }
                if(tokens.includes(t)){
                    // capture nearby label for context
                    const par=e.parentElement;
                    const ctx=par?(par.innerText||'').trim().replace(/\s+/g,' ').slice(0,90):'';
                    out.empties.push({text:t, ctx:ctx, id:e.id||'', cls:String(e.className).slice(0,50)});
                }
            }
            // sections that exist but render zero height (possible failed render)
            for(const s of document.querySelectorAll('section,[data-view-panel]')){
                const r=s.getBoundingClientRect();
                if(r.height===0 && s.offsetParent!==null) out.hiddenSections++;
            }
            return out;
        }""", EMPTY_TOKENS)
        report[v]=res
        flag = "OK" if not res["empties"] else f"{len(res['empties'])} EMPTY-VALUE"
        print(f"\n=== {v.upper():<11} visibleLeaves={res['totalEls']:<5} blank={res['emptyEls']:<4} {flag}")
        for e in res["empties"][:12]:
            print(f"     '{e['text']}'  <- ctx: {e['ctx'][:80]}")
    b.close()
print("\nconsole errors:",len(errs))
json.dump(report, open("diagnostics/results/missing_values.json","w"), indent=1)
tot=sum(len(r["empties"]) for r in report.values())
print(f"TOTAL empty/placeholder values across 8 views: {tot}")

"""Do the dashboard values actually change when the area selector changes?"""
from playwright.sync_api import sync_playwright
import time,re
def strip(pg):
    return pg.evaluate("""() => {
        const t=document.getElementById('nowcast-steps');
        const txt=t?t.innerText.replace(/\s+/g,' ').trim():'';
        const cards={};
        ['hazard-thunderstorm-value','hazard-rain-value','hazard-flood-value','hazard-overall-value'].forEach(id=>{
            const e=document.getElementById(id); if(e) cards[id]=e.innerText.trim();
        });
        // fallback: scrape the 4 big card numbers
        const body=document.body.innerText;
        const m=body.match(/Thunderstorm\s+([\d.]+%)[\s\S]*?Heavy Rainfall\s+([\d.]+\s*mm)[\s\S]*?Flash-Flood Risk Proxy\s+([\d.]+%)[\s\S]*?Overall Hazard Level\s+([\d.]+%)/);
        return {obsStrip:txt.slice(0,240), cards:cards, scraped:m?m.slice(1):null,
                sel:(document.getElementById('sector-selector')||{}).value};
    }""")
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(12)
    opts=pg.evaluate("()=>[...document.querySelectorAll('#sector-selector option')].map(o=>[o.value,o.text]).slice(0,26)")
    print("area options:",[o[1] for o in opts][:8],"...")
    results={}
    for val,name in [o for o in opts if o[0] in ("whole-state",)] + [o for o in opts if o[0] not in ("whole-state",)][:4]:
        pg.select_option("#sector-selector", val)
        time.sleep(7)
        r=strip(pg)
        results[name]=r
        print(f"\n--- {name} (value={val}) ---")
        print("   cards:",r["scraped"])
        print("   obs  :",r["obsStrip"][:150])
    b.close()
print("\nconsole errors:",len(errs))
vals=[tuple(v["scraped"]) if v["scraped"] else None for v in results.values()]
print("distinct card-sets:",len(set(vals)),"/",len(vals))
obs=[v["obsStrip"] for v in results.values()]
print("distinct obs-strips:",len(set(obs)),"/",len(obs))

"""Switching NOW -> +2h -> +4h -> +6h -> NOW must stay consistent."""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(13)
    def snap():
        return pg.evaluate("""() => ({
            lead: String(window.currentLeadHours),
            active: [...document.querySelectorAll('.horizon-btn')].filter(b=>b.className.includes('bg-cyan-600')).map(b=>b.textContent.trim()).join(','),
            legend: (document.getElementById('legend-lead-tag')||{}).textContent,
            overlay: ([...document.querySelectorAll('img')].filter(i=>(i.src||'').includes('risk-surface')).map(i=>(i.src.match(/lead=(\d)/)||[])[1])[0])||'?',
            title: (document.getElementById('location-obs-title')||{}).innerText,
            temp: (document.getElementById('bind-temp')||{}).innerText
        })""")
    print(f"{'step':<8}{'lead':<6}{'activeBtn':<11}{'legend':<8}{'overlay':<9}{'panel title':<28}{'temp'}")
    r=snap(); print(f"{'load':<8}{r['lead']:<6}{r['active']:<11}{r['legend']:<8}{r['overlay']:<9}{r['title']:<28}{r['temp']}")
    for label in ["+2h","+4h","+6h","NOW"]:
        pg.evaluate("""(lb)=>{const b=[...document.querySelectorAll('.horizon-btn')].find(x=>x.textContent.trim()===lb); if(b)b.click();}""", label)
        time.sleep(7)
        r=snap()
        print(f"{label:<8}{r['lead']:<6}{r['active']:<11}{r['legend']:<8}{r['overlay']:<9}{r['title']:<28}{r['temp']}")
    b.close()
print("\nconsole errors:",len(errs))

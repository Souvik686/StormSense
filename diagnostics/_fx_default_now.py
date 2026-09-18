"""Does the dashboard open on NOW, with the right surface, label and button?"""
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]; surf=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.on("request", lambda r: surf.append(r.url.split("/api/")[-1]) if "risk-surface" in r.url else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(13)
    st=pg.evaluate("""() => {
        const active=[...document.querySelectorAll('.horizon-btn')]
            .filter(b=>b.className.includes('bg-cyan-600')).map(b=>b.textContent.trim());
        const imgs=[...document.querySelectorAll('img')]
            .filter(i=>(i.src||'').includes('risk-surface')).map(i=>i.src.split('?')[1]);
        return {
          currentLeadHours: window.currentLeadHours,
          activeButtons: active,
          legendTag: (document.getElementById('legend-lead-tag')||{}).textContent,
          overlaySrc: imgs,
          obsTitle: (document.getElementById('location-obs-title')||{}).innerText,
          bindTemp: (document.getElementById('bind-temp')||{}).innerText,
          bindRain: (document.getElementById('bind-rain')||{}).innerText
        };
    }""")
    for k,v in st.items(): print(f"   {k:<18}: {v}")
    print("   surface requests   :", surf[:4])
    pg.screenshot(path="diagnostics/results/fx_default_now.png", clip={"x":330,"y":820,"width":900, "height":260})
    b.close()
print("\nconsole errors:",len(errs))

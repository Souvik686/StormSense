"""Capture the SAME fields in live -> historical -> live and diff them.
Any field that returns to historical values after switching back is leaking."""
from playwright.sync_api import sync_playwright
import time, json

FIELDS = """() => {
  const g=id=>{const e=document.getElementById(id); return e?(e.innerText||'').trim():'(missing)';};
  const out={};
  ['bind-temp','bind-rain','bind-humidity','bind-wind',
   'bind-temp-source','bind-rain-source','bind-humidity-source','bind-wind-source',
   'location-obs-title','location-obs-age','location-risk-horizon',
   'meter-ts-label','meter-rain-label','meter-flood-label',
   'profile-district','profile-coords','profile-meta',
   'nowcast-timeline-title','nowcast-timeline-badge',
   'last-updated-time','bind-operational-mode','desk-subtitle',
   'pipeline-status-text','legend-lead-tag'].forEach(i=>out[i]=g(i));
  // hazard cards
  const m=document.body.innerText.match(/Thunderstorm\s+([\d.]+%)[\s\S]{0,400}?Heavy Rainfall\s+([\d.]+\s*mm)[\s\S]{0,400}?Flash-Flood Risk Proxy\s+([\d.]+%)/);
  out._cards = m ? m.slice(1).join(' | ') : '(n/a)';
  out._obsStrip = (document.getElementById('nowcast-steps')||{innerText:''}).innerText.replace(/\s+/g,' ').trim().slice(0,150);
  out._mode = window.stormSenseMode;
  return out;
}"""

with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1200},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(14)
    live1=pg.evaluate(FIELDS)
    pg.evaluate("switchMode('historical')"); time.sleep(14)
    hist=pg.evaluate(FIELDS)
    pg.evaluate("switchMode('live')"); time.sleep(16)
    live2=pg.evaluate(FIELDS)
    b.close()

print(f"{'field':<26}{'LIVE (1st)':<30}{'HISTORICAL':<34}{'LIVE (after switch back)':<30}")
print("-"*122)
leaks=[]
for k in live1:
    a,h,c = str(live1[k])[:28], str(hist[k])[:32], str(live2[k])[:28]
    flag=""
    if c==h and a!=h:
        flag="  <== LEAK (still historical)"; leaks.append(k)
    print(f"{k:<26}{a:<30}{h:<34}{c:<30}{flag}")
print(f"\nLEAKING FIELDS: {len(leaks)}")
for l in leaks: print("   -",l)
print("console errors:",len(errs))

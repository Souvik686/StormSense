"""Does granting geolocation populate the 8 stuck fields?"""
from playwright.sync_api import sync_playwright
import time
def run(label, grant):
    with sync_playwright() as p:
        b=p.chromium.launch()
        kw={}
        if grant:
            kw=dict(permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
        ctx=b.new_context(viewport={"width":1600,"height":1200}, **kw)
        pg=ctx.new_page()
        pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
        time.sleep(14)
        r=pg.evaluate("""() => {
            const ids=['bind-temp','bind-rain','bind-humidity','bind-wind',
                       'meter-ts-label','meter-rain-label','meter-flood-label','location-obs-age',
                       'profile-district','profile-coords'];
            const o={};
            for(const id of ids){const e=document.getElementById(id); o[id]=e?(e.innerText||'').trim():'(missing)';}
            return o;
        }""")
        b.close()
    print(f"\n=== {label} ===")
    for k,v in r.items(): print(f"   {k:<20}: {v!r}")
    return r
a=run("geolocation DENIED (headless default)", False)
b_=run("geolocation GRANTED (like your browser)", True)
stuck=[k for k,v in b_.items() if v in ('...','—')]
print(f"\nstill stuck when granted: {stuck if stuck else 'NONE -- all populated'}")

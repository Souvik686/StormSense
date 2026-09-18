"""Verify: (1) map loads on the new key, (2) failover fires when a key fails."""
from playwright.sync_api import sync_playwright
import time

def run(label, force_fail_first):
    msgs=[]
    with sync_playwright() as p:
        b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
        pg.on("console", lambda m: msgs.append((m.type,m.text)))
        if force_fail_first:
            # Simulate key#1 exhaustion: fire gm_authFailure right after the SDK
            # attaches, exactly as Google would on a quota breach.
            pg.add_init_script("""
                (function(){
                  var tries=0;
                  var iv=setInterval(function(){
                    if (typeof window.gm_authFailure === 'function' && !window.__forced) {
                      window.__forced = true; clearInterval(iv);
                      window.gm_authFailure();   // pretend key #1 is over quota
                    }
                    if (++tries>200) clearInterval(iv);
                  }, 20);
                })();
            """)
        pg.goto("http://127.0.0.1:8000/index.html", wait_until="domcontentloaded", timeout=90000)
        time.sleep(14)
        st=pg.evaluate("""() => ({
            ready: !!window.STORMSENSE_MAPS_READY,
            exhausted: !!window.STORMSENSE_MAPS_EXHAUSTED,
            state: window.STORMSENSE_MAPS_KEY_STATE,
            hasGoogle: typeof google!=='undefined' && !!(google.maps),
            hasMap: !!(window.stormSenseMap && window.stormSenseMap._gmap),
            divText: (document.getElementById('map-radar-placeholder')||{innerText:''}).innerText.slice(0,110)
        })""")
        pg.screenshot(path=f"diagnostics/results/fx_maps_{label}.png", clip={"x":0,"y":700,"width":1000,"height":300})
        b.close()
    print(f"\n=== {label} ===")
    for k,v in st.items(): print(f"  {k}: {v}")
    for t,m in msgs:
        if "[maps]" in m or "authFailure" in m or "quota" in m.lower():
            print(f"  console[{t}]: {m[:150]}")
    return st

a=run("normal", False)
b_=run("forced_failover", True)
print("\n--- SUMMARY ---")
print(f"normal          : ready={a['ready']} activeKey#={a['state']['activeIndex']+1} map_built={a['hasMap']}")
print(f"forced failover : ready={b_['ready']} activeKey#={b_['state']['activeIndex']+1} failed={b_['state']['failed']} map_built={b_['hasMap']}")

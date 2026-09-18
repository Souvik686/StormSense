from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1000},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page()
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(13)
    pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
    time.sleep(3)
    info=pg.evaluate("""() => {
        const b=document.getElementById('btn-my-location');
        const r=b.getBoundingClientRect();
        const top=document.elementFromPoint(r.x+r.width/2, r.y+r.height/2);
        return {rect:{x:Math.round(r.x),y:Math.round(r.y)},
                topEl: top?(top.id||top.tagName+'.'+String(top.className).slice(0,40)):null,
                isButtonOnTop: !!(top && (top.id==='btn-my-location' || b.contains(top)))};
    }""")
    print("LIVE :",info)
    pg.locator("#btn-my-location").screenshot(path="diagnostics/results/fx_locbtn_live.png")
    pg.screenshot(path="diagnostics/results/fx_map_controls.png",
                  clip={"x":700,"y":150,"width":420,"height":420})
    b.close()

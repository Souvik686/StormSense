from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1600,"height":1000},
        permissions=["geolocation"], geolocation={"latitude":22.5726,"longitude":88.3639})
    pg=ctx.new_page(); errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(13)
    pg.evaluate("()=>document.getElementById('map-radar-placeholder').scrollIntoView({block:'center'})")
    time.sleep(3)
    print(pg.evaluate("""()=>{const b=document.getElementById('btn-my-location');
        const r=b.getBoundingClientRect();
        return {text:b.innerText.trim(), w:Math.round(r.width), h:Math.round(r.height), visible:b.offsetParent!==null};}"""))
    pg.screenshot(path="diagnostics/results/fx_locbtn_final.png", clip={"x":700,"y":150,"width":420,"height":300})
    b.close()
print("console errors:",len(errs))

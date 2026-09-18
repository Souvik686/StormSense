"""Render the popup helper functions directly (no Google Maps needed)."""
from playwright.sync_api import sync_playwright
import json,urllib.request
js=open("frontend/js/app.js",encoding="utf-8").read()
snip=js[js.index("var NOW_VERDICT_STYLE"):js.index("function handleMapClick")]
d=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/nowcast/point?lat=22.5726&lon=88.3639&lead=0&mode=live",timeout=60))
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":420,"height":700})
    pg.goto("data:text/html,<body style='background:#0f172a;margin:0;padding:10px;font-family:monospace'></body>")
    pg.add_script_tag(content=snip)
    html=pg.evaluate("d => nowEvidenceHtml(d.now_evidence) + nowProvenanceHtml(d.now_provenance)", d)
    pg.evaluate("h => document.body.innerHTML = h", html)
    print(pg.evaluate("() => document.body.innerText"))
    pg.screenshot(path="diagnostics/results/fx_popup_blocks.png", full_page=True)
    b.close()

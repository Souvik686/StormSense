"""Render each verdict's HTML through the REAL frontend function, in a browser."""
from playwright.sync_api import sync_playwright
import json,re,time
js=open("frontend/js/app.js",encoding="utf-8").read()
start=js.index("var NOW_VERDICT_STYLE")
end=js.index("function handleMapClick")
snippet=js[start:end]
cases={
 "OBSERVED_NOT_MODELLED":{"verdict":"OBSERVED_NOT_MODELLED","model_probability_pct":2.0,
   "observed":{"radar_echo_fraction":0.45,"radar_has_coverage":True,"radar_time_utc":"2026-09-18T18:10:00+00:00",
               "station_rain_1h_mm":0.0,"nearest_station":"Kolkata (0.00° away)","station_time_utc":"2026-09-18T18:12:00+00:00"},
   "model_input":{"analysis_time_utc":"2026-09-18T12:00:00+00:00"},
   "product_time_gaps_hours":{"reference_minus_analysis":6.25}},
 "MODELLED_NOT_OBSERVED":{"verdict":"MODELLED_NOT_OBSERVED","model_probability_pct":66.0,
   "observed":{"radar_echo_fraction":0.0,"radar_has_coverage":True,"radar_time_utc":"2026-09-18T18:10:00+00:00",
               "station_rain_1h_mm":0.0,"nearest_station":"Purulia (0.12° away)","station_time_utc":"2026-09-18T18:12:00+00:00"},
   "model_input":{"analysis_time_utc":"2026-09-18T12:00:00+00:00"},
   "product_time_gaps_hours":{"reference_minus_analysis":6.25}},
 "NO_OBSERVATION":{"verdict":"NO_OBSERVATION","model_probability_pct":30.0,
   "observed":{"radar_echo_fraction":None,"radar_has_coverage":False,"radar_time_utc":None,
               "station_rain_1h_mm":None,"nearest_station":None,"station_time_utc":None},
   "model_input":{"analysis_time_utc":"2026-09-18T12:00:00+00:00"},
   "product_time_gaps_hours":{"reference_minus_analysis":None}},
}
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":420,"height":900})
    pg.goto("data:text/html,<body style='background:#0f172a;margin:0;padding:10px;font-family:monospace'></body>")
    pg.add_script_tag(content=snippet)
    for name,ev in cases.items():
        html=pg.evaluate("ev => nowEvidenceHtml(ev)", ev)
        pg.evaluate("h => document.body.insertAdjacentHTML('beforeend','<div style=\"color:#64748b;font-size:10px;margin:8px 0 2px\">'+h.k+'</div>'+h.v)", {"k":name,"v":html})
        txt=pg.evaluate("h => {const d=document.createElement('div'); d.innerHTML=h; return d.innerText;}", html)
        print(f"=== {name} ===\n{txt}\n")
    pg.screenshot(path="diagnostics/results/shot_verdicts.png", full_page=True)
    b.close()
print("screenshot -> diagnostics/results/shot_verdicts.png")

"""Trace EVERY displayed number on each view back to its backend source."""
from playwright.sync_api import sync_playwright
import json,urllib.request,re,time
B="http://127.0.0.1:8000"
def api(u):
    return json.load(urllib.request.urlopen(B+u,timeout=90))

summ=api("/api/nowcast/summary?lead=2&mode=live")
thermo=api("/api/nowcast/thermodynamics?mode=live")
mi=api("/api/model-info")
dh=api("/api/data-health")
bm=api("/api/benchmark/models")
wx=api("/api/weather/current?lat=22.5726&lon=88.3639")
dist=api("/api/nowcast/districts?lead=2&mode=live")

print("=== BACKEND SOURCE VALUES ===")
h=summ["hazards"]
print(f"  cards: thunder={h['thunderstorm']['probability']}% rain={h['heavy_rainfall']['rate_mm_3h']}mm "
      f"ff={h['flash_flood']['probability']}% overall={h['overall']['probability']}%")
print(f"  obs  : T={wx['temperature']}C RH={wx['humidity']}% P={wx['pressure']}hPa "
      f"W={wx['wind_speed']}km/h vis={wx['visibility']}km rain1h={wx['rainfall_1h']}mm")
print(f"  model: params={mi['n_parameters']} grid={mi['grid_shape']} leads={mi['lead_times_hours']}")
print(f"  thresholds: {mi['calibrated_thresholds']}")
print(f"  temps     : {mi['calibrated_temperatures']}")
print(f"  districts : {len(dist)} entries")
print(f"  thermo    : CAPE={thermo.get('cape_j_kg')} CIN={thermo.get('cin_j_kg')} shear={thermo.get('bulk_shear_1000_700hpa_mps')}")
bmk = bm if isinstance(bm,list) else bm.get("models", bm)
print(f"  benchmark : {type(bmk).__name__} len={len(bmk) if hasattr(bmk,'__len__') else '?'}")

views=["dashboard","radar","advisories","wrf","xai","gis","threshold","streams"]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1200})
    errs=[]
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto(B+"/index.html", wait_until="networkidle", timeout=90000)
    time.sleep(9)
    for v in views:
        el=pg.query_selector(f"[data-view='{v}']") or pg.query_selector(f"#nav-{v}")
        if el: el.click()
        time.sleep(3.5)
        t=pg.evaluate("() => document.body.innerText")
        nums=re.findall(r"-?\d+(?:\.\d+)?\s*(?:%|mm|hPa|km/h|°C|km|J/kg|m/s)", t)
        print(f"\n--- {v.upper()} : {len(nums)} numeric values rendered ---")
        print("   ", " | ".join(dict.fromkeys(nums))[:520])
    b.close()
print("\nconsole errors:",len(errs))

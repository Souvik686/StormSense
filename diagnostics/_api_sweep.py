"""Sweep EVERY endpoint the app exposes, both modes, checking status + shape."""
import json,urllib.request,urllib.error
B="http://127.0.0.1:8000"
EPS=[
 ("/api/health",None),("/api/system/info",None),("/api/model-info",None),("/api/config",None),
 ("/api/weather/current",None),("/api/data-health",None),("/api/mode/status",None),
 ("/api/live/ml-status",None),("/api/live/surface",None),("/api/now/provenance",None),
 ("/api/satellite/info",None),("/api/benchmark/models",None),
 ("/api/boundaries/west-bengal",None),("/api/boundaries/state",None),("/api/geo/areas",None),
 ("/api/nowcast/risk-surface/bounds",None),
 ("/api/historical/case-study",None),("/api/historical/analysis-state",None),
 ("/api/historical/satellite",None),
 ("/api/observations/current",None),("/api/observations/surface",None),
 ("/api/forecast/conditions",None),("/api/stormsense/timeline",None),
]
MODED=["/api/nowcast/summary","/api/nowcast/districts","/api/nowcast/thermodynamics",
       "/api/nowcast/high-risk-cells","/api/nowcast/xai","/api/nowcast/risk-map",
       "/api/nowcast/explanation","/api/stormsense/nowcast","/api/time/reference"]
def get(u):
    try:
        r=urllib.request.urlopen(B+u,timeout=60)
        raw=r.read()
        try: j=json.loads(raw); return r.status,type(j).__name__,len(raw),j
        except Exception: return r.status,"non-json",len(raw),None
    except urllib.error.HTTPError as e: return e.code,"httperror",0,None
    except Exception as e: return -1,str(e)[:60],0,None
bad=[]
print("=== SIMPLE ENDPOINTS ===")
for u,_ in EPS:
    s,t,n,j=get(u)
    flag="" if s==200 else "  <== NON-200"
    if s!=200: bad.append((u,s))
    print(f"  {s:>4} {t:<12} {n:>8}B  {u}{flag}")
print("\n=== MODE + LEAD ENDPOINTS ===")
for u in MODED:
    for mode in ["live","historical"]:
        for lead in [0,2,6]:
            uu=f"{u}?mode={mode}&lead={lead}"
            s,t,n,j=get(uu)
            if s!=200: bad.append((uu,s))
            print(f"  {s:>4} {t:<10} {n:>8}B  {uu}")
print("\n=== RISK SURFACE (PNG) ===")
for mode in ["live","historical"]:
    for lead in [0,2,3,4,5,6]:
        s,t,n,_=get(f"/api/nowcast/risk-surface?lead={lead}&mode={mode}")
        if s!=200: bad.append((f"risk-surface {mode} {lead}",s))
        print(f"  {s:>4} {t:<10} {n:>8}B  lead={lead} mode={mode}")
print("\n=== INVALID INPUT HANDLING ===")
for uu in ["/api/nowcast/risk-surface?lead=99&mode=live","/api/nowcast/risk-map?lead=99&mode=live",
           "/api/nowcast/point?lat=0&lon=0&lead=0&mode=live","/api/nowcast/point?lat=22.5&lon=88.3&lead=99&mode=live"]:
    s,t,n,j=get(uu)
    note=""
    if j and isinstance(j,dict) and j.get("inside_monitored_region") is False: note=" (correctly rejected as outside region)"
    print(f"  {s:>4} {t:<10} {uu}{note}")
print(f"\nNON-200 COUNT: {len(bad)}")
for u,s in bad: print("   ",s,u)

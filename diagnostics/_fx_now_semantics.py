"""What does live NOW actually MEAN in each case?"""
from datetime import datetime,timezone,timedelta
import sys,os
sys.path.insert(0,os.path.abspath("."))
from src.inference import gfs_live as G
LAG=G.GFS_PRODUCTION_LAG_HOURS
print("For each wall-clock hour: what the NOW slot actually represents.\n")
print(f"{'wall':>6} {'fc_analysis':>13} {'now_analysis':>13} {'NOW valid':>12} {'vs wallclock':>13}  semantics")
same=0; diff=0
for hh in range(24):
    t=datetime(2026,9,18,hh,0,tzinfo=timezone.utc)
    c1=G._cycles_available_at(t,LAG); c2=G._cycles_available_at(t-timedelta(hours=2),LAG)
    if not c1 or not c2: continue
    a1,a2=c1[0],c2[0]
    now_valid=a2+timedelta(hours=2)     # T-2h run's +2h head is valid at a2+2h
    delta=(now_valid-t).total_seconds()/3600
    if a1==a2:
        same+=1; sem=f"IDENTICAL to +2h (valid {a1+timedelta(hours=2):%HZ})"
    else:
        diff+=1; sem="distinct from +2h"
    print(f"{hh:02d}:00Z {a1:%d %HZ} {a2:%d %HZ} {now_valid:%d %HZ} {delta:+12.1f}h  {sem}")
print(f"\nhours where NOW == +2h exactly: {same}/24   distinct: {diff}/24")
print("\nKEY POINT: NOW's valid time is analysis+2h, which is NOT 'now'.")
print("It is in the PAST relative to wall-clock whenever analysis+2h < wallclock.")

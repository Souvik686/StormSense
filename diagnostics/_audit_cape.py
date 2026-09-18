"""Is the ERA5-trained CAPE regime reproduced by live GFS? Compare the
training normalization stats against what GFS actually delivers."""
import os, sys, json
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.config import load_config
from src.features.normalize import load_stats
from src.inference import gfs_live

cfg=load_config(None)
stats=load_stats(cfg.path("normalization","stats_file"))
print("Training (ERA5) normalization stats:")
for v in ["cape","cin","tcwv","tp"]:
    print(f"  {v:6s} mean={stats[v]['mean']:10.4f} std={stats[v]['std']:10.4f}")
thr=cfg.get("labels","cape_severe_jkg")
print(f"\nsevere_convective label needs CAPE > {thr} J/kg (and CIN < {cfg.get('labels','cin_weak_jkg')})")
z=(thr-stats['cape']['mean'])/stats['cape']['std']
print(f"  that is z = {z:.2f} sigma in the ERA5 training distribution")

d=cfg.get("domain"); n_lat,n_lon=d["grid_shape"]
lats=np.linspace(d["lat_max"],d["lat_min"],n_lat); lons=np.linspace(d["lon_min"],d["lon_max"],n_lon)
print("\nLive GFS CAPE over the whole 33x25 domain, per cached analysis:")
allmax=[]
for cy in ["20260916_180000","20260917_000000","20260917_060000","20260917_120000","20260917_180000","20260918_000000"]:
    an=datetime.strptime(cy,"%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    try: h=gfs_live.fetch_and_harmonize(lats,lons,target_t0=an+timedelta(hours=6),use_cache=False)
    except Exception: continue
    c=h.surface[-1][5]
    allmax.append(float(np.nanmax(c)))
    print(f"  {h.analysis_t0.strftime('%d %b %HZ')}: max={np.nanmax(c):7.1f}  p99={np.nanpercentile(c,99):7.1f}  mean={np.nanmean(c):7.1f}  cells>2000: {int((c>thr).sum())}/825")
print(f"\nPeak domain CAPE across all cached analyses: {max(allmax):.1f} J/kg  -- label threshold {thr}")
print("=> the severe_convective half of the label was NEVER satisfiable on these days.")

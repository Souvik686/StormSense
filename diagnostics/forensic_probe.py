"""StormSense forensic probe -- radar vs ML risk, one location/time at a time.

Reuses the PRODUCTION pipeline (gfs_live.fetch_and_harmonize + NowcastPredictor)
so the numbers it prints are the numbers the app serves. No parallel maths.

Usage:
    python diagnostics/forensic_probe.py --lat 22.5726 --lon 88.3639
    python diagnostics/forensic_probe.py --lat 22.4950 --lon 88.3450 --at 2026-09-17T12:30Z
"""
import argparse, json, math, os, sys, urllib.request
os.environ.setdefault("STORMSENSE_DEVICE", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from datetime import datetime, timezone, timedelta

from src.utils.config import load_config
from src.inference import gfs_live
from src.inference.predictor import get_predictor
from src.features.normalize import SINGLE_VARS

IST = timedelta(hours=5, minutes=30)
CKPT = "Data/outputs/checkpoints/v2_calibrated_best.pt"


def band(p):
    return "WARNING" if p >= .75 else "ALERT" if p >= .50 else "WATCH" if p >= .25 else "NORMAL"


def radar_frame():
    """Latest RainViewer frame + whether the tile over the point has echo."""
    try:
        d = json.load(urllib.request.urlopen(
            "https://api.rainviewer.com/public/weather-maps.json", timeout=25))
        f = d["radar"]["past"][-1]
        return d["host"], f["path"], datetime.fromtimestamp(f["time"], timezone.utc)
    except Exception as e:
        return None, None, e


def tile_echo(host, path, lat, lon, z=7):
    """Fraction of non-transparent pixels = actual echo. Distinguishes
    'no precipitation' from 'no tile / no coverage'."""
    try:
        from PIL import Image
        import io
        n = 2 ** z
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
        b = urllib.request.urlopen(f"{host}{path}/256/{z}/{x}/{y}/2/1_1.png", timeout=25).read()
        a = np.array(Image.open(io.BytesIO(b)).convert("RGBA"))
        return f"{z}/{x}/{y}", 100.0 * (a[..., 3] > 0).sum() / (a.shape[0] * a.shape[1]), len(b)
    except Exception as e:
        return None, None, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--at", default=None, help="UTC wall-clock, e.g. 2026-09-17T12:30Z (default: now)")
    a = ap.parse_args()

    t0 = (datetime.now(timezone.utc) if not a.at
          else datetime.fromisoformat(a.at.replace("Z", "+00:00")))

    cfg = load_config(None)
    d = cfg.get("domain")
    n_lat, n_lon = d["grid_shape"]
    lats = np.linspace(d["lat_max"], d["lat_min"], n_lat)
    lons = np.linspace(d["lon_min"], d["lon_max"], n_lon)

    i = int(np.argmin(np.abs(lats - a.lat)))
    j = int(np.argmin(np.abs(lons - a.lon)))
    dkm = math.hypot((lats[i] - a.lat) * 111.0,
                     (lons[j] - a.lon) * 111.0 * math.cos(math.radians(a.lat)))

    print("=" * 74)
    print(f"FORENSIC PROBE  ({a.lat}, {a.lon})   wall-clock {t0.strftime('%Y-%m-%d %H:%M UTC')} "
          f"/ {(t0 + IST).strftime('%H:%M IST')}")
    print("=" * 74)

    print("\n[MODEL GRID]")
    print(f"  nearest cell [{i},{j}] = {lats[i]:.2f}N, {lons[j]:.2f}E   offset {dkm:.1f} km")
    print(f"  cell size 0.25 deg (~28 km) -- a single cell spans the whole Kolkata metro")

    print("\n[RADAR]  (observation; NOT a model input)")
    host, path, rt = radar_frame()
    if host:
        tid, pct, nb = tile_echo(host, path, a.lat, a.lon)
        print(f"  source        : RainViewer composite mosaic")
        print(f"  latest frame  : {rt.strftime('%Y-%m-%d %H:%M UTC')} / {(rt + IST).strftime('%H:%M IST')}")
        if tid:
            print(f"  tile {tid:<12}: {nb} bytes, echo covers {pct:.2f}% of tile")
            print(f"  -> {'tile served WITH echo (coverage is real)' if pct > 0 else 'tile served but EMPTY (no precip, coverage OK)'}")
    else:
        print(f"  unavailable: {rt}")

    print("\n[LIGHTNING]")
    print("  no live lightning feed is ingested anywhere in this system;")
    print("  Data/LIGHTNING is ISS-LIS 2020 only. Not an input, not a product.")

    pred = get_predictor(CKPT, None, device="cpu")
    dem = np.load(os.path.join(cfg.path("paths", "cache_root"), "era5_memmap",
                               "dem_elevation_m.npy")).astype(np.float32)
    h = gfs_live.fetch_and_harmonize(lats, lons, target_t0=t0, use_cache=False)
    hn = gfs_live.fetch_and_harmonize(lats, lons, target_t0=t0 - timedelta(hours=2), use_cache=False)

    print("\n[MODEL INPUT]  GFS f000 analyses only; no forecast hour is used as input")
    print(f"  analysis time : {h.analysis_t0.strftime('%Y-%m-%d %H:%M UTC')} / "
          f"{(h.analysis_t0 + IST).strftime('%H:%M IST')}")
    print(f"  analysis lag  : {h.wallclock_age_hours:.2f} h behind wall-clock")
    if host:
        print(f"  RADAR-vs-INPUT GAP : {(rt - h.analysis_t0).total_seconds()/3600:.2f} h "
              f"<-- the two products describe different instants")

    idx = {v: SINGLE_VARS.index(v) for v in SINGLE_VARS}
    s0 = h.surface[-1]
    print("\n[SURFACE INPUTS at this cell]")
    for v, u in [("cape", "J/kg"), ("cin", "J/kg"), ("tcwv", "kg/m2"), ("tp", "mm/h"),
                 ("t2m", "K"), ("d2m", "K"), ("sp", "Pa"), ("u10", "m/s"), ("v10", "m/s")]:
        print(f"  {v:5s} = {float(s0[idx[v]][i, j]):10.3f} {u}")
    tp3 = float(h.surface[-3:, idx["tp"], i, j].sum())
    print(f"  3h rain (sum of last 3 slots) = {tp3:.3f} mm   [label needs > "
          f"{cfg.get('labels','heavy_rain_3h_mm')} mm]")
    print(f"  severe-convective criterion: CAPE>{cfg.get('labels','cape_severe_jkg')} "
          f"AND CIN<{cfg.get('labels','cin_weak_jkg')} -> "
          f"{float(s0[idx['cape']][i,j]) > cfg.get('labels','cape_severe_jkg') and float(s0[idx['cin']][i,j]) < cfg.get('labels','cin_weak_jkg')}")

    now = pred.predict(surface=hn.surface, pressure=hn.pressure, dem=dem,
                       timestamp=hn.analysis_t0.isoformat())["severe_weather_prob"][0]
    fc = pred.predict(surface=h.surface, pressure=h.pressure, dem=dem,
                      timestamp=h.analysis_t0.isoformat())["severe_weather_prob"]

    print("\n[MODEL OUTPUT]  calibrated severe-weather probability (temperature-scaled)")
    print(f"  NOW (lead 0, from T-2h window) : {float(now[i,j])*100:6.2f}%  {band(float(now[i,j]))}")
    for li, lh in enumerate(pred.lead_times):
        print(f"  +{lh}h                           : {float(fc[li,i,j])*100:6.2f}%  {band(float(fc[li,i,j]))}")
    print(f"  domain max (any cell, lead 2)  : {fc[0].max()*100:6.2f}%")
    print(f"  calibration T per lead         : {pred.temperature}")
    print(f"  decision thresholds per lead   : {pred.threshold_per_lead}")

    print("\n[DISPLAY MAPPING]  <25 NORMAL | 25-50 WATCH | 50-75 ALERT | >=75 WARNING")
    print("  map PNG = same field, cubic-interpolated + smoothed for display only;")
    print("  the underlying ML grid stays 0.25 deg. Popup uses the raw cell value.")
    print("=" * 74)


if __name__ == "__main__":
    main()

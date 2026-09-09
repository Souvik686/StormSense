"""Programmatic inventory of every dataset in the workspace.

Writes reports/data_inventory.json and prints a human summary.
Read-only: never modifies Data/.
"""
from __future__ import annotations
import json, os, re, glob, zipfile, tempfile, collections, io
from datetime import datetime
import numpy as np, pandas as pd

ROOT = os.environ.get("WHN_DATA_ROOT", r"D:\Weather-Hackathon\Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def dir_size_mb(p):
    tot = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return round(tot / 1e6, 1)


def insat_timestamps(prod):
    ts = []
    for f in glob.glob(os.path.join(ROOT, "INSAT", prod, "*.h5")):
        m = re.match(r"3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_", os.path.basename(f))
        if m:
            ts.append(datetime(int(m[3]), MON[m[2]], int(m[1]), int(m[4]), int(m[5])))
    return pd.DatetimeIndex(sorted(ts))


def inventory_insat():
    import h5py
    out = {}
    for prod in ["HEM", "UTH", "CTP", "CMK"]:
        d = os.path.join(ROOT, "INSAT", prod)
        if not os.path.isdir(d):
            continue
        ts = insat_timestamps(prod)
        files = sorted(glob.glob(os.path.join(d, "*.h5")))
        gaps = np.diff(ts.values).astype("timedelta64[m]").astype(int) if len(ts) > 1 else []
        info = {"n_files": len(files), "size_mb": dir_size_mb(d),
                "n_timestamps": len(ts),
                "dates": sorted({str(x) for x in ts.date}),
                "modal_gap_min": int(pd.Series(gaps).mode()[0]) if len(gaps) else None}
        with h5py.File(files[0], "r") as f:
            vars_ = {}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    a = obj.attrs
                    dec = lambda x: x.decode() if isinstance(x, bytes) else x
                    vars_[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype),
                                   "long_name": dec(a.get("long_name", "")),
                                   "units": dec(a.get("units", "")),
                                   "fill_value": (float(np.asarray(a["_FillValue"]).ravel()[0])
                                                  if "_FillValue" in a else None)}

            f.visititems(visit)
            info["variables"] = vars_
            info["satellite"] = f.attrs.get("Satellite_Name", b"").decode()
            info["nadir_lat_lon"] = [float(x) for x in np.asarray(
                f.attrs["Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude"])]
            lat = f["Latitude"][:].astype(np.float64)
            lon = f["Longitude"][:].astype(np.float64)
            fv = float(np.asarray(f["Latitude"].attrs["_FillValue"]).ravel()[0])
            sf = float(np.asarray(f["Latitude"].attrs["scale_factor"]).ravel()[0])
            m = (lat != fv) & (lon != fv)
            lat = np.where(m, lat * sf, np.nan)
            lon = np.where(m, lon * sf, np.nan)
            info["lat_range"] = [float(np.nanmin(lat)), float(np.nanmax(lat))]
            info["lon_range"] = [float(np.nanmin(lon)), float(np.nanmax(lon))]
            info["grid"] = "geostationary projection; per-pixel lat/lon stored int16 x0.01 deg"
        out[prod] = info
    return out


def inventory_era5():
    import xarray as xr
    out = {}
    pl_files = sorted(glob.glob(os.path.join(ROOT, "ERA5", "pressure_levels", "*", "*", "*.nc")))
    groups = collections.defaultdict(list)
    times = []
    last = None
    for p in pl_files:
        ds = xr.open_dataset(p, decode_timedelta=False)
        key = ("+".join(sorted(ds.data_vars)),
               "/".join(str(int(x)) for x in ds.pressure_level.values))
        groups[key].append(p)
        times.append(pd.DatetimeIndex(ds.valid_time.values))
        last = ds
    t = pd.DatetimeIndex(np.unique(np.concatenate([x.values for x in times])))
    out["pressure_levels"] = {
        "n_files": len(pl_files),
        "size_mb": dir_size_mb(os.path.join(ROOT, "ERA5", "pressure_levels")),
        "groups": {f"{k[0]} @ {k[1]} hPa": len(v) for k, v in groups.items()},
        "n_hours": len(t), "t_min": str(t.min()), "t_max": str(t.max()),
        "lat": [float(last.latitude.min()), float(last.latitude.max())],
        "lon": [float(last.longitude.min()), float(last.longitude.max())],
        "shape_latlon": [int(last.sizes["latitude"]), int(last.sizes["longitude"])],
        "res_deg": 0.25, "freq": "hourly"}

    sl_zips = sorted(glob.glob(os.path.join(ROOT, "ERA5", "single_levels", "*.zip")))
    td = tempfile.mkdtemp()
    sinfo = {"n_zips": len(sl_zips),
             "size_mb": dir_size_mb(os.path.join(ROOT, "ERA5", "single_levels")),
             "members": {}}
    tall = collections.defaultdict(list)
    for z in sl_zips:
        with zipfile.ZipFile(z) as f:
            d = os.path.join(td, os.path.basename(z)[:-4])
            f.extractall(d)
        for n in sorted(os.listdir(d)):
            ds = xr.open_dataset(os.path.join(d, n), decode_timedelta=False)
            kind = "accum" if "accum" in n else "instant"
            sinfo["members"][kind] = {
                "variables": sorted(ds.data_vars),
                "units": {v: ds[v].attrs.get("units", "") for v in ds.data_vars}}
            tall[kind].append(pd.DatetimeIndex(ds.valid_time.values))
            ds.close()
    for k, v in tall.items():
        t = pd.DatetimeIndex(np.unique(np.concatenate([x.values for x in v])))
        sinfo["members"][k].update({"n_hours": len(t), "t_min": str(t.min()), "t_max": str(t.max())})
    sinfo["res_deg"] = 0.25
    sinfo["freq"] = "hourly"
    out["single_levels"] = sinfo
    return out


def inventory_imd():
    import xarray as xr
    fs = sorted(glob.glob(os.path.join(ROOT, "IMD_RAINFALL", "*.nc")))
    per = {}
    for p in fs:
        ds = xr.open_dataset(p, decode_timedelta=False)
        sub = ds.RAINFALL.sel(LATITUDE=slice(20, 28), LONGITUDE=slice(84, 90)).values
        per[os.path.basename(p)] = {
            "n_days": int(ds.sizes["TIME"]),
            "t_min": str(pd.Timestamp(ds.TIME.values[0]).date()),
            "t_max": str(pd.Timestamp(ds.TIME.values[-1]).date()),
            "nan_frac_wb_window": round(float(np.isnan(sub).mean()), 4),
            "max_mm_wb_window": round(float(np.nanmax(sub)), 1)}
        ds.close()
    return {"n_files": len(fs), "size_mb": dir_size_mb(os.path.join(ROOT, "IMD_RAINFALL")),
            "res_deg": 0.25, "freq": "daily", "units": "mm/day",
            "domain": "India 6.5-38.5N, 66.5-100E", "per_file": per}


def inventory_lightning():
    p = os.path.join(ROOT, "LIGHTNING", "india_lightning_Strike.csv")
    d = pd.read_csv(p, parse_dates=["utc_time"])
    wb = d[(d.latitude.between(20, 28)) & (d.longitude.between(84, 90))]
    return {"file": os.path.basename(p), "size_mb": round(os.path.getsize(p) / 1e6, 1),
            "n_strikes": len(d), "n_strikes_wb_domain": len(wb),
            "t_min": str(d.utc_time.min()), "t_max": str(d.utc_time.max()),
            "n_dates": int(d.date.nunique()), "n_dates_wb": int(wb.date.nunique()),
            "sensor": "ISS-LIS V3.0", "columns": list(d.columns),
            "note": "ISS orbital overpass sampling; not a continuous time series"}


def inventory_dem():
    import rasterio
    fs = sorted(glob.glob(os.path.join(ROOT, "DEM", "SRTM", "*.tif")))
    lats, lons = set(), set()
    for f in fs:
        m = re.search(r"n(\d+)_e(\d+)", os.path.basename(f))
        lats.add(int(m[1]))
        lons.add(int(m[2]))
    with rasterio.open(fs[0]) as s:
        meta = {"crs": str(s.crs), "shape": list(s.shape), "res_deg": list(s.res),
                "dtype": s.dtypes[0], "nodata": s.nodata}
    return {"n_tiles": len(fs), "size_mb": dir_size_mb(os.path.join(ROOT, "DEM")),
            "lat_tiles": sorted(lats), "lon_tiles": sorted(lons),
            "coverage": f"{min(lats)}-{max(lats) + 1}N, {min(lons)}-{max(lons) + 1}E",
            "tile_meta": meta, "res_m_approx": 30}


def inventory_csvs():
    out = {}
    p = os.path.join(ROOT, "HISTORICAL_WEATHER", "west_bengal_historical_weather_2020_2025.csv")
    pts = set()
    tmin = tmax = None
    n = 0
    for ch in pd.read_csv(p, usecols=["time", "latitude", "longitude"], chunksize=500_000):
        pts |= set(zip(ch.latitude.round(4), ch.longitude.round(4)))
        t = pd.to_datetime(ch.time)
        tmin = t.min() if tmin is None else min(tmin, t.min())
        tmax = t.max() if tmax is None else max(tmax, t.max())
        n += len(ch)
    arr = np.array(sorted(pts))
    out["west_bengal_historical_weather"] = {
        "size_mb": round(os.path.getsize(p) / 1e6, 1), "n_rows": n, "n_points": len(pts),
        "t_min": str(tmin), "t_max": str(tmax), "freq": "hourly",
        "lat_range": [float(arr[:, 0].min()), float(arr[:, 0].max())],
        "lon_range": [float(arr[:, 1].min()), float(arr[:, 1].max())],
        "n_unique_lat": int(len(np.unique(arr[:, 0]))),
        "note": ("incomplete grid download: only 5 latitude rows (21.69-21.97N) of a "
                 "requested WB-wide grid; source is a reanalysis API (ERA5-derived)")}
    p2 = os.path.join(ROOT, "HISTORICAL_WEATHER", "north_24_parganas_historical_weather_2020_2025.csv")
    d2 = pd.read_csv(p2, parse_dates=["time"])
    out["north_24_parganas_historical_weather"] = {
        "size_mb": round(os.path.getsize(p2) / 1e6, 1), "n_rows": len(d2),
        "t_min": str(d2.time.min()), "t_max": str(d2.time.max()), "freq": "hourly",
        "columns": list(d2.columns), "note": "single-point time series"}
    p3 = os.path.join(ROOT, "RAINFALL_DISTRICT", "rainfall_districtwise_daily_imd.csv")
    d3 = pd.read_csv(p3)
    d3.columns = [c.replace("\n", " ").strip() for c in d3.columns]
    dt = pd.to_datetime(d3["Date"], errors="coerce", dayfirst=True)
    out["rainfall_districtwise"] = {
        "size_mb": round(os.path.getsize(p3) / 1e6, 1), "n_rows": len(d3),
        "t_min": str(dt.min()), "t_max": str(dt.max()), "freq": "daily",
        "n_districts": int(d3["District"].nunique()),
        "states": sorted(d3["State"].dropna().unique().tolist()),
        "columns": list(d3.columns)}
    return out


def inventory_radar():
    p = os.path.join(ROOT, "RADAR", "Radar_data.zip")
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        cols = {}
        for n in names:
            df = pd.read_csv(io.BytesIO(z.read(n)), nrows=2)
            cols[n] = [c for c in df.columns if not c.startswith("Unnamed")]
    return {"size_mb": round(os.path.getsize(p) / 1e6, 1), "members": names,
            "header_groups": cols,
            "note": ("publication figure data: GPM vs ground-radar reflectivity calibration "
                     "for Kolkata/Chennai/Machilipatnam/Karaikal. No timestamps and no "
                     "gridded reflectivity fields.")}


def inventory_boundaries():
    d = os.path.join(ROOT, "BOUNDARIES")
    out = {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        e = {"size_mb": round(os.path.getsize(p) / 1e6, 1)}
        if f.endswith(".geojson"):
            with open(p, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
            e["n_features"] = len(gj.get("features", []))
            if gj.get("features"):
                e["properties"] = list(gj["features"][0].get("properties", {}).keys())
        out[f] = e
    return out


def inventory_normal_samples():
    d = os.path.join(ROOT, "NORMAL_SAMPLES")
    out = {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        df = pd.read_csv(p)
        out[f] = {"n_rows": len(df), "columns": list(df.columns)}
        if "timestamp" in df.columns:
            t = pd.to_datetime(df.timestamp)
            out[f].update({"t_min": str(t.min()), "t_max": str(t.max())})
    out["_note"] = ("pre-existing partial labelling attempt by the project author; "
                    "not used - labels are regenerated from scratch by this pipeline")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    inv = {"generated": datetime.now().isoformat(timespec="seconds"), "root": ROOT}
    for name, fn in [("INSAT", inventory_insat), ("ERA5", inventory_era5),
                     ("IMD_RAINFALL", inventory_imd), ("LIGHTNING", inventory_lightning),
                     ("DEM", inventory_dem), ("CSV_PRODUCTS", inventory_csvs),
                     ("RADAR", inventory_radar), ("BOUNDARIES", inventory_boundaries),
                     ("NORMAL_SAMPLES", inventory_normal_samples)]:
        print(f"[inventory] {name} ...", flush=True)
        inv[name] = fn()
    path = os.path.join(OUT, "data_inventory.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2, default=str)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

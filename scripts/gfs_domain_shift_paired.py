"""Paired GFS-vs-ERA5 domain-shift analysis (FAR-reduction phase, section E).

Pairs each cached GFS f000 analysis with the ERA5 field at the SAME valid time,
so differences are model-vs-model bias, not seasonal or sampling differences.
Every statistic is computed on identical (time, lat, lon) cells.

Read-only: touches no checkpoint, no config, no production path.
"""
import os, sys, glob, pickle, json
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("."))
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS, load_stats

CACHE = "processed/cache/era5_memmap"
GFS_DIR = "processed/cache/gfs"
OUT = "reports/gfs_domain_shift_paired.json"

# ERA5 surface variable order as stored in surface.npy
def main():
    times = np.load(os.path.join(CACHE, "valid_time.npy")).astype("datetime64[ns]")
    tindex = {}
    for i, t in enumerate(times):
        dt = datetime.fromtimestamp(t.astype("datetime64[s]").astype(int), tz=timezone.utc)
        tindex[dt] = i
    surface = np.load(os.path.join(CACHE, "surface.npy"), mmap_mode="r")
    pressure = np.load(os.path.join(CACHE, "pressure.npy"), mmap_mode="r")
    levels = list(np.load(os.path.join(CACHE, "pressure_levels_hpa.npy")))
    print("ERA5 surface array:", surface.shape, "pressure:", pressure.shape, "levels:", levels)

    stats = load_stats("Data/processed/cache/norm_stats.json")

    files = sorted(glob.glob(os.path.join(GFS_DIR, "*_0.pkl")))
    pairs = []
    for f in files:
        try:
            c = pickle.load(open(f, "rb"))
        except Exception:
            continue
        ct = c.cycle_time
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        if ct in tindex:
            pairs.append((ct, tindex[ct], c))
    print(f"paired GFS cycles with ERA5 valid times: {len(pairs)} of {len(files)}")
    if not pairs:
        print("no pairs; abort")
        return 1

    res = {"n_pairs": len(pairs),
           "pair_times": [p[0].isoformat() for p in pairs],
           "surface": {}, "pressure": {}}

    for vi, var in enumerate(SINGLE_VARS):
        G, E = [], []
        for ct, ti, c in pairs:
            if var not in c.surface:
                continue
            g = np.asarray(c.surface[var], dtype=np.float64)
            e = np.asarray(surface[ti, vi], dtype=np.float64)
            if g.shape != e.shape:
                continue
            G.append(g.ravel()); E.append(e.ravel())
        if not G:
            continue
        G = np.concatenate(G); E = np.concatenate(E)
        m = np.isfinite(G) & np.isfinite(E)
        G, E = G[m], E[m]
        sd = float(stats[var]["std"]) if var in stats else float(np.std(E))
        bias = float(np.mean(G - E))
        d = {
            "n": int(G.size),
            "era5_mean": float(np.mean(E)), "gfs_mean": float(np.mean(G)),
            "era5_std": float(np.std(E)), "gfs_std": float(np.std(G)),
            "era5_median": float(np.median(E)), "gfs_median": float(np.median(G)),
            "bias_gfs_minus_era5": bias,
            "bias_in_training_sd": float(bias / (sd + 1e-9)),
            "training_sd": sd,
            "mae": float(np.mean(np.abs(G - E))),
            "corr": float(np.corrcoef(G, E)[0, 1]) if G.size > 2 and np.std(G) > 0 and np.std(E) > 0 else None,
            "era5_p": {str(q): float(np.percentile(E, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "gfs_p": {str(q): float(np.percentile(G, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "era5_frac_zero": float(np.mean(E == 0)), "gfs_frac_zero": float(np.mean(G == 0)),
            "gfs_frac_negative": float(np.mean(G < 0)), "era5_frac_negative": float(np.mean(E < 0)),
        }
        res["surface"][var] = d
        print(f"{var:6s} bias={bias:+10.3f} ({d['bias_in_training_sd']:+.3f} sd)  "
              f"ERA5 mu={d['era5_mean']:9.3f}  GFS mu={d['gfs_mean']:9.3f}  corr={d['corr']}")

    # pressure: ERA5 pressure.npy is (T, n_vars, n_levels, H, W)
    for vi, var in enumerate(PRESSURE_VARS):
        res["pressure"][var] = {}
        for li, lvl in enumerate(levels):
            lvl = int(lvl)
            G, E = [], []
            for ct, ti, c in pairs:
                gl = c.pressure.get(var, {})
                if lvl not in gl:
                    continue
                g = np.asarray(gl[lvl], dtype=np.float64)
                e = np.asarray(pressure[ti, vi, li], dtype=np.float64)
                if g.shape != e.shape:
                    continue
                G.append(g.ravel()); E.append(e.ravel())
            if not G:
                continue
            G = np.concatenate(G); E = np.concatenate(E)
            m = np.isfinite(G) & np.isfinite(E)
            G, E = G[m], E[m]
            if G.size == 0:
                continue
            key = f"{var}{lvl}"
            sd = float(stats[key]["std"]) if key in stats else float(np.std(E))
            bias = float(np.mean(G - E))
            res["pressure"][var][str(lvl)] = {
                "n": int(G.size), "era5_mean": float(np.mean(E)), "gfs_mean": float(np.mean(G)),
                "bias_gfs_minus_era5": bias, "bias_in_training_sd": float(bias / (sd + 1e-9)),
                "mae": float(np.mean(np.abs(G - E))),
                "corr": float(np.corrcoef(G, E)[0, 1]) if np.std(G) > 0 and np.std(E) > 0 else None,
            }
            print(f"{var}{lvl:<5d} bias={bias:+10.3f} ({res['pressure'][var][str(lvl)]['bias_in_training_sd']:+.3f} sd) "
                  f"corr={res['pressure'][var][str(lvl)]['corr']}")

    os.makedirs("reports", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nWrote", OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

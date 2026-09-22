"""Label-quality audit of the ERA5 severe-weather proxy (FAR phase, section H).

Read-only. Does NOT change the label definition; it characterises the existing
production target so we can tell how much of the measured false-alarm rate is
attributable to the proxy rather than to the model.

Production proxy (configs: labels.*):
    severe = (rolling 3h precip > 15 mm) OR (CAPE > 2000 AND CIN < 50)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from src.features.normalize import SINGLE_VARS

CACHE = "processed/cache/era5_memmap"
OUT = "reports/label_quality_audit.json"

def main():
    sev = np.load(os.path.join(CACHE, "target_severe_weather.npy"), mmap_mode="r")
    rain = np.load(os.path.join(CACHE, "target_rain_3h_mm.npy"), mmap_mode="r")
    heavy = np.load(os.path.join(CACHE, "target_heavy_rain.npy"), mmap_mode="r")
    conv = np.load(os.path.join(CACHE, "target_severe_convective.npy"), mmap_mode="r")
    surf = np.load(os.path.join(CACHE, "surface.npy"), mmap_mode="r")
    T, H, W = sev.shape
    print(f"target array: {sev.shape}")

    # Subsample time for the heavier spatial stats; full pass for prevalence.
    res = {}
    s_all = np.asarray(sev[:], dtype=np.uint8)
    res["prevalence_overall"] = float(s_all.mean())
    print(f"overall prevalence: {res['prevalence_overall']:.4f}")

    h_all = np.asarray(heavy[:], dtype=np.uint8)
    c_all = np.asarray(conv[:], dtype=np.uint8)
    both = (h_all == 1) & (c_all == 1)
    only_rain = (h_all == 1) & (c_all == 0)
    only_conv = (h_all == 0) & (c_all == 1)
    npos = max(int(s_all.sum()), 1)
    res["component_share"] = {
        "heavy_rain_prevalence": float(h_all.mean()),
        "severe_convective_prevalence": float(c_all.mean()),
        "positives_from_rain_only": float(only_rain.sum() / npos),
        "positives_from_cape_cin_only": float(only_conv.sum() / npos),
        "positives_from_both": float(both.sum() / npos),
    }
    print("component share of positives:", json.dumps(res["component_share"], indent=2))

    # How intense are the CAPE/CIN-only positives? If they carry little or no
    # actual rain, the label is flagging "environment supportive of convection"
    # rather than "severe weather happened" -- which puts irreducible false-alarm
    # pressure on any model scored against observed impact.
    r_all = np.asarray(rain[:], dtype=np.float32)
    for name, mask in (("rain_only", only_rain), ("cape_cin_only", only_conv), ("both", both)):
        if mask.sum() == 0:
            continue
        rr = r_all[mask]
        res.setdefault("rain_within_positive_class", {})[name] = {
            "n": int(mask.sum()),
            "mean_rain_3h_mm": float(rr.mean()),
            "median_rain_3h_mm": float(np.median(rr)),
            "frac_below_1mm": float(np.mean(rr < 1.0)),
            "frac_below_5mm": float(np.mean(rr < 5.0)),
            "frac_at_or_above_15mm": float(np.mean(rr >= 15.0)),
        }
        print(f"  {name}: n={int(mask.sum())} mean_rain={rr.mean():.2f}mm "
              f"frac<1mm={np.mean(rr<1.0):.3f} frac>=15mm={np.mean(rr>=15.0):.3f}")

    # Spatial coherence: isolated single-cell positives are far more likely to be
    # label noise than a coherent storm area.
    step = max(1, T // 400)
    iso, tot, sizes = 0, 0, []
    for t in range(0, T, step):
        f = s_all[t]
        if f.sum() == 0:
            continue
        pad = np.pad(f, 1)
        neigh = sum(pad[1+dy:1+dy+H, 1+dx:1+dx+W]
                    for dy in (-1,0,1) for dx in (-1,0,1)) - f
        iso += int(((f == 1) & (neigh == 0)).sum())
        tot += int(f.sum())
        sizes.append(int(f.sum()))
    res["spatial_coherence"] = {
        "frames_sampled": len(sizes),
        "positive_cells_sampled": tot,
        "frac_isolated_single_cell": float(iso / tot) if tot else None,
        "mean_positive_cells_per_active_frame": float(np.mean(sizes)) if sizes else None,
        "median_positive_cells_per_active_frame": float(np.median(sizes)) if sizes else None,
    }
    print("spatial coherence:", json.dumps(res["spatial_coherence"], indent=2))

    # Temporal persistence at a fixed cell: how long does a positive last?
    ci, cj = H // 2, W // 2
    series = s_all[:, ci, cj]
    runs, cur = [], 0
    for v in series:
        if v: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    res["temporal_persistence_center_cell"] = {
        "n_events": len(runs),
        "mean_duration_hours": float(np.mean(runs)) if runs else None,
        "median_duration_hours": float(np.median(runs)) if runs else None,
        "frac_single_hour_events": float(np.mean(np.array(runs) == 1)) if runs else None,
    }
    print("temporal persistence:", json.dumps(res["temporal_persistence_center_cell"], indent=2))

    os.makedirs("reports", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nWrote", OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

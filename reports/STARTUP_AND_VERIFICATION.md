# StormSense — how to run it, and how to check it is telling the truth

## Starting the system

### Mode A — combined (one command)

```
python run_server.py
```

Dashboard and API on `http://127.0.0.1:8000`. Same-origin, so no CORS involved.

### Mode B — split frontend/backend

```
# terminal 1
python run_backend.py          # API on :8000

# terminal 2
python run_frontend.py         # dashboard on :3000
```

Both modes import the **same** `backend/main.py` app object, so the ML, GFS and
historical services have exactly one implementation.

Useful environment variables:

| variable | purpose |
|---|---|
| `STORMSENSE_DEVICE` | set `cpu` on memory-constrained machines |
| `STORMSENSE_CORS_ORIGINS` | comma-separated origin allowlist for split mode |
| `API_BASE_URL` | backend the served page should target |
| `STORMSENSE_CKPT` / `STORMSENSE_CONFIG` | drive a **candidate** checkpoint through the production path (offline evaluation only; unset in normal operation) |

The frontend resolves its API base through `frontend/js/config.js` — window
global, then `<meta>`, then `?api=`, then same-origin, then a dev-port fallback.
No hardcoded production URL.

---

## Verifying the temporal claims

The single most important thing to check is that a horizon label means what it
says. Ask the backend directly:

```
curl "http://127.0.0.1:8000/api/nowcast/wallclock-horizons"
```

For each of NOW/+2/+4/+6 this reports the true target time, the model lead
genuinely required (`analysis_age + horizon`), and either the lead that satisfies
it or **why no lead does**. A horizon marked `available: false` is meant to be
shown as unavailable — never back-filled with a neighbouring lead.

Worked example from a real run (2026-09-21, production v2 whose leads are 2–6h):

```
analysis 12:00Z, wall clock 18:02Z, age 6.03h
  +0h  OK    lead 6h, served 18:00Z, label error -0.03h
  +2h  NO    needs lead  8.03h  (model forecasts out to 6h)
  +4h  NO    needs lead 10.03h
  +6h  NO    needs lead 12.03h
```

That is the honest state of the **production** checkpoint: it can serve NOW, and
cannot serve wall-clock +2/+4/+6, because reaching two hours past *now* from a
six-hour-old analysis needs an eight-hour forecast.

### Measuring GFS latency yourself

```
python - <<'EOF'
import httpx, datetime as dt
from datetime import datetime, timezone, timedelta
from src.inference.gfs_live import _idx_url
c = httpx.Client(timeout=20.0, follow_redirects=True)
now = datetime.now(timezone.utc)
b = now.replace(minute=0, second=0, microsecond=0); b -= timedelta(hours=b.hour % 6)
for k in range(1, 7):
    cyc = b - timedelta(hours=6*k)
    r = c.head(_idx_url(cyc, 0)); lm = r.headers.get('last-modified')
    if lm:
        pub = dt.datetime.strptime(lm, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)
        print(cyc.strftime('%Y-%m-%d %HZ'), 'publish lag %.2fh' % ((pub-cyc).total_seconds()/3600))
EOF
```

Measured 2026-09-21 over 12 consecutive cycles: mean **3.60 h**, min 3.54, max 3.81.

---

## Test suite

```
python -m pytest tests/ -q --ignore=tests/test_browser_e2e.py
```

397 tests pass (measured 2026-09-21). The browser suite
(`tests/test_browser_e2e.py`) needs Playwright and is excluded from the default
run.

Tests that specifically pin the claims in this document:

| file | pins |
|---|---|
| `tests/test_target_time.py` | a horizon is served only when a lead is genuinely valid at `now + horizon`; hourly leads cover a whole cycle; coarse leads do not |
| `tests/test_frontend_offset_label.py` | "FROM NOW" is computed from the valid time and can render a negative offset |
| `tests/test_multi_location.py` | 7 WB locations → 7 distinct cells; risk is not broadcast; XAI is location-specific |
| `tests/test_temporal_leakage.py` | no cycle later than the simulated NOW is ever consumed |
| `tests/test_api.py` | `required_lead_hours == analysis_age + horizon` for every horizon |

---

## Reproducing the analyses

```
# paired GFS vs ERA5 input distributions (54 cycles, identical valid times)
python scripts/gfs_domain_shift_paired.py

# what the severe-weather proxy label actually contains
python scripts/label_quality_audit.py

# held-out GFS backtest (91h stride -- NOT a multiple of 24)
STORMSENSE_DEVICE=cpu \
STORMSENSE_CKPT=Data/outputs/checkpoints_v3/v2_best.pt \
STORMSENSE_CONFIG=configs/v3_longlead.yaml \
python scripts/historical_backtest.py --max-cases 14 --step-hours 91 \
  --out reports/gfs_backtest_v3_harmfix.json --dump-raw /tmp/raw.npz

# dense threshold sweep + reliability diagram on those same arrays
python scripts/far_analysis.py --raw /tmp/raw.npz --out reports/far_sweep.json

# GFS-domain recalibration: fit on 2023, report on untouched 2024
python scripts/calibrate_gfs_domain.py --val-raw /tmp/raw_val2023.npz \
  --test-raw /tmp/raw.npz --out reports/gfs_domain_calibration.json

# fair multi-candidate comparison (refuses a diurnally-aliased stride)
python scripts/compare_candidates.py \
  --candidate "v3:Data/outputs/checkpoints_v3/v2_best.pt:configs/v3_longlead.yaml" \
  --candidate "v4:Data/outputs/checkpoints_v4/v2_best.pt:configs/v4_wallclock.yaml" \
  --max-cases 14 --out reports/candidate_comparison.json
```

`--step-hours` must not be a multiple of 24. A stride of 72 or 96 pins every
simulated NOW to the same hour of day, aliases the diurnal convection cycle, and
**inverts** the per-lead skill ranking. `compare_candidates.py` refuses such a
stride outright.

---

## What the numbers mean, stated plainly

* **FAR cannot be pushed below ~0.80** at any useful detection rate. `FAR = 1 −
  precision`; the base rate is ≈ 0.065, and the model's genuine 2.2–3.0×
  precision lift caps precision at 0.14–0.20. This is arithmetic, not a tuning
  failure, and a dense sweep of every threshold 0.01–0.99 confirms it.
* **Calibration fixes reliability, not skill.** The GFS-domain recalibration
  improves ECE 6–23× and Brier ≈ 45 % out-of-sample, so a displayed "30 %"
  genuinely means 30 %. It is monotonic, so PR-AUC is unchanged to six decimals
  and FAR at matched POD does not move.
* **The severe-weather target is a proxy.** 77.7 % of its positives come from the
  CAPE/CIN branch alone and 56.9 % of all positives carry under 1 mm of rain, so
  it largely flags an environment supportive of convection rather than severe
  weather that occurred.
* **Live probabilities are directionally informative.** Inputs are GFS, training
  was ERA5, and measured live skill is roughly half the ERA5-input figure.

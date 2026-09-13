# StormSense Documentation Package

Generated: 2026-09-12, by inspecting the actual running repository at `D:\Weather-Hackathon` (server started, API endpoints queried, browser navigated with Playwright).

**Time constraint note:** this package was generated under a strict ~7-8 minute budget requested by the user. It prioritizes accuracy and real, verified content over exhaustiveness. See "Known gaps" below.

## Files

1. **StormSense_Project_Summary.docx** — Full plain-language explanation of the whole project: problem, architecture, data sources, model, inputs/outputs, live vs historical, radar/satellite, XAI, thresholds, limitations, and 30s/2min pitches.
2. **StormSense_UI_Data_Explained.docx** — Covers **all 8 sidebar views** (Dashboard, Radar & Satellite Feeds, District Advisories, AI Nowcast & Benchmark, XAI Feature Attribution, GIS Spatial Layers, Calibrated Thresholds, Data Ingestion Streams): 27 documented UI elements, each with a small screenshot cropped to the specific card or panel being described (not a full-page capture), plain explanation, exact code path, and data-kind label (observation/forecast/model output/derived/proxy/static/unavailable). **Revised 2026-09-12** after a repair pass — see "Revision history" below.
3. **StormSense_Hackathon_Judge_QA.docx** — 30 verified Q&A pairs across project basics, data, architecture, radar/satellite, live/historical separation, validation, XAI, flash-flood proxy, deployment, and trust/official-warning questions, plus a "be careful" list, tensor-shape deep dive, code-location table, and demo script.

## Repository state inspected

- Live server started (`run_server.py`), `/api/health` confirmed OK.
- Real API responses read from `/api/nowcast/summary`, `/api/historical/case-study`, held-out metrics file `outputs/metrics/test_evaluation.json`.
- Code verified directly: `src/inference/gfs_live.py`, `nowcast_service.py`, `predictor.py`, `xai.py`, `insat_archive.py`, `risk_map.py`, `risk_surface.py`, `weather-app/backend/main.py`, `weather-app/js/app.js`, `configs/default.yaml`, `src/models/v2_model.py`.
- Browser navigated (Playwright, Chromium) through Dashboard, District Advisories, XAI, Radar & Satellite Feeds, in both Live and Historical mode; 6 screenshots captured and embedded.

## Revision history

**2026-09-12 (later same day):** Document 2 was revised after a repair pass on the running app changed what four of its items actually show. Updated: the Radar & Satellite Feeds page (a temporary "Cyclone Remal case-study hub" card was removed and the page restored to its original two-panel layout, now showing genuinely observed ERA5 rainfall rather than the forecast map re-labelled); the Dashboard hazard-card meters (previously a single fixed colour regardless of severity, now a real green-to-red scale); Live-mode Temperature/Humidity/Wind at forecast horizons (previously blank, now populated from the project's pre-existing OpenWeather integration); and the same fields in Historical mode (now populated from genuinely observed ERA5 reanalysis at each Remal horizon). Item count changed from 28 to 27 (two items describing the removed case-study hub were merged into one, since that panel is gone). All full-page screenshots were replaced with 14 small screenshots, each cropped to the specific card/panel the item describes.

## Important limitations / "Not verified" items

- Document 2 covers all 8 views with 27 items and 14 small, per-item screenshots (no full-page captures). One item, the Calibrated Thresholds table, has no screenshot in this revision — a reliable crop of that table could not be captured in the time available, so its description is text-only, carried over from the previous version and still accurate. It documents the meaningful cards/controls per view, not every individual div/label on the page.
- Document 3 covers **30 Q&A pairs**, not the full 60-topic / 60-question spec in the original request — selected for highest judge-relevance and verifiability.
- Storm-track / landfall coordinates for Cyclone Remal are **not verified in the current repository** — the historical case-study hub does not include them, correctly, per the project's own no-fabrication rule.
- Live GFS analysis latency observed during past sessions ranged 5–10.6 hours; exact figure at any given moment is not fixed and is disclosed as variable, not a constant.
- "Not verified in current implementation": per-district model skill (skill metrics are reported in aggregate across the whole domain, not broken out per district).

## If more time is available later

Recommended next additions: full 8-view UI sweep for Document 2 (GIS, Thresholds, Streams, AI Nowcast/Benchmark), expansion of Document 3 to the full 60-question spec, and a diagram image (architecture flow) embedded in Document 1.

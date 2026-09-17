# Google Maps version — snapshot

Snapshot of the working Google Maps 2D implementation, taken 2026-09-17.

These files are a **safety copy**, not part of the running app. The live files
are the ones under `frontend/` and `src/`.

## What is here

| File | Belongs at |
|---|---|
| `index.html` | `frontend/index.html` |
| `app.js` | `frontend/js/app.js` |
| `styles.css` | `frontend/css/styles.css` |
| `risk_surface.py` | `src/inference/risk_surface.py` |
| `main.py` | `backend/main.py` |

## What this version contains

- Google Maps 2D base map (Leaflet removed; `window.L` is a compatibility shim
  inside `app.js` that translates Leaflet calls to Google Maps)
- Contour-style risk rendering (500x660 render grid in `risk_surface.py`)
- Map interaction fixes (drag / wheel zoom / click popup over the risk overlay)
- NOW = model lead 0; horizon synchronisation across cards, districts and XAI
- Google Maps browser key loaded from `.env` via `_google_maps_key()` in
  `main.py` (accepts `GOOGLEMAPS_API_KEY` or `GOOGLE_MAPS_API_KEY`)

## Restoring this version

```bash
cp backups/google-maps-version/index.html      frontend/index.html
cp backups/google-maps-version/app.js          frontend/js/app.js
cp backups/google-maps-version/styles.css      frontend/css/styles.css
cp backups/google-maps-version/risk_surface.py src/inference/risk_surface.py
cp backups/google-maps-version/main.py         backend/main.py
```

Then restart the server and hard-refresh the browser (Ctrl+Shift+R).

## Going back to the OLD Leaflet map instead

The pre-Google Leaflet version is committed in git at `59d5f53`. It is complete
and self-consistent (real Leaflet 1.9.4 from CDN). To restore just the frontend:

```bash
git checkout 59d5f53 -- frontend/index.html frontend/js/app.js
```

Note: that also reverts every frontend fix made after the migration, because
they were written against the Google shim. The backend and model work
(`src/inference/`, `backend/main.py`) is independent and is NOT reverted by the
command above — keep those unless you specifically want them rolled back too.

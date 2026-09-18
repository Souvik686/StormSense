"""Radar echo sampling for the NOW observation-evidence layer.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
This module samples the SAME RainViewer composite the Radar view displays, onto
the model's 33x25 grid, so the NOW panel can state what is being OBSERVED at the
same cells the model predicts for.

It does NOT convert echo into a severe-weather probability, and nothing here is
allowed to modify the model's calibrated probability. That restraint is a
measured result, not caution for its own sake:

  * Over West Bengal, RainViewer echo fraction and the provider's station
    rain_1h correlate at r = 0.104 (n = 48 grid points, 2026-09-18 17:10Z).
  * Echo covered ~1.1% of the state, and every echo pixel sampled fell in the
    LOWEST reflectivity band of RainViewer colour scheme 2 (light blue), i.e.
    drizzle/light rain, not convective cores.
  * There is no gridded severe-weather verification dataset in this repository
    against which a radar -> risk transfer function could be fitted or checked.

Any weight mapping echo onto probability would therefore be invented. Instead
this module reports echo as its own quantity, with its own timestamp, and the
NOW payload presents model and observation side by side so a disagreement is
visible rather than silently averaged away.

WHAT THE NUMBER MEANS
---------------------
RainViewer tiles carry no numeric dBZ: the COLOUR is the value. We therefore
report `echo_fraction` -- the fraction of non-transparent pixels in a small
window centred on each grid cell -- which is an honest measure of "is the radar
painting precipitation here", and NOT a reflectivity or a rain rate. Converting
the colour back to dBZ would be an inverse-palette guess, so it is not done.

COVERAGE vs SILENCE
-------------------
An all-transparent tile is ambiguous on its own: it can mean "no precipitation"
or "no radar coverage". RainViewer serves a ~334-byte placeholder PNG where it
has nothing at all (verified against the Sahara and the mid-Pacific) and a
substantially larger tile where it has a real mosaic (3376 bytes over West
Bengal in the same frame). We use tile payload size as the coverage signal and
report `coverage: True/False` rather than silently reading "no coverage" as
"no rain".
"""
from __future__ import annotations

import io
import json
import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RAINVIEWER_INDEX = "https://api.rainviewer.com/public/weather-maps.json"

# Zoom level used for sampling. z=7 gives ~1.2 km/pixel at this latitude, which
# is finer than the 0.25 deg (~28 km) model cell, so each cell is summarised
# from many radar pixels rather than a single sample. RainViewer publishes radar
# only to maxNativeZoom 7; requesting deeper would just be upscaling.
SAMPLE_ZOOM = 7

# Half-width, in tile pixels, of the window summarised per model cell. At z=7 a
# 0.25 deg cell spans roughly 22 px, so +/-11 px covers the cell without
# bleeding far into its neighbours.
CELL_HALF_PX = 11

# Tiles at or below this size are RainViewer's "nothing here" placeholder. Real
# mosaic tiles are several KB. Measured: 334 B over the Sahara and mid-Pacific
# (no coverage) vs 3376 B over West Bengal in the same frame.
EMPTY_TILE_MAX_BYTES = 400

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"frame_time": None, "grid": None, "fetched_at": 0.0,
                          "coverage": None, "error": None, "frame_path": None}
# One radar frame is published every ~10 min; re-sampling faster than that
# cannot produce a new value and only costs tile requests.
CACHE_TTL_SECONDS = 240


@dataclass
class RadarEchoField:
    """Echo sampled onto the model grid, with its own provenance."""
    echo_fraction: np.ndarray      # (H, W) in [0,1]; NaN where no coverage
    coverage: np.ndarray           # (H, W) bool -- radar actually reports here
    frame_time_utc: str            # the radar scan time (NOT wall clock)
    frame_age_seconds: int
    source: str
    zoom: int
    note: str

    def to_summary(self) -> Dict[str, Any]:
        valid = self.echo_fraction[np.isfinite(self.echo_fraction)]
        return {
            "source": self.source,
            "frame_time_utc": self.frame_time_utc,
            "frame_age_seconds": self.frame_age_seconds,
            "sample_zoom": self.zoom,
            "cells_with_coverage": int(self.coverage.sum()),
            "cells_total": int(self.coverage.size),
            "cells_with_echo": int((valid > 0).sum()) if valid.size else 0,
            "max_echo_fraction": round(float(valid.max()), 4) if valid.size else None,
            "mean_echo_fraction": round(float(valid.mean()), 4) if valid.size else None,
            "is_forecast": False,
            "is_model_input": False,
            "quantity": "echo_fraction",
            "note": self.note,
        }


def _deg_to_pixel(lat: float, lon: float, z: int) -> Tuple[float, float]:
    """Web-Mercator lat/lon -> global pixel coordinates at zoom z."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat_r = math.radians(max(min(lat, 85.05), -85.05))
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n * 256.0
    return x, y


def _latest_frame(timeout: float = 20.0) -> Tuple[str, str, datetime]:
    """(host, path, scan_time) of the newest published radar frame."""
    with urllib.request.urlopen(RAINVIEWER_INDEX, timeout=timeout) as r:
        idx = json.loads(r.read().decode("utf-8"))
    past = (idx.get("radar") or {}).get("past") or []
    if not past:
        raise RuntimeError("RainViewer published no radar frames.")
    newest = past[-1]
    return idx["host"], newest["path"], datetime.fromtimestamp(newest["time"], timezone.utc)


def sample_echo_to_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    timeout: float = 20.0,
    use_cache: bool = True,
) -> RadarEchoField:
    """Sample the newest RainViewer frame onto the (len(lats), len(lons)) grid.

    Raises on network/index failure so the caller can report the radar layer as
    unavailable. It never substitutes zeros for missing radar: absent data and
    observed "no echo" are different facts and are kept distinct via `coverage`.
    """
    from PIL import Image

    host, path, frame_time = _latest_frame(timeout=timeout)
    now = time.time()

    with _CACHE_LOCK:
        if (use_cache and _CACHE["grid"] is not None
                and _CACHE["frame_path"] == path
                and now - _CACHE["fetched_at"] < CACHE_TTL_SECONDS):
            # Same published frame -> identical pixels; only the age moves on.
            return RadarEchoField(
                echo_fraction=_CACHE["grid"],
                coverage=_CACHE["coverage"],
                frame_time_utc=frame_time.isoformat(),
                frame_age_seconds=int(now - frame_time.timestamp()),
                source="RainViewer composite radar mosaic",
                zoom=SAMPLE_ZOOM,
                note=_NOTE,
            )

    H, W = len(lats), len(lons)
    echo = np.full((H, W), np.nan, dtype=np.float32)
    cover = np.zeros((H, W), dtype=bool)

    tiles: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

    def tile(tx: int, ty: int) -> Optional[np.ndarray]:
        """Fetch one tile; None means RainViewer has no data there."""
        key = (tx, ty)
        if key in tiles:
            return tiles[key]
        url = f"{host}{path}/256/{SAMPLE_ZOOM}/{tx}/{ty}/2/1_1.png"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                raw = r.read()
            if len(raw) <= EMPTY_TILE_MAX_BYTES:
                tiles[key] = None          # placeholder tile => no coverage
            else:
                tiles[key] = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
        except Exception:
            tiles[key] = None
        return tiles[key]

    n_px = 2 ** SAMPLE_ZOOM * 256
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            px, py = _deg_to_pixel(float(la), float(lo), SAMPLE_ZOOM)
            x0, x1 = int(px) - CELL_HALF_PX, int(px) + CELL_HALF_PX + 1
            y0, y1 = int(py) - CELL_HALF_PX, int(py) + CELL_HALF_PX + 1
            hit = 0
            tot = 0
            for gx in range(x0, x1):
                for gy in range(y0, y1):
                    if gx < 0 or gy < 0 or gx >= n_px or gy >= n_px:
                        continue
                    arr = tile(gx // 256, gy // 256)
                    if arr is None:
                        continue
                    tot += 1
                    if arr[gy % 256, gx % 256, 3] > 0:
                        hit += 1
            if tot > 0:
                cover[i, j] = True
                echo[i, j] = hit / float(tot)

    with _CACHE_LOCK:
        _CACHE.update({"grid": echo, "coverage": cover, "fetched_at": time.time(),
                       "frame_path": path, "frame_time": frame_time.isoformat(),
                       "error": None})

    return RadarEchoField(
        echo_fraction=echo,
        coverage=cover,
        frame_time_utc=frame_time.isoformat(),
        frame_age_seconds=int(time.time() - frame_time.timestamp()),
        source="RainViewer composite radar mosaic",
        zoom=SAMPLE_ZOOM,
        note=_NOTE,
    )


_NOTE = (
    "Observed radar echo coverage, sampled from the same RainViewer composite the "
    "Radar view displays. This is a precipitation-echo observation, not a "
    "reflectivity value, not a rain rate, and not a severe-weather probability. "
    "It is NOT an input to SevereWeatherNetV2 and never modifies the model's "
    "calibrated probability."
)

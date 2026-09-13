"""Archived INSAT-3DR imagery for the historical case study.

SCIENTIFIC CONTRACT
-------------------
Everything served from here is GENUINE archived ISRO/SAC INSAT-3DR Level-2B
product data read from the HDF5 files in Data/INSAT/. Nothing is synthesized,
interpolated from a picture, or reconstructed from a rendering.

The archive is EPISODIC: it covers selected severe-convective periods, not a
continuous record. For Cyclone Remal the nearest available acquisitions are on
27 May 2024 (post-landfall), NOT the 26 May 12:00 UTC analysis time of the case
study. That gap is reported to the caller as `offset_hours` and must be shown in
the UI rather than hidden -- relabelling a 27 May acquisition as a 26 May
observation would be fabricating an observation timestamp.

If no acquisition exists near a requested time, this module returns
`status: "unavailable"` with the reason. It never substitutes another date's
imagery to fill the panel.
"""
from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSAT_ROOT = os.path.join(PROJECT_ROOT, "Data", "INSAT")

# Channel -> (subdirectory, HDF5 dataset name, human label, units)
CHANNELS: Dict[str, Dict[str, str]] = {
    "HEM": {"dir": "HEM", "dataset": "HEM", "label": "Hydro-Estimator Precipitation", "units": "mm/hr"},
    "CTP": {"dir": "CTP", "dataset": "CTP", "label": "Cloud Top Pressure", "units": "hPa"},
    "UTH": {"dir": "UTH", "dataset": "UTH", "label": "Upper Tropospheric Humidity", "units": "%"},
    "CMK": {"dir": "CMK", "dataset": "CMK", "label": "Cloud Mask", "units": "category"},
}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
)}
_NAME_RE = re.compile(r"3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_L2B_([A-Z]+)_")

# West Bengal domain, matching the model grid extent.
WB_LAT_MIN, WB_LAT_MAX = 20.0, 28.0
WB_LON_MIN, WB_LON_MAX = 84.0, 90.0


def _parse_acquisition(filename: str) -> Optional[datetime]:
    """Acquisition instant encoded in the ISRO product filename (UTC)."""
    m = _NAME_RE.search(filename)
    if not m:
        return None
    day, mon, year, hhmm, _chan = m.groups()
    if mon not in _MONTHS:
        return None
    return datetime(int(year), _MONTHS[mon], int(day),
                    int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)


def list_acquisitions(channel: str) -> List[Dict[str, Any]]:
    """Every archived acquisition for a channel, sorted by time."""
    spec = CHANNELS.get(channel.upper())
    if spec is None:
        return []
    d = os.path.join(INSAT_ROOT, spec["dir"])
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if not fn.endswith(".h5"):
            continue
        t = _parse_acquisition(fn)
        if t is not None:
            out.append({"file": fn, "path": os.path.join(d, fn), "time_utc": t})
    return sorted(out, key=lambda x: x["time_utc"])


def nearest_acquisition(channel: str, target: datetime,
                        max_offset_hours: float = 48.0) -> Optional[Dict[str, Any]]:
    """Closest archived acquisition to `target`, or None if nothing is near.

    `max_offset_hours` is deliberately generous because the archive is episodic,
    but the actual offset is always returned so the UI can state it plainly.
    """
    acqs = list_acquisitions(channel)
    if not acqs:
        return None
    best = min(acqs, key=lambda a: abs((a["time_utc"] - target).total_seconds()))
    offset_h = (best["time_utc"] - target).total_seconds() / 3600.0
    if abs(offset_h) > max_offset_hours:
        return None
    best = dict(best)
    best["offset_hours"] = round(offset_h, 2)
    return best


def read_wb_window(path: str, channel: str) -> Dict[str, Any]:
    """Read one product and clip it to the West Bengal domain.

    Fill values are masked to NaN rather than coerced to zero: an off-disk or
    unretrieved pixel is missing data, not a measurement of zero rainfall.
    """
    import h5py

    spec = CHANNELS[channel.upper()]
    with h5py.File(path, "r") as h:
        ds = h[spec["dataset"]]
        arr = ds[0] if ds.ndim == 3 else ds[:]
        arr = np.asarray(arr, dtype=np.float32)
        fill = ds.attrs.get("_FillValue")
        if fill is not None:
            arr = np.where(arr == np.asarray(fill).ravel()[0], np.nan, arr)

        lat_ds, lon_ds = h["Latitude"], h["Longitude"]
        lat = np.asarray(lat_ds[:], dtype=np.float32)
        lon = np.asarray(lon_ds[:], dtype=np.float32)
        # int16 storage with a documented scale factor; honour it rather than
        # assuming 0.01.
        lat *= float(np.asarray(lat_ds.attrs.get("scale_factor", 0.01)).ravel()[0])
        lon *= float(np.asarray(lon_ds.attrs.get("scale_factor", 0.01)).ravel()[0])

    inside = (
        (lat >= WB_LAT_MIN) & (lat <= WB_LAT_MAX)
        & (lon >= WB_LON_MIN) & (lon <= WB_LON_MAX)
    )
    if not inside.any():
        return {"ok": False, "reason": "Product does not cover the West Bengal domain."}

    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return {
        "ok": True,
        "values": arr[r0:r1, c0:c1],
        "lat": lat[r0:r1, c0:c1],
        "lon": lon[r0:r1, c0:c1],
        "units": spec["units"],
        "label": spec["label"],
    }


# Maximum offset at which an archived acquisition may be presented as
# CONTEMPORANEOUS with a requested target time. The INSAT archive is episodic:
# for Cyclone Remal the nearest acquisitions are 21-35 h after the case study's
# horizons (27 May, by which time the system had moved inland and weakened), so
# at this threshold the case study correctly reports "unavailable" rather than
# showing day-after imagery beside a 26 May label.
CONTEMPORANEOUS_MAX_OFFSET_HOURS = 3.0


def describe_for_target(channel: str, target: datetime) -> Dict[str, Any]:
    """Archive availability for one channel at one target time.

    Returns a payload the UI can render truthfully in every case: genuine
    imagery when an acquisition is genuinely contemporaneous, and an explicit
    unavailability reason (naming the nearest acquisition and its offset) when
    it is not. Never substitutes a distant acquisition for a missing one.
    """
    spec = CHANNELS.get(channel.upper())
    if spec is None:
        return {"status": "unavailable", "channel": channel,
                "reason": f"Unknown channel {channel!r}."}

    nearest = nearest_acquisition(channel, target)
    if nearest is None:
        return {
            "status": "unavailable",
            "channel": channel.upper(),
            "label": spec["label"],
            "target_utc": target.isoformat(),
            "reason": ("No archived INSAT-3DR acquisition exists near this time. "
                       "The archive is episodic, not continuous."),
        }

    offset = float(nearest["offset_hours"])
    base = {
        "channel": channel.upper(),
        "label": spec["label"],
        "units": spec["units"],
        "target_utc": target.isoformat(),
        "nearest_acquisition_utc": nearest["time_utc"].isoformat(),
        "offset_hours": offset,
        "product_file": nearest["file"],
        "source": "ISRO/SAC INSAT-3DR Level-2B archive (genuine acquisition)",
        "is_live": False,
        "is_model_output": False,
    }

    if abs(offset) > CONTEMPORANEOUS_MAX_OFFSET_HOURS:
        base["status"] = "unavailable"
        base["reason"] = (
            f"No INSAT-3DR acquisition within "
            f"{CONTEMPORANEOUS_MAX_OFFSET_HOURS:.0f} h of this target. The nearest "
            f"archived acquisition is {nearest['time_utc'].strftime('%d %b %Y %H:%M UTC')}, "
            f"{abs(offset):.1f} h {'after' if offset > 0 else 'before'} the target -- "
            f"too distant to represent conditions at this time, so it is not shown."
        )
        return base

    base["status"] = "ok"
    base["provenance_note"] = (
        f"Genuine archived INSAT-3DR {spec['label']} acquired "
        f"{nearest['time_utc'].strftime('%d %b %Y %H:%M UTC')} "
        f"({offset:+.2f} h from the selected target)."
    )
    return base

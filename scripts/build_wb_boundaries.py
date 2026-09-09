"""Regenerate complete West Bengal boundary GeoJSONs from the bundled national
ADM2 dataset (Data/BOUNDARIES/IND_ADM2.zip, Datameet, CC-BY-2.0, 2011 census).

WHY THIS EXISTS: the previously shipped Data/BOUNDARIES/west_bengal.geojson and
west_bengal_districts.geojson contained only 13 of West Bengal's 23 districts.
Darjeeling, Kalimpong, Cooch Behar, Purulia, Howrah, Hooghly, Purba/Paschim
Bardhaman, Jhargram and Paschim Medinipur were absent entirely, so the forecast
raster mask derived from that file had large holes and physically could not
render the whole state -- northern (Darjeeling/Kalimpong) and western (Purulia)
West Bengal were being clipped away.

The bundled ADM2 file uses 2011 census boundaries, in which several modern
districts had not yet been split out:
    Kalimpong           was part of  Darjiling
    Alipurduar          was part of  Jalpaiguri
    Jhargram            was part of  Pashchim Medinipur
    Purba/Paschim Bardhaman  were one undivided  Barddhaman
Those 19 source polygons therefore cover exactly the same land area as today's
23 districts; only the internal subdivision differs. For a state-wide forecast
mask this is exactly what we need, and it is stated here so nobody later
mistakes the 19-feature district file for missing data.

Run:  python scripts/build_wb_boundaries.py
Writes (does NOT overwrite the originals):
    Data/BOUNDARIES/west_bengal_full.geojson            (dissolved state outline)
    Data/BOUNDARIES/west_bengal_districts_full.geojson  (19 district polygons)
"""
from __future__ import annotations

import json
import os
import zipfile

from shapely.geometry import shape, mapping
from shapely.ops import unary_union

BOUNDARIES_DIR = os.path.join("Data", "BOUNDARIES")
SOURCE_ZIP = os.path.join(BOUNDARIES_DIR, "IND_ADM2.zip")
STATE_OUT = os.path.join(BOUNDARIES_DIR, "west_bengal_full.geojson")
DISTRICTS_OUT = os.path.join(BOUNDARIES_DIR, "west_bengal_districts_full.geojson")

# Exact ADM2 "Name" values for West Bengal in the 2011 Datameet dataset. Matched
# by exact name (not bounding box) so a neighbouring Bihar/Jharkhand/Odisha
# district can never be pulled in by accident.
WB_ADM2_NAMES = [
    "Bankura",
    "Barddhaman",
    "Birbhum",
    "Dakshin Dinajpur",
    "Darjiling",
    "Haora",
    "Hugli",
    "Jalpaiguri",
    "Koch Bihar",
    "Kolkata",
    "Maldah",
    "Murshidabad",
    "Nadia",
    "North 24 Parganas",
    "Pashchim Medinipur",
    "Purba Medinipur",
    "Puruliya",
    "South 24 Parganas",
    "Uttar Dinajpur",
]

# Present-day district names, for display. Where 2011 boundaries bundle several
# modern districts together, all are listed so the UI can say what a polygon covers.
MODERN_EQUIVALENT = {
    "Darjiling": "Darjeeling & Kalimpong",
    "Jalpaiguri": "Jalpaiguri & Alipurduar",
    "Pashchim Medinipur": "Paschim Medinipur & Jhargram",
    "Barddhaman": "Purba & Paschim Bardhaman",
    "Koch Bihar": "Cooch Behar",
    "Haora": "Howrah",
    "Hugli": "Hooghly",
    "Maldah": "Malda",
    "Puruliya": "Purulia",
}


def main() -> None:
    with zipfile.ZipFile(SOURCE_ZIP) as z:
        national = json.loads(z.read("IND_ADM2.geojson"))

    by_name = {f["properties"].get("Name"): f for f in national["features"]}

    missing = [n for n in WB_ADM2_NAMES if n not in by_name]
    if missing:
        raise SystemExit(f"ADM2 source is missing expected West Bengal districts: {missing}")

    district_features = []
    geoms = []
    for name in WB_ADM2_NAMES:
        feat = by_name[name]
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)  # repair self-intersections before dissolving
        geoms.append(geom)
        district_features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "Name": MODERN_EQUIVALENT.get(name, name),
                "source_name_2011": name,
                "Level": "ADM2",
            },
        })

    state_geom = unary_union(geoms)

    with open(DISTRICTS_OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": district_features}, f)

    with open(STATE_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": mapping(state_geom),
                "properties": {"name": "West Bengal", "admin_level": "ADM1"},
            }],
        }, f)

    print(f"districts -> {DISTRICTS_OUT} ({len(district_features)} features)")
    print(f"state     -> {STATE_OUT} bounds={[round(v, 4) for v in state_geom.bounds]}")


if __name__ == "__main__":
    main()

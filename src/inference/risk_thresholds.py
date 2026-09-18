"""THE single source of truth for severe-weather risk display bands.

Every backend module that classifies a probability into a band imports from
here. Before this module existed the same three cut-points were written out
independently in nowcast_service._level_for_prob, nowcast_service
.get_point_inspection, risk_map.predictions_to_geojson and
risk_surface.colormap_risk_surface, plus a set of dead `_BAND_*` constants in
risk_surface.py that disagreed with all of them (0.20 vs 0.25). They happened to
agree numerically, but nothing enforced it: editing one was silent divergence
between the popup, the GeoJSON and the painted map.

These are DISPLAY bands for `severe_weather_prob`. They are deliberately NOT the
model's decision thresholds. The checkpoint carries its own operating points
(threshold_per_lead = {2: 0.727, 3: 0.679, 4: 0.673, 5: 0.637, 6: 0.625}), which
were optimised on the validation set for binary detection skill. The two answer
different questions -- "what colour is this cell" vs "would the model raise a
detection here" -- and must not be reconciled to each other.

The frontend legend (`<25% Normal`, `25-50% Watch`, `50-75% Alert`,
`>=75% Warning`) mirrors these values; frontend/js/app.js compares percentages,
so 0.25 here == 25 there.
"""
from __future__ import annotations

from typing import Tuple

# Lower edge of each band, as a probability in [0, 1].
WATCH_MIN = 0.25     # below this: Normal
ALERT_MIN = 0.50
WARNING_MIN = 0.75

# Canonical band metadata: (lower_edge, level_key, stage_label, hex_colour).
# Order is ascending; `classify` walks it from the top.
BANDS: Tuple[Tuple[float, str, str, str], ...] = (
    (0.0,         "green",  "NORMAL",  "#10b981"),
    (WATCH_MIN,   "yellow", "WATCH",   "#f59e0b"),
    (ALERT_MIN,   "orange", "ALERT",   "#f97316"),
    (WARNING_MIN, "red",    "WARNING", "#ef4444"),
)


def level_for_prob(p: float) -> str:
    """Colour key ('green'|'yellow'|'orange'|'red') for a probability."""
    if p >= WARNING_MIN:
        return "red"
    if p >= ALERT_MIN:
        return "orange"
    if p >= WATCH_MIN:
        return "yellow"
    return "green"


def stage_for_prob(p: float) -> str:
    """Operational stage label ('NORMAL'|'WATCH'|'ALERT'|'WARNING')."""
    if p >= WARNING_MIN:
        return "WARNING"
    if p >= ALERT_MIN:
        return "ALERT"
    if p >= WATCH_MIN:
        return "WATCH"
    return "NORMAL"

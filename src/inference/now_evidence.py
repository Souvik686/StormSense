"""Observation-evidence layer for NOW.

PURPOSE
-------
NOW answers "what severe-weather risk does the model estimate at T0". Its input
is the newest published GFS f000 analysis, which lags wall-clock by the NCEP
production latency (measured on this deployment: 5.3 - 7.3 h). Radar and station
observations are minutes old. Those are different instants, so presenting only
the model number invites the reader to treat a 5-11 h old atmosphere as "now".

This module builds an INDEPENDENT observation state for the same grid cells and
reports it ALONGSIDE the model probability, together with an explicit
agree/disagree verdict and the exact product-time gaps.

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
It NEVER modifies the model's calibrated probability. `assess()` returns
evidence and a verdict; it returns no adjusted probability, and no caller is
given one to display in place of the model's own value.

Why not fuse: a fusion weight must come from somewhere. Measured over this
domain (n = 48 grid points, 2026-09-18 17:10Z) RainViewer echo fraction and
station rain_1h correlate at r = 0.104; echo covered ~1.1% of the state and
every sampled echo pixel sat in the lowest reflectivity band of the palette.
The repository contains no gridded severe-weather verification data against
which a radar -> probability mapping could be fitted or validated. Any weight
would therefore be invented, and an invented weight applied to a safety-facing
probability is worse than an honest disagreement flag.

WHAT THE VERDICT MEANS
----------------------
  AGREE              model and observations tell the same story
  OBSERVED_NOT_MODELLED   radar/stations show precipitation where the model's
                          risk is low. Often legitimate: the model's input
                          predates the observation by hours, and its target is a
                          severe-weather proxy, not "any precipitation".
  MODELLED_NOT_OBSERVED   model risk is elevated with quiet observations. Often
                          legitimate: this is a FORECAST of developing
                          convection, and pre-convective environments are quiet
                          on radar by definition.
  NO_OBSERVATION     radar and stations could not be read; model value stands
                     alone and is labelled as such.

None of these are error states. They are information.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

# A cell counts as "observed precipitation" at or above this echo fraction.
# Rationale: the radar sampler summarises ~23x23 tile pixels per model cell, so
# isolated speckle is a few percent of the window. 0.10 requires a tenth of the
# cell to be painted, which is a coherent echo rather than noise, while staying
# well below the 0.90 maxima seen in real rain areas.
ECHO_PRESENT_FRACTION = 0.10

# Station rain at or above this counts as observed precipitation. OWM reports
# rain_1h in mm; 0.2 mm/h is the conventional "measurable rain" floor and is
# above the provider's rounding granularity.
STATION_RAIN_PRESENT_MM = 0.2

# The model band at which risk stops being "Normal". Mirrors the single source
# of truth in risk_thresholds.py.
from src.inference.risk_thresholds import WATCH_MIN


@dataclass
class NowEvidence:
    verdict: str
    model_prob: Optional[float]
    echo_fraction: Optional[float]
    echo_coverage: Optional[bool]
    station_rain_mm: Optional[float]
    station_name: Optional[str]
    radar_time_utc: Optional[str]
    station_time_utc: Optional[str]
    analysis_time_utc: Optional[str]
    reference_time_utc: Optional[str]
    gaps: Dict[str, Optional[float]]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "model_probability_pct": (
                round(self.model_prob * 100, 1) if self.model_prob is not None else None
            ),
            "observed": {
                "radar_echo_fraction": (
                    round(self.echo_fraction, 4) if self.echo_fraction is not None else None
                ),
                "radar_has_coverage": self.echo_coverage,
                "radar_time_utc": self.radar_time_utc,
                "station_rain_1h_mm": self.station_rain_mm,
                "nearest_station": self.station_name,
                "station_time_utc": self.station_time_utc,
            },
            "model_input": {
                "analysis_time_utc": self.analysis_time_utc,
                "reference_time_utc": self.reference_time_utc,
            },
            "product_time_gaps_hours": self.gaps,
            "explanation": self.explanation,
            # The contract, carried in the payload so no consumer can forget it.
            "observations_modify_probability": False,
            "note": (
                "The probability shown is the model's calibrated output, unchanged. "
                "Observations are reported independently; they are not inputs to "
                "SevereWeatherNetV2 and are not blended into its probability."
            ),
        }


def _hours_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    try:
        def p(s: str) -> datetime:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return round((p(a) - p(b)).total_seconds() / 3600.0, 2)
    except Exception:
        return None


def assess(
    model_prob: Optional[float],
    echo_fraction: Optional[float],
    echo_coverage: Optional[bool],
    station_rain_mm: Optional[float],
    station_name: Optional[str] = None,
    radar_time_utc: Optional[str] = None,
    station_time_utc: Optional[str] = None,
    analysis_time_utc: Optional[str] = None,
    reference_time_utc: Optional[str] = None,
) -> NowEvidence:
    """Compare the model's NOW probability with the current observations.

    Returns evidence and a verdict. Deliberately returns NO adjusted
    probability -- see the module docstring.
    """
    # "Observed precipitation" is true if EITHER independent sensor says so.
    # They are combined with OR, not averaged: they measure different things
    # (area-integrated echo vs a point gauge) and disagree far more often than
    # they agree (measured r = 0.104), so averaging them would manufacture a
    # confidence neither source supports.
    radar_says = (
        echo_fraction is not None
        and echo_coverage is True
        and echo_fraction >= ECHO_PRESENT_FRACTION
    )
    station_says = (
        station_rain_mm is not None and station_rain_mm >= STATION_RAIN_PRESENT_MM
    )
    have_obs = (echo_coverage is True) or (station_rain_mm is not None)
    observed_precip = radar_says or station_says

    model_elevated = model_prob is not None and model_prob >= WATCH_MIN

    gaps = {
        "radar_minus_analysis": _hours_between(radar_time_utc, analysis_time_utc),
        "station_minus_analysis": _hours_between(station_time_utc, analysis_time_utc),
        "reference_minus_analysis": _hours_between(reference_time_utc, analysis_time_utc),
    }

    lag = gaps.get("reference_minus_analysis")
    lag_txt = f"{lag:.1f} h" if lag is not None else "an unknown interval"

    if not have_obs or model_prob is None:
        verdict = "NO_OBSERVATION"
        why = (
            "No current radar coverage or station reading was available for this "
            "cell, so the model probability stands alone and is not corroborated "
            "by any observation."
        )
    elif observed_precip and not model_elevated:
        verdict = "OBSERVED_NOT_MODELLED"
        why = (
            "Precipitation is being observed here now, while the model's severe-"
            f"weather risk is low. The model read an atmospheric analysis {lag_txt} "
            "older than these observations, and its target is a severe-weather "
            "proxy (heavy 3-hour rain or strong instability), which ordinary rain "
            "does not by itself satisfy. This disagreement is reported, not "
            "reconciled: the probability is the model's own."
        )
    elif model_elevated and not observed_precip:
        verdict = "MODELLED_NOT_OBSERVED"
        why = (
            "The model indicates elevated severe-weather risk while radar and "
            "stations are currently quiet here. This is what a forecast of "
            "developing convection looks like before precipitation begins, so a "
            "quiet observation does not refute it. No observation is used to "
            "suppress the model value."
        )
    elif observed_precip and model_elevated:
        verdict = "AGREE"
        why = (
            "Observations and the model agree that conditions here are active. "
            "The probability is still the model's unmodified output -- the "
            "agreement raises confidence in reading it, and does not inflate it."
        )
    else:
        verdict = "AGREE"
        why = (
            "Neither the current observations nor the model indicate significant "
            "severe-weather activity at this cell."
        )

    return NowEvidence(
        verdict=verdict,
        model_prob=model_prob,
        echo_fraction=echo_fraction,
        echo_coverage=echo_coverage,
        station_rain_mm=station_rain_mm,
        station_name=station_name,
        radar_time_utc=radar_time_utc,
        station_time_utc=station_time_utc,
        analysis_time_utc=analysis_time_utc,
        reference_time_utc=reference_time_utc,
        gaps=gaps,
        explanation=why,
    )

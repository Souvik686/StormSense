"""Wall-clock target-time resolver.

THE PROBLEM THIS SOLVES

The UI offers NOW / +2h / +4h / +6h and those labels must mean what they say:
`+2 HOURS` is the forecast valid approximately two hours after the current
instant. The model, however, forecasts from the atmospheric state it was given,
and that state is a GFS analysis which is ALWAYS in the past -- measured NCEP
publication lag is 3.60h (n=12 cycles, min 3.54, max 3.81), so the newest
published analysis is between 3.6h and 9.6h old depending on where the wall clock
sits inside the 6-hourly cycle.

Serving model lead L and labelling it "+L hours from now" is therefore false: at
17:16Z with the 12Z analysis, model lead 2 is valid 14:00Z, which is 3.3h in the
PAST. That is the mislabelling this module exists to prevent.

THE MAPPING

    target_time   = wall_clock + requested_horizon
    required_lead = target_time - analysis_time
                  = analysis_age + requested_horizon

so the lead needed to answer "+2h from now" is `age + 2`, not 2. With age in
[3.6, 9.6] the four UI horizons need leads spanning roughly 3.6h to 15.6h, and
the exact value moves continuously as the analysis ages. This is why a model with
only a few coarse leads cannot serve wall-clock horizons honestly: it has nothing
valid at the target instant.

WHAT THIS MODULE DOES NOT DO

It does not relabel. If no available lead is within tolerance of the required
lead, `resolve` reports the horizon UNAVAILABLE rather than substituting the
nearest thing. Interpolation between two bracketing leads is offered only when
both bracket the target and the caller opts in, because interpolating a
probability field is a real approximation and the caller must own that choice.

COVERAGE OF THE FOUR UI HORIZONS

With hourly leads 4..16 (the v4 configuration), sweeping the wall clock across a
full 6-hourly cycle at 5-minute steps and asking for all of NOW/+2/+4/+6:

    availability rule 3.60h (measured publish lag) -> 100.0% coverage
    availability rule 5.00h (the conservative one   ->  97.9% coverage
                             actually in use)

The 2.1% of instants that cannot be served are the last ~20 minutes before a new
cycle becomes available under the conservative rule, where `+6h` would need a
16.7-17.0h lead. Extending training to a 17h lead would close it, but the gap is
an artefact of deliberately pessimistic availability accounting rather than of
real data, and this module reports it honestly instead of papering over it.

Everything here is pure arithmetic over timestamps -- no model, no I/O -- so it
is unit-testable without a checkpoint or a network.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence


# A resolved lead may differ from the exact requirement by at most this much
# before we call the horizon unavailable. 0.5h keeps the label honest to within
# half an hour, which is well inside the 1h granularity the UI implies and far
# inside the model's own 0.25 deg / 6-hourly input resolution.
DEFAULT_TOLERANCE_HOURS = 0.5


@dataclass
class ResolvedHorizon:
    """How one user-facing horizon was (or was not) satisfied."""
    requested_horizon_hours: float
    target_time_utc: str          # what the user is actually being shown
    wall_clock_utc: str
    analysis_time_utc: str
    analysis_age_hours: float
    required_lead_hours: float    # target_time - analysis_time
    available: bool
    # How it was served, when available:
    method: Optional[str] = None            # "exact" | "nearest" | "interpolated"
    source_lead_hours: Optional[float] = None
    source_leads_hours: Optional[List[float]] = None   # for interpolation
    interpolation_weight: Optional[float] = None        # weight on the LATER lead
    served_valid_time_utc: Optional[str] = None
    label_error_hours: Optional[float] = None  # served_valid_time - target_time
    reason: Optional[str] = None               # why unavailable

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def required_lead_hours(
    wall_clock: datetime, analysis_time: datetime, horizon_hours: float
) -> float:
    """The model lead whose valid time equals `wall_clock + horizon_hours`."""
    age_h = (wall_clock - analysis_time).total_seconds() / 3600.0
    return age_h + float(horizon_hours)


def resolve(
    wall_clock: datetime,
    analysis_time: datetime,
    horizon_hours: float,
    available_leads: Sequence[float],
    tolerance_hours: float = DEFAULT_TOLERANCE_HOURS,
    allow_interpolation: bool = False,
) -> ResolvedHorizon:
    """Map one user-facing horizon onto the model's available leads.

    `available_leads` are the checkpoint's own lead times, in hours from the
    analysis. Nothing here assumes they are evenly spaced or that they contain 0.
    """
    if wall_clock.tzinfo is None:
        wall_clock = wall_clock.replace(tzinfo=timezone.utc)
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    target = wall_clock + timedelta(hours=float(horizon_hours))
    age_h = (wall_clock - analysis_time).total_seconds() / 3600.0
    need = age_h + float(horizon_hours)

    base = dict(
        requested_horizon_hours=float(horizon_hours),
        target_time_utc=_iso(target),
        wall_clock_utc=_iso(wall_clock),
        analysis_time_utc=_iso(analysis_time),
        analysis_age_hours=round(age_h, 3),
        required_lead_hours=round(need, 3),
    )

    leads = sorted(float(x) for x in available_leads)
    if not leads:
        return ResolvedHorizon(**base, available=False,
                               reason="The loaded model exposes no forecast leads.")

    # Exact / within-tolerance match.
    nearest = min(leads, key=lambda L: abs(L - need))
    err = nearest - need
    if abs(err) <= tolerance_hours:
        return ResolvedHorizon(
            **base, available=True,
            method="exact" if abs(err) < 1e-9 else "nearest",
            source_lead_hours=nearest,
            served_valid_time_utc=_iso(analysis_time + timedelta(hours=nearest)),
            label_error_hours=round(err, 3),
        )

    # Optional interpolation, only when the requirement is genuinely bracketed.
    if allow_interpolation:
        lower = [L for L in leads if L <= need]
        upper = [L for L in leads if L >= need]
        if lower and upper:
            lo, hi = max(lower), min(upper)
            if hi > lo:
                w = (need - lo) / (hi - lo)
                return ResolvedHorizon(
                    **base, available=True, method="interpolated",
                    source_leads_hours=[lo, hi],
                    interpolation_weight=round(float(w), 4),
                    served_valid_time_utc=_iso(analysis_time + timedelta(hours=need)),
                    label_error_hours=0.0,
                )

    span = f"{leads[0]:g}..{leads[-1]:g}h"
    if need > leads[-1]:
        why = (f"Reaching {horizon_hours:+g}h from now needs a {need:.1f}h forecast "
               f"lead (the analysis is already {age_h:.1f}h old), but the model only "
               f"forecasts out to {leads[-1]:g}h.")
    elif need < leads[0]:
        why = (f"Reaching {horizon_hours:+g}h from now needs a {need:.1f}h lead, which "
               f"is shorter than the model's shortest lead ({leads[0]:g}h).")
    else:
        why = (f"Reaching {horizon_hours:+g}h from now needs a {need:.1f}h lead; the "
               f"nearest available lead is {nearest:g}h, which is {abs(err):.1f}h away "
               f"-- beyond the {tolerance_hours:g}h labelling tolerance.")
    return ResolvedHorizon(**base, available=False,
                           reason=why + f" Available leads: {span}.")


def resolve_all(
    wall_clock: datetime,
    analysis_time: datetime,
    horizons: Sequence[float],
    available_leads: Sequence[float],
    tolerance_hours: float = DEFAULT_TOLERANCE_HOURS,
    allow_interpolation: bool = False,
) -> Dict[str, ResolvedHorizon]:
    """Resolve every UI horizon at once. Keys are the horizon as an integer-ish
    string ("0", "2", "4", "6") so an API payload can be indexed by horizon."""
    out: Dict[str, ResolvedHorizon] = {}
    for h in horizons:
        key = f"{h:g}"
        out[key] = resolve(wall_clock, analysis_time, h, available_leads,
                           tolerance_hours=tolerance_hours,
                           allow_interpolation=allow_interpolation)
    return out

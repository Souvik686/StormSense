"""The backtest must not alias the diurnal cycle into its per-lead scores.

A sampling stride that is a multiple of 24 pins every simulated NOW to the same
hour of day, so each forecast lead always verifies at a fixed local time. West
Bengal convection peaks in the afternoon/evening, so later leads then sample
systematically stormier hours.

Measured with a 96h stride, this produced skill that RISES with lead time
(1.84x at +8h up to 3.38x at +16h) while observed prevalence climbed 0.046 ->
0.104 in lockstep -- physically impossible for a forecast, and a pure artifact
of when the samples were taken.
"""
import ast
import os

import pytest

SCRIPT = os.path.join("scripts", "historical_backtest.py")


def _default_step_hours():
    """Read the --step-hours default straight from the parser definition."""
    with open(SCRIPT, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--step-hours":
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    raise AssertionError("--step-hours default not found")


def test_default_stride_is_coprime_with_24():
    step = _default_step_hours()
    assert step % 24 != 0, (
        f"default --step-hours={step} is a multiple of 24; every simulated NOW "
        "would land at the same hour of day and the diurnal cycle would alias "
        "into the per-lead metrics"
    )


def test_default_stride_rotates_through_the_day():
    """Over a realistic run the sampled hours must actually spread out."""
    step = _default_step_hours()
    hours = {(i * step) % 24 for i in range(14)}
    assert len(hours) >= 8, (
        f"stride {step} visits only {len(hours)} distinct hours-of-day across "
        f"14 cases: {sorted(hours)}"
    )


def test_script_warns_on_an_aliasing_stride():
    """An explicit bad stride must still be flagged, not silently accepted."""
    with open(SCRIPT, "r", encoding="utf-8") as f:
        src = f.read()
    assert "step_hours % 24 == 0" in src, (
        "the script must detect a stride that is a multiple of 24"
    )
    assert "WARNING" in src


@pytest.mark.parametrize("bad", [24, 48, 72, 96, 120])
def test_known_aliasing_strides_are_rejected_by_the_rule(bad):
    assert bad % 24 == 0
    hours = {(i * bad) % 24 for i in range(14)}
    assert hours == {0}, "these strides pin every case to one hour of day"


# ---------------------------------------------------------------------------
# Alias DETECTION on a finished report (scripts/report_backtest.py)
# ---------------------------------------------------------------------------

def _alias_verdict(prev, leads=(8, 10, 12, 14, 16)):
    """Mirror of report_backtest.py's detector, kept in sync by the test below."""
    n = len(prev)
    mean_p = sum(prev) / n
    mean_l = sum(leads) / n
    cov = sum((leads[i] - mean_l) * (prev[i] - mean_p) for i in range(n))
    var_l = sum((L - mean_l) ** 2 for L in leads)
    var_p = sum((p - mean_p) ** 2 for p in prev)
    corr = (cov / (var_l ** 0.5 * var_p ** 0.5)) if (var_l and var_p) else 0.0
    ratio = (max(prev) / min(prev)) if min(prev) > 0 else float("inf")
    return (corr >= 0.80) and (ratio >= 1.5)


def test_detector_flags_the_real_aliased_run():
    """The measured 96h-stride prevalences must be reported as aliased.

    Note the dip between +8h and +10h: a strict "every pair increases" test
    returns False here and wrongly passed this run. The detector must judge the
    TREND.
    """
    measured = [0.0461, 0.0386, 0.0625, 0.0865, 0.1043]
    assert not all(measured[i] < measured[i + 1] for i in range(4)), (
        "this sample is deliberately non-monotonic; that is why strict "
        "monotonicity was the wrong test"
    )
    assert _alias_verdict(measured) is True


def test_detector_accepts_unaliased_prevalence():
    """Prevalence that wobbles without trending must not be flagged."""
    assert _alias_verdict([0.062, 0.058, 0.065, 0.059, 0.061]) is False


def test_detector_ignores_small_noise_spreads():
    """A weak upward drift within noise must not trip the detector."""
    assert _alias_verdict([0.060, 0.061, 0.062, 0.063, 0.064]) is False


def test_detector_matches_the_script_implementation():
    """Guard against the script and this mirror drifting apart."""
    with open(SCRIPT.replace("historical_backtest", "report_backtest"),
              "r", encoding="utf-8") as f:
        src = f.read()
    assert "corr >= 0.80" in src and "ratio >= 1.5" in src, (
        "report_backtest.py's alias thresholds changed; update this mirror"
    )

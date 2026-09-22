"""The benchmark figures shown in the UI must state their evaluation provenance.

The dashboard quotes per-horizon CSI / PR-AUC / POD / FAR / Brier for the AI
forecast model. Those are held-out test-split figures, not live production
numbers, and the page has to say so: which split they came from, how many
sequences, that the split was chronological with no temporal leakage, and what
baseline they are being compared against.

These tests do not police the metric values themselves -- they police the
provenance labelling, so a future edit cannot quietly drop the context that
makes the numbers interpretable.
"""
import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

# The headline benchmark metrics that appear in the verification table.
BENCHMARK_FIGURES = ["0.4054", "0.6417"]


@pytest.fixture(scope="module")
def html():
    return INDEX.read_text(encoding="utf-8", errors="replace")


def _context(html: str, needle: str, before: int = 1200, after: int = 600) -> str:
    i = html.find(needle)
    assert i != -1, f"{needle!r} not found in index.html"
    return html[max(0, i - before): i + after]


# Each phrase states one part of the evaluation provenance. Asserting the exact
# phrases (rather than merely looking for a keyword nearby) means deleting a
# qualifier fails the test instead of silently passing on an incidental mention
# elsewhere in the page.
REQUIRED_PROVENANCE = [
    # the split the figures were measured on
    "Held-Out 2024 Test Verification",
    # sample size and the chronological, leak-free protocol
    "4,322 chronological test sequences",
    "Zero Temporal Leakage",
]


@pytest.mark.parametrize("phrase", REQUIRED_PROVENANCE)
def test_required_provenance_is_present(phrase, html):
    """Each provenance qualifier must be present verbatim."""
    assert phrase in html, (
        f"the provenance qualifier {phrase!r} is gone. Without it the benchmark "
        f"figures ({', '.join(BENCHMARK_FIGURES)}) read as unqualified live "
        "performance claims."
    )


@pytest.mark.parametrize("figure", BENCHMARK_FIGURES)
def test_benchmark_figures_are_labelled_as_held_out(figure, html):
    """Each headline metric must sit near its held-out evaluation context."""
    ctx = _context(html, figure).lower()
    assert "held-out" in ctx or "test" in ctx, (
        f"the figure {figure} appears without any held-out/test-split qualifier "
        "nearby; a reader would take it for live performance"
    )


def test_benchmark_names_its_comparison_baseline(html):
    """A skill claim is only meaningful against a stated baseline."""
    low = html.lower()
    assert "persistence" in low, (
        "the benchmark table does not name the baseline its relative gains are "
        "measured against"
    )


def test_leads_at_every_horizon_badge_states_its_horizon_range(html):
    """A badge claiming the model 'leads at every horizon' must say which
    horizons it actually covers, so it cannot be read as an unbounded claim."""
    m = re.search(r"AI FORECAST MODEL LEADS AT EVERY HORIZON[^<]*", html)
    if m:
        assert re.search(r"\+\d+h", m.group(0)), (
            "the 'leads at every horizon' badge does not state the horizon range "
            "it refers to"
        )


def test_evolution_timeline_names_its_horizon_range(html):
    """The multi-horizon evolution panel must state the horizons it spans."""
    ctx = _context(html, "evolution-timeline-container", before=1500, after=200)
    assert re.search(r"\+\d+h", ctx), (
        "the evolution timeline does not state the horizon range it covers"
    )


def test_calibration_panel_states_the_split(html):
    """The calibration panel quotes CSI/PR-AUC and must name the data they
    were measured on rather than presenting them bare."""
    ctx = _context(html, "HORIZON CALIBRATION", before=200, after=900).lower()
    assert "test data" in ctx or "test split" in ctx or "held-out" in ctx, (
        "the calibration panel reports skill figures without naming the "
        "evaluation data they came from"
    )

"""Lifecycle helpers for figures rendered by the Streamlit application.

Matplotlib figures are mutable UI resources, not cached analysis data.  Keeping
the small helpers here (rather than in ``app.py``) also makes rerun behaviour
testable without starting Streamlit.
"""

from __future__ import annotations

from io import BytesIO
from typing import MutableMapping

from matplotlib.figure import Figure


DIRECTIONAL_STATE_PREFIX = "guided_directional_comparisons"


def invalidate_guided_analysis_state(state: MutableMapping[str, object]) -> None:
    """Discard every selection-dependent guided-analysis value.

    Directional widgets contain condition labels and identifiers.  Clearing the
    complete namespace is intentional: Streamlit widget state can otherwise
    restore a medium removed on a previous full-script run.
    """
    for key in list(state):
        if key in {"guided_complete_result", "guided_decisions", "guided_figure_view"}:
            state.pop(key, None)
        elif key.startswith(DIRECTIONAL_STATE_PREFIX):
            state.pop(key, None)


def validate_figure_render(figure: Figure) -> dict[str, object]:
    """Validate and describe the actual PNG payload immediately before display."""
    if not isinstance(figure, Figure):
        raise TypeError(f"Expected a Matplotlib Figure, got {type(figure).__name__}")
    if not figure.axes:
        raise AssertionError("A publication figure has no axes")

    populated_axes = sum(bool(axis.lines or axis.collections) for axis in figure.axes)
    output = BytesIO()
    figure.savefig(output, format="png")
    png_size = output.tell()
    if png_size <= 0:
        raise AssertionError("A publication figure produced an empty PNG")
    return {
        "type": type(figure).__name__,
        "axes": len(figure.axes),
        "size_inches": tuple(float(value) for value in figure.get_size_inches()),
        "populated_axes": populated_axes,
        "png_size": png_size,
    }

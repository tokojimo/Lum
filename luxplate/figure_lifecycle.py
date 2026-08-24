"""Lifecycle helpers for figures rendered by the Streamlit application.

Matplotlib figures are mutable UI resources, not cached analysis data.  Keeping
the small helpers here (rather than in ``app.py``) also makes rerun behaviour
testable without starting Streamlit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from io import BytesIO
from typing import MutableMapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


DIRECTIONAL_STATE_PREFIX = "guided_directional_comparisons"
PUBLICATION_FIGURES_KEY = "guided_publication_figures"
PUBLICATION_SIGNATURE_KEY = "guided_publication_figure_signature"
PUBLICATION_DEPENDENT_KEYS = {
    PUBLICATION_FIGURES_KEY,
    "guided_publication_statistics",
    "guided_publication_diagnostics",
}

FigureGallery = list[tuple[str, Figure]]


def publication_figure_signature(**parameters: object) -> tuple[tuple[str, object], ...]:
    """Return an immutable, order-sensitive signature for a publication render.

    Callers must provide every option that influences either artists or attached
    statistical output.  Nested lists are frozen, but their order is deliberately
    retained: reversing hypotheses is meaningful for a directional test.
    """
    def freeze(value: object) -> object:
        if isinstance(value, dict):
            return tuple((key, freeze(item)) for key, item in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(freeze(item) for item in value))
        return value

    return tuple((name, freeze(value)) for name, value in sorted(parameters.items()))


def _close_gallery(gallery: object) -> None:
    """Release all Matplotlib resources found in a former gallery."""
    if not isinstance(gallery, Iterable):
        return
    for item in gallery:
        if (isinstance(item, tuple) and len(item) == 2
                and isinstance(item[1], Figure)):
            # ``Figure.clear`` guarantees that even a reference retained by a
            # frontend fragment cannot continue to display old annotations.
            item[1].clear()
            plt.close(item[1])


def rebuild_publication_figures(
    state: MutableMapping[str, object],
    signature: tuple[tuple[str, object], ...],
    builder: Callable[[], FigureGallery],
) -> FigureGallery:
    """Discard the former gallery and build new Matplotlib figures from zero.

    Scientific data may be cached by the caller, but figures, statistics and
    diagnostics are one inseparable render layer.  They are invalidated before
    invoking ``builder`` so a failed build cannot expose stale results.
    """
    previous = state.pop(PUBLICATION_FIGURES_KEY, None)
    _close_gallery(previous)
    for key in PUBLICATION_DEPENDENT_KEYS:
        state.pop(key, None)
    state[PUBLICATION_SIGNATURE_KEY] = signature

    figures = builder()
    state[PUBLICATION_FIGURES_KEY] = figures
    return figures


def invalidate_guided_analysis_state(state: MutableMapping[str, object]) -> None:
    """Discard every selection-dependent guided-analysis value.

    Directional widgets contain condition labels and identifiers.  Clearing the
    complete namespace is intentional: Streamlit widget state can otherwise
    restore a medium removed on a previous full-script run.
    """
    _close_gallery(state.get(PUBLICATION_FIGURES_KEY))
    for key in list(state):
        if key in {"guided_complete_result", "guided_decisions", "guided_figure_view"}:
            state.pop(key, None)
        elif key in PUBLICATION_DEPENDENT_KEYS or key == PUBLICATION_SIGNATURE_KEY:
            state.pop(key, None)
        elif key.startswith(DIRECTIONAL_STATE_PREFIX):
            state.pop(key, None)


def validate_figure_render(figure: Figure) -> dict[str, object]:
    """Validate and describe the actual PNG payload immediately before display."""
    if not isinstance(figure, Figure):
        raise TypeError(f"Expected a Matplotlib Figure, got {type(figure).__name__}")
    if not figure.axes:
        raise AssertionError("A publication figure has no axes")

    # A metric panel is primarily made of boxplot patches and scatter
    # collections; brackets and technical points are optional.  Checking all
    # artist families prevents a perfectly valid box-only plot from being
    # rejected, while still detecting a figure whose axes were cleared by a
    # stale/mutated cached Figure instance.
    populated_axes = sum(
        bool(axis.lines or axis.collections or axis.patches or axis.artists)
        for axis in figure.axes
    )
    if not populated_axes:
        raise AssertionError("A publication figure has no visible plot artists")
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

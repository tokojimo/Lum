"""Lifecycle helpers for figures rendered by the Streamlit application.

Matplotlib figures are mutable UI resources, not cached analysis data.  Keeping
the small helpers here (rather than in ``app.py``) also makes rerun behaviour
testable without starting Streamlit.
"""

from __future__ import annotations

from io import BytesIO
from typing import MutableMapping

import pandas as pd
from matplotlib.figure import Figure


DIRECTIONAL_STATE_PREFIX = "guided_directional_comparisons"
LEGACY_PUBLICATION_STATE_KEYS = {
    "guided_publication_figures",
    "guided_publication_figure_signature",
    "guided_publication_statistics",
    "guided_publication_diagnostics",
}


def filter_result_strains(
    data: pd.DataFrame, strains: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Return calculated rows for the strains selected only for display.

    This deliberately operates on normalized output rather than input data: a
    result-view choice must never become part of the scientific analysis
    signature or trigger any correction/calculation stage again.
    """
    if "souche" not in data.columns:
        raise ValueError("Les résultats normalisés ne contiennent pas de colonne 'souche'.")
    selected = set(strains)
    return data.loc[data["souche"].isin(selected)].copy()


def result_strain_options(data: pd.DataFrame) -> list[str]:
    """Return displayable sample strains, excluding blank-control rows."""
    if "souche" not in data.columns:
        raise ValueError("Les résultats normalisés ne contiennent pas de colonne 'souche'.")
    samples = data
    if "type" in samples.columns:
        sample_types = samples["type"].astype("string").str.strip().str.casefold()
        samples = samples.loc[sample_types.eq("souche")]
    return sorted(samples["souche"].dropna().astype(str).unique().tolist())


def invalidate_guided_analysis_state(state: MutableMapping[str, object]) -> None:
    """Discard every selection-dependent guided-analysis value.

    Directional widgets contain condition labels and identifiers.  Clearing the
    complete namespace is intentional: Streamlit widget state can otherwise
    restore a medium removed on a previous full-script run.
    """
    for key in list(state):
        if key in {"guided_complete_result", "guided_decisions", "guided_figure_view"}:
            state.pop(key, None)
        elif key in LEGACY_PUBLICATION_STATE_KEYS:
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

"""State and range helpers for the normalized-luminescence AUC control."""

from __future__ import annotations

import math
from collections.abc import MutableMapping

import numpy as np
import pandas as pd


DRAFT_KEY = "auc_lum_norm_do_max_draft"
VALIDATED_KEY = "auc_lum_norm_do_max_validated"


def auc_do_slider_bounds(data: pd.DataFrame) -> tuple[float, float]:
    """Return hundredth-aligned scientifically usable OD limits."""
    od = pd.to_numeric(data["DO_corr"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if od.empty:
        return 0.0, 0.01
    thresholds = (pd.to_numeric(data.get("effective_threshold", pd.Series(dtype=float)), errors="coerce")
                  .replace([np.inf, -np.inf], np.nan).dropna())
    minimum = float(thresholds.max()) if not thresholds.empty else 0.05
    lower = math.ceil(max(0.0, minimum) * 100 - 1e-9) / 100
    upper = math.ceil(float(od.max()) * 100 - 1e-9) / 100
    return lower, max(lower + 0.01, upper)


def initialize_auc_do_state(
    state: MutableMapping[str, object], lower: float, upper: float
) -> None:
    """Initialize/preserve draft and validated values across reruns."""
    validated = state.get(VALIDATED_KEY, None)
    if validated is not None and not lower <= float(validated) < upper:
        validated = None
    state[VALIDATED_KEY] = validated
    seed = state.get(DRAFT_KEY, upper if validated is None else validated)
    state[DRAFT_KEY] = min(upper, max(lower, float(seed)))


def validate_auc_do_draft(
    state: MutableMapping[str, object], upper: float
) -> float | None:
    """Publish the draft; the rightmost position is the no-cutoff sentinel."""
    draft = float(state[DRAFT_KEY])
    validated = None if np.isclose(draft, upper) else draft
    state[VALIDATED_KEY] = validated
    return validated

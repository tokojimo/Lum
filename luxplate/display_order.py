"""Small, UI-independent helpers for persistent display ordering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reconcile_display_order(previous: object, present: Iterable[object]) -> list[str]:
    """Keep known values in their chosen order and append newly observed values.

    Values absent from the current data are omitted, without losing the relative
    order of the remaining values.  Callers may keep the returned active list in
    session state; a reappearing value is then treated like a new value.
    """
    active = list(dict.fromkeys(str(value) for value in present if value is not None))
    active_set = set(active)
    if isinstance(previous, str):
        candidates = [previous]
    elif isinstance(previous, Iterable) and not isinstance(previous, dict):
        # A Streamlit component can leave dictionaries (or other implementation
        # details) in session state.  They are not display-order values and must
        # never reach the membership check against the set below.
        candidates = [value for value in previous if isinstance(value, str)]
    else:
        candidates = []
    retained = list(dict.fromkeys(value for value in candidates if value in active_set))
    retained_set = set(retained)
    return [*retained, *(value for value in active if value not in retained_set)]


def ordered_present(values: Iterable[object], preferred: Sequence[str]) -> list[str]:
    """Return unique present values ranked by ``preferred``, then natural order."""
    natural = list(dict.fromkeys(str(value) for value in values if value is not None))
    rank = {value: index for index, value in enumerate(preferred)}
    natural_rank = {value: index for index, value in enumerate(natural)}
    return sorted(natural, key=lambda value: (rank.get(value, len(rank)), natural_rank[value]))

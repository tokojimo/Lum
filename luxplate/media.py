"""Canonical identities for culture media.

``Groupe`` is deliberately an acquisition identifier: importers may namespace
it by format and experiment.  The helpers here are the single boundary between
that internal identity and the biological medium shown to users.
"""

from __future__ import annotations

import re

import pandas as pd


_INTERNAL_MEDIUM_PREFIX = re.compile(
    r"^\s*(?:(?:kinetic|endpoint)\s*\|\s*)?"
    r"(?:exp(?:eriment)?\s*\d+\s*\|\s*)?",
    flags=re.IGNORECASE,
)


def medium_label(value: object) -> str:
    """Return the biological medium contained in an internal group id."""
    label = str(value).strip()
    return _INTERNAL_MEDIUM_PREFIX.sub("", label, count=1).strip()


def logical_media(data: pd.DataFrame, *, samples_only: bool = True) -> list[str]:
    """Return unique biological media in their first-occurrence order."""
    if "Groupe" not in data:
        return []
    rows = data
    if samples_only and "type" in rows:
        rows = rows.loc[rows["type"].astype(str).str.strip().str.casefold().eq("souche")]
    return list(dict.fromkeys(rows["Groupe"].map(medium_label)))

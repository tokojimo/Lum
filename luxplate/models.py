"""Shared data-model definitions."""

from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    """An explicit user/data-review decision; uncertainty is never hidden."""

    REVIEW = "review"
    KEEP = "keep"
    REMOVE = "remove"


@dataclass(frozen=True)
class ProcessingSettings:
    blank_sd_multiplier: float = 3.0
    minimum_od: float = 0.05
    consecutive_points: int = 3


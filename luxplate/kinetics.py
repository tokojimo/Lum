"""Small, testable kinetic primitives used by later processing stages."""

import numpy as np


def _finite_sorted(time_h, values) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_h, dtype=float)
    value = np.asarray(values, dtype=float)
    if time.shape != value.shape:
        raise ValueError("Time and value arrays must have identical shapes.")
    valid = np.isfinite(time) & np.isfinite(value)
    order = np.argsort(time[valid], kind="stable")
    return time[valid][order], value[valid][order]


def calculate_auc(time_h, values) -> float:
    """Trapezoidal AUC without inventing or zero-filling missing points."""
    time, value = _finite_sorted(time_h, values)
    if time.size < 2:
        return float("nan")
    return float(np.trapezoid(value, time))


def calculate_peak(time_h, values) -> tuple[float, float]:
    """Return (peak value, first observed peak time), with no interpolation."""
    time, value = _finite_sorted(time_h, values)
    if time.size == 0:
        return float("nan"), float("nan")
    index = int(np.argmax(value))
    return float(value[index]), float(time[index])


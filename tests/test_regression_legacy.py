import numpy as np

from luxplate.kinetics import calculate_auc, calculate_peak


def test_synthetic_legacy_kinetic_primitives():
    """Pinned synthetic baseline until private legacy fixtures are supplied."""
    time = [0.0, 1.0, 2.0, 3.0]
    lum = [0.0, 2.0, 4.0, 2.0]
    np.testing.assert_allclose(calculate_auc(time, lum), 7.0)
    np.testing.assert_allclose(calculate_peak(time, lum), (4.0, 2.0))


def test_missing_point_is_not_replaced_by_zero():
    np.testing.assert_allclose(calculate_auc([0, 1, 2], [1, np.nan, 3]), 4.0)


import pandas as pd

from luxplate.auc_controls import (DRAFT_KEY, VALIDATED_KEY, auc_do_slider_bounds,
                                   initialize_auc_do_state, validate_auc_do_draft)


def test_slider_range_uses_normalization_threshold_and_observed_od():
    data = pd.DataFrame({
        "DO_corr": [0.031, 0.204, 0.253],
        "effective_threshold": [0.047, 0.047, 0.047],
    })
    assert auc_do_slider_bounds(data) == (0.05, 0.26)


def test_draft_change_does_not_modify_validated_cutoff_until_validation():
    state = {}
    initialize_auc_do_state(state, 0.05, 0.30)
    assert state[VALIDATED_KEY] is None

    state[DRAFT_KEY] = 0.20
    initialize_auc_do_state(state, 0.05, 0.30)  # a Streamlit draft-only rerun
    assert state[VALIDATED_KEY] is None

    assert validate_auc_do_draft(state, 0.30) == 0.20
    assert state[VALIDATED_KEY] == 0.20


def test_rightmost_slider_position_validates_as_no_cutoff():
    state = {DRAFT_KEY: 0.30, VALIDATED_KEY: 0.20}
    assert validate_auc_do_draft(state, 0.30) is None
    assert state[VALIDATED_KEY] is None

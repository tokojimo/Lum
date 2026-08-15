import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from luxplate.blanks import apply_qc_decisions, run_blank_correction


def blank_table() -> pd.DataFrame:
    rows = []
    for group, kind, strain, values in [
        ("A", "blanc", "BA", [(0, 0.10, 10), (1, 0.20, 20)]),
        ("A", "blanc", "BA", [(0, 0.20, 30), (1, 0.30, 40)]),
        ("A", "souche", "SA", [(0, 0.50, 110), (1, 0.80, 220)]),
        ("B", "blanc", "BB", [(0, 1.00, 1000), (1, 1.10, 1100)]),
        ("B", "souche", "SB", [(0, 2.00, 2000), (1, 2.10, 2100)]),
        ("C", "souche", "SC", [(0, 3.00, 3000), (1, 3.10, 3100)]),
    ]:
        replicate = 1 + sum(1 for row in rows if row["souche"] == strain) // 2
        header = f"{strain}_rep{replicate}"
        for time, od, lum in values:
            rows.append({"temps_h": time, "souche": strain, "Groupe": group,
                         "replicat": replicate, "sample_header": header,
                         "DO_brute": od, "Lum_brute": lum, "type": kind})
    return pd.DataFrame(rows)


def decisions(*values: str, scope: str = "point", header: str = "SA_rep1") -> pd.DataFrame:
    return pd.DataFrame([
        {"scope": scope, "sample_header": header, "replicat": 1, "temps_h": 0,
         "variable_cible": "DO", "decision_utilisateur": value}
        for value in values
    ])


@pytest.mark.parametrize("decision", ["review", "keep", "", None])
def test_non_exclusion_decisions_retain_every_row(decision):
    data = blank_table()
    retained, excluded = apply_qc_decisions(data, decisions(decision))
    assert len(retained) == len(data)
    assert excluded.empty


@pytest.mark.parametrize("decision", ["remove", "exclude", "drop", "exclure"])
def test_explicit_exclusion_synonyms_remove_target(decision):
    retained, excluded = apply_qc_decisions(blank_table(), decisions(decision))
    assert len(excluded) == 1
    assert excluded.iloc[0]["temps_h"] == 0
    assert not ((retained["sample_header"] == "SA_rep1") & (retained["temps_h"] == 0)).any()


def test_point_and_series_scopes_have_distinct_reach():
    data = blank_table()
    point_retained, _ = apply_qc_decisions(data, decisions("remove"))
    series_retained, series_excluded = apply_qc_decisions(data, decisions("remove", scope="serie"))
    assert len(data) - len(point_retained) == 1
    assert len(series_excluded) == 2
    assert not series_retained["sample_header"].eq("SA_rep1").any()


def test_variable_specific_exclusion_removes_atomic_observation():
    """The explicit rule is to remove the paired OD/Lum row, not one cell."""
    _, excluded = apply_qc_decisions(blank_table(), decisions("exclude"))
    assert excluded.iloc[0][["DO_brute", "Lum_brute"]].notna().all()


def test_profiles_are_group_specific_and_formulas_are_numeric():
    result = run_blank_correction(blank_table(), decisions("keep"))
    profiles = result.blank_profiles.set_index(["Groupe", "temps_h"])
    assert profiles.loc[("A", 0), "DO_blanc_moyenne"] == pytest.approx(0.15)
    assert profiles.loc[("A", 0), "Lum_blanc_moyenne"] == pytest.approx(20)
    assert profiles.loc[("B", 0), "DO_blanc_moyenne"] == pytest.approx(1.0)
    row = result.corrected_data.query("souche == 'SA' and temps_h == 0").iloc[0]
    assert row["DO_corr"] == pytest.approx(0.35)
    assert row["Lum_corr"] == pytest.approx(90)


def test_corrected_output_keeps_absolute_blanks_and_separate_residuals():
    result = run_blank_correction(blank_table())
    blanks = result.corrected_data.query("type == 'blanc'")
    assert len(blanks) == 6
    assert blanks[["DO_corr", "Lum_corr", "Lum_blank_residual"]].notna().all().all()
    assert blanks.query("Groupe == 'A' and temps_h == 0")["DO_corr"].tolist() == pytest.approx([-0.05, 0.05])
    at_zero = blanks.query("Groupe == 'A' and temps_h == 0")
    assert at_zero["Lum_corr"].tolist() == pytest.approx([10, 30])
    assert at_zero["Lum_blank_residual"].tolist() == pytest.approx([-10, 10])
    assert blanks.groupby(["Groupe", "temps_h"])["Lum_blank_residual"].mean().eq(0).all()


def test_missing_group_blank_is_not_borrowed_and_is_reported():
    result = run_blank_correction(blank_table())
    rows = result.corrected_data.query("Groupe == 'C'")
    assert rows[["DO_corr", "Lum_corr"]].isna().all().all()
    assert rows["Lum_blank_residual"].isna().all()
    assert set(result.warnings["Groupe"]) == {"C"}


def test_inputs_are_never_modified_in_place():
    data = blank_table()
    journal = decisions("exclude")
    original_data, original_journal = data.copy(deep=True), journal.copy(deep=True)
    run_blank_correction(data, journal)
    pdt.assert_frame_equal(data, original_data)
    pdt.assert_frame_equal(journal, original_journal)

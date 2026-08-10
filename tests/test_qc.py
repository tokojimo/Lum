import pandas as pd
import pytest

from luxplate.qc import REQUIRED_COLUMNS, run_quality_control, validate_and_prepare


def kinetic_table() -> pd.DataFrame:
    rows = []
    for time, values in [(0.0, [0.10, 0.11, 2.0]), (1.0, [0.20, 0.21, 0.22])]:
        for replicate, value in enumerate(values, start=1):
            rows.append(
                {
                    "temps_h": time,
                    "souche": "S1",
                    "replicat": replicate,
                    "sample_header": f"S1_rep{replicate}",
                    "puits": f"A{replicate}",
                    "DO_brute": value,
                    "Lum_brute": 100 * value,
                    "type": "souche",
                }
            )
    return pd.DataFrame(rows)


def test_qc_proposes_anomaly_without_excluding_observation():
    source = kinetic_table()
    result = run_quality_control(source)

    assert len(result.data) == len(source)
    assert len(result.anomalies) == 1
    point = result.decisions[result.decisions["scope"].eq("point")].iloc[0]
    assert point["sample_header"] == "S1_rep3"
    assert point["variable_cible"] == "both"
    assert point["decision_utilisateur"] == "review"


def test_qc_summarizes_missing_measurements_and_flags_series():
    source = kinetic_table().drop(index=5)
    result = run_quality_control(source)
    series = result.series_summary.set_index("sample_header")

    assert series.loc["S1_rep3", "n_temps_manquants"] == 1
    assert "temps_manquants=1" in series.loc["S1_rep3", "flags"]
    assert "serie|S1_rep3" in set(result.decisions["decision_id"])


def test_qc_reports_missing_required_columns():
    source = kinetic_table().drop(columns="Lum_brute")

    with pytest.raises(ValueError, match="Lum_brute"):
        validate_and_prepare(source)


def test_required_schema_matches_long_import_contract():
    assert set(REQUIRED_COLUMNS) == {
        "temps_h", "souche", "replicat", "DO_brute", "Lum_brute", "type"
    }

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from luxplate.blanks import BlankCorrectionResult
from luxplate.project import export_project, import_project


def test_project_round_trip_preserves_tables_results_and_settings():
    table = pd.DataFrame({"souche": ["P0", "Δlux"], "temps_h": [0.0, 1.5], "ok": [True, False]})
    empty = table.iloc[0:0].copy()
    correction = BlankCorrectionResult(table, empty, table, table, empty, table)
    state = {
        "source_name": "essai_été", "source_identity": [["plaque.xlsx", 42, "Lum"]],
        "long_data": table, "qc_validated": True, "blank_correction_result": correction,
        "guided_media": ["LB"], "guided_strains": ["P0", "Δlux"],
        "guided_min_od": 0.08, "guided_figure_panels": "Panneaux par souche",
        "guided_directional_comparisons_stack": [("P0\0LB", "Δlux\0LB")],
    }

    restored = import_project(export_project(state))

    assert restored["source_name"] == "essai_été"
    assert restored["source_identity"] == [["plaque.xlsx", 42, "Lum"]]
    assert restored["qc_validated"] is True
    assert restored["guided_media"] == ["LB"]
    assert restored["guided_strains"] == ["P0", "Δlux"]
    assert restored["guided_min_od"] == 0.08
    assert restored["guided_figure_panels"] == "Panneaux par souche"
    assert restored["guided_directional_comparisons_stack"] == [["P0\0LB", "Δlux\0LB"]]
    pd.testing.assert_frame_equal(restored["long_data"], table)
    pd.testing.assert_frame_equal(restored["blank_correction_result"].corrected_data, table)


def test_project_rejects_non_project_archives():
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("other.json", "{}")

    with pytest.raises(ValueError, match="projet LuxPlate"):
        import_project(output.getvalue())

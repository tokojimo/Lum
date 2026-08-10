from io import BytesIO

import numpy as np
import pytest
from openpyxl import Workbook, load_workbook

from luxplate.varioskan import inspect_workbook, parse_kinetic_workbook


def synthetic_workbook(*, second_luminescence=False) -> BytesIO:
    workbook = Workbook()
    workbook.remove(workbook.active)
    headers = ["Lecture en cours", "temps moy. [s]", "PAO1 (A01)", "PAO1 (A02)", "Blanc (B01)"]
    for name, scale in [("Absorbance 600", 0.01), ("Luminescence 1", 100.0)]:
        sheet = workbook.create_sheet(name)
        sheet.append(["export synthétique"])
        sheet.append(headers)
        sheet.append([1, 0, scale, scale * 2, scale / 2])
        sheet.append([2, 3600, scale * 3, scale * 4, scale / 2])
    if second_luminescence:
        sheet = workbook.create_sheet("Luminescence 2")
        sheet.append(headers)
        sheet.append([1, 0, 10, 20, 5])
        sheet.append([2, 3600, 30, 40, 5])
    plan = workbook.create_sheet("Plan de plaque")
    plan.append([])
    plan.append([])
    plan.append([])
    plan.append([None, *range(1, 13)])
    plan.append(["A", "PAO1", "PAO1"])
    plan.append([None, "Groupe 1", "Groupe 1"])
    plan.append(["B", "Blanc"])
    plan.append([None, "Groupe 1"])
    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return output


def test_parse_kinetic_workbook_preserves_raw_technical_wells():
    result = parse_kinetic_workbook(synthetic_workbook())
    assert len(result) == 6
    assert result["sample_header"].nunique() == 3
    assert result.loc[result["souche"].eq("PAO1"), "replicat"].unique().tolist() == [1, 2]
    assert result.loc[result["souche"].eq("PAO1"), "Groupe"].eq("Groupe 1").all()
    np.testing.assert_allclose(sorted(result["temps_h"].unique()), [0, 1])
    assert result.loc[result["type"].eq("blanc"), "souche"].eq("Blanc").all()


def test_multiple_luminescence_sheets_require_explicit_selection():
    source = synthetic_workbook(second_luminescence=True)
    absorbance, luminescence = inspect_workbook(source)
    assert absorbance == "Absorbance 600"
    assert luminescence == ["Luminescence 1", "Luminescence 2"]
    source.seek(0)
    with pytest.raises(ValueError, match="choisissez-en une explicitement"):
        parse_kinetic_workbook(source)
    source.seek(0)
    selected = parse_kinetic_workbook(source, "Luminescence 2")
    assert selected["Lum_brute"].max() == 40


def test_medium_prefix_is_separated_and_numbered_blank_is_associated():
    source = synthetic_workbook()
    workbook = load_workbook(source)
    renamed = {
        "PAO1 (A01)": "SCFM2 (Po) 14.1Ac attB::P0-lux (A01)",
        "PAO1 (A02)": "SCFM2 (Po) 14.1Ac attB::P0-lux (A02)",
        "Blanc (B01)": "Blanc1 (B01)",
    }
    for sheet_name in ("Absorbance 600", "Luminescence 1"):
        sheet = workbook[sheet_name]
        for cell in sheet[2]:
            if cell.value in renamed:
                cell.value = renamed[cell.value]
    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    result = parse_kinetic_workbook(output)

    strains = result.loc[result["type"].eq("souche")]
    blanks = result.loc[result["type"].eq("blanc")]
    assert strains["souche"].unique().tolist() == ["14.1Ac attB::P0-lux"]
    assert strains["Groupe"].unique().tolist() == ["SCFM2 (Po)"]
    assert blanks["Groupe"].unique().tolist() == ["SCFM2 (Po)"]

import pandas as pd

from luxplate.media import logical_media, medium_label


def test_medium_label_removes_mixed_and_historical_internal_prefixes():
    assert medium_label("kinetic|exp1|SCFM2-KPi") == "SCFM2-KPi"
    assert medium_label("endpoint | experiment3 | SCFM2 (Po)") == "SCFM2 (Po)"
    assert medium_label("exp1|BM2") == "BM2"
    assert medium_label("experiment2 | SCFM2") == "SCFM2"
    assert medium_label("LB") == "LB"


def test_logical_media_deduplicates_six_mixed_groups_to_two_media():
    data = pd.DataFrame({
        "Groupe": [
            *(f"kinetic|exp{number}|SCFM2-KPi" for number in range(1, 4)),
            *(f"endpoint|exp{number}|SCFM2 (Po)" for number in range(1, 4)),
        ],
        "type": ["souche"] * 6,
    })
    assert logical_media(data) == ["SCFM2-KPi", "SCFM2 (Po)"]

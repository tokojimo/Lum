import pandas as pd

from luxplate.experimental_design import biological_n, summarize_technical_replicates


def _frame(rows):
    return pd.DataFrame(rows, columns=["source_file", "condition_id", "biological_replicate_id", "technical_replicate_id", "temps_h", "value"])


def test_three_bioreps_three_technical_each_returns_N3():
    data = _frame(("one.xlsx", "c", f"Rep{bio}", tech, 0, bio) for bio in range(1, 4) for tech in range(1, 4))
    assert biological_n(data, "c") == 3


def test_one_biorep_across_three_files_returns_N1():
    data = _frame((f"file{i}.xlsx", "c", "Rep1", i, 0, i) for i in range(1, 4))
    assert biological_n(data, "c") == 1


def test_two_bioreps_inside_one_excel_returns_N2():
    data = _frame(("one.xlsx", "c", f"Rep{i}", 1, 0, i) for i in range(1, 3))
    assert biological_n(data, "c") == 2


def test_technical_wells_are_averaged_before_biological_summary():
    data = _frame([("one.xlsx", "c", "Rep1", 1, 0, 1), ("one.xlsx", "c", "Rep1", 2, 0, 3), ("two.xlsx", "c", "Rep2", 1, 0, 10)])
    result = summarize_technical_replicates(data, "value")
    assert result["value"].tolist() == [2.0, 10.0]


import matplotlib.pyplot as plt
import pandas as pd

from luxplate.plotting import build_guided_raw_figures
from test_workflow import workflow_table


def test_guided_raw_figures_separate_sample_types_and_biological_replicates():
    data = workflow_table()
    second_replicate = data.loc[data["type"].eq("souche")].copy()
    second_replicate["replicat"] = 2
    second_replicate["sample_header"] += " rep2"
    data = pd.concat([data, second_replicate], ignore_index=True)

    blank_figures = build_guided_raw_figures(data, sample_type="blanc")
    sample_figures = build_guided_raw_figures(data, sample_type="souche")

    assert len(blank_figures) == 1
    assert len(sample_figures) == 4
    assert all(len(figure.axes) == 2 for _, figure in blank_figures + sample_figures)
    for _, figure in blank_figures + sample_figures:
        plt.close(figure)

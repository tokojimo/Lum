import matplotlib.pyplot as plt
import pandas as pd

from luxplate.plotting import (build_guided_raw_figures, build_publication_figures,
                               plot_kinetics, plot_mixed_panels, plot_publication_panels)
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


def _publication_table():
    rows = []
    for strain_index, strain in enumerate(("P0-lux", "Reporter-lux")):
        for well in ("A1", "A2"):
            for time in (0.0, 1.0, 2.0):
                od = 0.1 + 0.08 * time + 0.01 * strain_index
                rows.append({"temps_h": time, "souche": strain, "Groupe": "DMEM",
                    "sample_header": f"{strain}-{well}", "puits": well, "replicat": 1,
                    "experience_id": "exp", "type": "souche", "DO_corr": od,
                    "Lum_corr": 100 + 50 * time + 20 * strain_index,
                    "Lum_norm": (100 + 50 * time + 20 * strain_index) / od})
    return pd.DataFrame(rows)


def test_publication_panels_accept_log_scale_and_strain_panels():
    figure = plot_publication_panels(
        _publication_table(), value="Lum_norm", group_by="souche", y_scale="log"
    )
    assert len(figure.axes) == 2
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    plt.close(figure)


def test_mixed_panels_offer_log_luminescence_and_error_bars():
    figure = plot_mixed_panels(_publication_table(), lum_scale="log", uncertainty="bars")
    assert len(figure.axes) == 2
    assert figure.axes[0].get_yscale() == "linear"
    assert figure.axes[1].get_yscale() == "log"
    plt.close(figure)


def test_gallery_can_select_curve_families():
    figures = build_publication_figures(
        _publication_table(), families=("growth", "mixed"), uncertainty="ribbon"
    )
    assert [name for name, _ in figures] == ["croissance", "croissance_luminescence_mixte"]
    for _, figure in figures:
        plt.close(figure)


def test_gallery_builds_one_targeted_control_comparison_per_reporter():
    figures = build_publication_figures(
        _publication_table(), families=("control",), control="P0-lux", lum_scale="log"
    )
    assert [name for name, _ in figures] == ["comparaison_Reporter-lux_vs_P0-lux"]
    assert figures[0][1].axes[1].get_yscale() == "log"
    plt.close(figures[0][1])


def test_kinetics_legend_excludes_blanks_and_deduplicates_strains():
    data = _publication_table()
    duplicate_experiment = data.copy()
    duplicate_experiment["experience_id"] = "exp2"
    blank = data.iloc[:3].copy()
    blank["type"] = "blanc"
    blank["souche"] = "Blanc1"
    plotted = pd.concat([data, duplicate_experiment, blank], ignore_index=True)
    metrics = pd.DataFrame(columns=[
        "experience_id", "souche", "Groupe", "sample_header", "puits", "replicat",
    ])

    figure = plot_kinetics(plotted, metrics)

    for axis in figure.axes:
        labels = axis.get_legend_handles_labels()[1]
        assert labels == ["P0-lux", "Reporter-lux"]
    first_axis_colors = {
        line.get_label(): line.get_color() for line in figure.axes[0].lines
        if not line.get_label().startswith("_")
    }
    second_axis_colors = {
        line.get_label(): line.get_color() for line in figure.axes[1].lines
        if not line.get_label().startswith("_")
    }
    assert first_axis_colors == second_axis_colors
    plt.close(figure)

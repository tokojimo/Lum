import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from luxplate.plotting import (build_guided_raw_figures, build_publication_figures,
                               plot_kinetics, plot_metric_points, plot_mixed_panels,
                               plot_publication_panels)
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
    assert all(axis.get_xlabel() == "Time (h)" for axis in figure.axes)
    assert all("Normalized luminescence" in axis.get_ylabel() for axis in figure.axes)
    plt.close(figure)


def test_mixed_panels_offer_log_luminescence_and_error_bars():
    figure = plot_mixed_panels(_publication_table(), lum_scale="log", uncertainty="bars")
    assert len(figure.axes) == 2
    assert figure.axes[0].get_yscale() == "linear"
    assert figure.axes[1].get_yscale() == "log"
    assert figure.axes[0].get_ylabel() == r"OD$_{600}$"
    assert figure.axes[1].get_ylabel() == "Luminescence (RLU)"
    assert sum(line.get_linestyle() == "--" for line in figure.axes[1].lines) == 2
    assert all(len(axis.child_axes) == 0 for axis in figure.axes)
    legend = figure.legends[0]
    assert legend.get_title().get_text() == "Promoteur (couleur) · Mesure (style)"
    assert legend.get_texts()[-1].get_text() == "Luminescence (RLU) — pointillés"
    plt.close(figure)


def test_linear_rlu_axes_use_scientific_notation_and_plain_title():
    figure = plot_publication_panels(_publication_table(), value="Lum_corr", title="Luminescence")
    axis = figure.axes[0]
    assert isinstance(axis.yaxis.get_major_formatter(), ScalarFormatter)
    figure.canvas.draw()
    assert axis.yaxis.get_offset_text().get_text()
    assert figure._suptitle.get_text() == "Luminescence"
    plt.close(figure)


def test_metric_points_average_technical_series_per_independent_experiment():
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": "SCFM2", "experience_id": experiment,
         "replicat": 1, "lum_norm_auc": value + technical}
        for experiment, value in (("exp1", 10.0), ("exp2", 20.0))
        for strain in ("P0-lux", "Reporter-lux")
        for technical in (0.0, 2.0, 4.0)
    ])

    figure = plot_metric_points(metrics, metric="lum_norm_auc", y_scale="log")

    points = [collection for collection in figure.axes[0].collections if len(collection.get_offsets())]
    assert len(points) == 4
    assert sorted(float(collection.get_offsets()[0, 1]) for collection in points) == [12, 12, 22, 22]
    assert figure.axes[0].get_legend().get_title().get_text() == "Replicate display"
    assert len(figure.axes[0].patches) == 2
    plt.close(figure)


def test_metric_points_fall_back_to_linear_when_log_data_are_non_positive():
    metrics = pd.DataFrame([
        {"souche": "P0-lux", "Groupe": "SCFM2", "lum_norm_auc": 0.0},
        {"souche": "Reporter-lux", "Groupe": "SCFM2", "lum_norm_auc": -2.0},
    ])

    figure = plot_metric_points(metrics, metric="lum_norm_auc", y_scale="log")

    axis = figure.axes[0]
    assert axis.get_yscale() == "linear"
    points = [collection for collection in axis.collections if len(collection.get_offsets())]
    assert sorted(float(collection.get_offsets()[0, 1]) for collection in points) == [-2.0, 0.0]
    plt.close(figure)


def test_metric_points_ignore_infinities_before_selecting_log_scale():
    metrics = pd.DataFrame([
        {"souche": "P0-lux", "Groupe": "SCFM2", "lum_norm_auc": np.inf},
        {"souche": "Reporter-lux", "Groupe": "SCFM2", "lum_norm_auc": -np.inf},
        {"souche": "Reporter-lux", "Groupe": "SCFM2", "lum_norm_auc": 0.0},
    ])

    figure = plot_metric_points(metrics, metric="lum_norm_auc", y_scale="log")

    axis = figure.axes[0]
    assert axis.get_yscale() == "linear"
    points = [collection for collection in axis.collections if len(collection.get_offsets())]
    assert [float(collection.get_offsets()[0, 1]) for collection in points] == [0.0]
    # Exercise the layout/rendering path where Matplotlib's LogLocator raised.
    figure.canvas.draw()
    plt.close(figure)


def test_metric_points_ignore_an_entirely_empty_experience_identifier():
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": "SCFM2", "experience_id": np.nan,
         "replicat": 1, "lum_norm_auc": value}
        for strain, value in (("P0-lux", 10.0), ("Reporter-lux", 20.0))
    ])

    figure = plot_metric_points(metrics, metric="lum_norm_auc", y_scale="log")

    axis = figure.axes[0]
    points = [collection for collection in axis.collections if len(collection.get_offsets())]
    assert axis.get_yscale() == "log"
    assert sorted(float(collection.get_offsets()[0, 1]) for collection in points) == [10.0, 20.0]
    # Exercise the layout/rendering path from the reported Matplotlib traceback.
    figure.canvas.draw()
    plt.close(figure)


def test_gallery_can_select_curve_families():
    figures = build_publication_figures(
        _publication_table(), families=("growth", "mixed"), uncertainty="ribbon"
    )
    assert [name for name, _ in figures] == ["croissance", "croissance_luminescence_mixte"]
    for _, figure in figures:
        plt.close(figure)


def test_gallery_adds_pooled_recap_figures_for_multiple_experiments():
    data = pd.concat([
        _publication_table().assign(experience_id=f"exp{number}",
                                    Groupe=f"Experiment {number} | DMEM (replicate)")
        for number in (1, 2, 3)
    ], ignore_index=True)

    figures = build_publication_figures(data, families=("growth", "corrected", "mixed"))

    assert [name for name, _ in figures] == [
        "croissance", "croissance_moyenne_experiences",
        "luminescence_corrigee", "luminescence_corrigee_moyenne_experiences",
        "croissance_luminescence_mixte",
    ]
    # Each recap has one medium panel pooling all three independent experiments.
    assert len(figures[1][1].axes) == 1
    assert len(figures[3][1].axes) == 1
    assert len(figures[4][1].axes) == 2  # one dual-axis medium panel
    for _, figure in figures:
        plt.close(figure)


def test_nearby_technical_times_are_aligned_before_biological_summary():
    data = _publication_table()
    data.loc[data["puits"].eq("A2"), "temps_h"] += 5 / 3600

    figure = plot_publication_panels(data, value="DO_corr")

    # One curve per strain, with the three planned acquisition times rather than
    # six successively connected well times (the source of artificial saw teeth).
    reporter_lines = [container.lines[0] for container in figure.axes[0].containers
                      if container.get_label() in {"P0-lux", "Reporter-lux"}]
    assert len(reporter_lines) == 2
    assert all(len(line.get_xdata()) == 3 for line in reporter_lines)
    plt.close(figure)


def test_metric_gallery_uses_boxplots_and_includes_peak_time():
    figures = build_publication_figures(_publication_table(), families=("peak", "peak_time", "auc", "doubling"),
                                        metric_scale="linear")
    assert [name for name, _ in figures] == ["pic_luminescence_normalisee",
        "temps_pic_luminescence_normalisee", "auc_luminescence_normalisee", "temps_doublement"]
    for _, figure in figures:
        assert len(figure.axes[0].patches) == 2
        assert not any(text.get_text().startswith("mean =") for text in figure.axes[0].texts)
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

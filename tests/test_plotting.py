import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from luxplate.plotting import (build_guided_raw_figures, build_publication_figures,
                               metric_fold_change_vs_control, plot_kinetics, plot_metric_points, plot_mixed_panels,
                               plot_publication_panels)
from test_workflow import workflow_table


def test_guided_raw_figures_group_technical_replicates_in_one_biological_row():
    data = workflow_table()
    second_replicate = data.loc[data["type"].eq("souche")].copy()
    second_replicate["replicat"] = 2
    second_replicate["sample_header"] += " rep2"
    data = pd.concat([data, second_replicate], ignore_index=True)

    blank_figures = build_guided_raw_figures(data, sample_type="blanc")
    sample_figures = build_guided_raw_figures(data, sample_type="souche")

    assert len(blank_figures) == 1
    assert len(sample_figures) == 2
    assert all(len(figure.axes) == 2 for _, figure in blank_figures + sample_figures)
    for _, figure in blank_figures + sample_figures:
        plt.close(figure)


def test_guided_raw_legend_identifies_each_technical_well_position():
    data = workflow_table()

    blank_figure = build_guided_raw_figures(data, sample_type="blanc")[0][1]
    sample_figure = build_guided_raw_figures(data, sample_type="souche")[0][1]

    assert [text.get_text() for text in blank_figure.axes[0].get_legend().get_texts()] == [
        "Rép. technique (B01)"
    ]
    assert [text.get_text() for text in sample_figure.axes[0].get_legend().get_texts()] == [
        "Rép. technique (A01)"
    ]
    plt.close(blank_figure)
    plt.close(sample_figure)


def test_guided_raw_figures_order_media_before_biological_conditions():
    # Workbook-major input must not result in M1/bio1, M2/bio1, M1/bio2, M2/bio2.
    first = pd.concat([
        workflow_table().query("type == 'souche'").iloc[:4].assign(Groupe=medium, souche="Bio 1")
        for medium in ("M1", "M2")
    ])
    second = pd.concat([
        workflow_table().query("type == 'souche'").iloc[:4].assign(Groupe=medium, souche="Bio 2")
        for medium in ("M1", "M2")
    ])

    figures = build_guided_raw_figures(pd.concat([first, second]), sample_type="souche")

    assert [title for title, _ in figures] == [
        "M1 · Bio 1", "M1 · Bio 2", "M2 · Bio 1", "M2 · Bio 2",
    ]
    for _, figure in figures:
        plt.close(figure)


def test_guided_raw_figures_put_excel_experiments_on_rows_with_shared_scales():
    first = workflow_table().assign(experience="Expérience 1", Groupe="exp1|M1")
    second = workflow_table().assign(
        experience="Expérience 2", Groupe="exp2|M1",
        DO_brute=lambda frame: frame["DO_brute"] * 3,
        Lum_brute=lambda frame: frame["Lum_brute"] * 10,
    )

    figures = build_guided_raw_figures(pd.concat([first, second]), sample_type="souche")

    assert len(figures) == 2
    for _, figure in figures:
        assert len(figure.axes) == 4
        assert figure.axes[0].get_ylim() == figure.axes[2].get_ylim()
        assert figure.axes[1].get_ylim() == figure.axes[3].get_ylim()
        assert "Expérience 1" in figure.axes[0].get_ylabel()
        assert "Expérience 2" in figure.axes[2].get_ylabel()
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
    assert legend.get_title().get_text() == "Promoter (color) · Measurement (line)"
    assert [text.get_text() for text in legend.get_texts()][-2:] == [
        r"OD$_{600}$", "Luminescence (RLU)",
    ]
    plt.close(figure)


def test_mixed_panels_share_od_and_luminescence_limits_across_media():
    data = pd.concat([
        _publication_table(),
        _publication_table().assign(Groupe="SCFM2", DO_corr=lambda frame: frame["DO_corr"] * 4,
                                    Lum_corr=lambda frame: frame["Lum_corr"] * 20),
    ], ignore_index=True)

    figure = plot_mixed_panels(data)

    od_axes = [axis for axis in figure.axes if axis.get_ylabel() == r"OD$_{600}$"]
    lum_axes = [axis for axis in figure.axes if axis.get_ylabel() == "Luminescence (RLU)"]
    assert len(od_axes) == len(lum_axes) == 2
    assert od_axes[0].get_ylim() == od_axes[1].get_ylim()
    assert lum_axes[0].get_ylim() == lum_axes[1].get_ylim()
    plt.close(figure)


def test_time_course_panels_end_at_shortest_experiment_duration():
    short = _publication_table().assign(experience_id="short")
    long = pd.concat([
        _publication_table().assign(experience_id="long"),
        _publication_table().query("temps_h == 2").assign(
            temps_h=4.0, experience_id="long"
        ),
    ], ignore_index=True)
    data = pd.concat([short, long], ignore_index=True)

    publication = plot_publication_panels(data, value="Lum_corr", group_by="souche")
    mixed = plot_mixed_panels(data)

    assert all(axis.get_xlim() == (0.0, 2.0) for axis in publication.axes)
    assert all(axis.get_xlim() == (0.0, 2.0) for axis in mixed.axes)
    plt.close(publication)
    plt.close(mixed)


def test_linear_rlu_axes_use_scientific_notation_and_plain_title():
    figure = plot_publication_panels(_publication_table(), value="Lum_corr", title="Luminescence")
    axis = figure.axes[0]
    assert isinstance(axis.yaxis.get_major_formatter(), FuncFormatter)
    figure.canvas.draw()
    assert not axis.yaxis.get_offset_text().get_visible()
    assert any("e+" in label.get_text() for label in axis.get_yticklabels())
    assert figure._suptitle.get_text() == "Luminescence"
    legend = figure.legends[0]
    assert legend.get_title().get_fontsize() == 18
    assert all(text.get_fontsize() == 14 for text in legend.get_texts())
    plt.close(figure)


def test_linear_rlu_figures_can_be_cached_with_pickle():
    figures = build_publication_figures(
        _publication_table(), families=("corrected", "mixed")
    )

    cached = pickle.dumps(figures)

    assert cached
    for _, figure in figures:
        plt.close(figure)


def test_reporter_colors_are_fixed_across_order_and_figure_families():
    expected = {
        "P0-lux": "#0072B2", "PspeD-lux": "#D55E00",
        "PspeD2-1A-lux": "#009E73", "PspeD2-3B-lux": "#CC79A7",
        "PspeE-lux": "#E69F00",
    }
    data = pd.concat([
        _publication_table().iloc[:6].assign(souche=strain)
        for strain in reversed(expected)
    ], ignore_index=True)

    corrected = plot_publication_panels(data, value="Lum_corr")
    mixed = plot_mixed_panels(data)

    corrected_colors = {
        container.get_label(): container.lines[0].get_color()
        for container in corrected.axes[0].containers
    }
    mixed_colors = {line.get_label(): line.get_color() for line in mixed.axes[0].lines
                    if not line.get_label().startswith("_")}
    assert corrected_colors == expected
    assert mixed_colors == expected
    plt.close(corrected)
    plt.close(mixed)


def test_construct_names_keep_reporter_labels_and_colors():
    constructs = {
        "14.1Ac attB::PspeD-lux": ("PspeD", "#D55E00"),
        "14.1Ac attB::PspeE-lux": ("PspeE", "#E69F00"),
    }
    data = pd.concat([
        _publication_table().iloc[:6].assign(souche=construct) for construct in constructs
    ], ignore_index=True)

    figure = plot_publication_panels(data, value="Lum_corr")

    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        expected[0] for expected in constructs.values()
    ]
    assert [container.lines[0].get_color() for container in figure.axes[0].containers] == [
        expected[1] for expected in constructs.values()
    ]
    plt.close(figure)


def test_large_legends_leave_space_above_medium_titles():
    strains = ["P0-lux", "PspeD-lux", "PspeD2-1A-lux", "PspeD2-3B-lux",
               "PspeE-lux", *[f"Reporter-{index}" for index in range(7)]]
    data = pd.concat([
        _publication_table().iloc[:6].assign(souche=strain) for strain in strains
    ], ignore_index=True)

    for figure in (plot_publication_panels(data, value="Lum_corr"), plot_mixed_panels(data)):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_bottom = figure.legends[0].get_window_extent(renderer).y0
        title_top = figure.axes[0].title.get_window_extent(renderer).y1
        assert title_top < legend_bottom
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


def test_metric_points_make_one_panel_per_medium_and_label_pvalues_with_stars():
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": medium, "experience_id": f"exp{experiment}",
         "replicat": 1, "lum_norm_auc": base + experiment + strain_index * 10}
        for medium, base in (("DMEM", 10), ("SCFM2", 100))
        for experiment in range(1, 6)
        for strain_index, strain in enumerate(("P0-lux", "Reporter-lux"))
    ])

    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", group_by="Groupe", title="Normalized AUC"
    )

    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["DMEM", "SCFM2"]
    labels = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert any(label.startswith("pHolm = ") and "(raw " in label for label in labels)
    plt.close(figure)


def test_metric_points_compare_every_strain_medium_pair_in_one_figure():
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": f"Experiment {experiment} | {medium}",
         "experience_id": f"exp{experiment}", "replicat": 1,
         "lum_norm_auc": experiment + strain_index * 10 + medium_index * 3}
        for experiment in range(1, 7)
        for strain_index, strain in enumerate(("PspeD-lux", "PspeE-lux", "P0-lux"))
        for medium_index, medium in enumerate(("Milieu 1", "Milieu 2"))
    ])

    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True, title="AUC"
    )

    assert len(figure.axes) == 1
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "PspeD\nMilieu 1", "PspeD\nMilieu 2", "PspeE\nMilieu 1",
        "PspeE\nMilieu 2", "P0\nMilieu 1", "P0\nMilieu 2",
    ]
    # Six conditions yield all 15 Holm-corrected pairwise comparisons.
    assert len(figure._luxplate_statistics) == 15
    assert len([text for text in figure.axes[0].texts if text.get_text().startswith("pHolm")]) == 15
    assert figure.get_figheight() > 6
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


def test_experiment_and_pooled_panel_labels_preserve_medium_parentheses():
    data = _publication_table().assign(Groupe="exp2|SCFM2 (Po)")
    figure = plot_publication_panels(data, value="DO_corr")
    assert figure.axes[0].get_title() == "Experiment 2 – SCFM2 (Po)"
    figures = build_publication_figures(
        pd.concat([data.assign(experience_id="exp2"),
                   data.assign(experience_id="exp3", Groupe="exp3|SCFM2 (Po)")]),
        families=("growth",),
    )
    assert figures[1][1].axes[0].get_title() == "SCFM2 (Po)"
    plt.close(figure)
    for _, item in figures:
        plt.close(item)


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


def test_fold_change_metrics_use_matched_p0_biological_mean_per_medium():
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": f"Experiment {experiment} | {medium}",
         "experience_id": f"exp{experiment}", "replicat": 1,
         "lum_norm_peak": control_value * ratio + technical}
        for experiment, control_value in ((1, 10.0), (2, 20.0))
        for medium in ("DMEM", "SCFM2")
        for strain, ratio in (("P0-lux", 1.0), ("Reporter-lux", 3.0))
        for technical in (0.0, 2.0)
    ])

    result = metric_fold_change_vs_control(metrics, metric="lum_norm_peak")

    p0 = result.loc[result["souche"].eq("P0-lux"), "lum_norm_peak_fold_change"]
    reporter = result.loc[result["souche"].eq("Reporter-lux"), "lum_norm_peak_fold_change"]
    assert set(result["Groupe"]) == {"DMEM", "SCFM2"}
    assert np.allclose(p0, [10 / 11, 12 / 11, 10 / 11, 12 / 11,
                            20 / 21, 22 / 21, 20 / 21, 22 / 21])
    assert np.allclose(sorted(reporter), sorted([
        30 / 11, 32 / 11, 30 / 11, 32 / 11,
        60 / 21, 62 / 21, 60 / 21, 62 / 21,
    ]))


def test_gallery_adds_peak_and_auc_fold_change_figures_by_medium():
    figures = build_publication_figures(
        _publication_table(), families=("peak_fc", "auc_fc"), metric_scale="linear"
    )

    assert [name for name, _ in figures] == [
        "pic_luminescence_normalisee_fold_change_P0",
        "auc_luminescence_normalisee_fold_change_P0",
    ]
    assert all("fold change vs P0" in figure.axes[0].get_ylabel() for _, figure in figures)
    assert all(len(figure.axes) == 1 for _, figure in figures)
    assert all(figure.axes[0].get_title().endswith("fold change vs P0") for _, figure in figures)
    assert all([tick.get_text() for tick in figure.axes[0].get_xticklabels()] ==
               ["P0\nDMEM", "Reporter-lux\nDMEM"] for _, figure in figures)
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

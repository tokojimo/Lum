import json
import inspect
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.ticker import FuncFormatter

from luxplate.plotting import (_aligned_biological_summary, _comparison_identifiers,
                               BRACKET_ENDPOINT_GAP, DEFAULT_BOXPLOT_Y_SCALE,
                               DEFAULT_CURVE_Y_SCALE, _bracket_draw_coordinates, _bracket_levels,
                               build_guided_corrected_figures,
                               build_guided_crosstalk_figures, build_guided_raw_figures,
                               build_publication_figures, collect_publication_statistics,
                               directional_condition_options,
                               directional_comparison_options, metric_fold_change_vs_control,
                               plot_kinetics, plot_metric_points, plot_mixed_panels,
                               plot_publication_panels)
from luxplate.statistics import _canonical_condition, paired_directional_t_tests
from test_workflow import workflow_table


def test_directional_condition_options_exposes_each_box_once():
    data = pd.DataFrame({
        "souche": ["P0-lux", "P0-lux", "PspeD-lux", "PspeD-lux"],
        "Groupe": ["Experiment 1 | SCFM2", "Experiment 2 | SCFM2",
                   "Experiment 1 | SCFM2", "Experiment 2 | SCFM2"],
    })

    assert directional_condition_options(data) == {
        "P0 · SCFM2": "P0-lux\0SCFM2",
        "PspeD · SCFM2": "PspeD-lux\0SCFM2",
    }
    assert directional_comparison_options(data) == {
        "P0 · SCFM2 > PspeD · SCFM2": ("P0-lux\0SCFM2", "PspeD-lux\0SCFM2"),
        "PspeD · SCFM2 > P0 · SCFM2": ("PspeD-lux\0SCFM2", "P0-lux\0SCFM2"),
    }


@pytest.mark.parametrize(("intervals", "level_count"), [
    ([(0, 1), (1, 2)], 1),
    ([(0, 1), (1, 2), (0, 2)], 2),
    ([(0, 1), (2, 3)], 1),
    ([(0, 2), (1, 3)], 2),
    ([(0, 3), (1, 2)], 2),
])
def test_bracket_levels_use_interior_overlap(intervals, level_count):
    levels = _bracket_levels(intervals)
    assert max(levels) + 1 == level_count


def test_adjacent_brackets_keep_level_and_get_symmetric_drawing_gap():
    intervals = [(0, 1), (1, 2), (0, 2)]
    levels = _bracket_levels(intervals)
    coordinates = _bracket_draw_coordinates(intervals, levels)

    assert levels[:2] == [0, 0]
    assert coordinates[:2] == [
        (0.0, 1.0 - BRACKET_ENDPOINT_GAP),
        (1.0 + BRACKET_ENDPOINT_GAP, 2.0),
    ]
    assert coordinates[2] == (0.0, 2.0)


def test_curve_and_boxplot_defaults_are_independent():
    assert DEFAULT_CURVE_Y_SCALE == "linear"
    assert DEFAULT_BOXPLOT_Y_SCALE == "log"
    assert inspect.signature(plot_publication_panels).parameters["y_scale"].default == "linear"
    assert inspect.signature(plot_metric_points).parameters["y_scale"].default == "log"
    statistical_defaults = inspect.signature(paired_directional_t_tests).parameters
    assert statistical_defaults["alternative"].default == "two-sided"
    assert statistical_defaults["transform"].default == "log10"


def test_bracket_drawing_gap_does_not_change_inferential_results():
    metrics = pd.DataFrame([
        {"experience_id": experiment, "souche": condition, "Groupe": "M1",
         "lum_norm_auc": value}
        for experiment, values in enumerate(((10, 20, 40), (12, 18, 45), (9, 22, 38)), 1)
        for condition, value in zip(("P0", "PspeD2-1A", "PspeD2-3B"), values)
    ])
    comparisons = (("P0", "PspeD2-1A"), ("PspeD2-1A", "PspeD2-3B"))
    expected = paired_directional_t_tests(
        metrics, value="lum_norm_auc", condition="souche", identity=("experience_id",),
        comparisons=comparisons, transform="log10", alternative="two-sided",
    )
    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", directional_comparisons=comparisons,
        statistical_transform="log10", alternative="two-sided",
    )
    actual = figure._luxplate_statistics

    columns = ["condition_1", "condition_2", "statistic", "p_raw", "p_adjusted", "significance"]
    pd.testing.assert_frame_equal(actual[columns], expected[columns])
    bracket_lines = [line for line in figure.axes[0].lines if len(line.get_xdata()) == 4]
    horizontal_endpoints = [(line.get_xdata()[0], line.get_xdata()[2]) for line in bracket_lines[-2:]]
    assert horizontal_endpoints == [
        (0.0, 1.0 - BRACKET_ENDPOINT_GAP),
        (1.0 + BRACKET_ENDPOINT_GAP, 2.0),
    ]
    plt.close(figure)


def test_directional_condition_options_excludes_blank_controls():
    data = pd.DataFrame({
        "souche": ["P0-lux", "Blanc", "blank-labelled strain"],
        "Groupe": ["BM2", "BM2", "BM2"],
        "type": ["souche", " blanc ", "souche"],
    })

    assert directional_condition_options(data) == {
        "P0 · BM2": "P0-lux\0BM2",
        "blank-labelled strain · BM2": "blank-labelled strain\0BM2",
    }


def test_comparison_identifier_repr_contains_the_real_nul_separator():
    identifiers = _comparison_identifiers(
        pd.Series(["14.1Ac attB::PspeD2-1A-lux"]), pd.Series(["BM2"])
    )

    assert repr(identifiers.iloc[0]) == "'14.1Ac attB::PspeD2-1A-lux\\x00BM2'"


def test_collect_publication_statistics_makes_one_visible_table():
    first = plt.figure()
    first._luxplate_statistics = pd.DataFrame([
        {"condition_1": "A\0M1", "condition_2": "B\0M1", "p_raw": .02}
    ])
    second = plt.figure()  # Time-course figures intentionally have no statistics.
    third = plt.figure()
    third._luxplate_statistics = pd.DataFrame([
        {"condition_1": "A\0M2", "condition_2": "B\0M2", "p_raw": .03}
    ])

    result = collect_publication_statistics([
        ("peak", first), ("growth", second), ("auc", third)
    ])

    assert result["figure"].tolist() == ["peak", "auc"]
    assert result["p_raw"].tolist() == [.02, .03]
    plt.close("all")


def test_collect_publication_statistics_is_empty_without_metric_tests():
    figure = plt.figure()
    assert collect_publication_statistics([("growth", figure)]).empty
    plt.close(figure)


def test_publication_figure_dimensions_can_be_scaled_independently():
    data = pd.DataFrame({
        "temps_h": [0.0, 1.0], "souche": ["P0-lux", "P0-lux"],
        "sample_header": ["P0", "P0"], "Groupe": ["SCFM2", "SCFM2"],
        "DO_corr": [0.1, 0.2], "type": ["souche", "souche"],
    })
    default = build_publication_figures(data, families=("growth",))[0][1]
    resized = build_publication_figures(
        data, families=("growth",), width_scale=1.25, height_scale=1.75
    )[0][1]

    np.testing.assert_allclose(
        resized.get_size_inches(), default.get_size_inches() * [1.25, 1.75]
    )
    plt.close(default)
    plt.close(resized)


@pytest.mark.parametrize(("width_scale", "height_scale"), [(0, 1), (1, 0), (-1, 1)])
def test_publication_figure_dimensions_must_be_positive(width_scale, height_scale):
    with pytest.raises(ValueError, match="must be positive"):
        build_publication_figures(
            workflow_table(), families=("growth",),
            width_scale=width_scale, height_scale=height_scale,
        )


def test_metric_points_can_show_every_technical_replicate_or_hide_them():
    metrics = pd.DataFrame({
        "souche": ["P0-lux"] * 25,
        "Groupe": ["SCFM2"] * 25,
        "replicat": range(1, 26),
        "lum_norm_peak": np.arange(1, 26, dtype=float),
    })

    visible = plot_metric_points(metrics, metric="lum_norm_peak")
    hidden = plot_metric_points(
        metrics, metric="lum_norm_peak", show_technical_replicates=False
    )

    # The boxplot creates other Line2D artists; technical points are the only
    # line containing all 25 observations at once.
    assert any(len(line.get_ydata()) == 25 for line in visible.axes[0].lines)
    assert not any(len(line.get_ydata()) == 25 for line in hidden.axes[0].lines)
    assert [text.get_text() for text in hidden.axes[0].get_legend().get_texts()] == [
        "Biological mean"
    ]
    plt.close(visible)
    plt.close(hidden)


@pytest.mark.parametrize("y_scale", ["linear", "log"])
@pytest.mark.parametrize("statistical_transform", ["none", "log10"])
def test_axis_scale_and_statistical_transform_are_independent(y_scale, statistical_transform):
    metrics = pd.DataFrame([
        {"experience_id": experiment, "souche": condition, "Groupe": "M1",
         "lum_norm_auc": value}
        for experiment, left, right in ((1, 2, 1), (2, 8, 2), (3, 12, 10), (4, 150, 100))
        for condition, value in (("A", left), ("B", right))
    ])
    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", y_scale=y_scale,
        statistical_transform=statistical_transform,
        directional_comparisons=(("A", "B"),),
    )
    result = figure._luxplate_statistics.iloc[0]
    records = json.loads(result["paired_values_json"])
    expected = np.array([2, 8, 12, 150], dtype=float)
    if statistical_transform == "log10":
        expected = np.log10(expected)
    assert figure.axes[0].get_yscale() == y_scale
    np.testing.assert_allclose([row["condition_1_transformed"] for row in records], expected)
    assert result["y_scale"] == y_scale
    plt.close(figure)


def test_changing_only_y_scale_leaves_statistics_strictly_unchanged():
    metrics = pd.DataFrame([
        {"experience_id": experiment, "souche": condition, "Groupe": "M1",
         "lum_norm_auc": value}
        for experiment, left, right in ((1, 2, 1), (2, 8, 2), (3, 12, 10), (4, 150, 100))
        for condition, value in (("A", left), ("B", right))
    ])
    kwargs = dict(metric="lum_norm_auc", statistical_transform="log10",
                  directional_comparisons=(("A", "B"),))
    linear = plot_metric_points(metrics, y_scale="linear", **kwargs)
    logged = plot_metric_points(metrics, y_scale="log", **kwargs)
    left, right = linear._luxplate_statistics.iloc[0], logged._luxplate_statistics.iloc[0]
    assert left["statistic"] == pytest.approx(right["statistic"])
    assert left["p_raw"] == pytest.approx(right["p_raw"])
    assert left["p_adjusted"] == pytest.approx(right["p_adjusted"])
    plt.close(linear); plt.close(logged)


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


def test_guided_corrected_figures_plot_corrected_values_and_labels():
    data = workflow_table().assign(DO_corr=0.25, Lum_corr=42.0)

    figure = build_guided_corrected_figures(data, sample_type="souche")[0][1]

    assert set(figure.axes[0].lines[0].get_ydata()) == {0.25}
    assert set(figure.axes[1].lines[0].get_ydata()) == {42.0}
    assert figure.axes[0].get_title() == "Densité optique corrigée"
    assert figure.axes[1].get_title() == "Luminescence corrigée"
    assert "corrigée" in figure.axes[0].get_ylabel()
    assert "corrigée" in figure.axes[1].get_ylabel()
    plt.close(figure)


def test_guided_crosstalk_figures_show_intermediate_signal_and_label():
    data = workflow_table().assign(RLU_corrected=-123.0)

    figure = build_guided_crosstalk_figures(data, sample_type="blanc")[0][1]

    assert set(figure.axes[1].lines[0].get_ydata()) == {-123.0}
    assert figure.axes[1].get_title() == "Luminescence après correction du cross-talk"
    assert "après cross-talk" in figure.axes[1].get_ylabel()
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
        metrics, metric="lum_norm_auc", group_by="Groupe", title="Normalized AUC",
        directional_comparisons=(("Reporter-lux", "P0-lux"),),
    )

    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["DMEM", "SCFM2"]
    labels = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert any(label in {"ns", "*", "**", "***", "****"} for label in labels)
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
    # Six compatible boxes produce every unordered pair once.
    assert len(figure._luxplate_statistics) == 15
    plt.close(figure)


@pytest.mark.parametrize(("condition_count", "rotation", "font_size"), [
    (3, 0, 10),
    (8, 30, 9),
    (15, 45, 8),
])
def test_comparison_tick_labels_adapt_without_changing_categories_or_statistics(
        condition_count, rotation, font_size):
    metrics = pd.DataFrame([
        {"souche": f"Reporter {condition}", "Groupe": f"Medium {condition}",
         "experience_id": f"exp{experiment}", "lum_norm_auc": 10 + condition + experiment}
        for experiment in range(1, 4)
        for condition in range(condition_count)
    ])
    expected_conditions = [
        f"Reporter {condition}\0Medium {condition}" for condition in range(condition_count)
    ]
    hypotheses = ((expected_conditions[1], expected_conditions[0]),)
    expected_statistics = paired_directional_t_tests(
        metrics.assign(_comparison=_comparison_identifiers(
            metrics["souche"], metrics["Groupe"])),
        value="lum_norm_auc", condition="_comparison", identity=("experience_id",),
        comparisons=hypotheses, transform="log10", alternative="two-sided",
    )

    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True,
        directional_comparisons=hypotheses,
    )
    axis = figure.axes[0]
    labels = axis.get_xticklabels()

    assert [label.get_text() for label in labels] == [
        f"Reporter {condition}\nMedium {condition}" for condition in range(condition_count)
    ]
    assert [label.get_rotation() for label in labels] == [rotation] * condition_count
    assert [label.get_fontsize() for label in labels] == [font_size] * condition_count
    assert [label.get_rotation_mode() for label in labels] == (
        ["default"] * condition_count if condition_count <= 6 else ["anchor"] * condition_count
    )
    np.testing.assert_allclose(axis.get_xticks(), range(condition_count))
    columns = ["condition_1", "condition_2", "statistic", "p_raw", "p_adjusted"]
    pd.testing.assert_frame_equal(
        figure._luxplate_statistics[columns], expected_statistics[columns]
    )
    plt.close(figure)


def test_metric_points_only_draws_selected_directional_hypotheses():
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": medium, "experience_id": f"exp{experiment}",
         "replicat": 1, "lum_norm_auc": experiment + strain_index * 10 + medium_index * 3}
        for experiment in range(1, 7)
        for strain_index, strain in enumerate(("A", "B", "C"))
        for medium_index, medium in enumerate(("M1", "M2"))
    ])

    conditions = (("A\0M2", "A\0M1"),)
    selected = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True,
        directional_comparisons=conditions,
    )
    assert len(selected._luxplate_statistics) == 1
    comparison = selected._luxplate_statistics.iloc[0]
    assert (comparison.condition_1, comparison.condition_2) == conditions[0]
    labels = [text.get_text() for text in selected.axes[0].texts
              if text.get_text() in {"ns", "*", "**", "***", "****"}]
    assert len(labels) == 1
    plt.close(selected)


def test_metric_points_treats_legacy_experience_as_biological_unit():
    metrics = pd.DataFrame([
        {"souche": "PspeD2-1A", "Groupe": medium, "experience": f"Rep{experiment}",
         "replicat": technical, "lum_norm_auc": value}
        for experiment, without_spd, with_spd in (
            (1, 1.243e8, 3.019e8), (2, 1.303e8, 2.033e8), (3, 1.101e8, 1.921e8)
        )
        for technical in (1, 2, 3)
        for medium, value in (("SCFM2-KPi", without_spd),
                              ("SCFM2-KPi (Spd)", with_spd))
    ])
    comparison = (("PspeD2-1A\0SCFM2-KPi (Spd)",
                   "PspeD2-1A\0SCFM2-KPi"),)
    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True,
        directional_comparisons=comparison,
    )

    result = figure._luxplate_statistics.iloc[0]
    assert result["pairing_columns"] == "experience"
    assert result["n_pairs"] == 3
    assert "*" in [text.get_text() for text in figure.axes[0].texts]
    plt.close(figure)


def test_bm2_auc_three_validated_hypotheses_produce_rows_and_brackets():
    """Reproduce the reported P0/PspeD2 BM2 guided-analysis scenario."""
    strains = ("P0-lux", "PspeD2-1A-lux", "PspeD2-3B-lux")
    metrics = pd.DataFrame([
        {
            "souche": strain,
            "Groupe": f"Experiment {experiment} | BM2",
            "experience_id": f"exp{experiment}",
            "replicat": 1,
            "lum_norm_auc": value,
        }
        for experiment, values in enumerate((
            (10, 18, 14), (12, 25, 20), (9, 22, 16), (14, 31, 24)
        ), start=1)
        for strain, value in zip(strains, values)
    ])
    # These are the exact identifiers returned by the guided UI, rather than
    # identifiers constructed independently by the test.
    conditions = directional_condition_options(metrics)
    p0 = conditions["P0 · BM2"]
    d21 = conditions["PspeD2-1A · BM2"]
    d23 = conditions["PspeD2-3B · BM2"]
    hypotheses = ((d21, p0), (d23, p0), (d21, d23))

    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True,
        title="Luminescence AUC / OD AUC",
        directional_comparisons=hypotheses,
    )

    statistics = figure._luxplate_statistics
    assert len(statistics) == 3
    assert list(zip(statistics["condition_1"], statistics["condition_2"])) == list(hypotheses)
    assert statistics["calculation_status"].eq("calculé").all()
    annotation_labels = [
        text.get_text() for text in figure.axes[0].texts
        if text.get_text() in {"NA", "ns", "*", "**", "***", "****"}
    ]
    assert len(annotation_labels) == 3
    # Each annotation label is paired with one four-segment bracket line.
    bracket_lines = [
        line for line in figure.axes[0].lines
        if len(line.get_xdata()) == 4 and np.asarray(line.get_ydata()).max() < 1
    ]
    assert len(bracket_lines) == 3
    plt.close(figure)


def test_real_ui_nul_conditions_keep_three_workbook_biological_pairs():
    """Exercise raw UI identifiers through the complete metric statistics path."""
    full_strains = (
        "14.1Ac attB::P0-lux",
        "14.1Ac attB::PspeD2-1A-lux",
        "14.1Ac attB::PspeD2-3B-lux",
    )
    raw = pd.DataFrame([
        {"souche": strain, "Groupe": f"exp{bio}|BM2"}
        for bio in range(1, 4) for strain in full_strains
    ])
    ui_conditions = directional_condition_options(raw)
    p0 = ui_conditions["P0 · BM2"]
    d21 = ui_conditions["PspeD2-1A · BM2"]
    d23 = ui_conditions["PspeD2-3B · BM2"]
    hypotheses = ((d21, d23), (d21, p0), (d23, p0))
    assert all("\0" in condition for pair in hypotheses for condition in pair)

    # The uploaded workbook name is the biological unit.  Replicate and well
    # values deliberately repeat in every workbook and are technical only.
    metrics = pd.DataFrame([
        {
            "souche": strain,
            "Groupe": f"exp{bio}|BM2",
            "experience": experience,
            "replicat": technical,
            "puits": f"{chr(65 + technical)}0{bio}",
            "lum_norm_auc": (10 + strain_index * 5 + bio) * (1 + technical / 100),
        }
        for bio, experience in enumerate((
            "260403_BM2_testsScreening", "070826_BM2_LB", "140826_BM2_LB_Rep3"
        ), start=1)
        for strain_index, strain in enumerate(full_strains)
        for technical in range(1, 4)
    ])
    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True,
        directional_comparisons=hypotheses,
    )
    result = figure._luxplate_statistics

    # On regression, expose every exact value involved rather than hiding the
    # NUL separator in an ordinary assertion diff or CSV renderer.
    pivot_columns = list(dict.fromkeys(
        metrics["souche"].astype(str) + "\0" + metrics["Groupe"].map(
            lambda group: group.split("|", 1)[-1]
        )
    ))
    diagnostic = "\n".join([
        f"requested left = {d21!r}",
        f"canonical left = {_canonical_condition(d21)!r}",
        "pivot columns:",
        *(f"  {column!r} -> {_canonical_condition(column)!r}" for column in pivot_columns),
    ])
    assert result["n_pairs"].tolist() == [3, 3, 3], diagnostic
    assert result["calculation_status"].eq("calculé").all(), diagnostic
    assert result["pairing_columns"].eq("experience").all(), diagnostic
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


def test_long_medium_names_wrap_in_time_course_panel_titles():
    medium = "Experiment 1 | DMEM-KPI (62mM) + 10% SVF + 5% SCFM2-KPI"
    data = pd.concat([
        _publication_table().assign(Groupe=medium),
        _publication_table().assign(Groupe=medium.replace("62mM", "31mM")),
    ], ignore_index=True)

    publication = plot_publication_panels(data, value="DO_corr")
    mixed = plot_mixed_panels(data)

    assert all("\n" in axis.get_title() for axis in publication.axes)
    assert all("\n" in axis.get_title() for axis in mixed.axes if axis.get_title())
    assert "DMEM-KPI" in publication.axes[0].get_title()
    publication.canvas.draw()
    mixed.canvas.draw()
    plt.close(publication)
    plt.close(mixed)


def test_long_medium_names_wrap_in_metric_condition_labels():
    medium = "DMEM-KPI (62mM) + 10% SVF + 5% SCFM2-KPI"
    metrics = pd.DataFrame([
        {"souche": strain, "Groupe": medium, "experience_id": "exp1",
         "replicat": 1, "lum_norm_auc": value}
        for strain, value in (("P0-lux", 10.0), ("Reporter-lux", 20.0))
    ])

    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True, title="AUC"
    )

    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert all(label.count("\n") >= 2 for label in labels)
    assert all("SCFM2-KPI" in label for label in labels)
    figure.canvas.draw()
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


def test_pooled_summary_matches_acquisition_sequence_across_shifted_experiments():
    first = _publication_table().query("souche == 'Reporter-lux'").assign(
        experience_id="exp1", Lum_corr=lambda frame: frame["Lum_corr"] * 1.0
    )
    second = _publication_table().query("souche == 'Reporter-lux'").assign(
        experience_id="exp2", temps_h=lambda frame: frame["temps_h"] + 0.05,
        Lum_corr=lambda frame: frame["Lum_corr"] * 3.0,
    )

    summary = _aligned_biological_summary(pd.concat([first, second]), "Lum_corr")

    # The three corresponding acquisitions have N=2 and retain the genuine
    # between-experiment dispersion despite their three-minute clock offset.
    assert summary["n_biological"].tolist() == [2, 2, 2]
    assert summary["std"].tolist() == pytest.approx([
        np.std([120, 360], ddof=1),
        np.std([170, 510], ddof=1),
        np.std([220, 660], ddof=1),
    ])


def test_pooled_summary_uses_group_experiments_not_technical_wells_for_sd():
    data = pd.DataFrame([
        {"Groupe": f"Experiment {experiment} | LB", "replicat": 1,
         "sample_header": f"PspeD2-1A-{well}", "temps_h": 7.0,
         "Lum_corr": biological_value + technical_offset}
        for experiment, biological_value in enumerate((2.3e6, 1.5e6, 4.4e6), start=1)
        for well, technical_offset in (("A1", -1e4), ("A2", 1e4))
    ])

    summary = _aligned_biological_summary(data, "Lum_corr")

    assert summary.loc[0, "n_biological"] == 3
    assert summary.loc[0, "mean"] == pytest.approx(np.mean([2.3e6, 1.5e6, 4.4e6]))
    assert summary.loc[0, "std"] == pytest.approx(np.std([2.3e6, 1.5e6, 4.4e6], ddof=1))


def test_technical_replicate_numbers_never_inflate_biological_sd():
    data = pd.DataFrame([
        {"Groupe": f"Experiment {experiment} | LB", "replicat": technical,
         "sample_header": f"well-{technical}", "temps_h": 1.0,
         "Lum_corr": biological_value + technical_offset}
        for experiment, biological_value in ((1, 100.0), (2, 200.0), (3, 400.0))
        for technical, technical_offset in ((1, -10.0), (2, 10.0))
    ])

    summary = _aligned_biological_summary(data, "Lum_corr")

    assert summary.loc[0, "n_biological"] == 3
    assert summary.loc[0, "mean"] == pytest.approx(np.mean([100, 200, 400]))
    assert summary.loc[0, "std"] == pytest.approx(np.std([100, 200, 400], ddof=1))


def test_unidentified_technical_series_do_not_manufacture_biological_sd():
    data = pd.DataFrame({
        "Groupe": ["LB", "LB"], "replicat": [1, 2],
        "sample_header": ["well-1", "well-2"], "temps_h": [1.0, 1.0],
        "Lum_corr": [90.0, 110.0],
    })

    summary = _aligned_biological_summary(data, "Lum_corr")

    assert summary.loc[0, "n_biological"] == 1
    assert summary.loc[0, "mean"] == pytest.approx(100.0)
    assert pd.isna(summary.loc[0, "std"])


def test_one_repeated_legacy_experiment_does_not_create_a_pooled_recap():
    data = _publication_table().assign(Groupe="Experiment 1 | DMEM")

    figures = build_publication_figures(data, families=("corrected",))

    assert [name for name, _ in figures] == ["luminescence_corrigee"]
    plt.close(figures[0][1])


def test_legacy_experiment_sd_survives_corrected_recap_and_mixed_pooling():
    biological_values = (2.3e6, 1.5e6, 4.4e6)
    data = pd.DataFrame([
        {"Groupe": f"Experiment {experiment} | LB", "replicat": 1,
         "sample_header": f"PspeD2-1A-{well}", "puits": well, "temps_h": 7.0,
         "souche": "PspeD2-lux", "type": "souche", "DO_corr": od_value,
         "Lum_corr": biological_value + technical_offset}
        for experiment, (biological_value, od_value) in enumerate(
            zip(biological_values, (.2, .4, .8)), start=1
        )
        for well, technical_offset in (("A1", -1e4), ("A2", 1e4))
    ])

    figures = dict(build_publication_figures(data, families=("corrected", "mixed")))
    expected_sd = np.std(biological_values, ddof=1)

    corrected = figures["luminescence_corrigee_moyenne_experiences"]
    mixed = figures["croissance_luminescence_mixte"]
    corrected_segment = corrected.axes[0].containers[0].lines[2][0].get_segments()[0]
    lum_axis = next(axis for axis in mixed.axes if axis.get_ylabel() == "Luminescence (RLU)")
    mixed_segment = lum_axis.containers[0].lines[2][0].get_segments()[0]
    assert np.diff(corrected_segment[:, 1])[0] / 2 == pytest.approx(expected_sd)
    assert np.diff(mixed_segment[:, 1])[0] / 2 == pytest.approx(expected_sd)
    for figure in figures.values():
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

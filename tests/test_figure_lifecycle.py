from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from pandas.testing import assert_frame_equal

from luxplate.export import figure_bytes
from luxplate.figure_lifecycle import (
    filter_result_strains,
    invalidate_guided_analysis_state,
    validate_figure_render,
)
from luxplate.plotting import build_publication_figures, directional_condition_options


def publication_table():
    rows = []
    for strain_index, strain in enumerate(("P0-lux", "Reporter-lux")):
        for well in ("A1", "A2"):
            for time in (0.0, 1.0, 2.0):
                od = 0.1 + 0.08 * time + 0.01 * strain_index
                rows.append({
                    "temps_h": time, "souche": strain, "Groupe": "SCFM2",
                    "sample_header": f"{strain}-{well}", "puits": well,
                    "replicat": 1, "experience_id": "exp", "type": "souche",
                    "DO_corr": od, "Lum_corr": 100 + 50 * time + 20 * strain_index,
                    "Lum_norm": (100 + 50 * time + 20 * strain_index) / od,
                })
    return pd.DataFrame(rows)


def auc_figure(data, **statistics):
    if "directional_comparisons" not in statistics:
        conditions = list(directional_condition_options(data).values())
        statistics["directional_comparisons"] = ((conditions[0], conditions[1]),)
    figures = dict(build_publication_figures(data, families=("auc",), **statistics))
    return figures["auc_luminescence_normalisee"]


def assert_visible_auc(figure):
    assert figure.axes
    assert any(
        axis.lines or axis.collections or axis.patches or axis.artists
        for axis in figure.axes
    )
    assert hasattr(figure, "_luxplate_statistics")
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.tell() > 0


def test_result_strain_navigation_rebuilds_auc_without_replacing_complete_result():
    """A result filter changes figures, not the completed scientific analysis."""
    normalized = publication_table()
    complete_result = object()
    original_result_id = id(complete_result)

    strain_a = filter_result_strains(normalized, ["P0-lux"])
    figure_a = auc_figure(strain_a, directional_comparisons=())
    strain_b = filter_result_strains(normalized, ["Reporter-lux"])
    figure_b = auc_figure(strain_b, directional_comparisons=())
    strain_a_again = filter_result_strains(normalized, ["P0-lux"])
    figure_a_again = auc_figure(strain_a_again, directional_comparisons=())

    assert id(complete_result) == original_result_id
    assert strain_a["souche"].unique().tolist() == ["P0-lux"]
    assert strain_b["souche"].unique().tolist() == ["Reporter-lux"]
    assert_frame_equal(strain_a.reset_index(drop=True), strain_a_again.reset_index(drop=True))
    for figure in (figure_a, figure_b, figure_a_again):
        assert_visible_auc(figure)
        plt.close(figure)


def test_selection_change_discards_all_old_directional_widget_state():
    removed = "P0\0SCFM2 (Aa)"
    retained = "P0\0SCFM2"
    state = {
        "guided_complete_result": object(),
        "guided_decisions": object(),
        "guided_figure_view": "Luminescence corrigée",
        "guided_directional_comparisons_stack": [(removed, retained)],
        "guided_directional_comparisons_validated": [(removed, retained)],
        "guided_directional_comparisons_reference": "P0 · SCFM2 (Aa)",
        "guided_directional_comparisons_comparators": ["P0 · SCFM2"],
        "guided_directional_comparisons_medium_a": "SCFM2 (Aa)",
        "guided_directional_comparisons_medium_b": "SCFM2",
        "guided_directional_comparisons_control": "P0",
        "guided_publication_figure_signature": (("alternative", "greater"),),
        "guided_publication_statistics": pd.DataFrame({"old": [True]}),
        "guided_publication_diagnostics": [{"old": True}],
        "guided_figure_lum_scale": "Logarithmique (base 10)",
        "guided_figure_width_percent": 120,
    }

    invalidate_guided_analysis_state(state)

    assert not any(key.startswith("guided_directional_comparisons") for key in state)
    assert "guided_complete_result" not in state
    assert "guided_decisions" not in state
    assert "guided_figure_view" not in state
    assert state == {
        "guided_figure_lum_scale": "Logarithmique (base 10)",
        "guided_figure_width_percent": 120,
    }


def test_legacy_gallery_is_removed_from_state_without_clearing_its_figure():
    figure = auc_figure(publication_table())
    state = {"guided_publication_figures": [("auc", figure)]}

    invalidate_guided_analysis_state(state)

    assert state == {}
    assert_visible_auc(figure)
    plt.close(figure)


def test_removed_medium_can_be_followed_by_rebuilt_nonempty_auc_figures():
    media_a = ["SCFM2", "SCFM2 (i)", "SCFM2 (M)", "SCFM2 (Aa)", "SCFM2 (c)", "SCFM2-KPi"]
    media_b = [medium for medium in media_a if medium != "SCFM2 (Aa)"]
    base = publication_table()
    cycles = []
    for media in (media_a, media_b, media_b):
        data = pd.concat([base.assign(Groupe=medium) for medium in media], ignore_index=True)
        figures = build_publication_figures(
            data, families=("growth", "corrected", "mixed", "auc"),
            statistical_transform="log10", alternative="two-sided",
        )
        assert figures
        diagnostics = [validate_figure_render(figure) for _, figure in figures]
        assert all(item["axes"] > 0 and item["png_size"] > 0 for item in diagnostics)
        assert any(name == "auc_luminescence_normalisee" for name, _ in figures)
        cycles.append((data, figures))

    final_options = directional_condition_options(cycles[-1][0])
    assert all("SCFM2 (Aa)" not in label for label in final_options)
    assert all("SCFM2 (Aa)" not in identifier for identifier in final_options.values())
    for _, figures in cycles:
        for _, figure in figures:
            plt.close(figure)


def test_auc_remains_visible_after_statistical_parameters_change():
    data = pd.concat(
        [publication_table().assign(experience_id=f"exp-{index}") for index in range(3)],
        ignore_index=True,
    )

    first = auc_figure(data, statistical_transform="log10", alternative="two-sided")
    rebuilt = auc_figure(data, statistical_transform="none", alternative="greater")

    assert first is not rebuilt
    assert_visible_auc(first)
    assert_visible_auc(rebuilt)
    assert not rebuilt._luxplate_statistics.empty
    assert rebuilt._luxplate_statistics["alternative"].eq("greater").all()
    assert rebuilt._luxplate_statistics["statistical_transform"].eq("none").all()
    plt.close(first)
    plt.close(rebuilt)


def test_auc_remains_visible_when_comparisons_are_not_calculable():
    # One biological experiment cannot support a paired t-test, but its boxes
    # and points must still be rendered and its diagnostic row retained.
    figure = auc_figure(
        publication_table(), statistical_transform="log10", alternative="two-sided"
    )

    assert_visible_auc(figure)
    assert not figure._luxplate_statistics.empty
    assert figure._luxplate_statistics["p_raw"].isna().all()
    assert figure._luxplate_statistics["calculation_status"].eq("non calculable").all()
    plt.close(figure)


def test_hypothesis_reruns_build_fresh_figures_statistics_and_brackets():
    data = pd.concat(
        [publication_table().assign(experience_id=f"exp-{index}") for index in range(3)],
        ignore_index=True,
    )
    data = pd.concat(
        [data.assign(Groupe=medium) for medium in ("SCFM2", "SCFM2-KPi", "SCFM2 (i)",
                                                   "SCFM2 (M)", "SCFM2 (c)")],
        ignore_index=True,
    )
    conditions = tuple(directional_condition_options(data).values())
    stacks = (
        tuple((conditions[0], item) for item in conditions[1:9]),
        tuple((conditions[0], item) for item in conditions[1:5]),
        tuple((conditions[1], item) for item in conditions[5:7]),
        (),
        tuple((conditions[2], item) for item in conditions[6:10]),
    )
    former = None
    former_preview = None

    for comparisons in stacks:
        figure = auc_figure(
            data,
            directional_comparisons=comparisons,
            statistical_transform="log10",
            alternative="greater",
        )

        if former is not None:
            assert former is not figure
            # Building the next render must not mutate a Figure that the
            # Streamlit frontend from the preceding run may still reference.
            assert_visible_auc(former)
        statistics = figure._luxplate_statistics
        expected = [tuple(tuple(item.split("\0", 1)) for item in pair) for pair in comparisons]
        assert list(zip(statistics["condition_1"], statistics["condition_2"])) == expected
        assert len(figure._luxplate_statistical_diagnostics) == len(comparisons)
        bracket_lines = [
            line for axis in figure.axes for line in axis.lines
            if len(line.get_xdata()) == 4 and len(line.get_ydata()) == 4
        ]
        assert len(bracket_lines) == len(comparisons)

        axes_before = tuple(figure.axes)
        artist_counts_before = tuple(
            len(axis.lines) + len(axis.collections) + len(axis.patches) + len(axis.artists)
            for axis in figure.axes
        )
        statistics_before = statistics.copy(deep=True)
        diagnostics_before = tuple(figure._luxplate_statistical_diagnostics)
        preview = figure_bytes(figure, "png", dpi=150)

        assert preview.startswith(b"\x89PNG")
        assert len(preview) > 0
        assert tuple(figure.axes) == axes_before
        assert tuple(
            len(axis.lines) + len(axis.collections) + len(axis.patches) + len(axis.artists)
            for axis in figure.axes
        ) == artist_counts_before
        assert_frame_equal(figure._luxplate_statistics, statistics_before)
        assert tuple(figure._luxplate_statistical_diagnostics) == diagnostics_before
        if former_preview is not None:
            assert preview != former_preview

        # Preview generation must leave this exact Figure reusable by all
        # publication export formats.
        for export_format, signature in (
            ("svg", b"<svg"), ("pdf", b"%PDF"),
            ("tiff", (b"II", b"MM")),
        ):
            payload = figure_bytes(figure, export_format, dpi=72)
            if isinstance(signature, tuple):
                assert payload[:2] in signature
            else:
                assert signature in payload[:500]

        former = figure
        former_preview = preview

    plt.close(former)

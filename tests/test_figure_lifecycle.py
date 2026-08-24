from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

from luxplate.figure_lifecycle import (
    PUBLICATION_FIGURES_KEY,
    PUBLICATION_SIGNATURE_KEY,
    invalidate_guided_analysis_state,
    publication_figure_signature,
    rebuild_publication_figures,
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


def test_hypothesis_reruns_replace_figures_statistics_and_brackets():
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
        tuple((conditions[1], item) for item in conditions[5:8]),
        (),
    )
    state = {}
    former = None

    for comparisons in stacks:
        signature = publication_figure_signature(
            directional_comparisons=comparisons,
            statistical_transform="log10",
            alternative="greater",
        )
        figure = rebuild_publication_figures(
            state,
            signature,
            lambda comparisons=comparisons: [(
                "auc_luminescence_normalisee",
                auc_figure(
                    data,
                    directional_comparisons=comparisons,
                    statistical_transform="log10",
                    alternative="greater",
                ),
            )],
        )[0][1]

        if former is not None:
            assert former is not figure
            assert not former.axes
        statistics = figure._luxplate_statistics
        expected = [tuple(tuple(item.split("\0", 1)) for item in pair) for pair in comparisons]
        assert list(zip(statistics["condition_1"], statistics["condition_2"])) == expected
        assert len(figure._luxplate_statistical_diagnostics) == len(comparisons)
        bracket_lines = [
            line for axis in figure.axes for line in axis.lines
            if len(line.get_xdata()) == 4 and len(line.get_ydata()) == 4
        ]
        assert len(bracket_lines) == len(comparisons)
        assert state[PUBLICATION_FIGURES_KEY][0][1] is figure
        assert state[PUBLICATION_SIGNATURE_KEY] == signature
        former = figure

    plt.close(former)


def test_figure_signature_tracks_statistical_content_and_hypothesis_order():
    comparisons = (("reporter\0medium-a", "reporter\0medium-b"),
                   ("reporter\0medium-a", "reporter\0medium-c"))
    baseline = publication_figure_signature(
        directional_comparisons=comparisons,
        statistical_transform="log10",
        alternative="greater",
    )

    assert baseline != publication_figure_signature(
        directional_comparisons=tuple(reversed(comparisons)),
        statistical_transform="log10", alternative="greater",
    )
    assert baseline != publication_figure_signature(
        directional_comparisons=comparisons,
        statistical_transform="none", alternative="greater",
    )
    assert baseline != publication_figure_signature(
        directional_comparisons=comparisons,
        statistical_transform="log10", alternative="two-sided",
    )

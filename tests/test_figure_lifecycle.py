import matplotlib.pyplot as plt
import pandas as pd

from luxplate.figure_lifecycle import invalidate_guided_analysis_state, validate_figure_render
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

import matplotlib.pyplot as plt
import pandas as pd

from luxplate.plotting import directional_condition_options, plot_metric_points
from luxplate.workflow import (build_bulk_point_decisions, build_manual_decisions,
                               filter_experiment_data, run_complete_analysis)


def workflow_table():
    rows = []
    for header, strain, kind, well, ods, lums in [
        ("S1 (A01)", "S1", "souche", "A01", [0.10, 0.20, 0.40, 0.80], [10, 30, 80, 120]),
        ("S2 (A02)", "S2", "souche", "A02", [0.10, 0.15, 0.20, 0.25], [8, 12, 15, 17]),
        ("Blanc (B01)", "Blanc", "blanc", "B01", [0.01] * 4, [1.0] * 4),
    ]:
        for time, od, lum in zip(range(4), ods, lums):
            rows.append({"temps_h": time, "souche": strain, "Groupe": "M1", "replicat": 1,
                         "DO_brute": od, "Lum_brute": lum, "type": kind, "puits": well,
                         "sample_header": header})
    return pd.DataFrame(rows)


def test_filter_keeps_selected_strain_and_required_blanks():
    selected = filter_experiment_data(workflow_table(), ["S1"])
    assert set(selected["souche"]) == {"S1", "Blanc"}


def test_filter_infers_only_groups_needed_by_selected_strains():
    data = workflow_table()
    unrelated = data.copy()
    unrelated["Groupe"] = "M2"
    unrelated["souche"] = unrelated["souche"].replace({"S1": "S3", "S2": "S4"})
    selected = filter_experiment_data(pd.concat([data, unrelated], ignore_index=True), ["S1"])
    assert set(selected["Groupe"]) == {"M1"}
    assert set(selected["souche"]) == {"S1", "Blanc"}


def test_filter_respects_selected_media_and_keeps_their_blanks():
    data = workflow_table()
    second_medium = data.copy()
    second_medium["Groupe"] = "M2"
    selected = filter_experiment_data(pd.concat([data, second_medium], ignore_index=True), ["S1"], ["M2"])
    assert set(selected["Groupe"]) == {"M2"}
    assert set(selected["souche"]) == {"S1", "Blanc"}


def test_manual_point_and_series_decisions_are_applied_end_to_end():
    data = filter_experiment_data(workflow_table(), ["S1"], ["M1"])
    point_index = data.index[(data["sample_header"] == "S1 (A01)") & (data["temps_h"] == 1)][0]
    decisions = build_manual_decisions(data, [point_index], [])
    result = run_complete_analysis(data, decisions, consecutive_points=2, growth_window_points=2)
    assert len(result.blank_correction.excluded_data) == 1
    assert len(result.kinetics.series_metrics) == 1


def test_manual_whole_curve_decision_removes_every_series_point():
    data = workflow_table()
    decisions = build_manual_decisions(data, [], ["S2 (A02)"])
    result = run_complete_analysis(data, decisions, consecutive_points=2, growth_window_points=2)
    assert len(result.blank_correction.excluded_data) == 4
    assert set(result.kinetics.series_metrics["souche"]) == {"S1"}


def test_manual_whole_blank_curve_decision_removes_every_blank_point():
    data = workflow_table()
    decisions = build_manual_decisions(data, [], ["Blanc (B01)"])
    result = run_complete_analysis(data, decisions, consecutive_points=2, growth_window_points=2)
    assert len(result.blank_correction.excluded_data) == 4
    assert result.blank_correction.excluded_data["type"].eq("blanc").all()


def test_complete_analysis_reports_each_calculation_stage():
    updates = []
    run_complete_analysis(
        workflow_table(), consecutive_points=2, growth_window_points=2,
        progress_callback=lambda percent, message: updates.append((percent, message)),
    )

    assert [percent for percent, _ in updates] == [5, 35, 70, 100]
    assert updates[-1][1] == "Calcul des paramètres cinétiques terminé"


def test_bulk_decisions_can_remove_first_time_from_every_blank_in_one_experiment():
    data = workflow_table().assign(experience="Expérience 2")
    decisions = build_bulk_point_decisions(
        data, experience="Expérience 2", times=[0.0], sample_type="blanc"
    )
    assert len(decisions) == 1
    assert decisions.iloc[0]["sample_header"] == "Blanc (B01)"
    result = run_complete_analysis(data, decisions, consecutive_points=2, growth_window_points=2)
    assert len(result.blank_correction.excluded_data) == 1
    assert result.blank_correction.excluded_data.iloc[0]["temps_h"] == 0


def test_three_uploaded_bm2_experiments_reach_paired_statistics_end_to_end():
    strains = (
        "14.1Ac attB::P0-lux",
        "14.1Ac attB::PspeD2-1A-lux",
        "14.1Ac attB::PspeD2-3B-lux",
    )
    experiences = (
        "260403_BM2_testsScreening",
        "070826_BM2_LB",
        "140826_BM2_LB_Rep3",
    )
    rows = []
    for experiment_number, experience in enumerate(experiences, start=1):
        group = f"exp{experiment_number}|BM2"
        for strain_number, strain in enumerate(strains, start=1):
            for technical in (1, 2):
                for time, od in enumerate((.1, .2, .4, .8)):
                    rows.append({
                        "experience": experience, "temps_h": time + experiment_number / 10,
                        "souche": strain, "Groupe": group, "replicat": technical,
                        "sample_header": f"exp{experiment_number}|{strain}|{technical}",
                        "puits": f"A{technical:02d}", "DO_brute": od,
                        "Lum_brute": od * (strain_number + experiment_number) * 100,
                        "type": "souche",
                    })
        for time in range(4):
            rows.append({
                "experience": experience, "temps_h": time + experiment_number / 10,
                "souche": "Blanc", "Groupe": group, "replicat": 1,
                "sample_header": f"exp{experiment_number}|blank", "puits": "H12",
                "DO_brute": .01, "Lum_brute": 1.0, "type": "blanc",
            })

    complete = run_complete_analysis(
        pd.DataFrame(rows), consecutive_points=3, growth_window_points=3
    )
    metrics = complete.kinetics.series_metrics
    assert metrics["experience"].nunique() == 3
    assert all(
        set(group["experience"]) == set(experiences)
        for _, group in metrics.groupby("souche")
    )

    conditions = directional_condition_options(metrics)
    p0 = conditions["P0 · BM2"]
    d21 = conditions["PspeD2-1A · BM2"]
    d23 = conditions["PspeD2-3B · BM2"]
    figure = plot_metric_points(
        metrics, metric="lum_norm_auc", compare_media=True,
        directional_comparisons=((d21, p0), (d23, p0), (d21, d23)),
    )
    biological = figure._luxplate_statistical_diagnostics[0]["biological_rows"]
    assert biological["experience"].nunique() == 3
    assert biological.iloc[0]["_comparison"][1] == "BM2"
    assert figure._luxplate_statistics["n_pairs"].tolist() == [3, 3, 3]
    plt.close(figure)

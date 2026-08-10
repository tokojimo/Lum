"""Streamlit entry point for the progressive legacy-pipeline integration."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from luxplate.blanks import run_blank_correction
from luxplate.normalization import run_normalization
from luxplate.kinetics import run_kinetics
from luxplate.export import package_figures
from luxplate.plotting import (build_guided_raw_figures, build_publication_figures, plot_blank_correction, plot_kinetics,
                               plot_normalization, plot_qc_curves, plot_raw_curves)
from luxplate.qc import run_quality_control
from luxplate.varioskan import combine_kinetic_tables, inspect_workbook, parse_kinetic_workbook
from luxplate.workflow import build_manual_decisions, filter_experiment_data, run_complete_analysis

st.set_page_config(page_title="LuxPlate Analyzer", page_icon="🧫", layout="wide")
st.title("LuxPlate Analyzer")
st.caption("Du classeur Varioskan brut aux données contrôlées, analysées et visualisées.")

guided_tab, import_tab, qc_tab, blanks_tab, normalization_tab, kinetics_tab, figures_tab = st.tabs(
    ["▶ Analyse guidée", "1 · Import et mise en forme", "2 · Contrôle qualité", "3 · Correction des blancs",
     "4 · Normalisation par la DO", "5 · Paramètres cinétiques", "6 · Figures et export"]
)

with guided_tab:
    st.header("Analyse guidée d'un classeur réel")
    st.write(
        "Déposez un ou plusieurs fichiers, vérifiez les souches détectées, retirez si nécessaire "
        "des points ou des courbes, puis lancez tout le calcul en une fois."
    )
    guided_uploads = st.file_uploader(
        "Déposer un ou plusieurs classeurs Varioskan (.xlsx ou .xlsm)", type=["xlsx", "xlsm"],
        accept_multiple_files=True, key="guided_upload"
    )
    if not guided_uploads:
        st.info("Les fichiers restent traités localement par l'application.")
    else:
        try:
            guided_inputs = []
            guided_identity = []
            for file_index, upload in enumerate(guided_uploads):
                payload = upload.getvalue()
                _, lum_sheets = inspect_workbook(BytesIO(payload))
                lum = st.selectbox(f"Luminescence — {upload.name}", lum_sheets,
                                   key=f"guided_lum_{file_index}_{upload.name}")
                guided_inputs.append((upload.name, parse_kinetic_workbook(BytesIO(payload), lum)))
                guided_identity.append((upload.name, hash(payload), lum))
            guided_data = combine_kinetic_tables(guided_inputs)
        except Exception as error:
            st.error(f"Import impossible : {error}")
        else:
            strain_options = sorted(guided_data.loc[guided_data["type"].eq("souche"), "souche"].unique())
            medium_options = sorted(guided_data.loc[guided_data["type"].eq("souche"), "Groupe"].unique())
            guided_media = st.multiselect(
                "Milieux à analyser", medium_options, default=medium_options
            )
            guided_strains = st.multiselect(
                "Souches à analyser", strain_options, default=strain_options
            )
            try:
                guided_selected = filter_experiment_data(guided_data, guided_strains, guided_media)
            except ValueError as error:
                st.warning(str(error))
            else:
                guided_signature = (
                    tuple(guided_identity), tuple(guided_media), tuple(guided_strains),
                )
                if st.session_state.get("guided_signature") != guided_signature:
                    st.session_state.pop("guided_complete_result", None)
                    st.session_state.pop("guided_decisions", None)
                    st.session_state["guided_signature"] = guided_signature
                blanks = guided_selected.loc[guided_selected["type"].eq("blanc")]
                if blanks.empty:
                    st.error("Aucun blanc n'a été détecté pour les milieux sélectionnés.")
                else:
                    st.success(
                        f"{len(guided_strains)} souche(s), {len(guided_uploads)} fichier(s), "
                        f"{guided_selected['sample_header'].nunique()} courbe(s) et "
                        f"{blanks['sample_header'].nunique()} blanc(s) détectés."
                    )
                info_tabs = st.tabs(["Courbes des blancs", "Courbes des échantillons"])
                with info_tabs[0]:
                    st.caption("Une figure statique par réplicat biologique, avec la DO et la luminescence.")
                    for _, figure in build_guided_raw_figures(guided_selected, sample_type="blanc"):
                        st.pyplot(figure, use_container_width=True)
                with info_tabs[1]:
                    st.caption("Une figure statique par souche, milieu et réplicat biologique.")
                    for _, figure in build_guided_raw_figures(guided_selected, sample_type="souche"):
                        st.pyplot(figure, use_container_width=True)

                st.subheader("Points et courbes à supprimer")
                st.caption(
                    "Cochez les lignes à supprimer. Une suppression de point retire ensemble la DO et la "
                    "luminescence mesurées à ce temps ; le fichier source n'est jamais modifié."
                )
                editable_points = guided_selected.reset_index(names="source_index").copy()
                editable_points.insert(0, "Supprimer", False)
                edited_points = st.data_editor(
                    editable_points[["Supprimer", "source_index", "type", "souche", "Groupe", "sample_header",
                                     "puits", "temps_h", "DO_brute", "Lum_brute"]],
                    disabled=["source_index", "type", "souche", "Groupe", "sample_header", "puits",
                              "temps_h", "DO_brute", "Lum_brute"],
                    hide_index=True, use_container_width=True, key="guided_point_editor",
                )
                strain_headers = guided_selected.loc[
                    guided_selected["type"].eq("souche"), "sample_header"
                ].drop_duplicates().tolist()
                removed_series = st.multiselect("Courbes entières à supprimer", strain_headers)

                with st.expander("Paramètres de calcul", expanded=False):
                    p1, p2, p3, p4 = st.columns(4)
                    guided_min_od = p1.number_input("DO minimale", min_value=0.0, value=0.05, step=0.01)
                    guided_consecutive = p2.number_input("Points consécutifs", min_value=1, value=3, step=1)
                    guided_window = p3.number_input("Points par fenêtre", min_value=2, value=3, step=1)
                    guided_r2 = p4.number_input("R² minimal", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

                if st.button("Lancer toute l'analyse", type="primary", disabled=blanks.empty):
                    removed_points = edited_points.loc[edited_points["Supprimer"], "source_index"].astype(int).tolist()
                    manual_decisions = build_manual_decisions(guided_selected, removed_points, removed_series)
                    try:
                        complete = run_complete_analysis(
                            guided_selected, manual_decisions, minimum_od=float(guided_min_od),
                            consecutive_points=int(guided_consecutive), growth_window_points=int(guided_window),
                            growth_rate_min_r_squared=float(guided_r2),
                        )
                    except ValueError as error:
                        st.error(f"Analyse impossible : {error}")
                    else:
                        st.session_state["guided_complete_result"] = complete
                        st.session_state["guided_decisions"] = manual_decisions

                if "guided_complete_result" in st.session_state:
                    complete = st.session_state["guided_complete_result"]
                    st.subheader("Résultats complets")
                    cards = st.columns(4)
                    cards[0].metric("Points exclus", len(complete.blank_correction.excluded_data))
                    cards[1].metric("Points normalisés", int(complete.normalization.normalized_data["normalization_ok"].sum()))
                    cards[2].metric("Séries analysées", len(complete.kinetics.series_metrics))
                    cards[3].metric("Avertissements", len(complete.kinetics.warnings))
                    result_tabs = st.tabs(["Blancs corrigés", "Normalisation", "Cinétique", "Courbes finales"])
                    with result_tabs[0]:
                        st.dataframe(complete.blank_correction.blank_profiles, use_container_width=True, hide_index=True)
                    with result_tabs[1]:
                        st.dataframe(complete.normalization.normalized_data, use_container_width=True, hide_index=True)
                    with result_tabs[2]:
                        st.dataframe(complete.kinetics.series_metrics, use_container_width=True, hide_index=True)
                    with result_tabs[3]:
                        st.caption(
                            "Choisissez les figures à afficher : elles sont régénérées dès qu'une option change, "
                            "sans relancer l'analyse."
                        )
                        guided_family_labels = {
                            "Croissance corrigée": "growth",
                            "Luminescence non normalisée": "corrected",
                            "Luminescence normalisée": "normalized",
                            "Double axe DO + luminescence non normalisée": "mixed",
                            "Pic normalisé": "peak",
                            "Temps du pic normalisé": "peak_time",
                            "AUC normalisée": "auc",
                            "Temps de doublement": "doubling",
                        }
                        guided_figure_labels = st.multiselect(
                            "Figures finales", list(guided_family_labels),
                            default=list(guided_family_labels), key="guided_figure_families",
                        )
                        guided_options = st.columns(3)
                        guided_panel_label = guided_options[0].selectbox(
                            "Organisation des panneaux", ["Panneaux par milieu", "Panneaux par souche"],
                            key="guided_figure_panels",
                        )
                        guided_lum_label = guided_options[1].selectbox(
                            "Luminescence", ["Linéaire", "Logarithmique"], key="guided_figure_lum_scale",
                        )
                        guided_norm_label = guided_options[2].selectbox(
                            "Lum/DO", ["Linéaire", "Logarithmique"], key="guided_figure_norm_scale",
                        )
                        if not guided_figure_labels:
                            st.info("Sélectionnez au moins une figure finale.")
                        else:
                            guided_figures = build_publication_figures(
                                complete.normalization.normalized_data,
                                title="Analyse complète",
                                families=tuple(guided_family_labels[label] for label in guided_figure_labels),
                                panel_by="Groupe" if guided_panel_label.endswith("milieu") else "souche",
                                lum_scale="log" if guided_lum_label == "Logarithmique" else "linear",
                                normalized_scale="log" if guided_norm_label == "Logarithmique" else "linear",
                            )
                            for guided_name, guided_figure in guided_figures:
                                st.subheader(guided_name.replace("_", " ").title())
                                st.pyplot(guided_figure, use_container_width=True)
                            st.caption(
                                "Les temps séparés de moins d'une minute sont alignés avant le tracé. Les puits "
                                "techniques sont moyennés dans chaque réplicat biologique, puis les courbes montrent "
                                "la moyenne biologique ± SD. Normalized "
                                "luminescence is blank-corrected RLU divided by blank-corrected OD600. "
                                "Summary boxplots show independent biological experiments and their exact mean; "
                                "small points are technical replicates and large points are biological means. "
                                "Statistics use biological means only (Friedman test with paired Wilcoxon "
                                "tests and Holm correction)."
                            )
                    base = guided_uploads[0].name.rsplit(".", 1)[0] if len(guided_uploads) == 1 else "analyse_multi_fichiers"
                    exports = st.columns(3)
                    exports[0].download_button("Données finales", complete.normalization.normalized_data.to_csv(
                        index=False).encode("utf-8-sig"), f"{base}_donnees_finales.csv", "text/csv")
                    exports[1].download_button("Métriques cinétiques", complete.kinetics.series_metrics.to_csv(
                        index=False).encode("utf-8-sig"), f"{base}_metriques_cinetiques.csv", "text/csv")
                    exports[2].download_button("Décisions d'exclusion", st.session_state["guided_decisions"].to_csv(
                        index=False).encode("utf-8-sig"), f"{base}_decisions_exclusion.csv", "text/csv")

with import_tab:
    st.header("1 · Import et mise en forme")
    st.write(
        "Cette première étape intègre `01_mise_en_forme_donnees.py` : elle associe "
        "les mesures DO/luminescence, conserve chaque puits technique et construit le tableau long."
    )
    uploads = st.file_uploader("Classeurs Varioskan cinétiques (.xlsx)", type=["xlsx", "xlsm"],
                               accept_multiple_files=True)

    if not uploads:
        st.info("Chargez un ou plusieurs classeurs pour choisir la lecture de luminescence et inspecter les données brutes.")
    else:
        try:
            parsed, source_identity = [], []
            for file_index, upload in enumerate(uploads):
                payload = upload.getvalue()
                absorbance_sheet, luminescence_sheets = inspect_workbook(BytesIO(payload))
                selected_luminescence = st.selectbox(
                    f"Luminescence — {upload.name} (absorbance : {absorbance_sheet})", luminescence_sheets,
                    key=f"import_lum_{file_index}_{upload.name}")
                parsed.append((upload.name, parse_kinetic_workbook(BytesIO(payload), selected_luminescence)))
                source_identity.append((upload.name, hash(payload), selected_luminescence))
            data = combine_kinetic_tables(parsed)
            previous_source = st.session_state.get("source_identity")
            if previous_source != source_identity:
                for key in ("qc_journal", "validated_qc_journal", "qc_validated",
                            "blank_correction_result", "normalization_result"):
                    st.session_state.pop(key, None)
            st.session_state["source_identity"] = source_identity
            st.session_state["long_data"] = data
            st.session_state["source_name"] = uploads[0].name.rsplit(".", 1)[0] if len(uploads) == 1 else "analyse_multi_fichiers"
        except Exception as error:
            st.error(f"Import impossible : {error}")
            st.stop()

        strains = int(data.loc[data["type"].eq("souche"), "souche"].nunique())
        max_gap = float(data["ecart_temps_s"].max()) if not data.empty else 0.0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes", len(data))
        col2.metric("Souches", strains)
        col3.metric("Puits", data["sample_header"].nunique())
        col4.metric("Écart DO/Lum max", f"{max_gap:.1f} s")
        if max_gap > 5:
            st.warning("L'écart entre les temps DO et luminescence dépasse 5 secondes pour au moins une lecture.")

        preview_tab, curves_tab = st.tabs(["Aperçu du tableau", "Courbes brutes"])
        with preview_tab:
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.download_button(
                "Télécharger le tableau long (CSV)", data.to_csv(index=False).encode("utf-8"),
                file_name=f"{st.session_state['source_name']}_format_long.csv", mime="text/csv",
            )
        with curves_tab:
            st.pyplot(plot_raw_curves(data), use_container_width=True)

with qc_tab:
    st.header("2 · Contrôle qualité interactif")
    st.caption(
        "Les anomalies sont uniquement proposées. Aucune observation n'est exclue sans une décision explicite."
    )
    if "long_data" not in st.session_state:
        st.info("Importez d'abord un classeur cinétique dans l'onglet 1.")
    else:
        threshold = st.number_input("Seuil du z-score robuste", min_value=0.1, value=3.5, step=0.1)
        try:
            qc = run_quality_control(st.session_state["long_data"], threshold)
        except ValueError as error:
            st.error(f"Contrôle qualité impossible : {error}")
            st.stop()

        indicators = dict(zip(qc.global_summary["indicateur"], qc.global_summary["valeur"]))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes contrôlées", int(indicators["n_lignes"]))
        col2.metric("Séries techniques", int(indicators["n_series_total"]))
        col3.metric("Séries signalées", int(indicators["n_series_flaggees"]))
        col4.metric("Points proposés", int(indicators["n_points_aberrants"]))

        summary_tab, series_tab, decisions_tab, plot_tab = st.tabs(
            ["Résumé global", "Séries techniques", "Anomalies et décisions", "Courbes annotées"]
        )
        with summary_tab:
            st.dataframe(qc.global_summary, use_container_width=True, hide_index=True)
        with series_tab:
            st.dataframe(qc.series_summary, use_container_width=True, hide_index=True)
        with decisions_tab:
            if qc.decisions.empty:
                st.success("Aucune anomalie ni série à revoir n'a été proposée avec ce seuil.")
                journal = qc.decisions.copy()
            else:
                editable = qc.decisions.copy()
                editable["Décision"] = editable["decision_utilisateur"].map(
                    {"review": "À revoir", "keep": "Conserver", "remove": "Exclure"}
                )
                editable["Justification"] = editable["raison_utilisateur"]
                edited = st.data_editor(
                    editable,
                    column_config={
                        "Décision": st.column_config.SelectboxColumn(
                            "Décision", options=["À revoir", "Conserver", "Exclure"], required=True
                        ),
                        "Justification": st.column_config.TextColumn("Justification (facultative)"),
                    },
                    disabled=[column for column in editable.columns if column not in {"Décision", "Justification"}],
                    hide_index=True, use_container_width=True, key="qc_decision_editor",
                )
                journal = edited[list(qc.decisions.columns)].copy()
                journal["decision_utilisateur"] = edited["Décision"].map(
                    {"À revoir": "review", "Conserver": "keep", "Exclure": "exclure"}
                )
                journal["raison_utilisateur"] = edited["Justification"].fillna("")
            validated_journal = st.session_state.get("validated_qc_journal")
            if validated_journal is not None and not journal.equals(validated_journal):
                st.session_state["qc_validated"] = False
            st.session_state["qc_journal"] = journal.copy(deep=True)
            counts = journal["decision_utilisateur"].value_counts() if not journal.empty else {}
            count_review = int(counts.get("review", 0))
            count_keep = int(counts.get("keep", 0))
            count_exclude = int(counts.get("exclure", 0))
            review_col, keep_col, exclude_col = st.columns(3)
            review_col.metric("À revoir", count_review)
            keep_col.metric("Conservées", count_keep)
            exclude_col.metric("Exclues", count_exclude)
            confirmed = True
            if count_review:
                st.warning("Certaines décisions sont encore « À revoir ». Elles ne provoqueront aucune exclusion.")
                confirmed = st.checkbox(
                    "Je confirme vouloir valider malgré les décisions encore à revoir",
                    key="confirm_pending_qc",
                )
            if st.button("Valider les décisions QC", type="primary", disabled=not confirmed):
                st.session_state["validated_qc_journal"] = journal.copy(deep=True)
                st.session_state["qc_validated"] = True
                st.success("Journal QC validé et figé. La correction des blancs est déverrouillée.")
            st.download_button(
                "Télécharger le journal des décisions (CSV)",
                journal.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{st.session_state.get('source_name', 'experience')}_outliers_decisions.csv",
                mime="text/csv",
            )
        with plot_tab:
            st.pyplot(plot_qc_curves(qc.data, qc.anomalies), use_container_width=True)

with blanks_tab:
    st.header("3 · Correction des blancs")
    if not st.session_state.get("qc_validated") or "validated_qc_journal" not in st.session_state:
        st.info("Validez d'abord les décisions QC dans l'onglet 2 pour déverrouiller cette étape.")
    else:
        try:
            correction = run_blank_correction(
                st.session_state["long_data"], st.session_state["validated_qc_journal"]
            )
        except ValueError as error:
            st.error(f"Correction des blancs impossible : {error}")
        else:
            st.session_state["blank_correction_result"] = correction
            metrics = dict(zip(correction.summary["metrique"], correction.summary["valeur"]))
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes avant QC", int(metrics["lignes_avant_qc"]))
            col2.metric("Lignes après QC", int(metrics["lignes_apres_qc"]))
            col3.metric("Observations exclues", int(metrics["lignes_exclues"]))
            col4.metric("Groupes sans blanc", int(metrics["groupes_sans_blanc"]))
            if not correction.warnings.empty:
                groups = ", ".join(sorted(correction.warnings["Groupe"].astype(str).unique()))
                st.warning(f"Groupes sans blanc disponible à au moins un temps : {groups}. Les valeurs corrigées restent vides.")

            profiles_tab, preview_tab, curves_tab = st.tabs(
                ["Profils moyens des blancs", "Données corrigées", "Courbes avant / après"]
            )
            with profiles_tab:
                st.dataframe(correction.blank_profiles, use_container_width=True, hide_index=True)
            with preview_tab:
                preview_columns = [
                    column for column in ("temps_h", "souche", "Groupe", "replicat", "DO_brute", "DO_corr", "Lum_brute", "Lum_corr")
                    if column in correction.corrected_data
                ]
                st.dataframe(correction.corrected_data[preview_columns], use_container_width=True, hide_index=True)
            with curves_tab:
                st.pyplot(plot_blank_correction(correction.corrected_data), use_container_width=True)

            base = st.session_state.get("source_name", "experience")
            downloads = st.columns(4)
            downloads[0].download_button(
                "Données corrigées", correction.corrected_data.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{base}_corrige_blancs.csv", mime="text/csv",
            )
            downloads[1].download_button(
                "Observations exclues", correction.excluded_data.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{base}_exclusions.csv", mime="text/csv",
            )
            downloads[2].download_button(
                "Résumé des blancs", correction.blank_profiles.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{base}_profils_blancs.csv", mime="text/csv",
            )
            downloads[3].download_button(
                "Journal QC validé", st.session_state["validated_qc_journal"].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{base}_journal_qc_valide.csv", mime="text/csv",
            )

with normalization_tab:
    st.header("4 · Normalisation de la luminescence par la DO")
    if "blank_correction_result" not in st.session_state:
        st.info("Exécutez d'abord la correction des blancs dans l'onglet 3.")
    else:
        correction = st.session_state["blank_correction_result"]
        parameter_columns = st.columns(3)
        k_sd = parameter_columns[0].number_input("Multiplicateur du blanc (k)", min_value=0.0, value=3.0, step=0.1)
        minimum_od = parameter_columns[1].number_input("DO minimale", min_value=0.0, value=0.05, step=0.01, format="%.3f")
        consecutive = parameter_columns[2].number_input("Points consécutifs", min_value=1, value=3, step=1)
        try:
            normalization = run_normalization(correction.corrected_data, k_sd, minimum_od, int(consecutive))
        except ValueError as error:
            st.error(f"Normalisation impossible : {error}")
        else:
            st.session_state["normalization_result"] = normalization
            if not normalization.warnings.empty:
                for message in normalization.warnings["message"]:
                    st.warning(message)
            metrics = dict(zip(normalization.summary["metric"], normalization.summary["value"]))
            metric_columns = st.columns(4)
            metric_columns[0].metric("Séries validées", int(metrics["valid_series"]))
            metric_columns[1].metric("Séries non validées", int(metrics["invalid_series"]))
            metric_columns[2].metric("Lignes normalisées", int(metrics["normalized_rows"]))
            metric_columns[3].metric("Lignes non normalisées", int(metrics["rejected_rows"]))
            threshold_tab, series_tab, preview_tab, curves_tab, rejected_tab = st.tabs([
                "Seuils", "Validation des séries", "Données normalisées", "Courbes", "Lignes non normalisées"
            ])
            with threshold_tab:
                st.dataframe(normalization.threshold_details, use_container_width=True, hide_index=True)
            with series_tab:
                st.dataframe(normalization.series_validation, use_container_width=True, hide_index=True)
            with preview_tab:
                st.dataframe(normalization.normalized_data, use_container_width=True, hide_index=True)
            with curves_tab:
                st.pyplot(plot_normalization(normalization.normalized_data), use_container_width=True)
            with rejected_tab:
                st.dataframe(normalization.rejected_rows, use_container_width=True, hide_index=True)
            base = st.session_state.get("source_name", "experience")
            exports = st.columns(4)
            for column, label, table, suffix in [
                (exports[0], "Données normalisées", normalization.normalized_data, "normalise_DO"),
                (exports[1], "Validation des séries", normalization.series_validation, "validation_series"),
                (exports[2], "Détails des seuils", normalization.threshold_details, "seuils_DO"),
                (exports[3], "Avertissements", normalization.warnings, "avertissements_normalisation"),
            ]:
                column.download_button(label, table.to_csv(index=False).encode("utf-8-sig"),
                                       file_name=f"{base}_{suffix}.csv", mime="text/csv")

with kinetics_tab:
    st.header("5 · Paramètres cinétiques")
    if "normalization_result" not in st.session_state:
        st.info("Exécutez d'abord la normalisation dans l'onglet 4.")
    else:
        parameters = st.columns(4)
        window = parameters[0].number_input("Points par fenêtre", min_value=2, value=3, step=1)
        duration = parameters[1].number_input("Durée minimale (h)", min_value=0.0, value=0.0, step=0.1)
        r_squared = parameters[2].number_input("R² minimal", min_value=0.0, max_value=1.0,
                                               value=0.0, step=0.05)
        auc_points = parameters[3].number_input("Points minimum pour l'AUC", min_value=2, value=2, step=1)
        try:
            kinetics = run_kinetics(
                st.session_state["normalization_result"].normalized_data,
                growth_window_points=int(window), minimum_auc_points=int(auc_points),
                growth_window_min_duration_h=float(duration), growth_rate_min_r_squared=float(r_squared),
            )
        except ValueError as error:
            st.error(f"Extraction cinétique impossible : {error}")
        else:
            counters = dict(zip(kinetics.summary["metric"], kinetics.summary["value"]))
            cards = st.columns(3)
            cards[0].metric("Séries analysées", int(counters["series_analyzed"]))
            cards[1].metric("Séries rejetées", int(counters["series_rejected"]))
            cards[2].metric("Avertissements", int(counters["warnings"]))
            metrics_tab, technical_tab, rejected_tab, warnings_tab, summary_tab, curves_tab = st.tabs([
                "Métriques par série", "Résumé technique", "Séries rejetées", "Avertissements",
                "Résumé d'exécution", "Courbes annotées",
            ])
            tables = [kinetics.series_metrics, kinetics.strain_summary, kinetics.rejected_series,
                      kinetics.warnings, kinetics.summary]
            for tab, table in zip((metrics_tab, technical_tab, rejected_tab, warnings_tab, summary_tab), tables):
                with tab:
                    st.dataframe(table, use_container_width=True, hide_index=True)
            with curves_tab:
                st.pyplot(plot_kinetics(st.session_state["normalization_result"].normalized_data,
                                        kinetics.series_metrics), use_container_width=True)
            base = st.session_state.get("source_name", "experience")
            labels = ("metriques_series", "resume_technique", "series_rejetees", "avertissements", "resume")
            export_columns = st.columns(5)
            for column, table, suffix in zip(export_columns, tables, labels):
                column.download_button(suffix.replace("_", " ").title(),
                    table.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{base}_cinetique_{suffix}.csv", mime="text/csv")

with figures_tab:
    st.header("6 · Figures finales et export publication")
    st.write(
        "Composez librement la galerie inspirée des scripts d'exemple. Les échelles, l'organisation des "
        "panneaux et la représentation de l'incertitude restent modifiables avant chaque export."
    )
    if "normalization_result" not in st.session_state:
        st.info("Exécutez d'abord la normalisation dans l'onglet 4.")
    else:
        family_labels = {
            "Croissance corrigée": "growth", "Luminescence corrigée": "corrected",
            "Luminescence normalisée": "normalized", "Double axe DO + luminescence non normalisée": "mixed",
            "Pic normalisé": "peak", "Temps du pic normalisé": "peak_time",
            "AUC normalisée": "auc", "Temps de doublement": "doubling",
            "Comparaisons ciblées au contrôle": "control",
        }
        selected_labels = st.multiselect(
            "Figures à produire", list(family_labels), default=list(family_labels)
        )
        option_columns = st.columns(4)
        panel_label = option_columns[0].selectbox("Organisation des courbes", ["Panneaux par milieu", "Panneaux par souche"])
        lum_label = option_columns[1].selectbox("Échelle de luminescence", ["Linéaire", "Logarithmique"])
        norm_label = option_columns[2].selectbox("Échelle Lum/DO", ["Linéaire", "Logarithmique"])
        metric_label = option_columns[3].selectbox("Échelle AUC et pic", ["Logarithmique", "Linéaire"])
        uncertainty_label = st.radio(
            "Incertitude de la figure mixte", ["Barres ± SD", "Ruban ± SD"], horizontal=True
        )
        available_strains = sorted(
            st.session_state["normalization_result"].normalized_data["souche"].dropna().astype(str).unique()
        )
        default_control = available_strains.index("P0-lux") if "P0-lux" in available_strains else 0
        control_strain = st.selectbox("Souche contrôle des comparaisons ciblées", available_strains,
                                      index=default_control)
        if st.button("Préparer la galerie", type="primary"):
            if not selected_labels:
                st.warning("Sélectionnez au moins une famille de figures.")
            else:
                with st.spinner("Création des figures haute résolution…"):
                    figures = build_publication_figures(
                        st.session_state["normalization_result"].normalized_data,
                        families=tuple(family_labels[label] for label in selected_labels),
                        panel_by="Groupe" if panel_label.endswith("milieu") else "souche",
                        lum_scale="log" if lum_label == "Logarithmique" else "linear",
                        normalized_scale="log" if norm_label == "Logarithmique" else "linear",
                        metric_scale="log" if metric_label == "Logarithmique" else "linear",
                        uncertainty="bars" if uncertainty_label.startswith("Barres") else "ribbon",
                        control=control_strain,
                    )
                    rendered, archive = package_figures(figures, dpi=600)
                    st.session_state["publication_export"] = (figures, rendered, archive)
        if "publication_export" in st.session_state:
            figures, rendered, archive = st.session_state["publication_export"]
            st.success(f"{len(rendered)} figure(s) préparée(s), dans deux formats chacune.")
            st.download_button("Télécharger toutes les figures (.zip)", archive,
                               file_name=f"{st.session_state.get('source_name', 'analyse')}_figures.zip",
                               mime="application/zip", type="primary")
            for (name, figure), item in zip(figures, rendered):
                st.subheader(name.replace("_", " ").title())
                st.pyplot(figure, use_container_width=True)
                st.caption(
                    "Les temps observés à moins d'une minute sont alignés; les puits techniques sont d'abord "
                    "moyennés par réplicat biologique, puis les courbes montrent moyenne biologique ± SD. "
                    "Normalized luminescence = blank-corrected RLU / blank-corrected OD600. Summary plots are "
                    "boxplots with their exact mean; small points are technical replicates and large points are "
                    "biological means. Statistical "
                    "tests use biological means only (Friedman; paired Wilcoxon post-hoc with Holm correction)."
                )
                png_column, pdf_column = st.columns(2)
                png_column.download_button(
                    "PNG · 600 dpi", item.png, f"{item.name}.png", "image/png", key=f"png_{item.name}"
                )
                pdf_column.download_button(
                    "PDF · vectoriel", item.pdf, f"{item.name}.pdf", "application/pdf", key=f"pdf_{item.name}"
                )

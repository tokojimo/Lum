"""Streamlit entry point for the progressive legacy-pipeline integration."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from luxplate.blanks import run_blank_correction
from luxplate.plotting import plot_blank_correction, plot_qc_curves, plot_raw_curves
from luxplate.qc import run_quality_control
from luxplate.varioskan import inspect_workbook, parse_kinetic_workbook

st.set_page_config(page_title="LuxPlate Analyzer", page_icon="🧫", layout="wide")
st.title("LuxPlate Analyzer")
st.caption("Du classeur Varioskan brut aux données contrôlées, analysées et visualisées.")

import_tab, qc_tab, blanks_tab = st.tabs(
    ["1 · Import et mise en forme", "2 · Contrôle qualité", "3 · Correction des blancs"]
)

with import_tab:
    st.header("1 · Import et mise en forme")
    st.write(
        "Cette première étape intègre `01_mise_en_forme_donnees.py` : elle associe "
        "les mesures DO/luminescence, conserve chaque puits technique et construit le tableau long."
    )
    uploaded = st.file_uploader("Classeur Varioskan cinétique (.xlsx)", type=["xlsx", "xlsm"])

    if uploaded is None:
        st.info("Chargez un classeur pour choisir la lecture de luminescence et inspecter les données brutes.")
    else:
        payload = uploaded.getvalue()
        try:
            absorbance_sheet, luminescence_sheets = inspect_workbook(BytesIO(payload))
            left, right = st.columns(2)
            left.text_input("Feuille d'absorbance détectée", absorbance_sheet, disabled=True)
            selected_luminescence = right.selectbox("Feuille de luminescence", luminescence_sheets)
            data = parse_kinetic_workbook(BytesIO(payload), selected_luminescence)
            previous_source = st.session_state.get("source_payload")
            if previous_source != payload:
                for key in ("qc_journal", "validated_qc_journal", "qc_validated"):
                    st.session_state.pop(key, None)
            st.session_state["source_payload"] = payload
            st.session_state["long_data"] = data
            st.session_state["source_name"] = uploaded.name.rsplit(".", 1)[0]
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

st.divider()
st.subheader("Étapes suivantes")
st.write("Normalisation par la DO → paramètres cinétiques → statistiques et figures.")

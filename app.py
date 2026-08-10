"""Streamlit entry point for the progressive legacy-pipeline integration."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from luxplate.plotting import plot_raw_curves
from luxplate.varioskan import inspect_workbook, parse_kinetic_workbook

st.set_page_config(page_title="LuxPlate Analyzer", page_icon="🧫", layout="wide")
st.title("LuxPlate Analyzer")
st.caption("Du classeur Varioskan brut aux données contrôlées, analysées et visualisées.")

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

    tabs = st.tabs(["Aperçu du tableau", "Courbes brutes"])
    with tabs[0]:
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.download_button(
            "Télécharger le tableau long (CSV)",
            data.to_csv(index=False).encode("utf-8"),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_format_long.csv",
            mime="text/csv",
        )
    with tabs[1]:
        st.pyplot(plot_raw_curves(data), use_container_width=True)

st.divider()
st.subheader("Étapes suivantes")
st.write("Contrôle qualité → correction des blancs → normalisation par la DO → paramètres cinétiques → statistiques et figures.")

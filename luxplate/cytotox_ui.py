"""Streamlit interface for inspecting Cytotox measurements by strain."""

import hashlib
from textwrap import fill

from matplotlib import colormaps
from matplotlib.figure import Figure
import streamlit as st

from .cytotox import CytotoxSignal, read_cytotox_workbook


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_workbook(payload: bytes):
    return read_cytotox_workbook(payload)


def _signal_figure(
    signal: CytotoxSignal, wells_by_strain: dict[str, list[str]], selected: list[str]
) -> tuple[Figure, list[str]]:
    palette = colormaps["tab20"].colors
    styles = ("-", "--", "-.", ":")
    figure = Figure(figsize=(10, 4), layout="constrained")
    axis = figure.subplots()
    missing = []
    handles = []
    labels = []

    for index, strain in enumerate(selected):
        drawn = False
        for well in wells_by_strain[strain]:
            if well not in signal.data:
                continue
            readings = signal.data[well]
            measured_count = readings.count()
            if not measured_count:
                continue
            (curve,) = axis.plot(
                readings.index, readings,
                color=palette[index % len(palette)], linestyle=styles[index // len(palette) % len(styles)],
                marker="." if measured_count == 1 else None,
                alpha=0.8,
            )
            if not drawn:
                handles.append(curve)
                labels.append(fill(strain, width=30))
                drawn = True
        if not drawn:
            missing.append(strain)

    axis.set(xlabel="Temps (h)", ylabel=signal.metric)
    axis.grid(axis="y", alpha=0.25)
    if handles:
        columns = min(3, len(handles))
        legend_rows = (len(handles) + columns - 1) // columns
        figure.set_figheight(4 + 0.42 * legend_rows)
        figure.legend(handles, labels, ncols=columns, loc="outside lower center")
    return figure, missing


def render_cytotox() -> None:
    st.header("Analyse Cytotox")
    upload = st.file_uploader(
        "Déposer un classeur Cytotox (.xlsx ou .xlsm)",
        type=["xlsx", "xlsm"], key="cytotox_workbook",
        help="Ordre des feuilles : GFP, IP, courbes finales GFP, plan de plaque 96 puits.",
    )
    if upload is None:
        return

    payload = upload.getvalue()
    try:
        workbook = _cached_workbook(payload)
    except ValueError as error:
        st.error(f"Import Cytotox impossible : {error}")
        return

    wells_by_strain: dict[str, list[str]] = {}
    for row, labels in workbook.plate.iterrows():
        for column, strain in labels.items():
            if strain:
                wells_by_strain.setdefault(strain, []).append(f"{row}{column}")

    with st.expander("Plan de plaque · 96 puits"):
        st.dataframe(workbook.plate.rename(columns=str), use_container_width=True)

    selected = st.multiselect(
        "Souches et conditions à afficher", sorted(wells_by_strain, key=str.casefold),
        key=f"cytotox_strains_{hashlib.sha256(payload).hexdigest()}",
    )
    if not selected:
        return

    st.caption("Chaque trait représente un puits. Les puits d’une même souche ont la même couleur.")
    for signal in workbook.signals:
        st.subheader(signal.name)
        st.caption(f"Feuille Excel « {signal.sheet_name.strip()} »")
        figure, missing = _signal_figure(signal, wells_by_strain, selected)
        if len(missing) < len(selected):
            st.pyplot(figure, use_container_width=True)
        if missing:
            st.info(f"Aucune mesure dans cette feuille pour {', '.join(missing)}.")

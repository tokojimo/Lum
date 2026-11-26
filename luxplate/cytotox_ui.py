"""Streamlit interface for inspecting Cytotox measurements by strain."""

from pathlib import Path
from textwrap import fill

from matplotlib import colormaps
from matplotlib.figure import Figure
import pandas as pd
import streamlit as st

from .cytotox import CytotoxSignal, CytotoxWorkbook, read_cytotox_workbook
from .export import figure_bytes


PLATE_STYLE = """
<style>
.cytotox-plate-scroll { overflow-x: auto; }
.cytotox-plate-scroll .cytotox-plate {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.cytotox-plate-scroll .cytotox-plate th,
.cytotox-plate-scroll .cytotox-plate td {
    border: 1px solid rgba(128, 128, 128, 0.5);
    padding: 0.25rem 0.35rem;
    text-align: center;
}
.cytotox-plate-scroll .cytotox-plate th {
    background: rgba(128, 128, 128, 0.12);
}
</style>
"""


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_workbook(payload: bytes) -> CytotoxWorkbook:
    return read_cytotox_workbook(payload)


def _wells_by_strain(plate: pd.DataFrame) -> dict[str, list[str]]:
    """Keep the plate order, reading each column from A to H."""
    grouped: dict[str, list[str]] = {}
    for column, labels in plate.items():
        for row, strain in labels.items():
            if strain:
                grouped.setdefault(strain, []).append(f"{row}{column}")
    return grouped


def _raw_figure(
    signal: CytotoxSignal, wells_by_strain: dict[str, list[str]], title: str,
) -> Figure:
    """Plot one panel per selected strain, with the original values from each well."""
    columns = min(4, len(wells_by_strain))
    rows = (len(wells_by_strain) + columns - 1) // columns
    palette = colormaps["tab10"].colors
    styles = ("-", "--", "-.", ":")
    figure = Figure(figsize=(4 * columns, 2.2 * rows + 1.2), layout="constrained")
    axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    figure.suptitle(title, fontsize=14, parse_math=False)
    time_min = signal.data.index * 60
    replicate_lines = {}
    lowest_value = float("inf")

    for index, (strain, wells) in enumerate(wells_by_strain.items()):
        row, column = divmod(index, columns)
        axis = axes[row, column]
        for replicate, well in enumerate(wells, start=1):
            if well not in signal.data:
                continue
            values = signal.data[well]
            if values.isna().all():
                continue
            lowest_value = min(lowest_value, values.min())
            (line,) = axis.plot(
                time_min, values, color=palette[(replicate - 1) % len(palette)],
                linestyle=styles[((replicate - 1) // len(palette)) % len(styles)],
                marker="o", markersize=2.4, linewidth=1.15, label=f"Rep {replicate}",
            )
            replicate_lines.setdefault(replicate, line)

        if not axis.lines:
            axis.text(0.5, 0.5, "Aucune mesure", transform=axis.transAxes, ha="center", va="center")
        axis.set_title(fill(strain, width=39), fontsize=10, parse_math=False)
        axis.tick_params(labelsize=8, labelbottom=True, labelleft=True)
        axis.grid(True, alpha=0.22)
        axis.set_axisbelow(True)
        if column == 0:
            axis.set_ylabel(fill(signal.metric, width=24), fontsize=9, parse_math=False)
        if index + columns >= len(wells_by_strain):
            axis.set_xlabel("Temps (min)", fontsize=9)

    for axis in axes.flat[len(wells_by_strain):]:
        axis.set_visible(False)
    if replicate_lines:
        if lowest_value >= 0:
            axes[0, 0].set_ylim(bottom=0)
        handles = [replicate_lines[key] for key in sorted(replicate_lines)]
        figure.legend(
            handles=handles, loc="outside lower center", title="Répétition technique",
            ncols=min(6, len(handles)), fontsize=9, frameon=False,
        )
    return figure


@st.cache_data(show_spinner="Préparation des courbes brutes…", max_entries=24)
def _raw_figure_png(
    signal: CytotoxSignal, wells_by_strain: dict[str, list[str]], title: str,
) -> bytes:
    return figure_bytes(_raw_figure(signal, wells_by_strain, title), "png", dpi=125)


def render_cytotox() -> None:
    st.header("Analyse Cytotox")
    uploads = st.file_uploader(
        "Déposer un ou plusieurs classeurs Cytotox (.xlsx ou .xlsm)",
        type=["xlsx", "xlsm"], accept_multiple_files=True, key="cytotox_workbooks",
        help="Ordre des feuilles : GFP, IP, courbes finales GFP, plan de plaque 96 puits.",
    )

    imports = []
    for upload in uploads:
        try:
            workbook = _cached_workbook(upload.getvalue())
        except ValueError as error:
            st.error(f"Import impossible pour « {upload.name} » : {error}")
            continue
        imports.append((upload.name, workbook, _wells_by_strain(workbook.plate)))

    if not imports:
        return

    options = list(dict.fromkeys(strain for _, _, groups in imports for strain in groups))
    selection_key = "cytotox_selected_strains"
    previous = st.session_state.get(selection_key, [])
    valid = [strain for strain in previous if strain in options]
    if st.button("All", key="cytotox_select_all", help="Sélectionner toutes les souches et conditions"):
        st.session_state[selection_key] = options
    elif previous != valid:
        st.session_state[selection_key] = valid
    selected = set(st.multiselect("Souches et conditions à analyser", options, key=selection_key))

    for filename, workbook, wells in imports:
        st.subheader(filename)
        with st.expander("Plan de plaque · 96 puits", expanded=len(imports) == 1 and not selected):
            plate_html = workbook.plate.to_html(classes="cytotox-plate", escape=True, border=0)
            st.html(f'{PLATE_STYLE}<div class="cytotox-plate-scroll">{plate_html}</div>')

        if not selected:
            continue
        selected_wells = {strain: group for strain, group in wells.items() if strain in selected}
        if not selected_wells:
            st.info("Aucune des souches sélectionnées ne figure sur cette plaque.")
            continue

        for title, signal in zip(("GFP raw", "IP raw"), workbook.signals[:2]):
            st.image(_raw_figure_png(signal, selected_wells, f"{Path(filename).stem} — {title}"))

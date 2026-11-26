"""Streamlit interface for inspecting Cytotox measurements by strain."""

from pathlib import Path
from textwrap import fill

from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st

from .cytotox import CytotoxSignal, CytotoxWorkbook, auc_by_well, normalize_ip, read_cytotox_workbook
from .cytotox_grid import review_exclusions
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


def _signal_figure(
    signal: CytotoxSignal, wells_by_strain: dict[str, list[str]], title: str,
) -> Figure:
    """Plot one panel per strain and one curve per retained well."""
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
    if 0 <= lowest_value < float("inf"):
        axes[0, 0].set_ylim(bottom=0)
    if replicate_lines:
        handles = [replicate_lines[key] for key in sorted(replicate_lines)]
        figure.legend(
            handles=handles, loc="outside lower center", title="Répétition technique",
            ncols=min(6, len(handles)), fontsize=9, frameon=False,
        )
    return figure


@st.cache_data(show_spinner="Préparation des courbes…", max_entries=24)
def _signal_figure_png(
    signal: CytotoxSignal, wells_by_strain: dict[str, list[str]], title: str,
) -> bytes:
    return figure_bytes(_signal_figure(signal, wells_by_strain, title), "png", dpi=125)


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
        imports.append((upload.file_id, upload.name, workbook, _wells_by_strain(workbook.plate)))

    if not imports:
        return

    options = list(dict.fromkeys(strain for _, _, _, groups in imports for strain in groups))
    selection_key = "cytotox_selected_strains"
    previous = st.session_state.get(selection_key, [])
    valid = [strain for strain in previous if strain in options]
    if st.button("All", key="cytotox_select_all", help="Sélectionner toutes les souches et conditions"):
        st.session_state[selection_key] = options
    elif previous != valid:
        st.session_state[selection_key] = valid
    validated_key = "cytotox_validated_strains"
    with st.form("cytotox_strain_selection"):
        draft = st.multiselect("Souches et conditions à analyser", options, key=selection_key)
        if st.form_submit_button("Valider les souches et conditions"):
            st.session_state[validated_key] = draft
    selected = set(st.session_state.get(validated_key, ())).intersection(options)

    for file_id, filename, workbook, wells in imports:
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

        figure_title = Path(filename).stem
        for title, signal in zip(("GFP raw", "IP raw"), workbook.signals[:2]):
            st.image(_signal_figure_png(signal, selected_wells, f"{figure_title} — {title}"))
        _render_normalized(file_id, figure_title, workbook, selected_wells)


def _render_normalized(
    file_id: str, title: str, workbook: CytotoxWorkbook, selected_wells: dict[str, list[str]],
) -> None:
    gfp, ip, gfp_final = workbook.signals
    last_minute = max(0.0, float(ip.data.index.max() * 60))
    cutoff_key = f"cytotox_ip_validated_end_{file_id}"
    with st.form(key=f"cytotox_ip_window_{file_id}"):
        draft_end = st.number_input(
            "Temps maximal pour l’IP (min)", min_value=0.0, max_value=last_minute,
            value=min(1200.0, last_minute), step=1.0,
        )
        if st.form_submit_button("Valider la période IP"):
            st.session_state[cutoff_key] = draft_end

    end_minutes = st.session_state.get(cutoff_key)
    if end_minutes is None:
        return

    try:
        ip_data = normalize_ip(gfp.data, ip.data, end_minutes)
    except ValueError as error:
        st.error(str(error))
        return

    wells = {well for group in selected_wells.values() for well in group}
    ip_data = ip_data.loc[:, ip_data.columns.isin(wells)]
    gfp_data = gfp_final.data.loc[:, gfp_final.data.columns.isin(wells)]
    missing_baseline = ip.data.loc[0].reindex(ip_data.columns).isna() | (
        gfp.data.loc[0].reindex(ip_data.columns).fillna(0).eq(0)
    )
    if missing_baseline.any():
        st.warning(
            "IP normalisé indisponible pour les puits dont IP à t = 0 est manquant "
            "ou GFP à t = 0 est nul ou manquant : " + ", ".join(missing_baseline[missing_baseline].index)
        )

    signals = (
        CytotoxSignal("GFP normalisé", gfp_final.sheet_name, "GFP normalisé", gfp_data),
        CytotoxSignal("IP normalisé", ip.sheet_name, "IP normalisé", ip_data),
    )
    with st.container():
        st.subheader("Courbes normalisées")
        st.caption(f"GFP repris de « {gfp_final.sheet_name.strip()} ». IP calculé jusqu’à {end_minutes:g} min inclus.")
        figures = st.container()
        st.subheader("Points à exclure")
        st.caption("Cliquez sur les cellules à exclure. Un second clic les décoche.")
        filtered, excluded_count, confirmed = review_exclusions(signals, file_id=file_id)
        if not confirmed:
            with figures:
                for signal in filtered:
                    st.image(_signal_figure_png(signal, selected_wells, f"{title} — {signal.name}"))
            return
        st.caption(f"Points exclus des courbes normalisées : {excluded_count}")
        _render_final_results(title, filtered, selected_wells)


def _render_final_results(
    title: str, signals: tuple[CytotoxSignal, ...], selected_wells: dict[str, list[str]],
) -> None:
    st.subheader("Courbes finales")
    for signal in signals:
        st.image(_signal_figure_png(signal, selected_wells, f"{title} — {signal.name}"))

    st.caption(
        "AUC par puits. Les trapèzes relient les mesures conservées. "
        "Un puits sans mesure à l’une des bornes est omis."
    )
    for signal in reversed(signals):  # IP first, then GFP
        label = f"AUC {signal.name}"
        st.subheader(label)
        auc = auc_by_well(signal.data)
        if auc.isna().all():
            st.info("Aucune AUC disponible. Chaque puits doit avoir une mesure à chacune des deux bornes.")
            continue

        start_minute = float(signal.data.index[0] * 60)
        end_minute = float(signal.data.index[-1] * 60)
        samples = {strain: auc.reindex(wells).dropna() for strain, wells in selected_wells.items()}
        st.image(_auc_boxplot_png(
            samples, f"{title} — {label} ({start_minute:g}–{end_minute:g} min)", label,
        ))


@st.cache_data(show_spinner=False, max_entries=12)
def _auc_boxplot_png(samples: dict[str, pd.Series], title: str, label: str) -> bytes:
    figure = Figure(figsize=(max(7, len(samples) * .68), 5.3), layout="constrained")
    axis = figure.subplots()
    positions = []
    boxes = []
    for position, values in enumerate(samples.values(), start=1):
        if values.empty:
            continue
        positions.append(position)
        boxes.append(values)
        offsets = np.linspace(-.13, .13, len(values)) if len(values) > 1 else [0]
        axis.plot(position + np.asarray(offsets), values, "ko", markersize=3.5, zorder=3)

    axis.boxplot(boxes, positions=positions, widths=.43, showfliers=False, manage_ticks=False,
                 medianprops={"color": "black"})
    axis.set_xlim(.55, len(samples) + .55)
    axis.set_xticks(range(1, len(samples) + 1), list(samples), rotation=90, fontsize=8, parse_math=False)
    axis.set(xlabel="Souches et conditions", ylabel=f"{label} (signal × min)")
    axis.set_title(title, parse_math=False)
    return figure_bytes(figure, "png", dpi=125)

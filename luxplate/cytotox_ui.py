"""Streamlit interface for inspecting Cytotox measurements by strain."""

from pathlib import Path
from contextlib import nullcontext
from hashlib import sha256
from textwrap import fill

from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from .cytotox import (
    REFERENCE_STRAIN, CytotoxSignal, CytotoxWorkbook, auc_by_well, common_auc_window,
    normalize_ip, read_cytotox_workbook, relative_auc_by_well,
)
from .cytotox_grid import review_exclusions
from .display_order import reconcile_display_order
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
    selected = [strain for strain in st.session_state.get(validated_key, ()) if strain in options]
    order_key = "cytotox_strain_order"
    ordered = reconcile_display_order(st.session_state.get(order_key, []), selected)
    if len(ordered) > 1:
        with st.expander("Ordre des souches et conditions"):
            st.caption("Glissez les étiquettes pour définir l’ordre des figures et des box plots.")
            membership = sha256(repr(sorted(selected)).encode()).hexdigest()
            ordered = reconcile_display_order(
                sort_items(ordered, direction="vertical", key=f"cytotox_sort_{membership}"), selected,
            )
    if ordered:
        st.session_state[order_key] = ordered

    multiple_files = len(imports) > 1
    participating = sum(any(strain in wells for strain in ordered) for _, _, _, wells in imports)
    compare = participating > 1
    workbook_labels = [f"{number} · {filename}" for number, (_, filename, _, _) in enumerate(imports, 1)]
    if multiple_files:
        st.caption("Ouvre l’onglet de chaque classeur pour valider sa période IP et ses exclusions.")
        panels = st.tabs(workbook_labels + (["Synthèse inter-manip"] if compare else []))
    else:
        panels = [st.container()]

    completed = []
    pending = []
    for number, (file_id, filename, workbook, wells) in enumerate(imports, start=1):
        with panels[number - 1]:
            st.subheader(filename)
            with st.expander("Plan de plaque · 96 puits", expanded=not multiple_files and not selected):
                plate_html = workbook.plate.to_html(classes="cytotox-plate", escape=True, border=0)
                st.html(f'{PLATE_STYLE}<div class="cytotox-plate-scroll">{plate_html}</div>')

            if not selected:
                continue
            selected_wells = {strain: wells[strain] for strain in ordered if strain in wells}
            if not selected_wells:
                st.info("Aucune des souches sélectionnées ne figure sur cette plaque.")
                continue

            figure_title = Path(filename).stem
            raw_figures = st.expander("Courbes brutes GFP et IP") if multiple_files else nullcontext()
            with raw_figures:
                for title, signal in zip(("GFP raw", "IP raw"), workbook.signals[:2]):
                    st.image(_signal_figure_png(signal, selected_wells, f"{figure_title} — {title}"))
            reference_wells = wells.get(REFERENCE_STRAIN, []) if compare else []
            signals, action = _render_normalized(
                file_id, figure_title, workbook, selected_wells,
                reference_wells=reference_wells, compact=multiple_files,
            )
            if signals is not None:
                completed.append((f"Rép. biologique {number} · {filename}", selected_wells, reference_wells, signals))
            else:
                pending.append(f"{workbook_labels[number - 1]} · {action}")

    if compare:
        with panels[-1]:
            st.header("Synthèse inter-manip")
            if pending:
                st.info(f"{len(completed)} classeur(s) sur {participating} validé(s).")
                for status in pending:
                    st.text(status)
                st.caption("Clique sur « Valider les exclusions » même si tu ne coches aucun point.")
            else:
                _render_intermanip_results(completed, ordered)


def _render_normalized(
    file_id: str, title: str, workbook: CytotoxWorkbook, selected_wells: dict[str, list[str]],
    *, reference_wells: list[str], compact: bool = False,
) -> tuple[tuple[CytotoxSignal, ...] | None, str]:
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
        return None, "Période IP à valider"

    try:
        ip_data = normalize_ip(gfp.data, ip.data, end_minutes)
    except ValueError as error:
        st.error(str(error))
        return None, "Normalisation IP indisponible"

    wells = set(reference_wells).union(well for group in selected_wells.values() for well in group)
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
        if reference_wells and REFERENCE_STRAIN not in selected_wells:
            st.caption(
                f"Puits de référence {REFERENCE_STRAIN}, inclus pour la synthèse inter-manip : "
                + ", ".join(reference_wells)
            )
        filtered, excluded_count, confirmed = review_exclusions(signals, file_id=file_id)
        if not confirmed:
            with figures:
                preview = st.expander("Aperçu des courbes normalisées") if compact else nullcontext()
                with preview:
                    for signal in filtered:
                        st.image(_signal_figure_png(signal, selected_wells, f"{title} — {signal.name}"))
            return None, "Exclusions à valider"
        st.caption(f"Points exclus des courbes normalisées : {excluded_count}")
        _render_final_results(title, filtered, selected_wells)
        return filtered, ""


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
        samples = {strain: {"": auc.reindex(wells).dropna()} for strain, wells in selected_wells.items()}
        if all(groups[""].empty for groups in samples.values()):
            st.info("Aucune AUC disponible. Chaque puits doit avoir une mesure à chacune des deux bornes.")
            continue

        start_minute = float(signal.data.index[0] * 60)
        end_minute = float(signal.data.index[-1] * 60)
        st.image(_auc_boxplot_png(
            samples, f"{title} — {label} ({start_minute:g}–{end_minute:g} min)", f"{label} (signal × min)",
        ))


def _render_intermanip_results(
    completed: list[tuple[str, dict[str, list[str]], list[str], tuple[CytotoxSignal, ...]]],
    ordered: list[str],
) -> None:
    st.caption(
        f"Chaque point est l’AUC d’un puits divisée par la médiane des AUC de {REFERENCE_STRAIN} "
        "du même classeur. Chaque couleur identifie un réplicat biologique."
    )
    for signal_index, label in ((1, "IP"), (0, "GFP")):
        st.subheader(f"AUC {label} normalisé inter-manip")
        window = common_auc_window([signals[signal_index].data for _, _, _, signals in completed])
        if window is None:
            st.info("Aucune fenêtre mesurée commune pour ce signal.")
            continue
        start, end = window
        st.caption(f"Fenêtre commune : {start:g}–{end:g} min.")
        samples: dict[str, dict[str, pd.Series]] = {strain: {} for strain in ordered}
        for name, wells, reference_wells, signals in completed:
            ratios = relative_auc_by_well(signals[signal_index].data, reference_wells, window)
            if ratios is None:
                st.warning(f"{name} : médiane de référence {REFERENCE_STRAIN} indisponible ou nulle pour {label}.")
                ratios = pd.Series(dtype=float)
            for strain in ordered:
                samples[strain][name] = ratios.reindex(wells.get(strain, [])).dropna()

        if all(values.empty for groups in samples.values() for values in groups.values()):
            st.info("Aucun puits exploitable pour cette comparaison.")
            continue
        st.image(_auc_boxplot_png(
            samples, f"AUC {label} inter-manip ({start:g}–{end:g} min)", f"AUC {label} / médiane {REFERENCE_STRAIN}",
        ))


@st.cache_data(show_spinner=False, max_entries=12)
def _auc_boxplot_png(samples: dict[str, dict[str, pd.Series]], title: str, label: str) -> bytes:
    figure = Figure(figsize=(max(7, len(samples) * .68), 5.3), layout="constrained")
    axis = figure.subplots()
    replicates = list(dict.fromkeys(name for groups in samples.values() for name in groups))
    combined = any(replicates)
    palette = colormaps["tab10"].colors
    colors = {name: palette[index % len(palette)] for index, name in enumerate(replicates)}
    offsets = np.linspace(-.22, .22, len(replicates)) if len(replicates) > 1 else [0]
    shifts = dict(zip(replicates, offsets))
    spread = .08 / len(replicates) if combined else .13
    labelled = set()
    positions = []
    boxes = []
    for position, groups in enumerate(samples.values(), start=1):
        observed = [values for values in groups.values() if not values.empty]
        if not observed:
            continue
        positions.append(position)
        boxes.append(pd.concat(observed, ignore_index=True))
        for name, values in groups.items():
            if values.empty:
                continue
            jitter = np.linspace(-spread, spread, len(values)) if len(values) > 1 else [0]
            axis.plot(
                position + shifts[name] + np.asarray(jitter), values, linestyle="none", marker="o",
                color=colors[name] if combined else "black", markersize=3.5, zorder=3,
                label=name if name not in labelled else "_nolegend_",
            )
            labelled.add(name)

    axis.boxplot(boxes, positions=positions, widths=.43, showfliers=False, manage_ticks=False,
                 medianprops={"color": "black"})
    axis.set_xlim(.55, len(samples) + .55)
    axis.set_xticks(range(1, len(samples) + 1), list(samples), rotation=90, fontsize=8, parse_math=False)
    axis.set(xlabel="Souches et conditions", ylabel=label)
    axis.set_title(title, parse_math=False)
    if combined:
        axis.axhline(1, color="0.6", linestyle=":", linewidth=1)
        figure.legend(loc="outside upper center", fontsize=8, ncols=min(3, len(labelled)), frameon=False)
    return figure_bytes(figure, "png", dpi=125)

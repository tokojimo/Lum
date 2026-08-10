"""Matplotlib scientific figures used by the application."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from luxplate.kinetics import run_kinetics
from luxplate.statistics import paired_nonparametric_tests


PUBLICATION_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def _publication_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.tick_params(width=0.8, length=3)


def _series_label(row: pd.Series) -> str:
    return str(row.get("souche", row.get("sample_header", "Series")))


def _display_strain(value: object) -> str:
    """Use compact reporter names on axes while retaining full names in the data."""
    strain = str(value)
    for reporter in ("PspeD2-1A-lux", "PspeD2-3B-lux", "P0-lux"):
        if reporter.casefold() in strain.casefold():
            return reporter.removesuffix("-lux")
    return strain


def _display_panel(value: object) -> str:
    """Turn machine-oriented group identifiers into publication labels."""
    label = str(value).strip()
    match = re.fullmatch(r"exp(?:eriment)?\s*(\d+)\s*\|\s*([^()]+)(?:\s*\([^)]*\))?", label,
                         flags=re.IGNORECASE)
    if match:
        return f"Experiment {match.group(1)} – {match.group(2).strip()}"
    return label.replace("|", " – ")


def _strain_colors(data: pd.DataFrame) -> dict[str, str]:
    """Assign each strain one deterministic color for every panel in a figure."""
    strains = list(dict.fromkeys(data["souche"].dropna().astype(str)))
    return {strain: PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)]
            for index, strain in enumerate(strains)}


def plot_publication_panels(data: pd.DataFrame, *, value: str, group_by: str = "Groupe",
                            title: str | None = None, y_scale: str = "linear"):
    """Make one publication-style mean ± SD time-course panel per group."""
    required = {"temps_h", "souche", "sample_header", value, group_by}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for figure: {sorted(missing)}")
    if y_scale not in {"linear", "log"}:
        raise ValueError("Scale must be 'linear' or 'log'.")
    work = data.loc[data[value].notna()].copy()
    if y_scale == "log":
        work = work.loc[pd.to_numeric(work[value], errors="coerce").gt(0)]
    if "type" in work:
        work = work.loc[work["type"].eq("souche")]
    panels = list(dict.fromkeys(work[group_by].dropna().astype(str)))
    if not panels:
        raise ValueError("No usable data available for the final figures.")
    strain_colors = _strain_colors(work)
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.15 * nrows), squeeze=False)
    ylabel = {"DO_corr": r"Blank-corrected OD$_{600}$",
              "Lum_corr": "Blank-corrected luminescence (RLU)",
              "Lum_norm": r"Normalized luminescence (RLU/OD$_{600}$)"}.get(value, value)
    for panel_index, panel in enumerate(panels):
        axis = axes.flat[panel_index]
        subset = work.loc[work[group_by].astype(str).eq(panel)]
        strains = list(dict.fromkeys(subset["souche"].astype(str)))
        for strain in strains:
            color = strain_colors[strain]
            strain_data = subset.loc[subset["souche"].astype(str).eq(strain)]
            summary = strain_data.groupby("temps_h", as_index=False)[value].agg(["mean", "std"]).reset_index()
            x = summary["temps_h"].to_numpy(float); y = summary["mean"].to_numpy(float)
            sd = summary["std"].fillna(0).to_numpy(float)
            axis.errorbar(x, y, yerr=sd, color=color, lw=1.6, capsize=2,
                          marker="o", markersize=2.5, label=strain)
        axis.set(title=_display_panel(panel), xlabel="Time (h)", ylabel=ylabel)
        axis.set_yscale(y_scale)
        axis.title.set_fontweight("bold"); axis.title.set_ha("left"); axis.title.set_position((0, 1.0))
        _publication_style(axis)
    for axis in axes.flat[len(panels):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, [_display_strain(label) for label in labels], title="Reporter",
                      frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, 0.93), ncol=min(5, len(labels)))
    figure.suptitle(title or f"{ylabel} over time", fontsize=13, fontweight="bold", y=0.995)
    # Identical limits prevent misleading visual comparisons between replicate panels.
    visible_axes = list(axes.flat[:len(panels)])
    if visible_axes:
        limits = np.asarray([axis.get_ylim() for axis in visible_axes], dtype=float)
        shared_limits = (float(np.nanmin(limits[:, 0])), float(np.nanmax(limits[:, 1])))
        for axis in visible_axes:
            axis.set_ylim(shared_limits)
    figure.subplots_adjust(top=0.84, bottom=0.12, hspace=0.48, wspace=0.34)
    return figure


def _mean_sd(data: pd.DataFrame, value: str) -> pd.DataFrame:
    return data.groupby("temps_h", as_index=False)[value].agg(["mean", "std"]).reset_index()


def plot_mixed_panels(data: pd.DataFrame, *, lum_scale: str = "linear",
                      uncertainty: str = "bars", title: str | None = None,
                      media: list[str] | None = None, strains: list[str] | None = None):
    """Plot aligned OD and normalized-luminescence panels without dual y-axes."""
    required = {"temps_h", "souche", "Groupe", "DO_corr", "Lum_norm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for combined figure: {sorted(missing)}")
    if lum_scale not in {"linear", "log"} or uncertainty not in {"bars", "ribbon"}:
        raise ValueError("Unknown scale or uncertainty representation.")
    work = data.loc[data.get("type", "souche").eq("souche")].copy() if "type" in data else data.copy()
    if media:
        work = work.loc[work["Groupe"].astype(str).isin(media)]
    if strains:
        work = work.loc[work["souche"].astype(str).isin(strains)]
    panels = list(dict.fromkeys(work["Groupe"].dropna().astype(str)))
    if not panels:
        raise ValueError("No condition selected for combined figure.")
    strain_colors = _strain_colors(work)
    ncols = min(3, len(panels)); blocks = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(2 * blocks, ncols, figsize=(4.3 * ncols, 5.4 * blocks), squeeze=False)
    legend_handles = []
    for panel_index, medium in enumerate(panels):
        block, column = divmod(panel_index, ncols)
        top, bottom = axes[2 * block, column], axes[2 * block + 1, column]
        subset = work.loc[work["Groupe"].astype(str).eq(medium)]
        for strain, strain_data in subset.groupby("souche", sort=False):
            color = strain_colors[str(strain)]
            od = _mean_sd(strain_data, "DO_corr"); lum = _mean_sd(strain_data, "Lum_norm")
            ox = od["temps_h"].to_numpy(float); oy = od["mean"].to_numpy(float)
            osd = od["std"].fillna(0).to_numpy(float)
            lx = lum["temps_h"].to_numpy(float); ly = lum["mean"].to_numpy(float)
            lsd = lum["std"].fillna(0).to_numpy(float)
            if uncertainty == "ribbon":
                top.fill_between(ox, oy - osd, oy + osd, color=color, alpha=.14, linewidth=0)
                lower = ly - lsd
                if lum_scale == "log": lower = np.where(lower > 0, lower, np.nan)
                bottom.fill_between(lx, lower, ly + lsd, color=color, alpha=.09, linewidth=0)
                od_line, = top.plot(ox, oy, color=color, lw=1.8, marker="o", markersize=2.5,
                                     markerfacecolor="white", label=str(strain))
                bottom.plot(lx, np.where(ly > 0, ly, np.nan) if lum_scale == "log" else ly,
                           color=color, lw=1.6, marker="s", markersize=2.3,
                           markerfacecolor="white")
            else:
                od_line = top.errorbar(ox, oy, yerr=osd, color=color, lw=1.35, marker="o",
                    markersize=2.5, markerfacecolor="white", capsize=2, label=str(strain)).lines[0]
                valid = ly > 0 if lum_scale == "log" else np.ones(len(ly), dtype=bool)
                bottom.errorbar(lx[valid], ly[valid], yerr=lsd[valid], color=color, lw=1.25,
                    marker="s", markersize=2.3, markerfacecolor="white", capsize=2)
            if panel_index == 0: legend_handles.append(od_line)
        top.set(title=_display_panel(medium), ylabel=r"Blank-corrected OD$_{600}$")
        bottom.set(xlabel="Time (h)", ylabel=r"Normalized luminescence (RLU/OD$_{600}$)")
        bottom.set_yscale(lum_scale); top.tick_params(labelbottom=False)
        top.title.set_fontweight("bold"); top.title.set_ha("left"); top.title.set_position((0, 1))
        _publication_style(top); _publication_style(bottom)
    for panel_index in range(len(panels), blocks * ncols):
        block, column = divmod(panel_index, ncols)
        axes[2 * block, column].remove(); axes[2 * block + 1, column].remove()
    figure.legend(legend_handles, [_display_strain(line.get_label()) for line in legend_handles], title="Reporter",
                  frameon=False, loc="upper center", bbox_to_anchor=(.5, .945),
                  ncol=min(5, len(legend_handles)))
    figure.suptitle(title or "Growth and normalized luminescence", fontweight="bold", y=.995)
    figure.subplots_adjust(top=.80, bottom=.11, hspace=.12, wspace=.38)
    return figure


def plot_metric_points(metrics: pd.DataFrame, *, metric: str, y_scale: str = "linear",
                       title: str | None = None):
    """Plot mean ± biological SD, technical wells, and biological means."""
    required = {"souche", "Groupe", metric}
    missing = required.difference(metrics.columns)
    if missing: raise ValueError(f"Missing columns for metric: {sorted(missing)}")
    if y_scale not in {"linear", "log"}:
        raise ValueError("Scale must be 'linear' or 'log'.")
    # Treat infinities like missing measurements.  In particular, ``+inf``
    # passes a simple ``> 0`` check but does not give Matplotlib a finite
    # positive bound for a logarithmic axis, causing LogLocator to fail during
    # layout with "Data cannot be log-scaled".
    numeric_metric = pd.to_numeric(metrics[metric], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    work = metrics.loc[numeric_metric.notna()].copy()
    work[metric] = numeric_metric.loc[work.index]
    effective_scale = y_scale
    if y_scale == "log":
        positive = work[metric].gt(0)
        if positive.any():
            work = work.loc[positive]
        else:
            # Matplotlib cannot lay out an empty/non-positive logarithmic axis.
            # Retain the available observations on a linear axis instead of
            # crashing the complete publication-figure gallery.
            effective_scale = "linear"
    strains = list(dict.fromkeys(work["souche"].astype(str)))
    # Some exports include an ``experience_id`` column even though it is wholly
    # empty.  Such a column is metadata, not a usable biological identifier:
    # comparing its NaN identity with ``Series.eq`` would select no rows and
    # leave the axis without points.  Prefer only identifiers that contain at
    # least one real value, falling back to the condition below when necessary.
    biological_columns = [
        column for column in ("experience_id", "replicat")
        if column in work and work[column].notna().any()
    ]
    group_columns = ["souche", "Groupe", *biological_columns]
    technical = work.copy()
    work = work.groupby(group_columns, dropna=False, sort=False)[metric].mean().reset_index()
    biological_columns = biological_columns or ["Groupe"]
    figure, axis = plt.subplots(figsize=(max(6, 1.15 * len(strains)), 4.5))
    rng = np.random.default_rng(1947)
    colors = _strain_colors(work)
    summaries = work.groupby("souche", sort=False)[metric].agg(["mean", "std"])
    for strain_index, strain in enumerate(strains):
        if strain not in summaries.index:
            continue
        row = summaries.loc[strain]
        axis.bar(strain_index, row["mean"], width=.62, color=colors[strain], alpha=.28,
                 edgecolor=colors[strain], linewidth=1, zorder=1)
        error = 0 if pd.isna(row["std"]) else row["std"]
        axis.plot([strain_index, strain_index], [row["mean"] - error, row["mean"] + error],
                  color=colors[strain], lw=1.3, zorder=4)
        axis.plot([strain_index - .05, strain_index + .05], [row["mean"] - error] * 2,
                  color=colors[strain], lw=1.3, zorder=4)
        axis.plot([strain_index - .05, strain_index + .05], [row["mean"] + error] * 2,
                  color=colors[strain], lw=1.3, zorder=4)
        raw = technical.loc[technical["souche"].astype(str).eq(strain), metric].to_numpy(float)
        axis.plot(strain_index + rng.uniform(-.16, .16, len(raw)), raw, linestyle="none",
                  marker="o", markersize=2.8, color=colors[strain], alpha=.32, zorder=2)
    identities = list(work[biological_columns].drop_duplicates().itertuples(index=False, name=None))
    for identity_index, identity in enumerate(identities):
        mask = np.ones(len(work), dtype=bool)
        for column, value in zip(biological_columns, identity):
            # ``NaN != NaN``; explicitly match missing identities so partially
            # populated identifier columns cannot silently discard observations.
            matches = work[column].isna() if pd.isna(value) else work[column].eq(value)
            mask &= matches.to_numpy()
        subset = work.loc[mask]
        xs, ys = [], []
        for strain_index, strain in enumerate(strains):
            values = subset.loc[subset["souche"].astype(str).eq(strain), metric].to_numpy(float)
            if len(values):
                x = strain_index + rng.uniform(-.025, .025)
                xs.append(x); ys.append(values[0])
                axis.scatter(x, values[0], s=48, marker="o",
                    color=colors[strain],
                    edgecolor="white", linewidth=.6, zorder=3,
                    label=" | ".join(map(str, identity)) if strain_index == 0 else None)
    metric_labels = {"lum_norm_peak": r"Peak normalized luminescence (RLU/OD$_{600}$)",
        "lum_norm_auc": r"Normalized luminescence AUC (RLU/OD$_{600}$)·h",
        "doubling_time_h": "Doubling time (h)"}
    axis.set_xticks(range(len(strains)), [_display_strain(s) for s in strains])
    axis.set(xlabel="Reporter", ylabel=metric_labels.get(metric, metric), title=title or metric_labels.get(metric, metric))
    axis.set_yscale(effective_scale)
    axis.title.set_fontweight("bold"); _publication_style(axis)
    omnibus, comparisons = paired_nonparametric_tests(work, value=metric)
    strain_positions = {strain: index for index, strain in enumerate(strains)}
    usable = comparisons.loc[
        comparisons["condition_1"].isin(strain_positions)
        & comparisons["condition_2"].isin(strain_positions)
    ]
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    for level, comparison in enumerate(usable.itertuples(index=False), start=1):
        left = strain_positions[comparison.condition_1]
        right = strain_positions[comparison.condition_2]
        y = .82 + .055 * level
        p = comparison.p_holm
        stars = "****" if p < .0001 else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        axis.plot([left, left, right, right], [y - .012, y, y, y - .012],
                  transform=transform, color="#333333", lw=.7, clip_on=False)
        axis.text((left + right) / 2, y + .006, stars, transform=transform,
                  ha="center", va="bottom", fontsize=7)
    axis.text(.99, .01, "Friedman: " + (f"p = {omnibus:.3g}" if np.isfinite(omnibus) else "not estimable"),
              transform=axis.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
    axis.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=4, alpha=.35,
               color="#555555", label="Technical replicate"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               color="#555555", markeredgecolor="white", label="Biological mean"),
    ], title="Replicate display", frameon=False, fontsize=7, loc="center left",
       bbox_to_anchor=(1.01, .5))
    figure._luxplate_statistics = comparisons
    figure.tight_layout(); return figure


def build_control_comparisons(data: pd.DataFrame, *, control: str = "P0-lux",
                              lum_scale: str = "linear", uncertainty: str = "bars",
                              title: str = "Control comparison") -> list[tuple[str, object]]:
    """Build one aligned-panel comparison figure per reporter strain."""
    strains = list(dict.fromkeys(data["souche"].dropna().astype(str)))
    matching = [strain for strain in strains if strain.casefold() == control.casefold()]
    if not matching:
        raise ValueError(f"Control {control!r} is absent from the selected data.")
    actual_control = matching[0]
    figures = []
    for reporter in (strain for strain in strains if strain != actual_control):
        subset = data.loc[data["souche"].astype(str).isin([actual_control, reporter])]
        figure = plot_mixed_panels(subset, lum_scale=lum_scale, uncertainty=uncertainty,
                                  title=f"{title} — {reporter} vs {actual_control}")
        figures.append((f"comparaison_{reporter}_vs_{actual_control}", figure))
    return figures


def build_publication_figures(data: pd.DataFrame, *, title: str = "",
    families: tuple[str, ...] = ("growth", "normalized", "mixed", "peak", "auc", "doubling"),
    panel_by: str = "Groupe", lum_scale: str = "linear", normalized_scale: str = "linear",
    metric_scale: str = "log", uncertainty: str = "bars", control: str = "P0-lux") -> list[tuple[str, object]]:
    """Build the curve families represented in the historical example scripts."""
    figures = []
    choices = (("growth", "DO_corr", "croissance", "linear", "Growth"),
               ("corrected", "Lum_corr", "luminescence_corrigee", lum_scale,
                "Blank-corrected luminescence"),
               ("normalized", "Lum_norm", "luminescence_normalisee", normalized_scale,
                "Normalized luminescence"))
    for family, value, suffix, scale, figure_label in choices:
        if family in families and value in data and data[value].notna().any():
            figures.append((suffix, plot_publication_panels(data, value=value, group_by=panel_by,
                y_scale=scale, title=figure_label)))
    if "mixed" in families:
        figures.append(("croissance_luminescence_mixte", plot_mixed_panels(
            data, lum_scale=lum_scale, uncertainty=uncertainty,
            title="Growth and normalized luminescence")))
    metric_families = {"peak": ("lum_norm_peak", "pic_luminescence_normalisee", "Peak normalized luminescence"),
                       "auc": ("lum_norm_auc", "auc_luminescence_normalisee", "Normalized luminescence AUC"),
                       "doubling": ("doubling_time_h", "temps_doublement", "Doubling time")}
    requested = set(families).intersection(metric_families)
    if requested:
        metrics = run_kinetics(data).series_metrics
        for family in ("peak", "auc", "doubling"):
            if family in requested:
                metric, suffix, figure_label = metric_families[family]
                scale = "linear" if family == "doubling" else metric_scale
                figures.append((suffix, plot_metric_points(metrics, metric=metric, y_scale=scale,
                    title=figure_label)))
    if "control" in families:
        figures.extend(build_control_comparisons(data, control=control, lum_scale=lum_scale,
            uncertainty=uncertainty, title="Targeted control comparison"))
    return figures


def plot_raw_curves(data: pd.DataFrame):
    """Plot every raw technical curve; no averaging or interpolation."""
    required = {"temps_h", "souche", "sample_header", "DO_brute", "Lum_brute"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for header, curve in data.groupby("sample_header", sort=False):
        curve = curve.sort_values("temps_h")
        label = str(curve["souche"].iloc[0])
        axes[0].plot(curve["temps_h"], curve["DO_brute"], marker="o", markersize=3, alpha=0.75, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_brute"], marker="o", markersize=3, alpha=0.75, label=label)
    axes[0].set(title="Raw growth", xlabel="Time (h)", ylabel=r"Raw OD$_{600}$")
    axes[1].set(title="Raw luminescence", xlabel="Time (h)", ylabel="Raw luminescence (RLU)")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def build_guided_raw_figures(data: pd.DataFrame, *, sample_type: str) -> list[tuple[str, object]]:
    """Build one static DO/luminescence figure per biological replicate.

    Technical wells belonging to the same strain, medium and biological replicate
    stay together in a figure.  Keeping these previews as Matplotlib figures avoids
    the much heavier interactive charts in the guided import screen.
    """
    required = {"temps_h", "souche", "Groupe", "replicat", "sample_header", "type",
                "DO_brute", "Lum_brute"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    work = data.loc[data["type"].astype(str).str.lower().eq(sample_type.lower())].copy()
    figures: list[tuple[str, object]] = []
    for keys, replicate in work.groupby(["Groupe", "souche", "replicat"], dropna=False, sort=False):
        medium, strain, biological_replicate = keys
        title = f"{medium} · {strain} · biological replicate {biological_replicate}"
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for header, curve in replicate.groupby("sample_header", sort=False):
            curve = curve.sort_values("temps_h", kind="stable")
            label = str(header)
            axes[0].plot(curve["temps_h"], curve["DO_brute"], lw=1.4, label=label)
            axes[1].plot(curve["temps_h"], curve["Lum_brute"], lw=1.4, label=label)
        axes[0].set(title="Optical density", xlabel="Time (h)", ylabel=r"Raw OD$_{600}$")
        axes[1].set(title="Luminescence", xlabel="Time (h)", ylabel="Raw luminescence (RLU)")
        for axis in axes:
            _publication_style(axis)
            axis.legend(fontsize="x-small", frameon=False, loc="best")
        figure.suptitle(title, fontsize=11, fontweight="bold")
        figures.append((title, figure))
    return figures


def plot_qc_curves(data: pd.DataFrame, anomalies: pd.DataFrame):
    """Plot unmodified curves and overlay proposed anomalies as red crosses."""
    figure = plot_raw_curves(data)
    axes = figure.axes
    if not anomalies.empty:
        axes[0].scatter(
            anomalies["temps_h"], anomalies["DO_brute"], marker="x", s=90,
            linewidths=2.2, color="crimson", zorder=10, label="Anomalie proposée",
        )
        axes[1].scatter(
            anomalies["temps_h"], anomalies["Lum_brute"], marker="x", s=90,
            linewidths=2.2, color="crimson", zorder=10, label="Anomalie proposée",
        )
        for axis in axes:
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
    return figure


def plot_blank_correction(data: pd.DataFrame):
    """Compare raw and blank-corrected strain curves."""
    required = {"temps_h", "souche", "sample_header", "DO_brute", "Lum_brute", "DO_corr", "Lum_corr"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for _, curve in data.groupby("sample_header", sort=False):
        curve = curve.sort_values("temps_h")
        label = str(curve["souche"].iloc[0])
        axes[0].plot(curve["temps_h"], curve["DO_brute"], color="0.7", alpha=0.45)
        axes[0].plot(curve["temps_h"], curve["DO_corr"], alpha=0.8, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_brute"], color="0.7", alpha=0.45)
        axes[1].plot(curve["temps_h"], curve["Lum_corr"], alpha=0.8, label=label)
    axes[0].set(title="OD: raw (gray) / blank-corrected", xlabel="Time (h)", ylabel=r"OD$_{600}$")
    axes[1].set(title="Luminescence: raw (gray) / blank-corrected", xlabel="Time (h)", ylabel="Luminescence (RLU)")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def plot_normalization(data: pd.DataFrame):
    """Compare corrected and OD-normalized luminescence for strain series."""
    required = {"temps_h", "souche", "sample_header", "type", "Lum_corr", "Lum_norm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    strains = data.loc[data["type"].eq("souche")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for _, curve in strains.groupby("sample_header", sort=False):
        curve = curve.sort_values("temps_h")
        label = str(curve["souche"].iloc[0])
        axes[0].plot(curve["temps_h"], curve["Lum_corr"], alpha=0.8, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_norm"], alpha=0.8, label=label)
    axes[0].set(title="Blank-corrected luminescence", xlabel="Time (h)", ylabel="Luminescence (RLU)")
    axes[1].set(title="OD-normalized luminescence", xlabel="Time (h)",
                ylabel=r"Normalized luminescence (RLU/OD$_{600}$)")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def plot_kinetics(data: pd.DataFrame, series_metrics: pd.DataFrame):
    """Plot strain curves with stable colors and one compact legend entry per strain."""
    required = {"temps_h", "sample_header", "DO_corr", "Lum_norm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    work = data.loc[data["type"].eq("souche")].copy() if "type" in data else data.copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    strain_colors = _strain_colors(work)
    labelled_strains: set[str] = set()
    group_columns = [column for column in
                     ("experience_id", "souche", "Groupe", "sample_header", "puits", "replicat")
                     if column in work]
    for keys, curve in work.groupby(group_columns, dropna=False, sort=False):
        curve = curve.sort_values("temps_h", kind="stable")
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        strain = str(identity.get("souche", _series_label(curve.iloc[0])))
        label = strain if strain not in labelled_strains else "_nolegend_"
        color = strain_colors[strain]
        axes[0].plot(curve["temps_h"], curve["DO_corr"], marker="o", markersize=2,
                     color=color, alpha=.75, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_norm"], marker="o", markersize=2,
                     color=color, alpha=.75, label=label)
        labelled_strains.add(strain)
        selected = series_metrics
        for column, value in identity.items():
            selected = selected.loc[selected[column].eq(value)]
        if selected.empty:
            continue
        metric = selected.iloc[0]
        axes[0].scatter(metric["od_max_time_h"], metric["od_max"], marker="*", s=90, zorder=5)
        axes[1].scatter(metric["lum_norm_peak_time_h"], metric["lum_norm_peak"], marker="*", s=90, zorder=5)
        if pd.notna(metric["max_growth_rate_start_h"]):
            axes[0].axvspan(metric["max_growth_rate_start_h"], metric["max_growth_rate_end_h"],
                            alpha=0.10, color="green")
    axes[0].set(title="Growth and maximum-rate window", xlabel="Time (h)",
                ylabel=r"Blank-corrected OD$_{600}$")
    axes[1].set(title="Luminescence and peak", xlabel="Time (h)",
                ylabel=r"Normalized luminescence (RLU/OD$_{600}$)")
    for axis in axes:
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, title="Strain", fontsize="small", loc="best")
    return figure

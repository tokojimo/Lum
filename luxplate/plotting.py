"""Matplotlib scientific figures used by the application."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from luxplate.kinetics import run_kinetics


PUBLICATION_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def _publication_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.tick_params(width=0.8, length=3)


def _series_label(row: pd.Series) -> str:
    return str(row.get("souche", row.get("sample_header", "Série")))


def plot_publication_panels(data: pd.DataFrame, *, value: str, group_by: str = "Groupe",
                            title: str | None = None, y_scale: str = "linear"):
    """Make one polished mean ± SD time-course panel per medium or strain.

    Fine lines retain the technical-series information used throughout the example
    scripts; the heavy line and translucent ribbon show mean and standard deviation.
    """
    required = {"temps_h", "souche", "sample_header", value, group_by}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    if y_scale not in {"linear", "log"}:
        raise ValueError("L'échelle doit être 'linear' ou 'log'.")
    work = data.loc[data[value].notna()].copy()
    if y_scale == "log":
        work = work.loc[pd.to_numeric(work[value], errors="coerce").gt(0)]
    if "type" in work:
        work = work.loc[work["type"].eq("souche")]
    panels = list(dict.fromkeys(work[group_by].dropna().astype(str)))
    if not panels:
        raise ValueError("Aucune donnée exploitable pour construire les figures finales.")
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.15 * nrows), squeeze=False)
    ylabel = {"DO_corr": "Densité optique corrigée", "Lum_corr": "Luminescence corrigée (RLU)",
              "Lum_norm": "Luminescence normalisée (RLU / DO)"}.get(value, value)
    for panel_index, panel in enumerate(panels):
        axis = axes.flat[panel_index]
        subset = work.loc[work[group_by].astype(str).eq(panel)]
        strains = list(dict.fromkeys(subset["souche"].astype(str)))
        for index, strain in enumerate(strains):
            color = PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)]
            strain_data = subset.loc[subset["souche"].astype(str).eq(strain)]
            for _, curve in strain_data.groupby("sample_header", sort=False):
                curve = curve.sort_values("temps_h")
                axis.plot(curve["temps_h"], curve[value], color=color, lw=0.7, alpha=0.20)
            summary = strain_data.groupby("temps_h", as_index=False)[value].agg(["mean", "std"]).reset_index()
            x = summary["temps_h"].to_numpy(float); y = summary["mean"].to_numpy(float)
            sd = summary["std"].fillna(0).to_numpy(float)
            axis.fill_between(x, y - sd, y + sd, color=color, alpha=0.14, linewidth=0)
            axis.plot(x, y, color=color, lw=2, label=strain)
        axis.set(title=panel, xlabel="Temps (h)", ylabel=ylabel)
        axis.set_yscale(y_scale)
        axis.title.set_fontweight("bold"); axis.title.set_ha("left"); axis.title.set_position((0, 1.0))
        _publication_style(axis)
    for axis in axes.flat[len(panels):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, title="Souche", frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, 0.93), ncol=min(5, len(labels)))
    figure.suptitle(title or f"{ylabel} au cours du temps", fontsize=13, fontweight="bold", y=0.995)
    figure.text(0.5, 0.955, "Lignes fines = puits techniques · ligne épaisse = moyenne · ruban = ± écart-type",
                ha="center", color="#555555", fontsize=8)
    figure.subplots_adjust(top=0.82, bottom=0.12, hspace=0.48, wspace=0.34)
    return figure


def _mean_sd(data: pd.DataFrame, value: str) -> pd.DataFrame:
    return data.groupby("temps_h", as_index=False)[value].agg(["mean", "std"]).reset_index()


def plot_mixed_panels(data: pd.DataFrame, *, lum_scale: str = "linear",
                      uncertainty: str = "bars", title: str | None = None,
                      media: list[str] | None = None, strains: list[str] | None = None):
    """Plot corrected OD and luminescence on configurable twin axes."""
    required = {"temps_h", "souche", "Groupe", "DO_corr", "Lum_corr"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure mixte : {sorted(missing)}")
    if lum_scale not in {"linear", "log"} or uncertainty not in {"bars", "ribbon"}:
        raise ValueError("Échelle ou représentation d'incertitude inconnue.")
    work = data.loc[data.get("type", "souche").eq("souche")].copy() if "type" in data else data.copy()
    if media:
        work = work.loc[work["Groupe"].astype(str).isin(media)]
    if strains:
        work = work.loc[work["souche"].astype(str).isin(strains)]
    panels = list(dict.fromkeys(work["Groupe"].dropna().astype(str)))
    if not panels:
        raise ValueError("Aucune condition sélectionnée pour la figure mixte.")
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), squeeze=False)
    legend_handles = []
    for panel_index, medium in enumerate(panels):
        left = axes.flat[panel_index]; right = left.twinx()
        subset = work.loc[work["Groupe"].astype(str).eq(medium)]
        for index, (strain, strain_data) in enumerate(subset.groupby("souche", sort=False)):
            color = PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)]
            od = _mean_sd(strain_data, "DO_corr"); lum = _mean_sd(strain_data, "Lum_corr")
            ox = od["temps_h"].to_numpy(float); oy = od["mean"].to_numpy(float)
            osd = od["std"].fillna(0).to_numpy(float)
            lx = lum["temps_h"].to_numpy(float); ly = lum["mean"].to_numpy(float)
            lsd = lum["std"].fillna(0).to_numpy(float)
            if uncertainty == "ribbon":
                left.fill_between(ox, oy - osd, oy + osd, color=color, alpha=.14, linewidth=0)
                lower = ly - lsd
                if lum_scale == "log": lower = np.where(lower > 0, lower, np.nan)
                right.fill_between(lx, lower, ly + lsd, color=color, alpha=.09, linewidth=0)
                od_line, = left.plot(ox, oy, color=color, lw=1.8, marker="o", markersize=2.5,
                                     markerfacecolor="white", label=str(strain))
                right.plot(lx, np.where(ly > 0, ly, np.nan) if lum_scale == "log" else ly,
                           color=color, lw=1.6, ls="--", marker="s", markersize=2.3,
                           markerfacecolor="white")
            else:
                od_line = left.errorbar(ox, oy, yerr=osd, color=color, lw=1.35, marker="o",
                    markersize=2.5, markerfacecolor="white", capsize=2, label=str(strain)).lines[0]
                od_line.set_label(str(strain))
                valid = ly > 0 if lum_scale == "log" else np.ones(len(ly), dtype=bool)
                right.errorbar(lx[valid], ly[valid], yerr=lsd[valid], color=color, lw=1.25,
                    ls="--", marker="s", markersize=2.3, markerfacecolor="white", capsize=2)
            if panel_index == 0: legend_handles.append(od_line)
        left.set(title=medium, xlabel="Temps (h)", ylabel="DO corrigée")
        right.set_ylabel("Luminescence corrigée (RLU)"); right.set_yscale(lum_scale)
        left.title.set_fontweight("bold"); left.title.set_ha("left"); left.title.set_position((0, 1))
        _publication_style(left); right.grid(False); right.spines["top"].set_visible(False)
    for axis in axes.flat[len(panels):]: axis.remove()
    figure.legend(legend_handles, [line.get_label() for line in legend_handles], title="Souche",
                  frameon=False, loc="upper center", bbox_to_anchor=(.5, .93),
                  ncol=min(5, len(legend_handles)))
    figure.suptitle(title or "Croissance et luminescence corrigée", fontweight="bold", y=.995)
    figure.text(.5, .80, "DO : trait plein, cercles · Luminescence : pointillés, carrés",
                ha="center", fontsize=8, color="#555555")
    figure.subplots_adjust(top=.72, bottom=.13, hspace=.68, wspace=.55)
    return figure


def plot_metric_points(metrics: pd.DataFrame, *, metric: str, y_scale: str = "linear",
                       title: str | None = None):
    """Plot one point per technical series, grouped by strain and medium."""
    required = {"souche", "Groupe", metric}
    missing = required.difference(metrics.columns)
    if missing: raise ValueError(f"Colonnes manquantes pour le paramètre : {sorted(missing)}")
    work = metrics.loc[pd.to_numeric(metrics[metric], errors="coerce").notna()].copy()
    if y_scale == "log": work = work.loc[work[metric].gt(0)]
    strains = list(dict.fromkeys(work["souche"].astype(str)))
    media = list(dict.fromkeys(work["Groupe"].astype(str)))
    figure, axis = plt.subplots(figsize=(max(6, 1.15 * len(strains)), 4.5))
    rng = np.random.default_rng(1947)
    markers = ("o", "s", "^", "D", "v", "P", "X")
    width = .65 / max(1, len(media))
    for medium_index, medium in enumerate(media):
        subset = work.loc[work["Groupe"].astype(str).eq(medium)]
        offset = (medium_index - (len(media) - 1) / 2) * width
        for strain_index, strain in enumerate(strains):
            values = subset.loc[subset["souche"].astype(str).eq(strain), metric].to_numpy(float)
            x = strain_index + offset + rng.uniform(-width * .22, width * .22, len(values))
            axis.scatter(x, values, s=32, marker=markers[medium_index % len(markers)],
                         color=PUBLICATION_COLORS[strain_index % len(PUBLICATION_COLORS)],
                         edgecolor="white", linewidth=.6, alpha=.95,
                         label=medium if strain_index == 0 else None)
    axis.set_xticks(range(len(strains)), strains, rotation=25, ha="right")
    axis.set(xlabel="Souche", ylabel=metric, title=title or metric); axis.set_yscale(y_scale)
    axis.title.set_fontweight("bold"); _publication_style(axis)
    if media: axis.legend(title="Milieu", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout(); return figure


def build_control_comparisons(data: pd.DataFrame, *, control: str = "P0-lux",
                              lum_scale: str = "linear", uncertainty: str = "bars",
                              title: str = "Comparaison au contrôle") -> list[tuple[str, object]]:
    """Build one twin-axis comparison figure per reporter strain."""
    strains = list(dict.fromkeys(data["souche"].dropna().astype(str)))
    matching = [strain for strain in strains if strain.casefold() == control.casefold()]
    if not matching:
        raise ValueError(f"Le contrôle {control!r} est absent des données sélectionnées.")
    actual_control = matching[0]
    figures = []
    for reporter in (strain for strain in strains if strain != actual_control):
        subset = data.loc[data["souche"].astype(str).isin([actual_control, reporter])]
        figure = plot_mixed_panels(subset, lum_scale=lum_scale, uncertainty=uncertainty,
                                  title=f"{title} — {reporter} vs {actual_control}")
        figures.append((f"comparaison_{reporter}_vs_{actual_control}", figure))
    return figures


def build_publication_figures(data: pd.DataFrame, *, title: str = "Analyse LuxPlate",
    families: tuple[str, ...] = ("growth", "normalized", "mixed", "peak", "auc", "doubling"),
    panel_by: str = "Groupe", lum_scale: str = "linear", normalized_scale: str = "linear",
    metric_scale: str = "log", uncertainty: str = "bars", control: str = "P0-lux") -> list[tuple[str, object]]:
    """Build the curve families represented in the historical example scripts."""
    figures = []
    choices = (("growth", "DO_corr", "croissance", "linear"),
               ("corrected", "Lum_corr", "luminescence_corrigee", lum_scale),
               ("normalized", "Lum_norm", "luminescence_normalisee", normalized_scale))
    for family, value, suffix, scale in choices:
        if family in families and value in data and data[value].notna().any():
            figures.append((suffix, plot_publication_panels(data, value=value, group_by=panel_by,
                y_scale=scale, title=f"{title} — {suffix.replace('_', ' ')}")))
    if "mixed" in families:
        figures.append(("croissance_luminescence_mixte", plot_mixed_panels(
            data, lum_scale=lum_scale, uncertainty=uncertainty, title=f"{title} — DO + luminescence")))
    metric_families = {"peak": ("lum_norm_peak", "pic_luminescence_normalisee"),
                       "auc": ("lum_norm_auc", "auc_luminescence_normalisee"),
                       "doubling": ("doubling_time_h", "temps_doublement")}
    requested = set(families).intersection(metric_families)
    if requested:
        metrics = run_kinetics(data).series_metrics
        for family in ("peak", "auc", "doubling"):
            if family in requested:
                metric, suffix = metric_families[family]
                scale = "linear" if family == "doubling" else metric_scale
                figures.append((suffix, plot_metric_points(metrics, metric=metric, y_scale=scale,
                    title=f"{title} — {suffix.replace('_', ' ')}")))
    if "control" in families:
        figures.extend(build_control_comparisons(data, control=control, lum_scale=lum_scale,
            uncertainty=uncertainty, title=f"{title} — comparaison ciblée"))
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
    axes[0].set(title="Croissance brute", xlabel="Temps (h)", ylabel="DO brute")
    axes[1].set(title="Luminescence brute", xlabel="Temps (h)", ylabel="Luminescence brute (RLU)")
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
        title = f"{medium} · {strain} · réplicat biologique {biological_replicate}"
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for header, curve in replicate.groupby("sample_header", sort=False):
            curve = curve.sort_values("temps_h", kind="stable")
            label = str(header)
            axes[0].plot(curve["temps_h"], curve["DO_brute"], lw=1.4, label=label)
            axes[1].plot(curve["temps_h"], curve["Lum_brute"], lw=1.4, label=label)
        axes[0].set(title="DO", xlabel="Temps (h)", ylabel="DO brute")
        axes[1].set(title="Luminescence", xlabel="Temps (h)", ylabel="Luminescence brute (RLU)")
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
    axes[0].set(title="DO : brute (gris) / corrigée", xlabel="Temps (h)", ylabel="DO")
    axes[1].set(title="Luminescence : brute (gris) / corrigée", xlabel="Temps (h)", ylabel="Luminescence (RLU)")
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
    axes[0].set(title="Luminescence corrigée", xlabel="Temps (h)", ylabel="Luminescence (RLU)")
    axes[1].set(title="Luminescence normalisée par la DO", xlabel="Temps (h)", ylabel="Lum_corr / DO_corr")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def plot_kinetics(data: pd.DataFrame, series_metrics: pd.DataFrame):
    """Plot each well and annotate OD/normalized-luminescence kinetic landmarks."""
    required = {"temps_h", "sample_header", "DO_corr", "Lum_norm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    work = data.loc[data["type"].eq("souche")].copy() if "type" in data else data.copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    group_columns = [column for column in
                     ("experience_id", "souche", "Groupe", "sample_header", "puits", "replicat")
                     if column in work]
    color_by_strain = {
        strain: PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)]
        for index, strain in enumerate(dict.fromkeys(work["souche"].astype(str)))
    }
    labelled_strains: set[str] = set()
    for keys, curve in work.groupby(group_columns, dropna=False, sort=False):
        curve = curve.sort_values("temps_h", kind="stable")
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        strain = str(identity.get("souche", "Série"))
        color = color_by_strain[strain]
        label = strain if strain not in labelled_strains else "_nolegend_"
        axes[0].plot(curve["temps_h"], curve["DO_corr"], color=color, alpha=.45,
                     marker="o", markersize=2, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_norm"], color=color, alpha=.45,
                     marker="o", markersize=2, label=label)
        labelled_strains.add(strain)
        selected = series_metrics
        for column, value in identity.items():
            selected = selected.loc[selected[column].eq(value)]
        if selected.empty:
            continue
        metric = selected.iloc[0]
        axes[0].scatter(metric["od_max_time_h"], metric["od_max"], color=color,
                        marker="*", s=70, zorder=5)
        axes[1].scatter(metric["lum_norm_peak_time_h"], metric["lum_norm_peak"], color=color,
                        marker="*", s=70, zorder=5)
        if pd.notna(metric["max_growth_rate_start_h"]):
            axes[0].axvspan(metric["max_growth_rate_start_h"], metric["max_growth_rate_end_h"],
                            alpha=0.10, color="green")
    axes[0].set(title="Croissance et fenêtre maximale", xlabel="Temps (h)", ylabel="DO corrigée")
    axes[1].set(title="Luminescence et pic", xlabel="Temps (h)", ylabel="Luminescence normalisée")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(title="Souche", fontsize="small", frameon=False, loc="best")
    return figure

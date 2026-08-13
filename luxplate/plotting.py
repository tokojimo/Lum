"""Matplotlib scientific figures used by the application."""

from __future__ import annotations

import re
import textwrap
from colorsys import hls_to_rgb
from hashlib import sha256
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from luxplate.kinetics import run_kinetics
from luxplate.statistics import paired_directional_t_tests


PUBLICATION_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")

# The reporters used routinely in the laboratory keep their visual identity in
# every figure, independently of CSV row order or newly added strains.
REPORTER_COLORS = {
    "p0": "#0072B2",
    "psped": "#D55E00",
    "psped2-1a": "#009E73",
    "psped2-3b": "#CC79A7",
    "pspee": "#E69F00",
}

REPORTER_NAMES = {
    "psped2-1a": "PspeD2-1A",
    "psped2-3b": "PspeD2-3B",
    "psped": "PspeD",
    "pspee": "PspeE",
    "p0": "P0",
}


def _reporter_key(value: object) -> str | None:
    """Find a reporter even in construct names such as ``attB::PspeE-lux``."""
    normalized = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    # Longest first: PspeD must not capture either PspeD2 reporter.
    return next((key for key in REPORTER_NAMES if re.sub(r"[^a-z0-9]", "", key) in normalized), None)


def _publication_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.tick_params(width=0.8, length=3)


def _scientific_tick_label(value, _position):
    """Format an RLU tick without capturing unpicklable local state."""
    return "0" if value == 0 else f"{value:.0e}"


def _scientific_rlu_axis(axis):
    """Put the exponent on every RLU tick instead of above the plotting area."""
    axis.yaxis.set_major_formatter(FuncFormatter(_scientific_tick_label))
    axis.yaxis.offsetText.set_visible(False)


def _series_label(row: pd.Series) -> str:
    return str(row.get("souche", row.get("sample_header", "Series")))


def _display_strain(value: object) -> str:
    """Use compact reporter names on axes while retaining full names in the data."""
    key = _reporter_key(value)
    return REPORTER_NAMES[key] if key else str(value)


def _display_panel(value: object) -> str:
    """Turn machine-oriented group identifiers into publication labels."""
    label = str(value).strip()
    match = re.fullmatch(r"exp(?:eriment)?\s*(\d+)\s*\|\s*(.+)", label,
                         flags=re.IGNORECASE)
    if match:
        return f"Experiment {match.group(1)} – {match.group(2).strip()}"
    return label.replace("|", " – ")


def _wrapped_label(value: object, *, width: int = 32) -> str:
    """Wrap long display labels at words so neighbouring panels stay distinct.

    Matplotlib does not wrap axes titles to the physical width of an axes.  A
    long culture-medium name consequently runs through the titles of adjacent
    panels.  Explicit newlines are deterministic in screen, PNG, TIFF and SVG
    exports, unlike renderer-dependent automatic wrapping.
    """
    label = str(value)
    return "\n".join(textwrap.wrap(
        label, width=width, break_long_words=False, break_on_hyphens=False,
    )) or label


def _panel_title(value: object, *, width: int = 32) -> str:
    """Return a publication panel label constrained to a few readable lines."""
    return _wrapped_label(_display_panel(value), width=width)


def _medium_label(value: object) -> str:
    """Remove an experiment prefix so replicate panels can be pooled by medium."""
    label = str(value).strip()
    match = re.fullmatch(
        r"exp(?:eriment)?\s*\d+\s*\|\s*(.+)",
        label,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else label


def directional_condition_options(data: pd.DataFrame) -> dict[str, str]:
    """Return the compact UI label and internal id of each boxplot condition.

    Keeping individual conditions separate lets the interface ask for one
    reference and then its comparators instead of displaying the quadratic
    list of every possible ordered pair.
    """
    if not {"souche", "Groupe"}.issubset(data.columns):
        return {}
    conditions = list(dict.fromkeys(
        zip(data["souche"].astype(str), data["Groupe"].map(_medium_label))
    ))
    return {
        f"{_display_strain(strain)} · {medium}": strain + "\0" + medium
        for strain, medium in conditions
    }


def directional_comparison_options(data: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Return UI labels and ordered internal condition pairs for metric plots."""
    conditions = directional_condition_options(data)
    return {
        f"{left_label} > {right_label}": (left, right)
        for left_label, left in conditions.items()
        for right_label, right in conditions.items()
        if left != right
    }


def _has_multiple_experiments(data: pd.DataFrame) -> bool:
    if "experience_id" in data and data["experience_id"].dropna().astype(str).nunique() > 1:
        return True
    experiments = data["Groupe"].astype(str).str.extract(
        r"^\s*(exp(?:eriment)?\s*\d+)\s*\|", flags=re.IGNORECASE, expand=False,
    )
    return experiments.dropna().str.casefold().nunique() > 1


def _pooled_media(data: pd.DataFrame) -> pd.DataFrame:
    pooled = data.copy()
    # Preserve the biological experiment encoded in legacy group labels before
    # the mixed plot replaces ``Groupe`` with the pooled medium.  Otherwise the
    # only experiment identifier is lost and technical wells can become the SD
    # observations (or make an error bar incorrectly collapse to zero).
    if "experience_id" not in pooled or not pooled["experience_id"].notna().any():
        pooled["_experiment_from_group"] = pooled["Groupe"].astype(str).str.extract(
            r"^\s*(exp(?:eriment)?\s*\d+)\s*\|", flags=re.IGNORECASE,
            expand=False,
        ).str.casefold()
    pooled["Milieu"] = pooled["Groupe"].map(_medium_label)
    return pooled


def _shortest_experiment_end(data: pd.DataFrame) -> float | None:
    """Return the earliest final acquisition time across experiments.

    Comparing curves beyond the end of the shortest experiment can make panels
    visually misleading.  Keep the full data for summaries, but use this value
    as the common right-hand x limit of every time-course panel.
    """
    times = pd.to_numeric(data.get("temps_h"), errors="coerce")
    finite = times[np.isfinite(times)]
    if finite.empty:
        return None
    for column in ("experience_id", "experience"):
        if column in data and data[column].notna().any():
            ends = (pd.DataFrame({"experiment": data[column], "time": times})
                    .dropna().groupby("experiment", dropna=False)["time"].max())
            if not ends.empty:
                return float(ends.min())
    return float(finite.max())


def _set_common_time_end(axes, data: pd.DataFrame) -> None:
    """Show time courses from zero to the end of the shortest experiment.

    Matplotlib otherwise adds a horizontal margin before the first acquisition.
    Its automatic tick locator can then label that empty margin with a negative
    time even when every experiment starts at zero.
    """
    time_end = _shortest_experiment_end(data)
    if time_end is not None:
        for axis in axes:
            axis.set_xlim(left=0, right=time_end)


def _strain_colors(data: pd.DataFrame) -> dict[str, str]:
    """Assign stable reporter colors, including across data order and releases."""
    strains = list(dict.fromkeys(data["souche"].dropna().astype(str)))
    colors: dict[str, str] = {}
    for strain in strains:
        key = _reporter_key(strain)
        if key:
            colors[strain] = REPORTER_COLORS[key]
            continue
        # A digest rather than the position in the input makes an unfamiliar
        # strain retain its color when strains are added or rows are reordered.
        fallback_key = re.sub(r"(?:[-_ ]?lux)$", "", strain.strip(), flags=re.IGNORECASE).casefold()
        digest = sha256(fallback_key.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") / 65535
        saturation = .58 + digest[2] / 255 * .16
        lightness = .38 + digest[3] / 255 * .12
        red, green, blue = hls_to_rgb(hue, lightness, saturation)
        colors[strain] = f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"
    return colors


def _legend_layout(item_count: int, *, extra_items: int = 0) -> tuple[int, float]:
    """Return legend columns and a safe axes ceiling below a multi-row legend."""
    total = item_count + extra_items
    columns = min(5, max(1, total))
    rows = int(np.ceil(total / columns))
    # A legend row is relatively tall because it includes error-bar handles;
    # reserve enough room even for the larger corrected-luminescence font.
    return columns, max(.30, .60 - .11 * (rows - 1))


def _aligned_biological_summary(data: pd.DataFrame, value: str,
                                tolerance_h: float = 1 / 60) -> pd.DataFrame:
    """Align near-identical acquisition times before biological mean ± SD.

    Technical wells are first averaged inside an independent experiment/biological
    replicate.  Times separated by at most one minute are assigned to one observed
    timepoint, preventing a line from joining successive wells a few seconds apart.
    No interpolation or smoothing is performed.
    """
    work = data.loc[data[value].notna()].copy()
    observed = np.sort(pd.to_numeric(work["temps_h"], errors="coerce").dropna().unique())
    if not len(observed):
        return pd.DataFrame(columns=["temps_h", "mean", "std", "n_biological"])
    clusters: list[list[float]] = []
    for time in observed:
        if not clusters or time - clusters[-1][-1] > tolerance_h:
            clusters.append([float(time)])
        else:
            clusters[-1].append(float(time))
    lookup = {time: float(np.median(cluster)) for cluster in clusters for time in cluster}
    work["temps_aligne_h"] = pd.to_numeric(work["temps_h"]).map(lookup)
    # Some historical imports identify independent runs only in Groupe (for
    # example ``Experiment 2 | LB``).  Recover that identity before pooling;
    # otherwise the sample-header fallback would incorrectly treat technical
    # wells as independent biological observations and shrink the recap SD.
    # Build one explicit independent-unit key.  ``replicat`` and
    # ``sample_header`` identify technical series in historical imports and must
    # never increase biological N.  When no independent identity is available,
    # collapse the technical readings into one (SD unavailable) observation
    # rather than manufacture an SD from pseudoreplicates.
    group_experiment = work.get("_experiment_from_group")
    if group_experiment is None or not group_experiment.notna().any():
        group_experiment = (work["Groupe"].astype(str).str.extract(
            r"^\s*(exp(?:eriment)?\s*\d+)\s*\|", flags=re.IGNORECASE,
            expand=False,
        ).str.casefold() if "Groupe" in work else
            pd.Series(pd.NA, index=work.index, dtype="string"))
    experiment = (work["experience_id"].astype("string")
                  if "experience_id" in work else
                  pd.Series(pd.NA, index=work.index, dtype="string"))
    experiment = experiment.fillna(group_experiment.astype("string"))
    if "biological_replicate_id" in work and work["biological_replicate_id"].notna().any():
        biological_replicate = work["biological_replicate_id"].astype("string")
        work["_biological_unit"] = (
            experiment.fillna("unidentified-experiment") + "\0" + biological_replicate
        )
    else:
        work["_biological_unit"] = experiment.fillna("unidentified-biological-unit")
    biological = (work.groupby(["_biological_unit", "temps_aligne_h"], dropna=False)[value]
                   .mean().reset_index())

    # Independent runs commonly start a few minutes apart.  A strict join on
    # elapsed time would then put the corresponding measurements into separate
    # bins: each bin can contain only one experiment and its SD becomes NaN
    # (displayed as zero).  For a pooled recap, match the observed acquisition
    # sequence instead.  This does not interpolate values: point k from every
    # experiment is merely shown at the median observed time of point k.
    if biological["_biological_unit"].nunique(dropna=True) > 1:
        biological = biological.sort_values(["_biological_unit", "temps_aligne_h"])
        biological["_acquisition"] = biological.groupby(
            "_biological_unit", dropna=False
        )["temps_aligne_h"].rank(method="dense").astype(int)
        display_times = biological.groupby("_acquisition")["temps_aligne_h"].median()
        biological["temps_aligne_h"] = biological["_acquisition"].map(display_times)
    return (biological.groupby("temps_aligne_h")[value].agg(["mean", "std", "count"])
            .rename(columns={"count": "n_biological"}).reset_index()
            .rename(columns={"temps_aligne_h": "temps_h"}))


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
    panel_title_width = {1: 50, 2: 30, 3: 22}[ncols]
    header_columns, _ = _legend_layout(len(strain_colors))
    figure_width = max(4.1 * ncols, 2.15 * header_columns)
    # The extra vertical room accommodates wrapped medium titles without the
    # preceding row's x label touching the next row's heading.
    figure, axes = plt.subplots(nrows, ncols, figsize=(figure_width, 3.8 * nrows), squeeze=False)
    ylabel = {"DO_corr": r"OD$_{600}$",
              "Lum_corr": "Luminescence (RLU)",
              "Lum_norm": r"Normalized luminescence (RLU/OD$_{600}$)"}.get(value, value)
    for panel_index, panel in enumerate(panels):
        axis = axes.flat[panel_index]
        subset = work.loc[work[group_by].astype(str).eq(panel)]
        strains = list(dict.fromkeys(subset["souche"].astype(str)))
        for strain in strains:
            color = strain_colors[strain]
            strain_data = subset.loc[subset["souche"].astype(str).eq(strain)]
            summary = _aligned_biological_summary(strain_data, value)
            x = summary["temps_h"].to_numpy(float); y = summary["mean"].to_numpy(float)
            sd = summary["std"].fillna(0).to_numpy(float)
            linestyle = "--" if value in {"Lum_corr", "Lum_norm"} else "-"
            axis.errorbar(x, y, yerr=sd, color=color, lw=1.6, capsize=2,
                          linestyle=linestyle,
                          marker="o", markersize=2.5, label=strain)
        axis.set(title=_panel_title(panel, width=panel_title_width),
                 xlabel="Time (h)", ylabel=ylabel)
        axis.set_yscale(y_scale)
        if value == "Lum_corr" and y_scale == "linear":
            _scientific_rlu_axis(axis)
        axis.title.set_fontweight("bold"); axis.title.set_ha("left"); axis.title.set_position((0, 1.0))
        _publication_style(axis)
    for axis in axes.flat[len(panels):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        legend_columns, axes_top = _legend_layout(len(labels))
        legend = figure.legend(handles, [_display_strain(label) for label in labels], title="Reporter",
                      frameon=False, loc="upper center", fontsize=14 if value == "Lum_corr" else None,
                      bbox_to_anchor=(0.5, 0.92), ncol=legend_columns)
        if value == "Lum_corr":
            legend.get_title().set_fontsize(18)
    else:
        axes_top = .72
    figure.suptitle(title or f"{ylabel} over time", fontsize=13, fontweight="bold", y=0.995)
    # Identical limits prevent misleading visual comparisons between replicate panels.
    visible_axes = list(axes.flat[:len(panels)])
    if visible_axes:
        limits = np.asarray([axis.get_ylim() for axis in visible_axes], dtype=float)
        shared_limits = (float(np.nanmin(limits[:, 0])), float(np.nanmax(limits[:, 1])))
        for axis in visible_axes:
            axis.set_ylim(shared_limits)
        _set_common_time_end(visible_axes, work)
    figure.subplots_adjust(top=axes_top, bottom=0.12, hspace=0.82, wspace=0.34)
    return figure


def _mean_sd(data: pd.DataFrame, value: str) -> pd.DataFrame:
    return _aligned_biological_summary(data, value)


def plot_mixed_panels(data: pd.DataFrame, *, lum_scale: str = "linear",
                      uncertainty: str = "bars", title: str | None = None,
                      media: list[str] | None = None, strains: list[str] | None = None,
                      lum_value: str = "Lum_corr"):
    """Plot aligned OD and luminescence on genuine dual-y-axis panels."""
    if lum_value not in {"Lum_corr", "Lum_norm"}:
        raise ValueError("lum_value must be 'Lum_corr' or 'Lum_norm'.")
    required = {"temps_h", "souche", "Groupe", "DO_corr", lum_value}
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
    strains_in_order = list(dict.fromkeys(work["souche"].dropna().astype(str)))
    strain_colors = _strain_colors(work)
    ncols = min(3, len(panels)); blocks = int(np.ceil(len(panels) / ncols))
    panel_title_width = {1: 50, 2: 30, 3: 22}[ncols]
    header_columns, _ = _legend_layout(len(strains_in_order), extra_items=2)
    figure_width = max(4.5 * ncols, 2.15 * header_columns)
    figure, axes = plt.subplots(blocks, ncols, figsize=(figure_width, 4.2 * blocks), squeeze=False)
    legend_handles = []
    od_axes = []
    lum_axes = []
    for panel_index, medium in enumerate(panels):
        block, column = divmod(panel_index, ncols)
        top = axes[block, column]
        bottom = top.twinx()
        od_axes.append(top)
        lum_axes.append(bottom)
        subset = work.loc[work["Groupe"].astype(str).eq(medium)]
        for strain, strain_data in subset.groupby("souche", sort=False):
            color = strain_colors[str(strain)]
            od = _mean_sd(strain_data, "DO_corr"); lum = _mean_sd(strain_data, lum_value)
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
                           color=color, lw=1.6, linestyle="--", marker="s", markersize=2.3,
                           markerfacecolor="white")
            else:
                od_line = top.errorbar(ox, oy, yerr=osd, color=color, lw=1.35, marker="o",
                    markersize=2.5, markerfacecolor="white", capsize=2, label=str(strain)).lines[0]
                valid = ly > 0 if lum_scale == "log" else np.ones(len(ly), dtype=bool)
                bottom.errorbar(lx[valid], ly[valid], yerr=lsd[valid], color=color, lw=1.25,
                    linestyle="--",
                    marker="s", markersize=2.3, markerfacecolor="white", capsize=2)
            # Errorbar stores the public label on its container rather than on
            # the first Line2D; the figure-level legend consumes Line2D handles.
            od_line.set_label(str(strain))
            if panel_index == 0: legend_handles.append(od_line)
        top.set(title=_panel_title(medium, width=panel_title_width), ylabel=r"OD$_{600}$")
        top.set(xlabel="Time (h)")
        lum_ylabel = ("Luminescence (RLU)" if lum_value == "Lum_corr" else
                      r"Normalized luminescence (RLU/OD$_{600}$)")
        bottom.set_ylabel(lum_ylabel)
        bottom.set_yscale(lum_scale)
        if lum_value == "Lum_corr" and lum_scale == "linear":
            _scientific_rlu_axis(bottom)
        top.title.set_fontweight("bold"); top.title.set_ha("left"); top.title.set_position((0, 1))
        _publication_style(top); _publication_style(bottom)
    # A dual-axis figure is only comparable across media when each measurement
    # uses one common range.  Sharing cannot be requested directly here because
    # the luminescence axes are created with ``twinx`` after the OD grid.
    for measurement_axes in (od_axes, lum_axes):
        limits = np.asarray([axis.get_ylim() for axis in measurement_axes], dtype=float)
        shared_limits = (float(np.nanmin(limits[:, 0])), float(np.nanmax(limits[:, 1])))
        for axis in measurement_axes:
            axis.set_ylim(shared_limits)
    _set_common_time_end(od_axes, work)
    for panel_index in range(len(panels), blocks * ncols):
        block, column = divmod(panel_index, ncols)
        axes[block, column].remove()
    style_handles = [
        Line2D([], [], color="#333333", lw=1.8, linestyle="-"),
        Line2D([], [], color="#333333", lw=1.6, linestyle="--"),
    ]
    legend_columns, axes_top = _legend_layout(len(legend_handles), extra_items=2)
    figure.legend([*legend_handles, *style_handles],
                  [*[_display_strain(line.get_label()) for line in legend_handles],
                   r"OD$_{600}$", "Luminescence (RLU)"],
                  title="Promoter (color) · Measurement (line)",
                  frameon=False, loc="upper center", bbox_to_anchor=(.5, .92),
                  ncol=legend_columns)
    figure.suptitle(title or "Growth and luminescence", fontweight="bold", y=.995)
    figure.subplots_adjust(top=axes_top, bottom=.13, hspace=.78, wspace=.52)
    return figure


def _significance_stars(p_value: float) -> str:
    """Return the conventional significance symbol for an adjusted p-value."""
    # A missing/uncomputable p-value is not evidence of non-significance.  In
    # particular, labelling NaN as ``ns`` hid failed or impossible tests.
    if not np.isfinite(p_value):
        return "NA"
    if p_value < .0001:
        return "****"
    if p_value < .001:
        return "***"
    if p_value < .01:
        return "**"
    if p_value < .05:
        return "*"
    return "ns"


def _draw_metric_panel(axis, technical: pd.DataFrame, biological: pd.DataFrame, *,
                       metric: str, condition: str, y_scale: str, panel_title: str,
                       seed: int, directional_comparisons: tuple[tuple[str, str], ...] = (),
                       significant_only: bool = False) -> pd.DataFrame:
    """Draw one metric panel and return its pairwise statistics."""
    conditions = list(dict.fromkeys(biological[condition].astype(str)))
    if condition == "_comparison":
        reporter_colors = _strain_colors(pd.DataFrame({"souche": biological["_reporter"]}))
        colors = {item: reporter_colors[biological.loc[
            biological[condition].astype(str).eq(item), "_reporter"
        ].iloc[0]] for item in conditions}
    else:
        colors = _strain_colors(biological) if condition == "souche" else {
            item: PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)]
            for index, item in enumerate(conditions)
        }
    rng = np.random.default_rng(seed)
    summaries = biological.groupby(condition, sort=False)[metric].agg(["mean", "std"])
    for condition_index, item in enumerate(conditions):
        raw = technical.loc[technical[condition].astype(str).eq(item), metric].to_numpy(float)
        values = biological.loc[biological[condition].astype(str).eq(item), metric].to_numpy(float)
        # Fold changes retain the technical ratios specifically so their boxes
        # show the observed dispersion.  Biological means remain the only
        # values used for the large points and inferential statistics below.
        box_values = raw if metric.endswith("_fold_change") else values
        color = colors[item]
        axis.boxplot([box_values], positions=[condition_index], widths=.55, patch_artist=True,
            showfliers=False, medianprops={"color": color, "linewidth": 1.5},
            boxprops={"facecolor": color, "alpha": .20, "edgecolor": color},
            whiskerprops={"color": color}, capprops={"color": color})
        highest = float(np.nanmax(np.concatenate([values, raw])))
        axis.annotate(f"{float(summaries.loc[item, 'mean']):.3g}", (condition_index, highest),
                      xytext=(0, 7), ha="center", va="bottom", textcoords="offset points",
                      fontsize=7, color=color)
        axis.plot(condition_index + rng.uniform(-.16, .16, len(raw)), raw, linestyle="none",
                  marker="o", markersize=2.8, color=color, alpha=.32, zorder=2)

    identity_columns = [column for column in ("experience_id", "replicat")
                        if column in biological and biological[column].notna().any()]
    identity_columns = identity_columns or (["Groupe"] if condition == "souche" else ["souche"])
    for identity in biological[identity_columns].drop_duplicates().itertuples(index=False, name=None):
        mask = np.ones(len(biological), dtype=bool)
        for column, value in zip(identity_columns, identity):
            matches = biological[column].isna() if pd.isna(value) else biological[column].eq(value)
            mask &= matches.to_numpy()
        subset = biological.loc[mask]
        for condition_index, item in enumerate(conditions):
            values = subset.loc[subset[condition].astype(str).eq(item), metric].to_numpy(float)
            if len(values):
                axis.scatter(condition_index + rng.uniform(-.025, .025), values[0], s=48,
                    marker="o", color=colors[item], edgecolor="white", linewidth=.6, zorder=3)

    metric_labels = {"lum_norm_peak": r"Peak normalized luminescence (RLU/OD$_{600}$)",
        "lum_norm_peak_time_h": "Time of normalized luminescence peak (h)",
        "lum_norm_auc": r"Normalized luminescence AUC (RLU/OD$_{600}$)·h",
        "lum_norm_peak_fold_change": "Peak normalized luminescence (fold change vs P0)",
        "lum_norm_auc_fold_change": "Normalized luminescence AUC (fold change vs P0)",
        "doubling_time_h": "Doubling time (h)"}
    display_conditions = []
    for item in conditions:
        if condition == "souche":
            display_conditions.append(_display_strain(item))
        elif condition == "_comparison":
            row = biological.loc[biological[condition].astype(str).eq(item)].iloc[0]
            medium = _wrapped_label(row["_medium"], width=18)
            display_conditions.append(f"{_display_strain(row['_reporter'])}\n{medium}")
        else:
            display_conditions.append(_panel_title(item))
    axis.set_xticks(range(len(conditions)), display_conditions)
    axis.set(xlabel="Reporter · medium" if condition == "_comparison" else
             ("Reporter" if condition == "souche" else "Medium"),
             ylabel=metric_labels.get(metric, metric), title=panel_title)
    axis.set_yscale(y_scale)
    # Combined reporter/medium figures need a genuine annotation band above the
    # boxes.  The extra margin keeps even a dense complete pairwise test matrix
    # from being printed over the data.
    axis.margins(y=.65 if condition == "_comparison" else .30)
    if condition == "_comparison":
        finite = biological[metric].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        if len(finite) and y_scale == "linear":
            low, high = float(finite.min()), float(finite.max())
            span = max(high - low, abs(high) * .05, 1e-9)
            axis.set_ylim(low - .12 * span, high + 1.20 * span)
        elif len(finite):
            low, high = float(finite.min()), float(finite.max())
            log_span = max(np.log(high) - np.log(low), .1)
            axis.set_ylim(np.exp(np.log(low) - .12 * log_span),
                          np.exp(np.log(high) + 1.20 * log_span))
    axis.title.set_fontweight("bold"); axis.title.set_ha("left"); axis.title.set_position((0, 1))
    _publication_style(axis)

    comparisons = paired_directional_t_tests(
        biological, value=metric, condition=condition,
        identity=tuple(identity_columns),
        comparisons=directional_comparisons,
    )
    positions = {item: index for index, item in enumerate(conditions)}
    usable = comparisons.loc[comparisons["condition_1"].isin(positions)
                             & comparisons["condition_2"].isin(positions)]
    if significant_only:
        usable = usable.loc[usable["p_holm"] < .05]
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    spacing = min(.055, .36 / max(1, len(usable)))
    for level, comparison in enumerate(usable.itertuples(index=False), start=1):
        left, right = positions[comparison.condition_1], positions[comparison.condition_2]
        y = .60 + spacing * level
        p_label = _significance_stars(comparison.p_holm)
        axis.plot([left, left, right, right], [y - .012, y, y, y - .012],
                  transform=transform, color="#333333", lw=.7, clip_on=False)
        axis.text((left + right) / 2, y + .006, p_label, transform=transform,
                  ha="center", va="bottom", fontsize=7)
    return comparisons


def plot_metric_points(metrics: pd.DataFrame, *, metric: str, y_scale: str = "linear",
                       title: str | None = None, group_by: str | None = None,
                       compare_media: bool = False,
                       directional_comparisons: tuple[tuple[str, str], ...] = (),
                       significant_only: bool = False):
    """Plot metric distributions, optionally with one panel per medium or strain."""
    required = {"souche", "Groupe", metric}
    if group_by is not None:
        required.add(group_by)
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing columns for metric: {sorted(missing)}")
    if y_scale not in {"linear", "log"}:
        raise ValueError("Scale must be 'linear' or 'log'.")
    numeric = pd.to_numeric(metrics[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    technical = metrics.loc[numeric.notna()].copy()
    technical[metric] = numeric.loc[technical.index]
    effective_scale = y_scale
    if y_scale == "log":
        if technical[metric].gt(0).any():
            technical = technical.loc[technical[metric].gt(0)]
        else:
            effective_scale = "linear"

    biological_ids = [column for column in ("experience_id", "replicat")
                      if column in technical and technical[column].notna().any()]
    group_columns = ["souche", "Groupe", *biological_ids]
    biological = technical.groupby(group_columns, dropna=False, sort=False)[metric].mean().reset_index()
    panels = ([None] if group_by is None else
              list(dict.fromkeys(biological[group_by].dropna().astype(str))))
    if not panels:
        raise ValueError("No usable data available for the metric figure.")
    if compare_media:
        biological["_reporter"] = biological["souche"].astype(str)
        biological["_medium"] = biological["Groupe"].map(_medium_label)
        biological["_comparison"] = (biological["_reporter"] + "\0" + biological["_medium"])
        technical["_reporter"] = technical["souche"].astype(str)
        technical["_medium"] = technical["Groupe"].map(_medium_label)
        technical["_comparison"] = technical["_reporter"] + "\0" + technical["_medium"]
        condition = "_comparison"
        panels = [None]
    else:
        condition = "Groupe" if group_by == "souche" else "souche"
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels) / ncols))
    condition_count = biological[condition].nunique()
    comparison_count = len(directional_comparisons)
    extra_height = min(8, .18 * comparison_count) if compare_media else 0
    width = max(6, 1.35 * condition_count) if compare_media else max(6, 4.3 * ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(width, (4.5 + extra_height) * nrows), squeeze=False)
    all_statistics = []
    for index, panel in enumerate(panels):
        panel_technical = technical if panel is None else technical.loc[technical[group_by].astype(str).eq(panel)]
        panel_biological = biological if panel is None else biological.loc[biological[group_by].astype(str).eq(panel)]
        panel_title = (title or metric) if panel is None else _display_panel(panel)
        statistics = _draw_metric_panel(axes.flat[index], panel_technical, panel_biological,
            metric=metric, condition=condition, y_scale=effective_scale,
            panel_title=panel_title, seed=1947 + index,
            directional_comparisons=directional_comparisons,
            significant_only=significant_only)
        if panel is not None:
            statistics = statistics.assign(panel=panel)
        all_statistics.append(statistics)
    for axis in axes.flat[len(panels):]:
        axis.remove()
    axes.flat[0].legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=4, alpha=.35,
               color="#555555", label="Technical replicate"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               color="#555555", markeredgecolor="white", label="Biological mean"),
    ], title="Replicate display", frameon=False, fontsize=7, loc="center left",
       bbox_to_anchor=(1.01, .5))
    if group_by is not None:
        figure.suptitle(title or metric, fontweight="bold", y=.995)
    figure._luxplate_statistics = pd.concat(all_statistics, ignore_index=True) if all_statistics else pd.DataFrame()
    figure.tight_layout(rect=(0, 0, 1, .96) if group_by is not None and not compare_media else None)
    return figure


def metric_fold_change_vs_control(metrics: pd.DataFrame, *, metric: str,
                                  control: str = "P0-lux") -> pd.DataFrame:
    """Return biological metric means divided by the matched control per medium.

    Ratios are calculated only between observations from the same independent
    experiment/replicate and medium.  Technical series are averaged before the
    division, so adding technical wells cannot inflate the biological N.
    """
    required = {"souche", "Groupe", metric}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing columns for fold change: {sorted(missing)}")
    work = metrics.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.loc[work[metric].notna()]
    work["Milieu"] = work["Groupe"].map(_medium_label)
    identities = [column for column in ("experience_id", "replicat")
                  if column in work and work[column].notna().any()]
    grouping = ["souche", "Milieu", *identities]
    biological = work.groupby(grouping, dropna=False, sort=False)[metric].mean().reset_index()
    exact_control = biological["souche"].astype(str).str.casefold().eq(control.casefold())
    reporter_control = biological["souche"].map(_reporter_key).eq("p0")
    control_rows = biological.loc[exact_control if exact_control.any() else reporter_control]
    if control_rows.empty:
        raise ValueError(f"Control {control!r} is absent from the kinetic metrics.")
    denominator_keys = ["Milieu", *identities]
    denominators = (control_rows.groupby(denominator_keys, dropna=False, sort=False)[metric]
                    .mean().rename("_control_value").reset_index())
    fold_metric = f"{metric}_fold_change"
    # Return the technical observations rather than the already collapsed
    # biological table.  Each value is still divided by its matched biological
    # P0 mean, but retaining these rows lets the figure display the experimental
    # dispersion while ``plot_metric_points`` independently reconstructs the
    # biological means used by the statistical tests.
    result = work.merge(denominators, on=denominator_keys, how="inner", validate="many_to_one")
    valid_denominator = result["_control_value"].ne(0) & result["_control_value"].notna()
    result[fold_metric] = result[metric] / result["_control_value"].where(valid_denominator)
    result["Groupe"] = result["Milieu"]
    return result.drop(columns=["_control_value"])

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
    families: tuple[str, ...] = ("growth", "corrected", "mixed", "peak",
                                "peak_time", "auc", "peak_fc", "auc_fc", "doubling"),
    panel_by: str = "Groupe", lum_scale: str = "linear", normalized_scale: str = "linear",
    metric_scale: str = "log", uncertainty: str = "bars", control: str = "P0-lux",
    directional_comparisons: tuple[tuple[str, str], ...] = (),
    significant_only: bool = False) -> list[tuple[str, object]]:
    """Build the curve families represented in the historical example scripts."""
    figures = []
    choices = (("growth", "DO_corr", "croissance", "linear", "Growth"),
               ("corrected", "Lum_corr", "luminescence_corrigee", lum_scale,
                "Luminescence"),
               ("normalized", "Lum_norm", "luminescence_normalisee", normalized_scale,
                "Normalized luminescence"))
    for family, value, suffix, scale, figure_label in choices:
        if family in families and value in data and data[value].notna().any():
            figures.append((suffix, plot_publication_panels(data, value=value, group_by=panel_by,
                y_scale=scale, title=figure_label)))
            if family in {"growth", "corrected"} and _has_multiple_experiments(data):
                pooled = _pooled_media(data)
                recap_group = "Milieu" if panel_by == "Groupe" else "souche"
                figures.append((f"{suffix}_moyenne_experiences", plot_publication_panels(
                    pooled, value=value, group_by=recap_group, y_scale=scale,
                    title=f"{figure_label} — mean of experiments")))
    if "mixed" in families:
        mixed_data = _pooled_media(data) if _has_multiple_experiments(data) else data
        if "Milieu" in mixed_data:
            mixed_data = mixed_data.copy()
            mixed_data["Groupe"] = mixed_data["Milieu"]
        figures.append(("croissance_luminescence_mixte", plot_mixed_panels(
            mixed_data, lum_scale=lum_scale, uncertainty=uncertainty,
            title="Growth and non-normalized luminescence", lum_value="Lum_corr")))
    metric_families = {"peak": ("lum_norm_peak", "pic_luminescence_normalisee", "Peak normalized luminescence"),
                       "peak_time": ("lum_norm_peak_time_h", "temps_pic_luminescence_normalisee",
                                     "Time of normalized luminescence peak"),
                       "auc": ("lum_norm_auc", "auc_luminescence_normalisee", "Normalized luminescence AUC"),
                       "peak_fc": ("lum_norm_peak", "pic_luminescence_normalisee_fold_change_P0",
                                   "Peak normalized luminescence — fold change vs P0"),
                       "auc_fc": ("lum_norm_auc", "auc_luminescence_normalisee_fold_change_P0",
                                  "Normalized luminescence AUC — fold change vs P0"),
                       "doubling": ("doubling_time_h", "temps_doublement", "Doubling time")}
    requested = set(families).intersection(metric_families)
    if requested:
        metrics = run_kinetics(data).series_metrics
        for family in ("peak", "peak_time", "auc", "peak_fc", "auc_fc", "doubling"):
            if family in requested:
                metric, suffix, figure_label = metric_families[family]
                if family.endswith("_fc"):
                    fold_changes = metric_fold_change_vs_control(metrics, metric=metric, control="P0-lux")
                    fold_metric = f"{metric}_fold_change"
                    figures.append((suffix, plot_metric_points(
                        fold_changes, metric=fold_metric, y_scale=metric_scale,
                        title=figure_label, compare_media=True,
                        directional_comparisons=directional_comparisons,
                        significant_only=significant_only)))
                else:
                    scale = "linear" if family in {"doubling", "peak_time"} else metric_scale
                    figures.append((suffix, plot_metric_points(metrics, metric=metric, y_scale=scale,
                        title=figure_label, compare_media=True,
                        directional_comparisons=directional_comparisons,
                        significant_only=significant_only)))
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
    """Build one static DO/luminescence figure per strain and medium.

    A workbook is one independent biological replicate; ``replicat`` and the
    sample headers inside that workbook identify technical replicates.  Each
    biological replicate therefore occupies one row and all of its technical
    curves are overlaid in that row.  Axis limits are shared across rows so the
    biological replicates can be compared without a misleading change of scale.
    """
    required = {"temps_h", "souche", "Groupe", "replicat", "sample_header", "type",
                "DO_brute", "Lum_brute"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    work = data.loc[data["type"].astype(str).str.lower().eq(sample_type.lower())].copy()
    work["_milieu"] = work["Groupe"].map(_medium_label)
    if "experience_id" in work and work["experience_id"].notna().any():
        work["_experience_biologique"] = work["experience_id"].astype(str)
    elif "experience" in work and work["experience"].notna().any():
        work["_experience_biologique"] = work["experience"].astype(str)
    else:
        # A table parsed from a single workbook has no experiment column yet.
        work["_experience_biologique"] = "Expérience 1"
    # Keep media as the outer display level.  Input tables are commonly
    # concatenated workbook by workbook; relying on their row order would show
    # every medium from biological replicate 1 before biological replicate 2.
    # Stable ranks instead produce M1/bio1, M1/bio2, M2/bio1, M2/bio2.
    medium_order = {value: index for index, value in enumerate(work["_milieu"].drop_duplicates())}
    strain_order = {value: index for index, value in enumerate(work["souche"].drop_duplicates())}
    work["_ordre_milieu"] = work["_milieu"].map(medium_order)
    work["_ordre_souche"] = work["souche"].map(strain_order)
    work = work.sort_values(["_ordre_milieu", "_ordre_souche"], kind="stable")
    figures: list[tuple[str, object]] = []
    for (medium, strain), condition in work.groupby(["_milieu", "souche"], dropna=False, sort=False):
        biological = list(condition.groupby("_experience_biologique", dropna=False, sort=False))
        title = f"{medium} · {strain}"
        figure, axes = plt.subplots(
            len(biological), 2, figsize=(10, 3.3 * len(biological)),
            constrained_layout=True, squeeze=False, sharex=True, sharey="col",
        )
        for row, (experience, replicate) in enumerate(biological):
            for technical_number, (header, curve) in enumerate(
                replicate.groupby("sample_header", sort=False), start=1
            ):
                curve = curve.sort_values("temps_h", kind="stable")
                well = curve["puits"].iloc[0] if "puits" in curve else pd.NA
                if pd.notna(well) and str(well).strip():
                    label = f"Rép. technique ({str(well).strip()})"
                else:
                    technical = curve["replicat"].iloc[0]
                    number = technical if pd.notna(technical) else technical_number
                    label = f"Rép. technique {number}"
                axes[row, 0].plot(curve["temps_h"], curve["DO_brute"], lw=1.4, label=label)
                axes[row, 1].plot(curve["temps_h"], curve["Lum_brute"], lw=1.4, label=label)
            axes[row, 0].set_ylabel(f"{experience}\n" + r"OD$_{600}$ brute")
            axes[row, 1].set_ylabel(f"{experience}\nLuminescence brute (RLU)")
            for axis in axes[row]:
                _publication_style(axis)
                axis.legend(fontsize="x-small", frameon=False, loc="best")
        axes[0, 0].set_title("Densité optique")
        axes[0, 1].set_title("Luminescence")
        axes[-1, 0].set_xlabel("Temps (h)")
        axes[-1, 1].set_xlabel("Temps (h)")
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

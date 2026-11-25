#!/usr/bin/env python3
"""Readable, data-preserving redraws of supplementary figures S1 and S5–S7.

The renderer uses only the published panel tables under data/summary and does
not refit models, recompute statistics, or edit the source package.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

try:
    from .layout_style import configure_style, style_figure
except ImportError:
    from layout_style import configure_style, style_figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


WIDTH_MM = 180.0
MAX_HEIGHT_MM = 198.0
BLUE = "#356C8C"
ORANGE = "#C87935"
GREEN = "#4D876F"
PURPLE = "#786A91"
GREY = "#666666"
LIGHT = "#D5DEE3"
PROFILE_COLORS = {"with_met": BLUE, "no_met": ORANGE}
TASK_COLORS = {"T2_spatial": BLUE, "T2_spatial_buffer50km": ORANGE}
FOLD_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "*"]

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
configure_style()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    candidates = (root, root / "input" / "github", root / "github")
    for candidate in candidates:
        if (candidate / "data" / "summary" / "figure_registry.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate data/summary/figure_registry.json below {root}"
    )


def _safe_panel_path(root: Path, relative: str) -> Path:
    panels_root = (root / "data" / "summary" / "figure_panels").resolve()
    path = (root / relative).resolve()
    if panels_root not in path.parents:
        raise ValueError(f"Registry panel is outside figure_panels: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_panel(root: Path, entry: dict[str, Any], key: str) -> tuple[pd.DataFrame, Path]:
    path = _safe_panel_path(root, entry["panels"][key])
    frame = pd.read_csv(path, low_memory=False, compression="infer")
    expected = int(entry.get("panel_rows", {}).get(key, len(frame)))
    if len(frame) != expected:
        raise ValueError(
            f"{entry['figure']} panel {key} has {len(frame)} rows; registry records {expected}"
        )
    return frame, path


def _figure_entry(registry: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in registry.get("figures", []) if item.get("figure") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registry entry for {name}, got {len(matches)}")
    return matches[0]


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _clean(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def _short_profile(value: Any) -> str:
    return {"with_met": "with MET", "no_met": "no MET"}.get(str(value), _clean(value))


def _alignment(value: Any) -> str:
    return {
        "same_named_date": "same date labels",
        "same_named_date_EPA_local_HMS_UTC": "same date labels",
        "plusminus1day_window_expansion": "±1 day",
        "nearest_date": "nearest date",
    }.get(str(value), str(value).replace("_", " "))


def _plume(value: Any) -> str:
    return {
        "any_plume": "any plume",
        "medium_or_heavy": "medium/heavy plume",
    }.get(str(value), str(value).replace("_", " "))


def _saved_interval(ax: Any, frame: pd.DataFrame, *, value_col: str,
                    low_col: str | None, high_col: str | None,
                    labels: list[str], colors: list[str] | None = None,
                    markers: list[str] | None = None, xlim: tuple[float, float] | None = None,
                    zero_line: bool = False) -> int:
    """Plot every saved estimate/interval; return the number of finite estimates."""
    n_finite = 0
    colors = colors or [BLUE] * len(frame)
    markers = markers or ["o"] * len(frame)
    for i, (_, row) in enumerate(frame.iterrows()):
        value = _float(row.get(value_col))
        low = _float(row.get(low_col)) if low_col else math.nan
        high = _float(row.get(high_col)) if high_col else math.nan
        color = colors[i]
        marker = markers[i]
        if math.isfinite(low) and math.isfinite(high):
            ax.plot([low, high], [i, i], color=color, lw=0.9, zorder=1)
        if math.isfinite(value):
            ax.scatter([value], [i], color=color, edgecolor="white", linewidth=0.45,
                       marker=marker, s=24, zorder=3)
            n_finite += 1
        else:
            ax.text(0.01, i, "NA", transform=ax.get_yaxis_transform(), color=GREY,
                    va="center", fontsize=8)
    ax.set_yticks(np.arange(len(frame)), labels, fontsize=8)
    ax.invert_yaxis()
    if zero_line:
        ax.axvline(0, color=GREY, lw=0.7, zorder=0)
    ax.grid(axis="x", color=LIGHT, lw=0.45)
    if xlim:
        ax.set_xlim(*xlim)
    return n_finite


def _save_figure(fig: Any, outdir: Path, stem: str, height_mm: float) -> dict[str, Any]:
    width_in, height_in = WIDTH_MM / 25.4, height_mm / 25.4
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"{stem} height {height_mm:.1f} mm exceeds {MAX_HEIGHT_MM:.0f} mm")
    fig.set_size_inches(width_in, height_in, forward=True)
    style_figure(fig)
    outputs: dict[str, Any] = {"dimensions_mm": {"width": WIDTH_MM, "height": height_mm}}
    fig.savefig(outdir / f"{stem}.pdf", format="pdf", bbox_inches=None)
    fig.savefig(outdir / f"{stem}.svg", format="svg", bbox_inches=None)
    fig.savefig(outdir / f"{stem}.png", format="png", bbox_inches=None, dpi=300)
    fig.savefig(outdir / f"{stem}.tiff", format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    for extension in ("pdf", "svg", "png", "tiff"):
        path = outdir / f"{stem}.{extension}"
        outputs[extension] = {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
    try:
        from .layout_style import write_layout_qa
    except ImportError:
        from layout_style import write_layout_qa
    write_layout_qa(fig, outdir, stem)
    plt.close(fig)
    return outputs


def _add_panel_header(axis: Any, label: str, title: str, *, title_x: float = 0.15,
                      y: float = 1.08) -> None:
    """Keep the bold panel letter separate from its compact descriptive title."""
    axis.text(0.0, y, label, transform=axis.transAxes, ha="left", va="bottom",
              fontsize=18, fontfamily="Arial", fontweight="bold", clip_on=False)
    axis.text(title_x, y, title, transform=axis.transAxes, ha="left", va="bottom",
              fontsize=10, fontfamily="Arial", fontweight="bold", clip_on=False)


def _s5_label(row: pd.Series, *, include_country: bool = False) -> str:
    parts: list[str] = []
    if include_country and "country_label" in row:
        parts.append(_clean(row["country_label"]))
    if "year" in row:
        parts.append(_clean(row["year"]))
    task = str(row.get("task", ""))
    if task:
        mode = "rolling" if "rolling" in task.lower() else "fixed" if "fixed" in task.lower() else task.replace("_", " ")
        if mode not in parts:
            parts.append(mode)
    if "profile" in row:
        parts.append(_short_profile(row["profile"]))
    return " · ".join(parts) if parts else "saved row"


def _draw_s5(frames: dict[str, pd.DataFrame], outdir: Path) -> tuple[dict[str, Any], str]:
    a, b, c, d = (frames[k].copy() for k in ("a", "b", "c", "d"))
    if len(a) != 16 or set(a.get("country", pd.Series(dtype=str)).astype(str)) != {"USA", "CAN"}:
        raise ValueError("S5 panel a must retain all 16 USA/CAN records")
    if len(b) != 8 or len(c) != 6 or len(d) != 12:
        raise ValueError("S5 panel b/c/d must retain 8/6/12 source rows")
    a_by_country = {country: a[a.country.astype(str) == country].copy() for country in ("USA", "CAN")}
    if any(len(a_by_country[country]) == 0 for country in a_by_country):
        raise ValueError("S5 country facets must both contain saved HMS records")

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 198 / 25.4))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.45), hspace=0.58, wspace=0.42)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :])]
    profile_handles = [
        Line2D([], [], color=PROFILE_COLORS[p], marker="o", lw=0, markersize=5,
               label=_short_profile(p)) for p in ("with_met", "no_met")
    ]
    mode_handles = [
        Line2D([], [], color=GREY, marker=mark, markerfacecolor=face,
               markeredgecolor=GREY, lw=0, markersize=5, label=label)
        for mark, face, label in (("o", GREY, "fixed"), ("D", "white", "rolling"))
    ]
    fig.legend(handles=profile_handles + mode_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.99), fontsize=8, handletextpad=0.4, columnspacing=1.0)

    finite_counts: dict[str, int] = {}
    group_label_sets: dict[str, list[str]] = {}

    def grouped_profile_plot(axis: Any, frame: pd.DataFrame, group_cols: list[str],
                             label_fn: Any, title: str, *,
                             value_col: str = "pearson_r", low_col: str = "ci_low",
                             high_col: str = "ci_high", zero_line: bool = False) -> tuple[int, list[str]]:
        group_keys = list(dict.fromkeys(tuple(_clean(row[column]) for column in group_cols)
                                        for _, row in frame.iterrows()))
        labels: list[str] = []
        plotted = 0
        for group_index, key in enumerate(group_keys):
            mask = np.ones(len(frame), dtype=bool)
            for column, value in zip(group_cols, key):
                mask &= frame[column].fillna("NA").astype(str).to_numpy() == value
            subset = frame.loc[mask]
            labels.append(label_fn(subset.iloc[0]))
            profiles = [profile for profile in ("with_met", "no_met")
                        if profile in set(subset.get("profile", pd.Series(dtype=str)).astype(str))]
            profiles += [str(profile) for profile in subset.get("profile", pd.Series(dtype=str)).dropna().unique()
                         if str(profile) not in profiles]
            for profile_index, profile in enumerate(profiles):
                profile_rows = subset[subset.profile.astype(str) == profile] if "profile" in subset else subset
                profile_offset = (profile_index - (len(profiles) - 1) / 2) * 0.22
                for replicate_index, (_, row) in enumerate(profile_rows.iterrows()):
                    y = group_index + profile_offset + (replicate_index - (len(profile_rows) - 1) / 2) * 0.035
                    value, low, high = (_float(row.get(column)) for column in (value_col, low_col, high_col))
                    color = PROFILE_COLORS.get(profile, PURPLE)
                    task = str(row.get("task", ""))
                    marker = "D" if "rolling" in task.lower() else "o"
                    if math.isfinite(low) and math.isfinite(high):
                        axis.plot([low, high], [y, y], color=color, lw=0.9, zorder=1)
                        axis.vlines([low, high], y-0.055, y+0.055, color=color, lw=0.8, zorder=4)
                    if math.isfinite(value):
                        axis.scatter([value], [y], color='white', marker=marker, edgecolor=color,
                                     linewidth=0.7, s=9, zorder=3)
                        if math.isfinite(low) and math.isfinite(high):
                            axis.plot([low, high], [y, y], color=color, lw=0.9, zorder=4)
                        plotted += 1
                    else:
                        axis.text(0.01, y, "NA", transform=axis.get_yaxis_transform(),
                                  color=GREY, va="center", fontsize=8)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        axis.invert_yaxis()
        axis.grid(axis="x", color=LIGHT, lw=0.45)
        axis.set_xlim(-1, 1)
        if zero_line:
            axis.axvline(0, color=GREY, lw=0.7, zorder=0)
        axis.set_xlabel("Pearson r")
        panel_label, _, panel_title = title.partition("  ")
        _add_panel_header(axis, panel_label.upper(), panel_title, title_x=0.15)
        return plotted, labels

    def year_mode_label(row: pd.Series) -> str:
        year = _clean(row.get("year", ""))
        task = str(row.get("task", ""))
        mode = "rolling" if "rolling" in task.lower() else "fixed" if "fixed" in task.lower() else task.replace("_", " ")
        return f"{year} {mode}".strip()

    finite_counts["a"], group_label_sets["a"] = grouped_profile_plot(
        axes[0], a_by_country["USA"], ["year", "task"], year_mode_label, "a  USA HMS")
    finite_counts["b"], group_label_sets["b"] = grouped_profile_plot(
        axes[1], a_by_country["CAN"], ["year", "task"], year_mode_label, "b  CAN HMS")
    if {"scope", "comparator"}.issubset(b.columns):
        b["_country"] = b.scope.astype(str).str.split(":").str[1].fillna("")
    else:
        b["_country"] = b.get("country_label", pd.Series(["NA"] * len(b)))
    product_label = lambda row: f"{_clean(row.get('_country', 'NA'))} · {_clean(row.get('comparator', 'product'))}"
    finite_counts["c"], group_label_sets["c"] = grouped_profile_plot(
        axes[2], b, ["_country", "comparator"], product_label, "c  Fire-product agreement",
        low_col="pearson_r_ci_low", high_col="pearson_r_ci_high")
    method_label = lambda row: f"{_clean(row.get('parameter_code', 'NA'))}/{_clean(row.get('method_code', 'NA'))}"
    finite_counts["d"], group_label_sets["d"] = grouped_profile_plot(
        axes[3], c, ["parameter_code", "method_code"], method_label, "d  OC tracer",
        low_col="pearson_r_ci_low", high_col="pearson_r_ci_high")

    e_labels = []
    for _, row in d.iterrows():
        code = f"{_clean(row.get('parameter_code', 'NA'))}/{_clean(row.get('method_code', 'NA'))}"
        alignment = "same" if "same" in str(row.get("alignment", "")) else "±1d" if "plusminus1day" in str(row.get("alignment", "")) else "other"
        plume = "any" if "any_plume" in str(row.get("plume_definition", "")) else "M/H" if "medium_or_heavy" in str(row.get("plume_definition", "")) else "other"
        e_labels.append(f"{code} · {alignment} · {plume}")
    finite_counts["e"] = _saved_interval(axes[4], d, value_col="mean_station_oc_difference",
        low_col="ci_low", high_col="ci_high", labels=e_labels, colors=[PURPLE] * len(d), zero_line=True)
    group_label_sets["e"] = e_labels
    axes[4].set_xlabel("Observed plume − control OC (µg m$^{-3}$)")
    _add_panel_header(axes[4], "E", "Observed OC difference", title_x=0.07)
    axes[4].tick_params(axis="y", pad=3)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.07, top=0.86)

    caption = (
        "Supplementary Figure S5 | Complementary fire-related consistency evidence. "
        f"(A,B) All 16 saved HMS plume-fraction correlations with intervals, grouped by year and "
        f"evaluation mode for USA (n={len(a_by_country['USA'])}) and CAN "
        f"(n={len(a_by_country['CAN'])}); paired vertical offsets/color distinguish MET profiles, "
        "and circles/diamonds denote fixed/rolling evaluation. No country records are connected. "
        "(C) All eight 2023 fire-product comparator correlations are grouped by country/product, "
        "with paired MET profiles. (D) All six EPA organic-carbon (OC) tracer correlations are "
        "grouped by parameter/method code, with paired MET profiles. (E) All 12 station-level "
        "plume-minus-control OC contrasts use compact labels: same = same date labels; ±1d = "
        "expanded local-date window; any = any plume; M/H = medium/heavy plume. HMS dates are UTC "
        "and EPA OC dates are local, so equal date labels do not establish within-day overlap. "
        "The records share source supports, OC is nonspecific, and these comparisons do not identify "
        "ground-level smoke mass."
    )
    metadata = {
        "source_rows": {"a": len(a), "b": len(b), "c": len(c), "d": len(d)},
        "plotted_estimates": finite_counts,
        "retention": {"a/b": "all 16 HMS rows grouped into year/mode strata; 10 USA and 6 CAN",
                      "c": "all 8 comparator rows grouped into four country/product pairs",
                      "d": "all 6 OC correlation rows grouped into three code pairs",
                      "e": "all 12 observed OC contrast rows",
                      "excluded_rows": 0},
        "country_rows": {country: len(subset) for country, subset in a_by_country.items()},
        "panel_mapping": {"A": "USA HMS correlations grouped by year/evaluation mode",
                          "B": "CAN HMS correlations grouped by year/evaluation mode",
                          "C": "fire-product correlations grouped by country/product",
                          "D": "EPA OC-tracer correlations grouped by parameter/method",
                          "E": "station-level observed OC contrasts"},
        "group_labels": group_label_sets,
        "interpretive_notes": ["HMS dates are UTC; EPA OC dates are local.",
                               "Same-date matches date labels; ±1 day expands the local-date window.",
                               "No country-spanning lines are drawn."],
        "caption_en": caption,
        "outputs": _save_figure(fig, outdir, "Supplementary_Figure_S5", 198.0),
    }
    return metadata, caption


def _gate_label(value: Any) -> str:
    number = _float(value)
    return f"P{number * 100:g}" if math.isfinite(number) else _clean(value)


def _s6_row_label(row: pd.Series) -> str:
    parts: list[str] = []
    task = str(row.get("task", ""))
    if task:
        task = task.replace("T3_fixed_", "fixed ").replace("T3_rolling_", "rolling ").replace("T3_", "")
        parts.append(task.replace("_", " "))
    if "year" in row and str(row.get("year")) not in task:
        parts.append(str(row.get("year")))
    if "profile" in row:
        parts.append("MET" if str(row.profile) == "with_met" else "no MET" if str(row.profile) == "no_met" else _clean(row.profile))
    param, method = _clean(row.get("parameter_code", "")), _clean(row.get("method_code", ""))
    if param not in ("", "NA", "not_applicable") or method not in ("", "NA", "not_applicable"):
        parts.append(f"{param}/{method}")
    if "alignment" in row:
        parts.append(_alignment(row.alignment))
    if "plume_definition" in row:
        parts.append(_plume(row.plume_definition))
    return " · ".join(parts) if parts else "saved stratum"


def _s6_compact_row_label(row: pd.Series, scope: str) -> str:
    parts: list[str] = []
    if scope == "PM_future" or scope.startswith("PM_"):
        task = str(row.get("task", ""))
        year = str(row.get("year", ""))
        if task:
            label = task.replace("T3_fixed_", "fixed ").replace("T3_rolling_", "rolling ").replace("T3_", "")
            label = label.replace("_", " ")
            if year and year not in label:
                label = f"{year} {label}"
            parts.append(label)
        elif year:
            parts.append(year)
    parameter = _clean(row.get("parameter_code", ""))
    method = _clean(row.get("method_code", ""))
    if parameter not in ("", "NA", "not_applicable") or method not in ("", "NA", "not_applicable"):
        parts.append(f"{parameter}/{method}")
    if "profile" in row:
        parts.append(_short_profile(row.profile))
    return " · ".join(parts) if parts else "saved stratum"


def _draw_s6(aframe: pd.DataFrame, bframe: pd.DataFrame, outdir: Path) -> tuple[dict[str, Any], str]:
    """Render all 264 correlations as a compact three-facet, 12-column matrix."""
    if len(aframe) != 264 or len(bframe) != 264:
        raise ValueError("S6 source must retain all 264 correlation and 264 coverage rows")
    required = {"pearson_r", "country", "model_scope", "alignment", "plume_definition", "gate_quantile"}
    if not required.issubset(aframe.columns):
        raise ValueError(f"S6a is missing fields: {sorted(required - set(aframe.columns))}")
    data = aframe.copy().reset_index(drop=True)
    alignments = list(dict.fromkeys(data.alignment.dropna().astype(str).tolist()))
    alignments.sort(key=lambda value: (0 if "same_named" in value else 1, value))
    plumes = list(dict.fromkeys(data.plume_definition.dropna().astype(str).tolist()))
    plumes.sort(key=lambda value: (0 if value == "any_plume" else 1, value))
    gates = sorted(data.gate_quantile.dropna().unique().tolist(), key=lambda value: _float(value))
    column_keys = [(alignment, plume, str(gate))
                   for alignment, plume in itertools.product(alignments, plumes)
                   for gate in gates]
    if len(column_keys) != 12:
        raise ValueError(f"S6 pivot expected 12 date×plume×gate columns, found {len(column_keys)}")
    col_positions = {key: i for i, key in enumerate(column_keys)}
    gate_labels = [_gate_label(gate) for gate in gates]
    gate_roles: dict[str, list[str]] = {}
    for gate in gates:
        subset = data[data.gate_quantile == gate]
        roles = list(dict.fromkeys(subset.get("gate_role", pd.Series(dtype=str)).dropna().astype(str).tolist()))
        gate_roles[_gate_label(gate)] = [role.replace("prespecified_sensitivity", "sensitivity").replace("_", " ")
                                         for role in roles]
    facets = list(dict.fromkeys((str(row.country), str(row.model_scope)) for row in data.itertuples()))
    country_order = {"USA": 0, "CAN": 1}
    facets.sort(key=lambda item: (0 if item[1] == "PM_future" or item[1].startswith("PM_") else 1,
                                  country_order.get(item[0], 9), item[0], item[1]))
    if len(facets) != 3:
        raise ValueError(f"S6 pivot expected USA-PM, CAN-PM, and USA-OC facets, found {facets}")

    key_cols = [column for column in ("task", "year", "parameter_code", "method_code", "profile")
                if column in data.columns]
    indicator_values = list(dict.fromkeys(data.get("HMS_indicator", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    if len(indicator_values) > 1:
        key_cols.append("HMS_indicator")
    matrices: list[tuple[tuple[str, str], list[str], np.ndarray, int, int]] = []
    total_mapped = finite_values = 0
    facet_records: list[dict[str, Any]] = []
    for country, scope in facets:
        subset = data[(data.country.astype(str) == country) & (data.model_scope.astype(str) == scope)]
        row_positions: dict[tuple[tuple[str, ...], int], int] = {}
        row_first: dict[tuple[tuple[str, ...], int], pd.Series] = {}
        row_order: list[tuple[tuple[str, ...], int]] = []
        matrix_rows: list[list[float]] = []
        duplicates: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        for _, row in subset.iterrows():
            base_key = tuple(_clean(row[column]) for column in key_cols)
            column_key = (str(row.alignment), str(row.plume_definition), str(row.gate_quantile))
            if column_key not in col_positions:
                raise ValueError(f"Unmapped S6 date/plume/gate combination: {column_key}")
            duplicate = duplicates.get((base_key, column_key), 0)
            duplicates[(base_key, column_key)] = duplicate + 1
            row_key = (base_key, duplicate)
            if row_key not in row_positions:
                row_positions[row_key] = len(row_order)
                row_order.append(row_key)
                row_first[row_key] = row
                matrix_rows.append([math.nan] * len(column_keys))
            row_index = row_positions[row_key]
            col_index = col_positions[column_key]
            if math.isfinite(matrix_rows[row_index][col_index]):
                raise AssertionError("S6 duplicate heatmap cell was not assigned a separate row")
            value = _float(row.pearson_r)
            matrix_rows[row_index][col_index] = value
            finite_values += int(math.isfinite(value))
            total_mapped += 1
        labels = []
        for row_key in row_order:
            label = _s6_compact_row_label(row_first[row_key], scope)
            if row_key[1] > 0:
                label += f" · rep {row_key[1] + 1}"
            labels.append(label)
        matrix = np.asarray(matrix_rows, dtype=float)
        matrices.append(((country, scope), labels, matrix, len(subset), int(np.isfinite(matrix).sum())))
        facet_records.append({"country": country, "model_scope": scope, "source_rows": len(subset),
                              "heatmap_rows": len(labels), "mapped_cells": len(subset), "row_labels": labels})
    if total_mapped != len(data):
        raise AssertionError(f"S6 heatmap maps {total_mapped} of {len(data)} rows")

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 190 / 25.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[10, 6, 6], hspace=0.60,
                          left=0.27, right=0.89, bottom=0.26, top=0.90)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E5E5E5")
    axes, images = [], []
    for index, ((country, scope), labels, matrix, _, _) in enumerate(matrices):
        axis = fig.add_subplot(gs[index, 0], sharex=axes[0] if axes else None)
        axes.append(axis)
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1,
                            aspect="auto", interpolation="nearest")
        images.append(image)
        axis.set_xticks(np.arange(12), gate_labels * 4, fontsize=7.5)
        axis.tick_params(axis="x", labelbottom=True, labelsize=7.5, length=2, pad=2)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        axis.tick_params(axis="y", length=0, pad=3)
        scope_title = ("PM model" if scope == "PM_future" or scope.startswith("PM_")
                       else "EPA OC (10-km exclusion)" if "OC" in scope else scope.replace("_", " "))
        _add_panel_header(axis, chr(65 + index), f"{country} · {scope_title}", title_x=0.14,
                               y=1.20 if index == 0 else 1.08)
        if index == 2:
            axis.set_xlabel("Gate quantile: P80/P95 sensitivity; P90 primary", fontsize=8, labelpad=4)
        axis.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", bottom=False, left=False)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                axis.text(x, y, f"{value:.3f}" if math.isfinite(value) else "NA",
                          ha="center", va="center", fontsize=7.5, color="#17232A")
    def short_alignment(value: str) -> str:
        return "same date" if "same_named_date" in value else "±1 day" if "plusminus1day" in value else _alignment(value)
    def short_plume(value: str) -> str:
        return "any plume" if "any_plume" in value else "medium/heavy" if "medium_or_heavy" in value else _plume(value)
    group_labels = [f"{short_alignment(alignment)}\n{short_plume(plume)}"
                    for alignment, plume in itertools.product(alignments, plumes)]
    group_axis = axes[0].secondary_xaxis("top")
    group_axis.set_xticks([i * 3 + 1 for i in range(4)], group_labels)
    group_axis.tick_params(axis="x", labelsize=7.5, length=0, pad=3)
    group_axis.spines["top"].set_visible(False)
    cbar_ax = fig.add_axes([0.91, 0.43, 0.020, 0.36])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.ax.set_title("r", fontsize=8, pad=5)
    cbar.ax.tick_params(labelsize=7.5, pad=2)

    known = pd.to_numeric(bframe.known_fraction, errors="coerce").to_numpy(float)
    retained = pd.to_numeric(bframe.retained_fraction, errors="coerce").to_numpy(float)
    def summarize(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        return {"median": float(np.median(finite)) if len(finite) else None,
                "min": float(np.min(finite)) if len(finite) else None,
                "max": float(np.max(finite)) if len(finite) else None,
                "finite_n": int(len(finite)), "missing_n": int(len(values) - len(finite))}
    known_summary, retained_summary = summarize(known), summarize(retained)
    def pct_summary(label: str, summary: dict[str, Any]) -> str:
        if summary["finite_n"] == 0:
            return f"{label}: no finite values; missing={summary['missing_n']}"
        return (f"{label}: median {summary['median'] * 100:.2f}%; range "
                f"{summary['min'] * 100:.2f}–{summary['max'] * 100:.2f}%; "
                f"n={summary['finite_n']}; missing={summary['missing_n']}")
    fig.text(0.035, 0.205, "D", ha="left", va="center", fontsize=18,
             fontfamily="Arial", fontweight="bold")
    fig.text(0.14, 0.205, "HMS availability and observation retention · 264 comparisons",
             ha="left", va="center", fontsize=10, fontweight="bold")
    fig.text(0.14, 0.135, pct_summary("Days with known HMS status", known_summary),
             ha="left", va="center", fontsize=8.2)
    fig.text(0.14, 0.075, pct_summary("Retained observation fraction", retained_summary),
             ha="left", va="center", fontsize=8.2)
    indicator_note = f" using {indicator_values[0].replace('_', ' ')}" if len(indicator_values) == 1 else ""
    caption = (
        "Supplementary Figure S6 | HMS definition, date alignment, and fire-gate sensitivity. "
        f"(A–C) All 264 strict log-smear/smearing correlations{indicator_note} are shown as three "
        "country/model-scope facets (USA PM, CAN PM, USA EPA OC). Rows identify year/task/profile "
        "or OC parameter/method/profile; the 12 columns group both date alignments and both plume "
        "definitions, each crossed with the P80/P95 sensitivity and P90 primary gates. Cell values "
        "are saved Pearson r on a common −1 to 1 scale. (D) Known-day and retained-row fractions "
        "are summarized by median, range, and available/unavailable counts across all 264 coverage rows. "
        "Unknown archive days remain unknown; matching date labels do not establish OC sampling-window "
        "overlap. Analytical strata share supports and are not independent model replications."
    )
    metadata = {
        "source_rows": {"a": len(aframe), "b": len(bframe)},
        "retention": {"a": "all 264 correlation rows mapped; no exclusions",
                      "b": "all 264 coverage rows summarized; no exclusions", "excluded_rows": 0,
                      "mapped_correlation_values": total_mapped, "finite_correlations": finite_values},
        "heatmap": {"facets": facet_records, "row_key_columns": key_cols,
                    "column_groups": [{"alignment": alignment, "plume_definition": plume,
                                       "gates": gate_labels, "gate_roles": gate_roles}
                                      for alignment, plume in itertools.product(alignments, plumes)],
                    "HMS_indicators": indicator_values, "color_scale": "shared Pearson r from -1 to 1",
                    "annotated_values": len(aframe)},
        "availability": {"source_rows": len(bframe), "known_fraction": known_summary,
                         "retained_fraction": retained_summary,
                         "visual_summary": "Median, observed range, and available/unavailable counts in compact footer."},
        "display_panel_mapping": {"A": "USA PM correlation heatmap", "B": "CAN PM correlation heatmap",
                                  "C": "USA EPA OC correlation heatmap", "D": "HMS known-day and retained-row summary"},
        "caption_en": caption,
        "outputs": _save_figure(fig, outdir, "Supplementary_Figure_S6", 190.0),
    }
    return metadata, caption


def _month_number(value: Any) -> int | None:
    number = _float(value)
    if math.isfinite(number):
        return int(number)
    return {"jan": 1, "january": 1, "apr": 4, "april": 4,
            "jul": 7, "july": 7, "oct": 10, "october": 10}.get(str(value).lower())


def _truthy(series: pd.Series) -> np.ndarray:
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def _draw_s7(source_path: Path, expected_rows: int, outdir: Path) -> tuple[dict[str, Any], str]:
    """Two-pass gzip stream: count/scale first, then plot every finite cell."""
    required = {"model_component", "country", "month_panel", "longitude", "latitude"}
    chunksize = 200_000
    totals: dict[tuple[str, int], dict[str, Any]] = {}
    total_rows = total_finite = total_nodata = total_support_discordant = 0
    finite_without_coordinates = 0
    value_min, value_max = math.inf, -math.inf
    months_order = [1, 4, 7, 10]
    countries_order = ["USA", "CAN"]

    for chunk in pd.read_csv(source_path, low_memory=False, chunksize=chunksize, compression="infer"):
        if not required.issubset(chunk.columns):
            raise ValueError(f"S7 source missing columns: {sorted(required - set(chunk.columns))}")
        total_rows += len(chunk)
        values = pd.to_numeric(chunk.model_component, errors="coerce").to_numpy(float)
        lon = pd.to_numeric(chunk.longitude, errors="coerce").to_numpy(float)
        lat = pd.to_numeric(chunk.latitude, errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        coords_finite = np.isfinite(lon) & np.isfinite(lat)
        finite_without_coordinates += int((finite & ~coords_finite).sum())
        countries = chunk.country.fillna("NA").astype(str).to_numpy()
        months = np.array([_month_number(value) for value in chunk.month_panel], dtype=object)
        supported = _truthy(chunk["is_supported"]) if "is_supported" in chunk else finite
        total_support_discordant += int((supported != finite).sum())
        total_finite += int(finite.sum())
        total_nodata += int((~finite).sum())
        if finite.any():
            value_min = min(value_min, float(np.min(values[finite])))
            value_max = max(value_max, float(np.max(values[finite])))
        for country in dict.fromkeys(countries.tolist()):
            for month in months_order:
                mask = (countries == country) & (months == month)
                if not mask.any():
                    continue
                key = (country, month)
                record = totals.setdefault(key, {"rows": 0, "finite": 0, "nodata": 0,
                    "supported": 0, "longitude_min": math.inf, "longitude_max": -math.inf,
                    "latitude_min": math.inf, "latitude_max": -math.inf, "latitude_sum": 0.0,
                    "coordinate_rows": 0})
                record["rows"] += int(mask.sum())
                record["finite"] += int((finite & mask).sum())
                record["nodata"] += int((~finite & mask).sum())
                record["supported"] += int((supported & mask).sum())
                # Map extent follows finite saved values only; all source rows still remain
                # in the denominator and all finite cells are plotted in the second pass.
                coord_mask = mask & finite & coords_finite
                if coord_mask.any():
                    record["longitude_min"] = min(record["longitude_min"], float(np.min(lon[coord_mask])))
                    record["longitude_max"] = max(record["longitude_max"], float(np.max(lon[coord_mask])))
                    record["latitude_min"] = min(record["latitude_min"], float(np.min(lat[coord_mask])))
                    record["latitude_max"] = max(record["latitude_max"], float(np.max(lat[coord_mask])))
                    record["latitude_sum"] += float(np.sum(lat[coord_mask]))
                    record["coordinate_rows"] += int(coord_mask.sum())

    if total_rows != expected_rows:
        raise ValueError(f"S7 gzip rows {total_rows} do not match registry {expected_rows}")
    if finite_without_coordinates:
        raise ValueError(f"S7 has {finite_without_coordinates} finite cells without finite coordinates")
    if not math.isfinite(value_min) or not math.isfinite(value_max):
        raise ValueError("S7 source has no finite model_component values")
    if value_min < 0:
        norm = SymLogNorm(linthresh=max(0.01, min(abs(value_min), abs(value_max)) / 10),
                          vmin=value_min, vmax=value_max)
        cmap = "RdBu_r"
        scale_description = "symmetric-log color mapping; colorbar ticks show raw model-component units"
    else:
        upper = value_max if value_max > 0 else 1.0
        norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=upper)
        cmap = "viridis"
        scale_description = "square-root color mapping; colorbar ticks show raw model-component units"

    fig, axes = plt.subplots(2, 4, figsize=(WIDTH_MM / 25.4, 105 / 25.4), squeeze=False)
    finite_extent_records = [record for record in totals.values() if record["coordinate_rows"]]
    if not finite_extent_records:
        raise ValueError("S7 has no finite cells with coordinates")
    finite_extent = (
        min(record["longitude_min"] for record in finite_extent_records),
        max(record["longitude_max"] for record in finite_extent_records),
        min(record["latitude_min"] for record in finite_extent_records),
        max(record["latitude_max"] for record in finite_extent_records),
    )
    union_mean_lat = sum(record["latitude_sum"] for record in finite_extent_records) / sum(
        record["coordinate_rows"] for record in finite_extent_records)
    union_aspect = 1 / max(0.15, abs(math.cos(math.radians(union_mean_lat))))
    plotted_finite = 0
    for chunk in pd.read_csv(source_path, low_memory=False, chunksize=chunksize, compression="infer"):
        values = pd.to_numeric(chunk.model_component, errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        lon = pd.to_numeric(chunk.longitude, errors="coerce").to_numpy(float)
        lat = pd.to_numeric(chunk.latitude, errors="coerce").to_numpy(float)
        countries = chunk.country.fillna("NA").astype(str).to_numpy()
        months = np.array([_month_number(value) for value in chunk.month_panel], dtype=object)
        for row, country in enumerate(countries_order):
            for col, month in enumerate(months_order):
                mask = finite & (countries == country) & (months == month)
                if not mask.any():
                    continue
                axes[row, col].scatter(lon[mask], lat[mask], c=values[mask], cmap=cmap, norm=norm,
                                       s=0.8, marker="s", linewidths=0, rasterized=True)
                plotted_finite += int(mask.sum())
    month_names = {1: "January", 4: "April", 7: "July", 10: "October"}
    for row, country in enumerate(countries_order):
        lon_min, lon_max, lat_min, lat_max = finite_extent
        for col, month in enumerate(months_order):
            axis = axes[row, col]
            record = totals.get((country, month), {"rows": 0, "finite": 0, "nodata": 0})
            axis.set_xlim(lon_min, lon_max)
            axis.set_ylim(lat_min, lat_max)
            axis.set_aspect(union_aspect, adjustable="box")
            label = chr(ord("A") + row * len(months_order) + col)
            axis.text(0.03, 1.19, label, transform=axis.transAxes, ha="left", va="bottom",
                      fontsize=18, fontfamily="Arial", fontweight="bold", clip_on=False)
            axis.text(0.68, 1.19, f"{country} · {month_names[month]}",
                      transform=axis.transAxes, ha="center", va="bottom", fontsize=10,
                      fontfamily="Arial", fontweight="bold", clip_on=False)
            axis.text(0.68, 1.06, f"n={record['finite']:,}/{record['rows']:,}",
                      transform=axis.transAxes, ha="center", va="bottom", fontsize=8.2,
                      fontfamily="Arial", fontweight="bold", clip_on=False)
            axis.tick_params(labelsize=8, length=2, pad=1)
            if row == 1:
                axis.set_xlabel("Longitude (°)", fontsize=8)
            else:
                axis.set_xlabel("")
            if col == 0:
                axis.set_ylabel("Latitude (°)", fontsize=8)
            else:
                axis.set_ylabel("")
            axis.grid(color="#E3E7E9", linewidth=0.35, alpha=0.65)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.09, top=0.86, wspace=0.28, hspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.025, 0.60])
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.set_label("Modeled fire-related component (µg m$^{-3}$)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Fire-related model component · selected months in 2023", y=0.985, fontsize=10)
    if plotted_finite != total_finite:
        raise AssertionError(f"S7 plotted {plotted_finite} of {total_finite} finite cells")
    support_counts = {
        f"{country}/{month_names[month]}": {
            "rows": int(totals.get((country, month), {}).get("rows", 0)),
            "finite_model_component": int(totals.get((country, month), {}).get("finite", 0)),
            "nodata_model_component": int(totals.get((country, month), {}).get("nodata", 0)),
            "support_flag_true": int(totals.get((country, month), {}).get("supported", 0)),
        }
        for country in countries_order for month in months_order
    }
    caption = (
        "Supplementary Figure S7 | Sparse reference-grid integration for January, April, July, "
        "and October 2023. (A–D) United States maps; (E–H) Canada maps, with columns ordered "
        "January, April, July, and October. All finite saved strict-QA with-MET log-smear/smearing P90 model-component "
        "cells are shown for the United States and Canada using one shared square-root color mapping "
        "and a dedicated colorbar labeled in µg m⁻³. All panels share the union extent of finite-value "
        "coordinates; the complete grid footprint and row denominators are larger. Nonfinite cells "
        "remain nodata and are counted in the source summary; no interpolation is performed. "
        "Grid cells are spatially dependent. "
        "The model component is a statistical PM2.5 output, not a direct smoke-mass measurement or a "
        "complete national/global exposure release."
    )
    metadata = {
        "source_rows": total_rows,
        "finite_model_component_rows": total_finite,
        "nodata_model_component_rows": total_nodata,
        "finite_cells_plotted": plotted_finite,
        "support_flag_discordant_rows": total_support_discordant,
        "retention": {"all_rows_streamed": total_rows, "all_finite_cells_plotted": plotted_finite,
                      "nodata_rows_retained_and_counted": total_nodata,
                      "interpolation": "none", "finite_cells_without_coordinates": finite_without_coordinates},
        "support_counts_by_country_month": support_counts,
        "display_panel_mapping": {chr(ord("A") + row * 4 + col): f"{country} · {month_names[month]}"
                                  for row, country in enumerate(countries_order)
                                  for col, month in enumerate(months_order)},
        "shared_color_scale": {"vmin": 0.0 if value_min >= 0 else value_min,
                                "vmax": value_max, "mapping": scale_description,
                                "raw_finite_minimum": value_min, "raw_finite_maximum": value_max},
        "coordinate_extent": {"basis": "union of coordinates for finite model-component cells",
                              "longitude_min": finite_extent[0], "longitude_max": finite_extent[1],
                              "latitude_min": finite_extent[2], "latitude_max": finite_extent[3],
                              "full_grid_rows_in_denominators": total_rows},
        "coordinate_aspect": "common longitude/latitude degree aspect adjusted by secant of mean latitude for union finite cells",
        "caption_en": caption,
        "outputs": _save_figure(fig, outdir, "Supplementary_Figure_S7", 105.0),
    }
    return metadata, caption


def _fold_marker_map(frame: pd.DataFrame) -> dict[str, str]:
    def sort_key(value: str) -> tuple[int, Any]:
        numeric = _float(value)
        return (0, numeric) if math.isfinite(numeric) else (1, value)
    levels = sorted(frame.get("fold_id", pd.Series(dtype=str)).dropna().astype(str).unique(), key=sort_key)
    return {value: FOLD_MARKERS[index % len(FOLD_MARKERS)] for index, value in enumerate(levels)}


def _task_name(value: Any) -> str:
    return {"T2_spatial": "T2 spatial", "T2_spatial_buffer50km": "T2 buffer (50 km)"}.get(
        str(value), str(value).replace("_", " "))


def _draw_s8(a: pd.DataFrame, b: pd.DataFrame, outdir: Path) -> tuple[dict[str, Any], str]:
    if len(a) != 154 or len(b) != 78:
        raise ValueError("S8 source must retain all 154 split rows and all 78 distance rows")
    required_a = {"country", "task", "fold_id", "source_job_id", "n_train_stations", "n_test_stations"}
    required_b = {"country", "task", "fold_id", "source_job_id", "distance_km_min",
                  "distance_km_median", "distance_km_p90"}
    if not required_a.issubset(a.columns) or not required_b.issubset(b.columns):
        raise ValueError("S8 panel is missing country/task/fold/source-job or requested measurement columns")
    a = a.copy().reset_index(drop=True)
    b = b.copy().reset_index(drop=True)
    for column in ("n_train_stations", "n_test_stations"):
        a[column] = pd.to_numeric(a[column], errors="coerce")
    for column in ("distance_km_min", "distance_km_median", "distance_km_p90"):
        b[column] = pd.to_numeric(b[column], errors="coerce")
    usa_can = b.country.astype(str).isin(("USA", "CAN"))
    if b.loc[usa_can, ["distance_km_min", "distance_km_median", "distance_km_p90"]].notna().any().any():
        raise ValueError("S8 USA/CAN distances must remain NA in every source row")

    countries = list(dict.fromkeys(a.country.fillna("NA").astype(str).tolist()))
    for country in b.country.fillna("NA").astype(str).tolist():
        if country not in countries:
            countries.append(country)
    tasks = list(dict.fromkeys(a.task.fillna("NA").astype(str).tolist()))
    fold_markers = _fold_marker_map(pd.concat([a[["fold_id"]], b[["fold_id"]]], ignore_index=True))
    task_colors = {task: TASK_COLORS.get(task, [BLUE, ORANGE, GREEN, PURPLE][i % 4])
                   for i, task in enumerate(tasks)}
    task_base_offsets = {task: (index - (len(tasks) - 1) / 2) * 0.22 for index, task in enumerate(tasks)}

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_MM / 25.4, 198 / 25.4),
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    country_x = {country: index for index, country in enumerate(countries)}
    # Within a country/task/fold group, source jobs get a small display-only offset.
    def source_job_offset(frame: pd.DataFrame, row: pd.Series) -> float:
        group = frame[(frame.country.astype(str) == str(row.country)) &
                      (frame.task.astype(str) == str(row.task)) &
                      (frame.fold_id.astype(str) == str(row.fold_id))]
        jobs = list(dict.fromkeys(group.source_job_id.fillna("NA").astype(str).tolist()))
        job = str(row.source_job_id)
        if len(jobs) <= 1:
            return 0.0
        return float(np.linspace(-0.045, 0.045, len(jobs))[jobs.index(job)])

    for _, row in a.iterrows():
        country, task = str(row.country), str(row.task)
        if country not in country_x:
            continue
        x = country_x[country] + task_base_offsets.get(task, 0.0) + source_job_offset(a, row)
        train, test = _float(row.n_train_stations), _float(row.n_test_stations)
        color = task_colors.get(task, GREY)
        marker = fold_markers.get(str(row.fold_id), "o")
        if math.isfinite(train) and math.isfinite(test):
            axes[0].plot([x, x], [train, test], color=LIGHT, lw=0.65, zorder=1)
        if math.isfinite(train):
            axes[0].scatter([x], [train], marker=marker, s=28, color=color,
                            edgecolor=color, linewidth=0.55, zorder=2)
        if math.isfinite(test):
            axes[0].scatter([x], [test], marker=marker, s=30, facecolor="white",
                            edgecolor=color, linewidth=0.9, zorder=3)
    finite_counts = pd.concat([a.n_train_stations, a.n_test_stations]).dropna().to_numpy(float)
    if len(finite_counts):
        low = float(np.min(finite_counts)); high = float(np.max(finite_counts))
        if low > 0 and high / low >= 50:
            axes[0].set_yscale("log")
            count_scale = "logarithmic"
        elif low == 0 and high >= 50:
            axes[0].set_yscale("symlog", linthresh=1)
            count_scale = "symmetric logarithmic"
        else:
            count_scale = "linear"
    else:
        count_scale = "no finite station counts"
    axes[0].set_xticks(range(len(countries)), countries, fontsize=8)
    axes[0].set_xlim(-0.55, max(0.55, len(countries) - 0.45))
    axes[0].set_ylabel("Stations per validation fold", fontsize=8)
    _add_panel_header(axes[0], "A", "Training and test station counts · 154 validation records", title_x=0.08)
    axes[0].grid(axis="y", color=LIGHT, lw=0.45)

    missing_by_country: dict[str, int] = {}
    finite_distance_rows = 0
    for country in countries:
        subset = b[b.country.astype(str) == country]
        missing_by_country[country] = int(subset.distance_km_median.isna().sum())
    for _, row in b.iterrows():
        country = str(row.country)
        if country not in country_x:
            continue
        x = country_x[country] + source_job_offset(b, row)
        lo, med, hi = (_float(row.get(column)) for column in
                       ("distance_km_min", "distance_km_median", "distance_km_p90"))
        color = task_colors.get(str(row.task), BLUE)
        marker = fold_markers.get(str(row.fold_id), "o")
        if math.isfinite(lo) and math.isfinite(hi):
            axes[1].plot([x, x], [lo, hi], color=color, lw=1.0, alpha=0.75, zorder=1)
        if math.isfinite(med):
            axes[1].scatter([x], [med], marker=marker, s=29, color=color,
                            edgecolor="white", linewidth=0.5, zorder=3)
            finite_distance_rows += 1
    axes[1].set_xticks(range(len(countries)),
        [f"{country}\nNA:{missing_by_country.get(country, 0)}" if missing_by_country.get(country, 0)
         else country for country in countries], fontsize=8)
    axes[1].set_xlim(-0.55, max(0.55, len(countries) - 0.45))
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Train-to-test station distance (km)", fontsize=8)
    _add_panel_header(axes[1], "B", "T2 station distances · 78 records (min–P90; marker = median)",
                      title_x=0.08)
    axes[1].grid(axis="y", color=LIGHT, lw=0.45)
    axes[1].tick_params(axis="x", pad=2)

    task_handles = [Line2D([], [], color=task_colors[task], marker="o", lw=0,
                           markersize=5, label=_task_name(task)) for task in tasks]
    train_test_handles = [
        Line2D([], [], color=GREY, marker="o", markerfacecolor=GREY, lw=0,
               markersize=5, label="training (filled)"),
        Line2D([], [], color=GREY, marker="o", markerfacecolor="white", markeredgecolor=GREY,
               lw=0, markersize=5, label="test (open)"),
    ]
    fold_handles = [Line2D([], [], color=GREY, marker=marker, lw=0, markersize=5,
                           label=f"fold {fold}") for fold, marker in fold_markers.items()]
    handles = task_handles + train_test_handles + fold_handles
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=min(5, max(1, len(handles))), fontsize=8, handletextpad=0.35,
               columnspacing=0.85)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.82, hspace=0.42)

    caption = (
        "Supplementary Figure S1 | Saved spatial-fold geometry summaries without map reconstruction. "
        "(A) All 154 country/task/source-job/fold rows show paired training (filled) and test (open) "
        "station counts; color denotes task, marker shape denotes fold, and the count axis is "
        f"{count_scale}. (B) All 78 saved T2 distance rows show minimum-to-P90 ranges with a median "
        "marker. Country labels give the number of rows with unavailable distance summaries; USA and "
        "CAN values remain NA. Small horizontal offsets separate source jobs and have no quantitative "
        "meaning. No coordinates, block maps, or interpolated geometry are inferred."
    )
    country_task_rows = {
        f"{country}/{task}": int(((a.country.astype(str) == country) & (a.task.astype(str) == task)).sum())
        for country in countries for task in tasks
    }
    country_distance_rows = {country: int((b.country.astype(str) == country).sum()) for country in countries}
    metadata = {
        "source_rows": {"a": len(a), "b": len(b)},
        "retention": {"a": "all 154 split rows; every finite train/test count plotted",
                      "b": "all 78 distance rows; every finite range/median plotted and NA rows counted",
                      "excluded_rows": 0},
        "plotted_counts": {"train_finite": int(a.n_train_stations.notna().sum()),
                           "test_finite": int(a.n_test_stations.notna().sum()),
                           "distance_rows_with_finite_median": finite_distance_rows,
                           "distance_rows_with_missing_median": int(b.distance_km_median.isna().sum())},
        "station_count_scale": count_scale,
        "fold_marker_map": fold_markers,
        "country_task_split_rows": country_task_rows,
        "country_distance_rows": country_distance_rows,
        "distance_missing_by_country": missing_by_country,
        "display_panel_mapping": {"A": "training/test station counts", "B": "saved T2 distance ranges"},
        "usa_can_distance_values_verified_na": True,
        "caption_en": caption,
        "outputs": _save_figure(fig, outdir, "Supplementary_Figure_S1", 198.0),
    }
    return metadata, caption


def render(repo_root: str | Path, outdir: str | Path) -> dict[str, Any]:
    """Render S1 and S5–S7 and return a machine-readable source/retention record."""
    root = _resolve_repo_root(repo_root)
    output = Path(outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path = root / "data" / "summary" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "backend": "Python/matplotlib",
        "figure_width_mm": WIDTH_MM,
        "maximum_height_mm": MAX_HEIGHT_MM,
        "text_minimum_pt": 8.2,
        "nominal_text_pt": {"S1": 8.2, "S5": 8.2, "S6_dense_heatmap": 8.2,
                            "S6_availability_footer": 8.2, "S7": 8.2},
        "source_files": {
            "figure_registry": {"path": "data/summary/figure_registry.json",
                                "sha256": _sha256(registry_path)},
        },
        "figures": {},
        "caption_proposals_en": {},
        "source_integrity": "panel CSV/CSV.GZ files are read-only; no source data are rewritten",
    }
    expected_names = {
        "Supplementary_Figure_S1": ("a", "b"),
        "Supplementary_Figure_S5": ("a", "b", "c", "d"),
        "Supplementary_Figure_S6": ("a", "b"),
        "Supplementary_Figure_S7": ("a",),
    }
    entries = {name: _figure_entry(registry, name) for name in expected_names}
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for name, panels in expected_names.items():
        loaded[name] = {}
        for panel in panels:
            frame, path = _read_panel(root, entries[name], panel)
            loaded[name][panel] = frame
            relative = path.relative_to(root).as_posix()
            result["source_files"][f"{name}/{panel}"] = {
                "path": relative, "sha256": _sha256(path), "rows": len(frame),
                "retained_rows": len(frame), "excluded_rows": 0,
            }

    figures: dict[str, dict[str, Any]] = {}
    figures["Supplementary_Figure_S5"], cap5 = _draw_s5(loaded["Supplementary_Figure_S5"], output)
    figures["Supplementary_Figure_S6"], cap6 = _draw_s6(
        loaded["Supplementary_Figure_S6"]["a"], loaded["Supplementary_Figure_S6"]["b"], output)
    s7_entry = entries["Supplementary_Figure_S7"]
    s7_path = _safe_panel_path(root, s7_entry["panels"]["a"])
    # Hash the gzip source by streaming bytes; the render itself is also chunked.
    result["source_files"]["Supplementary_Figure_S7/a"] = {
        "path": s7_path.relative_to(root).as_posix(), "sha256": _sha256(s7_path),
        "rows": int(s7_entry["panel_rows"]["a"]), "retained_rows": int(s7_entry["panel_rows"]["a"]),
        "excluded_rows": 0,
    }
    figures["Supplementary_Figure_S7"], cap7 = _draw_s7(
        s7_path, int(s7_entry["panel_rows"]["a"]), output)
    figures["Supplementary_Figure_S1"], cap8 = _draw_s8(
        loaded["Supplementary_Figure_S1"]["a"], loaded["Supplementary_Figure_S1"]["b"], output)

    # Record design source hashes when the parent workspace includes the source snapshots.
    design_root = root.parent
    design_sources = {
        "old_plot_sources/plot_revision_figures.py": design_root / "old_plot_sources" / "plot_revision_figures.py",
        "old_plot_sources/build_supp_semantics.py": design_root / "old_plot_sources" / "build_supp_semantics.py",
        "src/pm25_repro/figures.py": root / "src" / "pm25_repro" / "figures.py",
    }
    for label, path in design_sources.items():
        if path.is_file():
            result["source_files"][f"design_reference:{label}"] = {"sha256": _sha256(path), "used_for": "redesign reference only"}
    script_path = Path(__file__).resolve()
    result["source_files"]["renderer_script"] = {"path": script_path.name, "sha256": _sha256(script_path)}
    result["figures"] = figures
    result["caption_proposals_en"] = {
        "Supplementary_Figure_S5": cap5,
        "Supplementary_Figure_S6": cap6,
        "Supplementary_Figure_S7": cap7,
        "Supplementary_Figure_S1": cap8,
    }
    metadata_path = output / "supplementary_readability_s1_s5_s7_metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    result["metadata_file"] = {"path": metadata_path.name, "sha256": _sha256(metadata_path)}
    # Add the self-reference only to the returned object; avoid a recursive JSON hash.
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True,
                        help="Repository root containing data/summary/figure_registry.json")
    parser.add_argument("--outdir", required=True, help="Directory for the four figures and metadata")
    args = parser.parse_args()
    result = render(args.repo_root, args.outdir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

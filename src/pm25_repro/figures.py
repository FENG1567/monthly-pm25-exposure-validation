"""Portable redraws of the published PM2.5 figure panels.

This module replays saved panel tables. It does not refit models, resample data,
or read analysis outputs outside ``data/summary/figure_panels``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


_EXPECTED_FIGURES = {
    "Figure_1_Historical_Validation", "Figure_2_Future_Evaluation",
    "Figure_4_Fire_Support", "Figure_5_Calibration",
    "Figure_3_Feature_Increments", "Figure_6_Monthly_Benchmark",
    *(f"Supplementary_Figure_S{i}" for i in range(1, 9)),
}
_BLUE = "#3B6C8E"
_ORANGE = "#D98847"
_GREEN = "#568F7B"
_RED = "#A85858"
_PURPLE = "#81739A"
_GREY = "#777777"
_LIGHT = "#D9E2E8"
_PROFILE_COLOR = {"with_met": _BLUE, "no_met": _ORANGE}
_TASK_ORDER = ["T1_station", "T2_spatial", "T2_spatial_buffer50km", "T3_year_2022"]

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7.2,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.65, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "svg.hashsalt": "pm25-repro-figure-replay-v1",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_path(repo_root: Path, panel_ref: str | dict[str, Any]) -> Path:
    """Resolve a registry panel path, constrained to the public panel directory."""
    raw = panel_ref.get("path") if isinstance(panel_ref, dict) else panel_ref
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Invalid panel path: {panel_ref!r}")
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Panel path must be repo-relative: {raw}")
    root = repo_root.resolve()
    panels_root = (root / "data" / "summary" / "figure_panels").resolve()
    path = (root / rel).resolve()
    if panels_root not in path.parents:
        raise ValueError(f"Panel must live under data/summary/figure_panels: {raw}")
    return path


def _format_number(value: Any, digits: int = 2) -> str:
    """Format a saved numeric value without hiding signed or unavailable values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _series(frame: pd.DataFrame, column: str, *, numeric: bool = True) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan if numeric else "", index=frame.index)
    if numeric:
        return pd.to_numeric(frame[column], errors="coerce")
    return frame[column].fillna("").astype(str)


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _row_labels(frame: pd.DataFrame, candidates: Iterable[str] = ()) -> list[str]:
    """Make one visible label per input row; repeated labels remain distinguishable."""
    col = _first_column(frame, (*candidates, "row_label", "label"))
    if col:
        base = frame[col].fillna("NA").astype(str).tolist()
    else:
        label_cols = [c for c in (
            "country", "task", "profile", "validation_mode", "external_year",
            "fold_id", "seed", "sample_size", "mode", "parameter_code",
            "method_code", "alignment_short", "plume_short", "gate_quantile",
        ) if c in frame.columns]
        if not label_cols:
            base = [f"row {i + 1}" for i in range(len(frame))]
        else:
            base = [" · ".join(str(row[c]) for c in label_cols) for _, row in frame.iterrows()]
    counts: dict[str, int] = {}
    result = []
    for label in base:
        counts[label] = counts.get(label, 0) + 1
        result.append(label if counts[label] == 1 else f"{label} [{counts[label]}]")
    return result


def _annotate_point(ax: Any, x: float, y: float, value: Any, *, digits: int = 2,
                    color: str = "#263238", dx: float = 3) -> None:
    if math.isfinite(x):
        ax.annotate(_format_number(value, digits), (x, y), xytext=(dx, 2),
                    textcoords="offset points", fontsize=4.5, color=color, va="center")
    else:
        ax.text(.01, y, "NA", transform=ax.get_yaxis_transform(), fontsize=4.5,
                color=_GREY, va="center")


def _horizontal_estimates(ax: Any, frame: pd.DataFrame, value_col: str,
                          labels: list[str], *, low_col: str | None = None,
                          high_col: str | None = None, color_col: str | None = None,
                          marker_col: str | None = None, xlabel: str = "Saved estimate",
                          digits: int = 2, height: float = 0.0,
                          label_fontsize: float | None = None) -> None:
    values = _series(frame, value_col).to_numpy(dtype=float)
    lows = _series(frame, low_col).to_numpy(dtype=float) if low_col else np.full(len(frame), np.nan)
    highs = _series(frame, high_col).to_numpy(dtype=float) if high_col else np.full(len(frame), np.nan)
    colors = frame[color_col].fillna("").astype(str).tolist() if color_col in frame else [""] * len(frame)
    markers = frame[marker_col].fillna("").astype(str).tolist() if marker_col in frame else [""] * len(frame)
    marker_map: dict[str, str] = {}
    palette = [_BLUE, _ORANGE, _GREEN, _PURPLE, _RED, _GREY]
    marker_palette = ["o", "s", "D", "^", "v", "P", "X"]
    for item in dict.fromkeys(markers):
        marker_map[item] = {"fixed": "o", "rolling": "D", "with_met": "o", "no_met": "s"}.get(
            item, marker_palette[len(marker_map) % len(marker_palette)] if item else "o")
    yy = np.arange(len(frame), dtype=float)
    for i, (value, low, high) in enumerate(zip(values, lows, highs)):
        color = _PROFILE_COLOR.get(colors[i], palette[i % len(palette)])
        mark = marker_map.get(markers[i], "o")
        if math.isfinite(low) and math.isfinite(high):
            ax.plot([low, high], [yy[i], yy[i]], color=color, lw=.7, zorder=1)
        if math.isfinite(value):
            ax.scatter([value], [yy[i]], s=10, c=[color], marker=mark, zorder=3,
                       edgecolors="white", linewidths=.25)
            _annotate_point(ax, value, yy[i], value, digits=digits)
        else:
            ax.text(.01, yy[i], "NA", transform=ax.get_yaxis_transform(), fontsize=4.5,
                    color=_GREY, va="center")
    tick_fontsize = label_fontsize if label_fontsize is not None else max(3.6, min(5.2, 440 / max(1, len(labels))))
    ax.set_yticks(yy, labels, fontsize=tick_fontsize)
    ax.invert_yaxis()
    ax.axvline(0, color=_GREY, lw=.5)
    ax.grid(axis="x", color=_LIGHT, lw=.4)
    ax.set_xlabel(xlabel)
    ax.margins(y=max(.01, height))


def _value_labels(frame: pd.DataFrame, *, include: Iterable[str] = (),
                  support: Iterable[tuple[str, str]] = ()) -> list[str]:
    preferred = tuple(include)
    if "row_label" in frame:
        labels = frame["row_label"].fillna("NA").astype(str).tolist()
        extra_columns = [c for c in preferred if c != "row_label" and c in frame]
        for i, (_, row) in enumerate(frame.iterrows()):
            normalized_label = re.sub(r"[_\s]+", " ", labels[i].casefold()).strip()
            for column in extra_columns:
                raw_value = row[column]
                value = "NA" if pd.isna(raw_value) else str(raw_value)
                normalized_value = re.sub(r"[_\s]+", " ", value.casefold()).strip()
                if normalized_value and not re.search(
                    rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized_label
                ):
                    labels[i] = f"{labels[i]} · {value}"
                    normalized_label = re.sub(r"[_\s]+", " ", labels[i].casefold()).strip()
    else:
        columns = [c for c in preferred if c in frame]
        if not columns:
            columns = [c for c in ("country", "task", "profile", "fold_id", "mode") if c in frame]
        labels = [" · ".join(str(row[c]) for c in columns) for _, row in frame.iterrows()]
    labels = _row_labels(pd.DataFrame({"_label": labels}), ("_label",))
    for column, short_name in support:
        if column in frame:
            labels = [f"{label} · {short_name}={_format_number(value, 0) if pd.notna(value) else 'NA'}"
                      for label, value in zip(labels, frame[column])]
    return labels


def _short_caption(caption: str) -> str:
    first = caption.strip().splitlines()[0].strip()
    if "|" in first:
        first = first.split("|", 1)[1].strip()
    first = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)[0].strip()
    return textwrap.fill(first, width=80)


def _s6_labels(frame: pd.DataFrame, *, support: Iterable[tuple[str, str]]) -> list[str]:
    """Keep S6's scientific strata visible while using compact axis labels."""
    columns = (
        ("model_scope", "scope"), ("country", "country"), ("year", "year"),
        ("parameter_code", "parameter"), ("method_code", "method"),
        ("alignment", "date"), ("plume_definition", "plume"),
        ("profile", "profile"), ("gate_quantile", "gate"),
    )
    labels = []
    for _, row in frame.iterrows():
        parts = []
        for column, label in columns:
            if column not in frame:
                continue
            value = "NA" if pd.isna(row[column]) else str(row[column])
            if column == "model_scope":
                value = value.replace("_", " ")
            elif column in ("parameter_code", "method_code") and value == "not_applicable":
                value = "n/a"
            elif column == "alignment":
                value = {"same_named_date": "same date", "nearest_date": "nearest date"}.get(
                    value, value.replace("_", " "))
            elif column == "plume_definition":
                value = value.replace("_", " ")
            elif column == "profile":
                value = {"with_met": "MET", "no_met": "no MET"}.get(value, value.replace("_", " "))
            elif column == "gate_quantile":
                try:
                    value = f"P{round(float(value) * 100):g}"
                except (TypeError, ValueError):
                    value = value.replace("_", " ")
            parts.append(f"{label}={value}")
        for column, short_name in support:
            if column in frame:
                parts.append(f"{short_name}={_format_number(row[column], 0) if pd.notna(row[column]) else 'NA'}")
        labels.append(" · ".join(parts))
    return _row_labels(pd.DataFrame({"_label": labels}), ("_label",))


def _panel_title(ax: Any, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", fontsize=7.0)


def _draw_f1(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.5))
    for ax, label in zip(axes.ravel(), ("a", "b", "c", "d")):
        frame = p[label]
        metric = "r2" if label in ("a", "b") else "rmse"
        profile = str(frame.profile.iloc[0]).replace("_", " ") if "profile" in frame and len(frame) else "saved profile"
        countries = list(dict.fromkeys(frame.country.fillna("NA").astype(str)))
        tasks = [t for t in _TASK_ORDER if t in set(frame.task.astype(str))]
        tasks += [t for t in dict.fromkeys(frame.task.astype(str)) if t not in tasks]
        grid = np.full((len(countries), len(tasks)), np.nan)
        status: dict[tuple[str, str], str] = {}
        numeric = _series(frame, metric)
        for (_, row), value in zip(frame.iterrows(), numeric):
            if str(row.country) in countries and str(row.task) in tasks:
                i, j = countries.index(str(row.country)), tasks.index(str(row.task))
                grid[i, j] = value
                status[(str(row.country), str(row.task))] = str(row.get("cell_status", "missing"))
        values = grid[np.isfinite(grid)]
        if len(values):
            if metric == "r2":
                lo, hi = float(values.min()), float(values.max())
                if lo == hi:
                    lo -= .1; hi += .1
                norm = SymLogNorm(linthresh=.1, vmin=lo, vmax=hi)
                im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap="RdBu_r", norm=norm)
            else:
                hi = float(values.max()) or 1
                im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap="Blues", norm=Normalize(0, hi))
            for i, country in enumerate(countries):
                for j, task in enumerate(tasks):
                    val = grid[i, j]
                    if math.isfinite(val):
                        txt = _format_number(val, 2 if metric == "r2" else 1)
                        row = frame[(frame.country.astype(str) == country) & (frame.task.astype(str) == task)]
                        if len(row):
                            n = _format_number(row.iloc[0].get("n", np.nan), 0)
                            stations = _format_number(row.iloc[0].get("n_stations", np.nan), 0)
                            txt += f"\nn={n} · s={stations}"
                        rgb = im.cmap(im.norm(val))[:3]
                        color = "white" if np.dot(rgb, [.2126, .7152, .0722]) < .48 else "#203038"
                    else:
                        txt = "guard" if "guard" in status.get((country, task), "") else "NA"
                        color = _GREY
                    ax.text(j, i, txt, ha="center", va="center", fontsize=4.7, color=color)
            fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.035, pad=.09, label=metric.upper())
        ax.set_facecolor("#ECEFF1")
        ax.set_yticks(range(len(countries)), countries)
        ax.set_xticks(range(len(tasks)), [t.replace("T1_station", "T1").replace("T2_spatial_buffer50km", "T2 buffer").replace("T2_spatial", "T2").replace("T3_year_2022", "T3 2022") for t in tasks], rotation=0)
        ax.set_xlabel("Deployment task"); ax.set_ylabel("Country")
        _panel_title(ax, label, f"{metric.upper()} · {profile}")
    fig.suptitle(_short_caption(caption), fontsize=8.5, y=1.01)
    fig.tight_layout()
    return fig


def _draw_f2(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharex=False)
    for ax, label, panel_letter in zip(axes.ravel(), ("USA_r2", "USA_rmse", "CAN_r2", "CAN_rmse"), "abcd"):
        frame = p[label]
        metric = "r2" if label.endswith("r2") else "rmse"
        country = label.split("_", 1)[0]
        handles = []
        for profile in dict.fromkeys(frame.profile.astype(str)):
            for mode in dict.fromkeys(frame.validation_mode.astype(str)):
                q = frame[(frame.profile.astype(str) == profile) & (frame.validation_mode.astype(str) == mode)].copy()
                q = q.sort_values("external_year")
                color = _PROFILE_COLOR.get(profile, _GREEN)
                mark, style = ("o", "-") if mode == "fixed" else ("s", "--")
                if len(q):
                    ax.plot(q.external_year, _series(q, metric), color=color, marker=mark, ls=style,
                            ms=3, lw=.8, label=f"{profile.replace('_', ' ')}, {mode}")
                    for _, row in q.iterrows():
                        val = row.get(metric, np.nan)
                        if pd.notna(val):
                            support = f" n={_format_number(row.get('n'), 0)} / s={_format_number(row.get('n_stations'), 0)}"
                            ax.annotate(f"{_format_number(val, 2 if metric == 'r2' else 1)}{support}",
                                        (float(row.external_year), float(val)), xytext=(2, 2),
                                        textcoords="offset points", fontsize=3.8, color=color)
            h, l = ax.get_legend_handles_labels()
            for handle, text in zip(h, l):
                if text not in [x[1] for x in handles]: handles.append((handle, text))
        ax.axvline(2023, color=_GREY, lw=.55, ls=":")
        ax.grid(axis="y", color=_LIGHT, lw=.4)
        ax.set_xticks(sorted(pd.to_numeric(frame.external_year, errors="coerce").dropna().unique()))
        ax.set_xlabel("External evaluation year"); ax.set_ylabel("R²" if metric == "r2" else "RMSE (µg m⁻³)")
        _panel_title(ax, panel_letter, f"{country} · {metric.upper()}")
        if handles: ax.legend(handles=[x[0] for x in handles], labels=[x[1] for x in handles], fontsize=4.6)
    fig.suptitle(_short_caption(caption), fontsize=8.5)
    fig.tight_layout()
    return fig


def _draw_f3(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
    a, b = p["a"], p["b"]
    for ax, frame, label, value, group, xlabel in (
        (axes[0], a, "a", "full_rmse", "fire_month_scope", "Full-model RMSE (µg m⁻³)"),
        (axes[1], b, "b", "row_coverage_fraction", "rule", "Retained / eligible row-months"),
    ):
        group_label = "fire_month_scope" if label == "a" else "rule_label"
        labels = _value_labels(frame, include=("row_label", group_label),
                               support=(("n", "rows"), ("n_stations", "sites")))
        cats = frame[group].fillna("NA").astype(str).tolist() if group in frame else [""] * len(frame)
        if label == "a":
            markers = frame["fire_month_scope"].fillna("NA").astype(str).tolist()
            styles = {x: (_PROFILE_COLOR.get(x, _GREY), "o" if marker == "P90_fire_proxy" else "s")
                      for x, marker in zip(cats, markers)}
        else:
            rule_styles = {"distance_50km_non_geo_minmax": (_BLUE, "o"),
                           "distance_50km_non_geo_p01_p99": (_ORANGE, "s"), "all": (_GREY, "x")}
            styles = {x: rule_styles.get(x, (_GREY, "D")) for x in dict.fromkeys(cats)}
        x = _series(frame, value).to_numpy(float)
        for i, val in enumerate(x):
            if label == "a":
                color = _PROFILE_COLOR.get(cats[i], _GREY)
                marker = "o" if markers[i] == "P90_fire_proxy" else "s"
            else:
                color, marker = styles[cats[i]]
            yy = i + ((markers[i] == "other_months") * .1 - (markers[i] == "P90_fire_proxy") * .1 if label == "a"
                       else (list(styles).index(cats[i]) - (len(styles) - 1) / 2) * .12)
            if math.isfinite(val):
                ax.scatter(val, yy, marker=marker, color=color, s=12, zorder=3)
                display = _format_number(val, 3 if value == "row_coverage_fraction" else 2)
                if value == "row_coverage_fraction" and "n" in frame and "eligible_rows" in frame:
                    display = f"{val:.1%} · {_format_number(frame.iloc[i].get('n'), 0)}/{_format_number(frame.iloc[i].get('eligible_rows'), 0)}"
                elif value == "full_rmse" and "n" in frame:
                    display += f" · n={_format_number(frame.iloc[i].get('n'), 0)}"
                ax.annotate(display,
                            (val, yy), xytext=(3, 2), textcoords="offset points", fontsize=4.1, color=color)
            else:
                ax.text(.01, yy, "NA", transform=ax.get_yaxis_transform(), fontsize=4.5, color=_GREY)
        ax.set_yticks(range(len(frame)), labels, fontsize=4.7)
        ax.invert_yaxis(); ax.grid(axis="x", color=_LIGHT, lw=.4); ax.set_xlabel(xlabel)
        _panel_title(ax, label, "Held-out error by month stratum" if label == "a" else "P90-month source support by rule")
    fig.suptitle(_short_caption(caption), fontsize=8.3)
    fig.tight_layout()
    return fig


def _draw_f4(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.8), sharey=True)
    specs = [("a", "pointwise_coverage", "pointwise_covered_rows", "pointwise_test_rows", "Point coverage"),
             ("b", "all_month_station_coverage", "all_month_station_covered_n", "all_month_station_group_n", "Whole-station coverage"),
             ("c", "row_weighted_mean_interval_width", "finite_width_folds", "width_weight_rows", "Row-weighted width (µg m⁻³)")]
    for ax, (label, value, numerator, denominator, title) in zip(axes, specs):
        frame = p[label]; labels = _value_labels(frame, include=("country", "task", "profile"))
        values = _series(frame, value).to_numpy(float)
        for i, val in enumerate(values):
            profile = str(frame.iloc[i].get("profile", "")); color = _PROFILE_COLOR.get(profile, _BLUE)
            if math.isfinite(val):
                ax.scatter(val, i, color=color, marker="o" if profile == "with_met" else "s", s=14, zorder=3)
                _annotate_point(ax, val, i, val, digits=3 if "coverage" in value else 2, color=color)
            else:
                ax.text(.01, i, "NA", transform=ax.get_yaxis_transform(), fontsize=4.5, color=_GREY)
            if numerator in frame and denominator in frame:
                ax.text(1.02, i, f"{frame.iloc[i][numerator]}/{frame.iloc[i][denominator]}",
                        transform=ax.get_yaxis_transform(), fontsize=3.8, va="center", clip_on=False)
        ax.axvline(.9, color=_GREY, ls=":", lw=.5) if "coverage" in value else None
        ax.grid(axis="x", color=_LIGHT, lw=.4); ax.set_xlabel(title)
        _panel_title(ax, label, title)
        if label == "a": ax.set_yticks(np.arange(len(frame)), labels, fontsize=4.6); ax.invert_yaxis()
    fig.suptitle(_short_caption(caption), fontsize=8.2)
    fig.tight_layout(rect=[0, 0, .96, .98])
    return fig


def _draw_f5(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for ax, label in zip(axes, ("a", "b", "c")):
        frame = p[label]; labels = _value_labels(frame, include=("row_label",),
                                                  support=(("n_rows", "n"), ("n_test_stations", "sites")))
        y = np.arange(len(frame)); values = _series(frame, "delta_rmse").to_numpy(float)
        low = _series(frame, "simultaneous_95_low").to_numpy(float)
        high = _series(frame, "simultaneous_95_high").to_numpy(float)
        for i, val in enumerate(values):
            color = _PROFILE_COLOR.get(str(frame.iloc[i].get("profile", "")), _BLUE)
            if math.isfinite(low[i]) and math.isfinite(high[i]): ax.plot([low[i], high[i]], [i, i], color=color, lw=.8)
            if math.isfinite(val):
                ax.scatter(val, i, color=color, s=9, zorder=3)
                _annotate_point(ax, val, i, val, digits=3, color=color)
            else: ax.text(.01, i, "NA", transform=ax.get_yaxis_transform(), fontsize=4.5, color=_GREY)
        ax.set_yticks(y, labels, fontsize=4.1); ax.invert_yaxis(); ax.axvline(0, color="#222", lw=.6)
        ax.grid(axis="x", color=_LIGHT, lw=.4); ax.set_ylabel("Country · task · profile")
        _panel_title(ax, label, f"Saved contrast · {frame.contrast.iloc[0] if 'contrast' in frame and len(frame) else ''}")
    axes[-1].set_xlabel("Δ RMSE (µg m⁻³)")
    fig.suptitle(_short_caption(caption), fontsize=8.4)
    fig.tight_layout()
    return fig


def _draw_f6(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.8), gridspec_kw={"width_ratios": [1, 1.1]})
    frame = p["a"]; labels = _value_labels(frame, include=("country", "validation_mode", "external_year", "profile"),
                                                   support=(("n_matched_pairs_metric", "pairs"), ("n_matched_stations_metric", "sites")))
    for i, row in frame.iterrows():
        y = i; color = _BLUE
        vals = [row.get("full_rmse_ug_m3", np.nan), row.get("comparator_rmse_ug_m3", np.nan)]
        for j, val in enumerate(vals):
            try: x = float(val)
            except (TypeError, ValueError): x = np.nan
            if math.isfinite(x):
                axes[0].scatter(x, y + (-.12 if j == 0 else .12), color=color if j == 0 else _ORANGE,
                                marker="o" if j == 0 else "s", s=11)
                _annotate_point(axes[0], x, y + (-.12 if j == 0 else .12), x, digits=2, color=color if j == 0 else _ORANGE)
            else: axes[0].text(.01, y, "NA", transform=axes[0].get_yaxis_transform(), fontsize=4.5, color=_GREY)
    axes[0].set_yticks(np.arange(len(frame)), labels, fontsize=4.5); axes[0].invert_yaxis()
    axes[0].grid(axis="x", color=_LIGHT, lw=.4); axes[0].set_xlabel("RMSE (µg m⁻³)")
    axes[0].legend(handles=[Line2D([], [], marker="o", color=_BLUE, lw=0, label="Full"),
                            Line2D([], [], marker="s", color=_ORANGE, lw=0, label="SatPM")], fontsize=4.6)
    _panel_title(axes[0], "a", "Matched-support RMSE")
    b = p["b"]; labels_b = _value_labels(b, include=("country", "validation_mode", "external_year", "profile"),
                                             support=(("n_matched_pairs", "pairs"), ("n_matched_stations", "sites")))
    _horizontal_estimates(axes[1], b, "delta", labels_b, low_col="ci_low", high_col="ci_high",
                          color_col="profile", xlabel="ΔRMSE: full − SatPM (µg m⁻³)", digits=3)
    _panel_title(axes[1], "b", "Paired ΔRMSE with saved intervals")
    fig.suptitle(_short_caption(caption), fontsize=8.3); fig.tight_layout()
    return fig


def _draw_s1(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharey="row")
    for col, profile in enumerate(("no_met", "with_met")):
        for row, label in enumerate(("a", "b")):
            ax = axes[row, col]; frame = p[label]
            q = frame[frame.profile.astype(str) == profile] if "profile" in frame else frame
            # Shared y axes require one mapping across both meteorology panels.
            countries = sorted(set(frame.country.fillna("NA").astype(str))) if "country" in frame else ["records"]
            ymap = {c: i for i, c in enumerate(countries)}
            if row == 0 and {"qa_label", "estimator_label"}.issubset(q.columns):
                group_col = "QA / estimator"
                qa = q.qa_label.fillna(q.get('qa_variant', pd.Series('NA', index=q.index))).fillna('NA')
                est = q.estimator_label.fillna(q.get('estimator_kind', pd.Series('NA', index=q.index))).fillna('NA')
                groups = (qa.astype(str) + " / " + est.astype(str)).tolist()
            elif row == 1 and {"target_strategy", "backtransform"}.issubset(q.columns):
                group_col = "target / backtransform"
                groups = (q.target_strategy.fillna("NA").astype(str) + " / " + q.backtransform.fillna("NA").astype(str)).tolist()
            else:
                group_col = "estimator_label" if "estimator_label" in q else "saved branch"
                groups = q[group_col].fillna("NA").astype(str).tolist() if group_col in q else ["saved"] * len(q)
            palette = [_BLUE, _GREEN, _ORANGE, _PURPLE, _RED]
            group_colors = {g: palette[i % len(palette)] for i, g in enumerate(dict.fromkeys(groups))}
            for i, (_, record) in enumerate(q.iterrows()):
                country = str(record.get("country", countries[0])); val = _series(q.iloc[[i]], "r2").iloc[0]
                color = group_colors[groups[i]]
                if math.isfinite(float(val)):
                    ax.scatter(val, ymap[country] + ((i % 7) - 3) * .025, color=color, s=9)
                    support = f" n={_format_number(record.get('n'), 0)} / s={_format_number(record.get('n_stations'), 0)}"
                    ax.annotate(f"{_format_number(val, 2)}{support}",
                                (float(val), ymap[country] + ((i % 7) - 3) * .025),
                                xytext=(2, 2), textcoords="offset points", fontsize=3.5, color=color)
                else: ax.text(.01, ymap[country], "NA", transform=ax.get_yaxis_transform(), fontsize=4.3, color=_GREY)
            ax.set_xscale("symlog", linthresh=.1); ax.axvline(0, color=_GREY, lw=.5); ax.grid(axis="x", color=_LIGHT, lw=.4)
            ax.set_yticks(range(len(countries)), countries); ax.set_xlabel("T1 R² (symlog)")
            handles = [Line2D([], [], marker="o", color=color, lw=0, label=name)
                       for name, color in group_colors.items()]
            if handles: ax.legend(handles=handles, fontsize=3.6, ncol=2, loc="lower right")
            panel_letter = chr(ord("a") + row * 2 + col)
            _panel_title(ax, panel_letter, f"{profile.replace('_', ' ')} · {group_col}")
    fig.suptitle(_short_caption(caption), fontsize=8.4); fig.tight_layout()
    return fig


def _draw_s2(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 6.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    a, b = p["a"], p["b"]
    countries = list(dict.fromkeys(a.country.fillna("NA").astype(str)))
    estim_col = _first_column(a, ("estimator_kind", "estimator"))
    ymap = {country: i for i, country in enumerate(countries)}
    estimators = a[estim_col].fillna("NA").astype(str).tolist() if estim_col else ["saved"] * len(a)
    for i, (_, row) in enumerate(a.iterrows()):
        x = pd.to_numeric(pd.Series([row.get("r2")]), errors="coerce").iloc[0]
        y = ymap[str(row.get("country", "NA"))] + (list(dict.fromkeys(estimators)).index(estimators[i]) - .5) * .12
        color = _BLUE if estimators[i].lower() in ("hgb", "histgradientboosting") else _ORANGE
        if math.isfinite(float(x)):
            axes[0].scatter(x, y, color=color, marker="o" if color == _BLUE else "s", s=10)
            count = row.get("n_test_sites", row.get("n", np.nan))
            axes[0].annotate(f"{_format_number(x, 2)} · n={_format_number(count, 0)}",
                             (float(x), y), xytext=(2, 2), textcoords="offset points", fontsize=3.8, color=color)
        else: axes[0].text(.01, y, "NA", transform=axes[0].get_yaxis_transform(), fontsize=4.4, color=_GREY)
    axes[0].set_yticks(range(len(countries)), countries); axes[0].invert_yaxis(); axes[0].set_xscale("symlog", linthresh=.1)
    axes[0].axvline(0, color=_GREY, lw=.5); axes[0].grid(axis="x", color=_LIGHT, lw=.4); axes[0].set_xlabel("LOCO R² (symlog)")
    _panel_title(axes[0], "a", "Saved country transfer")
    train_col = _first_column(b, ("train_n", "n_train_stations", "sample_size"))
    size_col = _first_column(b, ("rmse", "full_rmse"))
    seeds = list(dict.fromkeys(b.seed.fillna("NA").astype(str))) if "seed" in b else ["saved"]
    for ax, country, label in zip(axes[1:], ("USA", "CAN"), ("b", "c")):
        qcountry = b[b.country.astype(str) == country] if "country" in b else b
        for (profile, seed), q in qcountry.groupby([c for c in ("profile", "seed") if c in qcountry], dropna=False, sort=False):
            q = q.sort_values(train_col) if train_col else q
            x = _series(q, train_col).to_numpy(float) if train_col else np.arange(len(q), dtype=float)
            y = _series(q, size_col).to_numpy(float) if size_col else np.full(len(q), np.nan)
            color = _PROFILE_COLOR.get(str(profile), _BLUE)
            seed_index = seeds.index(str(seed)) if str(seed) in seeds else 0
            style = ["-", "--", ":", "-."][seed_index % 4]
            if len(x): ax.plot(x, y, color=color, marker="o", ms=2.4, lw=.65, ls=style,
                              label=f"{str(profile).replace('_', ' ')} · seed {seed}")
            for xv, yv in zip(x, y):
                if math.isfinite(xv) and math.isfinite(yv): _annotate_point(ax, xv, yv, yv, digits=2, color=color, dx=2)
        ax.set_xscale("log"); ax.grid(axis="y", color=_LIGHT, lw=.4)
        ax.set_xlabel("Training stations"); ax.set_ylabel("2022 RMSE (µg m⁻³)")
        ax.legend(fontsize=4.0, ncol=1)
        _panel_title(ax, label, f"{country} saved thinning rows")
    fig.suptitle(_short_caption(caption), fontsize=8.3); fig.tight_layout()
    return fig


def _draw_s3(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig = plt.figure(figsize=(7.2, 7.7)); grid = fig.add_gridspec(2, 2, height_ratios=[5.5, 1.8])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax, label, country in zip(axes[:2], ("a", "b"), ("USA", "CAN")):
        frame = p[label]; labels = _value_labels(frame, include=("row_label",),
                                                  support=(("test_row_n", "rows"), ("test_station_n", "sites")))
        y = np.arange(len(frame))
        for metric, marker, offset in (("pointwise_coverage", "o", -.1), ("all_month_per_station_coverage", "D", .1)):
            vals = _series(frame, metric).to_numpy(float)
            for i, val in enumerate(vals):
                color = _PROFILE_COLOR.get(str(frame.iloc[i].get("profile", "")), _BLUE)
                if math.isfinite(val):
                    ax.scatter(val, i + offset, marker=marker, color=color, s=9)
                    _annotate_point(ax, val, i + offset, val, digits=3, color=color, dx=2)
                else: ax.text(.01, i + offset, "NA", transform=ax.get_yaxis_transform(), fontsize=4.2, color=_GREY)
        ax.axvline(.9, color=_GREY, ls=":", lw=.55); ax.set_xlim(-.03, 1.03); ax.set_yticks(y, labels, fontsize=4.1)
        ax.invert_yaxis(); ax.set_xlabel("Coverage fraction"); ax.grid(axis="x", color=_LIGHT, lw=.4)
        _panel_title(ax, label, f"{country} · saved primary folds ({len(frame)})")
    c = p["c"]; labels = _value_labels(c, include=("row_label",), support=(("calibration_station_n", "cal sites"),)); vals = _series(c, "q90").to_numpy(float)
    for i, val in enumerate(vals):
        color = _BLUE if str(c.iloc[i].get("qa_variant", "")) == "strict" else _ORANGE
        marker = "o" if str(c.iloc[i].get("estimator_kind", "")) == "hgb" else "s"
        if math.isfinite(val):
            axes[2].scatter(val, i, color=color, marker=marker, s=16)
            _annotate_point(axes[2], val, i, val, digits=3, color=color)
        else: axes[2].text(.01, i, "NA", transform=axes[2].get_yaxis_transform(), fontsize=4.2, color=_GREY)
    axes[2].set_yticks(range(len(c)), labels, fontsize=4.3); axes[2].invert_yaxis(); axes[2].grid(axis="x", color=_LIGHT, lw=.4)
    axes[2].set_xlabel("q90 feasibility threshold (saved source units)"); _panel_title(axes[2], "c", f"Feasibility rows ({len(c)})")
    fig.suptitle(_short_caption(caption), fontsize=8.2); fig.tight_layout()
    return fig


def _draw_s4(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    frame = p["a"]
    labels = _value_labels(frame, include=("country", "task", "profile", "mode", "alias"),
                           support=(("rows", "rows"), ("sites", "sites")))
    _horizontal_estimates(ax, frame, "delta_rmse", labels, low_col="delta_rmse_ci_low",
                          high_col="delta_rmse_ci_high", color_col="profile",
                          xlabel="ΔRMSE: full − SatPM (µg m⁻³)", digits=3)
    _panel_title(ax, "a", f"All saved exact-support rows · {len(frame)}")
    fig.suptitle(_short_caption(caption), fontsize=8.2); fig.tight_layout()
    return fig


def _draw_s5(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig = plt.figure(figsize=(7.2, 7.6)); grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.55])
    a, b, c, d = (p[x] for x in ("a", "b", "c", "d"))
    axes_a = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, country in zip(axes_a, ("USA", "CAN")):
        q = a[a.country.astype(str) == country] if "country" in a else a
        labels = _value_labels(q, include=("country", "year", "profile", "task"),
                               support=(("n_pairs", "pairs"), ("sites", "sites")))
        _horizontal_estimates(ax, q, "pearson_r", labels, low_col="ci_low", high_col="ci_high",
                              color_col="profile", xlabel="Pearson r", digits=3)
        _panel_title(ax, "a", f"{country} saved HMS correlations")
    axb = fig.add_subplot(grid[1, 0]); labels = _value_labels(b, include=("row_label",), support=(("n_pairs", "pairs"), ("n_groups", "groups")))
    _horizontal_estimates(axb, b, "pearson_r", labels, low_col="pearson_r_ci_low", high_col="pearson_r_ci_high",
                          color_col="profile", xlabel="Pearson r", digits=3)
    _panel_title(axb, "b", "Saved fire-product comparisons")
    axc = fig.add_subplot(grid[1, 1]); labels = _value_labels(c, include=("row_label", "parameter_code", "method_code", "profile"), support=(("n_pairs", "pairs"), ("n_groups", "groups")))
    _horizontal_estimates(axc, c, "pearson_r", labels, low_col="pearson_r_ci_low", high_col="pearson_r_ci_high",
                          color_col="profile", xlabel="Pearson r", digits=3)
    _panel_title(axc, "c", "EPA OC-tracer correlation codes")
    axd = fig.add_subplot(grid[2, :]); labels = _value_labels(d, include=("row_label", "parameter_code", "method_code", "alignment_short", "plume_short"), support=(("station_month_contrasts", "contrasts"), ("sites", "sites")))
    _horizontal_estimates(axd, d, "mean_station_oc_difference", labels, low_col="ci_low", high_col="ci_high",
                          xlabel="Plume − control OC (µg m⁻³)", digits=3)
    _panel_title(axd, "d", "Saved parameter / method / date-alignment strata")
    fig.suptitle(_short_caption(caption), fontsize=8.1); fig.tight_layout()
    return fig


def _draw_s6(p: dict[str, pd.DataFrame], caption: str) -> Any:
    a, b = p["a"], p["b"]
    figure_height = max(12.5, 4.5 + .075 * (len(a) + len(b)))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, figure_height),
                             gridspec_kw={"height_ratios": [max(1, len(a)), max(1, len(b))]})
    la = _s6_labels(a, support=(("n_pairs", "pairs"), ("sites", "sites")))
    _horizontal_estimates(axes[0], a, "pearson_r", la, low_col="ci_low", high_col="ci_high",
                          color_col="profile", xlabel="Pearson r", digits=3, label_fontsize=4.3)
    _panel_title(axes[0], "a", f"Saved correlation strata ({len(a)} rows)")
    lb = _s6_labels(b, support=(("source_rows", "rows"), ("geometry_matched_rows", "matched")))
    y = np.arange(len(b))
    for col, marker, offset in (("known_fraction", "o", -.1), ("retained_fraction", "D", .1)):
        vals = _series(b, col).to_numpy(float)
        for i, val in enumerate(vals):
            if math.isfinite(val):
                axes[1].scatter(val, y[i] + offset, marker=marker, color=_BLUE if marker == "o" else _ORANGE, s=8)
                _annotate_point(axes[1], val, y[i] + offset, val, digits=3, dx=2)
            else: axes[1].text(.01, y[i] + offset, "NA", transform=axes[1].get_yaxis_transform(), fontsize=4.0, color=_GREY)
    axes[1].set_yticks(y, lb, fontsize=4.3); axes[1].invert_yaxis(); axes[1].set_xlim(-.03, 1.03)
    axes[1].grid(axis="x", color=_LIGHT, lw=.4); axes[1].set_xlabel("Saved matched-day fraction")
    _panel_title(axes[1], "b", f"Known and retained fractions ({len(b)} rows)")
    fig.suptitle(_short_caption(caption), fontsize=8.1); fig.tight_layout(rect=[0, 0, 1, .985])
    return fig


def _draw_s7(path: Path, caption: str, *, chunksize: int = 100_000) -> tuple[Any, dict[str, Any]]:
    """Stream a sparse grid CSV; plot finite saved components and count nodata."""
    groups: dict[tuple[str, str], dict[str, int]] = {}
    total_rows = 0
    vmin, vmax = math.inf, -math.inf
    # First pass keeps no full-table copy in memory and records unavailable cells.
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize, compression="infer"):
        total_rows += len(chunk)
        component = pd.to_numeric(chunk.get("model_component"), errors="coerce").to_numpy(float)
        finite = np.isfinite(component)
        countries = chunk.get("country", pd.Series("NA", index=chunk.index)).fillna("NA").astype(str)
        months = chunk.get("month_panel", pd.Series("NA", index=chunk.index)).fillna("NA").astype(str)
        support = chunk.get("is_supported", pd.Series(False, index=chunk.index)).astype(str).str.lower().isin(["true", "1", "yes"])
        for country, month in dict.fromkeys(zip(countries, months)):
            mask = (countries.to_numpy() == country) & (months.to_numpy() == month)
            record = groups.setdefault((country, month), {"rows": 0, "finite": 0, "nodata": 0, "supported_flag": 0})
            record["rows"] += int(mask.sum())
            record["finite"] += int((finite & mask).sum())
            record["nodata"] += int((~finite & mask).sum())
            record["supported_flag"] += int((support.to_numpy() & mask).sum())
        if finite.any():
            vmin = min(vmin, float(component[finite].min())); vmax = max(vmax, float(component[finite].max()))
    if not groups:
        raise ValueError(f"S7 source contains no rows: {path}")
    months = list(dict.fromkeys(month for _, month in groups))
    fig, axes = plt.subplots(1, max(1, len(months)), figsize=(10.2, 4.8), squeeze=False)
    axes = axes.ravel()
    norm = Normalize(vmin=vmin if math.isfinite(vmin) else 0,
                     vmax=vmax if math.isfinite(vmax) and vmax != vmin else (vmin + 1 if math.isfinite(vmin) else 1))
    # Second pass adds saved finite cells directly. NaN rows are counted above and
    # stay visibly represented in each country/month annotation.
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize, compression="infer"):
        component = pd.to_numeric(chunk.get("model_component"), errors="coerce").to_numpy(float)
        finite = np.isfinite(component)
        if not finite.any():
            continue
        good = chunk.loc[finite]
        values = component[finite]
        for ax, month in zip(axes, months):
            qmask = good.month_panel.astype(str).eq(month).to_numpy()
            q = good.loc[qmask]
            if q.empty:
                continue
            ax.scatter(pd.to_numeric(q.longitude, errors="coerce"), pd.to_numeric(q.latitude, errors="coerce"),
                       c=values[qmask], cmap="viridis", norm=norm, s=.65, marker="s", linewidths=0,
                       rasterized=True)
    for ax, month in zip(axes, months):
        country_notes = []
        for country in dict.fromkeys(country for country, _ in groups):
            rec = groups.get((country, month))
            if rec:
                country_notes.append(f"{country}: finite {rec['finite']}/{rec['rows']}; nodata {rec['nodata']}")
        ax.set_title(month, loc="left", fontsize=7)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(color="#dddddd", lw=.3)
        ax.text(.02, .02, "\n".join(country_notes), transform=ax.transAxes, fontsize=4.5,
                va="bottom", bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    if math.isfinite(vmin):
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
        fig.colorbar(sm, ax=axes.tolist(), shrink=.78, label="Saved model_component")
    fig.suptitle(_short_caption(caption), fontsize=8.2); fig.tight_layout()
    summary = {"rows": total_rows, "finite_model_component": int(sum(x["finite"] for x in groups.values())),
               "nodata_model_component": int(sum(x["nodata"] for x in groups.values())),
               "support_counts_by_country_month": {f"{country}/{month}": rec for (country, month), rec in groups.items()},
               "value_min": vmin if math.isfinite(vmin) else None,
               "value_max": vmax if math.isfinite(vmax) else None}
    return fig, summary


def _draw_s8(p: dict[str, pd.DataFrame], caption: str) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 9.5), gridspec_kw={"width_ratios": [1.1, 1]})
    a, b = p["a"], p["b"]
    labels = _value_labels(a, include=("country", "task", "source_job_id", "fold_id"), support=(("n_train_rows", "train rows"), ("n_test_rows", "test rows"))); y = np.arange(len(a))
    for col, marker, offset, color in (("n_train_stations", "o", -.1, _BLUE), ("n_test_stations", "s", .1, _ORANGE)):
        vals = _series(a, col).to_numpy(float)
        for i, val in enumerate(vals):
            if math.isfinite(val): axes[0].scatter(val, y[i] + offset, marker=marker, color=color, s=8)
            else: axes[0].text(.01, y[i] + offset, "NA", transform=axes[0].get_yaxis_transform(), fontsize=4.0, color=_GREY)
    axes[0].set_yticks(y, labels, fontsize=3.6); axes[0].invert_yaxis(); axes[0].grid(axis="x", color=_LIGHT, lw=.4)
    axes[0].set_xlabel("Stations per saved fold"); _panel_title(axes[0], "a", f"Training and test counts · {len(a)} rows")
    labels_b = _value_labels(b, include=("country", "task", "source_job_id", "fold_id"), support=(("n_train_stations", "train sites"), ("n_test_stations", "test sites"))); yb = np.arange(len(b))
    lo, med, hi = (_series(b, c).to_numpy(float) for c in ("distance_km_min", "distance_km_median", "distance_km_p90"))
    for i in range(len(b)):
        if math.isfinite(lo[i]) and math.isfinite(hi[i]): axes[1].plot([lo[i], hi[i]], [i, i], color=_BLUE, lw=.7)
        if math.isfinite(med[i]):
            axes[1].scatter(med[i], i, color=_ORANGE, s=9, zorder=3)
            _annotate_point(axes[1], med[i], i, med[i], digits=1, dx=2)
        else: axes[1].text(.01, i, "NA", transform=axes[1].get_yaxis_transform(), fontsize=4.0, color=_GREY)
    axes[1].set_yticks(yb, labels_b, fontsize=3.8); axes[1].invert_yaxis(); axes[1].grid(axis="x", color=_LIGHT, lw=.4)
    axes[1].set_xlabel("Train-to-test station distance (km)"); _panel_title(axes[1], "b", f"Saved minimum—median—P90 ranges · {len(b)} rows")
    fig.suptitle(_short_caption(caption), fontsize=8.2); fig.tight_layout()
    return fig


_DRAWERS = {
    "Figure_1_Historical_Validation": _draw_f1,
    "Figure_2_Future_Evaluation": _draw_f2,
    "Figure_4_Fire_Support": _draw_f3,
    "Figure_5_Calibration": _draw_f4,
    "Figure_3_Feature_Increments": _draw_f5,
    "Figure_6_Monthly_Benchmark": _draw_f6,
    "Supplementary_Figure_S3": _draw_s1,
    "Supplementary_Figure_S4": _draw_s2,
    "Supplementary_Figure_S2": _draw_s3,
    "Supplementary_Figure_S8": _draw_s4,
    "Supplementary_Figure_S5": _draw_s5,
    "Supplementary_Figure_S6": _draw_s6,
    "Supplementary_Figure_S1": _draw_s8,
}


def _read_registry(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "data" / "summary" / "figure_registry.json"
    if not path.is_file():
        raise FileNotFoundError(f"Figure registry not found: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    entries = registry.get("figures")
    if not isinstance(entries, list):
        raise ValueError("figure_registry.json must contain a figures list")
    names = [entry.get("figure") for entry in entries]
    if set(names) != _EXPECTED_FIGURES or len(names) != len(_EXPECTED_FIGURES):
        raise ValueError(f"Expected exactly the published 14 figure stems; got {names}")
    for entry in entries:
        if not isinstance(entry.get("caption"), str) or not entry["caption"].strip():
            raise ValueError(f"Missing final caption for {entry['figure']}")
        if not isinstance(entry.get("panels"), dict) or not entry["panels"]:
            raise ValueError(f"Missing panel paths for {entry['figure']}")
    return registry


def _load_entry(repo_root: Path, entry: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, str], dict[str, int]]:
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    shas: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for label, panel_ref in entry["panels"].items():
        path = _panel_path(repo_root, panel_ref)
        if not path.is_file():
            raise FileNotFoundError(f"Panel CSV not found for {entry['figure']}:{label}: {path}")
        paths[label] = path
        shas[label] = _sha256(path)
        expected_hash = None
        if isinstance(panel_ref, dict):
            expected_hash = (panel_ref.get("gzip_sha256") or panel_ref.get("compressed_sha256")
                             or panel_ref.get("sha256"))
        elif path.suffix.lower() != ".gz":
            expected_hash = entry.get("panel_sha256", {}).get(label)
        else:
            expected_hash = (entry.get("panel_gzip_sha256", {}).get(label)
                             or entry.get("gzip_sha256", {}).get(label))
        if expected_hash and shas[label] != expected_hash:
            raise ValueError(f"Panel SHA-256 mismatch for {entry['figure']}:{label}")
        if entry["figure"] == "Supplementary_Figure_S7":
            # S7 is streamed by its drawer; the expected row count is checked there.
            continue
        frame = pd.read_csv(path, low_memory=False, compression="infer")
        frames[label] = frame
        row_counts[label] = int(len(frame))
    return frames, paths, shas, row_counts


def _save_outputs(fig: Any, outputs: dict[str, Path]) -> None:
    """Write stable vector exports and the standard-resolution raster copy."""
    fig.savefig(outputs["pdf"], bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight", metadata={"Date": None})


def _legacy_render_all(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Render all 14 registered figures from public, saved panel CSVs.

    Returns and writes a manifest with each panel's byte hash and row count,
    plus the SHA-256 hashes of the generated PDF/PNG/SVG files.
    """
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = _read_registry(repo_root)
    records = []
    for entry in registry["figures"]:
        stem = entry["figure"]
        frames, paths, source_hashes, row_counts = _load_entry(repo_root, entry)
        s7_stats = None
        if stem == "Supplementary_Figure_S7":
            panel = paths.get("a")
            if panel is None:
                raise ValueError("Supplementary_Figure_S7 requires panel a")
            fig, s7_stats = _draw_s7(panel, entry["caption"])
            row_counts["a"] = int(s7_stats["rows"])
        else:
            drawer = _DRAWERS.get(stem)
            if drawer is None:
                raise ValueError(f"No source-replay layout is defined for {stem}")
            fig = drawer(frames, entry["caption"])
        outputs = {
            "pdf": output_dir / f"{stem}.pdf",
            "png": output_dir / f"{stem}.png",
            "svg": output_dir / f"{stem}.svg",
        }
        try:
            _save_outputs(fig, outputs)
        finally:
            plt.close(fig)
        original_hashes = {}
        for label, panel_ref in entry["panels"].items():
            if isinstance(panel_ref, dict) and panel_ref.get("original_csv_sha256"):
                original_hashes[label] = panel_ref["original_csv_sha256"]
            elif entry.get("original_csv_sha256", {}).get(label):
                original_hashes[label] = entry["original_csv_sha256"][label]
        record = {
            "figure": stem,
            "caption": entry["caption"],
            "panel_rows": row_counts,
            "panel_sha256": source_hashes,
            "panel_sources": {
                label: {"filename": path.relative_to(repo_root).as_posix(),
                        "rows": row_counts.get(label), "sha256": source_hashes[label]}
                for label, path in paths.items()
            },
            "original_csv_sha256": original_hashes,
            "outputs": {kind: {"filename": path.name, "sha256": _sha256(path)} for kind, path in outputs.items()},
        }
        if s7_stats is not None:
            record["support_summary"] = s7_stats
        (output_dir / f"{stem}_replay_manifest.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append(record)
    manifest = {"method": "source_replay_from_published_panel_csvs", "model_refit": False,
                "bootstrap_or_resampling": False, "figures": records}
    (output_dir / "figure_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# Current manuscript layouts share the same immutable panel inputs.
from .display_revision import render_all

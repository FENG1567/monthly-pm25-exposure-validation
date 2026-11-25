#!/usr/bin/env python3
"""Redraw main Figures1–6 from their registered final panel CSVs.

This is a presentation-only renderer. It does not fit models, recalculate
estimates, create intervals, or filter source rows. Run with the repository
root (the directory containing ``data/summary/figure_registry.json``) and an
output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, PercentFormatter

try:
    from .layout_style import configure_style, style_figure
except ImportError:
    from layout_style import configure_style, style_figure


BLUE = "#3B6C8E"
ORANGE = "#D98847"
INK = "#23313A"
GREY = "#737F86"
GRID = "#DCE2E6"
CONCENTRATION_UNIT = r"µg m$^{-3}$"
COUNTRIES = ["AUS", "BD", "BRA", "CAN", "CL", "ES", "GR", "IN", "IT", "MEX", "NP", "PK", "PT", "TH", "USA", "VN", "ZA"]
TASKS = ["T1_station", "T2_spatial", "T2_spatial_buffer50km", "T3_year_2022"]
TASK_LABELS = {
    "T1_station": "T1 station",
    "T2_spatial": "T2 spatial",
    "T2_spatial_buffer50km": "T2 buffer 50 km",
    "T3_year_2022": "T3 2022",
}
TASK_TICK_LABELS = {
    "T1_station": "T1\nstation",
    "T2_spatial": "T2\nspatial",
    "T2_spatial_buffer50km": "T2 buffer\n50 km",
    "T3_year_2022": "T3\n2022",
}
PROFILE_LABELS = {"with_met": "with MET", "no_met": "without MET"}
PROFILE_COLORS = {"with_met": BLUE, "no_met": ORANGE}
FIGURE_KEYS = [
    "Figure_1_Historical_Validation",
    "Figure_2_Future_Evaluation",
    "Figure_3_Feature_Increments",
    "Figure_4_Fire_Support",
    "Figure_5_Calibration",
    "Figure_6_Monthly_Benchmark",
]
WIDTH_MM = 180.0

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.default": "regular",
    }
)
configure_style()


def _panel_header(ax: plt.Axes, label: str, title: str) -> None:
    """Place an independent large panel letter beside its compact title."""
    ax.text(
        0.0, 1.018, label, transform=ax.transAxes,
        ha="left", va="baseline", fontsize=18, fontweight="bold",
        color=INK, clip_on=False,
    )
    ax.text(
        0.145, 1.018, title, transform=ax.transAxes,
        ha="left", va="baseline", fontsize=10, fontweight="bold",
        color=INK, clip_on=False,
    )


def _rmse_detail_table(ax: plt.Axes, values: dict[str, dict[str, float]]) -> None:
    """Show the four saved Canadian 2024 RMSE values in aligned Arial columns."""
    x0, y0, width, height = 0.40, 0.375, 0.59, 0.605
    ax.add_patch(
        Rectangle(
            (x0, y0), width, height, transform=ax.transAxes,
            facecolor="white", edgecolor=GRID, linewidth=0.6,
            zorder=5, clip_on=False,
        )
    )
    ax.text(
        0.975, 0.965, f"2024 RMSE detail\n({CONCENTRATION_UNIT})",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
        color=INK, zorder=6,
    )
    for x, heading in ((0.82, "Fixed"), (0.975, "Rolling")):
        ax.text(
            x, 0.695, heading, transform=ax.transAxes,
            ha="right", va="center", fontsize=8.2, color=INK, zorder=6,
        )
    for y, profile, label in ((0.575, "with_met", "with MET"), (0.455, "no_met", "without MET")):
        ax.text(
            0.66, y, label, transform=ax.transAxes,
            ha="right", va="center", fontsize=8.2, color=INK, zorder=6,
        )
        ax.text(
            0.82, y, f"{values[profile]['fixed']:.3f}", transform=ax.transAxes,
            ha="right", va="center", fontsize=8.2, color=INK, zorder=6,
        )
        ax.text(
            0.975, y, f"{values[profile]['rolling']:.3f}", transform=ax.transAxes,
            ha="right", va="center", fontsize=8.2, color=INK, zorder=6,
        )
def _registry(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / "data" / "summary" / "figure_registry.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    figures = raw.get("figures")
    if not isinstance(figures, list):
        raise ValueError("figure_registry.json must contain a figures list")
    by_name = {str(item.get("figure")): item for item in figures if isinstance(item, dict)}
    absent = [key for key in FIGURE_KEYS if key not in by_name]
    if absent:
        raise KeyError(f"Main figures missing from registry: {absent}")
    return by_name


def _load_panels(
    repo_root: Path, entry: dict[str, Any], requested: list[str]
) -> dict[str, pd.DataFrame]:
    paths = entry.get("panels", {})
    counts = entry.get("panel_rows", {})
    frames: dict[str, pd.DataFrame] = {}
    for label in requested:
        if label not in paths:
            raise KeyError(f"{entry.get('figure')} has no registered panel {label!r}")
        source = (repo_root / str(paths[label])).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source, low_memory=False)
        expected = counts.get(label)
        if expected is not None and len(frame) != int(expected):
            raise ValueError(
                f"{entry.get('figure')} panel {label} has {len(frame)} rows; registry says {expected}"
            )
        frames[label] = frame
    return frames


def _need(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _profile(frame: pd.DataFrame, context: str) -> str:
    _need(frame, {"profile"}, context)
    values = [
        value for value in frame["profile"].astype(str).unique().tolist()
        if value not in {"nan", "None", ""}
    ]
    if len(values) != 1 or values[0] not in PROFILE_LABELS:
        raise ValueError(f"{context} must contain one recognized profile; found {values}")
    return values[0]


def _save(fig: plt.Figure, outdir: Path, stem: str, height_mm: float) -> list[str]:
    if height_mm > 198:
        raise ValueError(f"Figure {stem} exceeds the 198 mm height target")
    fig.set_size_inches(WIDTH_MM / 25.4, height_mm / 25.4, forward=True)
    style_figure(fig)
    outputs = [f"{stem}.pdf", f"{stem}.svg", f"{stem}.png", f"{stem}.tiff"]
    fig.savefig(outdir / outputs[0], facecolor="white")
    fig.savefig(outdir / outputs[1], facecolor="white")
    fig.savefig(outdir / outputs[2], dpi=300, facecolor="white")
    fig.savefig(outdir / f"{stem}.tiff", format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    try:
        from .layout_style import write_layout_qa
    except ImportError:
        from layout_style import write_layout_qa
    write_layout_qa(fig, outdir, stem)
    plt.close(fig)
    return outputs


def _panel_sources(
    repo_root: Path, entry: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> dict[str, dict[str, Any]]:
    paths = entry["panels"]
    return {
        label: {
            "source": str(paths[label]).replace("\\", "/"),
            "rows_read": int(len(frame)),
            "rows_used": int(len(frame)),
            "rows_dropped": 0,
        }
        for label, frame in frames.items()
    }


def _record(
    repo_root: Path,
    entry: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    outputs: list[str],
    height_mm: float,
    represented: dict[str, int],
    caption_notes: list[str],
) -> dict[str, Any]:
    sources = _panel_sources(repo_root, entry, frames)
    return {
        "figure": entry["figure"],
        "registry_caption_en": entry.get("caption", ""),
        "proposed_caption_edits_en": caption_notes,
        "source_panels": sources,
        "input_rows_total": int(sum(v["rows_read"] for v in sources.values())),
        "rows_dropped": 0,
        "represented_observations_by_panel": represented,
        "outputs": outputs,
        "width_mm": WIDTH_MM,
        "height_mm": height_mm,
        "png_dpi": 300,
        "vector_text": "editable text in PDF and SVG",
        "backend": "Python/matplotlib",
        "data_changes": "none; registered panel CSV rows and saved values are used as supplied",
    }


def _figure1(
    repo_root: Path, outdir: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    frames = _load_panels(repo_root, entry, ["a", "b", "c", "d"])
    mapping = {"a": ("r2", "with_met"), "b": ("r2", "no_met"), "c": ("rmse", "with_met"), "d": ("rmse", "no_met")}
    for label, (metric, expected_profile) in mapping.items():
        frame = frames[label]
        _need(frame, {"country", "task", "profile", metric, "cell_status", "eligible_folds", "ineligible_folds"}, f"F1 panel {label}")
        if _profile(frame, f"F1 panel {label}") != expected_profile:
            raise ValueError(f"F1 panel {label} is not the registered {expected_profile} profile")
        if len(frame) != len(COUNTRIES) * len(TASKS):
            raise ValueError(f"F1 panel {label} must preserve the full 17 × 4 country-task grid")
        if frame.duplicated(["country", "task"]).any():
            raise ValueError(f"F1 panel {label} has duplicate country-task cells")
        if set(frame["country"].astype(str)) != set(COUNTRIES) or set(frame["task"].astype(str)) != set(TASKS):
            raise ValueError(f"F1 panel {label} does not contain the registered 17-country × 4-task grid")

    r2_values = np.concatenate([_numeric(frames[k], "r2") for k in ("a", "b")])
    rmse_values = np.concatenate([_numeric(frames[k], "rmse") for k in ("c", "d")])
    r2_values = r2_values[np.isfinite(r2_values)]
    rmse_values = rmse_values[np.isfinite(rmse_values)]
    if not len(r2_values) or not len(rmse_values):
        raise ValueError("F1 requires finite R² and RMSE values")
    r2_limit = float(np.max(np.abs(r2_values)))
    if r2_limit == 0:
        r2_limit = 1.0
    r2_norm = SymLogNorm(linthresh=0.1, vmin=-r2_limit, vmax=r2_limit)
    rmse_max = float(np.max(rmse_values))
    rmse_norm = Normalize(vmin=0.0, vmax=rmse_max if rmse_max > 0 else 1.0)
    signed_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "signed_blue_orange", ["#4477AA", "#F7F7F5", ORANGE]
    )
    rmse_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rmse_blue", ["#F6F8F9", "#3B6C8E"]
    )
    fig = plt.figure(constrained_layout=False)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.12, 1.0, 0.12])
    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "c": fig.add_subplot(grid[2, 0]),
        "d": fig.add_subplot(grid[2, 1]),
    }
    cmap_norm = {"a": (signed_cmap, r2_norm), "b": (signed_cmap, r2_norm), "c": (rmse_cmap, rmse_norm), "d": (rmse_cmap, rmse_norm)}
    titles = {"a": "R² · with MET", "b": "R² · without MET", "c": "RMSE · with MET", "d": "RMSE · without MET"}
    for label, (metric, _) in mapping.items():
        ax = axes[label]
        frame = frames[label].set_index(["country", "task"]).reindex(
            pd.MultiIndex.from_product([COUNTRIES, TASKS], names=["country", "task"])
        )
        matrix = _numeric(frame.reset_index(), metric).reshape(len(COUNTRIES), len(TASKS))
        ineligible = pd.to_numeric(frame["ineligible_folds"], errors="coerce").fillna(0).to_numpy(dtype=float).reshape(len(COUNTRIES), len(TASKS))
        cmap, norm = cmap_norm[label]
        render_cmap = cmap.copy()
        render_cmap.set_bad((1, 1, 1, 0))
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=render_cmap, norm=norm, interpolation="none", zorder=1)
        status = frame["cell_status"].fillna("missing").astype(str).to_numpy().reshape(len(COUNTRIES), len(TASKS))
        for row in range(len(COUNTRIES)):
            for col in range(len(TASKS)):
                value = matrix[row, col]
                if not np.isfinite(value):
                    is_ineligible = "ineligible" in status[row, col].lower() or "guard" in status[row, col].lower()
                    fill = "#D9E0E4" if is_ineligible else "#F0F2F3"
                    ax.add_patch(
                        Rectangle(
                            (col - 0.5, row - 0.5), 1, 1,
                            facecolor=fill,
                            edgecolor="#96A1A7" if is_ineligible else "white",
                            linewidth=0.45,
                            hatch="///" if is_ineligible else None,
                            zorder=2,
                        )
                    )
                    label_text = "I" if is_ineligible else "NA"
                    ax.text(col, row, label_text, ha="center", va="center", color=INK, fontsize=7.2, zorder=3)
                else:
                    rgb = cmap(norm(value))[:3]
                    luminance = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
                    text_color = INK if luminance > 0.56 else "white"
                    shown = f"{value:.2f}" if metric == "r2" else f"{value:.1f}"
                    if ineligible[row, col] > 0:
                        shown += "†"
                    ax.text(col, row, shown, ha="center", va="center", color=text_color, fontsize=7.2, zorder=3)
        _panel_header(ax, label.upper(), titles[label])
        ax.set_xticks(range(len(TASKS)), [TASK_TICK_LABELS[x] for x in TASKS])
        ax.set_yticks(range(len(COUNTRIES)), COUNTRIES if label in {"a", "c"} else [""] * len(COUNTRIES))
        ax.set_xlim(-0.5, len(TASKS) - 0.5)
        ax.set_ylim(len(COUNTRIES) - 0.5, -0.5)
        ax.tick_params(length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, len(TASKS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(COUNTRIES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        if label in {"a", "c"}:
            ax.set_ylabel("Country", labelpad=4)

    cb_r2 = fig.colorbar(
        mpl.cm.ScalarMappable(norm=r2_norm, cmap=signed_cmap),
        cax=fig.add_subplot(grid[1, :]), orientation="horizontal"
    )
    cb_rmse = fig.colorbar(
        mpl.cm.ScalarMappable(norm=rmse_norm, cmap=rmse_cmap),
        cax=fig.add_subplot(grid[3, :]), orientation="horizontal"
    )
    cb_r2.ax.xaxis.set_ticks_position("bottom")
    cb_rmse.ax.xaxis.set_ticks_position("bottom")
    fig.legend(
        handles=[
            Patch(facecolor="#F0F2F3", edgecolor="#D0D6DA", label="NA · unavailable"),
            Patch(facecolor="#D9E0E4", edgecolor="#96A1A7", hatch="///", label="I · ineligible"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.94, bottom=0.07, wspace=0.08, hspace=0.30)
    stem = entry["figure"]
    height = 198.0
    outputs = _save(fig, outdir, stem, height)
    return _record(
        repo_root, entry, frames, outputs, height,
        {"R2_with_met": 68, "R2_without_met": 68, "RMSE_with_met": 68, "RMSE_without_met": 68},
        [
            "Add: ‘† marks a finite estimate with at least one ineligible fold; hatched I marks no estimate under a guard, and NA marks unavailable. R² labels show raw values on the shared signed-log color scale.’"
        ],
    )


def _legend_future() -> list[Line2D]:
    return [
        Line2D([], [], color=BLUE, marker="o", lw=0.9, label="with MET"),
        Line2D([], [], color=ORANGE, marker="s", lw=0.9, label="without MET"),
        Line2D([], [], color=INK, marker="o", markerfacecolor=INK, linestyle="-", lw=0.9, label="fixed · filled"),
        Line2D([], [], color=INK, marker="o", markerfacecolor="white", markeredgecolor=INK, linestyle="--", lw=0.9, label="rolling · open"),
    ]


def _figure2(
    repo_root: Path, outdir: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    names = ["USA_r2", "USA_rmse", "CAN_r2", "CAN_rmse"]
    frames = _load_panels(repo_root, entry, names)
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
    country_order = ["USA", "CAN"]
    metric_order = ["r2", "rmse"]
    labels = [["USA · R²", "USA · RMSE"], ["CAN · R²", "CAN · RMSE"]]
    represented: dict[str, int] = {}
    for row, country in enumerate(country_order):
        for col, metric in enumerate(metric_order):
            panel = f"{country}_{metric}"
            frame = frames[panel]
            _need(frame, {"country", "profile", "validation_mode", "external_year", metric}, f"F2 {panel}")
            if not frame["country"].astype(str).eq(country).all():
                raise ValueError(f"F2 {panel} contains a different country")
            if frame.duplicated(["country", "profile", "validation_mode", "external_year"]).any():
                raise ValueError(f"F2 {panel} has duplicate country/profile/mode/year cells")
            if "metric" in frame and not frame["metric"].astype(str).eq(metric).all():
                raise ValueError(f"F2 {panel} has inconsistent metric labels")
            ax = axes[row, col]
            years = sorted(pd.to_numeric(frame["external_year"], errors="raise").astype(int).unique().tolist())
            if not years:
                raise ValueError(f"F2 {panel} has no observed evaluation years")
            for profile in ("with_met", "no_met"):
                for mode, linestyle in (("fixed", "-"), ("rolling", "--")):
                    q = frame[(frame["profile"].astype(str) == profile) & (frame["validation_mode"].astype(str) == mode)].copy()
                    if q.empty:
                        continue
                    q["_year"] = pd.to_numeric(q["external_year"], errors="raise")
                    q = q.sort_values("_year")
                    ax.plot(
                        q["_year"], _numeric(q, metric), color=PROFILE_COLORS[profile],
                        marker="o" if profile == "with_met" else "s",
                        markerfacecolor="white" if mode == "rolling" else PROFILE_COLORS[profile],
                        markeredgecolor=PROFILE_COLORS[profile], markersize=3.6 if mode == "rolling" else 4.8,
                        linewidth=1.15, linestyle=linestyle, zorder=3,
                    )
            ax.axvline(2023, color=GREY, linewidth=0.75, linestyle=":", zorder=1)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            _panel_header(ax, "ABCD"[row * 2 + col], labels[row][col])
            ax.set_xticks(years, [str(year) for year in years])
            lo, hi = min(years), max(years)
            if country == "CAN" and metric == "rmse" and set(years) == {2023, 2024}:
                ax.set_xlim(lo - 0.25, hi + 1.35)
            else:
                ax.set_xlim((lo - 0.25, hi + 0.25) if hi > lo else (lo - 0.35, hi + 0.35))
            ax.set_xlabel("External evaluation year")
            ax.set_ylabel("R²" if metric == "r2" else f"RMSE ({CONCENTRATION_UNIT})")
            if country == "CAN" and metric == "rmse":
                detail = frame[pd.to_numeric(frame["external_year"], errors="raise").astype(int) == 2024]
                if len(detail) != 4 or detail.duplicated(["profile", "validation_mode"]).any():
                    raise ValueError("F2 CAN 2024 RMSE detail must contain four unique profile/update values")
                detail_values: dict[str, dict[str, float]] = {}
                for profile in ("with_met", "no_met"):
                    profile_values: dict[str, float] = {}
                    for mode in ("fixed", "rolling"):
                        selected = detail[
                            (detail["profile"].astype(str) == profile)
                            & (detail["validation_mode"].astype(str) == mode)
                        ]
                        if len(selected) != 1:
                            raise ValueError(f"F2 CAN 2024 RMSE missing unique {profile}/{mode} value")
                        profile_values[mode] = float(pd.to_numeric(selected["rmse"], errors="raise").iloc[0])
                    detail_values[profile] = profile_values
                _rmse_detail_table(ax, detail_values)
            represented[panel] = int(len(frame))
    fig.legend(handles=_legend_future(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.20, wspace=0.25, hspace=0.52)
    height = 124.0
    outputs = _save(fig, outdir, entry["figure"], height)
    return _record(
        repo_root, entry, frames, outputs, height, represented,
        ["Add: ‘The x-axis in each country panel lists observed years only. Fixed estimates use filled markers and solid lines; rolling estimates use open markers and dashed lines. The Canadian panel ends at 2024; its four 2024 RMSE values are listed to three decimals.’"],
    )


def _year_from_task(task: str) -> int:
    digits = "".join(ch for ch in str(task) if ch.isdigit())
    if len(digits) < 4:
        raise ValueError(f"Could not read evaluation year from task {task!r}")
    return int(digits[-4:])


def _normalize_profile_suffix(task: str) -> str:
    """Remove only a terminal MET-profile token used in some task identifiers."""
    value = str(task)
    for suffix in ("_with_met", "_no_met"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _task_display(task: str) -> str:
    value = _normalize_profile_suffix(task)
    if value in TASK_LABELS:
        return TASK_LABELS[value]
    if "rolling" in value and "2024" in value:
        return "T4 rolling 2024"
    if "external" in value:
        year = "2023" if "2023" in value else "2024" if "2024" in value else "2025" if "2025" in value else ""
        return f"T4 fixed {year}".strip()
    return value.replace("_", " ")


def _figure3(
    repo_root: Path, outdir: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    frames = _load_panels(repo_root, entry, ["a", "b"])
    a, b = frames["a"], frames["b"]
    _need(a, {"country", "profile", "task", "fire_month_scope", "full_rmse"}, "F3 panel a")
    _need(b, {"country", "profile", "rule", "row_coverage_fraction", "rule_label"}, "F3 panel b")
    if len(a) != 20 or len(b) != 12:
        raise ValueError(f"F3 must retain 20 held-out error rows and 12 support rows; got {len(a)} and {len(b)}")
    if a.duplicated(["country", "task", "profile", "fire_month_scope"]).any():
        raise ValueError("F3 panel a has duplicate country-year/profile/month-scope rows")
    if b.duplicated(["country", "task", "profile", "rule"]).any():
        raise ValueError("F3 panel b has duplicate country/profile/rule rows")
    if set(a["fire_month_scope"].astype(str)) != {"P90_fire_proxy", "other_months"}:
        raise ValueError("F3 panel a must preserve both P90 fire-proxy and other-month strata")
    expected_rules = {"all", "distance_50km_non_geo_minmax", "distance_50km_non_geo_p01_p99"}
    if set(b["rule"].astype(str)) != expected_rules:
        raise ValueError("F3 panel b does not contain all three registered support rules")

    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1.0]})
    country_order = ["USA", "CAN"]
    year_order = sorted({_year_from_task(task) for task in a["task"].astype(str)})
    groups = [(country, year) for country in country_order for year in year_order if ((a.country.astype(str) == country) & a.task.astype(str).map(_year_from_task).eq(year)).any()]
    ymap = {key: i for i, key in enumerate(groups)}
    scope_styles = {"P90_fire_proxy": ("o", -0.055), "other_months": ("s", 0.055)}
    profile_offsets = {"with_met": -0.16, "no_met": 0.16}
    ax = axes[0]
    for (country, task, profile), q in a.groupby(["country", "task", "profile"], sort=False):
        year = _year_from_task(str(task))
        group_y = ymap[(str(country), year)] + profile_offsets[str(profile)]
        for _, record in q.iterrows():
            scope = str(record["fire_month_scope"])
            marker, scope_jitter = scope_styles[scope]
            ax.scatter(
                float(record["full_rmse"]), group_y + scope_jitter,
                color=PROFILE_COLORS[str(profile)], marker=marker, s=30,
                edgecolors="white", linewidths=0.45, zorder=3,
            )
    ax.set_yticks(range(len(groups)), [f"{country} · {year}" for country, year in groups])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel(f"Full-model RMSE ({CONCENTRATION_UNIT})")
    _panel_header(ax, "A", "Held-out error by country and year")

    rule_styles = {
        "distance_50km_non_geo_minmax": ("o", -0.10, "50 km + non-geographic min–max"),
        "distance_50km_non_geo_p01_p99": ("s", 0.0, "50 km + non-geographic P01–P99"),
        "all": ("x", 0.10, "All eligible months"),
    }
    ax = axes[1]
    for _, record in b.iterrows():
        country = str(record["country"])
        profile = str(record["profile"])
        rule = str(record["rule"])
        marker, rule_jitter, _ = rule_styles[rule]
        group_y = country_order.index(country) + profile_offsets[profile] + rule_jitter
        scatter_kwargs = {
            "color": PROFILE_COLORS[profile], "marker": marker, "s": 31,
            "linewidths": 0.45, "zorder": 3,
        }
        if marker != "x":
            scatter_kwargs["edgecolors"] = "white"
        ax.scatter(float(record["row_coverage_fraction"]), group_y, **scatter_kwargs)
    ax.set_yticks(range(len(country_order)), country_order)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.23)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("Retained / eligible station-months")
    _panel_header(ax, "B", "2023 P90-month support")

    legend_handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="none", label="with MET"),
        Line2D([], [], marker="o", color=ORANGE, linestyle="none", label="without MET"),
        Line2D([], [], marker="o", color=INK, linestyle="none", label="P90 fire-proxy months"),
        Line2D([], [], marker="s", color=INK, linestyle="none", label="Other months"),
        Line2D([], [], marker="o", color=INK, linestyle="none", label="50 km + min–max rule"),
        Line2D([], [], marker="s", color=INK, linestyle="none", label="50 km + P01–P99 rule"),
        Line2D([], [], marker="x", color=INK, linestyle="none", label="All eligible months"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), columnspacing=1.0, handletextpad=0.35)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.91, bottom=0.20, wspace=0.22)
    height = 120.0
    outputs = _save(fig, outdir, entry["figure"], height)
    return _record(
        repo_root, entry, frames, outputs, height,
        {"a": 20, "b": 12},
        ["Add: ‘Panel a groups estimates by country-year, with vertical offsets for MET profile and marker shape for P90 fire-proxy versus other months. Panel b groups by country, offsets MET profiles, and uses marker shape for the support rule.’"],
    )


def _figure4(
    repo_root: Path, outdir: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    frames = _load_panels(repo_root, entry, ["a", "b", "c"])
    value_cols = {"a": "pointwise_coverage", "b": "all_month_station_coverage", "c": "row_weighted_mean_interval_width"}
    keys = ["country", "task", "profile"]
    key_sets: list[set[tuple[str, str, str]]] = []
    expected_folds = {"T1_station": 5, "T2_spatial": 5, "T2_spatial_buffer50km": 5, "T3_year_2022": 1}
    for label, frame in frames.items():
        _need(frame, set(keys + [value_cols[label], "fold_n"]), f"F4 panel {label}")
        if len(frame) != 16 or frame.duplicated(keys).any():
            raise ValueError(f"F4 panel {label} must preserve 16 unique country-task-profile rows")
        observed_folds = pd.to_numeric(frame["fold_n"], errors="raise").astype(int)
        for task, expected in expected_folds.items():
            selected = observed_folds[frame["task"].astype(str) == task]
            if len(selected) and not selected.eq(expected).all():
                raise ValueError(f"F4 panel {label} has unexpected source fold counts for {task}")
        key_sets.append(set(tuple(str(row[k]) for k in keys) for _, row in frame.iterrows()))
    if any(found != key_sets[0] for found in key_sets[1:]):
        raise ValueError("F4 panels do not contain matching country-task-profile rows")
    if len({key[2] for key in key_sets[0]}) != 2:
        raise ValueError("F4 should contain both MET profiles")

    countries = [country for country in ["USA", "CAN"] if any(key[0] == country for key in key_sets[0])]
    tasks = [task for task in TASKS if any(key[1] == task for key in key_sets[0])]
    groups = [(country, task) for country in countries for task in tasks if any(key[:2] == (country, task) for key in key_sets[0])]
    ymap = {key: i for i, key in enumerate(groups)}
    offsets = {"with_met": -0.17, "no_met": 0.17}
    fig, axes = plt.subplots(1, 3, sharey=True)
    titles = {"a": "Station-month coverage", "b": "Whole-station coverage", "c": "Mean interval width"}
    for label, frame in frames.items():
        ax = axes[ord(label) - ord("a")]
        col = value_cols[label]
        for _, record in frame.iterrows():
            country, task, profile = (str(record[k]) for k in keys)
            value = pd.to_numeric(pd.Series([record[col]]), errors="coerce").iloc[0]
            y = ymap[(country, task)] + offsets[profile]
            if not np.isfinite(value):
                ax.text(0.01, y, "NA", transform=ax.get_yaxis_transform(), va="center", color=GREY, fontsize=7.0)
                continue
            ax.scatter(value, y, s=27, color=PROFILE_COLORS[profile], marker="o", zorder=3)
            text = f"{value:.3f}" if label in {"a", "b"} else f"{value:.1f}"
            ax.annotate(
                text, (value, y), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6.9, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            )
        _panel_header(ax, label.upper(), titles[label])
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlabel("Coverage fraction" if label in {"a", "b"} else f"Interval width ({CONCENTRATION_UNIT})")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if label in {"a", "b"}:
            ax.axvline(0.9, color=GREY, linewidth=0.8, linestyle=":", zorder=1)
            vals = _numeric(frame, col)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lower = max(0.0, float(np.floor((float(vals.min()) - 0.08) * 10.0) / 10.0))
                upper = min(1.25, max(1.15, float(vals.max()) + 0.16))
                ax.set_xlim(lower, upper)
                step = 0.1 if upper - lower < 0.45 else 0.25
                start = int(np.ceil(lower / step))
                stop = int(np.floor(1.0 / step))
                ax.set_xticks([round(tick * step, 3) for tick in range(start, stop + 1)])
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        else:
            vals = _numeric(frame, col)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                pad = max(0.5, (float(vals.max()) - float(vals.min())) * 0.12)
                ax.set_xlim(float(vals.min()) - pad, float(vals.max()) + pad * 2.5)
        if label == "a":
            ax.set_yticks(range(len(groups)), [f"{country} · {TASK_LABELS.get(task, task)}" for country, task in groups])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="none", label="with MET"),
            Line2D([], [], marker="o", color=ORANGE, linestyle="none", label="without MET"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.20, right=0.995, top=0.92, bottom=0.22, wspace=0.30)
    height = 122.0
    outputs = _save(fig, outdir, entry["figure"], height)
    record = _record(
        repo_root, entry, frames, outputs, height,
        {label: 16 for label in frames},
        ["Add: ‘Rows group country-task strata; vertical offsets distinguish with-MET and no-MET profiles (blue and orange). Dotted lines mark nominal 90% coverage.’"],
    )
    record["source_fold_counts_by_task"] = expected_folds
    return record


def _figure5(
    repo_root: Path, outdir: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    frames = _load_panels(repo_root, entry, ["a", "b", "c"])
    contrasts = ["B+AOD - B", "full - B+AOD", "full - B"]
    keys = ["country", "task", "profile"]
    all_lows: list[np.ndarray] = []
    all_highs: list[np.ndarray] = []
    groups: list[tuple[str, str]] | None = None
    expected_counts = [24, 12, 26]
    counts: list[int] = []
    for i, label in enumerate(["a", "b", "c"]):
        frame = frames[label]
        _need(frame, set(keys + ["contrast", "delta_rmse", "simultaneous_95_low", "simultaneous_95_high"]), f"F5 panel {label}")
        if len(frame) != 26 or frame.duplicated(keys).any():
            raise ValueError(f"F5 panel {label} must preserve 26 unique country-task-profile contrasts")
        if not frame["contrast"].astype(str).eq(contrasts[i]).all():
            raise ValueError(f"F5 panel {label} does not match contrast {contrasts[i]!r}")
        found_groups = list(
            dict.fromkeys(
                (str(row.country), _normalize_profile_suffix(str(row.task)))
                for row in frame.itertuples()
            )
        )
        if len(found_groups) != 13:
            raise ValueError(f"F5 panel {label} must contain 13 country-task groups")
        normalized_keys = [
            (str(row.country), _normalize_profile_suffix(str(row.task)), str(row.profile))
            for row in frame.itertuples()
        ]
        if len(set(normalized_keys)) != 26:
            raise ValueError(f"F5 panel {label} does not contain 26 distinct normalized group/profile rows")
        if groups is None:
            groups = found_groups
        elif set(found_groups) != set(groups):
            raise ValueError("F5 panels do not contain the same 13 country-task groups")
        low, high, delta = (_numeric(frame, col) for col in ("simultaneous_95_low", "simultaneous_95_high", "delta_rmse"))
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.isfinite(delta).all()):
            raise ValueError(f"F5 panel {label} has missing simultaneous intervals or estimates")
        if np.any(low > delta) or np.any(delta > high):
            raise ValueError(f"F5 panel {label} has a point estimate outside its simultaneous interval")
        all_lows.append(low)
        all_highs.append(high)
        counts.append(int(((low > 0) | (high < 0)).sum()))
    if counts != expected_counts:
        raise ValueError(f"F5 family interval counts changed: expected {expected_counts}, found {counts}")
    if any(set(frame["profile"].astype(str)) != {"with_met", "no_met"} for frame in frames.values()):
        raise ValueError("F5 each country-task group must retain both MET profiles")

    assert groups is not None
    ymap = {key: i for i, key in enumerate(groups)}
    lows = np.concatenate(all_lows)
    highs = np.concatenate(all_highs)
    span = float(highs.max() - lows.min())
    pad = max(0.15, span * 0.045)
    xlim = (float(lows.min()) - pad, float(highs.max()) + pad)
    fig, axes = plt.subplots(1, 3, sharey=True)
    for i, label in enumerate(["a", "b", "c"]):
        ax = axes[i]
        frame = frames[label]
        for _, record in frame.iterrows():
            key = (str(record["country"]), _normalize_profile_suffix(str(record["task"])))
            profile = str(record["profile"])
            y = ymap[key] + (-0.18 if profile == "with_met" else 0.18)
            color = PROFILE_COLORS[profile]
            low = float(record["simultaneous_95_low"])
            high = float(record["simultaneous_95_high"])
            ax.hlines(y, low, high, color=color, linewidth=1.15, zorder=2)
            ax.vlines([low, high], y - 0.055, y + 0.055, color=color, linewidth=1.0, zorder=2)
            ax.scatter(float(record["delta_rmse"]), y, color=color, marker="o", s=13, zorder=3)
        ax.axvline(0, color=INK, linewidth=0.85, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlim(xlim)
        panel_title = ["B+AOD − B", "Full − (B+AOD)", "Full − B"][i]
        _panel_header(ax, label.upper(), panel_title)
        ax.set_xlabel(f"ΔRMSE ({CONCENTRATION_UNIT})")
        if label == "a":
            ax.set_yticks(range(len(groups)), [f"{country} · {_task_display(task)}" for country, task in groups])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="none", label="with MET"),
            Line2D([], [], marker="o", color=ORANGE, linestyle="none", label="without MET"),
            Line2D([], [], color=INK, linewidth=1.1, label="simultaneous 95% max-|t| CI"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.24, right=0.995, top=0.92, bottom=0.20, wspace=0.22)
    height = 140.0
    outputs = _save(fig, outdir, entry["figure"], height)
    return _record(
        repo_root, entry, frames, outputs, height,
        {label: 26 for label in frames},
        ["No statistical-wording change. Preserve the registry wording: ‘simultaneous 95% max-|t| intervals across all 78 contrasts (three contrasts × 26 units), using 10,000 prespecified Exp(1) station-multiplier draws.’ Add only the display key: ‘Rows group 13 country-task units; vertical offsets and color distinguish MET profiles. The 24/26, 12/26 and 26/26 intervals exclude zero, respectively.’"],
    )


def _figure6(
    repo_root: Path, outdir: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    frames = _load_panels(repo_root, entry, ["a", "b"])
    a, b = frames["a"], frames["b"]
    keys = ["country", "external_year", "validation_mode", "profile"]
    required_a = set(keys + ["full_rmse_ug_m3", "comparator_rmse_ug_m3", "comparison"])
    required_b = set(keys + ["delta", "ci_low", "ci_high", "metric", "contrast", "ci_status", "bootstrap_replicates", "bootstrap_seed"])
    _need(a, required_a, "F6 panel a")
    _need(b, required_b, "F6 panel b")
    if len(a) != 12 or len(b) != 12 or a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("F6 must preserve 12 unique same-support cells in both registered panels")
    def key_tuple(row: pd.Series) -> tuple[str, ...]:
        return tuple(
            str(int(float(row[k]))) if k == "external_year" else str(row[k])
            for k in keys
        )

    key_a = {key_tuple(row) for _, row in a.iterrows()}
    key_b = {key_tuple(row) for _, row in b.iterrows()}
    if key_a != key_b:
        raise ValueError("F6 panel a and b do not contain the same 12 country/year/mode/profile cells")
    if not b["ci_status"].astype(str).eq("ci_computed").all() or not pd.to_numeric(b["bootstrap_replicates"], errors="raise").eq(500).all() or not pd.to_numeric(b["bootstrap_seed"], errors="raise").eq(20261116).all():
        raise ValueError("F6 bootstrap interval metadata differs from the registered 500-resample source")
    if not b["metric"].astype(str).eq("rmse").all() or not b["contrast"].astype(str).eq("full minus comparator").all():
        raise ValueError("F6 panel b is not the registered paired RMSE contrast")
    if set(pd.to_numeric(a["external_year"], errors="raise").astype(int)) != {2023, 2024}:
        raise ValueError("F6 must contain only the registered 2023–2024 comparisons")
    if not a["comparison"].astype(str).eq("SatPM_V6GL03_monthly").all():
        raise ValueError("F6 panel a comparator is not native monthly SatPM V6GL03")
    paired = a[keys].merge(b[keys + ["delta"]], on=keys, how="left", validate="one_to_one", sort=False)
    xdiff = _numeric(a, "full_rmse_ug_m3") - _numeric(a, "comparator_rmse_ug_m3")
    if not np.allclose(xdiff, _numeric(paired, "delta"), atol=1e-8, rtol=0):
        raise ValueError("F6 saved paired deltas do not match full minus comparator RMSE")

    groups = list(dict.fromkeys((str(row.country), int(row.external_year), str(row.validation_mode)) for row in a.itertuples()))
    group_order = [(country, year, mode) for country in ["USA", "CAN"] for year in [2023, 2024] for mode in ["fixed", "rolling"] if (country, year, mode) in groups]
    ymap = {key: i for i, key in enumerate(group_order)}
    offsets = {"with_met": -0.17, "no_met": 0.17}
    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.0, 1.08]})
    b_lookup = {key_tuple(row): row for _, row in b.iterrows()}
    for _, record in a.iterrows():
        key = key_tuple(record)
        country, year, mode, profile = key
        y = ymap[(country, int(year), mode)] + offsets[profile]
        color = PROFILE_COLORS[profile]
        full = float(record["full_rmse_ug_m3"])
        comparator = float(record["comparator_rmse_ug_m3"])
        axes[0].plot([full, comparator], [y, y], color=GRID, linewidth=1.1, zorder=1)
        axes[0].scatter(full, y - 0.035, color=color, marker="o", s=29, zorder=3)
        axes[0].scatter(comparator, y + 0.035, facecolors="white", edgecolors=color, marker="s", s=31, linewidths=1.0, zorder=3)
        interval = b_lookup[key]
        low, high, delta = (float(interval[col]) for col in ("ci_low", "ci_high", "delta"))
        axes[1].hlines(y, low, high, color=color, linewidth=1.2, zorder=2)
        axes[1].vlines([low, high], y - 0.06, y + 0.06, color=color, linewidth=1.0, zorder=2)
        axes[1].scatter(delta, y, color=color, marker="o", s=13, zorder=3)
    labels = [f"{country} · {year} {mode}" for country, year, mode in group_order]
    axes[0].set_yticks(range(len(group_order)), labels)
    axes[0].invert_yaxis()
    _panel_header(axes[0], "A", "Matched-sample RMSE")
    axes[0].set_xlabel(f"RMSE ({CONCENTRATION_UNIT})")
    axes[0].grid(axis="x", color=GRID, linewidth=0.6)
    axes[1].axvline(0, color=INK, linewidth=0.85, zorder=1)
    axes[1].grid(axis="x", color=GRID, linewidth=0.6)
    _panel_header(axes[1], "B", "Paired ΔRMSE · 95% CI")
    axes[1].set_xlabel(f"ΔRMSE: full − SatPM ({CONCENTRATION_UNIT})")
    profile_handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="none", label="with MET"),
        Line2D([], [], marker="o", color=ORANGE, linestyle="none", label="without MET"),
    ]
    mark_handles = [
        Line2D([], [], marker="o", color=INK, linestyle="none", label="Full model"),
        Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=INK, linestyle="none", label="SatPM V6.GL.03"),
    ]
    fig.legend(handles=profile_handles + mark_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.23, right=0.99, top=0.92, bottom=0.20, wspace=0.25)
    height = 132.0
    outputs = _save(fig, outdir, entry["figure"], height)
    return _record(
        repo_root, entry, frames, outputs, height,
        {"a": 12, "b": 12},
        ["Add: ‘Rows group country-year-update mode; vertical offsets and color distinguish MET profiles. The comparison includes fixed 2023, fixed 2024 and rolling 2024 cells only.’"],
    )


def render(repo_root: str | Path, outdir: str | Path) -> dict[str, Any]:
    """Render the registered main Figures1–6 and return retention metadata."""
    repo_root = Path(repo_root).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    registry = _registry(repo_root)
    renderers = {
        "Figure_1_Historical_Validation": _figure1,
        "Figure_2_Future_Evaluation": _figure2,
        "Figure_3_Feature_Increments": _figure5,
        "Figure_4_Fire_Support": _figure3,
        "Figure_5_Calibration": _figure4,
        "Figure_6_Monthly_Benchmark": _figure6,
    }
    figures = [renderers[key](repo_root, outdir, registry[key]) for key in FIGURE_KEYS]
    return {
        "figure_count": len(figures),
        "backend": "Python/matplotlib",
        "output_directory": str(outdir),
        "width_mm": WIDTH_MM,
        "height_limit_mm": 210,
        "source_data_policy": "use registered exact panel CSVs; no row filtering or estimate changes",
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.repo_root, args.outdir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Redraw supplementary figures S1-S4 from the registered source panels.

The source CSVs remain unchanged. Each input record is represented directly in
the corresponding figure; no model fitting, pooling, interval estimation, or
interpolation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D

try:
    from .layout_style import configure_style, style_figure
except ImportError:
    try:
        from layout_style import configure_style, style_figure
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from layout_style import configure_style, style_figure


FIGURE_WIDTH_MM = 180.0
PT_PER_IN = 72.0
MM_PER_IN = 25.4
BLUE = "#2878B5"  # with-MET profile
ORANGE = "#D98232"  # no-MET profile
INK = "#20252B"
GREY = "#69727D"
LIGHT = "#D9DEE4"
PROFILE_ORDER = ("no_met", "with_met")
PROFILE_LABEL = {"no_met": "no MET", "with_met": "with MET"}
PROFILE_COLOR = {"no_met": ORANGE, "with_met": BLUE}


CAPTIONS = {
    "Supplementary_Figure_S3": (
        "Supplementary Figure S3 | Historical T1 station-grouped sensitivity. "
        "Heatmaps show every saved country-level signed R² estimate. The upper "
        "row separates strict/sensitivity QA and HGB/Ridge branches under "
        "log-smear with smearing; the lower row separates the strict-HGB "
        "target/retransformation branches. Columns separate no-MET and with-MET "
        "profiles. Cell labels show signed estimates and color uses a symmetric "
        "log scale. Portugal is labelled PT. No interval or new inference is added."
    ),
    "Supplementary_Figure_S4": (
        "Supplementary Figure S4 | Transfer and monitoring-network sensitivity. "
        "(a) All 34 strict-QA, no-MET full-arm leave-one-country-out R² estimates "
        "for 17 held-out countries under HGB and Ridge; the signed-log axis retains "
        "negative values. (b,c) All saved 2022 RMSE observations across the three "
        "fixed thinning seeds, shown separately for the United States and Canada "
        "and by MET profile. Marker shape identifies seed. Open diamonds show the "
        "shared full-network endpoints once per country/profile. Only observed "
        "sample sizes are plotted; no connecting curves or interpolation are used. "
        "These are descriptive sensitivity runs, not confidence intervals."
    ),
    "Supplementary_Figure_S2": (
        "Supplementary Figure S2 | Country-specific fold-level calibration detail. "
        "(a,b) Heatmaps show each of the 32 primary 80/20 folds for the United States "
        "and Canada. Row labels identify task, MET profile and fold; columns show "
        "pointwise and whole-station coverage. Cell values are saved coverage "
        "fractions, with 0.90 marked in the shared scale. (c) The eight Brazil "
        "external-2023 feasibility-only q90 thresholds are shown separately by source "
        "class and MET profile. The complete 440-row interval source remains with "
        "the analysis; records without sufficient task metadata are not plotted in "
        "these summaries. T1 is station grouping, T2 spatial blocking, T2b the "
        "50-km buffer, and T3/22 the 2022 task. Primary folds and feasibility "
        "thresholds are not pooled, and no interval is added."
    ),
    "Supplementary_Figure_S8": (
        "Supplementary Figure S8 | Annual SatPM comparator contrasts on primary "
        "exact-12-month support. The 24 plotted estimates use the primary "
        "strict-HGB/log-smear/smearing full-model specification. Each country panel "
        "shows six tasks, with paired no-MET and with-MET estimates: T1 station, "
        "T2 spatial, T2 50-km buffer, T3 2022, and fixed 2023 and 2024 external "
        "tasks. Points and horizontal intervals show the saved annual estimates and "
        "their source intervals. For ΔRMSE = full − SatPM, positive values favor "
        "SatPM and negative values favor the full model. Each comparison uses 12 "
        "exact matched months. Annual raw-CAMS comparisons are not shown; the fitted-"
        "CAMS comparison is reported in main Table 6. No new inference is added."
    ),
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    configure_style()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repository(repo_root: Path) -> tuple[Path, Path]:
    """Return (repository root, data/summary root) for supported task layouts."""
    repo_root = repo_root.expanduser().resolve()
    candidates = [repo_root, repo_root / "input" / "github", repo_root / "github"]
    if repo_root.name == "summary" and (repo_root / "figure_registry.json").is_file():
        return repo_root.parent.parent, repo_root
    for candidate in candidates:
        summary = candidate / "data" / "summary"
        if (summary / "figure_registry.json").is_file():
            return candidate, summary
    raise FileNotFoundError(
        "Could not find data/summary/figure_registry.json under repo_root; "
        "pass the repository root or its input/github directory."
    )


def _entry_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = registry.get("figures")
    if not isinstance(entries, list):
        raise ValueError("figure_registry.json must contain a figures list")
    mapped = {str(entry.get("figure")): entry for entry in entries if isinstance(entry, dict)}
    required = {f"Supplementary_Figure_S{i}" for i in range(1, 5)}
    missing = required - set(mapped)
    if missing:
        raise ValueError(f"Registry is missing requested supplementary figures: {sorted(missing)}")
    return mapped


def _panel_path(repo: Path, panel_ref: Any) -> Path:
    if isinstance(panel_ref, dict):
        panel_ref = panel_ref.get("path") or panel_ref.get("file")
    if not isinstance(panel_ref, str) or not panel_ref.strip():
        raise ValueError(f"Invalid panel reference: {panel_ref!r}")
    path = Path(panel_ref)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _read_panel(repo: Path, entry: dict[str, Any], label: str) -> tuple[pd.DataFrame, Path]:
    panel_ref = entry.get("panels", {}).get(label)
    if panel_ref is None:
        raise ValueError(f"{entry.get('figure')} has no registered panel {label!r}")
    path = _panel_path(repo, panel_ref)
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered source panel: {path}")
    frame = pd.read_csv(path, low_memory=False, compression="infer")
    expected = entry.get("panel_rows", {}).get(label)
    if expected is not None and len(frame) != int(expected):
        raise ValueError(
            f"{entry.get('figure')} panel {label} has {len(frame)} rows; "
            f"registry records {expected}."
        )
    return frame, path


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not find {role} column; tried {candidates}")


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _profile(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]", "", _clean_text(value).lower())
    if text in {"nom et", "nomet", "nometeorology", "nometeo"}:
        return "no_met"
    if text in {"withmet", "withmeteorology", "withmeteo"}:
        return "with_met"
    text = text.replace(" ", "")
    if "no" in text and "met" in text:
        return "no_met"
    if "with" in text and "met" in text:
        return "with_met"
    raise ValueError(f"Unrecognized MET profile value: {value!r}")


def _task_label(task: Any) -> str:
    text = _clean_text(task)
    if text == "T1_station":
        return "T1"
    if text == "T2_spatial_buffer50km":
        return "T2b"
    if text == "T2_spatial":
        return "T2"
    if text == "T3_year_2022":
        return "T3/22"
    if "2023" in text:
        return "T4/23"
    if "2024" in text:
        return "T4/24"
    return text


def _s4_task_base(task: Any) -> str:
    """Remove only the redundant profile suffix used in the annual source labels."""
    return re.sub(r"_(?:no_met|with_met)$", "", _clean_text(task), flags=re.IGNORECASE)


def _task_sort_key(task: Any) -> tuple[int, str]:
    text = _clean_text(task)
    if text == "T1_station":
        return (0, text)
    if text == "T2_spatial":
        return (1, text)
    if text == "T2_spatial_buffer50km":
        return (2, text)
    if text == "T3_year_2022":
        return (3, text)
    if "2023" in text:
        return (4, text)
    if "2024" in text:
        return (5, text)
    return (9, text)


def _country_column(frame: pd.DataFrame) -> str:
    return _first_column(
        frame,
        ("country", "normalized_country_for_grouping", "mapped_country"),
        "country",
    )


def _profile_column(frame: pd.DataFrame) -> str:
    return _first_column(
        frame,
        ("profile", "normalized_profile_for_grouping", "mapped_profile", "profile_label"),
        "profile",
    )


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.12,
        1.025,
        label.upper(),
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> dict[str, dict[str, Any]]:
    style_figure(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", metadata={"Creator": "redraw_supp_first.py"})
    fig.savefig(output_dir / f"{stem}.svg", format="svg", metadata={"Date": None, "Creator": "redraw_supp_first.py"})
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300)
    fig.savefig(output_dir / f"{stem}.tiff", format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    for suffix in ("pdf", "svg", "png", "tiff"):
        path = output_dir / f"{stem}.{suffix}"
        outputs[suffix] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    try:
        from .layout_style import write_layout_qa
    except ImportError:
        from layout_style import write_layout_qa
    write_layout_qa(fig, output_dir, stem)
    plt.close(fig)
    return outputs


def _source_record(
    figure: str,
    panel: str,
    frame: pd.DataFrame,
    path: Path,
    *,
    rendered_rows: int | None = None,
    rendered_values: int | None = None,
) -> dict[str, Any]:
    return {
        "panel": panel,
        "source_file": path.name,
        "source_sha256": _sha256(path),
        "source_rows": int(len(frame)),
        "rendered_source_rows": int(len(frame) if rendered_rows is None else rendered_rows),
        "rendered_values": int(len(frame) if rendered_values is None else rendered_values),
        "excluded_source_rows": 0,
    }


def _branch_sort_value(value: Any, order: tuple[str, ...]) -> tuple[int, str]:
    text = _clean_text(value)
    try:
        return (order.index(text.lower()), text.lower())
    except ValueError:
        return (len(order), text.lower())


def _s1_branch_table(
    frame: pd.DataFrame,
    *,
    country_col: str,
    profile_col: str,
    branch_cols: tuple[str, ...],
    branch_formatter: Any,
    countries: list[str],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    work = frame.copy()
    work["_profile"] = work[profile_col].map(_profile)
    work["_country"] = work[country_col].map(_clean_text)
    if "r2" not in work.columns:
        raise ValueError("S1 source panels must contain the raw signed r2 field")
    work["_r2"] = pd.to_numeric(work["r2"], errors="coerce")
    missing_branch_cols = set(branch_cols) - set(work.columns)
    if missing_branch_cols:
        raise ValueError(f"S1 source panel is missing branch fields: {sorted(missing_branch_cols)}")
    for column in branch_cols:
        work[f"_{column}"] = work[column].map(_clean_text)
    result: dict[str, tuple[np.ndarray, list[str]]] = {}
    for profile in PROFILE_ORDER:
        subset = work[work["_profile"] == profile].copy()
        tuples = list(
            subset[list(f"_{column}" for column in branch_cols)]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        tuples.sort(
            key=lambda row: tuple(
                _branch_sort_value(value, ("strict", "sensitivity", "hgb", "ridge"))
                for value in row
            )
        )
        labels = [branch_formatter(row) for row in tuples]
        if len(labels) != len(set(labels)):
            raise ValueError(f"S1 branch labels are not unique for profile {profile}")
        subset["_branch"] = subset.apply(
            lambda row: branch_formatter(tuple(row[f"_{column}"] for column in branch_cols)),
            axis=1,
        )
        if subset.duplicated(["_country", "_branch"]).any():
            duplicates = subset.loc[
                subset.duplicated(["_country", "_branch"], keep=False),
                ["_country", "_branch", *branch_cols],
            ]
            raise ValueError(
                f"S1 source contains multiple estimates for a country/branch cell "
                f"in {profile}: {duplicates.head(6).to_dict(orient='records')}"
            )
        matrix = (
            subset.pivot(index="_country", columns="_branch", values="_r2")
            .reindex(index=countries, columns=labels)
            .to_numpy(dtype=float)
        )
        result[profile] = (matrix, labels)
    return result


def _s1_figure(
    frames: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    a, b = frames["a"], frames["b"]
    ca, cb = _country_column(a), _country_column(b)
    pa, pb = _profile_column(a), _profile_column(b)
    qa = _first_column(a, ("qa_variant", "qa_label"), "QA variant")
    est_a = _first_column(a, ("estimator_kind", "estimator_label"), "estimator")
    target = _first_column(b, ("target_strategy",), "target strategy")
    backtransform = _first_column(b, ("backtransform",), "backtransform")
    qa_b = _first_column(b, ("qa_variant", "qa_label"), "QA variant")
    est_b = _first_column(b, ("estimator_kind", "estimator_label"), "estimator")

    if set(a[qa].dropna().astype(str).str.lower()) != {"strict", "sensitivity"}:
        raise ValueError("S1a must retain strict and sensitivity QA branches")
    if set(a[est_a].dropna().astype(str).str.lower()) != {"hgb", "ridge"}:
        raise ValueError("S1a must retain HGB and Ridge branches")
    if set(b[qa_b].dropna().astype(str).str.lower()) != {"strict"}:
        raise ValueError("S1b contains an unexpected QA branch; no rows were filtered")
    if set(b[est_b].dropna().astype(str).str.lower()) != {"hgb"}:
        raise ValueError("S1b contains an unexpected estimator branch; no rows were filtered")
    if target in a.columns and a[target].nunique(dropna=False) > 1:
        raise ValueError("S1a mixes target-strategy branches outside its registered QA/estimator layout")
    if b[target].nunique(dropna=False) > 3:
        raise ValueError("S1b has more than three target strategies; review the registered layout")

    country_values = sorted(set(a[ca].dropna().map(_clean_text)) | set(b[cb].dropna().map(_clean_text)))
    branch_a = _s1_branch_table(
        a,
        country_col=ca,
        profile_col=pa,
        branch_cols=(qa, est_a),
        branch_formatter=lambda row: (
            f"{('sens.' if row[0].lower() == 'sensitivity' else row[0].lower())}\n"
            f"{row[1].upper() if row[1].lower() == 'hgb' else 'Ridge'}"
        ),
        countries=country_values,
    )
    branch_b = _s1_branch_table(
        b,
        country_col=cb,
        profile_col=pb,
        branch_cols=(target, backtransform),
        branch_formatter=lambda row: (
            f"{row[0].replace('_', '-')}\n"
            f"{ {'smearing': 'smear', 'original': 'original', 'model_native': 'native'}.get(row[1], row[1]) }"
        ),
        countries=country_values,
    )

    finite_values = np.concatenate(
        [
            np.r_[branch_a[p][0].ravel(), branch_b[p][0].ravel()]
            for p in PROFILE_ORDER
        ]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if not finite_values.size:
        raise ValueError("S1 source panels contain no finite R² values")
    max_abs = max(0.1, float(np.max(np.abs(finite_values))))
    norm = SymLogNorm(linthresh=0.1, linscale=0.8, vmin=-max_abs, vmax=max_abs, base=10)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E7E9EC")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH_MM / MM_PER_IN, 5.45),
        sharey="row",
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    panel_specs = [
        (axes[0, 0], "a", "AOD screening × estimator · no MET", branch_a["no_met"][0], branch_a["no_met"][1], True),
        (axes[0, 1], "b", "AOD screening × estimator · with MET", branch_a["with_met"][0], branch_a["with_met"][1], False),
        (axes[1, 0], "c", "Target / retransformation · no MET", branch_b["no_met"][0], branch_b["no_met"][1], True),
        (axes[1, 1], "d", "Target / retransformation · with MET", branch_b["with_met"][0], branch_b["with_met"][1], False),
    ]
    for ax, panel, title, values, labels, show_y in panel_specs:
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(title, loc="left", pad=4, fontsize=9, color=PROFILE_COLOR.get("no_met" if "no MET" in title else "with_met", INK), fontweight="bold")
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7.2)
        ax.set_yticks(np.arange(len(country_values)), country_values, fontsize=7.5)
        if not show_y:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(country_values), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        _panel_label(ax, panel)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    rgba = cmap(norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    text_color = INK if luminance > 0.55 else "white"
                    label_text = f"{value:.2f}"
                else:
                    text_color = GREY
                    label_text = "—"
                ax.text(column_index, row_index, label_text, ha="center", va="center", fontsize=8.2, color=text_color)
    axes[0, 0].set_ylabel("Country", fontsize=8)
    axes[1, 0].set_ylabel("Country", fontsize=8)
    fig.suptitle("Historical T1 performance sensitivity", fontsize=9.5, fontweight="bold", y=0.98)
    fig.text(
        0.50,
        0.035,
        "Strict = strict QA; sens. = sensitivity QA. Cell labels are signed R² estimates.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=GREY,
    )
    cbar_ax = fig.add_axes([0.87, 0.25, 0.018, 0.53])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    ticks = [-0.1, 0.0, 0.1]
    power = 0
    while 10**power < max_abs:
        ticks.extend([-float(10**power), float(10**power)])
        power += 1
    ticks = sorted(set(tick for tick in ticks if -max_abs <= tick <= max_abs))
    colorbar.set_ticks(ticks)
    colorbar.ax.set_yticklabels([f"{tick:g}" for tick in ticks], fontsize=6.5)
    colorbar.set_label("Signed R²\n(symlog color)", fontsize=7.5, labelpad=2)
    fig.subplots_adjust(left=0.105, right=0.845, bottom=0.12, top=0.86, hspace=0.42, wspace=0.24)

    figures = _save_figure(fig, output_dir, "Supplementary_Figure_S3")
    sources = [
        _source_record("S1", "a", a, paths["a"], rendered_rows=len(a), rendered_values=int(np.isfinite(pd.to_numeric(a["r2"], errors="coerce")).sum())),
        _source_record("S1", "b", b, paths["b"], rendered_rows=len(b), rendered_values=int(np.isfinite(pd.to_numeric(b["r2"], errors="coerce")).sum())),
    ]
    metadata = {
        "sources": sources,
        "countries": country_values,
        "qa_estimator_rows": {p: int((a[pa].map(_profile) == p).sum()) for p in PROFILE_ORDER},
        "target_retransformation_rows": {p: int((b[pb].map(_profile) == p).sum()) for p in PROFILE_ORDER},
        "r2_scale": "raw signed R² annotations; symmetric-log color scale centered at zero; no rows filtered",
        "caption": CAPTIONS["Supplementary_Figure_S3"],
        "caption_change": "Recast the two source tables as four country-by-branch heatmaps; clarified signed R² color scale and retained the no-inference statement.",
        "outputs": figures,
    }
    return sources, metadata


def _s2_figure(
    frames: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loco, thin = frames["a"], frames["b"]
    country_loco = _country_column(loco)
    country_thin = _country_column(thin)
    profile_col = _profile_column(thin)
    if "r2" not in loco or "estimator_kind" not in loco:
        raise ValueError("S2a requires country, estimator_kind and signed r2 columns")
    if not {"hgb", "ridge"}.issubset(set(loco["estimator_kind"].dropna().astype(str).str.lower())):
        raise ValueError("S2a must include HGB and Ridge for each transfer target")
    loco_work = loco.copy()
    loco_work["_country"] = loco_work[country_loco].map(_clean_text)
    loco_work["_estimator"] = loco_work["estimator_kind"].astype(str).str.lower()
    loco_work["_r2"] = pd.to_numeric(loco_work["r2"], errors="coerce")
    if loco_work.duplicated(["_country", "_estimator"]).any():
        raise ValueError("S2a has duplicate country/estimator estimates")
    countries = sorted(loco_work["_country"].unique())
    if len(countries) != 17 or len(loco_work) != 34:
        raise ValueError(f"S2a must contain 34 estimates for 17 countries; got {len(loco_work)} rows and {len(countries)} countries")

    required_thin = {"seed", "rmse", "sample_size", "n_train_stations", "profile", "arm"}
    missing = required_thin - set(thin.columns)
    if missing:
        raise ValueError(f"S2b is missing thinning fields: {sorted(missing)}")
    thin_work = thin.copy()
    thin_work["_country"] = thin_work[country_thin].map(_clean_text)
    thin_work["_profile"] = thin_work[profile_col].map(_profile)
    thin_work["_seed"] = thin_work["seed"].map(_clean_text)
    thin_work["_rmse"] = pd.to_numeric(thin_work["rmse"], errors="coerce")
    sample_size = pd.to_numeric(thin_work["sample_size"], errors="coerce")
    train_stations = pd.to_numeric(thin_work["n_train_stations"], errors="coerce")
    thin_work["_train_n"] = sample_size.fillna(train_stations)
    if not thin_work["arm"].astype(str).str.lower().eq("full").all():
        raise ValueError("S2b contains an arm other than full; no records were filtered")
    if set(thin_work["_country"]) != {"USA", "CAN"}:
        raise ValueError("S2b must contain the USA and CAN network panels")
    if set(thin_work["_profile"]) != set(PROFILE_ORDER):
        raise ValueError("S2b must retain both MET profiles")
    if thin_work["_train_n"].isna().any() or (thin_work["_train_n"] <= 0).any():
        raise ValueError("S2b contains a missing or non-positive training-site coordinate")
    if thin_work["_rmse"].isna().any():
        raise ValueError("S2b contains a missing RMSE estimate")
    key = ["_country", "_profile", "_seed", "_train_n"]
    if thin_work.duplicated(key).any():
        raise ValueError("S2b has duplicate country/profile/seed/training-size records")
    shared = thin_work[thin_work["_seed"].str.lower().isin({"shared_full", "full", "full_network"})]
    fixed = thin_work.loc[~thin_work.index.isin(shared.index)].copy()
    seeds = sorted(fixed["_seed"].unique(), key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else v))
    if len(seeds) != 3:
        raise ValueError(f"S2b must retain all three fixed thinning seeds; found {seeds}")
    if len(shared) != 4 or shared.duplicated(["_country", "_profile"]).any():
        raise ValueError("S2b must have one distinct shared full-network endpoint per country/profile")

    seed_markers = {seed: marker for seed, marker in zip(seeds, ("o", "s", "^"))}
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_MM / MM_PER_IN, 4.9),
        gridspec_kw={"width_ratios": [1.32, 1.0, 1.0]},
        sharey=False,
    )
    country_y = {country: index for index, country in enumerate(countries)}
    for estimator, color, offset in (("hgb", BLUE, -0.13), ("ridge", ORANGE, 0.13)):
        subset = loco_work[loco_work["_estimator"] == estimator]
        positions = subset["_country"].map(country_y).to_numpy(dtype=float) + offset
        axes[0].scatter(subset["_r2"], positions, s=24, color=color, marker="o", edgecolors="white", linewidths=0.35, zorder=3)
    axes[0].axvline(0, color=GREY, linewidth=0.7)
    axes[0].set_xscale("symlog", linthresh=0.1)
    axes[0].set_yticks(np.arange(len(countries)), countries)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("LOCO R² (signed log scale)")
    axes[0].set_ylabel("Held-out country")
    axes[0].grid(axis="x", color=LIGHT, linewidth=0.45)
    axes[0].set_title("Strict-QA, no-MET transfer", loc="left", pad=4, fontweight="bold")
    _panel_label(axes[0], "a")
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=BLUE, label="HGB", markersize=5),
            Line2D([0], [0], marker="o", linestyle="", color=ORANGE, label="Ridge", markersize=5),
        ],
        loc="upper right",
        ncol=2,
        fontsize=7,
        handletextpad=0.35,
        columnspacing=0.7,
    )
    for ax, country, panel in ((axes[1], "USA", "b"), (axes[2], "CAN", "c")):
        subset = thin_work[thin_work["_country"] == country]
        for seed, marker in seed_markers.items():
            for profile in PROFILE_ORDER:
                rows = subset[(subset["_seed"] == seed) & (subset["_profile"] == profile)]
                ax.scatter(
                    rows["_train_n"],
                    rows["_rmse"],
                    marker=marker,
                    s=20,
                    color=PROFILE_COLOR[profile],
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )
        endpoint = shared[shared["_country"] == country]
        for profile in PROFILE_ORDER:
            rows = endpoint[endpoint["_profile"] == profile]
            ax.scatter(
                rows["_train_n"],
                rows["_rmse"],
                marker="D",
                s=35,
                facecolors="white",
                edgecolors=PROFILE_COLOR[profile],
                linewidths=1.0,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Training sites (log scale)")
        ax.set_title(f"{country} thinning runs", loc="left", pad=4, fontweight="bold")
        ax.grid(axis="y", color=LIGHT, linewidth=0.45)
        _panel_label(ax, panel)
        if country == "USA":
            ax.set_ylabel("2022 RMSE (µg m$^{-3}$)")
    fig.suptitle("Transfer and network-size sensitivity", fontsize=9.5, fontweight="bold", y=0.98)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=PROFILE_COLOR["with_met"], label="with MET", markersize=5),
        Line2D([0], [0], marker="o", linestyle="", color=PROFILE_COLOR["no_met"], label="no MET", markersize=5),
    ]
    legend_handles.extend(
        Line2D([0], [0], marker=seed_markers[seed], linestyle="", color=GREY, label=f"seed {seed}", markersize=5)
        for seed in seeds
    )
    legend_handles.append(
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor=INK, color=INK, label="shared full network", markersize=5)
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.91),
        ncol=6,
        fontsize=7.2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.15, top=0.78, wspace=0.37)

    figures = _save_figure(fig, output_dir, "Supplementary_Figure_S4")
    sources = [
        _source_record("S2", "a", loco, paths["a"], rendered_rows=len(loco), rendered_values=int(loco_work["_r2"].notna().sum())),
        _source_record("S2", "b", thin, paths["b"], rendered_rows=len(thin), rendered_values=int(thin_work["_rmse"].notna().sum())),
    ]
    metadata = {
        "sources": sources,
        "transfer_countries": countries,
        "transfer_estimators": sorted(loco_work["_estimator"].unique()),
        "thinning_rows_by_country": {c: int((thin_work["_country"] == c).sum()) for c in ("USA", "CAN")},
        "fixed_seeds": seeds,
        "fixed_seed_rows": int(len(fixed)),
        "shared_full_endpoint_rows": int(len(shared)),
        "thinning_connectors": "none; source observations only",
        "caption": CAPTIONS["Supplementary_Figure_S4"],
        "caption_change": "Clarified the 34 transfer estimates, assigned separate country panels, and replaced seed curves with observed-point markers plus distinct shared endpoints.",
        "outputs": figures,
    }
    return sources, metadata


def _coverage_columns(frame: pd.DataFrame) -> tuple[str, str]:
    point = _first_column(frame, ("pointwise_coverage",), "pointwise coverage")
    station = _first_column(
        frame,
        (
            "all_month_per_station_coverage",
            "all_month_station_coverage_recomputed_from_counts",
            "all_month_station_coverage",
        ),
        "whole-station coverage",
    )
    return point, station


def _s3_figure(
    frames: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usa, can, bra = frames["a"], frames["b"], frames["c"]
    country_columns = {key: _country_column(frame) for key, frame in frames.items()}
    profile_columns = {key: _profile_column(frame) for key, frame in frames.items()}
    task_col = _first_column(usa, ("task",), "task")
    fold_col = _first_column(usa, ("fold_id",), "fold")
    point_col, station_col = _coverage_columns(usa)
    countries = ("USA", "CAN")
    primary_frames = {"USA": usa.copy(), "CAN": can.copy()}
    for country, frame in primary_frames.items():
        ccol = country_columns["a" if country == "USA" else "b"]
        pcol = profile_columns["a" if country == "USA" else "b"]
        if len(frame) != 32:
            raise ValueError(f"S3 {country} must retain 32 primary fold rows; got {len(frame)}")
        frame["_country"] = frame[ccol].map(_clean_text)
        frame["_profile"] = frame[pcol].map(_profile)
        frame["_task"] = frame[task_col].map(_clean_text)
        frame["_fold"] = frame[fold_col].map(_clean_text)
        frame["_point"] = pd.to_numeric(frame[point_col], errors="coerce")
        frame["_station"] = pd.to_numeric(frame[station_col], errors="coerce")
        if set(frame["_country"]) != {country}:
            raise ValueError(f"S3 {country} panel contains a different country")
        if set(frame["_profile"]) != set(PROFILE_ORDER):
            raise ValueError(f"S3 {country} panel must include both MET profiles")
        if frame.duplicated(["_task", "_profile", "_fold"]).any():
            raise ValueError(f"S3 {country} panel has duplicate task/profile/fold keys")
        if len(frame.groupby(["_task", "_fold"], dropna=False)) != 16:
            raise ValueError(f"S3 {country} should contain 16 task/fold keys across two profiles")
        for metric in ("_point", "_station"):
            finite = frame[metric].dropna()
            if ((finite < 0) | (finite > 1)).any():
                raise ValueError(f"S3 {country} has a coverage fraction outside [0, 1]")

    task_order = sorted(
        set(usa[task_col].map(_clean_text)) | set(can[task_col].map(_clean_text)),
        key=_task_sort_key,
    )
    task_codes = {task: _task_label(task) for task in task_order}
    ordered_primary: dict[str, pd.DataFrame] = {}
    ordered_keys: dict[str, list[tuple[str, str, str]]] = {}
    for country, frame in primary_frames.items():
        frame = frame.copy()
        frame["_task_sort"] = frame["_task"].map(lambda value: _task_sort_key(value)[0])
        frame["_profile_sort"] = frame["_profile"].map({"no_met": 0, "with_met": 1})
        frame["_fold_sort"] = pd.to_numeric(frame["_fold"], errors="coerce")
        frame = frame.sort_values(
            ["_task_sort", "_task", "_fold_sort", "_fold", "_profile_sort"],
            kind="stable",
        ).reset_index(drop=True)
        ordered_primary[country] = frame
        ordered_keys[country] = list(zip(frame["_task"], frame["_profile"], frame["_fold"]))
    if ordered_keys["USA"] != ordered_keys["CAN"]:
        raise ValueError("S3 USA/CAN primary folds do not share the same ordered task/profile/fold keys")
    row_labels = [
        f"{task_codes[task]} · {PROFILE_LABEL[profile]}"
        + ("" if task == "T3_year_2022" else f" · f{fold}")
        for task, profile, fold in ordered_keys["USA"]
    ]

    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_IN, 7.35))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[5.2, 0.28, 1.55],
        hspace=0.23,
        wspace=0.42,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[1, :])
    bra_ax = fig.add_subplot(grid[2, :])
    image = None
    for ax, country in zip(axes, countries):
        frame = ordered_primary[country]
        matrix = frame[["_point", "_station"]].to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="cividis", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(frame)), row_labels if country == "USA" else [""] * len(frame), fontsize=9.1)
        for tick, profile in zip(ax.get_yticklabels(), frame["_profile"]):
            tick.set_color(PROFILE_COLOR[profile])
        ax.set_xticks([0, 1], ["Pointwise", "Whole-station"], fontsize=8.5)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(frame), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                label = f"{value:.3f}" if np.isfinite(value) else "NA"
                rgba = image.cmap(image.norm(value)) if np.isfinite(value) else (0.8, 0.8, 0.8, 1)
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=9.1, color=INK if luminance > 0.55 else "white")
        ax.set_title(f"{country} · 32 primary folds", loc="left", pad=4, fontsize=9, fontweight="bold")
        if country == "CAN":
            ax.tick_params(axis="y", labelleft=False)
        _panel_label(ax, "a" if country == "USA" else "b")
    assert image is not None
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0.0, 0.5, 0.9, 1.0])
    colorbar.set_label("Coverage fraction (0.90 nominal target)", fontsize=8.0, labelpad=3)
    colorbar.ax.tick_params(labelsize=7.0, length=2)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("bottom")

    bra_country = country_columns["c"]
    bra_profile = profile_columns["c"]
    task_bra = _first_column(bra, ("task",), "Brazil feasibility task")
    qa_bra = _first_column(bra, ("qa_variant", "mapped_qa_variant", "qa"), "Brazil QA class")
    estimator_bra = _first_column(bra, ("estimator_kind", "mapped_estimator", "estimator"), "Brazil estimator class")
    q90_col = _first_column(bra, ("q90",), "Brazil q90 threshold")
    if len(bra) != 8:
        raise ValueError(f"S3 Brazil panel must retain eight feasibility rows; got {len(bra)}")
    bra_work = bra.copy()
    bra_work["_country"] = bra_work[bra_country].map(_clean_text)
    bra_work["_profile"] = bra_work[bra_profile].map(_profile)
    bra_work["_task"] = bra_work[task_bra].map(_clean_text)
    bra_work["_qa"] = bra_work[qa_bra].map(_clean_text)
    bra_work["_estimator"] = bra_work[estimator_bra].map(_clean_text)
    bra_work["_q90"] = pd.to_numeric(bra_work[q90_col], errors="coerce")
    if not bra_work["_country"].eq("BRA").all():
        raise ValueError("S3 feasibility panel includes records outside Brazil")
    if set(bra_work["_profile"]) != set(PROFILE_ORDER):
        raise ValueError("S3 feasibility panel must retain both MET profiles")
    bra_work = bra_work.sort_values(["_qa", "_estimator", "_profile"], kind="stable").reset_index(drop=True)
    bra_work["_qa"] = bra_work["_qa"].str.lower()
    bra_work["_estimator"] = bra_work["_estimator"].str.lower()
    class_order = (("strict", "hgb"), ("strict", "ridge"), ("sensitivity", "hgb"), ("sensitivity", "ridge"))
    class_labels = {
        ("strict", "hgb"): "Strict HGB",
        ("strict", "ridge"): "Strict Ridge",
        ("sensitivity", "hgb"): "Sensitivity HGB",
        ("sensitivity", "ridge"): "Sensitivity Ridge",
    }
    bra_work["_source_class"] = list(zip(bra_work["_qa"], bra_work["_estimator"]))
    if set(bra_work["_source_class"]) != set(class_order):
        raise ValueError("S3 Brazil panel must retain strict/sensitivity HGB/Ridge source classes")
    if bra_work.duplicated(["_source_class", "_profile"]).any():
        raise ValueError("S3 Brazil panel must have one record per source class/MET profile")
    class_index = {key: index for index, key in enumerate(class_order)}
    bra_work["_class_order"] = bra_work["_source_class"].map(class_index)
    bra_work["_profile_order"] = bra_work["_profile"].map({"with_met": 0, "no_met": 1})
    bra_work = bra_work.sort_values(["_class_order", "_profile_order"], kind="stable").reset_index(drop=True)
    y = np.arange(len(class_order))
    finite_q90 = bra_work.loc[np.isfinite(bra_work["_q90"]), "_q90"]
    if finite_q90.empty:
        max_q90 = 1.0
    else:
        max_q90 = float(finite_q90.max())
        if max_q90 <= 0:
            max_q90 = 1.0
    right_limit = max_q90 * 1.12
    q90_inf_column = next((column for column in ("q90_is_infinite", "q90_is_inf") if column in bra_work.columns), None)
    profile_offsets = {"with_met": -0.13, "no_met": 0.13}
    for _, row in bra_work.iterrows():
        position = class_index[row["_source_class"]] + profile_offsets[row["_profile"]]
        value = row["_q90"]
        color = PROFILE_COLOR[row["_profile"]]
        if np.isfinite(value):
            bra_ax.scatter(value, position, color=color, marker="o", s=28, edgecolors="white", linewidths=0.35, zorder=3)
        elif np.isposinf(value) or (q90_inf_column is not None and str(row[q90_inf_column]).strip().lower() in {"true", "1", "yes"}):
            bra_ax.scatter(right_limit * 0.96, position, color=color, marker=">", s=28, zorder=3)
            bra_ax.text(right_limit * 0.985, position, "∞", ha="left", va="center", fontsize=8, color=color)
        else:
            bra_ax.text(0.01, position, "NA", transform=bra_ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5, color=GREY)
    bra_ax.set_xlim(0, right_limit * 1.5)
    bra_ax.set_yticks(y, [class_labels[key] for key in class_order], fontsize=8.5)
    bra_ax.invert_yaxis()
    bra_ax.set_xlabel("q90 threshold (µg m$^{-3}$)", fontsize=8.0)
    bra_ax.set_title("Brazil feasibility", loc="left", pad=4, fontsize=9, fontweight="bold")
    bra_ax.grid(axis="x", color=LIGHT, linewidth=0.45)
    _panel_label(bra_ax, "c")
    bra_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=PROFILE_COLOR["with_met"], label="with MET", markersize=5),
            Line2D([0], [0], marker="o", linestyle="", color=PROFILE_COLOR["no_met"], label="no MET", markersize=5),
        ],
        loc="upper right",
        ncol=2,
        fontsize=8,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    fig.suptitle("Fold-level coverage and separate feasibility thresholds", fontsize=9.5, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.075, top=0.91, wspace=0.42, hspace=0.28)
    figures = _save_figure(fig, output_dir, "Supplementary_Figure_S2")
    sources = [
        _source_record("S3", "a", usa, paths["a"], rendered_rows=len(usa), rendered_values=int(usa.shape[0] * 2)),
        _source_record("S3", "b", can, paths["b"], rendered_rows=len(can), rendered_values=int(can.shape[0] * 2)),
        _source_record("S3", "c", bra, paths["c"], rendered_rows=len(bra), rendered_values=int(bra_work["_q90"].notna().sum())),
    ]
    metadata = {
        "sources": sources,
        "primary_fold_rows_by_country": {"USA": int(len(usa)), "CAN": int(len(can))},
        "primary_folds_by_task": {
            country: {task: int((frame["_task"] == task).sum()) for task in sorted(frame["_task"].unique(), key=_task_sort_key)}
            for country, frame in primary_frames.items()
        },
        "brazil_feasibility_rows": int(len(bra_work)),
        "brazil_source_classes": [class_labels[key] for key in class_order],
        "brazil_profile_rows_per_class": 2,
        "brazil_finite_q90": int(np.isfinite(bra_work["_q90"]).sum()),
        "brazil_nonfinite_q90": int((~np.isfinite(bra_work["_q90"])).sum()),
        "caption": CAPTIONS["Supplementary_Figure_S2"],
        "caption_change": "Reorganized country-specific fold records as labeled coverage heatmaps and kept all eight Brazil q90 feasibility rows in a separate panel; clarified that no pooling or interval was added.",
        "outputs": figures,
    }
    return sources, metadata


def _s4_figure(
    frames: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = frames["a"].copy()
    country_col = _country_column(frame)
    profile_col = _profile_column(frame)
    task_col = _first_column(frame, ("task",), "annual task")
    required = {"delta_rmse", "delta_rmse_ci_low", "delta_rmse_ci_high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"S4 is missing estimate/interval fields: {sorted(missing)}")
    frame["_country"] = frame[country_col].map(_clean_text)
    frame["_profile"] = frame[profile_col].map(_profile)
    frame["_source_task"] = frame[task_col].map(_clean_text)
    frame["_task"] = frame[task_col].map(_s4_task_base)
    frame["_estimate"] = pd.to_numeric(frame["delta_rmse"], errors="coerce")
    frame["_low"] = pd.to_numeric(frame["delta_rmse_ci_low"], errors="coerce")
    frame["_high"] = pd.to_numeric(frame["delta_rmse_ci_high"], errors="coerce")
    if len(frame) != 24:
        raise ValueError(f"S4 must retain all 24 registered annual contrasts; got {len(frame)}")
    if set(frame["_country"]) != {"USA", "CAN"}:
        raise ValueError("S4 must retain the USA and CAN country panels")
    if set(frame["_profile"]) != set(PROFILE_ORDER):
        raise ValueError("S4 must retain both MET profiles")
    if frame[["_estimate", "_low", "_high"]].isna().any().any():
        raise ValueError("S4 estimate or saved CI is missing; no source interval was dropped")
    if (frame["_low"] > frame["_high"]).any() or (frame["_estimate"] < frame["_low"]).any() or (frame["_estimate"] > frame["_high"]).any():
        raise ValueError("S4 saved point estimates and interval limits are inconsistent")
    if frame.duplicated(["_country", "_task", "_profile"]).any():
        raise ValueError("S4 contains duplicate country/task/profile annual rows")
    counts = frame.groupby("_country", dropna=False).size().to_dict()
    if counts != {"CAN": 12, "USA": 12}:
        raise ValueError(f"S4 expects 12 rows per country; found {counts}")
    task_order = sorted(frame["_task"].unique(), key=_task_sort_key)
    if len(task_order) != 6:
        raise ValueError(f"S4 expects six source tasks; found {task_order}")
    for country, group in frame.groupby("_country"):
        if len(group) != 12 or group.groupby("_task")["_profile"].nunique().ne(2).any():
            raise ValueError(f"S4 {country} does not have two MET profiles for each task")

    x_extent = max(abs(float(frame["_low"].min())), abs(float(frame["_high"].max())))
    x_extent = max(x_extent, 0.1) * 1.08
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_MM / MM_PER_IN, 4.75),
        sharex=True,
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    task_y = {task: index for index, task in enumerate(task_order)}
    offsets = {"with_met": -0.17, "no_met": 0.17}
    for ax, country, panel in ((axes[0], "USA", "a"), (axes[1], "CAN", "b")):
        group = frame[frame["_country"] == country]
        for _, row in group.iterrows():
            y = task_y[row["_task"]] + offsets[row["_profile"]]
            color = PROFILE_COLOR[row["_profile"]]
            ax.errorbar(
                row["_estimate"],
                y,
                xerr=np.array([[row["_estimate"] - row["_low"]], [row["_high"] - row["_estimate"]]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=4.2,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.8,
                zorder=3,
            )
        ax.axvline(0, color=GREY, linewidth=0.75)
        ax.set_xlim(-x_extent, x_extent)
        ax.set_yticks(np.arange(len(task_order)), [_task_label(task) for task in task_order])
        ax.invert_yaxis()
        ax.set_xlabel("ΔRMSE: full − SatPM (µg m$^{-3}$)")
        ax.set_title(country, loc="left", pad=4, fontweight="bold")
        ax.grid(axis="x", color=LIGHT, linewidth=0.45)
        _panel_label(ax, panel)
    axes[0].set_ylabel("Annual task")
    fig.suptitle("Annual SatPM comparator contrasts", fontsize=9.5, fontweight="bold", y=0.98)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=PROFILE_COLOR["with_met"], label="with MET", markersize=5),
            Line2D([0], [0], marker="o", linestyle="", color=PROFILE_COLOR["no_met"], label="no MET", markersize=5),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.91),
        ncol=2,
        fontsize=7.7,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.text(
        0.52,
        0.035,
        "ΔRMSE = full − SatPM; positive values favor SatPM, negative values favor the full model.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.80, wspace=0.28)
    figures = _save_figure(fig, output_dir, "Supplementary_Figure_S8")
    sources = [
        _source_record("S4", "a", frames["a"], paths["a"], rendered_rows=len(frame), rendered_values=int(len(frame) * 3)),
    ]
    metadata = {
        "sources": sources,
        "rows_by_country": {country: int(counts[country]) for country in ("USA", "CAN")},
        "tasks": [
            {
                "source_task_variants": sorted(frame.loc[frame["_task"] == task, "_source_task"].unique()),
                "display_task": task,
                "display_label": _task_label(task),
            }
            for task in task_order
        ],
        "profiles": list(PROFILE_ORDER),
        "saved_ci_rows": int(frame[["_low", "_high"]].notna().all(axis=1).sum()),
        "saved_signs": {"positive": int((frame["_estimate"] > 0).sum()), "negative": int((frame["_estimate"] < 0).sum()), "zero": int((frame["_estimate"] == 0).sum())},
        "caption": CAPTIONS["Supplementary_Figure_S8"],
        "caption_change": "Kept the 24 exact-12-month annual contrasts and saved intervals; compressed the layout into six paired MET rows per country and stated the signed ΔRMSE direction.",
        "outputs": figures,
    }
    return sources, metadata


def render(
    repo_root: str | Path,
    outdir: str | Path,
    only_figures: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Render S2-S4 and S8, or a selected subset, and return a retention manifest."""
    _configure_matplotlib()
    repo, summary_root = _resolve_repository(Path(repo_root))
    output_dir = Path(outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = summary_root / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = _entry_map(registry)

    builders = {
        "Supplementary_Figure_S2": ("abc", _s3_figure),
        "Supplementary_Figure_S3": ("ab", _s1_figure),
        "Supplementary_Figure_S4": ("ab", _s2_figure),
        "Supplementary_Figure_S8": ("a", _s4_figure),
    }
    if only_figures is None:
        selected = list(builders)
    else:
        selected = []
        for value in only_figures:
            normalized = value if value.startswith("Supplementary_Figure_") else f"Supplementary_Figure_{value}"
            if normalized not in builders:
                raise ValueError(f"Unknown selected supplementary figure: {value}")
            if normalized not in selected:
                selected.append(normalized)
    if not selected:
        raise ValueError("At least one supplementary figure must be selected")

    manifest_path = output_dir / "supplementary_s2_s4_s8_manifest.json"
    prior_records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_records = {record["figure"]: record for record in prior.get("figures", []) if isinstance(record, dict) and "figure" in record}
        except (OSError, json.JSONDecodeError, TypeError):
            prior_records = {}

    updated_records: dict[str, dict[str, Any]] = {}
    for figure in selected:
        panel_labels, builder = builders[figure]
        entry = entries[figure]
        frames: dict[str, pd.DataFrame] = {}
        paths: dict[str, Path] = {}
        for panel in panel_labels:
            frames[panel], paths[panel] = _read_panel(repo, entry, panel)
        sources, details = builder(frames, paths, output_dir)
        details["caption"] = entry.get("caption", details.get("caption", ""))
        record = {
            "figure": figure,
            "registered_caption": entry.get("caption", ""),
            "panel_rows_registered": entry.get("panel_rows", {}),
            "sources": sources,
            **details,
        }
        updated_records[figure] = record

    figure_manifest = [
        (updated_records[figure] if figure in updated_records else prior_records[figure])
        for figure in builders
        if figure in updated_records or figure in prior_records
    ]
    all_outputs = {record["figure"]: record.get("outputs", {}) for record in figure_manifest}

    manifest = {
        "status": "rendered",
        "backend": "Python/matplotlib",
        "figure_width_mm": FIGURE_WIDTH_MM,
        "font_contract": "Arial bold; all text at least 8.2 pt; titles 9–11 pt; panel labels 18 pt",
        "export_contract": ["editable-text PDF", "editable-text SVG", "300 dpi PNG"],
        "source_data_changed": False,
        "new_statistical_inference": False,
        "registry_file": registry_path.name,
        "figures_rendered_this_run": selected,
        "figures": figure_manifest,
        "outputs": all_outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_file"] = manifest_path.name
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="repository root, input/github directory, or data/summary directory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="directory for S1-S4 PDF, SVG, PNG, and manifest outputs",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("S2", "S3", "S4", "S8"),
        help="optional subset to render; the default renders S1-S4",
    )
    args = parser.parse_args()
    result = render(args.repo_root, args.outdir, only_figures=args.figures)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

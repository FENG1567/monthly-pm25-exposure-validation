"""Derive a small set of audit summaries from author-owned public source tables."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_CONFORMAL_TASKS = {"T1_station": "T1", "T2_spatial": "T2",
                    "T2_spatial_buffer50km": "T2+50km", "T3_year_2022": "T3 2022"}
_MONTHLY_KEYS = (
    "country", "validation_mode", "external_year", "profile", "qa_variant", "estimator_kind",
    "arm", "target_strategy", "backtransform", "task", "model_alias", "comparison",
)
_MONTHLY_COLUMNS = {
    "r2": ("full_r2", "comparator_r2"),
    "rmse": ("full_rmse_ug_m3", "comparator_rmse_ug_m3"),
    "mae": ("full_mae_ug_m3", "comparator_mae_ug_m3"),
    "bias": ("full_bias_ug_m3", "comparator_bias_ug_m3"),
}
_MULTIPLICITY_CONTRASTS = ("B+AOD - B", "full - B", "full - B+AOD")


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing source summary: {path}") from exc


def _num(row: dict[str, str], name: str, context: str, *, finite: bool = True) -> float:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {name} must be numeric") from exc
    if math.isnan(value) or (finite and not math.isfinite(value)):
        raise ValueError(f"{context}: {name} must be finite")
    return value


def _int(row: dict[str, str], name: str, context: str) -> int:
    value = _num(row, name, context)
    if not value.is_integer():
        raise ValueError(f"{context}: {name} must be an integer")
    return int(value)


def _same(a: float, b: float, *, tolerance: float = 1e-8) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_table5(root: Path) -> list[list[Any]]:
    registry = json.loads((root / "data/summary/table_registry.json").read_text(encoding="utf-8-sig"))
    table = next((item for item in registry.get("supplementary_tables", []) if item.get("semantic_id") == "primary_conformal"), None) or next((item for item in registry.get("main_tables", []) if item.get("id") == "5"), None)
    if table is None:
        raise ValueError("Table registry has no main Table 5")
    return table.get("rows", [])


def _parse_ratio_cell(value: Any, context: str) -> tuple[float, int, int]:
    text = str(value)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)%\s*\(([\d,]+)\s*/\s*([\d,]+)\)", text)
    if not match:
        raise ValueError(f"{context}: cannot parse saved ratio/count cell {text!r}")
    return (float(match.group(1)) / 100.0, int(match.group(2).replace(",", "")),
            int(match.group(3).replace(",", "")))


def _derive_conformal(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _read_csv(root / "data/summary/source_tables/conformal_primary_fold_summary.csv")
    folds = [row for row in source if (
        row.get("normalized_country_for_grouping") in {"CAN", "USA"}
        and row.get("normalized_profile_for_grouping") in {"no_met", "with_met"}
        and row.get("normalized_qa_variant_for_grouping") == "strict"
        and row.get("normalized_estimator_for_grouping") == "hgb"
        and row.get("split_label") == "primary_80_20"
        and row.get("arm") == "full"
        and row.get("task") in _CONFORMAL_TASKS
    )]
    if len(folds) != 64:
        raise ValueError(f"Expected 64 primary conformal fold rows, found {len(folds)}")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    unique_keys: set[str] = set()
    for row in folds:
        key = row.get("unique_key", "")
        if not key or key in unique_keys:
            raise ValueError(f"Missing or duplicate conformal unique_key: {key!r}")
        unique_keys.add(key)
        if row.get("q90_status") != "finite":
            raise ValueError(f"{key}: q90 is not finite")
        groups[(row["normalized_country_for_grouping"], row["task"],
                row["normalized_profile_for_grouping"])].append(row)

    saved = _load_table5(root)
    derived: list[dict[str, Any]] = []
    saved_groups = 0
    for table_row in saved:
        if len(table_row) < 7:
            raise ValueError("Malformed Table 5 row")
        country, label = str(table_row[0]).split(" / ", 1)
        task = next((code for code, task_label in _CONFORMAL_TASKS.items() if task_label == label), None)
        profile = {"No MET": "no_met", "With MET": "with_met"}.get(str(table_row[1]))
        if task is None or profile is None:
            raise ValueError(f"Unrecognized Table 5 group: {table_row[0]!r}, {table_row[1]!r}")
        records = groups.get((country, task, profile), [])
        if not records:
            raise ValueError(f"No conformal folds for {country} {label} {profile}")
        test_n = sum(_int(row, "test_row_n", country) for row in records)
        point_n = sum(_int(row, "pointwise_covered_row_n_derived", country) for row in records)
        station_n = sum(_int(row, "all_month_station_group_n_source", country) for row in records)
        station_covered = sum(_int(row, "all_month_station_covered_n_source", country) for row in records)
        if test_n <= 0 or station_n <= 0 or point_n > test_n or station_covered > station_n:
            raise ValueError(f"Invalid pooled conformal counts for {country} {label} {profile}")
        widths = [(_num(row, "mean_interval_width", country, finite=False),
                   _int(row, "test_row_n", country)) for row in records]
        if any(width < 0 for width, _ in widths):
            raise ValueError(f"Negative interval width for {country} {label} {profile}")
        weighted_width = math.inf if any(math.isinf(width) for width, _ in widths) else (
            sum(width * n for width, n in widths) / test_n
        )
        point_coverage, point_saved_n, point_saved_d = _parse_ratio_cell(table_row[4], "Table 5 point coverage")
        station_coverage, station_saved_n, station_saved_d = _parse_ratio_cell(table_row[5], "Table 5 station coverage")
        expected_width = _num({"width": str(table_row[6])}, "width", "Table 5 weighted width", finite=False)
        if (point_n, test_n) != (point_saved_n, point_saved_d) or (station_covered, station_n) != (station_saved_n, station_saved_d):
            raise ValueError(f"Computed pooled conformal counts disagree with Table 5 for {country} {label} {profile}")
        if not _same(point_coverage, point_n / test_n, tolerance=0.00051):
            raise ValueError(f"Computed point coverage disagrees with Table 5 for {country} {label} {profile}")
        if not _same(station_coverage, station_covered / station_n, tolerance=0.00051):
            raise ValueError(f"Computed whole-station coverage disagrees with Table 5 for {country} {label} {profile}")
        if not (_same(weighted_width, expected_width, tolerance=0.00051) or weighted_width == expected_width):
            raise ValueError(f"Computed weighted interval width disagrees with Table 5 for {country} {label} {profile}")
        derived.append({
            "country": country, "task": task, "profile": profile, "fold_n": len(records),
            "pointwise_covered_rows": point_n, "test_rows": test_n,
            "pointwise_coverage": point_n / test_n,
            "whole_station_covered_groups": station_covered, "whole_station_groups": station_n,
            "whole_station_coverage": station_covered / station_n,
            "test_row_weighted_mean_interval_width": weighted_width,
        })
        saved_groups += 1
    if saved_groups != 16:
        raise ValueError(f"Expected 16 pooled Table 5 groups, derived {saved_groups}")
    return derived, {"status": "passed", "fold_rows": len(folds), "pooled_groups": saved_groups,
                    "matched_saved_table_rows": saved_groups}


def _derive_monthly(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = _read_csv(root / "data/summary/source_tables/monthly_benchmark_metrics.csv")
    saved = _read_csv(root / "data/summary/source_tables/monthly_benchmark_paired_deltas.csv")
    saved_index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in saved:
        key = tuple(row.get(column, "") for column in (*_MONTHLY_KEYS, "metric"))
        if key in saved_index:
            raise ValueError(f"Duplicate saved paired monthly delta: {key}")
        saved_index[key] = row
    derived: list[dict[str, Any]] = []
    for row in metrics:
        for metric, (full_column, comparator_column) in _MONTHLY_COLUMNS.items():
            context = "/".join(row.get(field, "") for field in _MONTHLY_KEYS) + f"/{metric}"
            delta = _num(row, full_column, context) - _num(row, comparator_column, context)
            key = tuple(row.get(column, "") for column in _MONTHLY_KEYS) + (metric,)
            saved_row = saved_index.get(key)
            if saved_row is None:
                raise ValueError(f"Missing saved monthly paired delta for {context}")
            if not _same(delta, _num(saved_row, "delta", context)):
                raise ValueError(f"Computed monthly {metric} delta disagrees with saved paired delta: {context}")
            if _int(row, "n_matched_pairs", context) != _int(saved_row, "n_matched_pairs", context):
                raise ValueError(f"Matched-pair count disagrees for {context}")
            derived.append({**{field: row.get(field, "") for field in _MONTHLY_KEYS},
                            "metric": metric, "delta_full_minus_comparator": delta,
                            "n_matched_pairs": _int(row, "n_matched_pairs", context),
                            "n_matched_stations": _int(row, "n_matched_stations", context)})
    if len(derived) != 96 or len(saved_index) != len(derived):
        raise ValueError(f"Expected 96 paired monthly metric deltas; derived {len(derived)}, saved {len(saved_index)}")
    return derived, {"status": "passed", "metric_deltas": len(derived),
                    "matched_saved_deltas": len(derived), "bootstrap_refit": False}


def _derive_multiplicity(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _read_csv(root / "data/summary/source_tables/multiplicity_contrasts.csv")
    summaries = _read_csv(root / "data/summary/source_tables/multiplicity_contrast_summary.csv")
    summaries_by_contrast = {row.get("contrast", ""): row for row in summaries}
    relevant = [row for row in source if row.get("contrast") in _MULTIPLICITY_CONTRASTS
                and row.get("family_eligible", "").strip().lower() == "true"]
    if len(source) != 78 or len(relevant) != 78:
        raise ValueError(f"Expected 78 eligible multiplicity contrast rows; source={len(source)}, eligible={len(relevant)}")
    output: list[dict[str, Any]] = []
    counts = {contrast: {"record_n": 0, "delta_rmse_nonmissing_n": 0,
                         "all_below_zero_n": 0, "all_above_zero_n": 0,
                         "contains_zero_n": 0, "interval_missing_n": 0}
              for contrast in _MULTIPLICITY_CONTRASTS}
    for row in relevant:
        contrast = row["contrast"]
        tally = counts[contrast]
        tally["record_n"] += 1
        low_raw, high_raw = row.get("simultaneous_95_low", ""), row.get("simultaneous_95_high", "")
        if not low_raw or not high_raw:
            tally["interval_missing_n"] += 1
            continue
        low, high = _num(row, "simultaneous_95_low", contrast), _num(row, "simultaneous_95_high", contrast)
        if low > high:
            raise ValueError(f"Reversed simultaneous interval for {contrast}")
        tally["delta_rmse_nonmissing_n"] += 1
        if high < 0:
            tally["all_below_zero_n"] += 1
        elif low > 0:
            tally["all_above_zero_n"] += 1
        else:
            tally["contains_zero_n"] += 1
    for contrast in _MULTIPLICITY_CONTRASTS:
        derived = counts[contrast]
        saved = summaries_by_contrast.get(contrast)
        if saved is None:
            raise ValueError(f"Missing saved multiplicity summary for {contrast}")
        for field in derived:
            if derived[field] != _int(saved, field, contrast):
                raise ValueError(f"Recomputed multiplicity {field} disagrees with saved summary for {contrast}")
        output.append({"contrast": contrast, **derived})
    return output, {"status": "passed", "source_rows": len(source), "contrasts": len(output),
                    "summary_matched": True, "interval_source": "stored simultaneous_95_low/high; bootstrap not rerun"}


def _select_annual(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_csv(root / "data/summary/source_tables/annual_comparator_primary_subtable.csv")
    aliases=set()
    for country in ('can','usa'):
        for profile in ('no_met','with_met'):
            aliases.add(f'can_hist_primary_strict_{profile}_hgb' if country=='can' else f'usa_hist_strict_{profile}_hgb')
            for year in (2023,2024):
                aliases.add(f'{country}_primary_all_published_fixed_{year}_strict_hgb__{profile}' if country=='can' else f'usa_primary_fixed_{year}_strict_hgb__{profile}')
    selected = [row for row in rows if row.get("source_file", "annual_deltas.csv") == "annual_deltas.csv"
                and row.get("support_scope") == "primary_exact_12_months"
                and row.get("contrast") == "full minus SatPM"
                and row.get('status')=='eligible' and row.get('qa_variant')=='strict'
                and row.get('estimator_kind')=='hgb' and row.get('target_strategy')=='log_smear'
                and row.get('backtransform')=='smearing' and row.get('arm')=='full'
                and row.get('alias') in aliases
                and row.get("primary_filter_match", "").strip().lower() == "true"]
    if len(selected) != 24:
        raise ValueError(f"Expected 24 primary_exact_12_months full minus SatPM rows, found {len(selected)}")
    return selected, {"status": "passed", "selected_rows": len(selected),
                     "scope": "row selection from annual_comparator_primary_subtable; annual estimates not recalculated"}


def derive_summaries(repo_root: str | Path, outdir: str | Path) -> dict[str, Any]:
    """Write recomputed aggregate summaries and one explicitly scoped row selection."""
    root, output = Path(repo_root).resolve(), Path(outdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    conformal, conformal_check = _derive_conformal(root)
    monthly, monthly_check = _derive_monthly(root)
    multiplicity, multiplicity_check = _derive_multiplicity(root)
    annual, annual_check = _select_annual(root)

    files = {
        "conformal": "conformal_table5_derived.csv",
        "monthly": "monthly_paired_deltas_derived.csv",
        "multiplicity": "multiplicity_contrasts_derived.csv",
        "annual_selection": "annual_satpm_primary_exact_12_months.csv",
    }
    _write_csv(output / files["conformal"], conformal, list(conformal[0]))
    _write_csv(output / files["monthly"], monthly, list(monthly[0]))
    _write_csv(output / files["multiplicity"], multiplicity, list(multiplicity[0]))
    annual_fields = list(annual[0]) if annual else []
    _write_csv(output / files["annual_selection"], annual, annual_fields)
    result = {
        "status": "passed", "output_files": files,
        "checks": {"conformal": conformal_check, "monthly": monthly_check,
                   "multiplicity": multiplicity_check, "annual_selection": annual_check},
        "scope": [
            "Conformal pooled coverages and test-row-weighted interval widths were recomputed from the 64 selected fold records.",
            "Monthly point-metric deltas were recomputed from matched full/comparator metrics; saved paired deltas were used only for consistency checks. Bootstrap intervals were not recomputed.",
            "Multiplicity sign counts were recomputed from the stored simultaneous interval bounds; bootstrap replicates were not rerun.",
            "Annual comparator output is a 24-row selection of stored primary_exact_12_months full minus SatPM records; estimates were not recalculated.",
        ],
    }
    (output / "derive_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result

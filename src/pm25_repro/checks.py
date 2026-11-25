"""Scientific consistency checks for the published summary results."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_CONFORMAL_TASKS = {
    "T1_station": "T1",
    "T2_spatial": "T2",
    "T2_spatial_buffer50km": "T2+50km",
    "T3_year_2022": "T3 2022",
}
_MONTHLY_KEYS = (
    "country", "validation_mode", "external_year", "profile", "qa_variant",
    "estimator_kind", "arm", "target_strategy", "backtransform", "task",
    "model_alias", "comparison",
)
_MONTHLY_METRICS = {
    "r2": ("full_r2", "comparator_r2"),
    "rmse": ("full_rmse_ug_m3", "comparator_rmse_ug_m3"),
    "mae": ("full_mae_ug_m3", "comparator_mae_ug_m3"),
    "bias": ("full_bias_ug_m3", "comparator_bias_ug_m3"),
}
_MULTIPLICITY_EXPECTATIONS = {
    "B+AOD - B": (26, 26, 24, 0, 2),
    "full - B": (26, 26, 26, 0, 0),
    "full - B+AOD": (26, 26, 12, 0, 14),
}


def _read_csv(root: Path, relative: str) -> list[dict[str, str]]:
    path = root / relative
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing summary input: {path}") from exc


def _number(row: dict[str, str], column: str, context: str) -> float:
    raw = row.get(column, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {column} is not numeric ({raw!r})") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}: {column} is not finite ({raw!r})")
    return value


def _integer(row: dict[str, str], column: str, context: str) -> int:
    value = _number(row, column, context)
    if not value.is_integer():
        raise ValueError(f"{context}: {column} must be an integer ({value})")
    return int(value)


def _close(observed: float, expected: float, *, tolerance: float = 1e-8) -> bool:
    return math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance)


def _check_conformal(root: Path) -> dict[str, Any]:
    rows = _read_csv(root, "data/summary/source_tables/conformal_primary_fold_summary.csv")
    tasks = set(_CONFORMAL_TASKS)
    folds = [row for row in rows if (
        row.get("normalized_country_for_grouping") in {"CAN", "USA"}
        and row.get("normalized_profile_for_grouping") in {"no_met", "with_met"}
        and row.get("normalized_qa_variant_for_grouping") == "strict"
        and row.get("normalized_estimator_for_grouping") == "hgb"
        and row.get("split_label") == "primary_80_20"
        and row.get("arm") == "full"
        and row.get("task") in tasks
    )]
    if len(folds) != 64:
        raise ValueError(f"Primary conformal check expected 64 CAN/USA fold records; found {len(folds)}")
    seen: set[str] = set()
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in folds:
        context = row.get("unique_key", "conformal fold")
        key = row.get("unique_key")
        if not key or key in seen:
            raise ValueError(f"Duplicate or missing conformal fold key: {key!r}")
        seen.add(key)
        if row.get("q90_status") != "finite":
            raise ValueError(f"{context}: primary q90 must be finite")
        _number(row, "q90", context)
        test_n = _integer(row, "test_row_n", context)
        covered_n = _integer(row, "pointwise_covered_row_n_derived", context)
        if not 0 <= covered_n <= test_n or test_n <= 0:
            raise ValueError(f"{context}: invalid pointwise covered/test counts")
        pointwise = _number(row, "pointwise_coverage", context)
        if not _close(pointwise, covered_n / test_n):
            raise ValueError(f"{context}: pointwise_coverage is not the covered/test ratio")
        station_n = _integer(row, "all_month_station_group_n_source", context)
        station_covered = _integer(row, "all_month_station_covered_n_source", context)
        if station_n <= 0 or not 0 <= station_covered <= station_n:
            raise ValueError(f"{context}: invalid all-month station counts")
        station_ratio = _number(row, "all_month_station_coverage_recomputed_from_counts", context)
        if not _close(station_ratio, station_covered / station_n):
            raise ValueError(f"{context}: station coverage is not the covered/group ratio")
        grouped[(row["normalized_country_for_grouping"], row["task"],
                 row["normalized_profile_for_grouping"])].append(row)

    registry_path = root / "data" / "summary" / "table_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing table registry: {registry_path}") from exc
    conformal_table = next((item for item in registry.get("supplementary_tables", []) if item.get("semantic_id") == "primary_conformal"), None) or next((table for table in registry.get("main_tables", []) if table.get("id") == "5"), None)
    if conformal_table is None:
        raise ValueError("Table registry is missing primary conformal Table 5")

    checked_groups = 0
    for table_row in conformal_table.get("rows", []):
        if len(table_row) < 6:
            raise ValueError("Primary conformal Table 5 contains a malformed row")
        country_task = str(table_row[0]).split(" / ", 1)
        if len(country_task) != 2:
            raise ValueError(f"Unexpected Table 5 country/task label: {table_row[0]!r}")
        country, task_label = country_task
        task = next((key for key, label in _CONFORMAL_TASKS.items() if label == task_label), None)
        profile = {"No MET": "no_met", "With MET": "with_met"}.get(str(table_row[1]))
        if task is None or profile is None:
            raise ValueError(f"Unexpected Table 5 group: {table_row[0]!r}, {table_row[1]!r}")
        group = grouped.get((country, task, profile), [])
        if not group:
            raise ValueError(f"No primary conformal folds found for {country} {task_label} {profile}")
        point_counts = (sum(_integer(row, "pointwise_covered_row_n_derived", "Table 5 pool") for row in group),
                        sum(_integer(row, "test_row_n", "Table 5 pool") for row in group))
        station_counts = (sum(_integer(row, "all_month_station_covered_n_source", "Table 5 pool") for row in group),
                          sum(_integer(row, "all_month_station_group_n_source", "Table 5 pool") for row in group))
        point_match = re.search(r"\((\d[\d,]*)\s*/\s*(\d[\d,]*)\)", str(table_row[4]))
        station_match = re.search(r"\((\d[\d,]*)\s*/\s*(\d[\d,]*)\)", str(table_row[5]))
        if not point_match or not station_match:
            raise ValueError(f"Table 5 pooled ratios are missing count anchors for {country} {task_label}")
        if tuple(int(value.replace(",", "")) for value in point_match.groups()) != point_counts:
            raise ValueError(f"Table 5 pointwise pooled counts disagree for {country} {task_label} {profile}")
        if tuple(int(value.replace(",", "")) for value in station_match.groups()) != station_counts:
            raise ValueError(f"Table 5 station pooled counts disagree for {country} {task_label} {profile}")
        checked_groups += 1
    if checked_groups != 16:
        raise ValueError(f"Primary conformal check expected 16 pooled groups; found {checked_groups}")
    return {"status": "passed", "fold_records": len(folds), "pooled_groups": checked_groups}


def _check_monthly_deltas(root: Path) -> dict[str, Any]:
    metric_rows = _read_csv(root, "data/summary/source_tables/monthly_benchmark_metrics.csv")
    delta_rows = _read_csv(root, "data/summary/source_tables/monthly_benchmark_paired_deltas.csv")
    metric_index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in metric_rows:
        key = tuple(row.get(field, "") for field in _MONTHLY_KEYS)
        if key in metric_index:
            raise ValueError(f"Duplicate monthly benchmark metric key: {key}")
        metric_index[key] = row
    checked = 0
    for row in delta_rows:
        context = "/".join(row.get(field, "") for field in _MONTHLY_KEYS)
        key = tuple(row.get(field, "") for field in _MONTHLY_KEYS)
        metric_row = metric_index.get(key)
        if metric_row is None:
            raise ValueError(f"Paired monthly delta has no matching metric row: {context}")
        metric = row.get("metric", "")
        if metric not in _MONTHLY_METRICS:
            raise ValueError(f"Unknown paired monthly metric {metric!r}: {context}")
        full_column, comparator_column = _MONTHLY_METRICS[metric]
        expected_delta = _number(metric_row, full_column, context) - _number(metric_row, comparator_column, context)
        observed_delta = _number(row, "delta", context)
        if not _close(observed_delta, expected_delta):
            raise ValueError(f"Paired monthly {metric} delta disagrees with the matched metric row: {context}")
        for column in ("n_matched_pairs", "n_matched_stations"):
            if _integer(row, column, context) != _integer(metric_row, column, context):
                raise ValueError(f"Paired monthly {column} disagrees with metric row: {context}")
        low, high = _number(row, "ci_low", context), _number(row, "ci_high", context)
        if low > high:
            raise ValueError(f"Paired monthly interval bounds are reversed: {context}")
        if row.get("ci_status") == "ci_computed" and _integer(row, "bootstrap_valid_replicates", context) <= 0:
            raise ValueError(f"Computed paired monthly interval has no valid replicates: {context}")
        checked += 1
    if checked == 0:
        raise ValueError("No paired monthly benchmark deltas were checked")
    return {"status": "passed", "paired_deltas_checked": checked,
            "metric_rows": len(metric_rows), "metric_types": sorted(_MONTHLY_METRICS)}


def _check_feature_increment_signs(root: Path) -> dict[str, Any]:
    rows = _read_csv(root, "data/summary/source_tables/multiplicity_contrast_summary.csv")
    by_contrast = {row.get("contrast", ""): row for row in rows}
    observed: dict[str, dict[str, int]] = {}
    for contrast, expected in _MULTIPLICITY_EXPECTATIONS.items():
        row = by_contrast.get(contrast)
        if row is None:
            raise ValueError(f"Missing multiplicity contrast: {contrast}")
        fields = ("record_n", "delta_rmse_nonmissing_n", "all_below_zero_n",
                  "all_above_zero_n", "contains_zero_n")
        values = tuple(_integer(row, field, contrast) for field in fields)
        if values != expected:
            raise ValueError(f"Unexpected feature-increment sign counts for {contrast}: {values}; expected {expected}")
        if _integer(row, "interval_missing_n", contrast) != 0:
            raise ValueError(f"Multiplicity contrast {contrast} has missing intervals")
        observed[contrast] = dict(zip(fields, values))
    return {"status": "passed", "contrasts_checked": observed}


def verify_results(repo_root: str | Path) -> dict[str, Any]:
    """Check pooled conformal ratios, paired monthly deltas, and feature increments."""
    root = Path(repo_root).resolve()
    checks = {
        "pooled_conformal": _check_conformal(root),
        "monthly_paired_deltas": _check_monthly_deltas(root),
        "feature_increment_signs": _check_feature_increment_signs(root),
    }
    return {"status": "passed", "checks": checks}

#!/usr/bin/env python3
"""Check bundled annual export consistency and deterministic bootstrap seeds.

This verifies already-exported tables. It does not rebuild the summaries or
rerun the bootstrap. Original project code, including the seed algorithm below,
is available under the MIT licence; see LICENSE and ANNUAL_EXPORT_CHAIN.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METRICS = ("rmse", "mae", "bias", "r2")
KEYS = (
    "alias",
    "qa",
    "profile",
    "estimator",
    "task",
    "target_strategy",
    "backtransform",
    "country",
    "support_scope",
    "benchmark_version",
)
SEED_IDENTITY_FIELDS = (
    "alias",
    "country",
    "qa",
    "profile",
    "estimator",
    "task",
    "target_strategy",
    "backtransform",
)
EXPECTED_CELLS = 352
EXPECTED_METRIC_ROWS = 1056
EXPECTED_DELTA_ROWS = 704
EXPECTED_REPLICATES = 500
EXPECTED_BENCHMARK = "V6GL03"
MAX_POINT_ERROR = 1e-10


def seed_for(labels: Sequence[str]) -> int:
    """Return the seed used by the archived annual comparison generator.

    This is the exact seed_for algorithm from
    analysis/original/annual/compare_cams.py: NUL-delimit the prefix and labels,
    then interpret the first eight SHA-256 hex digits as an integer.
    """
    raw = ("cams-matched-station-bootstrap-v1\0" + "\0".join(labels)).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_file_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def read_export(path: Path, spec: Mapping[str, object]) -> List[Dict[str, str]]:
    require(path.is_file(), "Missing export: {}".format(path))
    actual_sha = sha256_file(path)
    require(actual_sha == spec["sha256"], "SHA-256 mismatch for {}".format(path.name))

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == spec["columns"], "Column header mismatch for {}".format(path.name))
        rows: List[Dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            require(None not in row, "Extra CSV fields in {} at line {}".format(path.name, line_number))
            rows.append(row)

    require(len(rows) == spec["rows"], "Row count mismatch for {}".format(path.name))
    return rows


def finite_number(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def integer(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid integer for {}: {!r}".format(label, value))
    return parsed


def validate_catalog(catalog_path: Path) -> Mapping[str, object]:
    require(catalog_path.is_file(), "Missing package catalog: {}".format(catalog_path))
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read package catalog: {}".format(error))

    require(catalog.get("schema_version") == 1, "Unsupported package catalog schema")
    files = catalog.get("files")
    require(isinstance(files, list), "Catalog field 'files' must be a list")
    by_name = {entry.get("name"): entry for entry in files if isinstance(entry, dict)}
    require(set(by_name) == {"annual_metrics.csv", "annual_deltas.csv"}, "Catalog must bind both annual exports")
    return by_name


def validate_exports(
    metrics: Sequence[Mapping[str, str]], deltas: Sequence[Mapping[str, str]]
) -> Dict[str, object]:
    metrics_by_key: Dict[Tuple[str, ...], Mapping[str, str]] = {}
    arms_by_cell: Dict[Tuple[str, ...], set] = {}
    cells = set()

    for row in metrics:
        require(row["qa_variant"] == row["qa"], "qa_variant differs from qa")
        require(row["estimator_kind"] == row["estimator"], "estimator_kind differs from estimator")
        key = tuple(row[field] for field in KEYS)
        arm_key = key + (row["arm"],)
        require(arm_key not in metrics_by_key, "Duplicate annual metric key: {}".format(arm_key))
        metrics_by_key[arm_key] = row
        cells.add(key)
        arms_by_cell.setdefault(key, set()).add(row["arm"])

    require(len(cells) == EXPECTED_CELLS, "Expected {} model support cells; found {}".format(EXPECTED_CELLS, len(cells)))
    expected_arms = {"full", "raw_CAMS", "SatPM"}
    for key, arms in arms_by_cell.items():
        require(arms == expected_arms, "Unexpected arms for model support cell {}: {}".format(key, sorted(arms)))

    errors: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    delta_keys = set()
    seed_matches = 0
    eligible = 0
    withheld = 0

    for row in deltas:
        require(row["qa_variant"] == row["qa"], "delta qa_variant differs from qa")
        require(row["estimator_kind"] == row["estimator"], "delta estimator_kind differs from estimator")

        contrast = row["contrast"]
        prefix = "full minus "
        require(contrast.startswith(prefix), "Unexpected annual contrast: {}".format(contrast))
        control = contrast[len(prefix):]
        require(control in ("raw_CAMS", "SatPM"), "Unsupported annual comparator: {}".format(control))

        key = tuple(row[field] for field in KEYS)
        delta_key = key + (contrast,)
        require(delta_key not in delta_keys, "Duplicate annual delta key: {}".format(delta_key))
        delta_keys.add(delta_key)

        require(row["benchmark_version"] == EXPECTED_BENCHMARK, "Unexpected benchmark version")
        seed_labels = (
            "fair-totalpm-v1",
            *(str(row[field]) for field in SEED_IDENTITY_FIELDS),
            str(row["support_scope"]),
            EXPECTED_BENCHMARK,
            control,
        )
        expected_seed = seed_for(seed_labels)
        require(integer(row["bootstrap_seed"], "bootstrap_seed") == expected_seed,
                "Bootstrap seed mismatch for {}".format(delta_key))
        require(integer(row["bootstrap_replicates"], "bootstrap_replicates") == EXPECTED_REPLICATES,
                "Unexpected bootstrap replicate count for {}".format(delta_key))
        seed_matches += 1

        full_key = key + ("full",)
        control_key = key + (control,)
        require(full_key in metrics_by_key, "Missing full-arm metric for {}".format(delta_key))
        require(control_key in metrics_by_key, "Missing comparator metric for {}".format(delta_key))
        full = metrics_by_key[full_key]
        other = metrics_by_key[control_key]
        rows = integer(row["rows"], "delta rows")
        sites = integer(row["sites"], "delta sites")
        require(integer(full["rows"], "metric rows") == rows == integer(other["rows"], "comparator rows"),
                "Row support differs across metric and delta exports for {}".format(delta_key))
        require(integer(full["sites"], "metric sites") == sites == integer(other["sites"], "comparator sites"),
                "Site support differs across metric and delta exports for {}".format(delta_key))

        for metric in METRICS:
            actual = finite_number(row["delta_" + metric])
            full_value = finite_number(full[metric])
            other_value = finite_number(other[metric])
            expected = full_value - other_value if full_value is not None and other_value is not None else None
            require((actual is None) == (expected is None),
                    "Finite-value mismatch for {} {}".format(delta_key, metric))
            if actual is not None and expected is not None:
                errors[metric].append(abs(actual - expected))

            low = finite_number(row["delta_" + metric + "_ci_low"])
            high = finite_number(row["delta_" + metric + "_ci_high"])
            if sites < 2 or rows == 0:
                require(low is None and high is None,
                        "Unsupported interval is present for {} {}".format(delta_key, metric))
            elif low is not None or high is not None:
                require(low is not None and high is not None and low <= high,
                        "Malformed interval for {} {}".format(delta_key, metric))

        if sites < 2 or rows == 0:
            withheld += 1
        else:
            eligible += 1

    require(len(delta_keys) == EXPECTED_DELTA_ROWS,
            "Expected {} annual contrasts; found {}".format(EXPECTED_DELTA_ROWS, len(delta_keys)))
    require(seed_matches == EXPECTED_DELTA_ROWS, "Not all annual contrast seeds were checked")
    require(eligible + withheld == EXPECTED_DELTA_ROWS, "Annual contrast accounting mismatch")
    max_errors = {metric: max(values) if values else None for metric, values in errors.items()}
    for metric, error in max_errors.items():
        require(error is None or error < MAX_POINT_ERROR,
                "Point-difference error exceeds tolerance for {}: {}".format(metric, error))

    return {
        "metric_rows": len(metrics),
        "contrast_rows": len(deltas),
        "model_support_cells": len(cells),
        "all_seed_matches": seed_matches,
        "point_difference_max_abs_error": max_errors,
        "eligible_contrasts": eligible,
        "withheld_contrasts": withheld,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    package_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=package_root / "data" / "provenance" / "catalog.json")
    parser.add_argument("--metrics", type=Path, default=package_root / "data" / "summary" / "source_tables" / "annual_metrics.csv")
    parser.add_argument("--deltas", type=Path, default=package_root / "data" / "summary" / "source_tables" / "annual_deltas.csv")
    parser.add_argument("--receipt", type=Path, default=package_root / "results" / "generated" / "annual_package_validation.json")
    args = parser.parse_args(argv)

    try:
        for input_path in (args.catalog, args.metrics, args.deltas):
            require(not same_file_path(args.receipt, input_path),
                    "Receipt path must not overwrite an input file: {}".format(input_path))
        specs = validate_catalog(args.catalog)
        metrics = read_export(args.metrics, specs["annual_metrics.csv"])
        deltas = read_export(args.deltas, specs["annual_deltas.csv"])
        require(len(metrics) == EXPECTED_METRIC_ROWS, "Unexpected annual metric row count")
        require(len(deltas) == EXPECTED_DELTA_ROWS, "Unexpected annual delta row count")
        checks = validate_exports(metrics, deltas)
        result: Dict[str, object] = {
            "status": "passed",
            "validation_scope": "export table consistency and deterministic seed matching",
            **checks,
            "distributed_sha256": {
                "annual_metrics": sha256_file(args.metrics),
                "annual_deltas": sha256_file(args.deltas),
            },
            "catalog_sha256": sha256_file(args.catalog),
            "original_source_csv_bytes_recovered": False,
            "raw_source_numerical_comparison": False,
            "bootstrap_reexecuted": False,
            "estimates_or_intervals_regenerated": False,
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, csv.Error, KeyError, TypeError) as error:
        parser.error(str(error))
    print("ANNUAL_PACKAGE_VALIDATION_PASSED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

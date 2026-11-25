"""Focused tests for registered table output and scientific summary checks."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pm25_repro.checks import verify_results
from pm25_repro.tables import build_tables


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_check_fixture(root: Path) -> None:
    source = root / "data" / "summary" / "source_tables"
    conformal_fields = [
        "normalized_country_for_grouping", "normalized_profile_for_grouping",
        "normalized_qa_variant_for_grouping", "normalized_estimator_for_grouping",
        "split_label", "arm", "task", "unique_key", "q90_status", "q90",
        "test_row_n", "pointwise_covered_row_n_derived", "pointwise_coverage",
        "all_month_station_group_n_source", "all_month_station_covered_n_source",
        "all_month_station_coverage_recomputed_from_counts",
    ]
    table_rows = []
    conformal_rows = []
    task_labels = {
        "T1_station": "T1", "T2_spatial": "T2",
        "T2_spatial_buffer50km": "T2+50km", "T3_year_2022": "T3 2022",
    }
    for country in ("CAN", "USA"):
        for task, task_label in task_labels.items():
            fold_count = 1 if task == "T3_year_2022" else 5
            for profile, profile_label in (("no_met", "No MET"), ("with_met", "With MET")):
                for fold in range(fold_count):
                    conformal_rows.append({
                        "normalized_country_for_grouping": country,
                        "normalized_profile_for_grouping": profile,
                        "normalized_qa_variant_for_grouping": "strict",
                        "normalized_estimator_for_grouping": "hgb",
                        "split_label": "primary_80_20", "arm": "full", "task": task,
                        "unique_key": f"{country}|{task}|{profile}|{fold}", "q90_status": "finite",
                        "q90": "1.0", "test_row_n": "1", "pointwise_covered_row_n_derived": "1",
                        "pointwise_coverage": "1.0", "all_month_station_group_n_source": "1",
                        "all_month_station_covered_n_source": "1",
                        "all_month_station_coverage_recomputed_from_counts": "1.0",
                    })
                count = fold_count
                table_rows.append([
                    f"{country} / {task_label}", profile_label, "1–1", f"{fold_count}/{fold_count}",
                    f"100.0% ({count}/{count})", f"100.0% ({count}/{count})", "1.000",
                ])
    _write_csv(source / "conformal_primary_fold_summary.csv", conformal_fields, conformal_rows)
    registry = {"main_tables": [{"id": "5", "rows": table_rows}], "supplementary_tables": []}
    path = root / "data" / "summary" / "table_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry), encoding="utf-8")

    common = {
        "country": "CAN", "validation_mode": "fixed", "external_year": "2023",
        "profile": "no_met", "qa_variant": "strict", "estimator_kind": "hgb", "arm": "full",
        "target_strategy": "log_smear", "backtransform": "smearing", "task": "T4_external_2023",
        "model_alias": "can_fixed_2023_strict_no_met_hgb", "comparison": "SatPM_V6GL03_monthly",
    }
    metric = {**common, "n_matched_pairs": "2", "n_matched_stations": "1", "full_r2": "0.5",
              "full_rmse_ug_m3": "2.0", "full_mae_ug_m3": "1.0", "full_bias_ug_m3": "0.2",
              "comparator_r2": "0.3", "comparator_rmse_ug_m3": "3.0", "comparator_mae_ug_m3": "1.5",
              "comparator_bias_ug_m3": "0.1"}
    _write_csv(source / "monthly_benchmark_metrics.csv", list(metric), [metric])
    delta = {**common, "metric": "r2", "delta": "0.2", "ci_low": "0.1", "ci_high": "0.3",
             "n_matched_pairs": "2", "n_matched_stations": "1", "bootstrap_valid_replicates": "10",
             "ci_status": "ci_computed"}
    _write_csv(source / "monthly_benchmark_paired_deltas.csv", list(delta), [delta])
    multiplicity = []
    for contrast, below, contains in (("B+AOD - B", 24, 2), ("full - B", 26, 0), ("full - B+AOD", 12, 14)):
        multiplicity.append({"contrast": contrast, "record_n": "26", "delta_rmse_nonmissing_n": "26",
                             "all_below_zero_n": str(below), "all_above_zero_n": "0",
                             "contains_zero_n": str(contains), "interval_missing_n": "0"})
    _write_csv(source / "multiplicity_contrast_summary.csv", list(multiplicity[0]), multiplicity)


class TableRenderingTests(unittest.TestCase):
    def test_csv_and_markdown_output_matches_reference_csv_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = {
                "main_tables": [{"id": "1", "caption": "Table 1. Example", "columns": ["a", "b"],
                                 "rows": [["x,y", "one|two"]]}],
                "supplementary_tables": [{"id": "S1", "caption": "Supplement", "columns": ["v"], "rows": [["ok"]]}],
            }
            registry_path = root / "data" / "summary" / "table_registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            references = root / "results" / "reference" / "tables"
            references.mkdir(parents=True)
            expected = 'a,b\r\n"x,y",one|two\r\n'.encode("utf-8")
            (references / "Table_1.csv").write_bytes(expected)
            (references / "Table_S1.csv").write_bytes(b"v\r\nok\r\n")
            result = build_tables(root, root / "results" / "generated" / "tables")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["tables_generated"], 2)
            self.assertEqual(result["reference_check"]["matched"], 2)
            self.assertEqual((root / "results" / "generated" / "tables" / "Table_1.csv").read_bytes(), expected)
            markdown = (root / "results" / "generated" / "tables" / "Table_1.md").read_text(encoding="utf-8")
            self.assertIn("one\\|two", markdown)

    def test_malformed_registry_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "data" / "summary" / "table_registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(json.dumps({"main_tables": [{"id": "1", "caption": "x",
                "columns": ["a", "b"], "rows": [["only one"]]}], "supplementary_tables": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_tables(root, root / "out")


class SummaryCheckTests(unittest.TestCase):
    def test_scientific_summary_anchors_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_check_fixture(root)
            result = verify_results(root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["checks"]["pooled_conformal"]["fold_records"], 64)
            self.assertEqual(result["checks"]["pooled_conformal"]["pooled_groups"], 16)
            self.assertEqual(result["checks"]["monthly_paired_deltas"]["paired_deltas_checked"], 1)

    def test_multiplicity_anchor_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_check_fixture(root)
            path = root / "data" / "summary" / "source_tables" / "multiplicity_contrast_summary.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["all_below_zero_n"] = "23"
            _write_csv(path, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "Unexpected feature-increment sign counts"):
                verify_results(root)


if __name__ == "__main__":
    unittest.main()

"""Tests for direct derivations from the public summary source tables."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pm25_repro.derive import derive_summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> None:
    summary = root / "data" / "summary"
    source = summary / "source_tables"
    task_labels = {"T1_station": "T1", "T2_spatial": "T2",
                   "T2_spatial_buffer50km": "T2+50km", "T3_year_2022": "T3 2022"}
    conformal = []
    table5 = []
    for country in ("CAN", "USA"):
        for task, task_label in task_labels.items():
            nfold = 1 if task == "T3_year_2022" else 5
            for profile, profile_label in (("no_met", "No MET"), ("with_met", "With MET")):
                for fold in range(nfold):
                    conformal.append({
                        "normalized_country_for_grouping": country,
                        "normalized_profile_for_grouping": profile,
                        "normalized_qa_variant_for_grouping": "strict",
                        "normalized_estimator_for_grouping": "hgb", "split_label": "primary_80_20",
                        "arm": "full", "task": task, "unique_key": f"{country}-{task}-{profile}-{fold}",
                        "q90_status": "finite", "test_row_n": 10,
                        "pointwise_covered_row_n_derived": 9, "all_month_station_group_n_source": 5,
                        "all_month_station_covered_n_source": 4, "mean_interval_width": 2.0,
                    })
                table5.append([f"{country} / {task_label}", profile_label, "1–1", f"{nfold}/{nfold}",
                               f"90.0% ({nfold * 9}/{nfold * 10})",
                               f"80.0% ({nfold * 4}/{nfold * 5})", "2.000"])
    _write_csv(source / "conformal_primary_fold_summary.csv", conformal)
    (summary / "table_registry.json").parent.mkdir(parents=True, exist_ok=True)
    (summary / "table_registry.json").write_text(json.dumps({"main_tables": [{"id": "5", "rows": table5}]}), encoding="utf-8")

    metric_fields = ["country", "validation_mode", "external_year", "profile", "qa_variant", "estimator_kind",
                     "arm", "target_strategy", "backtransform", "task", "model_alias", "comparison",
                     "n_matched_pairs", "n_matched_stations", "full_r2", "comparator_r2", "full_rmse_ug_m3",
                     "comparator_rmse_ug_m3", "full_mae_ug_m3", "comparator_mae_ug_m3", "full_bias_ug_m3",
                     "comparator_bias_ug_m3"]
    metric_rows = []
    paired_rows = []
    for i in range(24):
        row = dict(zip(metric_fields, ["CAN", "fixed", str(2023 + i % 3), "no_met", "strict", "hgb", "full",
                                       "log_smear", "smearing", f"task{i}", f"model{i}", "SatPM_V6GL03_monthly",
                                       2, 1, 0.5, 0.3, 2.0, 1.0, 1.0, 0.5, 0.2, 0.1]))
        metric_rows.append(row)
        for metric, full, comp in (("r2", "full_r2", "comparator_r2"),
                                   ("rmse", "full_rmse_ug_m3", "comparator_rmse_ug_m3"),
                                   ("mae", "full_mae_ug_m3", "comparator_mae_ug_m3"),
                                   ("bias", "full_bias_ug_m3", "comparator_bias_ug_m3")):
            paired_rows.append({**{k: row[k] for k in metric_fields[:12]}, "metric": metric,
                                "delta": float(row[full]) - float(row[comp]),
                                "n_matched_pairs": 2, "n_matched_stations": 1})
    _write_csv(source / "monthly_benchmark_metrics.csv", metric_rows)
    _write_csv(source / "monthly_benchmark_paired_deltas.csv", paired_rows)

    multiplicity = []
    summaries = []
    counts = {"B+AOD - B": (24, 2), "full - B": (26, 0), "full - B+AOD": (12, 14)}
    for contrast, (below, contains) in counts.items():
        tally = {"record_n": 26, "delta_rmse_nonmissing_n": 26, "all_below_zero_n": below,
                 "all_above_zero_n": 0, "contains_zero_n": contains, "interval_missing_n": 0}
        summaries.append({"contrast": contrast, **tally})
        for i in range(26):
            low, high = (-2, -1) if i < below else (-1, 1)
            multiplicity.append({"contrast": contrast, "family_eligible": "True",
                                 "simultaneous_95_low": low, "simultaneous_95_high": high})
    _write_csv(source / "multiplicity_contrasts.csv", multiplicity)
    _write_csv(source / "multiplicity_contrast_summary.csv", summaries)
    annual = [{"source_file":"annual_deltas.csv", "support_scope": "primary_exact_12_months", "contrast": "full minus SatPM",
               "primary_filter_match": "True", "country": "CAN", "year": str(2000 + i),
               "status":"eligible","qa_variant":"strict","estimator_kind":"hgb","target_strategy":"log_smear",
               "backtransform":"smearing","arm":"full","alias":"can_hist_primary_strict_no_met_hgb"} for i in range(24)]
    _write_csv(source / "annual_comparator_primary_subtable.csv", annual)


class DeriveSummaryTests(unittest.TestCase):
    def test_derivations_match_stored_summary_anchors_and_write_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _fixture(root)
            result = derive_summaries(root, Path(tmp) / "out")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["checks"]["conformal"]["pooled_groups"], 16)
            self.assertEqual(result["checks"]["monthly"]["metric_deltas"], 96)
            self.assertEqual(result["checks"]["multiplicity"]["source_rows"], 78)
            self.assertEqual(result["checks"]["annual_selection"]["selected_rows"], 24)
            out = Path(tmp) / "out"
            self.assertTrue((out / "derive_summary.json").is_file())
            self.assertEqual(len(list(out.glob("*.csv"))), 4)
            with (out / "conformal_table5_derived.csv").open(encoding="utf-8", newline="") as stream:
                first = next(csv.DictReader(stream))
            self.assertEqual(first["pointwise_coverage"], "0.9")
            self.assertEqual(first["whole_station_coverage"], "0.8")
            self.assertEqual(first["test_row_weighted_mean_interval_width"], "2.0")


if __name__ == "__main__":
    unittest.main()

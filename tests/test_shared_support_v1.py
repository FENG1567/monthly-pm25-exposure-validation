"""Synthetic contract tests for shared_support_v1; no project data is read."""
from __future__ import annotations

import sys
import unittest
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "original" / "annual"))
from shared_support_v1 import annualize_exact_shared_support, join_monthly_model_oof_raw_cams


def monthly_rows(site: str, year: int, months: list[int], *, alias: str = "alias-a",
                 qa: str = "strict", estimator: str = "HGB", task: str = "T1_station",
                 fold_by_month: dict[int, str] | None = None, prediction: float = 4.0,
                 cams_value: float = 8.0, satpm_value: float = 12.0,
                 obs_value: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for month in months:
        rows.append({
            "country": "USA", "site_id": site, "year": year, "month": month,
            "obs": obs_value, "pred": prediction, "raw_cams": cams_value,
            "satpm": satpm_value, "alias": alias, "qa": qa, "profile": "historical",
            "estimator": estimator, "task": task,
            "fold_id": (fold_by_month or {}).get(month, "0"), "arm": "full",
            "target_strategy": "log_smear", "backtransform": "smearing",
        })
    frame = pd.DataFrame(rows)
    model = frame[["country", "site_id", "year", "month", "obs", "pred", "alias", "qa",
                   "profile", "estimator", "task", "fold_id", "arm", "target_strategy",
                   "backtransform"]].copy()
    cams = frame[["country", "site_id", "year", "month", "obs", "raw_cams"]].copy()
    satpm = frame[["country", "site_id", "year", "month", "obs", "satpm"]].copy()
    return model, cams, satpm


class SharedSupportTests(unittest.TestCase):
    def test_leap_year_weighting_uses_calendar_days(self) -> None:
        model, cams, satpm = monthly_rows("leap", 2024, [1, 2])
        model.loc[model.month.eq(1), "pred"] = 10.0
        model.loc[model.month.eq(2), "pred"] = 20.0
        cams.loc[cams.month.eq(1), "raw_cams"] = 30.0
        cams.loc[cams.month.eq(2), "raw_cams"] = 40.0
        satpm.loc[satpm.month.eq(1), "satpm"] = 50.0
        satpm.loc[satpm.month.eq(2), "satpm"] = 60.0

        _, annual, audit = annualize_exact_shared_support(
            model, cams, satpm, benchmark_version="V6GL03",
        )
        expected = (31 * 10.0 + 29 * 20.0) / (31 + 29)
        self.assertEqual(int(annual.iloc[0].shared_calendar_days), 60)
        self.assertEqual(annual.iloc[0].shared_month_set, (1, 2))
        self.assertAlmostEqual(annual.iloc[0].model_annual_pm25, expected)
        self.assertEqual(audit["benchmark_version"], "V6GL03")
        self.assertFalse(bool(annual.iloc[0].primary_month_support))

    def test_twelve_month_primary_ten_month_sensitivity_and_nine_excluded(self) -> None:
        parts = [monthly_rows("twelve", 2023, list(range(1, 13))),
                 monthly_rows("ten", 2023, list(range(1, 13))),
                 monthly_rows("nine", 2023, list(range(1, 13)))]
        model = pd.concat([p[0] for p in parts], ignore_index=True)
        cams = pd.concat([p[1] for p in parts], ignore_index=True)
        satpm = pd.concat([p[2] for p in parts], ignore_index=True)
        satpm = satpm[~((satpm.site_id == "ten") & ~satpm.month.isin(range(3, 13)))
                      & ~((satpm.site_id == "nine") & ~satpm.month.isin(range(1, 10)))]

        _, annual, audit = annualize_exact_shared_support(
            model, cams, satpm, benchmark_version="V6NA01",
        )
        by_site = annual.set_index("site_id")
        self.assertTrue(bool(by_site.loc["twelve", "primary_month_support"]))
        self.assertTrue(bool(by_site.loc["twelve", "sensitivity_month_support"]))
        self.assertEqual(int(by_site.loc["ten", "shared_months"]), 10)
        self.assertEqual(by_site.loc["ten", "shared_month_set"], tuple(range(3, 13)))
        self.assertFalse(bool(by_site.loc["ten", "primary_month_support"]))
        self.assertTrue(bool(by_site.loc["ten", "sensitivity_month_support"]))
        self.assertEqual(int(by_site.loc["nine", "shared_months"]), 9)
        self.assertFalse(bool(by_site.loc["nine", "sensitivity_month_support"]))
        self.assertEqual(audit["primary_annual_rows"], 1)
        self.assertEqual(audit["sensitivity_annual_rows"], 2)

    def test_satpm_observed_target_mismatch_is_rejected(self) -> None:
        model, cams, satpm = monthly_rows("target", 2023, [1, 2])
        satpm.loc[satpm.month.eq(2), "obs"] = 2.5
        with self.assertRaisesRegex(ValueError, "observed targets differ"):
            annualize_exact_shared_support(model, cams, satpm, benchmark_version="V6GL03")

    def test_raw_cams_observed_target_mismatch_is_rejected(self) -> None:
        model, cams, _ = monthly_rows("target", 2023, [1, 2])
        cams.loc[cams.month.eq(2), "obs"] = 2.5
        with self.assertRaisesRegex(ValueError, "observed targets differ"):
            join_monthly_model_oof_raw_cams(model, cams)

    def test_same_key_count_with_different_months_is_rejected(self) -> None:
        model, cams, _ = monthly_rows("months", 2023, [1, 2])
        cams = cams[cams.month.eq(2)].copy()
        extra = cams.iloc[0].copy()
        extra["month"] = 3
        cams = pd.concat([cams, extra.to_frame().T], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "monthly key sets differ"):
            join_monthly_model_oof_raw_cams(model, cams)

    def test_duplicate_month_keys_are_rejected_on_each_source(self) -> None:
        model, cams, satpm = monthly_rows("duplicate", 2023, [1, 2])
        with self.assertRaisesRegex(ValueError, "duplicate monthly rows"):
            join_monthly_model_oof_raw_cams(pd.concat([model, model.iloc[[0]]]), cams)
        with self.assertRaisesRegex(ValueError, "duplicate country/site/year/month"):
            join_monthly_model_oof_raw_cams(model, pd.concat([cams, cams.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "duplicate country/site/year/month"):
            annualize_exact_shared_support(model, cams, pd.concat([satpm, satpm.iloc[[0]]]),
                                           benchmark_version="V6GL03")

    def test_qa_estimator_and_other_model_identity_cells_stay_separate(self) -> None:
        strict = monthly_rows("same", 2023, list(range(1, 13)), alias="strict-hgb",
                              qa="strict", estimator="HGB", prediction=3.0)[0]
        sensitivity = monthly_rows("same", 2023, list(range(1, 13)), alias="sensitivity-xgb",
                                   qa="sensitivity", estimator="XGB", prediction=9.0)[0]
        _, cams, satpm = monthly_rows("same", 2023, list(range(1, 13)))
        monthly, annual, _ = annualize_exact_shared_support(
            pd.concat([strict, sensitivity], ignore_index=True), cams, satpm,
            benchmark_version="V6GL03",
        )
        self.assertEqual(len(monthly), 24)
        self.assertEqual(len(annual), 2)
        by_alias = annual.set_index("alias")
        self.assertAlmostEqual(by_alias.loc["strict-hgb", "model_annual_pm25"], 3.0)
        self.assertAlmostEqual(by_alias.loc["sensitivity-xgb", "model_annual_pm25"], 9.0)
        self.assertEqual(by_alias.loc["strict-hgb", "qa"], "strict")
        self.assertEqual(by_alias.loc["sensitivity-xgb", "estimator"], "XGB")
        self.assertTrue(by_alias.target_strategy.eq("log_smear").all())
        self.assertTrue(by_alias.backtransform.eq("smearing").all())
        self.assertTrue(by_alias.arm.eq("full").all())

    def test_station_heldout_annual_cannot_cross_folds(self) -> None:
        model, cams, _ = monthly_rows("folds", 2023, [1, 2], fold_by_month={1: "0", 2: "1"})
        with self.assertRaisesRegex(ValueError, "cross fold_id"):
            join_monthly_model_oof_raw_cams(model, cams)

    def test_t3_year_annualizes_within_its_single_heldout_year_cell(self) -> None:
        model, cams, satpm = monthly_rows("t3", 2022, list(range(1, 13)), task="T3_year_2022")
        _, annual, _ = annualize_exact_shared_support(model, cams, satpm,
                                                      benchmark_version="V6NA01")
        self.assertEqual(len(annual), 1)
        self.assertEqual(annual.iloc[0].task, "T3_year_2022")
        self.assertEqual(annual.iloc[0].fold_id, "0")
        self.assertTrue(bool(annual.iloc[0].primary_month_support))

    def test_missing_benchmark_and_2025_unavailable_are_explicit_without_zero_fill(self) -> None:
        model, cams, satpm = monthly_rows("missing", 2024, [1, 2])
        satpm = satpm[satpm.month.eq(1)]
        model25, cams25, satpm25 = monthly_rows("future", 2025, list(range(1, 13)))
        model = pd.concat([model, model25], ignore_index=True)
        cams = pd.concat([cams, cams25], ignore_index=True)
        satpm = pd.concat([satpm, satpm25], ignore_index=True)
        monthly, annual, audit = annualize_exact_shared_support(
            model, cams, satpm, benchmark_version="V6GL03",
        )
        by_site = annual.set_index("site_id")
        self.assertEqual(by_site.loc["missing", "annual_status"], "below_sensitivity_support")
        self.assertEqual(int(by_site.loc["missing", "excluded_missing_benchmark_months"]), 1)
        self.assertEqual(by_site.loc["future", "annual_status"], "benchmark_unavailable_year")
        self.assertEqual(int(by_site.loc["future", "shared_months"]), 0)
        self.assertTrue(np.isnan(by_site.loc["future", "satpm_annual_pm25"]))
        self.assertTrue(monthly.loc[monthly.year.eq(2025), "support_status"].str.contains(
            "benchmark_unavailable_year").all())
        self.assertEqual(audit["exclusions_and_support"]["finite_benchmark_rows_ignored_in_unavailable_years"], 12)

    def test_satpm_versions_cannot_be_mixed_in_one_call(self) -> None:
        model, cams, satpm = monthly_rows("versions", 2023, [1, 2])
        satpm["satpm_version"] = ["V6NA01", "V6GL03"]
        with self.assertRaisesRegex(ValueError, "must contain only the requested"):
            annualize_exact_shared_support(model, cams, satpm, benchmark_version="V6GL03")

    def test_missing_prediction_is_excluded_not_zero_filled(self) -> None:
        model, cams, satpm = monthly_rows("nan", 2023, [1, 2])
        model.loc[model.month.eq(2), "pred"] = np.nan
        monthly, annual, _ = annualize_exact_shared_support(model, cams, satpm,
                                                            benchmark_version="V6NA01")
        self.assertEqual(int(annual.iloc[0].shared_months), 1)
        self.assertEqual(int(annual.iloc[0].excluded_missing_model_prediction_months), 1)
        self.assertTrue(np.isnan(monthly.loc[monthly.month.eq(2), "model_prediction"].iloc[0]))
        self.assertAlmostEqual(annual.iloc[0].model_annual_pm25, 4.0)

    def test_calendar_support_and_day_count_use_actual_months(self) -> None:
        model, cams, satpm = monthly_rows("months", 2024, [2, 4])
        _, annual, _ = annualize_exact_shared_support(model, cams, satpm,
                                                      benchmark_version="V6NA01")
        self.assertEqual(int(annual.iloc[0].shared_calendar_days),
                         monthrange(2024, 2)[1] + monthrange(2024, 4)[1])


if __name__ == "__main__":
    unittest.main()

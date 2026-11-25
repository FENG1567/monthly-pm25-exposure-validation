"""Focused checks for source-replay input and labeling helpers."""
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pm25_repro.figures import (
    _draw_f3,
    _draw_s6,
    _format_number,
    _horizontal_estimates,
    _panel_path,
    _row_labels,
    _save_outputs,
    _s6_labels,
    _short_caption,
    _value_labels,
)


class FigureReplayHelperTests(unittest.TestCase):
    def test_panel_path_is_limited_to_public_panel_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panels = root / "data" / "summary" / "figure_panels"
            panels.mkdir(parents=True)
            (panels / "panel.csv").write_text("value\n1\n", encoding="utf-8")
            self.assertEqual(_panel_path(root, "data/summary/figure_panels/panel.csv"), panels / "panel.csv")
            with self.assertRaises(ValueError):
                _panel_path(root, "data/summary/figure_panels/../../private.csv")
            with self.assertRaises(ValueError):
                _panel_path(root, str(panels / "panel.csv"))

    def test_number_format_keeps_sign_and_marks_nonfinite_as_na(self):
        self.assertEqual(_format_number(-0.125, 3), "-0.125")
        self.assertEqual(_format_number(float("nan")), "NA")
        self.assertEqual(_format_number("not numeric"), "NA")

    def test_duplicate_row_labels_are_retained_and_distinguished(self):
        frame = pd.DataFrame({"row_label": ["USA · T1", "USA · T1", "CAN · T2"]})
        self.assertEqual(_row_labels(frame), ["USA · T1", "USA · T1 [2]", "CAN · T2"])

    def test_caption_uses_wrapped_first_sentence(self):
        caption = "Figure1 | " + "A detailed first sentence " * 5 + ". The remaining caption is long."
        short = _short_caption(caption)
        self.assertNotIn("remaining caption", short)
        self.assertTrue(all(len(line) <= 80 for line in short.splitlines()))

    def test_value_labels_add_explicit_strata_to_row_label(self):
        frame = pd.DataFrame({
            "row_label": ["CAN 2023 with met", "CAN 2023 with met"],
            "profile": ["with_met", "with_met"],
            "fire_month_scope": ["P90_fire_proxy", "other_months"],
        })
        self.assertEqual(
            _value_labels(frame, include=("row_label", "profile", "fire_month_scope")),
            ["CAN 2023 with met · P90_fire_proxy", "CAN 2023 with met · other_months"],
        )

    def test_figure3_shows_fire_month_and_support_rule_labels(self):
        panel_a = pd.DataFrame({
            "row_label": ["CAN 2023 with met", "CAN 2023 with met"],
            "profile": ["with_met", "with_met"],
            "fire_month_scope": ["P90_fire_proxy", "other_months"],
            "full_rmse": [11.7, 1.8], "n": [466, 725], "n_stations": [35, 38],
        })
        panel_b = pd.DataFrame({
            "row_label": ["CAN 2023 with met", "CAN 2023 with met"],
            "profile": ["with_met", "with_met"],
            "rule": ["distance_50km_non_geo_minmax", "distance_50km_non_geo_p01_p99"],
            "rule_label": ["50 km + min–max", "50 km + P01–P99"],
            "row_coverage_fraction": [.643, .417], "n": [466, 302],
            "eligible_rows": [725, 725], "n_stations": [35, 38],
        })
        fig = _draw_f3({"a": panel_a, "b": panel_b}, "Figure4 | Short caption. Full details.")
        try:
            labels_a = [label.get_text() for label in fig.axes[0].get_yticklabels()]
            labels_b = [label.get_text() for label in fig.axes[1].get_yticklabels()]
            self.assertTrue(any("P90_fire_proxy" in label for label in labels_a))
            self.assertTrue(any("other_months" in label for label in labels_a))
            self.assertTrue(any("min–max" in label for label in labels_b))
            self.assertTrue(any("P01–P99" in label for label in labels_b))
        finally:
            plt.close(fig)

    def test_s6_labels_keep_compact_scientific_strata(self):
        frame = pd.DataFrame({
            "model_scope": ["PM_future"], "country": ["USA"], "year": [2023],
            "parameter_code": ["not_applicable"], "method_code": ["not_applicable"],
            "alignment": ["same_named_date"], "plume_definition": ["any_plume"],
            "profile": ["with_met"], "gate_quantile": [.9], "n_pairs": [120],
        })
        label = _s6_labels(frame, support=(("n_pairs", "pairs"),))[0]
        for term in ("parameter=n/a", "method=n/a", "date=same date", "profile=MET", "gate=P90", "pairs=120"):
            self.assertIn(term, label)

    def test_s6_layout_is_vertical_wide_and_scales_with_rows(self):
        n = 64
        shared = {
            "model_scope": ["PM_future"] * n, "country": ["USA"] * n,
            "year": [2023] * n, "parameter_code": ["not_applicable"] * n,
            "method_code": ["not_applicable"] * n, "alignment": ["same_named_date"] * n,
            "plume_definition": ["any_plume"] * n, "profile": ["with_met"] * n,
            "gate_quantile": [.9] * n,
        }
        panel_a = pd.DataFrame({
            **shared, "n_pairs": [120] * n, "sites": [40] * n,
            "pearson_r": [i / n for i in range(n)], "ci_low": [0.1] * n,
            "ci_high": [0.8] * n,
        })
        panel_b = pd.DataFrame({
            **shared, "source_rows": [120] * n, "geometry_matched_rows": [100] * n,
            "known_fraction": [.8] * n, "retained_fraction": [.7] * n,
        })
        fig = _draw_s6({"a": panel_a, "b": panel_b}, "Figure S6 | A short caption. Full details.")
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertAlmostEqual(fig.get_size_inches()[0], 12.0)
            self.assertGreater(fig.get_size_inches()[1], 12.5)
            self.assertGreater(fig.axes[0].get_position().y0, fig.axes[1].get_position().y0)
        finally:
            plt.close(fig)

    def test_unrecognized_marker_groups_use_valid_markers(self):
        fig, ax = plt.subplots()
        try:
            frame = pd.DataFrame({"value": [.1, .2], "group": ["custom_a", "custom_b"]})
            _horizontal_estimates(ax, frame, "value", ["a", "b"], marker_col="group")
            self.assertEqual(len(ax.collections), 2)
        finally:
            plt.close(fig)

    def test_vector_exports_have_stable_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            try:
                outputs = []
                for name in ("first", "second"):
                    folder = root / name
                    folder.mkdir()
                    paths = {kind: folder / f"plot.{kind}" for kind in ("pdf", "png", "svg")}
                    _save_outputs(fig, paths)
                    outputs.append(paths)
                for kind in ("pdf", "svg"):
                    self.assertEqual(outputs[0][kind].read_bytes(), outputs[1][kind].read_bytes())
            finally:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()


class CountryAxisTests(unittest.TestCase):
    def test_s1_shared_axes_keep_country_identity_when_profile_order_differs(self):
        from pm25_repro.figures import _draw_s1, plt
        frame=pd.DataFrame([
            {'country':'PT','profile':'no_met','r2':-13.4,'qa_label':'strict','estimator_label':'hgb','target_strategy':'log_smear','backtransform':'smearing'},
            {'country':'CAN','profile':'no_met','r2':.7,'qa_label':'strict','estimator_label':'hgb','target_strategy':'log_smear','backtransform':'smearing'},
            {'country':'CAN','profile':'with_met','r2':.75,'qa_label':'strict','estimator_label':'hgb','target_strategy':'log_smear','backtransform':'smearing'},
            {'country':'PT','profile':'with_met','r2':-14.,'qa_label':'strict','estimator_label':'hgb','target_strategy':'log_smear','backtransform':'smearing'},
        ])
        fig=_draw_s1({'a':frame,'b':frame},'Country ordering test.')
        try:
            self.assertEqual([x.get_text() for x in fig.axes[0].get_yticklabels()],['CAN','PT'])
            for axis in fig.axes:
                for collection in axis.collections:
                    x,y=collection.get_offsets()[0]
                    self.assertAlmostEqual(round(float(y)),1 if x<0 else 0)
        finally:plt.close(fig)

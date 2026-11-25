"""Focused checks for safe analysis replay planning and input validation."""
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pm25_repro.analysis import (  # noqa: E402
    AnalysisConfigError,
    build_analysis_command,
    run_analysis,
    self_check,
)


class AnalysisReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "analysis/original/historical/validation").mkdir(parents=True)
        (self.root / "analysis/original/portugal/validation").mkdir(parents=True)
        (self.root / "analysis/original/loco/validation").mkdir(parents=True)
        (self.root / "analysis/original/extras").mkdir(parents=True)
        (self.root / "analysis/original/historical/checks").mkdir(parents=True)
        for relative in (
            "analysis/original/historical/validation/run_validation.py",
            "analysis/original/historical/validation/validation_core.py",
            "analysis/original/portugal/validation/run_validation.py",
            "analysis/original/historical/validation/run_external_year.py",
            "analysis/original/historical/checks/multiplicity_final.py",
            "analysis/original/loco/validation/run_loco.py",
            "analysis/original/extras/run_station_thinning.py",
        ):
            (self.root / relative).write_text("# test runner\n"+'\n'.join("parser.add_argument("+repr(flag)+")" for flag in ['--country','--training-features','--external-features','--out','--manifest','--workspace','--registry','--output-root','--execute','--run']), encoding="utf-8")
        (self.root / "data/source").mkdir(parents=True)
        (self.root / "data/features.csv").write_text("x\n1\n", encoding="utf-8")
        (self.root / "data/manifest.json").write_text("{}\n", encoding="utf-8")

    def write_config(self, components, *, output_root="results/analysis-replay"):
        path = self.root / "analysis.json"
        path.write_text(
            json.dumps({"version": 1, "output_root": output_root, "components": components}),
            encoding="utf-8",
        )
        return path

    def historical_settings(self, *, country="USA", feature_path="data/features.csv", meteorology="source_declared"):
        return {
            "source_root": "data/source",
            "countries": [country],
            "feature_files": {country: {"path": feature_path}},
            "meteorology": meteorology,
        }

    def test_published_template_is_valid_json_and_uses_source_declared_defaults(self):
        template = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/analysis.template.json").read_text(
                encoding="utf-8",
            )
        )

        self.assertEqual(template["version"], 1)
        self.assertEqual(template["components"]["historical"]["meteorology"], "source_declared")
        self.assertEqual(template["components"]["portugal"]["meteorology"], "source_declared")
        self.assertIn("validation_dir", template["components"]["thinning"])

    def test_historical_dry_run_builds_original_command_without_creating_output(self):
        config = self.write_config({"historical": self.historical_settings()})

        plan = run_analysis(self.root, config, "historical")

        self.assertFalse(plan["executed"])
        self.assertEqual(plan["runner"], "analysis/original/historical/validation/run_validation.py")
        self.assertIn("--feature-override", plan["command"])
        self.assertFalse((self.root / "results/analysis-replay/historical").exists())

    def test_portugal_source_declared_matches_older_runner_cli(self):
        config = self.write_config({
            "portugal": self.historical_settings(country="PT", meteorology="source_declared")
        })

        plan = build_analysis_command(self.root, config, "portugal")

        self.assertEqual(plan["runner"], "analysis/original/portugal/validation/run_validation.py")
        self.assertNotIn("--met-profile", plan["command"])

    def test_portugal_no_met_is_supported_but_era5_flag_is_rejected(self):
        no_met_config = self.write_config({
            "portugal": self.historical_settings(country="PT", meteorology="no_met")
        })
        plan = build_analysis_command(self.root, no_met_config, "portugal")
        self.assertIn("--no-met", plan["command"])

        era5_config = self.write_config({
            "portugal": self.historical_settings(country="PT", meteorology="era5_six")
        })
        with self.assertRaisesRegex(AnalysisConfigError, "does not support --met-profile era5_six"):
            build_analysis_command(self.root, era5_config, "portugal")

    def test_missing_feature_input_fails_instead_of_using_source_defaults(self):
        config = self.write_config({
            "historical": self.historical_settings(feature_path="data/not-provided.csv")
        })

        with self.assertRaisesRegex(AnalysisConfigError, "Missing historical feature file for USA"):
            run_analysis(self.root, config, "historical")

    def test_missing_input_has_cli_exit_code_2(self):
        (self.root / "data/summary").mkdir(parents=True)
        (self.root / "data/summary/table_registry.json").write_text("{}\n", encoding="utf-8")
        config = self.write_config({
            "historical": self.historical_settings(feature_path="data/not-provided.csv")
        })
        from pm25_repro.__main__ import main

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main([
                "--repo-root", str(self.root), "analysis", "--component", "historical",
                "--config", str(config),
            ])

        self.assertEqual(code, 2)
        self.assertIn("Missing historical feature file for USA", stderr.getvalue())

    def test_repo_reference_is_validated_and_resolved_for_thinning_manifest(self):
        config = self.write_config({"thinning": {
            "validation_dir": "analysis/original/historical/validation",
            "arguments": ["--manifest", "@repo/data/manifest.json"],
        }})

        plan = build_analysis_command(self.root, config, "thinning")

        input_index = plan["command"].index("--manifest") + 1
        self.assertEqual(plan["command"][input_index], str(self.root / "data/manifest.json"))
        self.assertIn("--output-root", plan["command"])
        self.assertEqual(
            plan["environment"]["CAND1_FINAL_VALIDATION_DIR"],
            str(self.root / "analysis/original/historical/validation"),
        )
        self.assertEqual(plan["runner"], "analysis/original/extras/run_station_thinning.py")

    def test_raw_component_without_existing_input_fails(self):
        config = self.write_config({"thinning": {
            "arguments": ["--manifest", "data/not-provided.json"],
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "Missing thinning input"):
            build_analysis_command(self.root, config, "thinning")

    def test_multiplicity_requires_explicit_workspace_and_registry_and_maps_output(self):
        with self.assertRaisesRegex(AnalysisConfigError, "must provide --workspace exactly once"):
            build_analysis_command(self.root, self.write_config({"multiplicity": {
                "runner": "analysis/original/historical/checks/multiplicity_final.py",
                "arguments": ["--registry", "@repo/data/manifest.json"],
            }}), "multiplicity")

        config = self.write_config({"multiplicity": {
            "runner": "analysis/original/historical/checks/multiplicity_final.py",
            "arguments": [
                "--workspace", "@repo/.", "--registry", "@repo/data/manifest.json",
            ],
        }})
        plan = build_analysis_command(self.root, config, "multiplicity")
        self.assertIn("--out-root", plan["command"])
        registry_index = plan["command"].index("--registry") + 1
        self.assertEqual(plan["command"][registry_index], str(self.root / "data/manifest.json"))

    def test_fixed_path_scripts_are_refused_until_they_have_portable_clis(self):
        for component in ("monthly-comparison", "pt-diagnostic"):
            config = self.write_config({component: {"arguments": []}})
            with self.assertRaisesRegex(AnalysisConfigError, "no portable CLI"):
                build_analysis_command(self.root, config, component)

    def test_future_requires_explicit_existing_feature_inputs(self):
        config = self.write_config({"future": {
            "arguments": [
                "--country", "USA", "--training-features", "data/features.csv",
                "--external-features", "data/missing.csv",
            ],
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "Missing future --external-features input"):
            build_analysis_command(self.root, config, "future")

    def test_future_rejects_equals_form_path_override(self):
        config = self.write_config({"future": {
            "arguments": [
                "--country", "USA", "--training-features", "data/features.csv",
                "--external-features", "data/features.csv",
                "--external-features=/private/override.csv",
            ],
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "--flag=value is not allowed"):
            build_analysis_command(self.root, config, "future")

    def test_loco_manifest_nested_paths_must_be_local_and_exist(self):
        manifest_path = self.root / "data/loco.json"
        manifest_path.write_text(json.dumps({
            "format": "repair-v2-loco-input-manifest-v1",
            "countries": [{"feature_path": "C:/private/features.csv"}],
        }), encoding="utf-8")
        config = self.write_config({"loco": {
            "manifest": {"path": "data/loco.json"},
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "feature_path must be repository-relative"):
            build_analysis_command(self.root, config, "loco")

    def test_manifest_url_metadata_is_not_treated_as_a_local_path(self):
        manifest_path = self.root / "data/loco.json"
        manifest_path.write_text(json.dumps({
            "source_url": "https://data.example.test/source.csv",
            "metadata_url": "https://data.example.test/metadata.json",
        }), encoding="utf-8")
        config = self.write_config({"loco": {
            "manifest": {"path": "data/loco.json"},
        }})

        plan = build_analysis_command(self.root, config, "loco")

        self.assertEqual(plan["runner"], "analysis/original/loco/validation/run_loco.py")

    def test_thinning_manifest_nested_paths_must_resolve_within_repository(self):
        manifest_path = self.root / "data/manifest.json"
        manifest_path.write_text(json.dumps({
            "candidate_manifest": {"path": str(self.root / "data/features.csv")},
        }), encoding="utf-8")
        config = self.write_config({"thinning": {
            "validation_dir": "analysis/original/historical/validation",
            "arguments": ["--manifest", "@repo/data/manifest.json"],
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "input manifest path must be repository-relative"):
            build_analysis_command(self.root, config, "thinning")

    def test_multiplicity_nested_paths_resolve_from_workspace_and_reject_traversal(self):
        registry_path = self.root / "data/manifest.json"
        registry_path.write_text(json.dumps({"protocol_path": "../private/protocol.json"}), encoding="utf-8")
        config = self.write_config({"multiplicity": {
            "runner": "analysis/original/historical/checks/multiplicity_final.py",
            "arguments": [
                "--workspace", "@repo/.", "--registry", "@repo/data/manifest.json",
            ],
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "protocol_path cannot traverse outside its manifest directory"):
            build_analysis_command(self.root, config, "multiplicity")

    def test_output_root_must_stay_under_results_outside_reference(self):
        components = {"historical": self.historical_settings()}
        outside_config = self.write_config(components, output_root="analysis/original")
        with self.assertRaisesRegex(AnalysisConfigError, "output_root must be inside results/"):
            build_analysis_command(self.root, outside_config, "historical")

        reference_config = self.write_config(components, output_root="results/reference")
        with self.assertRaisesRegex(AnalysisConfigError, "cannot be inside results/reference"):
            build_analysis_command(self.root, reference_config, "historical")

    def test_raw_equals_form_output_or_input_override_is_rejected(self):
        config = self.write_config({"thinning": {
            "validation_dir": "analysis/original/historical/validation",
            "arguments": [
                "--manifest", "@repo/data/manifest.json",
                "--manifest=/private/override.json",
            ],
        }})

        with self.assertRaisesRegex(AnalysisConfigError, "--flag=value is not allowed"):
            build_analysis_command(self.root, config, "thinning")

    def test_self_check_keeps_historical_and_portugal_runner_snapshots_separate(self):
        snapshots = []
        for relative in (
            "analysis/original/historical/validation/run_validation.py",
            "analysis/original/portugal/validation/run_validation.py",
        ):
            path = self.root / relative
            snapshots.append({
                "file": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "original_sha256": "0" * 64,
            })
        (self.root / "analysis/original/SOURCE_SNAPSHOTS.json").write_text(
            json.dumps({"files": snapshots}), encoding="utf-8",
        )
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        with patch("pm25_repro.analysis.subprocess.run", side_effect=fake_run):
            result = self_check(self.root)

        self.assertTrue(result["ok"])
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["source_manifest"]["original_sha256_compared"])
        self.assertEqual(len(calls), 2)
        self.assertIn("historical/validation/run_validation.py", calls[0][0][1].replace("\\", "/"))
        self.assertIn("portugal/validation/run_validation.py", calls[1][0][1].replace("\\", "/"))
        self.assertEqual(calls[0][0][-1], "--self-check-only")
        self.assertEqual(calls[1][0][-1], "--self-check-only")

    def test_self_check_skips_runner_when_published_hash_is_wrong(self):
        snapshots_path = self.root / "analysis/original/SOURCE_SNAPSHOTS.json"
        snapshots_path.write_text(json.dumps({"files": [{
            "file": "analysis/original/historical/validation/run_validation.py",
            "sha256": "0" * 64,
            "original_sha256": "0" * 64,
        }]}), encoding="utf-8")
        with patch("pm25_repro.analysis.subprocess.run") as run:
            result = self_check(self.root)

        self.assertFalse(result["ok"])
        self.assertEqual(result["returncode"], 2)
        self.assertIn("Published source SHA-256 mismatch", result["source_manifest"]["error"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

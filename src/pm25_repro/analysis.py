"""Portable adapters for replaying the package's original analysis runners.

This module only validates inputs and builds subprocess argument vectors. It
does not reimplement model fitting or select a transformation from its score.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from typing import Any, Mapping, Sequence


_COMPONENTS = (
    "historical", "portugal", "future", "loco", "multiplicity", "thinning",
    "monthly-comparison", "pt-diagnostic",
)
_DEFAULT_RUNNERS = {
    "historical": "analysis/original/historical/validation/run_validation.py",
    "portugal": "analysis/original/portugal/validation/run_validation.py",
    "future": "analysis/original/historical/validation/run_external_year.py",
    "loco": "analysis/original/loco/validation/run_loco.py",
    "thinning": "analysis/original/extras/run_station_thinning.py",
    "monthly-comparison": "analysis/original/extras/run_monthly_benchmark.py",
    "pt-diagnostic": "analysis/original/extras/pt_tail_diagnostic.py",
}
_RAW_COMPONENTS = {"multiplicity", "thinning", "monthly-comparison", "pt-diagnostic"}
_RAW_UNSUPPORTED_COMPONENTS = {"monthly-comparison", "pt-diagnostic"}
_RAW_OUTPUT_FLAGS = {
    "multiplicity": {"--out", "--out-root"},
    "thinning": {"--output-root"},
}
_RAW_DEFAULT_OUTPUT_FLAGS = {"multiplicity": "--out-root", "thinning": "--output-root"}
_RAW_REQUIRED_INPUT_FLAGS = {
    "multiplicity": {"--workspace", "--registry"},
    "thinning": {"--manifest"},
}
_FUTURE_INPUT_FLAGS = {
    "--training-features", "--external-features", "--training-target-override",
    "--external-target-override", "--training-input-provenance",
    "--external-input-provenance",
}
_REPO_PATH_PREFIXES = ("@repo/", "repo:")


class AnalysisConfigError(ValueError):
    """Raised when the portable replay configuration is incomplete or unsafe."""


def _repo_path(repo_root: Path, raw: str, label: str, *, must_exist: bool = True) -> Path:
    """Resolve a repo-relative path and reject absolute paths and traversal."""
    if not isinstance(raw, str) or not raw.strip():
        raise AnalysisConfigError(f"{label} must be a nonempty repo-relative path")
    raw = raw.strip()
    candidate = Path(raw)
    win_candidate = PureWindowsPath(raw)
    if candidate.is_absolute() or win_candidate.is_absolute() or win_candidate.drive:
        raise AnalysisConfigError(f"{label} must be relative to the repository: {raw}")
    if ".." in candidate.parts or ".." in win_candidate.parts:
        raise AnalysisConfigError(f"{label} cannot traverse outside the repository: {raw}")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise AnalysisConfigError(f"{label} resolves outside the repository: {raw}")
    if must_exist and not resolved.exists():
        raise AnalysisConfigError(f"Missing {label}: {raw} (expected at {resolved})")
    return resolved


def _load_config(repo_root: Path, config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Read JSON configuration; a relative config path is rooted at repo_root."""
    raw_path = Path(config_path)
    path = raw_path.resolve() if raw_path.is_absolute() else (repo_root / raw_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisConfigError(f"Analysis config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisConfigError(f"Invalid JSON in analysis config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisConfigError("Analysis config must contain a JSON object")
    if value.get("version") != 1:
        raise AnalysisConfigError("Analysis config must declare version: 1")
    return value, path


def _input_entry(
    repo_root: Path,
    entries: Mapping[str, Any],
    key: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if key not in entries:
        raise AnalysisConfigError(f"Missing configured {label} for {key}; no source fallback is used")
    entry = entries[key]
    if isinstance(entry, str):
        entry = {"path": entry}
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise AnalysisConfigError(f"{label} entry for {key} must be a path or an object with path")
    path = _repo_path(repo_root, entry["path"], f"{label} for {key}")
    digest = entry.get("sha256")
    if digest is not None:
        expected = str(digest).lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise AnalysisConfigError(f"Invalid SHA-256 for {label} {key}: expected 64 hex characters")
        actual = _sha256(path)
        if actual != expected:
            raise AnalysisConfigError(
                f"SHA-256 mismatch for {label} {key}: configured {expected}, observed {actual}"
            )
    return path, entry


def _resolve_manifest_reference(
    repo_root: Path, base_dir: Path, raw: Any, label: str,
) -> Path:
    """Resolve a path declared inside an input manifest, within this checkout."""
    if not isinstance(raw, str) or not raw.strip():
        raise AnalysisConfigError(f"{label} must be a nonempty repository-relative path")
    value = raw.strip()
    candidate = Path(value)
    win_candidate = PureWindowsPath(value)
    if (
        value.startswith("~") or
        candidate.is_absolute() or candidate.expanduser().is_absolute()
        or win_candidate.is_absolute() or win_candidate.drive
    ):
        raise AnalysisConfigError(f"{label} must be repository-relative, not absolute: {value}")
    if ".." in candidate.parts or ".." in win_candidate.parts:
        raise AnalysisConfigError(f"{label} cannot traverse outside its manifest directory: {value}")
    root = repo_root.resolve()
    try:
        resolved = (base_dir / candidate).resolve(strict=True)
    except FileNotFoundError as exc:
        raise AnalysisConfigError(f"Missing {label}: {value}") from exc
    if resolved != root and root not in resolved.parents:
        raise AnalysisConfigError(f"{label} resolves outside the repository: {value}")
    return resolved


def _validate_manifest_path_tree(
    repo_root: Path,
    manifest_path: Path,
    label: str,
    *,
    reference_base: Path | None = None,
) -> None:
    """Validate known path references recursively through JSON input manifests.

    LOCO and thinning references are relative to the JSON file that declares
    them. Multiplicity references are all rooted at its explicit workspace,
    matching the archived runner's path resolution. Only `path` and `*_path`
    keys are treated as paths; fields such as `source_url` remain metadata.
    """
    root = repo_root.resolve()
    current = manifest_path.resolve(strict=True)
    if current != root and root not in current.parents:
        raise AnalysisConfigError(f"{label} resolves outside the repository: {manifest_path}")
    visited: set[Path] = set()

    def visit_json(path: Path, depth: int = 0, *, force: bool = False) -> None:
        if path in visited or not path.is_file():
            return
        if path.suffix.lower() != '.json' and not force:
            # Archived readers accept JSON under arbitrary filenames, including
            # references shaped as candidate_manifest: {path: candidate.dat}.
            with path.open('rb') as stream:
                prefix=stream.read(256).lstrip(b'\xef\xbb\xbf \r\n\t')
            if not prefix.startswith((b'{',b'[')):
                return
        if depth > 32:
            raise AnalysisConfigError(f"{label} references exceed the 32-level JSON nesting limit")
        if len(visited) >= 4096:
            raise AnalysisConfigError(f"{label} references exceed the 4096-file validation limit")
        visited.add(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisConfigError(f"Cannot read referenced JSON in {label} at {path}: {exc}") from exc
        base = reference_base.resolve() if reference_base is not None else path.parent

        def walk(node: Any, semantic_context: bool = False) -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    is_path_key = isinstance(key, str) and (key == "path" or key.endswith("_path"))
                    child_context = semantic_context or (isinstance(key,str) and ('manifest' in key or 'provenance' in key))
                    if is_path_key and isinstance(child, str):
                        referenced = _resolve_manifest_reference(
                            root, base, child, f"{label} {key}",
                        )
                        visit_json(referenced, depth + 1, force=child_context)
                    elif is_path_key and isinstance(child, list):
                        for item in child:
                            if isinstance(item, str):
                                referenced = _resolve_manifest_reference(
                                    root, base, item, f"{label} {key}",
                                )
                                visit_json(referenced, depth + 1, force=child_context)
                    walk(child,child_context)
            elif isinstance(node, list):
                for item in node:
                    walk(item,semantic_context)

        walk(value)

    visit_json(current, force=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runner_path(repo_root: Path, component: str, settings: Mapping[str, Any]) -> Path:
    raw = settings.get("runner", _DEFAULT_RUNNERS.get(component))
    if not raw:
        raise AnalysisConfigError(
            f"Component {component!r} requires an explicit runner path in the config"
        )
    return _repo_path(repo_root, str(raw), f"{component} runner")


def _validate_original_flags(runner: Path, arguments: Sequence[str]) -> None:
    """Reject argparse abbreviations as well as unknown or equals-form flags."""
    tree=ast.parse(runner.read_text(encoding='utf-8-sig'))
    allowed={'--help'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            allowed.update(arg.value for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str) and arg.value.startswith('--'))
    invalid=[arg for arg in arguments if arg.startswith('--') and arg not in allowed]
    if invalid:
        raise AnalysisConfigError(f'Only complete original runner option names are allowed; unknown, abbreviated or equals-form option(s): {invalid}')


def _output_path(repo_root: Path, config: Mapping[str, Any], component: str) -> Path:
    raw_root = config.get("output_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise AnalysisConfigError("Config must set output_root to a repo-relative output directory")
    root = _repo_path(repo_root, raw_root, "output_root", must_exist=False)
    checkout = repo_root.resolve()
    results_root = (checkout / "results").resolve()
    if results_root != checkout and checkout not in results_root.parents:
        raise AnalysisConfigError("Repository results directory resolves outside the repository")
    if root != results_root and results_root not in root.parents:
        raise AnalysisConfigError("output_root must be inside results/")
    reference_root = (results_root / "reference").resolve()
    if root == reference_root or reference_root in root.parents:
        raise AnalysisConfigError("output_root cannot be inside results/reference/")
    output = (root / component).resolve()
    if root != output and root not in output.parents:
        raise AnalysisConfigError("Component output directory resolves outside output_root")
    return output


def _check_auxiliary_inputs(repo_root: Path, entry: Mapping[str, Any], country: str, label: str) -> list[str]:
    """Resolve optional per-country target/provenance paths and verify hashes."""
    paths: list[str] = []
    for field, flag in (("target_files", "--target-override"), ("provenance_files", "--input-provenance")):
        group = entry.get(field, {})
        if group is None:
            continue
        if not isinstance(group, dict):
            raise AnalysisConfigError(f"{label}.{field} must be an object keyed by country")
        if country in group:
            path, _ = _input_entry(repo_root, group, country, field.replace("_", " "))
            paths.extend((flag, f"{country}={path}"))
    return paths


def _build_historical(
    repo_root: Path, component: str, settings: Mapping[str, Any], output: Path,
) -> tuple[Path, Path, list[str]]:
    runner = _runner_path(repo_root, component, settings)
    countries_raw = settings.get("countries")
    if not isinstance(countries_raw, list) or not countries_raw:
        raise AnalysisConfigError(f"{component} requires a nonempty countries list")
    countries = [str(country).strip().upper() for country in countries_raw]
    if any(not code for code in countries) or len(set(countries)) != len(countries):
        raise AnalysisConfigError(f"{component}.countries must contain unique nonempty country codes")
    feature_files = settings.get("feature_files")
    if not isinstance(feature_files, dict):
        raise AnalysisConfigError(f"{component} requires feature_files keyed by country")
    args: list[str] = [
        "--source-root", str(_repo_path(repo_root, str(settings.get("source_root", "")), f"{component} source_root")),
        "--out", str(output), "--countries", ",".join(countries),
    ]
    # Require one explicit feature table for every requested country. This
    # prevents the original runner's source-tree defaults from filling gaps.
    for country in countries:
        feature, _ = _input_entry(repo_root, feature_files, country, f"{component} feature file")
        args.extend(("--feature-override", f"{country}={feature}"))
        args.extend(_check_auxiliary_inputs(repo_root, settings, country, component))
    args.extend(("--max-iter", str(settings.get("max_iter", 400))))
    args.extend(("--estimator-kind", str(settings.get("estimator_kind", "hgb"))))
    args.extend(("--smear-strategies", str(settings.get("smear_strategies", "crossfit_station,in_sample"))))
    args.extend(("--bootstrap-replicates", str(settings.get("bootstrap_replicates", 500))))
    args.extend(("--bootstrap-engine", str(settings.get("bootstrap_engine", "cluster_stats"))))
    meteorology = settings.get("meteorology")
    if not isinstance(meteorology, str):
        raise AnalysisConfigError(
            f"{component}.meteorology must explicitly be no_met or source_declared; "
            "era5_six is supported by the historical runner only"
        )
    if meteorology == "no_met":
        args.append("--no-met")
    elif meteorology == "source_declared" and component == "portugal":
        # The archived Portugal runner predates --met-profile. Its default is
        # the source-declared feature set; passing the newer flag would make
        # its own CLI reject the command.
        pass
    elif meteorology in {"era5_six", "source_declared"} and component == "historical":
        args.extend(("--met-profile", str(meteorology)))
    elif meteorology == "era5_six" and component == "portugal":
        raise AnalysisConfigError(
            "portugal runner snapshot does not support --met-profile era5_six; "
            "use source_declared or no_met"
        )
    else:
        raise AnalysisConfigError(
            f"{component}.meteorology must explicitly be no_met or source_declared; "
            "era5_six is supported by the historical runner only"
        )
    return runner, runner.parent, args


def _decode_repo_reference(repo_root: Path, value: str, label: str) -> str:
    """Expand @repo/path or repo:path references in raw runner argument lists."""
    raw = value
    for prefix in _REPO_PATH_PREFIXES:
        if raw.startswith(prefix):
            suffix = raw[len(prefix):]
            # Keep the decoded value relative until the component builder
            # resolves it. This lets raw-argument builders apply the same
            # existence and containment checks as ordinary relative paths.
            return _repo_path(repo_root, suffix, label).relative_to(repo_root.resolve()).as_posix()
    if Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise AnalysisConfigError(f"Absolute paths are not portable in analysis config ({label}): {raw}")
    return raw


def _build_future(
    repo_root: Path, settings: Mapping[str, Any], output: Path,
) -> tuple[Path, Path, list[str]]:
    runner = _runner_path(repo_root, "future", settings)
    raw_args = settings.get("arguments")
    if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
        raise AnalysisConfigError("future.arguments must be an array of original CLI argument strings")
    equals_flags = [item for item in raw_args if item.startswith("--") and "=" in item]
    if equals_flags:
        raise AnalysisConfigError(
            f"future.arguments must use separate flag/value items; --flag=value is not allowed: {equals_flags}"
        )
    args = [_decode_repo_reference(repo_root, item, "future argument") for item in raw_args]
    _validate_original_flags(runner,args)
    seen_paths: set[str] = set()
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in _FUTURE_INPUT_FLAGS:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise AnalysisConfigError(f"future arguments require a value after {flag}")
            if flag in seen_paths:
                raise AnalysisConfigError(f"future arguments contain repeated {flag}")
            seen_paths.add(flag)
            value = args[index + 1]
            # Raw paths are repo-relative, and must name files explicitly.
            path = Path(value)
            if not path.is_absolute() and not PureWindowsPath(value).drive:
                resolved = _repo_path(repo_root, value, f"future {flag} input")
                args[index + 1] = str(resolved)
        index += 1
    required = {"--country", "--training-features", "--external-features"}
    missing = sorted(flag for flag in required if args.count(flag) != 1)
    if missing:
        raise AnalysisConfigError(
            f"future.arguments must provide each required original flag exactly once: {missing}"
        )
    for flag in required:
        index = args.index(flag)
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            raise AnalysisConfigError(f"future.arguments requires a value after {flag}")
    if "--out" in args:
        if args.count("--out") != 1:
            raise AnalysisConfigError("future.arguments may contain --out only once")
        out_index = args.index("--out")
        if out_index + 1 >= len(args):
            raise AnalysisConfigError("future.arguments requires a value after --out")
        args[out_index + 1] = str(output)
    else:
        args.extend(("--out", str(output)))
    return runner, runner.parent, args


def _build_loco(
    repo_root: Path, settings: Mapping[str, Any], output: Path,
) -> tuple[Path, Path, list[str]]:
    runner = _runner_path(repo_root, "loco", settings)
    manifest, _ = _input_entry(
        repo_root, {"manifest": settings.get("manifest")}, "manifest", "LOCO manifest"
    )
    _validate_manifest_path_tree(repo_root, manifest, "LOCO input manifest")
    args = ["--manifest", str(manifest), "--out", str(output)]
    if bool(settings.get("include_ridge", False)):
        args.append("--include-ridge")
    if bool(settings.get("run", False)):
        args.append("--run")
    args.extend(("--bootstrap-replicates", str(settings.get("bootstrap_replicates", 500))))
    return runner, runner.parent, args


def _build_raw(
    repo_root: Path, component: str, settings: Mapping[str, Any], output: Path,
) -> tuple[Path, Path, list[str]]:
    if component in _RAW_UNSUPPORTED_COMPONENTS:
        raise AnalysisConfigError(
            f"{component} cannot be safely replayed: its archived script has no portable CLI "
            "for explicit inputs and outputs; do not execute its fixed-path defaults"
        )
    runner = _runner_path(repo_root, component, settings)
    raw_args = settings.get("arguments")
    if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
        raise AnalysisConfigError(
            f"{component}.arguments must preserve the exact original CLI argument array"
        )
    equals_flags = [item for item in raw_args if item.startswith("--") and "=" in item]
    if equals_flags:
        raise AnalysisConfigError(
            f"{component}.arguments must use separate flag/value items; --flag=value is not allowed: "
            f"{equals_flags}"
        )
    args = [_decode_repo_reference(repo_root, item, f"{component} argument") for item in raw_args]
    _validate_original_flags(runner,args)
    path_flags = {
        "--input", "--input-root", "--source-root", "--feature", "--features",
        "--feature-file", "--target", "--target-file", "--manifest", "--provenance",
        "--input-manifest", "--feature-manifest", "--registry", "--workspace",
    }
    output_flags = _RAW_OUTPUT_FLAGS[component]
    present_output_flags = [flag for flag in output_flags if flag in args]
    other_output_flags = [
        flag for flag in ("--out", "--out-root", "--output-root")
        if flag in args and flag not in output_flags
    ]
    if other_output_flags:
        raise AnalysisConfigError(
            f"{component} original CLI does not support output flag(s): {other_output_flags}"
        )
    if len(present_output_flags) > 1:
        raise AnalysisConfigError(f"{component}.arguments may contain only one output flag")
    for required_flag in _RAW_REQUIRED_INPUT_FLAGS[component]:
        if args.count(required_flag) != 1:
            raise AnalysisConfigError(
                f"{component}.arguments must provide {required_flag} exactly once to avoid runner defaults"
            )
        required_index = args.index(required_flag)
        if required_index + 1 >= len(args) or args[required_index + 1].startswith("--"):
            raise AnalysisConfigError(f"{component}.arguments requires a value after {required_flag}")
    output_flag = present_output_flags[0] if present_output_flags else _RAW_DEFAULT_OUTPUT_FLAGS[component]
    path_suffixes = {
        ".csv", ".parquet", ".json", ".jsonl", ".npz", ".xlsx", ".shp", ".gpkg",
        ".tif", ".tiff", ".feather", ".pkl", ".joblib", ".txt", ".geojson", ".zip",
    }
    has_input = False
    for index, value in enumerate(args):
        if value in output_flags or (index and args[index - 1] in output_flags):
            continue
        if value.startswith("--"):
            continue
        previous = args[index - 1] if index else ""
        is_path_value = (
            previous in path_flags
            or any(part in previous.lower() for part in (
                "input", "feature", "target", "manifest", "provenance", "source",
                "data", "file", "workspace", "registry",
            ))
            or any(value.lower().endswith(suffix) for suffix in path_suffixes)
        )
        if not is_path_value:
            continue
        assignment = ""
        raw_path = value
        if "=" in value and not value.startswith("--"):
            assignment, raw_path = value.split("=", 1)
            assignment += "="
        if not raw_path:
            raise AnalysisConfigError(f"{component} has an empty path argument")
        resolved = _repo_path(repo_root, raw_path, f"{component} input")
        args[index] = assignment + str(resolved)
        has_input = True
    if not has_input:
        raise AnalysisConfigError(
            f"{component} requires at least one explicit existing input path in its original CLI arguments"
        )
    if output_flag in args:
        if args.count(output_flag) != 1:
            raise AnalysisConfigError(f"{component}.arguments may contain {output_flag} only once")
        out_index = args.index(output_flag)
        if out_index + 1 >= len(args) or args[out_index + 1].startswith("--"):
            raise AnalysisConfigError(f"{component}.arguments requires a value after {output_flag}")
        args[out_index + 1] = str(output)
    else:
        args.extend((output_flag, str(output)))
    if component == "thinning":
        manifest_path = Path(args[args.index("--manifest") + 1])
        _validate_manifest_path_tree(repo_root, manifest_path, "thinning input manifest")
    elif component == "multiplicity":
        workspace = Path(args[args.index("--workspace") + 1])
        registry = Path(args[args.index("--registry") + 1])
        if not workspace.is_dir():
            raise AnalysisConfigError(f"multiplicity --workspace is not a directory: {workspace}")
        _validate_manifest_path_tree(
            repo_root, registry, "multiplicity registry", reference_base=workspace,
        )
    return runner, runner.parent, args


def build_analysis_command(
    repo_root: str | Path,
    config_path: str | Path,
    component: str,
) -> dict[str, Any]:
    """Validate config and return a portable, shell-free command description."""
    root = Path(repo_root).resolve()
    if component not in _COMPONENTS:
        raise AnalysisConfigError(f"Unknown component {component!r}; choose from {', '.join(_COMPONENTS)}")
    config, _ = _load_config(root, config_path)
    components = config.get("components")
    if not isinstance(components, dict) or not isinstance(components.get(component), dict):
        raise AnalysisConfigError(f"Config must provide a components.{component} object")
    settings = components[component]
    output = _output_path(root, config, component)
    if component in {"historical", "portugal"}:
        runner, cwd, runner_args = _build_historical(root, component, settings, output)
    elif component == "future":
        runner, cwd, runner_args = _build_future(root, settings, output)
    elif component == "loco":
        runner, cwd, runner_args = _build_loco(root, settings, output)
    else:
        runner, cwd, runner_args = _build_raw(root, component, settings, output)
    command = [sys.executable, str(runner), *runner_args]
    plan = {
        "component": component,
        "runner": runner.relative_to(root).as_posix(),
        "cwd": str(cwd),
        "output_root": str(_repo_path(root, str(config["output_root"]), "output_root", must_exist=False)),
        "output_dir": str(output),
        "command": command,
    }
    if component == "thinning":
        raw_validation_dir = settings.get("validation_dir")
        validation_dir = _repo_path(
            root, str(raw_validation_dir or ""), "thinning validation_dir"
        )
        if not validation_dir.is_dir() or not all(
            (validation_dir / name).is_file() for name in ("run_validation.py", "validation_core.py")
        ):
            raise AnalysisConfigError(
                "thinning validation_dir must contain the run_validation.py and validation_core.py "
                "snapshot bound by its input manifest"
            )
        # The archived script exposes this path override specifically for its
        # imported core snapshot. Keep the CLI unchanged and surface the exact
        # environment in the dry-run plan.
        plan["environment"] = {"CAND1_FINAL_VALIDATION_DIR": str(validation_dir)}
    return plan


def run_analysis(
    repo_root: str | Path,
    config_path: str | Path,
    component: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Print-plan by default; optionally launch the exact original runner."""
    plan = build_analysis_command(repo_root, config_path, component)
    plan["executed"] = bool(execute)
    if not execute:
        return plan
    Path(plan["output_dir"]).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(plan.get("environment", {}))
    completed = subprocess.run(
        plan["command"], cwd=plan["cwd"], env=environment,
        capture_output=True, text=True, check=False,
    )
    plan.update({
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    })
    return plan


def self_check(repo_root: str | Path) -> dict[str, Any]:
    """Verify published source hashes, then run historical runner self-checks.

    The published hashes in SOURCE_SNAPSHOTS.json are authoritative for this
    distribution. ``original_sha256`` records source history and is not an
    assertion that every packaged file is byte-identical to its upstream copy.
    """
    root = Path(repo_root).resolve()
    manifest_path = root / "analysis" / "original" / "SOURCE_SNAPSHOTS.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, list) or not files:
            raise AnalysisConfigError("SOURCE_SNAPSHOTS.json must contain a nonempty files list")
        seen: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise AnalysisConfigError("SOURCE_SNAPSHOTS.json entries must be objects")
            relative = entry.get("file")
            expected = entry.get("sha256")
            if not isinstance(relative, str) or not relative:
                raise AnalysisConfigError("SOURCE_SNAPSHOTS.json entry is missing file")
            if relative in seen:
                raise AnalysisConfigError(f"Duplicate SOURCE_SNAPSHOTS.json file entry: {relative}")
            seen.add(relative)
            if not isinstance(expected, str) or len(expected) != 64 or any(
                ch not in "0123456789abcdefABCDEF" for ch in expected
            ):
                raise AnalysisConfigError(f"Invalid published SHA-256 for {relative}")
            path = _repo_path(root, relative, "published source file")
            observed = _sha256(path)
            if observed.lower() != expected.lower():
                raise AnalysisConfigError(
                    f"Published source SHA-256 mismatch for {relative}: "
                    f"manifest {expected.lower()}, observed {observed}"
                )
        source_manifest = {
            "ok": True,
            "manifest": "analysis/original/SOURCE_SNAPSHOTS.json",
            "files_checked": len(files),
            "hash_basis": "manifest sha256 (published distribution)",
            "original_sha256_compared": False,
        }
    except (OSError, json.JSONDecodeError, AnalysisConfigError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "returncode": 2,
            "source_manifest": {"ok": False, "error": error},
            "checks": {
                component: {"ok": False, "error": "Skipped because source manifest verification failed"}
                for component in ("historical", "portugal")
            },
        }

    outcomes: dict[str, Any] = {}
    for component in ("historical", "portugal"):
        raw = _DEFAULT_RUNNERS[component]
        try:
            runner = _repo_path(root, raw, f"{component} self-check runner")
        except AnalysisConfigError as exc:
            outcomes[component] = {"ok": False, "error": str(exc)}
            continue
        completed = subprocess.run(
            [sys.executable, str(runner), "--self-check-only"],
            cwd=runner.parent, capture_output=True, text=True, check=False,
        )
        outcomes[component] = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    ok = all(result.get("ok", False) for result in outcomes.values())
    return {
        "ok": ok,
        "returncode": 0 if ok else 2,
        "source_manifest": source_manifest,
        "checks": outcomes,
    }


def analytical_cli(repo_root: str | Path, args: Sequence[str] | argparse.Namespace) -> int:
    """CLI entry point for `analysis <component> --config ... [--execute]`."""
    parser = argparse.ArgumentParser(prog="pm25-repro analysis")
    parser.add_argument("component", choices=_COMPONENTS)
    parser.add_argument("--config", required=True, help="repo-relative JSON config path")
    parser.add_argument("--execute", action="store_true", help="launch the configured original runner")
    parsed = args if isinstance(args, argparse.Namespace) else parser.parse_args(list(args))
    try:
        result = run_analysis(repo_root, parsed.config, parsed.component, parsed.execute)
    except (AnalysisConfigError, OSError) as exc:
        parser.error(str(exc))
        return 2  # argparse.error exits; retained for type checkers.
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("returncode", 0))


__all__ = [
    "AnalysisConfigError", "analytical_cli", "build_analysis_command", "run_analysis", "self_check",
]

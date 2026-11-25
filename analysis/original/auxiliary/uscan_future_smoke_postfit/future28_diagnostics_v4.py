"""No-fit, source-derived future28 support and conformal diagnostics.

The two public checks here deliberately consume only the output directory,
source-derived frames, and the frozen external runner. They never fit a model.
All saved tables are checked against recomputation with the pinned helper code.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROFILES = ("with_met", "no_met")
ARMS = ("B", "B+AOD", "B+fire", "full")
OOF_STRATEGIES = {
    ("log_smear", "crossfit_station", "smearing"),
    ("log_smear", "crossfit_station", "original"),
    ("original_scale", "not_applicable", "model_native"),
}
SUPPORT_GROUPS = {
    "geography", "meteorology", "aod", "fire", "non_geography",
    "all_model_features_including_geography",
}
TOL = 1e-10


class DiagnosticError(ValueError):
    """A persisted diagnostic differs from the frozen producer contract."""


def _require(ok: bool, message: str) -> None:
    if not bool(ok):
        raise DiagnosticError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _static_module(name: str, runner: Any):
    """Return a helper imported from the runtime's immutable source snapshot."""
    runner_file = getattr(runner, "__file__", None)
    expected = None
    if runner_file:
        expected = (Path(runner_file).resolve().parent.parent /
                    "source_snapshot" / "validation" / f"{name}.py").resolve()
    module = sys.modules.get(name)
    if module is not None:
        origin = getattr(module, "__file__", None)
        if expected is not None and origin is not None:
            _require(Path(origin).resolve() == expected,
                     f"{name} import origin is not the frozen source snapshot: {origin}")
        return module
    _require(expected is not None and expected.is_file(),
             f"cannot locate frozen {name}.py from runner path")
    parent = str(expected.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(name)
    _require(Path(module.__file__).resolve() == expected,
             f"{name} import did not resolve to frozen snapshot")
    return module


def _string_key(series: pd.Series) -> pd.Series:
    _require(not series.isna().any(), f"null key in {series.name}")
    return series.astype(str)


def _normalised_keys(frame: pd.DataFrame, keys: Sequence[str], numeric: set[str]) -> pd.DataFrame:
    missing = sorted(set(keys) - set(frame.columns))
    _require(not missing, f"missing key columns: {missing}")
    out = pd.DataFrame(index=frame.index)
    for name in keys:
        if name in numeric:
            values = pd.to_numeric(frame[name], errors="coerce")
            _require(values.notna().all() and np.isfinite(values).all() and
                     np.equal(values, np.floor(values)).all() and
                     values.ge(0).all() and values.le(np.iinfo(np.int64).max).all(), f"invalid numeric key {name}")
            out[name] = values.astype("int64")
        else:
            out[name] = _string_key(frame[name])
    return out


def _typed_table_equal(expected: pd.DataFrame, observed: pd.DataFrame,
                       keys: Sequence[str], *, numeric_keys: Iterable[str] = (),
                       label: str) -> None:
    """Fail-closed exact-key comparison followed by typed value comparison."""
    _require(set(expected.columns) == set(observed.columns),
             f"{label}: columns differ expected={sorted(expected.columns)} observed={sorted(observed.columns)}")
    numeric_keys = set(numeric_keys)
    akey = _normalised_keys(expected, keys, numeric_keys)
    bkey = _normalised_keys(observed, keys, numeric_keys)
    _require(not akey.duplicated().any(), f"{label}: duplicate expected keys")
    _require(not bkey.duplicated().any(), f"{label}: duplicate saved keys")
    order = list(keys)
    ai = akey.sort_values(order, kind="mergesort").reset_index(drop=True)
    bi = bkey.sort_values(order, kind="mergesort").reset_index(drop=True)
    _require(len(ai) == len(bi) and ai.equals(bi),
             f"{label}: exact typed key roster differs ({len(ai)} != {len(bi)})")
    # Reorder full frames by their normalized keys so key types don't get
    # weakened merely because a CSV parser inferred an integer as a string.
    a = expected.copy(); b = observed.copy()
    a["__audit_order"] = list(map(tuple, akey.to_numpy().tolist()))
    b["__audit_order"] = list(map(tuple, bkey.to_numpy().tolist()))
    a = a.sort_values("__audit_order", kind="mergesort").reset_index(drop=True)
    b = b.sort_values("__audit_order", kind="mergesort").reset_index(drop=True)
    for col in sorted(expected.columns):
        x, y = a[col], b[col]
        if pd.api.types.is_bool_dtype(x) or pd.api.types.is_bool_dtype(y):
            _require(pd.api.types.is_bool_dtype(x) and pd.api.types.is_bool_dtype(y),
                     f"{label}: non-boolean representation in {col}")
            _require(x.fillna(False).astype(bool).equals(y.fillna(False).astype(bool)) and
                     x.isna().equals(y.isna()), f"{label}: boolean value mismatch in {col}")
        elif (pd.api.types.is_numeric_dtype(x) or pd.api.types.is_numeric_dtype(y)) and col not in keys:
            try:
                xn = pd.to_numeric(x, errors="raise").to_numpy(dtype="float64")
                yn = pd.to_numeric(y, errors="raise").to_numpy(dtype="float64")
            except (ValueError, TypeError) as exc:
                raise DiagnosticError(f"{label}: invalid numeric representation in {col}") from exc
            _require(np.allclose(xn, yn, rtol=TOL, atol=TOL, equal_nan=True),
                     f"{label}: numeric value mismatch in {col}")
        else:
            xs = x.fillna("<NA>").astype(str).reset_index(drop=True)
            ys = y.fillna("<NA>").astype(str).reset_index(drop=True)
            _require(xs.equals(ys), f"{label}: string value mismatch in {col}")


def _same_float(a: Any, b: Any, *, label: str, atol: float = TOL) -> None:
    x, y = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    _require(x.shape == y.shape and np.allclose(x, y, rtol=1e-12, atol=atol,
                                                equal_nan=True), label)


def _expected_oof(oof: pd.DataFrame, test: pd.DataFrame,
                  country: str) -> tuple[dict[str, str], dict[str, str]]:
    required = {
        "country", "cohort_label", "profile", "task", "fold_id", "arm",
        "target_strategy", "smear_strategy", "backtransform", "row_id",
        "site_id", "year", "month", "obs", "pred",
    }
    _require(required <= set(oof.columns),
             f"OOF columns missing for support recomputation: {sorted(required-set(oof.columns))}")
    _require(len(test) > 0 and "_row_id" in test.columns,
             "test frame must be nonempty and retain _row_id")
    ids = pd.to_numeric(test._row_id, errors="coerce").to_numpy(dtype="float64")
    _require(np.array_equal(ids, np.arange(len(test), dtype="float64")),
             "test _row_id is not the zero-based source-order census")
    years = pd.to_numeric(test.year, errors="coerce")
    _require(years.notna().all() and years.nunique() == 1,
             "external support test must contain one exact year")
    year = int(years.iloc[0]); fold = str(year); n = len(test)
    _require(oof.country.astype(str).eq(str(country)).all(), "OOF country mismatch")
    tasks: dict[str, str] = {}; cohorts: dict[str, str] = {}
    for profile in PROFILES:
        task = f"T4_external_{year}_{profile}"
        part = oof.loc[oof.profile.astype(str).eq(profile)].copy()
        _require(len(part) == n * len(ARMS) * len(OOF_STRATEGIES),
                 f"{profile}: OOF row count mismatch")
        _require(part.task.astype(str).eq(task).all() and part.fold_id.astype(str).eq(fold).all(),
                 f"{profile}: OOF task/fold mismatch")
        _require(part.cohort_label.astype(str).nunique() == 1,
                 f"{profile}: OOF cohort label is not unique")
        cohorts[profile] = str(part.cohort_label.iloc[0]); tasks[profile] = task
        key = ["country", "cohort_label", "profile", "task", "fold_id", "target_strategy",
               "smear_strategy", "backtransform", "arm", "row_id"]
        _require(not part.duplicated(key).any(), f"{profile}: duplicate OOF diagnostic keys")
        observed_strategies = set(zip(part.target_strategy.astype(str), part.smear_strategy.astype(str),
                                      part.backtransform.astype(str)))
        _require(observed_strategies == OOF_STRATEGIES, f"{profile}: OOF strategy roster differs")
        _require(set(part.arm.astype(str)) == set(ARMS), f"{profile}: OOF arm roster differs")
        _require(set(pd.to_numeric(part.row_id, errors="coerce").astype(int)) == set(range(n)),
                 f"{profile}: OOF row ID census differs")
        by_id = test.copy().set_index(test._row_id.astype(int))
        for _, rows in part.groupby(["target_strategy", "smear_strategy", "backtransform", "arm"],
                                    sort=False, dropna=False):
            _require(len(rows) == n and set(rows.row_id.astype(int)) == set(range(n)),
                     f"{profile}: OOF arm/strategy row roster differs")
            rows = rows.sort_values("row_id"); target = by_id.loc[rows.row_id.astype(int).to_numpy()]
            _require(rows.site_id.astype(str).reset_index(drop=True).equals(
                         target.site_id.astype(str).reset_index(drop=True)) and
                     np.array_equal(rows.year.astype(int).to_numpy(), target.year.astype(int).to_numpy()) and
                     np.array_equal(rows.month.astype(int).to_numpy(), target.month.astype(int).to_numpy()),
                     f"{profile}: OOF test site/year/month binding differs")
            _same_float(rows.obs.to_numpy(float), target.pm25.to_numpy(float),
                        label=f"{profile}: OOF targets differ from assigned external test targets")
            _require(np.isfinite(rows[["obs", "pred"]].to_numpy(dtype=float)).all(),
                     f"{profile}: nonfinite OOF observation or prediction")
    _require(set(oof.profile.astype(str)) == set(PROFILES), "OOF contains extra or missing profiles")
    return tasks, cohorts


def check_support(out, runner, oof, train, test, country, scope) -> dict[str, Any]:
    """Recompute 48*N support flags and all coverage/trust-risk summaries."""
    out = Path(out); country = str(country).upper()
    tasks, cohorts = _expected_oof(oof, test, country)
    support = _static_module("support_diagnostics", runner)
    core_path = Path(runner.__file__).resolve().parent.parent / "source_snapshot" / "validation" / "validation_core.py"
    flags_path = out / "support_row_flags.parquet"
    ranges_path = out / "training_feature_ranges.csv"
    coverage_path = out / "support_coverage.csv"
    risk_path = out / "trust_risk_metrics.csv"
    for p in (flags_path, ranges_path, coverage_path, risk_path):
        _require(p.is_file(), f"{scope}: missing required support output {p.name}")
    saved_flags = pd.read_parquet(flags_path)
    saved_ranges = pd.read_csv(ranges_path, float_precision="round_trip")
    saved_coverage = pd.read_csv(coverage_path, float_precision="round_trip")
    saved_risk = pd.read_csv(risk_path, float_precision="round_trip")
    fold = str(int(pd.to_numeric(test.year, errors="raise").iloc[0]))
    nearest = np.asarray(runner._nearest_distances(test, train), dtype="float64")
    _require(len(nearest) == len(test), "nearest-train distance vector length mismatch")
    range_rows: list[pd.DataFrame] = []; flag_rows: list[pd.DataFrame] = []
    for profile in PROFILES:
        base = list(runner._profile_features(profile))
        met = list(runner.MET) if profile == "with_met" else []
        arms = {"B": base, "B+AOD": base + list(runner.AOD),
                "B+fire": base + list(runner.FIRE),
                "full": base + list(runner.AOD) + list(runner.FIRE)}
        task = tasks[profile]; cohort = cohorts[profile]
        for arm in ARMS:
            features = arms[arm]
            groups = support.feature_groups_for_arm(
                features, geography=runner.GEO, meteorology=met,
                aod=runner.AOD, fire=runner.FIRE)
            _require(set(groups) == SUPPORT_GROUPS,
                     f"{scope}/{profile}/{arm}: frozen support helper group roster changed")
            ranges, flags = support.training_feature_support(train, test, features, groups)
            meta = {"country": country, "cohort_label": cohort, "profile": profile,
                    "task": task, "fold_id": fold, "arm": arm}
            for frame in (ranges, flags):
                for name, value in meta.items():
                    frame[name] = value
            flags["site_id"] = flags.row_id.map(dict(zip(test._row_id.astype(int), test.site_id.astype(str))))
            flags = support.add_distance_support(flags, nearest, test._row_id.to_numpy(dtype="int64"))
            range_rows.append(ranges); flag_rows.append(flags)
    expected_ranges = pd.concat(range_rows, ignore_index=True)
    expected_flags = pd.concat(flag_rows, ignore_index=True)
    expected_flag_n = 48 * len(test)
    _require(len(expected_flags) == expected_flag_n and
             set(expected_flags.feature_group.astype(str)) == SUPPORT_GROUPS,
             f"{scope}: frozen helper did not produce exactly six groups x eight arm/profile cells x N")
    flag_keys = ["country", "cohort_label", "profile", "task", "fold_id", "arm", "feature_group", "row_id"]
    string_keys = set(flag_keys) - {"row_id"}
    _typed_table_equal(expected_flags, saved_flags, flag_keys,
                       numeric_keys={"row_id"}, label=f"{scope}/support_row_flags")
    range_keys = ["country", "cohort_label", "profile", "task", "fold_id", "arm", "feature"]
    _typed_table_equal(expected_ranges, saved_ranges, range_keys,
                       label=f"{scope}/training_feature_ranges")
    expected_coverage = support.support_coverage_summary(expected_flags)
    expected_risk = support.trust_risk_metrics(oof, expected_flags)
    # Distance strata intentionally have no feature_group; CSV encodes this as NA.
    # Require that exact applicability mask before using a non-null comparison key.
    for risk_table in (expected_risk, saved_risk):
        distance_rows = risk_table.support_dimension.astype(str).eq('nearest_training_station_50km')
        _require(np.array_equal(risk_table.feature_group.isna().to_numpy(), distance_rows.to_numpy()),
                 f'{scope}/trust_risk_metrics: feature_group NA outside distance-only strata')
        risk_table['feature_group'] = risk_table.feature_group.fillna('<not_applicable_station_distance>')
    cov_keys = ["country", "cohort_label", "profile", "task", "fold_id", "arm", "feature_group"]
    risk_keys = ["country", "cohort_label", "profile", "task", "fold_id", "target_strategy",
                 "smear_strategy", "backtransform", "arm", "feature_group",
                 "support_dimension", "support_group", "support_stratum"]
    _typed_table_equal(expected_coverage, saved_coverage, cov_keys,
                       label=f"{scope}/support_coverage")
    _typed_table_equal(expected_risk, saved_risk, risk_keys,
                       label=f"{scope}/trust_risk_metrics")
    return {"status": "checked", "range_rows": int(len(expected_ranges)),
            "row_flag_rows": int(len(expected_flags)), "coverage_rows": int(len(expected_coverage)),
            "trust_risk_rows": int(len(expected_risk)),
            "method": "recomputed train-only ranges, six-group row flags, 50 km distances, coverage, and stratified risk using frozen source helpers"}


def validate_bundle(bundle, core, config, features, train, target_strategy='log_smear'):
    """Check the actual estimator and calibration contract without fitting."""
    _require(bundle.get('config') == config and bundle.get('schema') == core.schema_for(features), 'Bundle config/schema mismatch')
    _require(bundle.get('fit_n') == len(train) and bundle.get('fit_stations') == train.site_id.astype(str).nunique(),
             'Bundle fit denominators differ from bound training rows')
    kind = config['estimator_kind']
    _require(kind in ('hgb', 'ridge') and bundle.get('estimator_kind') == kind and
             bundle.get('estimator_class') == core.ESTIMATOR_CLASS_NAMES[kind], 'Bundle estimator declaration mismatch')
    actual, expected = bundle.get('estimator'), core.estimator(config)
    _require(type(actual) is type(expected), 'Actual estimator class mismatch')

    def parameters_equal(a, b):
        _require(set(a) == set(b), 'Estimator parameter roster mismatch')
        for key in a:
            av, bv = a[key], b[key]
            both_nan = isinstance(av, (float, np.floating)) and isinstance(bv, (float, np.floating)) and np.isnan(av) and np.isnan(bv)
            _require(both_nan or av == bv, 'Actual estimator parameter differs: '+key)

    if kind == 'ridge':
        _require([n for n, _ in actual.steps] == [n for n, _ in expected.steps], 'Ridge preprocessing step roster differs')
        parameters_equal({k: v for k, v in actual.get_params(deep=False).items() if k != 'steps'},
                         {k: v for k, v in expected.get_params(deep=False).items() if k != 'steps'})
        for (_, a), (_, b) in zip(actual.steps, expected.steps):
            _require(type(a) is type(b), 'Ridge preprocessing estimator class differs')
            parameters_equal(a.get_params(deep=False), b.get_params(deep=False))
    else:
        parameters_equal(actual.get_params(deep=False), expected.get_params(deep=False))
    _require(bundle.get('target_strategy') == target_strategy and bundle.get('target_col') == 'pm25' and
             bundle.get('group_col') == 'site_id', 'Bundle target/group contract mismatch')
    info = bundle.get('smear_info', {})
    method = 'crossfit_station' if target_strategy == 'log_smear' else 'not_applicable'
    splits = min(3, int(train.site_id.astype(str).nunique())) if target_strategy == 'log_smear' else 0
    _require(info.get('method') == method and info.get('n_splits') == splits and
             info.get('n_rows') == len(train) and info.get('n_stations') == train.site_id.astype(str).nunique(),
             'Bundle smearing calibration contract mismatch')
    _require(np.isfinite(bundle.get('smear', np.nan)) and float(bundle['smear']) > 0, 'Invalid smearing factor')
    if target_strategy == 'original_scale':
        _require(float(bundle['smear']) == 1, 'Original-scale bundle must have unit smearing factor')


def check_conformal(out, runner, train, test, country, scope):
    """Replay the frozen producer using saved proper-training bundles, never fit."""
    out = Path(out)
    core = _static_module('validation_core', runner)
    helper = _static_module('conformal_sensitivity', runner)
    manifest = _load_json(out/'external_run_manifest.json')
    config = manifest['model_config']
    cohort = manifest['external_target_source']['status']
    year = int(test.year.iloc[0]); fold = str(year)
    _require(test.year.eq(year).all() and manifest['external_year'] == year and manifest['country'] == country,
             'Conformal external identity mismatch')
    records = []
    for prefix in ('conformal', 'conformal_feasibility'):
        cm = _load_json(out/f'{prefix}_manifest.json')
        _require(cm.get('nominal_coverage') == .9 and set(cm.get('models', {})) == set(PROFILES),
                 f'{scope}/{prefix}: exact 90-percent profile manifest required')
        if prefix == 'conformal':
            _require(cm.get('proper_train_fraction') == .8, 'Primary conformal split fraction changed')
        else:
            _require(cm.get('minimum_calibration_stations') == 9 and cm.get('minimum_proper_training_stations') == 2,
                     'Feasibility allocation limits changed')
        expected = {name: [] for name in ('intervals', 'station_assignments', 'calibration_scores')}
        summaries = []
        for profile in PROFILES:
            task = f'T4_external_{year}_{profile}'
            context = dict(country=country, cohort_label=cohort, profile=profile, task=task, fold_id=fold, arm='full')
            rec = cm['models'][profile]
            _require(all(rec.get(k) == v for k, v in context.items()), 'Conformal model context mismatch')
            ntrain = int(train.site_id.astype(str).nunique())
            ncal = helper.feasibility_calibration_station_n(ntrain) if prefix == 'conformal_feasibility' else None
            label = 'finite_q90_feasibility_min9_calibration' if prefix == 'conformal_feasibility' else 'primary_80_20'
            if prefix == 'conformal_feasibility' and ncal is None:
                _require(ntrain < 11 and rec.get('feasibility_status') == 'ineligible' and not rec.get('path') and
                         rec.get('outer_train_station_n') == ntrain and bool(rec.get('eligibility_reason')), 'Invalid feasibility ineligibility')
                summaries.append({**context, 'split_label': label, 'feasibility_status': 'ineligible',
                    'eligibility_reason': 'requires at least 11 outer-training stations for 9 calibration and 2 proper-training stations',
                    'outer_train_station_n': ntrain, 'minimum_calibration_station_n': 9, 'minimum_proper_train_station_n': 2,
                    'requested_calibration_station_n': np.nan, 'nominal_coverage': .9, 'q90': np.nan, 'q90_is_infinite': np.nan})
                records.append({'job_id': scope, 'prefix': prefix, 'profile': profile, 'status': 'ineligible', 'q90': None})
                continue
            proper, calibration, membership, seed = helper.deterministic_station_split(
                train, seed_parts=(country, 'T4_external', fold), proper_fraction=.8, calibration_station_n=ncal)
            features = runner._profile_features(profile) + runner.AOD + runner.FIRE
            model_path = Path(rec['path'])
            suffix = 'full_split_conformal' if prefix == 'conformal' else 'full_finite_q90_feasibility'
            _require(model_path.resolve() == (out/'models'/prefix/f'{country}_{task}_{suffix}.joblib').resolve() and
                     _sha(model_path) == rec['sha256'], 'Conformal model path/hash mismatch')
            bundle = joblib.load(model_path)
            validate_bundle(bundle, core, config, features, proper)
            _require(rec.get('fit_n') == len(proper) and rec.get('proper_train_station_n') == proper.site_id.nunique() and
                     rec.get('calibration_station_n') == calibration.site_id.nunique(), 'Conformal manifest fit denominators mismatch')
            if prefix == 'conformal_feasibility':
                _require(rec.get('feasibility_status') == 'eligible' and rec.get('training_support_fingerprint') ==
                         runner.training_support_fingerprint(proper, features, target_col='pm25'), 'Feasibility proper-training fingerprint mismatch')
            original_fit = helper.fit_model
            calls = []

            def saved_bundle_only(frame, selected_features, **kwargs):
                _require(frame.equals(proper) and list(selected_features) == list(features), 'Replay fit request used different proper-training inputs')
                _require(kwargs.get('target_col') == 'pm25' and kwargs.get('group_col') == 'site_id' and
                         kwargs.get('smear_splits') == 3 and kwargs.get('smear_strategy') == 'crossfit_station' and
                         kwargs.get('target_strategy') == 'log_smear', 'Replay fit request changed calibration configuration')
                _require(kwargs['config'].__dict__ == config, 'Replay requested config differs')
                calls.append(1)
                return bundle

            helper.fit_model = saved_bundle_only
            try:
                replay = helper.station_split_conformal(train, test, features, config=core.ModelConfig(**config),
                    seed_parts=(country, 'T4_external', fold), target_col='pm25', group_col='site_id', smear_splits=3,
                    coverage=.9, proper_fraction=.8, calibration_station_n=ncal, split_label=label)
            finally:
                helper.fit_model = original_fit
            _require(len(calls) == 1 and replay['seed'] == seed, 'Replay did not use exactly one saved proper-training bundle')
            _same_float([rec['q90']], [replay['summary']['q90']], label='Conformal manifest quantile differs from replay')
            for name in expected:
                frame = replay[name].copy()
                for key, value in context.items():
                    frame[key] = value
                if prefix == 'conformal_feasibility':
                    frame['split_label'] = label
                expected[name].append(frame)
            summary = {**replay['summary'], **context, 'split_seed': seed}
            if prefix == 'conformal_feasibility':
                summary.update(split_label=label, feasibility_status='eligible', eligibility_reason='', minimum_calibration_station_n=9,
                               minimum_proper_train_station_n=2, requested_calibration_station_n=int(ncal))
            summaries.append(summary)
            q = replay['summary']['q90']
            records.append({'job_id': scope, 'prefix': prefix, 'profile': profile, 'status': 'replayed_no_fit',
                            'q90': 'infinity' if np.isinf(q) else float(q), 'proper_train_rows': len(proper),
                            'calibration_rows': len(calibration), 'test_rows': len(test)})
        summary_saved = pd.read_csv(out/f'{prefix}_summary.csv', float_precision='round_trip')
        if 'eligibility_reason' in summary_saved:
            # The producer intentionally writes an empty explanation for eligible rows.
            summary_saved['eligibility_reason'] = summary_saved.eligibility_reason.fillna('')
        _typed_table_equal(pd.DataFrame(summaries), summary_saved, ['country', 'cohort_label', 'profile', 'task', 'fold_id', 'arm'],
                           label=f'{scope}/{prefix}/summary')
        for name, suffix in (('intervals', 'intervals.parquet'), ('station_assignments', 'station_splits.csv'),
                             ('calibration_scores', 'calibration_scores.csv')):
            path = out/f'{prefix}_{suffix}'
            if not expected[name]:
                _require(not path.exists(), 'Ineligible feasibility unexpectedly has detail artifacts')
                continue
            saved = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, float_precision='round_trip')
            key = 'row_id' if name == 'intervals' else 'station_id'
            _typed_table_equal(pd.concat(expected[name], ignore_index=True), saved,
                               ['country', 'cohort_label', 'profile', 'task', 'fold_id', 'arm', key] + (['split_role'] if name == 'station_assignments' else []),
                               numeric_keys={'row_id'} if name == 'intervals' else (), label=f'{scope}/{prefix}/{name}')
    return records


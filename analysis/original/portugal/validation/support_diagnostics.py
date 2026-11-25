"""Training-only feature-support and held-out risk diagnostics.

The fixed central-range cutoffs are the 1st and 99th percentiles computed
from each outer-training fold. Observed min/max ranges are retained separately.
These are descriptive coverage diagnostics and never change model selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

SUPPORT_LOWER_QUANTILE = 0.01
SUPPORT_UPPER_QUANTILE = 0.99
SUPPORT_DISTANCE_KM = 50.0


def feature_groups_for_arm(
    features: Sequence[str],
    *,
    geography: Sequence[str],
    meteorology: Sequence[str],
    aod: Sequence[str],
    fire: Sequence[str],
) -> dict[str, list[str]]:
    """Return named, overlapping groups so geographic extrapolation stays distinct."""
    selected = list(features)
    sets = {
        "geography": set(geography),
        "meteorology": set(meteorology),
        "aod": set(aod),
        "fire": set(fire),
    }
    groups = {
        name: [feature for feature in selected if feature in names]
        for name, names in sets.items()
    }
    groups["non_geography"] = [
        feature for feature in selected if feature not in sets["geography"]
    ]
    groups["all_model_features_including_geography"] = selected
    return groups


def training_feature_support(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-feature training ranges and per-row coverage flags.

    Quantiles and min/max are derived only from finite values in ``train``.
    Missing test values are reported separately, not counted as range failures.
    """
    features = list(features)
    required = sorted(set(features))
    missing_train = [name for name in required if name not in train.columns]
    missing_test = [name for name in required if name not in test.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"support audit schema mismatch: train_missing={missing_train}, "
            f"test_missing={missing_test}"
        )
    if "_row_id" not in test.columns:
        raise ValueError("test frame must retain the frozen _row_id column")

    n_test = len(test)
    test_values: dict[str, np.ndarray] = {}
    test_finite: dict[str, np.ndarray] = {}
    train_valid: dict[str, np.ndarray] = {}
    bounds: dict[str, dict[str, float | int]] = {}
    range_rows: list[dict[str, object]] = []

    feature_group = {}
    for group in ("geography", "meteorology", "aod", "fire", "non_geography"):
        for name in groups.get(group, []):
            feature_group.setdefault(name, group)
    for name in features:
        tr_values = pd.to_numeric(train[name], errors="raise").to_numpy(dtype="float64")
        te_values = pd.to_numeric(test[name], errors="raise").to_numpy(dtype="float64")
        tr_valid = tr_values[np.isfinite(tr_values)]
        te_finite = np.isfinite(te_values)
        test_values[name] = te_values
        test_finite[name] = te_finite
        train_valid[name] = tr_valid
        if len(tr_valid):
            q01, q99 = np.quantile(
                tr_valid, [SUPPORT_LOWER_QUANTILE, SUPPORT_UPPER_QUANTILE]
            )
            lo, hi = float(np.min(tr_valid)), float(np.max(tr_valid))
            q01, q99 = float(q01), float(q99)
        else:
            lo = hi = q01 = q99 = np.nan
        bounds[name] = {
            "n_train_finite": int(len(tr_valid)),
            "train_min": lo,
            "train_max": hi,
            "train_p01": q01,
            "train_p99": q99,
        }
        row = {
            "feature": name,
            "feature_group": feature_group.get(name, "other"),
            **bounds[name],
            "n_test_finite": int(te_finite.sum()),
            "n_test_missing": int((~te_finite).sum()),
            "n_test_outside_train_minmax": int(
                np.sum(te_finite & ((te_values < lo) | (te_values > hi)))
            ) if len(tr_valid) else int(te_finite.sum()),
            "n_test_outside_train_p01_p99": int(
                np.sum(te_finite & ((te_values < q01) | (te_values > q99)))
            ) if len(tr_valid) else int(te_finite.sum()),
            "range_quantiles": "training_only_p01_p99",
        }
        row["pct_test_finite_outside_train_minmax"] = (
            row["n_test_outside_train_minmax"] / row["n_test_finite"]
            if row["n_test_finite"] else np.nan
        )
        row["pct_test_finite_outside_train_p01_p99"] = (
            row["n_test_outside_train_p01_p99"] / row["n_test_finite"]
            if row["n_test_finite"] else np.nan
        )
        range_rows.append(row)

    row_frames: list[pd.DataFrame] = []
    for group_name, group_features_raw in groups.items():
        group_features = [name for name in group_features_raw if name in features]
        base = pd.DataFrame({"row_id": test["_row_id"].to_numpy(dtype="int64")})
        base["feature_group"] = group_name
        k = len(group_features)
        base["n_selected_features"] = k
        if not k:
            base["n_finite_features"] = 0
            base["n_missing_features"] = 0
            base["n_no_training_reference_features"] = 0
            base["n_outside_train_minmax"] = 0
            base["n_outside_train_p01_p99"] = 0
            base["pct_finite_within_train_minmax"] = np.nan
            base["pct_finite_within_train_p01_p99"] = np.nan
            base["all_observed_within_train_minmax"] = False
            base["all_observed_within_train_p01_p99"] = False
            base["all_selected_finite_within_train_minmax"] = False
            base["all_selected_finite_within_train_p01_p99"] = False
            base["status_minmax"] = "not_in_arm"
            base["status_p01_p99"] = "not_in_arm"
            row_frames.append(base)
            continue

        finite_matrix = np.column_stack([test_finite[name] for name in group_features])
        n_finite = finite_matrix.sum(axis=1)
        n_missing = k - n_finite
        minmax_out_matrix = []
        central_out_matrix = []
        train_ref_matrix = []
        for name in group_features:
            value = test_values[name]
            finite = test_finite[name]
            b = bounds[name]
            has_reference = b["n_train_finite"] > 0
            train_ref_matrix.append(np.full(n_test, bool(has_reference), dtype=bool))
            if has_reference:
                minmax_out = finite & ((value < b["train_min"]) | (value > b["train_max"]))
                central_out = finite & ((value < b["train_p01"]) | (value > b["train_p99"]))
            else:
                minmax_out = finite.copy()
                central_out = finite.copy()
            minmax_out_matrix.append(minmax_out)
            central_out_matrix.append(central_out)
        minmax_out = np.column_stack(minmax_out_matrix)
        central_out = np.column_stack(central_out_matrix)
        train_ref = np.column_stack(train_ref_matrix)
        n_out_minmax = minmax_out.sum(axis=1)
        n_out_central = central_out.sum(axis=1)
        n_no_ref = ((~train_ref) & finite_matrix).sum(axis=1)
        n_within_minmax = (finite_matrix & ~minmax_out).sum(axis=1)
        n_within_central = (finite_matrix & ~central_out).sum(axis=1)
        pct_minmax = np.divide(
            n_within_minmax, n_finite,
            out=np.full(n_test, np.nan, dtype="float64"), where=n_finite > 0,
        )
        pct_central = np.divide(
            n_within_central, n_finite,
            out=np.full(n_test, np.nan, dtype="float64"), where=n_finite > 0,
        )
        ref_complete = train_ref.all(axis=1)
        all_observed_minmax = (n_finite > 0) & (n_out_minmax == 0)
        all_observed_central = (n_finite > 0) & (n_out_central == 0)
        all_selected_minmax = ref_complete & (n_missing == 0) & (n_out_minmax == 0)
        all_selected_central = ref_complete & (n_missing == 0) & (n_out_central == 0)

        def statuses(out_count: np.ndarray, label: str) -> list[str]:
            output = []
            for i in range(n_test):
                if not ref_complete[i]:
                    output.append("incomplete_training_reference")
                elif out_count[i] > 0:
                    output.append(f"outside_train_{label}")
                elif n_missing[i] > 0:
                    output.append("missing_predictor")
                else:
                    output.append(f"within_train_{label}")
            return output

        base["n_finite_features"] = n_finite
        base["n_missing_features"] = n_missing
        base["n_no_training_reference_features"] = n_no_ref
        base["n_outside_train_minmax"] = n_out_minmax
        base["n_outside_train_p01_p99"] = n_out_central
        base["pct_finite_within_train_minmax"] = pct_minmax
        base["pct_finite_within_train_p01_p99"] = pct_central
        base["all_observed_within_train_minmax"] = all_observed_minmax
        base["all_observed_within_train_p01_p99"] = all_observed_central
        base["all_selected_finite_within_train_minmax"] = all_selected_minmax
        base["all_selected_finite_within_train_p01_p99"] = all_selected_central
        base["status_minmax"] = statuses(n_out_minmax, "observed_minmax")
        base["status_p01_p99"] = statuses(n_out_central, "central_p01_p99")
        row_frames.append(base)

    ranges = pd.DataFrame(range_rows)
    row_flags = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    return ranges, row_flags


def add_distance_support(
    row_flags: pd.DataFrame,
    nearest_train_distance_km: Sequence[float],
    test_row_ids: Sequence[int],
) -> pd.DataFrame:
    """Attach the pre-specified nearest-training-station 50 km coverage flag."""
    out = row_flags.copy()
    distances = np.asarray(nearest_train_distance_km, dtype="float64")
    row_ids = np.asarray(test_row_ids, dtype="int64")
    if len(row_ids) != len(distances):
        raise ValueError("distance vector length does not match held-out row count")
    distance_map = dict(zip(row_ids, distances))
    out["nearest_train_station_km"] = out.row_id.map(distance_map).astype(float)
    out["within_50km_train"] = (
        np.isfinite(out.nearest_train_station_km)
        & (out.nearest_train_station_km < SUPPORT_DISTANCE_KM)
    )
    out["distance_support_status"] = np.where(
        ~np.isfinite(out.nearest_train_station_km), "no_training_distance",
        np.where(out.within_50km_train, "within_50km", "beyond_50km"),
    )
    return out


def support_coverage_summary(row_flags: pd.DataFrame) -> pd.DataFrame:
    """Summarize training-derived range and distance coverage by held-out fold."""
    if row_flags.empty:
        return pd.DataFrame()
    metadata = [
        name for name in ("country", "cohort_label", "profile", "task", "fold_id", "arm", "feature_group")
        if name in row_flags.columns
    ]
    rows = []
    for key, part in row_flags.groupby(metadata, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(metadata, key))
        record.update({
            "test_rows": int(part.row_id.nunique()),
            "test_stations": int(part.site_id.nunique()) if "site_id" in part else np.nan,
            "n_selected_features": int(part.n_selected_features.iloc[0]),
            "pct_rows_with_missing_predictor": float(
                (part.n_missing_features > 0).mean()
            ) if part.n_selected_features.iloc[0] else np.nan,
            "pct_rows_all_selected_within_train_minmax": float(
                part.all_selected_finite_within_train_minmax.mean()
            ) if part.n_selected_features.iloc[0] else np.nan,
            "pct_rows_all_selected_within_train_p01_p99": float(
                part.all_selected_finite_within_train_p01_p99.mean()
            ) if part.n_selected_features.iloc[0] else np.nan,
            "pct_feature_values_within_train_minmax": float(
                part.pct_finite_within_train_minmax.mean()
            ) if part.n_selected_features.iloc[0] else np.nan,
            "pct_feature_values_within_train_p01_p99": float(
                part.pct_finite_within_train_p01_p99.mean()
            ) if part.n_selected_features.iloc[0] else np.nan,
            "pct_rows_within_50km_train_station": float(part.within_50km_train.mean()),
            "median_nearest_train_station_km": float(part.nearest_train_station_km.median()),
            "distance_cutoff_km": SUPPORT_DISTANCE_KM,
            "central_range_cutoffs": "training_only_p01_p99",
            "geography_is_separate_from_environmental_groups": True,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def _metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if not len(y):
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "bias": float(np.mean(pred - y)),
        "r2": float(1 - np.sum((y - pred) ** 2) / denom) if denom > 0 else np.nan,
    }


def trust_risk_metrics(oof: pd.DataFrame, row_flags: pd.DataFrame) -> pd.DataFrame:
    """Score held-out predictions within fixed distance/range coverage strata.

    Subgroup metrics are descriptive and intentionally carry no additional
    confidence interval. The primary paired station bootstrap remains separate.
    """
    if oof.empty or row_flags.empty:
        return pd.DataFrame()
    row_keys = [name for name in (
        "country", "cohort_label", "task", "fold_id", "row_id", "arm", "profile", "site_id"
    )
                if name in oof.columns and name in row_flags.columns]
    joined = oof.merge(row_flags, on=row_keys, how="inner", validate="many_to_many")
    prediction_keys = [
        name for name in (
            "country", "cohort_label", "profile", "task", "fold_id", "arm",
            "target_strategy", "smear_strategy", "backtransform",
        ) if name in joined.columns
    ]
    rows: list[dict[str, object]] = []

    distance_part = joined.drop_duplicates(prediction_keys + ["row_id"])
    for key, part in distance_part.groupby(prediction_keys + ["distance_support_status"], dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        info = dict(zip(prediction_keys + ["distance_support_status"], key))
        metric = _metric_values(part.obs.to_numpy(float), part.pred.to_numpy(float))
        rows.append({
            **info, "support_dimension": "nearest_training_station_50km",
            "support_group": "station_distance", "support_stratum": info.pop("distance_support_status"),
            "n_rows": metric["n"], "n_stations": int(part.site_id.nunique()),
            **{k: v for k, v in metric.items() if k != "n"},
            "uncertainty_scope": "descriptive subgroup metric; no separate interval",
        })

    for column, dimension in (
        ("status_p01_p99", "training_central_percentile_range"),
        ("status_minmax", "observed_training_minmax"),
    ):
        for key, part in joined.groupby(prediction_keys + ["feature_group", column], dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            info = dict(zip(prediction_keys + ["feature_group", column], key))
            status = info.pop(column)
            if status == "not_in_arm":
                continue
            metric = _metric_values(part.obs.to_numpy(float), part.pred.to_numpy(float))
            rows.append({
                **info, "support_dimension": dimension,
                "support_group": info.get("feature_group"), "support_stratum": status,
                "n_rows": metric["n"], "n_stations": int(part.site_id.nunique()),
                **{k: v for k, v in metric.items() if k != "n"},
                "uncertainty_scope": "descriptive subgroup metric; no separate interval",
            })
    return pd.DataFrame(rows)


def write_support_outputs(
    out_dir: Path,
    range_frames: Sequence[pd.DataFrame],
    row_frames: Sequence[pd.DataFrame],
    oof: pd.DataFrame,
) -> None:
    """Write additive support artifacts alongside a completed validation run."""
    out_dir = Path(out_dir)
    ranges = pd.concat(range_frames, ignore_index=True) if range_frames else pd.DataFrame()
    row_flags = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    ranges.to_csv(out_dir / "training_feature_ranges.csv", index=False)
    if row_flags.empty and not len(row_flags.columns):
        row_flags = pd.DataFrame({"row_id": pd.Series(dtype="int64")})
    row_flags.to_parquet(out_dir / "support_row_flags.parquet", index=False)
    support_coverage_summary(row_flags).to_csv(out_dir / "support_coverage.csv", index=False)
    trust_risk_metrics(oof, row_flags).to_csv(out_dir / "trust_risk_metrics.csv", index=False)

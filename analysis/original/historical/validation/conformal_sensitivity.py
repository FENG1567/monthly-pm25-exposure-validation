"""Station-level split-conformal sensitivity for original-scale PM2.5.

Each outer-training set is split by physical station into proper-training and
calibration groups. The interval point model is fitted only on proper-training
stations; calibration quantiles are never attached to a full-data fit.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from validation_core import ModelConfig, fit_model, predict_model

CONFORMAL_COVERAGE = 0.90
PROPER_TRAIN_FRACTION = 0.80
CONFORMAL_SCORE = "station_max_absolute_original_scale_residual"
FEASIBILITY_MIN_CALIBRATION_STATIONS = 9
FEASIBILITY_MIN_PROPER_STATIONS = 2


def feasibility_calibration_station_n(
    outer_train_station_n: int,
    *,
    proper_fraction: float = PROPER_TRAIN_FRACTION,
    minimum_calibration_stations: int = FEASIBILITY_MIN_CALIBRATION_STATIONS,
    minimum_proper_stations: int = FEASIBILITY_MIN_PROPER_STATIONS,
) -> int | None:
    """Choose a larger calibration group that permits a finite 90% rank."""
    if outer_train_station_n < minimum_calibration_stations + minimum_proper_stations:
        return None
    return max(
        int(math.ceil((1 - proper_fraction) * outer_train_station_n)),
        int(minimum_calibration_stations),
    )


def deterministic_station_split(
    train: pd.DataFrame,
    *,
    seed_parts: Sequence[str],
    proper_fraction: float = PROPER_TRAIN_FRACTION,
    base_seed: int = 0,
    group_col: str = "site_id",
    calibration_station_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Split sorted physical stations deterministically, independent of outcomes."""
    if not 0 < proper_fraction < 1:
        raise ValueError("proper_fraction must be between zero and one")
    if group_col not in train:
        raise ValueError(f"training group column {group_col!r} is absent")
    stations = sorted(train[group_col].astype(str).unique())
    if len(stations) < 2:
        raise ValueError("split conformal requires at least two training stations")
    payload = "|".join([str(base_seed), *map(str, seed_parts)]).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(stations, dtype=object)).tolist()
    if calibration_station_n is None:
        n_proper = min(
            max(int(math.floor(proper_fraction * len(stations))), 1),
            len(stations) - 1,
        )
    else:
        calibration_station_n = int(calibration_station_n)
        if not 1 <= calibration_station_n < len(stations):
            raise ValueError("calibration_station_n must leave at least one proper-training station")
        n_proper = len(stations) - calibration_station_n
    proper_ids = set(shuffled[:n_proper])
    calibration_ids = set(shuffled[n_proper:])
    proper_mask = train[group_col].astype(str).isin(proper_ids)
    calibration_mask = train[group_col].astype(str).isin(calibration_ids)
    proper = train.loc[proper_mask].copy()
    calibration = train.loc[calibration_mask].copy()
    counts = train.assign(_station_id=train[group_col].astype(str)).groupby(
        "_station_id", sort=True
    ).size()
    assignments = pd.DataFrame({
        "station_id": sorted(stations),
        "split_role": ["proper_train" if s in proper_ids else "calibration" for s in sorted(stations)],
        "n_training_rows": [int(counts.loc[s]) for s in sorted(stations)],
    })
    return proper, calibration, assignments, seed


def finite_sample_quantile(
    scores: Sequence[float], coverage: float = CONFORMAL_COVERAGE
) -> tuple[float, int, int]:
    """Return the finite-sample order statistic, or infinity when rank exceeds n."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    values = np.asarray(scores, dtype="float64")
    values = values[np.isfinite(values)]
    rank = int(math.ceil((len(values) + 1) * coverage))
    if rank > len(values) or len(values) == 0:
        return float("inf"), rank, int(len(values))
    return float(np.sort(values)[rank - 1]), rank, int(len(values))


def _coverage_note(task: str) -> str:
    if task.startswith("T1_station"):
        return (
            "station-level exchangeability is an approximation; spatial dependence "
            "may violate it; observed held-out coverage is reported"
        )
    if task.startswith("T2"):
        return "empirical coverage under spatial shift; no formal block-level or transport guarantee"
    return "empirical coverage for the observed temporal/external split; no new-year guarantee"


def station_split_conformal(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_features: Sequence[str],
    *,
    config: ModelConfig,
    seed_parts: Sequence[str],
    target_col: str = "pm25",
    group_col: str = "site_id",
    smear_splits: int = 3,
    coverage: float = CONFORMAL_COVERAGE,
    proper_fraction: float = PROPER_TRAIN_FRACTION,
    calibration_station_n: int | None = None,
    split_label: str = "primary_80_20",
) -> dict[str, Any]:
    """Fit a separate proper-training point model and evaluate conformal bands."""
    proper, calibration, assignments, seed = deterministic_station_split(
        train, seed_parts=seed_parts, proper_fraction=proper_fraction,
        group_col=group_col, calibration_station_n=calibration_station_n,
    )
    bundle = fit_model(
        proper, base_features, target_col=target_col, group_col=group_col,
        config=config, smear_strategy="crossfit_station", smear_splits=smear_splits,
        target_strategy="log_smear",
    )
    calibration_prediction = predict_model(bundle, calibration, backtransform="smearing")
    calibration_targets = pd.to_numeric(calibration[target_col], errors="coerce").to_numpy(float)
    calibration_rows = pd.DataFrame({
        "station_id": calibration[group_col].astype(str).to_numpy(),
        "absolute_residual": np.abs(calibration_targets - calibration_prediction),
    })
    calibration_scores = calibration_rows.groupby("station_id", sort=True).absolute_residual.max()
    q, order_rank, calibration_score_station_n = finite_sample_quantile(
        calibration_scores.to_numpy(dtype="float64"), coverage
    )
    calibration_station_n = int(calibration[group_col].astype(str).nunique())

    prediction = predict_model(bundle, test, backtransform="smearing")
    observed = pd.to_numeric(test[target_col], errors="coerce").to_numpy(float)
    lower = np.maximum(0.0, prediction - q)
    upper = prediction + q
    width = upper - lower
    covered = (observed >= lower) & (observed <= upper)
    intervals = pd.DataFrame({
        "row_id": test["_row_id"].to_numpy(dtype="int64") if "_row_id" in test else np.arange(len(test)),
        "site_id": test[group_col].astype(str).to_numpy(),
        "year": pd.to_numeric(test["year"], errors="raise").to_numpy(dtype="int64") if "year" in test else np.nan,
        "month": pd.to_numeric(test["month"], errors="raise").to_numpy(dtype="int64") if "month" in test else np.nan,
        "obs": observed,
        "point_prediction": prediction,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_width": width,
        "covered_pointwise": covered,
        "station_max_abs_residual_q90": q,
    })
    all_month_coverage = intervals.groupby("site_id", sort=True).covered_pointwise.all()
    test_months_per_station = intervals.groupby("site_id", sort=True).size()
    calibration_year_months = sorted({
        f"{int(year):04d}-{int(month):02d}"
        for year, month in zip(calibration["year"], calibration["month"])
    }) if {"year", "month"}.issubset(calibration.columns) else []
    test_year_months = sorted({
        f"{int(year):04d}-{int(month):02d}"
        for year, month in zip(test["year"], test["month"])
    }) if {"year", "month"}.issubset(test.columns) else []
    finite_width = width[np.isfinite(width)]
    summary = {
        "nominal_coverage": float(coverage),
        "split_label": str(split_label),
        "proper_train_fraction": float(proper_fraction),
        "proper_train_actual_fraction": (
            float(proper[group_col].astype(str).nunique())
            / float(train[group_col].astype(str).nunique())
        ),
        "calibration_actual_fraction": (
            float(calibration[group_col].astype(str).nunique())
            / float(train[group_col].astype(str).nunique())
        ),
        "outer_train_station_n": int(train[group_col].astype(str).nunique()),
        "proper_train_station_n": int(proper[group_col].astype(str).nunique()),
        "proper_train_row_n": int(len(proper)),
        "calibration_station_n": int(calibration_station_n),
        "calibration_score_station_n": int(calibration_score_station_n),
        "calibration_row_n": int(len(calibration)),
        "proper_train_years": ",".join(map(str, sorted(pd.to_numeric(proper.year, errors="raise").unique())))
        if "year" in proper else "",
        "calibration_years": ",".join(map(str, sorted(pd.to_numeric(calibration.year, errors="raise").unique())))
        if "year" in calibration else "",
        "calibration_year_month_n": int(len(calibration_year_months)),
        "calibration_year_months": ",".join(calibration_year_months),
        "test_station_n": int(test[group_col].astype(str).nunique()),
        "test_row_n": int(len(test)),
        "test_years": ",".join(map(str, sorted(pd.to_numeric(test.year, errors="raise").unique())))
        if "year" in test else "",
        "test_year_month_n": int(len(test_year_months)),
        "test_year_months": ",".join(test_year_months),
        "calibration_score_order_rank_1based": int(order_rank),
        "q90": float(q),
        "q90_is_infinite": bool(not np.isfinite(q)),
        "pointwise_coverage": float(np.mean(covered)) if len(covered) else np.nan,
        "all_month_per_station_coverage": float(all_month_coverage.mean())
        if len(all_month_coverage) else np.nan,
        "all_month_per_station_covered_n": int(all_month_coverage.sum()),
        "all_month_per_station_group_n": int(len(all_month_coverage)),
        "test_months_per_station_min": int(test_months_per_station.min())
        if len(test_months_per_station) else 0,
        "test_months_per_station_median": float(test_months_per_station.median())
        if len(test_months_per_station) else np.nan,
        "test_months_per_station_max": int(test_months_per_station.max())
        if len(test_months_per_station) else 0,
        "mean_interval_width": float(np.mean(width)) if len(width) else np.nan,
        "median_interval_width": float(np.median(width)) if len(width) else np.nan,
        "finite_interval_width_row_n": int(len(finite_width)),
        "coverage_note": _coverage_note(str(seed_parts[1]) if len(seed_parts) > 1 else ""),
        "point_model": "separate_80pct_proper_train_full_arm_crossfit_log_smear",
        "calibration_score": CONFORMAL_SCORE,
    }
    calibration_scores_frame = calibration_scores.rename("station_max_abs_residual").reset_index()
    calibration_scores_frame["calibration_score_order_rank_1based"] = (
        calibration_scores_frame.station_max_abs_residual.rank(method="first").astype(int)
    )
    assignments["proper_train_station_n"] = int(proper[group_col].astype(str).nunique())
    assignments["calibration_station_n"] = int(calibration_station_n)
    assignments["test_station_n"] = int(test[group_col].astype(str).nunique())
    test_counts = test.groupby(group_col, sort=True).size()
    test_assignments = pd.DataFrame({
        "station_id": test_counts.index.astype(str),
        "split_role": "test",
        "n_test_rows": test_counts.to_numpy(dtype="int64"),
        "proper_train_station_n": int(proper[group_col].astype(str).nunique()),
        "calibration_station_n": int(calibration_station_n),
        "test_station_n": int(test[group_col].astype(str).nunique()),
    })
    assignment_output = pd.concat([assignments, test_assignments], ignore_index=True, sort=False)
    return {
        "bundle": bundle,
        "intervals": intervals,
        "summary": summary,
        "station_assignments": assignment_output,
        "calibration_scores": calibration_scores_frame,
        "seed": int(seed),
    }

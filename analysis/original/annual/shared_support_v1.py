"""Pure monthly pairing and calendar-day annual shared-support helpers.

The helpers deliberately do no file I/O and compute no scores.  Call them once
per benchmark version; V6NA01 and V6GL03 outputs must remain separate.
"""
from __future__ import annotations

import calendar
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


MONTH_KEYS = ("country", "site_id", "year", "month")
ANNUAL_KEYS = ("country", "site_id", "year")
# A model cell is identified by all of these dimensions.  fold_id is retained
# to keep station-held-out folds separate while forming annual aggregates.
MODEL_IDENTITY = (
    "alias", "qa", "profile", "estimator", "task", "fold_id", "arm",
    "target_strategy", "backtransform",
)
DEFAULT_UNAVAILABLE_YEARS = (2025,)
OBS_TOLERANCE = 1e-8


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalize_calendar(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    _require_columns(out, MONTH_KEYS, label)
    if out[list(MONTH_KEYS)].isna().any().any():
        raise ValueError(f"{label} contains null monthly keys")
    for column in ("year", "month"):
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(values).all()) or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label} has noninteger {column} values")
        out[column] = values.astype(np.int64)
    if not out.month.between(1, 12).all() or out.year.lt(1).any():
        raise ValueError(f"{label} has invalid calendar fields")
    if out["country"].astype(str).str.len().eq(0).any() or out["site_id"].astype(str).str.len().eq(0).any():
        raise ValueError(f"{label} contains an empty country or site_id")
    return out


def _numeric(frame: pd.DataFrame, column: str, label: str, *, allow_missing: bool) -> pd.DataFrame:
    out = frame.copy()
    values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
    if not allow_missing and not np.isfinite(values).all():
        raise ValueError(f"{label}.{column} must be finite")
    # Invalid and missing prediction values remain NaN and are later recorded
    # as exclusions; they are never replaced with zero.
    values[~np.isfinite(values)] = np.nan
    out[column] = values
    return out


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(frame[list(MONTH_KEYS)].itertuples(index=False, name=None))


def _obs_equal(left: pd.Series, right: pd.Series, *, label: str) -> None:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = np.isfinite(a) & np.isfinite(b) & np.isclose(a, b, rtol=0.0, atol=OBS_TOLERANCE)
    if not equal.all():
        bad = int(np.flatnonzero(~equal)[0])
        raise ValueError(f"{label} observed targets differ at paired row {bad}")


def _validate_fold_stability(frame: pd.DataFrame, identity_columns: Sequence[str]) -> None:
    """Station-held-out T1/T2 cells must use one fold per station-year."""
    station_held_out = frame.task.astype(str).str.startswith(("T1", "T2"))
    if not station_held_out.any():
        return
    without_fold = [column for column in identity_columns if column != "fold_id"]
    group_columns = without_fold + ["country", "site_id", "year"]
    counts = (frame.loc[station_held_out]
              .groupby(group_columns, dropna=False, sort=False).fold_id.nunique())
    if counts.gt(1).any():
        raise ValueError("T1/T2 station-year monthly rows cross fold_id values")


def join_monthly_model_oof_raw_cams(
    model_oof: pd.DataFrame,
    raw_cams: pd.DataFrame,
    *,
    model_prediction: str = "pred",
    cams_prediction: str = "raw_cams",
    observation: str = "obs",
    identity_columns: Sequence[str] = MODEL_IDENTITY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pair monthly OOF predictions with raw CAMS on exact keys and targets.

    ``raw_cams`` is a monthly source table keyed once per country/site/month;
    model metadata stays on the OOF side and therefore cannot be collapsed
    across aliases, QA, profiles, estimators, tasks, folds, arms, or targets.
    The distinct monthly key sets must match exactly, though OOF may contain
    multiple independent model cells for the same monthly key.
    """
    identities = tuple(identity_columns)
    if len(set(identities)) != len(identities) or set(identities).intersection(MONTH_KEYS):
        raise ValueError("identity_columns must be unique and separate from monthly keys")
    required_model = list(MONTH_KEYS) + [observation, model_prediction, *identities]
    required_cams = list(MONTH_KEYS) + [observation, cams_prediction]
    model = _normalize_calendar(model_oof, "model_oof")
    cams = _normalize_calendar(raw_cams, "raw_cams")
    _require_columns(model, required_model, "model_oof")
    _require_columns(cams, required_cams, "raw_cams")
    if model.empty:
        raise ValueError("model_oof is empty")
    if model[list(identities)].isna().any().any():
        raise ValueError("model_oof has null model-cell identity values")
    model = _numeric(model, observation, "model_oof", allow_missing=False)
    model = _numeric(model, model_prediction, "model_oof", allow_missing=True)
    cams = _numeric(cams, observation, "raw_cams", allow_missing=False)
    cams = _numeric(cams, cams_prediction, "raw_cams", allow_missing=True)

    model_unique_key = list(identities) + list(MONTH_KEYS)
    if model.duplicated(model_unique_key).any():
        raise ValueError("model_oof has duplicate monthly rows within a model cell")
    if cams.duplicated(list(MONTH_KEYS)).any():
        raise ValueError("raw_cams has duplicate country/site/year/month rows")
    _validate_fold_stability(model, identities)

    model_keys, cams_keys = _key_set(model), _key_set(cams)
    if model_keys != cams_keys:
        model_only = sorted(model_keys - cams_keys, key=repr)
        cams_only = sorted(cams_keys - model_keys, key=repr)
        raise ValueError(
            "model_oof/raw_cams monthly key sets differ "
            f"(model_only={len(model_only)}, raw_cams_only={len(cams_only)}; "
            f"examples={model_only[:2]} / {cams_only[:2]})"
        )

    left = model.rename(columns={model_prediction: "model_prediction", observation: "obs"})
    right = cams[list(MONTH_KEYS) + [observation, cams_prediction]].rename(
        columns={observation: "raw_cams_obs", cams_prediction: "raw_cams_prediction"}
    )
    paired = left.merge(right, on=list(MONTH_KEYS), how="left", validate="many_to_one", sort=False)
    _obs_equal(paired.obs, paired.raw_cams_obs, label="model_oof/raw_cams")
    paired = paired.drop(columns=["raw_cams_obs"])
    paired = paired.sort_values(list(identities) + list(MONTH_KEYS), kind="mergesort").reset_index(drop=True)
    audit: dict[str, object] = {
        "monthly_key_set_equal": True,
        "model_month_rows": int(len(model)),
        "raw_cams_month_rows": int(len(cams)),
        "paired_month_rows": int(len(paired)),
        "unique_month_keys": int(len(model_keys)),
        "model_identity_columns": list(identities),
        "observed_targets_equal_within_tolerance": True,
        "observation_tolerance_absolute": OBS_TOLERANCE,
    }
    return paired, audit


def annualize_exact_shared_support(
    model_oof: pd.DataFrame,
    raw_cams: pd.DataFrame,
    satpm_monthly: pd.DataFrame,
    *,
    benchmark_version: str,
    model_prediction: str = "pred",
    cams_prediction: str = "raw_cams",
    observation: str = "obs",
    satpm_prediction: str = "satpm",
    satpm_observation: str = "obs",
    identity_columns: Sequence[str] = MODEL_IDENTITY,
    unavailable_years: Sequence[int] = DEFAULT_UNAVAILABLE_YEARS,
    min_sensitivity_months: int = 10,
    primary_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return monthly pairs and annual means on exact shared support.

    The annual target, model, raw-CAMS, and SatPM values use the same finite
    monthly rows, weighted by calendar days.  A monthly target mismatch is a
    hard error.  Missing comparator values and unavailable years are retained
    as statuses and excluded from annual means without zero filling.
    """
    version = str(benchmark_version).strip()
    if not version:
        raise ValueError("benchmark_version must identify one SatPM product")
    if not (1 <= min_sensitivity_months <= primary_months <= 12):
        raise ValueError("support thresholds must satisfy 1 <= sensitivity <= primary <= 12")
    unavailable = tuple(sorted({int(year) for year in unavailable_years}))
    paired, join_audit = join_monthly_model_oof_raw_cams(
        model_oof, raw_cams, model_prediction=model_prediction,
        cams_prediction=cams_prediction, observation=observation,
        identity_columns=identity_columns,
    )

    benchmark = _normalize_calendar(satpm_monthly, "satpm_monthly")
    _require_columns(benchmark, list(MONTH_KEYS) + [satpm_observation, satpm_prediction], "satpm_monthly")
    for version_column in ("satpm_version", "benchmark_version"):
        if version_column in benchmark:
            source_versions = benchmark[version_column].dropna().astype(str).unique().tolist()
            if source_versions != [version]:
                raise ValueError(
                    f"satpm_monthly {version_column} must contain only the requested "
                    f"benchmark_version {version!r}; found {source_versions}"
                )
    if benchmark.duplicated(list(MONTH_KEYS)).any():
        raise ValueError("satpm_monthly has duplicate country/site/year/month rows")
    benchmark = _numeric(benchmark, satpm_observation, "satpm_monthly", allow_missing=False)
    benchmark = _numeric(benchmark, satpm_prediction, "satpm_monthly", allow_missing=True)
    satpm_keys = _key_set(benchmark)
    prediction_keys = _key_set(paired)

    right = benchmark[list(MONTH_KEYS) + [satpm_observation, satpm_prediction]].rename(
        columns={satpm_observation: "satpm_obs", satpm_prediction: "satpm_prediction"}
    )
    monthly = paired.merge(right, on=list(MONTH_KEYS), how="left", validate="many_to_one",
                           indicator="_satpm_merge", sort=False)
    present = monthly._satpm_merge.eq("both")
    if present.any():
        _obs_equal(monthly.loc[present, "obs"], monthly.loc[present, "satpm_obs"],
                   label="model_oof/satpm_monthly")
    monthly["benchmark_row_present"] = present
    monthly["benchmark_unavailable_year"] = monthly.year.isin(unavailable)
    model_finite = np.isfinite(monthly.model_prediction.to_numpy(dtype=float))
    cams_finite = np.isfinite(monthly.raw_cams_prediction.to_numpy(dtype=float))
    satpm_finite = np.isfinite(monthly.satpm_prediction.to_numpy(dtype=float)) & present.to_numpy()
    monthly["missing_model_prediction"] = ~model_finite
    monthly["missing_raw_cams"] = ~cams_finite
    monthly["missing_benchmark"] = ~satpm_finite & ~monthly.benchmark_unavailable_year.to_numpy()
    monthly["shared_support"] = (
        model_finite & cams_finite & satpm_finite & ~monthly.benchmark_unavailable_year.to_numpy()
    )

    reasons = pd.Series("", index=monthly.index, dtype=object)
    for flag, label in (
        (monthly.benchmark_unavailable_year, "benchmark_unavailable_year"),
        (monthly.missing_model_prediction, "missing_model_prediction"),
        (monthly.missing_raw_cams, "missing_raw_cams"),
        (monthly.missing_benchmark, "missing_benchmark"),
    ):
        mask = flag.to_numpy(dtype=bool)
        existing = reasons.loc[mask]
        reasons.loc[mask] = np.where(existing.eq(""), label, existing + ";" + label)
    monthly["support_status"] = reasons.mask(reasons.eq(""), "included_shared_support")
    monthly = monthly.drop(columns=["_satpm_merge", "satpm_obs"])
    monthly["calendar_days"] = [calendar.monthrange(int(year), int(month))[1]
                                for year, month in zip(monthly.year, monthly.month)]

    identities = list(identity_columns)
    groups = identities + list(ANNUAL_KEYS)
    annual_rows: list[dict[str, object]] = []
    for labels, part in monthly.groupby(groups, dropna=False, sort=True):
        if not isinstance(labels, tuple):
            labels = (labels,)
        record = dict(zip(groups, labels))
        supported = part.loc[part.shared_support]
        support_months = sorted(int(value) for value in supported.month.tolist())
        shared_days = int(supported.calendar_days.sum())
        record.update({
            "source_month_rows": int(len(part)),
            "shared_months": len(support_months),
            "shared_month_set": tuple(support_months),
            "shared_calendar_days": shared_days,
            "excluded_missing_model_prediction_months": int(part.missing_model_prediction.sum()),
            "excluded_missing_raw_cams_months": int(part.missing_raw_cams.sum()),
            "excluded_missing_benchmark_months": int(part.missing_benchmark.sum()),
            "excluded_unavailable_year_months": int(part.benchmark_unavailable_year.sum()),
            "benchmark_version": version,
            "primary_month_support": len(support_months) == primary_months,
            "sensitivity_month_support": len(support_months) >= min_sensitivity_months,
        })
        values = {
            "observed_annual_pm25": "obs",
            "model_annual_pm25": "model_prediction",
            "raw_cams_annual_pm25": "raw_cams_prediction",
            "satpm_annual_pm25": "satpm_prediction",
        }
        for output_column, source_column in values.items():
            if shared_days:
                weighted = (supported[source_column].to_numpy(dtype=float)
                            * supported.calendar_days.to_numpy(dtype=float)).sum()
                record[output_column] = float(weighted / shared_days)
            else:
                record[output_column] = float("nan")
        if int(record["year"]) in unavailable:
            status = "benchmark_unavailable_year"
        elif not support_months:
            status = "missing_benchmark_or_prediction_support"
        elif len(support_months) < min_sensitivity_months:
            status = "below_sensitivity_support"
        elif len(support_months) == primary_months:
            status = "primary_12_month_support"
        else:
            status = "sensitivity_support_only"
        record["annual_status"] = status
        annual_rows.append(record)

    annual = pd.DataFrame.from_records(annual_rows)
    exclusion_counts = {
        "missing_model_prediction_month_rows": int(monthly.missing_model_prediction.sum()),
        "missing_raw_cams_month_rows": int(monthly.missing_raw_cams.sum()),
        "missing_benchmark_month_rows": int(monthly.missing_benchmark.sum()),
        "unavailable_year_month_rows": int(monthly.benchmark_unavailable_year.sum()),
        "shared_support_month_rows": int(monthly.shared_support.sum()),
        "benchmark_rows_outside_model_key_set": int(len(satpm_keys - prediction_keys)),
        "finite_benchmark_rows_ignored_in_unavailable_years": int(
            (benchmark.year.isin(unavailable)
             & np.isfinite(benchmark[satpm_prediction].to_numpy(dtype=float))).sum()
        ),
    }
    audit: dict[str, object] = {
        "format": "annual-exact-shared-support-kernel-v1",
        "benchmark_version": version,
        "monthly_join": join_audit,
        "monthly_key_join_to_satpm": "left; exact country/site/year/month; benchmark may be incomplete",
        "observed_targets_equal_within_tolerance": True,
        "observation_tolerance_absolute": OBS_TOLERANCE,
        "annual_weighting": "calendar days in month over exact common finite monthly support",
        "primary_months": primary_months,
        "minimum_sensitivity_months": min_sensitivity_months,
        "unavailable_years": list(unavailable),
        "annual_rows": int(len(annual)),
        "primary_annual_rows": int(annual.primary_month_support.sum()) if len(annual) else 0,
        "sensitivity_annual_rows": int(annual.sensitivity_month_support.sum()) if len(annual) else 0,
        "excluded_annual_rows_below_sensitivity": int(annual.annual_status.eq("below_sensitivity_support").sum()) if len(annual) else 0,
        "exclusions_and_support": exclusion_counts,
        "zero_filling_performed": False,
        "metric_computation_performed": False,
    }
    return monthly, annual, audit

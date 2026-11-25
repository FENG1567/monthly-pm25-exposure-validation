"""Shared, strict prediction interface for the repair-v2 validation pipeline.

The estimator predicts z = log(PM2.5 + 0.1). Primary inverse:
smear * exp(z) - 0.1. The sensitivity inverse is exp(z) - 0.1.
The default smearing estimate is cross-fitted over training stations only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "repair-v2-1"


@dataclass(frozen=True)
class ModelConfig:
    max_iter: int = 400
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 25
    l2_regularization: float = 1.0
    early_stopping: bool = True
    validation_fraction: float = 0.15
    random_state: int = 0
    estimator_kind: str = "hgb"
    ridge_alpha: float = 1.0


ESTIMATOR_CLASS_NAMES = {
    "hgb": "HistGradientBoostingRegressor",
    "ridge": "Ridge",
}


def _model_config(config: ModelConfig | Mapping[str, Any] | None = None) -> ModelConfig:
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config
    return ModelConfig(**dict(config))


def estimator(config: ModelConfig | Mapping[str, Any] | None = None):
    cfg = _model_config(config)
    if cfg.estimator_kind == "hgb":
        # Keep the historical HGB arguments explicit and unchanged. Ridge-only
        # configuration fields must not be passed to this estimator.
        return HistGradientBoostingRegressor(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            early_stopping=cfg.early_stopping,
            validation_fraction=cfg.validation_fraction,
            random_state=cfg.random_state,
        )
    if cfg.estimator_kind == "ridge":
        if not np.isfinite(cfg.ridge_alpha) or cfg.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=cfg.ridge_alpha)),
        ])
    raise ValueError("estimator_kind must be 'hgb' or 'ridge'")


def schema_for(
    base_features: Sequence[str],
    indicator_source_by_feature: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return an ordered schema; no data-dependent column filtering."""
    base = list(base_features)
    if not base or len(base) != len(set(base)):
        raise ValueError("base_features must be nonempty and unique")
    sources = dict(indicator_source_by_feature or {})
    if "aod_used" in base:
        sources.setdefault("aod_used", "aod_raw")
    indicators = ["aod_missing" if c == "aod_used" else f"missing__{c}" for c in base]
    if len(indicators) != len(set(indicators)) or set(base) & set(indicators):
        raise ValueError("base feature names collide with generated missing indicators")
    return {
        "schema_version": SCHEMA_VERSION,
        "base_features": base,
        "indicator_source_by_feature": sources,
        "required_source_columns": sorted(set(sources.values())),
        "indicator_columns": indicators,
        "model_columns": base + indicators,
    }


def _validate_frame(frame: pd.DataFrame, schema: Mapping[str, Any]) -> None:
    required = list(schema["base_features"]) + list(schema.get("required_source_columns", []))
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required schema columns: {missing}")
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema.get('schema_version')!r}")


def feature_matrix(frame: pd.DataFrame, schema: Mapping[str, Any]) -> np.ndarray:
    """Build an ordered float32 matrix and a missing indicator per base feature."""
    _validate_frame(frame, schema)
    values, flags = [], []
    for name in schema["base_features"]:
        col = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype="float32").copy()
        values.append(col)
        source = schema.get("indicator_source_by_feature", {}).get(name, name)
        source_col = pd.to_numeric(frame[source], errors="raise").to_numpy(dtype="float32")
        flags.append((~np.isfinite(source_col)).astype("float32"))
        col[~np.isfinite(col)] = np.nan
    matrix = np.column_stack(values + flags).astype("float32", copy=False)
    if matrix.shape[1] != len(schema["model_columns"]):
        raise AssertionError("generated matrix width does not match frozen schema")
    return matrix


def _target(train: pd.DataFrame, target_col: str) -> np.ndarray:
    if target_col not in train:
        raise ValueError(f"training target column {target_col!r} is absent")
    y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("training targets must be finite and nonnegative")
    return np.log(y + 0.1)


def _target_raw(train: pd.DataFrame, target_col: str) -> np.ndarray:
    if target_col not in train:
        raise ValueError(f"training target column {target_col!r} is absent")
    y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("training targets must be finite and nonnegative")
    return y


def _crossfit_smear(train, matrix, log_target, groups, config, n_splits):
    group_values = pd.Series(groups).astype(str).to_numpy()
    n_groups = len(np.unique(group_values))
    if n_groups < 2:
        raise ValueError("cross-fitted smearing requires at least two training stations")
    splits = min(int(n_splits), n_groups)
    residuals = np.full(len(train), np.nan, dtype="float64")
    for tr_i, cal_i in GroupKFold(n_splits=splits).split(matrix, groups=group_values):
        inner = estimator(config)
        inner.fit(matrix[tr_i], log_target[tr_i])
        residuals[cal_i] = log_target[cal_i] - inner.predict(matrix[cal_i])
    if not np.isfinite(residuals).all():
        raise AssertionError("cross-fit did not predict every training row")
    smear = float(np.mean(np.exp(residuals)))
    if not np.isfinite(smear) or smear <= 0:
        raise ValueError(f"invalid cross-fitted smearing factor {smear}")
    return smear, {
        "method": "crossfit_station", "n_splits": splits,
        "n_rows": int(len(train)), "n_stations": int(n_groups),
        "residual_source": "training-only group out-of-fold",
    }


def fit_model(
    train: pd.DataFrame,
    base_features: Sequence[str],
    *,
    target_col: str = "pm25",
    group_col: str = "site_id",
    config: ModelConfig | Mapping[str, Any] | None = None,
    smear_strategy: str = "crossfit_station",
    smear_splits: int = 5,
    target_strategy: str = "log_smear",
) -> dict[str, Any]:
    """Fit an estimator and train-only back-transform calibration."""
    if group_col not in train:
        raise ValueError(f"training group column {group_col!r} is absent")
    if train.empty:
        raise ValueError("cannot fit on an empty training frame")
    schema = schema_for(base_features)
    matrix = feature_matrix(train, schema)
    if target_strategy == "log_smear":
        model_target = _target(train, target_col)
        if smear_strategy == "crossfit_station":
            smear, smear_info = _crossfit_smear(
                train, matrix, model_target, train[group_col], config, smear_splits
            )
        elif smear_strategy == "in_sample":
            # Explicit product-parity sensitivity; in-sample residuals are optimistic.
            temp = estimator(config)
            temp.fit(matrix, model_target)
            smear = float(np.mean(np.exp(model_target - temp.predict(matrix))))
            if not np.isfinite(smear) or smear <= 0:
                raise ValueError(f"invalid in-sample smearing factor {smear}")
            smear_info = {
                "method": "in_sample", "n_splits": 0, "n_rows": int(len(train)),
                "n_stations": int(train[group_col].nunique()),
                "residual_source": "training in-sample; optimistic sensitivity",
            }
        else:
            raise ValueError("smear_strategy must be 'crossfit_station' or 'in_sample'")
        backtransform_primary = "smear*exp(z)-0.1"
        backtransform_sensitivity = "exp(z)-0.1"
    elif target_strategy == "original_scale":
        # Squared-error regression on concentration directly targets the
        # conditional arithmetic mean, as a distinct prespecified sensitivity.
        model_target = _target_raw(train, target_col)
        smear = 1.0
        smear_info = {
            "method": "not_applicable", "n_splits": 0, "n_rows": int(len(train)),
            "n_stations": int(train[group_col].nunique()),
            "residual_source": "direct original-scale squared-error fit",
        }
        backtransform_primary = "model_native_original_scale"
        backtransform_sensitivity = None
    else:
        raise ValueError("target_strategy must be 'log_smear' or 'original_scale'")
    cfg = _model_config(config)
    model = estimator(cfg)
    model.fit(matrix, model_target)
    config_values = asdict(cfg)
    return {
        "bundle_version": SCHEMA_VERSION, "schema": schema, "estimator": model,
        "config": config_values, "estimator_kind": cfg.estimator_kind,
        "estimator_class": ESTIMATOR_CLASS_NAMES[cfg.estimator_kind],
        "target_col": target_col, "group_col": group_col,
        "target_strategy": target_strategy,
        "smear": smear, "smear_info": smear_info,
        "backtransform_primary": backtransform_primary,
        "backtransform_sensitivity": backtransform_sensitivity,
        "fit_n": int(len(train)), "fit_stations": int(train[group_col].nunique()),
    }


def predict_model(bundle: Mapping[str, Any], frame: pd.DataFrame, *,
                  backtransform: str = "smearing", clip_at_zero: bool = True) -> np.ndarray:
    """Predict original-scale PM2.5 using an explicitly selected inverse."""
    if bundle.get("bundle_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or invalid model bundle")
    matrix = feature_matrix(frame, bundle["schema"])
    raw = np.asarray(bundle["estimator"].predict(matrix), dtype="float64")
    target_strategy = bundle.get("target_strategy", "log_smear")
    if target_strategy == "original_scale":
        if backtransform not in {"model_native", "original_scale"}:
            raise ValueError("original_scale target strategy requires backtransform='model_native'")
        pred = raw
    elif target_strategy == "log_smear" and backtransform == "smearing":
        pred = float(bundle["smear"]) * np.exp(raw) - 0.1
    elif target_strategy == "log_smear" and backtransform == "original":
        pred = np.exp(raw) - 0.1
    else:
        raise ValueError("log_smear requires backtransform='smearing' or 'original'")
    return np.maximum(pred, 0.0) if clip_at_zero else pred


def predict_grid_rows(bundle: Mapping[str, Any], frame: pd.DataFrame, *,
                      chunk_size: int = 250_000,
                      backtransform: str = "smearing") -> np.ndarray:
    """Chunked grid prediction with the same ordered schema as station prediction."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    out = np.empty(len(frame), dtype="float64")
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        out[start:stop] = predict_model(
            bundle, frame.iloc[start:stop], backtransform=backtransform
        )
    return out


def counterfactual_frame(frame: pd.DataFrame, fire_features: Sequence[str]) -> pd.DataFrame:
    """Return a copy with every declared fire predictor zeroed."""
    missing = [c for c in fire_features if c not in frame.columns]
    if missing:
        raise ValueError(f"counterfactual requires absent fire columns: {missing}")
    out = frame.copy()
    for name in fire_features:
        out[name] = 0.0
    return out


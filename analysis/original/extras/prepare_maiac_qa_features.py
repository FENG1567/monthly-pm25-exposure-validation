#!/usr/bin/env python
"""Attach MAIAC QA summaries to a frozen station-month feature cohort.

The join is performed by verified 0.1-degree cell geometry and year/month,
not station identifiers. This supports canonicalized monitor IDs while
preserving the original rows, targets, coordinates, and their order. Rows
outside the MAIAC cell domain remain explicitly missing; they are never given
zero AOD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_validation import sha256_file

KEYS = ["cell_id", "year", "month"]
STATION_KEYS = ["country", "site_id", "year", "month"]
QA_COPY_PREFIX = "maiac_qa_"
GEOMETRY_TOLERANCE = 1e-6
CELL_EDGE_TOLERANCE = 0.05001


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _normalize_country(frame: pd.DataFrame, country: str, label: str) -> pd.DataFrame:
    if "country" not in frame:
        raise ValueError(f"{label} has no country column")
    out = frame.copy()
    out["country"] = out.country.astype(str).str.upper()
    if not out.country.eq(country).all():
        observed = sorted(out.country.dropna().unique().tolist())
        raise ValueError(f"{label} contains countries {observed}, expected only {country}")
    return out


def _keyed_features(frame: pd.DataFrame, country: str) -> tuple[pd.DataFrame, list[str]]:
    frame = _normalize_country(frame, country, "features")
    required = ["site_id", "year", "month", "lat", "lon"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"features missing required station-month fields: {missing}")
    if {"grid_lat", "grid_lon"}.issubset(frame.columns):
        geometry = ["grid_lat", "grid_lon"]
    elif {"cell_lat", "cell_lon"}.issubset(frame.columns):
        geometry = ["cell_lat", "cell_lon"]
    else:
        raise ValueError(
            "features need explicit sampled-cell geometry columns grid_lat/grid_lon "
            "or cell_lat/cell_lon; station coordinates alone are insufficient"
        )
    out = frame.copy()
    out["site_id"] = out.site_id.astype(str)
    for name in ("year", "month"):
        out[name] = pd.to_numeric(out[name], errors="raise").astype("int64")
    for name in ("lat", "lon", *geometry):
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if out[STATION_KEYS].isna().any().any():
        raise ValueError("feature station-month keys contain missing values")
    if out.duplicated(STATION_KEYS).any():
        dup = out.loc[out.duplicated(STATION_KEYS, keep=False), STATION_KEYS].head(5)
        raise ValueError(f"feature station-month keys are not unique:\n{dup}")
    if not np.isfinite(out[geometry].to_numpy(dtype="float64")).all():
        raise ValueError("feature sampled-cell geometry contains non-finite values")
    out["_qa_geo_lat"] = out[geometry[0]].round(6)
    out["_qa_geo_lon"] = out[geometry[1]].round(6)
    if (
        (out.lat - out["_qa_geo_lat"]).abs().gt(CELL_EDGE_TOLERANCE).any()
        or (out.lon - out["_qa_geo_lon"]).abs().gt(CELL_EDGE_TOLERANCE).any()
    ):
        raise ValueError(
            "station coordinates disagree with their sampled 0.1-degree cell "
            f"by more than {CELL_EDGE_TOLERANCE} degrees"
        )
    return out, geometry


def _cell_geometry(lookup_path: Path) -> pd.DataFrame:
    lookup = _read(lookup_path)
    if {"cell_lat", "cell_lon"}.issubset(lookup.columns):
        geometry_columns = ["cell_lat", "cell_lon"]
    elif {"lat", "lon"}.issubset(lookup.columns):
        geometry_columns = ["lat", "lon"]
    else:
        geometry_columns = []
    if geometry_columns == ["lat", "lon"]:
        lookup = lookup.rename(columns={"lat": "cell_lat", "lon": "cell_lon"})
    required = ["cell_id", "cell_lat", "cell_lon"]
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(f"MAIAC cell lookup missing columns: {missing}")
    lookup = lookup[required].copy()
    lookup["cell_id"] = pd.to_numeric(lookup.cell_id, errors="raise").astype("int64")
    for name in ("cell_lat", "cell_lon"):
        lookup[name] = pd.to_numeric(lookup[name], errors="raise").round(6)
    if lookup.duplicated("cell_id").any():
        consistency = lookup.groupby("cell_id")[["cell_lat", "cell_lon"]].nunique()
        if (consistency > 1).any().any():
            raise ValueError("MAIAC lookup maps a cell_id to more than one geometry")
        lookup = lookup.drop_duplicates("cell_id")
    if lookup.duplicated(["cell_lat", "cell_lon"]).any():
        consistency = lookup.groupby(["cell_lat", "cell_lon"]).cell_id.nunique()
        if (consistency > 1).any():
            raise ValueError("MAIAC lookup maps one cell geometry to multiple cell IDs")
        lookup = lookup.drop_duplicates(["cell_lat", "cell_lon"])
    return lookup


def _validate_station_crosswalk(
    features: pd.DataFrame,
    station_qa_path: Path | None,
    *,
    country: str,
) -> dict[str, int]:
    if station_qa_path is None:
        return {"exact_station_keys_checked": 0, "exact_station_key_cell_mismatches": 0}
    station_qa = _read(station_qa_path)
    missing = [name for name in [*STATION_KEYS, "cell_id", "lat", "lon"] if name not in station_qa]
    if missing:
        raise ValueError(f"MAIAC station-month QA missing crosswalk columns: {missing}")
    if "country" not in station_qa:
        raise ValueError("MAIAC station-month QA has no country column")
    station_qa["country"] = station_qa.country.astype(str).str.upper()
    station_qa = station_qa.loc[station_qa.country.eq(country)].copy()
    if station_qa.empty:
        raise ValueError(f"MAIAC station-month QA has no {country} rows")
    station_qa = station_qa[[*STATION_KEYS, "cell_id", "lat", "lon"]].copy()
    station_qa["site_id"] = station_qa.site_id.astype(str)
    station_qa["year"] = pd.to_numeric(station_qa.year, errors="raise").astype("int64")
    station_qa["month"] = pd.to_numeric(station_qa.month, errors="raise").astype("int64")
    if station_qa.duplicated(STATION_KEYS).any():
        raise ValueError("MAIAC station-month QA has duplicate composite station keys")
    joined = features[[
        *STATION_KEYS, "maiac_cell_id", "maiac_cell_lat", "maiac_cell_lon"
    ]].merge(
        station_qa, on=STATION_KEYS, how="inner", validate="one_to_one",
        suffixes=("_feature", "_qa"),
    )
    feature_cell = pd.to_numeric(joined.maiac_cell_id, errors="coerce")
    qa_cell = pd.to_numeric(joined.cell_id, errors="coerce")
    mismatch = (
        feature_cell.isna()
        | qa_cell.isna()
        | feature_cell.ne(qa_cell)
        | (joined.maiac_cell_lat - joined.lat).abs().gt(GEOMETRY_TOLERANCE)
        | (joined.maiac_cell_lon - joined.lon).abs().gt(GEOMETRY_TOLERANCE)
    )
    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} exact station-key matches disagree in cell or station geometry"
        )
    return {
        "exact_station_keys_checked": int(len(joined)),
        "exact_station_key_cell_mismatches": int(mismatch.sum()),
    }


def _prepare_variant(
    features: pd.DataFrame,
    cell_month: pd.DataFrame,
    *,
    definition: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strict = definition == "strict"
    observed_name = "aod_mean" if strict else "sensitivity_aod_mean"
    fill_name = "aod_fill" if strict else "sensitivity_aod_fill"
    used_name = "aod_used" if strict else "sensitivity_aod_used"
    missing_name = "aod_missing" if strict else "sensitivity_aod_missing"

    out = features.copy()
    for name in (
        "aod", "aod_raw", "aod_fill", "aod_used", "aod_missing",
        "aod_raw_missing", "aod_used_missing",
    ):
        legacy_name = f"legacy_{name}"
        if name in out and legacy_name not in out:
            out[legacy_name] = out[name]
    qa_columns = [name for name in cell_month.columns if name not in KEYS]
    collision = [
        f"{QA_COPY_PREFIX}{name}" for name in qa_columns
        if f"{QA_COPY_PREFIX}{name}" in out
    ]
    if collision:
        raise ValueError(f"features already contain MAIAC QA columns: {collision[:5]}")
    qa = cell_month.rename(
        columns={
            "cell_id": "maiac_cell_id",
            **{name: f"{QA_COPY_PREFIX}{name}" for name in qa_columns},
        }
    )
    joined = out.merge(
        qa, on=["maiac_cell_id", "year", "month"], how="left",
        validate="many_to_one", sort=False, indicator="_maiac_cell_join",
    )
    qa_match = joined[f"{QA_COPY_PREFIX}{observed_name}"].notna() | joined[
        f"{QA_COPY_PREFIX}{missing_name}"
    ].notna()
    joined["maiac_qa_definition"] = definition
    joined["maiac_cell_geometry_verified"] = joined.maiac_cell_id.notna()
    joined["maiac_qa_cell_month_present"] = qa_match
    joined["maiac_qa_match_status"] = np.where(
        joined.maiac_cell_id.isna(), "no_cell_in_qa_lookup",
        np.where(qa_match, "matched_cell_month", "cell_known_month_missing"),
    )
    raw_aod = pd.to_numeric(joined[f"{QA_COPY_PREFIX}{observed_name}"], errors="coerce")
    fill_aod = pd.to_numeric(joined[f"{QA_COPY_PREFIX}{fill_name}"], errors="coerce")
    used_aod = pd.to_numeric(joined[f"{QA_COPY_PREFIX}{used_name}"], errors="coerce")
    raw_finite = np.isfinite(raw_aod.to_numpy(dtype="float64"))
    used_finite = np.isfinite(used_aod.to_numpy(dtype="float64"))
    expected_used = np.where(
        raw_finite,
        raw_aod.to_numpy(dtype="float64"),
        fill_aod.to_numpy(dtype="float64"),
    )
    if not np.array_equal(
        expected_used, used_aod.to_numpy(dtype="float64"), equal_nan=True
    ):
        raise ValueError(
            f"{definition} MAIAC used AOD is not observed AOD or its valid neighborhood fill"
        )
    joined["aod"] = raw_aod
    joined["aod_raw"] = raw_aod
    joined["aod_fill"] = fill_aod
    joined["aod_used"] = used_aod
    joined["aod_missing"] = (~raw_finite).astype("int8")
    joined["aod_raw_missing"] = (~raw_finite).astype("int8")
    joined["aod_used_missing"] = (~used_finite).astype("int8")

    producer_missing = pd.to_numeric(
        joined[f"{QA_COPY_PREFIX}{missing_name}"], errors="coerce"
    ).to_numpy(dtype="float64")
    expected_producer_missing = (~used_finite).astype("float64")
    matched_rows = qa_match.to_numpy(dtype=bool)
    if not np.array_equal(
        producer_missing[matched_rows], expected_producer_missing[matched_rows], equal_nan=True
    ):
        raise ValueError(
            f"{definition} namespaced MAIAC post-fill missing flag disagrees with used AOD"
        )

    if len(joined) != len(features):
        raise AssertionError("MAIAC QA join changed the frozen feature row count")
    if not joined[STATION_KEYS].reset_index(drop=True).equals(
        features[STATION_KEYS].reset_index(drop=True)
    ):
        raise AssertionError("MAIAC QA join changed feature row ordering or keys")
    targets_unchanged = True
    if "pm25" in features:
        targets_unchanged = bool(np.array_equal(
            features.pm25.to_numpy(copy=True), joined.pm25.to_numpy(), equal_nan=True
        ))
        if not targets_unchanged:
            raise AssertionError("MAIAC QA join changed frozen targets")
    replaced_aod_columns = {
        "aod", "aod_raw", "aod_fill", "aod_used", "aod_missing",
        "aod_raw_missing", "aod_used_missing",
    }
    preserved_feature_columns = [
        name for name in features.columns if name not in replaced_aod_columns
    ]
    changed = [
        name for name in preserved_feature_columns
        if not features[name].reset_index(drop=True).equals(
            joined[name].reset_index(drop=True)
        )
    ]
    if changed:
        raise AssertionError(f"MAIAC QA join changed original non-AOD columns: {changed[:10]}")

    imputed_rows = (~raw_finite) & used_finite

    summary = {
        "definition": definition,
        "row_count": int(len(joined)),
        "site_count": int(joined.site_id.nunique()),
        "cell_geometry_matched_rows": int(joined.maiac_cell_id.notna().sum()),
        "cell_month_qa_matched_rows": int(qa_match.sum()),
        "cell_month_missing_rows": int((joined.maiac_qa_match_status == "cell_known_month_missing").sum()),
        "no_cell_lookup_rows": int((joined.maiac_qa_match_status == "no_cell_in_qa_lookup").sum()),
        "observed_aod_finite_rows": int(np.isfinite(pd.to_numeric(joined.aod_raw, errors="coerce")).sum()),
        "used_aod_finite_rows": int(np.isfinite(pd.to_numeric(joined.aod_used, errors="coerce")).sum()),
        "aod_missing_flagged_rows": int(joined.aod_missing.astype(bool).sum()),
        "raw_aod_missing_rows": int((~raw_finite).sum()),
        "used_aod_missing_rows": int((~used_finite).sum()),
        "raw_missing_but_used_finite_rows": int(imputed_rows.sum()),
        "model_aod_equals_observed_aod": bool(np.array_equal(
            joined.aod.to_numpy(dtype="float64"),
            joined.aod_raw.to_numpy(dtype="float64"), equal_nan=True,
        )),
        "observed_aod_matches_namespaced_qa": bool(np.array_equal(
            joined.aod_raw.to_numpy(dtype="float64"),
            pd.to_numeric(joined[f"{QA_COPY_PREFIX}{observed_name}"], errors="coerce")
            .to_numpy(dtype="float64"), equal_nan=True,
        )),
        "fill_aod_matches_namespaced_qa": bool(np.array_equal(
            joined.aod_fill.to_numpy(dtype="float64"),
            pd.to_numeric(joined[f"{QA_COPY_PREFIX}{fill_name}"], errors="coerce")
            .to_numpy(dtype="float64"), equal_nan=True,
        )),
        "used_aod_matches_namespaced_qa": bool(np.array_equal(
            joined.aod_used.to_numpy(dtype="float64"),
            pd.to_numeric(joined[f"{QA_COPY_PREFIX}{used_name}"], errors="coerce")
            .to_numpy(dtype="float64"), equal_nan=True,
        )),
        "raw_missing_flags_match_observed_aod": bool(np.array_equal(
            joined.aod_missing.to_numpy(dtype="int8"), (~raw_finite).astype("int8")
        )),
        "used_missing_flags_match_used_aod": bool(np.array_equal(
            joined.aod_used_missing.to_numpy(dtype="int8"), (~used_finite).astype("int8")
        )),
        "imputed_rows_have_raw_missing_and_used_present": bool(
            joined.loc[imputed_rows, "aod_raw_missing"].eq(1).all()
            and joined.loc[imputed_rows, "aod_used_missing"].eq(0).all()
        ),
        "original_non_aod_columns_preserved": True,
        "aod_columns": {
            "observed": observed_name,
            "fill": fill_name,
            "used": used_name,
            "missing": missing_name,
        },
        "feature_row_count_unchanged": True,
        "feature_key_order_unchanged": True,
        "targets_unchanged": targets_unchanged,
    }
    return joined, summary


def prepare(
    *,
    features_path: Path,
    cell_month_path: Path,
    cell_lookup_path: Path,
    out_dir: Path,
    prefix: str,
    country: str,
    station_month_path: Path | None = None,
    overlay_status: str | None = None,
) -> dict[str, Any]:
    country = country.upper()
    features = _read(features_path)
    if {"maiac_cell_id", "_qa_geo_lat", "_qa_geo_lon"} & set(features.columns):
        raise ValueError("features already contain MAIAC QA geometry fields")
    features, geometry_columns = _keyed_features(features, country)
    cell_lookup = _cell_geometry(cell_lookup_path)
    feature_cells = features[["_qa_geo_lat", "_qa_geo_lon"]].copy()
    feature_cells["_qa_row_order"] = np.arange(len(feature_cells), dtype="int64")
    feature_cells = feature_cells.merge(
        cell_lookup,
        left_on=["_qa_geo_lat", "_qa_geo_lon"],
        right_on=["cell_lat", "cell_lon"],
        how="left", validate="many_to_one", sort=False,
    )
    if len(feature_cells) != len(features):
        raise AssertionError("cell geometry lookup changed feature row count")
    if not np.array_equal(
        feature_cells._qa_row_order.to_numpy(dtype="int64"),
        np.arange(len(features), dtype="int64"),
    ):
        raise AssertionError("cell geometry lookup changed feature row ordering")
    features = features.copy()
    if "cell_id" in features:
        original_cell_id = pd.to_numeric(features.cell_id, errors="coerce")
        mapped_cell_id = pd.to_numeric(feature_cells.cell_id, errors="coerce")
        if original_cell_id.notna().any() and not original_cell_id.equals(mapped_cell_id):
            raise ValueError("feature cell_id disagrees with the verified geometry lookup")
    features["maiac_cell_id"] = feature_cells.cell_id.to_numpy()
    features["maiac_cell_lat"] = feature_cells.cell_lat.to_numpy()
    features["maiac_cell_lon"] = feature_cells.cell_lon.to_numpy()
    mapped = features.maiac_cell_id.notna()
    if mapped.any():
        lat_delta = (features.loc[mapped, "_qa_geo_lat"] - features.loc[mapped, "maiac_cell_lat"]).abs()
        lon_delta = (features.loc[mapped, "_qa_geo_lon"] - features.loc[mapped, "maiac_cell_lon"]).abs()
        if (lat_delta.gt(GEOMETRY_TOLERANCE) | lon_delta.gt(GEOMETRY_TOLERANCE)).any():
            raise ValueError("feature sampled-cell geometry mismatches the MAIAC lookup")
    station_crosswalk = _validate_station_crosswalk(
        features, station_month_path, country=country
    )

    cell_month = _read(cell_month_path)
    missing = [name for name in [*KEYS, "lat", "lon", "aod_mean", "aod_used", "aod_missing",
                                 "sensitivity_aod_mean", "sensitivity_aod_used",
                                 "sensitivity_aod_missing"] if name not in cell_month]
    if missing:
        raise ValueError(f"MAIAC cell-month QA missing required columns: {missing}")
    for name in ("cell_id", "year", "month"):
        cell_month[name] = pd.to_numeric(cell_month[name], errors="raise").astype("int64")
    for name in ("lat", "lon"):
        cell_month[name] = pd.to_numeric(cell_month[name], errors="raise").round(6)
    if cell_month.duplicated(KEYS).any():
        raise ValueError("MAIAC cell-month QA has duplicate cell/year/month keys")
    qa_geometry = cell_month[["cell_id", "lat", "lon"]].drop_duplicates()
    qa_geometry = qa_geometry.merge(
        cell_lookup, on="cell_id", how="left", validate="many_to_one",
        suffixes=("_qa", "_lookup"),
    )
    if qa_geometry.cell_lat.isna().any():
        raise ValueError("MAIAC cell-month table contains cell IDs absent from its lookup")
    geom_delta = (
        (qa_geometry.lat - qa_geometry.cell_lat).abs().gt(GEOMETRY_TOLERANCE)
        | (qa_geometry.lon - qa_geometry.cell_lon).abs().gt(GEOMETRY_TOLERANCE)
    )
    if geom_delta.any():
        raise ValueError(f"{int(geom_delta.sum())} MAIAC QA cell geometries disagree with lookup")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    outputs = {}
    for definition in ("strict", "sensitivity"):
        result, summary = _prepare_variant(features, cell_month, definition=definition)
        result.drop(columns=["_qa_geo_lat", "_qa_geo_lon"], inplace=True)
        path = out_dir / f"{prefix}_{country.lower()}_{definition}.parquet"
        temp = path.with_suffix(path.suffix + ".tmp")
        result.to_parquet(temp, index=False)
        temp.replace(path)
        outputs[definition] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": int(len(result)),
        }
        variants[definition] = summary

    manifest = {
        "protocol": "left-join station features to MAIAC cell-month QA by verified sampled 0.1-degree geometry and year/month",
        "country": country,
        "prefix": prefix,
        "overlay_status": overlay_status or "status_unspecified",
        "builder_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "feature_input": {"path": str(features_path.resolve()), "sha256": sha256_file(features_path)},
        "maiac_cell_month_input": {"path": str(cell_month_path.resolve()), "sha256": sha256_file(cell_month_path)},
        "maiac_cell_lookup_input": {"path": str(cell_lookup_path.resolve()), "sha256": sha256_file(cell_lookup_path)},
        "maiac_station_month_crosswalk_input": {
            "path": str(station_month_path.resolve()), "sha256": sha256_file(station_month_path)
        } if station_month_path else None,
        "geometry_columns": geometry_columns,
        "geometry_tolerance_degrees": GEOMETRY_TOLERANCE,
        "station_within_sampled_cell_edge_tolerance_degrees": CELL_EDGE_TOLERANCE,
        "join_key": ["cell_id", "year", "month"],
        "canonical_site_ids_supported_by_geometry": True,
        "station_crosswalk_check": station_crosswalk,
        "unmatched_aod_policy": "preserve row; strict and sensitivity AOD remain missing and aod_missing=1; never fill with zero",
        "preserved_legacy_columns": [
            "legacy_aod", "legacy_aod_raw", "legacy_aod_fill", "legacy_aod_used",
            "legacy_aod_missing", "legacy_aod_raw_missing", "legacy_aod_used_missing",
        ],
        "outputs": outputs,
        "variants": variants,
    }
    manifest_path = out_dir / f"{prefix}_{country.lower()}_qa_merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cell-month-qa", required=True)
    parser.add_argument("--cell-lookup", required=True)
    parser.add_argument("--station-month-qa")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument(
        "--overlay-status",
        help="record the validation status of the MAIAC support overlay in the merge manifest",
    )
    args = parser.parse_args()
    result = prepare(
        features_path=Path(args.features).resolve(),
        cell_month_path=Path(args.cell_month_qa).resolve(),
        cell_lookup_path=Path(args.cell_lookup).resolve(),
        station_month_path=Path(args.station_month_qa).resolve() if args.station_month_qa else None,
        out_dir=Path(args.out_dir).resolve(), prefix=args.prefix, country=args.country,
        overlay_status=args.overlay_status,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

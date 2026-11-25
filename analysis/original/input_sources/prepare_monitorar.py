"""QC and aggregate official MonitorAr PM2.5 archives for 2022-2025.

Source ZIPs are read-only. The primary series applies a conservative,
operational VA-only selection; alternative statuses remain separately audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(r"F:\cand1_20261105\incoming\monitorar")
OUT = Path(__file__).resolve().parents[3] / "outputs" / "repair_v2" / "monitorar"
YEARS_ALL = [2022, 2023, 2024, 2025]
ITEM_TARGET = "Material Particulado com diâmetro de até 2,5 μm"
NORMALIZED_TARGET = "MP2,5"
ITEM_NORM = unicodedata.normalize("NFKC", ITEM_TARGET).casefold()
MIN_VALID_HOURS_PER_DAY = 18
MIN_VALID_DAYS_PER_MONTH = 20
RELOCATION_THRESHOLD_METERS = 100.0
SENTINEL_VALUES = {985.0, 995.0}
NONPRIMARY_STATUS = {"IV", "IM", "ID"}
SOURCE_COLUMNS = [
    "no_fonte_dados", "dh_medicao", "nu_iqar", "nu_concentracao",
    "cd_flag", "ds_flag", "st_situacao_iqar", "no_item_monitorado", "cd_normalizado",
    "id_estacao", "cod_estacao", "no_estacao", "nu_latitude", "nu_longitude",
    "st_estacao", "id_municipio", "no_municipio", "sg_uf",
]
STATUS_FIELD_CANDIDATES = ("st_situacao", "st_situacao_item")

# Descriptions explicitly indicating failed, missing, or unusable measurements.
# Unknown descriptions are retained in the VA primary series and reported.
INVALID_FLAG_TERMS = [
    "invalid", "inval", "invlid", "missing", "sem dado", "sem medicao",
    "ausente", "rompimento de fita", "sem fita", "fluxo inadequado",
    "fluxo fora dos limites", "amostra insuficiente", "desvio da densidade",
    "desvio de fluxo", "desvio de leitura", "pm2,5>pm10", "pm 2,5 > pm 10",
    "falha", "erro de leitura", "impossibilidade de verificacao",
    "interrupcao no fornecimento", "manutencao", "calibracao do analisador",
    "termino da fita", "substituicao da fita",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    # Some archived free-text fields contain a replacement character or '?'
    # in place of an accented byte; remove it for conservative term matching.
    s = s.replace("\ufffd", "").replace("?", "")
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s).casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def invalid_flag_reason(code: Any, description: Any) -> str:
    text = normalized_text(f"{code or ''} {description or ''}")
    for term in INVALID_FLAG_TERMS:
        if term in text:
            return term
    return ""


def safe_counter(counter: Counter, name: str, year: int) -> list[dict]:
    return [{"year": year, name: k, "rows": int(v)} for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))]


def histogram_rows(values: pd.Series, year: int, category: str) -> list[dict]:
    if values.empty:
        return []
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return []
    edges = [-np.inf, 0, 1, 5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 300, 500, 985, 995, 1000, np.inf]
    labels = ["<0", "[0,1)", "[1,5)", "[5,10)", "[10,15)", "[15,25)", "[25,35)",
              "[35,50)", "[50,75)", "[75,100)", "[100,150)", "[150,200)", "[200,300)",
              "[300,500)", "[500,985)", "[985,995)", "[995,1000)", ">=1000"]
    counts = np.histogram(numeric, bins=edges)[0]
    return [{"year": year, "category": category, "value_bin_ug_m3": label,
             "rows": int(n)} for label, n in zip(labels, counts) if n]


def distance_m(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6_371_000.0
    a1, a2 = np.radians(lat1.astype(float)), np.radians(lat2.astype(float))
    da = a2 - a1
    dl = np.radians(lon2.astype(float) - lon1.astype(float))
    hav = np.sin(da / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.minimum(1, hav)))


def profile_archive(year: int) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]:
    path = SOURCE_ROOT / f"dados_monitorar_{year}.zip"
    if not path.exists():
        raise FileNotFoundError(path)
    inventory = {"year_archive": year, "source_path": str(path), "zip_bytes": path.stat().st_size,
                 "zip_sha256": sha256(path), "zip_crc_error_member": "", "csv_member": "",
                 "status_source_field": "",
                 "csv_uncompressed_bytes": 0, "source_rows": 0, "timestamp_valid_rows": 0,
                 "source_year_rows": 0, "out_of_source_year_rows": 0,
                 "target_item_rows": 0, "target_item_code_mismatch_rows": 0,
                 "pm25_exact_item_and_code_rows": 0}
    item_codes: Counter = Counter()
    status_flags: Counter = Counter()
    funnels: Counter = Counter()
    histograms: list[dict] = []
    candidate_frames = []
    secondary_frames = []
    min_timestamp = None
    max_timestamp = None

    with zipfile.ZipFile(path) as zf:
        if len(zf.namelist()) != 1:
            raise ValueError(f"Expected one CSV member in {path.name}; found {zf.namelist()}")
        inventory["csv_member"] = zf.namelist()[0]
        inventory["csv_uncompressed_bytes"] = int(zf.getinfo(zf.namelist()[0]).file_size)
        inventory["zip_crc_error_member"] = zf.testzip() or ""
        if inventory["zip_crc_error_member"]:
            raise ValueError(f"ZIP CRC failure in {path}: {inventory['zip_crc_error_member']}")
        header = pd.read_csv(zf.open(zf.namelist()[0]), nrows=0, encoding="utf-8").columns.tolist()
        missing = set(SOURCE_COLUMNS) - set(header)
        if missing:
            raise ValueError(f"{path.name} is missing required source columns: {sorted(missing)}")
        status_field = next((name for name in STATUS_FIELD_CANDIDATES if name in header), None)
        if status_field is None:
            raise ValueError(f"{path.name} has no recognized validation status field: {STATUS_FIELD_CANDIDATES}")
        inventory["status_source_field"] = status_field
        with zf.open(zf.namelist()[0]) as f:
            for chunk in pd.read_csv(f, usecols=SOURCE_COLUMNS + [status_field], dtype=str,
                                     chunksize=500_000, encoding="utf-8"):
                # The 2024 archive renamed the measurement-validation field to
                # st_situacao_item. Keep the original source column and provide
                # a stable alias for shared QC logic/output tables.
                chunk["status_source_field"] = status_field
                chunk["st_situacao"] = chunk[status_field]
                n = len(chunk)
                inventory["source_rows"] += n
                item = chunk["no_item_monitorado"].fillna("").map(
                    lambda x: unicodedata.normalize("NFKC", str(x)).casefold().strip())
                code = chunk["cd_normalizado"].fillna("").astype(str).str.strip().str.upper()
                combos = pd.DataFrame({"item": chunk["no_item_monitorado"].fillna("<NA>"),
                                       "normalized_code": chunk["cd_normalizado"].fillna("<NA>")})
                item_codes.update({tuple(k): int(v) for k, v in combos.value_counts(dropna=False).items()})

                ts = pd.to_datetime(chunk["dh_medicao"], errors="coerce")
                inventory["timestamp_valid_rows"] += int(ts.notna().sum())
                if ts.notna().any():
                    lo, hi = ts.min(), ts.max()
                    min_timestamp = lo if min_timestamp is None or lo < min_timestamp else min_timestamp
                    max_timestamp = hi if max_timestamp is None or hi > max_timestamp else max_timestamp
                source_year = ts.dt.year.eq(year)
                inventory["source_year_rows"] += int(source_year.sum())
                inventory["out_of_source_year_rows"] += int((ts.notna() & ~source_year).sum())

                target_item = item.eq(ITEM_NORM)
                target_code = code.eq(NORMALIZED_TARGET)
                inventory["target_item_rows"] += int(target_item.sum())
                inventory["target_item_code_mismatch_rows"] += int((target_item & ~target_code).sum())
                exact = chunk.loc[target_item & target_code].copy()
                if exact.empty:
                    continue
                exact_ts = ts.loc[exact.index]
                exact_status = exact["st_situacao"].fillna("").astype(str).str.strip().str.upper()
                exact_values = pd.to_numeric(exact["nu_concentracao"], errors="coerce")
                exact_lat = pd.to_numeric(exact["nu_latitude"], errors="coerce")
                exact_lon = pd.to_numeric(exact["nu_longitude"], errors="coerce")
                reasons = [invalid_flag_reason(c, d) for c, d in zip(exact["cd_flag"], exact["ds_flag"])]
                exact["timestamp_local"] = exact_ts
                exact["source_archive_year"] = year
                exact["value_ug_m3"] = exact_values.astype(float)
                exact["latitude_numeric"] = exact_lat.astype(float)
                exact["longitude_numeric"] = exact_lon.astype(float)
                exact["status_code_normalized"] = exact_status
                exact["invalid_flag_reason"] = reasons
                exact["known_invalid_missing_flag"] = pd.Series(reasons, index=exact.index).ne("")
                exact["is_sentinel_985_995"] = exact_values.isin(SENTINEL_VALUES)
                exact["source_year_matches_archive"] = exact_ts.dt.year.eq(year)
                exact["has_valid_station_id"] = exact["id_estacao"].fillna("").astype(str).str.strip().ne("")
                exact["has_valid_timestamp"] = exact_ts.notna()
                exact["has_valid_coordinates"] = (exact_lat.between(-90, 90) & exact_lon.between(-180, 180))
                exact["has_valid_value"] = exact_values.notna() & np.isfinite(exact_values) & exact_values.ge(0) & ~exact["is_sentinel_985_995"]
                exact["primary_selected"] = (
                    exact["status_code_normalized"].eq("VA")
                    & ~exact["known_invalid_missing_flag"]
                    & exact["has_valid_value"]
                    & exact["source_year_matches_archive"]
                    & exact["has_valid_station_id"]
                    & exact["has_valid_timestamp"]
                )
                exact["secondary_nonprimary_status"] = (
                    ~exact["status_code_normalized"].isin({"VA", *NONPRIMARY_STATUS})
                    & ~exact["known_invalid_missing_flag"]
                    & exact["has_valid_value"]
                    & exact["source_year_matches_archive"]
                    & exact["has_valid_station_id"]
                    & exact["has_valid_timestamp"]
                )
                inventory["pm25_exact_item_and_code_rows"] += int(len(exact))
                for (status, flag, desc), count in exact.groupby(
                        ["status_code_normalized", "cd_flag", "ds_flag"], dropna=False).size().items():
                    status_flags[(str(status or "<NA>"), str(flag or "<NA>"), str(desc or "<NA>"))] += int(count)
                funnel_mask_counts = {
                    "target_item_and_code_all_status": np.ones(len(exact), dtype=bool),
                    "status_VA_before_flag_value_year_checks": exact["status_code_normalized"].eq("VA"),
                    "status_IV_IM_ID_excluded": exact["status_code_normalized"].isin(NONPRIMARY_STATUS),
                    "other_nonVA_status_codes": ~exact["status_code_normalized"].isin({"VA", *NONPRIMARY_STATUS}),
                    "known_invalid_or_missing_flag_description": exact["known_invalid_missing_flag"],
                    "sentinel_985_or_995": exact["is_sentinel_985_995"],
                    "missing_or_nonnumeric_concentration": exact["value_ug_m3"].isna(),
                    "negative_concentration": exact["value_ug_m3"].notna() & exact["value_ug_m3"].lt(0),
                    "invalid_timestamp": ~exact["has_valid_timestamp"],
                    "missing_station_id": ~exact["has_valid_station_id"],
                    "invalid_or_missing_coordinates": ~exact["has_valid_coordinates"],
                    "primary_after_all_rules_before_spatial_coord_check": exact["primary_selected"],
                    "secondary_nonprimary_candidate": exact["secondary_nonprimary_status"],
                }
                for name, mask in funnel_mask_counts.items():
                    funnels[name] += int(np.asarray(mask).sum())
                values = exact["value_ug_m3"]
                categories = {
                    "raw_exact_item_and_code_all_status": values,
                    "raw_status_VA_before_quality_filters": values.loc[exact["status_code_normalized"].eq("VA")],
                    "raw_status_IV_IM_ID": values.loc[exact["status_code_normalized"].isin(NONPRIMARY_STATUS)],
                    "raw_other_nonVA_status": values.loc[~exact["status_code_normalized"].isin({"VA", *NONPRIMARY_STATUS})],
                    "raw_known_invalid_missing_flag": values.loc[exact["known_invalid_missing_flag"]],
                    "raw_sentinel_985_995": values.loc[exact["is_sentinel_985_995"]],
                    "primary_after_all_rules": values.loc[exact["primary_selected"]],
                    "secondary_nonprimary_candidate": values.loc[exact["secondary_nonprimary_status"]],
                }
                for category, series in categories.items():
                    histograms.extend(histogram_rows(series, year, category))
                secondary = exact.loc[exact["secondary_nonprimary_status"]].copy()
                if not secondary.empty:
                    secondary_frames.append(secondary)
                candidate_frames.append(exact)

    inventory["source_datetime_min"] = min_timestamp.isoformat(sep=" ") if min_timestamp is not None else ""
    inventory["source_datetime_max"] = max_timestamp.isoformat(sep=" ") if max_timestamp is not None else ""
    item_rows = []
    for (item_name, normalized_code), count in item_codes.items():
        item_rows.append({"year": year, "no_item_monitorado": item_name,
                          "cd_normalizado": normalized_code, "rows": count})
    flag_rows = [{"year": year, "status_source_field": inventory["status_source_field"],
                  "st_situacao": status, "cd_flag": flag,
                  "ds_flag": desc, "rows": count}
                 for (status, flag, desc), count in status_flags.items()]
    funnel_rows = [{"year": year, "stage": stage, "rows": count}
                   for stage, count in funnels.items()]
    candidate_data = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    return inventory, candidate_data, {
        "item_code": pd.DataFrame(item_rows),
        "status_flag": pd.DataFrame(flag_rows),
        "funnel": pd.DataFrame(funnel_rows),
        "histogram": pd.DataFrame(histograms),
        "secondary": pd.concat(secondary_frames, ignore_index=True) if secondary_frames else pd.DataFrame(),
    }


def aggregate_primary(pm25: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pm25.loc[pm25.primary_selected].copy()
    if primary.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    primary["station_id"] = "BR_" + primary["id_estacao"].astype(str).str.strip()
    primary["local_hour"] = primary.timestamp_local.dt.floor("h")
    primary["date"] = primary.timestamp_local.dt.date
    primary["year"] = primary.timestamp_local.dt.year.astype(int)
    primary["month"] = primary.timestamp_local.dt.month.astype(int)
    primary["has_source_flag"] = primary.cd_flag.fillna("").astype(str).str.strip().ne("")
    hourly = (primary.groupby(["station_id", "id_estacao", "local_hour", "date", "year", "month"], as_index=False, dropna=False)
              .agg(pm25=("value_ug_m3", "mean"), n_source_rows=("value_ug_m3", "size"),
                   n_rows_with_any_cd_flag=("has_source_flag", "sum")))
    hourly = hourly.rename(columns={"local_hour": "datetime_local"})
    hourly["iso"] = "BR"
    hourly["country"] = "Brazil"
    hourly["region"] = "MonitorAr"
    hourly["site_id"] = hourly.station_id
    hourly["unit"] = "µg/m³ (MMA dictionary mapping for MP2,5)"
    hourly["time_basis"] = "source naive local wall time; timezone not supplied"

    daily = (hourly.groupby(["iso", "region", "station_id", "id_estacao", "date", "year", "month"], as_index=False, dropna=False)
             .agg(pm25=("pm25", "mean"), n_valid_hours=("datetime_local", "nunique"),
                  n_source_rows=("n_source_rows", "sum"), n_rows_with_any_cd_flag=("n_rows_with_any_cd_flag", "sum")))
    daily = daily.loc[daily.n_valid_hours.ge(MIN_VALID_HOURS_PER_DAY)].copy()
    daily["site_id"] = daily.station_id
    daily["unit"] = "µg/m³ (MMA dictionary mapping for MP2,5)"
    daily["time_basis"] = "source naive local wall time; timezone not supplied"

    # Coordinates are station metadata. Use all date-valid PM2.5 source rows,
    # even when a reading fails the concentration validation rule.
    coord_source = pm25.loc[
        pm25.has_valid_coordinates & pm25.has_valid_timestamp
        & pm25.source_year_matches_archive & pm25.has_valid_station_id
    ].copy()
    coord_source["station_id"] = "BR_" + coord_source["id_estacao"].astype(str).str.strip()
    coord_source["date"] = coord_source.timestamp_local.dt.date
    coord_source["year"] = coord_source.timestamp_local.dt.year.astype(int)
    coord_source["month"] = coord_source.timestamp_local.dt.month.astype(int)
    coordinate_month = (coord_source.groupby(["station_id", "id_estacao", "year", "month"], as_index=False)
                        .agg(lat=("latitude_numeric", "median"), lon=("longitude_numeric", "median"),
                             n_coordinate_rows=("latitude_numeric", "size"),
                             n_unique_coordinate_pairs=("latitude_numeric", lambda x: 0)))
    # Count unique (latitude, longitude) pairs without letting missing pairs enter.
    pair_counts = (coord_source.drop_duplicates(["station_id", "year", "month", "latitude_numeric", "longitude_numeric"])
                   .groupby(["station_id", "year", "month"]).size().rename("n_unique_coordinate_pairs").reset_index())
    coordinate_month = coordinate_month.drop(columns="n_unique_coordinate_pairs").merge(
        pair_counts, on=["station_id", "year", "month"], how="left")
    station_coordinate = (coord_source.groupby(["station_id", "id_estacao"], as_index=False)
                          .agg(station_lat_reference=("latitude_numeric", "median"),
                               station_lon_reference=("longitude_numeric", "median")))
    coordinate_month = coordinate_month.merge(station_coordinate, on=["station_id", "id_estacao"], how="left")
    coordinate_month["distance_from_station_reference_m"] = distance_m(
        coordinate_month.lat, coordinate_month.lon,
        coordinate_month.station_lat_reference, coordinate_month.station_lon_reference)
    coord_with_month_median = coord_source.merge(
        coordinate_month[["station_id", "year", "month", "lat", "lon"]],
        on=["station_id", "year", "month"], how="left")
    coord_with_month_median["distance_from_month_median_m"] = distance_m(
        coord_with_month_median.latitude_numeric, coord_with_month_median.longitude_numeric,
        coord_with_month_median.lat, coord_with_month_median.lon)
    within_month_spread = (coord_with_month_median.groupby(["station_id", "year", "month"], as_index=False)
                           .agg(max_within_month_coordinate_spread_m=("distance_from_month_median_m", "max")))
    coordinate_month = coordinate_month.merge(within_month_spread,
                                                on=["station_id", "year", "month"], how="left")
    coordinate_month["coordinate_in_brazil_bbox"] = (
        coordinate_month.lat.between(-35.5, 6.5) & coordinate_month.lon.between(-74.5, -32.0))
    coordinate_month["relocation_suspected"] = (
        coordinate_month.distance_from_station_reference_m.gt(RELOCATION_THRESHOLD_METERS)
        | coordinate_month.max_within_month_coordinate_spread_m.gt(RELOCATION_THRESHOLD_METERS)
    )
    coordinate_month["spatially_ready"] = (
        coordinate_month.lat.between(-90, 90) & coordinate_month.lon.between(-180, 180)
        & coordinate_month.coordinate_in_brazil_bbox
    )
    monthly = (daily.groupby(["iso", "region", "station_id", "id_estacao", "year", "month"], as_index=False, dropna=False)
              .agg(pm25=("pm25", "mean"), n_valid_dates=("date", "nunique"),
                   n_valid_hours=("n_valid_hours", "sum"), n_source_rows=("n_source_rows", "sum"),
                   n_rows_with_any_cd_flag=("n_rows_with_any_cd_flag", "sum")))
    monthly = monthly.loc[monthly.n_valid_dates.ge(MIN_VALID_DAYS_PER_MONTH)].copy()
    monthly = monthly.merge(coordinate_month[["station_id", "id_estacao", "year", "month", "lat", "lon",
                                             "n_coordinate_rows", "n_unique_coordinate_pairs",
                                             "distance_from_station_reference_m", "max_within_month_coordinate_spread_m",
                                             "coordinate_in_brazil_bbox",
                                             "relocation_suspected", "spatially_ready"]],
                            on=["station_id", "id_estacao", "year", "month"], how="left")
    monthly["coordinate_missing"] = monthly.lat.isna() | monthly.lon.isna()
    monthly["spatially_ready"] = monthly.spatially_ready.astype("boolean").fillna(False).astype(bool)
    monthly["relocation_suspected"] = monthly.relocation_suspected.astype("boolean").fillna(False).astype(bool)
    monthly["time_basis"] = "source naive local wall time; timezone not supplied"
    monthly["unit"] = "µg/m³ (MMA dictionary mapping for MP2,5)"
    monthly["site_id"] = monthly.station_id
    monthly["date"] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    if (monthly.pm25 < 0).any():
        raise ValueError("Negative monthly PM2.5 values remain after nonnegative hourly selection")
    # Valid days/months must meet the requested coverage gates and be nonnegative.
    return hourly, daily, monthly, coordinate_month


def run(years: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventories, candidates, tables = [], [], {k: [] for k in ["item_code", "status_flag", "funnel", "histogram", "secondary"]}
    for year in years:
        inv, pm, profile = profile_archive(year)
        inventories.append(inv)
        candidates.append(pm)
        for key, frame in profile.items():
            if not frame.empty:
                tables[key].append(frame)
        print(f"profiled MonitorAr {year}: source_rows={inv['source_rows']:,} PM2.5_rows={inv['pm25_exact_item_and_code_rows']:,}", flush=True)
    inventory_df = pd.DataFrame(inventories)
    pm25 = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    hourly, daily, monthly, coord = aggregate_primary(pm25)
    label = f"{min(years)}_{max(years)}"
    inventory_df.to_csv(OUT / f"archive_inventory_{label}.csv", index=False)
    for key, filename in [("item_code", f"item_code_counts_{label}.csv"),
                          ("status_flag", f"status_flag_counts_{label}.csv"),
                          ("funnel", f"qc_row_funnel_{label}.csv"),
                          ("histogram", f"concentration_histogram_{label}.csv")]:
        frames = tables[key]
        pd.concat(frames, ignore_index=True).to_csv(OUT / filename, index=False) if frames else pd.DataFrame().to_csv(OUT / filename, index=False)
    secondary = pd.concat(tables["secondary"], ignore_index=True) if tables["secondary"] else pd.DataFrame()
    if not pm25.empty:
        pm25.to_parquet(OUT / f"pm25_qc_observations_{label}.parquet", index=False)
    if not secondary.empty:
        secondary.to_parquet(OUT / f"pm25_secondary_nonprimary_{label}.parquet", index=False)
    hourly.to_parquet(OUT / f"station_hour_{label}.parquet", index=False)
    daily.to_parquet(OUT / f"station_day_{label}.parquet", index=False)
    monthly.to_parquet(OUT / f"station_month_{label}.parquet", index=False)
    coordinate_audit = monthly[["station_id", "id_estacao", "year", "month", "lat", "lon",
                                "n_coordinate_rows", "n_unique_coordinate_pairs",
                                "distance_from_station_reference_m", "max_within_month_coordinate_spread_m",
                                "coordinate_in_brazil_bbox", "relocation_suspected", "coordinate_missing",
                                "spatially_ready"]].copy()
    coordinate_audit.to_csv(OUT / f"station_month_coordinate_audit_{label}.csv", index=False)
    dup = (pm25.loc[pm25.primary_selected].assign(
        _station_id=lambda d: "BR_" + d.id_estacao.astype(str).str.strip(),
        _hour=lambda d: d.timestamp_local.dt.floor("h"))
        .groupby(["_station_id", "_hour"]).size()) if not pm25.empty else pd.Series(dtype=int)
    accepted_rows = int(pm25.primary_selected.sum()) if not pm25.empty else 0
    hourly_rows = int(len(hourly))
    duplicate_excess = int(accepted_rows - hourly_rows)
    duplicate_groups = int((dup > 1).sum()) if not dup.empty else 0
    extreme = (pm25.loc[pm25.primary_selected & pm25.value_ug_m3.gt(200)].copy()
               if not pm25.empty else pd.DataFrame())
    if not extreme.empty:
        extreme.to_csv(OUT / f"extreme_primary_pm25_gt200_{label}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years": years,
        "source_root_read_only": str(SOURCE_ROOT),
        "archives": inventories,
        "source_total_rows": int(inventory_df.source_rows.sum()),
        "pm25_exact_item_and_code_rows": int(inventory_df.pm25_exact_item_and_code_rows.sum()),
        "primary_selected_readings": accepted_rows,
        "primary_unique_station_hours": hourly_rows,
        "accepted_duplicate_excess_rows_averaged_within_station_hour": duplicate_excess,
        "station_hour_groups_with_duplicates": duplicate_groups,
        "primary_valid_days_ge18_hours": int(len(daily)),
        "primary_station_months_ge20_days": int(len(monthly)),
        "stations": int(monthly.station_id.nunique()) if not monthly.empty else 0,
        "station_months_spatially_ready": int(monthly.spatially_ready.sum()) if not monthly.empty else 0,
        "station_months_with_missing_coordinates": int(monthly.coordinate_missing.sum()) if not monthly.empty else 0,
        "station_months_with_relocation_signal_gt100m": int(monthly.relocation_suspected.sum()) if not monthly.empty else 0,
        "primary_readings_gt200_ug_m3_retained_for_review": int(len(extreme)),
        "primary_readings_gt1000_ug_m3_retained_for_review": int(extreme.value_ug_m3.gt(1000).sum()) if not extreme.empty else 0,
        "secondary_nonprimary_candidate_readings": int(len(secondary)),
        "status_source_field_by_year": {str(row["year_archive"]): row["status_source_field"] for row in inventories},
        "status_code_rule": "Conservative operational VA-only primary; IV, IM, ID and all other non-VA codes are not mixed into primary. Other non-VA records satisfying value/year/timestamp/flag checks are separately preserved as candidates.",
        "invalid_flag_rule": INVALID_FLAG_TERMS,
        "sentinel_exclusions_ug_m3": sorted(SENTINEL_VALUES),
        "hourly_deduplication": "Average accepted nonnegative PM2.5 readings within BR_{id_estacao} and floored source-local hour.",
        "daily_rule": f"Average unique accepted station-hour means; keep days with at least {MIN_VALID_HOURS_PER_DAY} distinct valid hours.",
        "monthly_rule": f"Average retained daily means; keep months with at least {MIN_VALID_DAYS_PER_MONTH} distinct retained dates; monthly PM2.5 is nonnegative.",
        "unit": "MMA data dictionary maps cd_normalizado=MP2,5 to µg/m³. The CSV has no separate unit column.",
        "timezone": "dh_medicao is naive in the source; retained as provider-local wall time with no UTC conversion. No timezone field is supplied.",
        "coordinates": f"Station-month coordinate is the median of date-valid PM2.5 source metadata coordinate pairs (independent of measurement status/value); relocation audit flags >{RELOCATION_THRESHOLD_METERS:g} m within month or from station-wide median. Spatial readiness also requires a broad Brazil bounding box.",
        "official_references": {
            "data_dictionary": "https://dados.mma.gov.br/dataset/5be05b46-3bda-4f6e-9bf2-810e716fff33/resource/aa83d585-2064-4424-8324-d5da520a0470/download/dicionario_monitorar.pdf",
            "field_dictionary_and_quality_notice": "https://dados.mma.gov.br/dataset/5be05b46-3bda-4f6e-9bf2-810e716fff33/resource/09c0df22-3688-45f6-b8da-40eb55012477/download/arquivo-auxiliar-dados-monitorar.pdf",
            "dataset": "https://dados.mma.gov.br/dataset/ar-puro-monitorar",
        },
        "caveat": "The source uses st_situacao_item in 2024 and st_situacao in other years; both are retained and normalized to one QC alias. The official field dictionary defines status and flag fields but does not expand the observed status codes. VA-only is an operational conservative rule, not an official code interpretation. MMA warns that automatic data may contain inconsistencies, gaps, and records not technically validated; it recommends obtaining validated data from the station operators for high-precision use.",
    }
    (OUT / f"qc_summary_{label}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"archives", "official_references", "invalid_flag_rule"}}, indent=2, ensure_ascii=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2022,2023", help="Comma-separated archive years (default: critical checkpoint years 2022,2023)")
    args = parser.parse_args()
    years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    if not years or any(y not in YEARS_ALL for y in years):
        raise ValueError(f"Years must be selected from {YEARS_ALL}")
    run(years)


if __name__ == "__main__":
    main()

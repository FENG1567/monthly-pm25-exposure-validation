"""Pinned, outcome-free normalization of historical/future HMS plume months."""
from pathlib import PureWindowsPath
import json
from pathlib import Path

import numpy as np
import pandas as pd
import future_core_gate_v1 as gate_api


OUTPUT_COLUMNS = [
    'country', 'site_id', 'lat', 'lon', 'year', 'month', 'point_id',
    'alignment', 'plume_definition', 'calendar_days', 'n_available_days',
    'n_missing_calendar_days', 'n_plume_days_with_available_calendar',
    'plume_fraction_available_calendar', 'n_plume_days',
    'n_known_classification_days', 'n_unknown_classification_days',
    'plume_fraction_known_days', 'calendar_availability_fraction', 'support_80pct',
]
ALIGNMENTS = ('same_named_date', 'plusminus1day')
DEFINITIONS = {'any_plume': 1, 'medium_heavy': 2}
FUTURE_STATS = ('n_plume_days', 'n_known_classification_days',
                'n_unknown_classification_days', 'plume_fraction_known_days',
                'n_plume_days_with_available_calendar',
                'plume_fraction_available_calendar', 'n_available_days',
                'n_missing_calendar_days', 'calendar_availability_fraction')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _basename(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def _bool(series, name, nullable=False):
    if not nullable:
        require(series.notna().all(), name + ' contains missing values')
    vals = series.dropna()
    require(vals.isin([True, False]).all(), name + ' is not Boolean')
    return series.astype('boolean' if nullable else bool)


def _num_equal(actual, expected, name):
    a = pd.to_numeric(actual, errors='raise').to_numpy(dtype=float)
    b = pd.to_numeric(expected, errors='raise').to_numpy(dtype=float)
    require(a.shape == b.shape and np.array_equal(a, b, equal_nan=True),
            'HMS exact numeric source reconstruction differs: ' + name)


def _eq_frame(left, right, cols, name):
    require(len(left) == len(right), name + ' row count differs')
    for col in cols:
        if col in ('lat', 'lon'):
            _num_equal(left[col], right[col], name + ':' + col)
        else:
            a, b = left[col].reset_index(drop=True), right[col].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                _num_equal(a, b, name + ':' + col)
            else:
                require(a.astype('string').equals(b.astype('string')),
                        name + ' differs: ' + col)


def _read_pins(gate):
    cfg = gate.get('configuration', {})
    pins = cfg.get('HMS_postfit_input_pins')
    require(isinstance(pins, dict) and len(pins) == 15,
            'complete accepted HMS_postfit_input_pins required')
    paths, by_name = {}, {}
    # Hash every configured object before opening any JSON or Parquet.
    for raw, digest in sorted(pins.items()):
        path = gate_api.pinned(raw, digest, gate['used'])
        require(path.name not in by_name, 'duplicate HMS pinned basename: ' + path.name)
        paths[path.name] = path
        by_name[path.name] = digest
    expected = {
        'hms_2023': {'station_points.parquet', 'hms_station_day_2023.parquet',
                     'hms_station_month_2023.parquet', 'hms_validation_provenance.json'},
        'hms_future': {'hms_future_station_points.parquet',
                       'hms_future_station_month_points.parquet',
                       'hms_station_day_2024.parquet', 'hms_station_day_2025.parquet',
                       'hms_station_month_day_detail_2024.parquet',
                       'hms_station_month_day_detail_2025.parquet',
                       'hms_station_month_2024.parquet', 'hms_station_month_2025.parquet',
                       'root_table_acceptance.json', 'FAILED_PARTIAL.json',
                       'hms_future_recovery_provenance.json'},
    }
    require(expected['hms_2023'].issubset(paths) and expected['hms_future'].issubset(paths)
            and set(paths) == expected['hms_2023'] | expected['hms_future'],
            'HMS pin census/schema differs')
    return paths, by_name


def _metadata(paths, digests):
    meta = {n: json.loads(paths[n].read_text(encoding='utf-8')) for n in paths
            if n.endswith('.json')}
    root = meta['root_table_acceptance.json']
    recovery = meta['hms_future_recovery_provenance.json']
    failed = meta['FAILED_PARTIAL.json']
    old = meta['hms_validation_provenance.json']
    require(root.get('status') == 'root_saved_table_hash_schema_and_rowcount_checks_passed'
            and root.get('no_pm_or_model_scores_computed') is True,
            'future HMS root table acceptance metadata differs')
    require(root.get('source_provenance_sha256') == digests['hms_future_recovery_provenance.json'],
            'future HMS root acceptance does not bind the recovery provenance')
    require(recovery.get('status') == 'metadata_only_recovery_verified'
            and recovery.get('prepared_version') == 'v3_polygon_only_invalid_geometry_repair_with_degenerate_feature_ledger'
            and recovery.get('all_official_assets_rehashed_and_match_manifest') is True
            and recovery.get('target_or_model_columns_read') is False
            and recovery.get('future_PM_scores_read_or_used') is False
            and recovery.get('overlay_rerun') is False,
            'future HMS metadata-only recovery provenance differs')
    require(failed.get('status') == 'failed_partial_tables_written_provenance_not_written'
            and failed.get('official_raw_source_status') ==
            'Both NOAA annual bundles and 24 monthly indexes remain verified and valid.',
            'preserved HMS partial-run marker differs')
    require(old.get('phase') == 'pre-MAIAC-QA supplementary 2023 consistency analysis'
            and old.get('unique_station_coordinates') == 1217
            and old.get('daily_point_rows') == 444205
            and old.get('target_or_model_columns_read', False) is False,
            '2023 HMS provenance metadata differs')
    records = root.get('tables', [])
    require(len(records) == 8, 'future HMS accepted table metadata census differs')
    for record in records:
        name = _basename(record['path'])
        require(name in digests and record.get('sha256') == digests[name]
                and record.get('no_pm_targets_or_model_prediction_columns') is True,
                'future HMS accepted table pin differs: ' + name)
    outputs = recovery.get('prepared_table_files', [])
    require(len(outputs) == 8, 'future HMS recovery table metadata census differs')
    for record in outputs:
        name = _basename(record['path'])
        require(name in digests and record.get('sha256') == digests[name],
                'future HMS recovery table hash differs: ' + name)
    return meta


def _frame(paths, name):
    return pd.read_parquet(paths[name])


def _columns(frame, required, name):
    require(set(required).issubset(frame.columns), name + ' columns missing')
    forbidden = ('prediction', 'model_score', 'pm25', 'pm10', 'target_value')
    require(not any(any(word in str(c).lower() for word in forbidden) for c in frame.columns),
            name + ' unexpectedly contains model/PM outcome data')


def _identity_unique(frame, keys, name):
    require(not frame.duplicated(keys).any(), name + ' contains duplicate source keys')


def _valid_geometry(points, id_col='point_id', name='HMS points'):
    _columns(points, [id_col, 'country', 'site_id', 'lat', 'lon'], name)
    _identity_unique(points, [id_col], name + ' point IDs')
    xy = points[['lat', 'lon']].to_numpy(dtype=float)
    require(np.isfinite(xy).all() and (np.abs(xy[:, 0]) <= 90).all()
            and (np.abs(xy[:, 1]) <= 180).all(), name + ' coordinates invalid')
    _identity_unique(points, ['country', 'site_id', 'lat', 'lon'], name + ' coordinate keys')


def _dates(daily, year, id_col='point_id'):
    d = daily.copy()
    d['date'] = pd.to_datetime(d['date'], errors='raise')
    require(d.date.notna().all() and d.date.dt.normalize().eq(d.date).all(),
            'HMS daily date is not a named calendar date')
    require(d.date.dt.year.eq(year).all(), f'HMS daily table year differs: {year}')
    if 'year' in d:
        require(pd.to_numeric(d.year, errors='raise').to_numpy().astype('int64').tolist()
                == d.date.dt.year.astype('int64').tolist(), 'HMS source year disagrees with date')
    if 'month' in d:
        require(pd.to_numeric(d.month, errors='raise').to_numpy().astype('int64').tolist()
                == d.date.dt.month.astype('int64').tolist(), 'HMS source month disagrees with date')
    d['year'] = d.date.dt.year.astype('int64')
    d['month'] = d.date.dt.month.astype('int64')
    _identity_unique(d, [id_col, 'date'], f'HMS {year} daily rows')
    return d


def _check_calendar(daily, year, expected_ids):
    days = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
    require(len(days) == (366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365),
            'internal HMS calendar length error')
    counts = daily.groupby('point_id').date.nunique()
    require(set(counts.index) == set(expected_ids) and counts.eq(len(days)).all()
            and len(daily) == len(expected_ids) * len(days),
            f'HMS {year} daily point/calendar census differs')
    observed_dates = set(daily.date.drop_duplicates())
    require(observed_dates == set(days), f'HMS {year} calendar has missing or extra dates')
    return len(days)


def _lag_density(d):
    g = d.groupby('point_id', sort=False)
    before = g.density_code.shift(1).fillna(0).to_numpy(dtype=np.uint8)
    after = g.density_code.shift(-1).fillna(0).to_numpy(dtype=np.uint8)
    return np.maximum.reduce([d.density_code.to_numpy(dtype=np.uint8), before, after])


def _reaggregate_2023(points, daily):
    required = ['date', 'point_id', 'density_code', 'archive_available',
                'density_code_plusminus1day', 'archive_available_plusminus1day']
    _columns(daily, required, '2023 HMS daily')
    _valid_geometry(points, name='2023 HMS points')
    d = _dates(daily, 2023).sort_values(['point_id', 'date'], kind='stable').reset_index(drop=True)
    require(set(d.point_id) == set(points.point_id), '2023 HMS daily point IDs do not map to source points')
    _check_calendar(d, 2023, points.point_id)
    for col in ('archive_available', 'archive_available_plusminus1day'):
        d[col] = _bool(d[col], col)
    code = d.density_code.to_numpy(dtype=np.uint8)
    require(np.isin(code, [0, 1, 2, 3]).all(), '2023 HMS density code outside 0-3')
    lag = _lag_density(d)
    require(np.array_equal(lag, d.density_code_plusminus1day.to_numpy(dtype=np.uint8)),
            '2023 HMS plus/minus-one-day density is not reconstructed from daily rows')
    g = d.groupby('point_id', sort=False)
    av = d.archive_available.to_numpy(bool)
    before = g.archive_available.shift(1).fillna(False).to_numpy(bool)
    after = g.archive_available.shift(-1).fillna(False).to_numpy(bool)
    lag_av = av & before & after
    require(np.array_equal(lag_av, d.archive_available_plusminus1day.to_numpy(bool)),
            '2023 HMS lag availability does not exclude incomplete year edges')
    d['_density_same'] = code
    d['_density_lag'] = lag
    records = []
    for alignment, density_col, avail_col in (
        ('same_named_date', '_density_same', 'archive_available'),
        ('plusminus1day', '_density_lag', 'archive_available_plusminus1day')):
        for definition, threshold in DEFINITIONS.items():
            avail = d[avail_col].to_numpy(bool)
            positive = d[density_col].to_numpy(dtype=np.uint8) >= threshold
            # 2023 source has a zero-filled raw density matrix. Mask it by
            # indexed archives before counting; unavailable dates are unknown.
            d['_pos'] = (positive & avail).astype('int64')
            d['_avail'] = avail.astype('int64')
            x = d.groupby(['point_id', 'year', 'month'], sort=True).agg(
                calendar_days=('date', 'nunique'), n_plume_days=('_pos', 'sum'),
                n_available_days=('_avail', 'sum')).reset_index()
            x['n_missing_calendar_days'] = x.calendar_days - x.n_available_days
            x['n_known_classification_days'] = x.n_available_days
            x['n_unknown_classification_days'] = x.n_missing_calendar_days
            x['n_plume_days_with_available_calendar'] = x.n_plume_days
            x['plume_fraction_available_calendar'] = x.n_plume_days / x.n_available_days.replace(0, np.nan)
            x['plume_fraction_known_days'] = x.plume_fraction_available_calendar
            x['calendar_availability_fraction'] = x.n_available_days / x.calendar_days
            x['support_80pct'] = x.n_available_days.ge(np.ceil(.8 * x.calendar_days))
            x['alignment'] = alignment
            x['plume_definition'] = definition
            records.append(x)
    long = pd.concat(records, ignore_index=True).merge(
        points[['point_id', 'country', 'site_id', 'lat', 'lon']],
        on='point_id', how='left', validate='many_to_one')
    require(long.country.notna().all(), '2023 normalized HMS geometry is incomplete')
    return long


def _verify_2023_monthly(long, monthly):
    _columns(monthly, ['point_id', 'year', 'month', 'n_available', 'any_plume_days',
                       'medium_heavy_days', 'calendar_days', 'any_plume_fraction',
                       'medium_heavy_fraction'], '2023 HMS monthly')
    require(monthly.year.eq(2023).all(), '2023 HMS monthly contains another year')
    _identity_unique(monthly, ['point_id', 'year', 'month'], '2023 HMS monthly')
    expected = long.loc[long.alignment.eq('same_named_date')].copy()
    _identity_unique(expected, ['point_id', 'year', 'month', 'plume_definition'], '2023 reconstruction')
    keys = ['point_id', 'year', 'month']
    for definition, count_col, frac_col in (
        ('any_plume', 'any_plume_days', 'any_plume_fraction'),
        ('medium_heavy', 'medium_heavy_days', 'medium_heavy_fraction')):
        part = expected.loc[expected.plume_definition.eq(definition)].sort_values(keys).reset_index(drop=True)
        pub = monthly.sort_values(keys).reset_index(drop=True)
        for key in keys:
            _num_equal(part[key], pub[key], '2023 HMS daily/monthly key ' + key)
        for actual, source in (('calendar_days', 'calendar_days'),
                               ('n_available_days', 'n_available'),
                               ('n_plume_days_with_available_calendar', count_col),
                               ('plume_fraction_available_calendar', frac_col)):
            _num_equal(part[actual], pub[source], '2023 ' + source)


def _future_daily(points, months, daily, year):
    required = ['date', 'point_id', 'year', 'month', 'density_code', 'archive_available',
                'archive_missing', 'density_code_plusminus1day',
                'plusminus1day_window_complete_within_classification_year',
                'plusminus1day_incomplete_reason', 'same_named_any_plume',
                'same_named_medium_heavy', 'plusminus1day_any_plume',
                'plusminus1day_medium_heavy', 'country', 'site_id', 'lat', 'lon', 'cohort']
    _columns(daily, required, f'{year} HMS daily')
    d = _dates(daily, year).sort_values(['point_id', 'date'], kind='stable').reset_index(drop=True)
    ids = months.point_id.unique()
    require(set(d.point_id) == set(ids), f'{year} HMS daily vs month-point IDs differ')
    calendar_days = _check_calendar(d, year, ids)
    _bool(d.archive_available, 'archive_available')
    _bool(d.archive_missing, 'archive_missing')
    require(np.array_equal(d.archive_missing.to_numpy(bool), ~d.archive_available.to_numpy(bool)),
            f'{year} HMS archive_missing disagrees with archive_available')
    require(np.isin(d.density_code.to_numpy(), [0, 1, 2, 3]).all(), 'invalid future HMS density code')
    expected_lag = _lag_density(d)
    require(np.array_equal(expected_lag, d.density_code_plusminus1day.to_numpy(dtype=np.uint8)),
            f'{year} HMS lag density differs from the within-year daily window')
    g = d.groupby('point_id', sort=False)
    av = d.archive_available.to_numpy(bool)
    complete = av & g.archive_available.shift(1).fillna(False).to_numpy(bool) & \
        g.archive_available.shift(-1).fillna(False).to_numpy(bool)
    lag_flag = _bool(d.plusminus1day_window_complete_within_classification_year,
                     'plusminus1day_window_complete_within_classification_year')
    require(np.array_equal(complete, lag_flag.to_numpy(bool)),
            f'{year} HMS lag archive completeness differs at boundary/missing dates')
    reason = np.where(complete, '', 'one_or_more_daily_archive_indexes_missing')
    at_edge = g.cumcount().to_numpy() == 0
    at_edge |= g.cumcount(ascending=False).to_numpy() == 0
    reason[at_edge] = 'outside_classification_year_window'
    require(np.array_equal(reason, d.plusminus1day_incomplete_reason.astype(str).to_numpy()),
            f'{year} HMS lag incompleteness reason differs')
    for name, density, available in (
        ('same_named_any_plume', d.density_code.to_numpy(), av),
        ('same_named_medium_heavy', d.density_code.to_numpy(), av),
        ('plusminus1day_any_plume', expected_lag, complete),
        ('plusminus1day_medium_heavy', expected_lag, complete)):
        threshold = 1 if name.endswith('any_plume') else 2
        positive = density >= threshold
        flag = _bool(d[name], name, nullable=True)
        require(np.array_equal(flag.notna().to_numpy(bool), positive | available)
                and np.array_equal(flag.eq(True).fillna(False).to_numpy(bool), positive),
                f'{year} HMS nullable classification differs: {name}')
    # Point IDs bind the complete source identity, including relocations.
    p = points.set_index('point_id')
    for frame_name, frame in (('daily', d), ('station month', months)):
        for col in ('country', 'site_id', 'cohort'):
            if col in frame.columns:
                expected = frame.point_id.map(p[col])
                require(expected.notna().all() and expected.astype(str).reset_index(drop=True).equals(
                    frame[col].astype(str).reset_index(drop=True)),
                    f'{year} HMS {frame_name} point {col} differs')
        for col in ('lat', 'lon'):
            if col in frame.columns:
                _num_equal(frame[col], frame.point_id.map(p[col]), f'{year} HMS {frame_name} point {col}')
    return d, calendar_days


def _future_month_detail(months, daily, detail, year):
    idcols = ['station_month_id', 'country', 'site_id', 'year', 'month', 'lat', 'lon',
              'cohort', 'source_path', 'point_id']
    _columns(months, idcols, f'{year} HMS station-month points')
    _columns(detail, idcols + ['date', 'archive_available',
        'plusminus1day_window_complete_within_classification_year',
        'plusminus1day_incomplete_reason', 'same_named_any_plume',
        'same_named_medium_heavy', 'plusminus1day_any_plume', 'plusminus1day_medium_heavy'],
        f'{year} HMS detail')
    require(months.year.eq(year).all() and detail.year.eq(year).all(),
            f'{year} HMS month table has another year')
    require(months.month.between(1, 12).all(), 'invalid HMS month')
    _identity_unique(months, ['station_month_id'], f'{year} HMS station-month IDs')
    source_key = ['country', 'site_id', 'lat', 'lon', 'year', 'month', 'point_id']
    _identity_unique(months, source_key, f'{year} HMS source coordinate-month')
    _identity_unique(detail, ['station_month_id', 'date'], f'{year} HMS detail keys')
    for col in ('archive_available', 'plusminus1day_window_complete_within_classification_year'):
        _bool(detail[col], col)
    day_cols = ['point_id', 'year', 'month', 'date', 'archive_available',
        'plusminus1day_window_complete_within_classification_year',
        'plusminus1day_incomplete_reason', 'same_named_any_plume',
        'same_named_medium_heavy', 'plusminus1day_any_plume', 'plusminus1day_medium_heavy']
    expected = months[idcols].merge(daily[day_cols], on=['point_id', 'year', 'month'],
        how='left', validate='one_to_many', sort=False, suffixes=('_month', '_daily'))
    require(expected.date.notna().all() and len(expected) == len(detail),
            f'{year} HMS month/detail calendar row census differs')
    expected_counts = expected.groupby('station_month_id').date.nunique()
    month_lengths = expected[['station_month_id', 'year', 'month']].drop_duplicates().set_index('station_month_id').loc[expected_counts.index]
    expected_lengths = pd.PeriodIndex(year=month_lengths.year, month=month_lengths.month, freq='M').days_in_month
    require(expected_counts.eq(expected_lengths).all(), f'{year} HMS station-month detail calendar is incomplete')
    expected = expected.rename(columns={c + '_month': c for c in
        ['country', 'site_id', 'lat', 'lon', 'cohort']})
    # Daily metadata is independently required to agree with source-month geometry.
    for col in ('country', 'site_id', 'lat', 'lon', 'cohort'):
        right = col + '_daily'
        if right in expected:
            if col in ('lat', 'lon'):
                _num_equal(expected[col], expected[right], f'{year} detail daily geometry {col}')
            else:
                require(expected[col].astype(str).equals(expected[right].astype(str)),
                        f'{year} detail daily identity {col} differs')
    compare = ['station_month_id', 'point_id', 'country', 'site_id', 'year', 'month',
               'lat', 'lon', 'cohort', 'source_path', 'date', 'archive_available',
               'plusminus1day_window_complete_within_classification_year',
               'plusminus1day_incomplete_reason', 'same_named_any_plume',
               'same_named_medium_heavy', 'plusminus1day_any_plume', 'plusminus1day_medium_heavy']
    left = expected[compare].sort_values(['station_month_id', 'date'], kind='stable').reset_index(drop=True)
    right = detail[compare].sort_values(['station_month_id', 'date'], kind='stable').reset_index(drop=True)
    for col in compare:
        a, b = left[col], right[col]
        if col in ('lat', 'lon'):
            _num_equal(a, b, f'{year} month detail coordinate {col}')
        elif col in ('same_named_any_plume', 'same_named_medium_heavy',
                     'plusminus1day_any_plume', 'plusminus1day_medium_heavy'):
            require(np.array_equal(a.isna().to_numpy(), b.isna().to_numpy()) and
                    np.array_equal(a.eq(True).fillna(False).to_numpy(bool), b.eq(True).fillna(False).to_numpy(bool)),
                    f'{year} month detail classification differs: {col}')
        else:
            require(a.astype('string').equals(b.astype('string')),
                    f'{year} month detail/daily source differs: {col}')
    return detail.copy()


def _reaggregate_future(detail):
    d = detail.copy()
    d['date'] = pd.to_datetime(d.date, errors='raise')
    _identity_unique(d, ['station_month_id', 'date'], 'future HMS detail date keys')
    base = d.groupby('station_month_id', sort=True).agg(
        calendar_days=('date', 'nunique')).reset_index()
    for alignment in ALIGNMENTS:
        avail_col = ('archive_available' if alignment == 'same_named_date'
                     else 'plusminus1day_window_complete_within_classification_year')
        avail = _bool(d[avail_col], avail_col).to_numpy(bool)
        d['_available'] = avail.astype('int64')
        for definition in DEFINITIONS:
            col = (('same_named_' if alignment == 'same_named_date' else 'plusminus1day_') +
                   ('any_plume' if definition == 'any_plume' else 'medium_heavy'))
            flag = _bool(d[col], col, nullable=True)
            pos = flag.eq(True).fillna(False).to_numpy(bool)
            known = flag.notna().to_numpy(bool)
            d['_positive'] = pos.astype('int64')
            d['_known'] = known.astype('int64')
            d['_available_positive'] = (pos & avail).astype('int64')
            prefix = alignment + '_' + definition
            g = d.groupby('station_month_id', sort=True).agg(
                n_plume_days=('_positive', 'sum'),
                n_known_classification_days=('_known', 'sum'),
                n_plume_days_with_available_calendar=('_available_positive', 'sum'),
                n_available_days=('_available', 'sum')).reset_index()
            g['n_unknown_classification_days'] = base.calendar_days.to_numpy() - g.n_known_classification_days.to_numpy()
            g['n_missing_calendar_days'] = base.calendar_days.to_numpy() - g.n_available_days.to_numpy()
            g['plume_fraction_known_days'] = g.n_plume_days / g.n_known_classification_days.replace(0, np.nan)
            g['plume_fraction_available_calendar'] = g.n_plume_days_with_available_calendar / g.n_available_days.replace(0, np.nan)
            g['calendar_availability_fraction'] = g.n_available_days / base.calendar_days.to_numpy()
            for stat in FUTURE_STATS:
                base[prefix + '_' + stat] = g[stat].to_numpy()
    same = base['same_named_date_any_plume_n_available_days']
    lag = base['plusminus1day_any_plume_n_available_days']
    base['same_named_date_support_80pct'] = same.ge(np.ceil(.8 * base.calendar_days))
    base['plusminus1day_support_80pct'] = lag.ge(np.ceil(.8 * base.calendar_days))
    base['support_80pct_archive_availability'] = base.same_named_date_support_80pct
    base['support_80pct_lag_window_completeness'] = base.plusminus1day_support_80pct
    return base


def _verify_future_monthly(monthly, months, detail, reagg, year):
    _columns(monthly, ['station_month_id', 'country', 'site_id', 'year', 'month', 'lat', 'lon',
        'cohort', 'source_path', 'point_id', 'calendar_days',
        'support_80pct_archive_availability', 'same_named_date_support_80pct',
        'plusminus1day_support_80pct', 'support_80pct_lag_window_completeness'],
        f'{year} HMS wide monthly')
    _identity_unique(monthly, ['station_month_id'], f'{year} HMS wide monthly IDs')
    _identity_unique(months, ['station_month_id'], f'{year} HMS month point IDs')
    ids = ['station_month_id']
    expected = months[ids + ['country', 'site_id', 'year', 'month', 'lat', 'lon', 'cohort', 'source_path', 'point_id']].merge(
        reagg, on=ids, how='outer', validate='one_to_one', indicator=True, sort=True)
    require(expected._merge.eq('both').all() and len(expected) == len(monthly),
            f'{year} HMS month points/detail/wide keys differ')
    expected = expected.drop(columns='_merge')
    _eq_frame(expected.sort_values(ids).reset_index(drop=True),
        monthly.sort_values(ids).reset_index(drop=True),
        ['country', 'site_id', 'year', 'month', 'lat', 'lon', 'cohort', 'source_path', 'point_id'],
        f'{year} HMS wide metadata')
    # Every published counter, fraction, calendar count, and support flag is reconstructed.
    compare = ['calendar_days']
    for alignment in ALIGNMENTS:
        for definition in DEFINITIONS:
            compare.extend(alignment + '_' + definition + '_' + stat for stat in FUTURE_STATS)
        compare.append(alignment + '_support_80pct')
    compare.extend(['support_80pct_archive_availability', 'support_80pct_lag_window_completeness'])
    for col in compare:
        require(col in monthly.columns, f'{year} HMS published statistic missing: {col}')
        _num_equal(expected[col], monthly.sort_values(ids).reset_index(drop=True)[col],
                    f'{year} HMS monthly {col}')


def _long_future(months, reagg):
    merged = months.merge(reagg, on='station_month_id', how='left', validate='one_to_one', sort=True)
    rows = []
    base_cols = ['country', 'site_id', 'lat', 'lon', 'year', 'month', 'point_id', 'calendar_days']
    for alignment in ALIGNMENTS:
        for definition in DEFINITIONS:
            prefix = alignment + '_' + definition + '_'
            part = merged[base_cols + [prefix + s for s in FUTURE_STATS]].copy()
            part = part.rename(columns={prefix + 'n_plume_days': 'n_plume_days',
                prefix + 'n_known_classification_days': 'n_known_classification_days',
                prefix + 'n_unknown_classification_days': 'n_unknown_classification_days',
                prefix + 'plume_fraction_known_days': 'plume_fraction_known_days',
                prefix + 'n_plume_days_with_available_calendar': 'n_plume_days_with_available_calendar',
                prefix + 'plume_fraction_available_calendar': 'plume_fraction_available_calendar',
                prefix + 'n_available_days': 'n_available_days',
                prefix + 'n_missing_calendar_days': 'n_missing_calendar_days',
                prefix + 'calendar_availability_fraction': 'calendar_availability_fraction'})
            part['alignment'] = alignment
            part['plume_definition'] = definition
            part['support_80pct'] = merged[alignment + '_support_80pct'].to_numpy()
            rows.append(part[OUTPUT_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def _long_2023(long):
    rows = []
    for alignment in ALIGNMENTS:
        for definition in DEFINITIONS:
            part = long.loc[long.alignment.eq(alignment) & long.plume_definition.eq(definition)].copy()
            part = part.rename(columns={'n_available_days': 'n_available_days'})
            rows.append(part[OUTPUT_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def load_monthly(gate):
    """Return normalized HMS coordinate-month classifications and source audit.

    Every configured HMS source pin is hashed before any metadata or table read.
    No PM targets, predictions, or model artifacts are opened here.
    """
    paths, digests = _read_pins(gate)
    metadata = _metadata(paths, digests)
    # Only the pinned HMS tables are opened; no model/PM inputs are consumed.
    points23 = _frame(paths, 'station_points.parquet')
    day23 = _frame(paths, 'hms_station_day_2023.parquet')
    month23 = _frame(paths, 'hms_station_month_2023.parquet')
    points_f = _frame(paths, 'hms_future_station_points.parquet')
    months_f = _frame(paths, 'hms_future_station_month_points.parquet')
    _valid_geometry(points_f, name='future HMS points')
    _columns(months_f, ['station_month_id', 'country', 'site_id', 'year', 'month',
        'lat', 'lon', 'cohort', 'source_path', 'point_id'], 'future HMS month points')
    _identity_unique(months_f, ['station_month_id'], 'future HMS station-month points')
    _identity_unique(months_f, ['country', 'site_id', 'lat', 'lon', 'year', 'month', 'point_id'],
                     'future HMS source coordinate-month')
    long23 = _reaggregate_2023(points23, day23)
    _verify_2023_monthly(long23, month23)
    future_longs, validation = [], {}
    for year in (2024, 2025):
        months = months_f.loc[months_f.year.eq(year)].copy().reset_index(drop=True)
        daily = _frame(paths, f'hms_station_day_{year}.parquet')
        detail = _frame(paths, f'hms_station_month_day_detail_{year}.parquet')
        monthly = _frame(paths, f'hms_station_month_{year}.parquet')
        daily, days = _future_daily(points_f, months, daily, year)
        detail = _future_month_detail(months, daily, detail, year)
        reagg = _reaggregate_future(detail)
        _verify_future_monthly(monthly, months, detail, reagg, year)
        future_longs.append(_long_future(months, reagg))
        validation[str(year)] = {'station_points': int(months.point_id.nunique()),
            'station_months': int(len(months)), 'calendar_days': int(days),
            'daily_rows': int(len(daily)), 'month_day_detail_rows': int(len(detail)),
            'wide_month_rows': int(len(monthly)), 'wide_monthly_counters_reconstructed': True,
            'detail_matches_daily_point_date_and_geometry': True}
    long23 = _long_2023(long23)
    out = pd.concat([long23, *future_longs], ignore_index=True)
    out = out[OUTPUT_COLUMNS].sort_values(
        ['country', 'site_id', 'lat', 'lon', 'year', 'month', 'alignment', 'plume_definition'],
        kind='stable').reset_index(drop=True)
    key = ['country', 'site_id', 'lat', 'lon', 'year', 'month', 'alignment', 'plume_definition']
    _identity_unique(out, key, 'normalized HMS coordinate-month definitions')
    require(set(out.year) == {2023, 2024, 2025} and set(out.alignment) == set(ALIGNMENTS)
            and set(out.plume_definition) == set(DEFINITIONS), 'normalized HMS source census incomplete')
    gate_api.assert_unchanged(gate)
    audit = {
        'status': 'PASS',
        'input_paths': {name: str(path) for name, path in paths.items()},
        'input_sha256': dict(digests),
        'metadata_status': {name: obj.get('status', obj.get('phase')) for name, obj in metadata.items()},
        'hms_2023_daily_path': str(paths['hms_station_day_2023.parquet']),
        'hms_2023_points_path': str(paths['station_points.parquet']),
        'hms_2023_monthly_path': str(paths['hms_station_month_2023.parquet']),
        'normalized_rows': int(len(out)),
        'normalized_rows_by_year': {str(y): int(n) for y, n in out.groupby('year').size().items()},
        'source_rows': {'2023_points': int(len(points23)), '2023_daily': int(len(day23)),
                        '2023_monthly': int(len(month23)), 'future_points': int(len(points_f)),
                        'future_station_month_points': int(len(months_f))},
        'source_validation': {'2023_daily_calendar_density_lag_availability_and_month_reconstruction': True,
                              'future': validation,
                              'future_monthly_all_counters_fractions_and_support_reconstructed': True,
                              'all_configured_pins_hashed_before_reads': True,
                              'target_or_model_columns_read': False},
        'normalization': {'alignments': list(ALIGNMENTS), 'plume_definitions': list(DEFINITIONS),
                          'source_point_id_preserved': True,
                          'coordinate_identity': 'source country/site/lat/lon/year/month; no nearest-point substitution',
                          '2023_unavailable_density_zero_masked_as_unknown': True,
                          'future_nullable_positive_vs_archive_unknown_preserved': True,
                          'lag_window_year_edges_unavailable': True},
    }
    return out, audit

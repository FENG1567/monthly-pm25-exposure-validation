# Monitoring and predictor source recipes

`analysis/original/input_sources/` preserves the recovered project source
recipes. These are original source snapshots, not a newly integrated
raw-to-result pipeline. They retain historical paths, imported helpers and
provider interfaces. Their source checksums and the recovered lineage scope
are recorded under `data/provenance/`.

The OpenAQ recipe selects stationary reference-flagged sources before
sensor averaging, normalizes recognized units, applies 24-hour/coverage
criteria, and requires at least 20 distinct valid days per month. Queensland
uses at least 18 valid hours in [−5, 2000) µg/m³ per day and 20 valid days per
month, retaining valid negative hours but excluding negative month means.
SINAICA uses source-valid nonnegative hours and quarantines stations 503/504
in 2022. MonitorAr uses VA status with documented invalid-flag/sentinel
exclusions and 18-hour/20-day completeness rules. These source flags are
not a uniform regulatory-certification classification.

The archived EPA preparation recipe has now reproduced the original AQS
raw-to-target stage for all eight years, 2018–2025. Each yearly ZIP matched
its source-ledger checksum. All 2,299,990 daily rows and 70,378 monthly rows
matched the saved target keys and compared numerical fields exactly. The
historical six-year and extended eight-year target generations remain
separately pinned. See [EPA target reconstruction](EPA_LINEAGE.md) for the
rules, explicit-input CLI and evidence. Canada retains all-published and
method-excluded target definitions.

Thirteen country inputs were recovered with exact final-input SHA-256 matches
and matched their source monthly PM values at every selected station-month.
The saved OpenAQ daily data reaggregated exactly to 14,494 saved monthly
values. Indian and Brazilian valid-day summaries came from their hash-pinned
selected target masters; final post-overlay feature files for those countries
were not reread in this check. USA/CAN historical parent population and six
meteorological fields matched the original grids exactly across 51,016 rows.
The historical AOD overlay's preserved-field contract links these parent
fields to the selected generation; this update did not refit the models.

ERA5-derived wind is the magnitude of monthly mean vector components, not
mean instantaneous speed. Precipitation is MODA mean daily accumulation
times 1,000 in mm/day, despite an old mm/month comment. Relative humidity
uses monthly mean temperature/dewpoint with Magnus coefficients 17.67 and
243.5, clipped to 1–100%; ventilation is the product of monthly fields.

Complete original raster-to-feature mappings for every historical country
are still incomplete. The original regional meteorology manifest is now preserved as a path-free
metadata extract in `data/provenance/regional_weather_manifest_extract.json`.
It records the definitions, sampling/fill rules and raw-source checksums; the
earlier helper source hash remains unresolved, and exact fire-download release mappings are not inferred from
script comments. The old FIRMS download script containing a literal
credential is not distributed. The credential was not used.

No raw observation or gridded third-party data are added to this package.
Code remains MIT; author-owned summaries remain CC BY 4.0. Source datasets
retain their own terms. See `THIRD_PARTY_DATA.md` and `ANNUAL_REPLAY.md`.

# Data inventory

`summary/table_registry.json` is the final English table content, with captions,
column names and rows: eight main blocks (including 4a/4b) and eight supplementary
tables. It fixes the final table sequence; legacy staging numbers are not used.

`summary/figure_registry.json` maps 14 figures to 35 saved panel datasets, records
their row counts and includes the full final captions and statistical meaning.
`summary/figure_panels/` contains those datasets. The large S7 grid is gzip CSV;
decompression restores the exact original bytes, including unavailable cells.
S7 is a model-derived geographic output, not monitoring observations or satellite
input. The renderer streams it without interpolating missing cells.

`summary/source_tables/` contains 20 aggregate result tables used to validate
reported numerical relationships. These are not individual station-month data.
`summary/source_manifest.json` records original and public-distribution checksums
and path-only redactions. `summary/data_dictionary.json` inventories each CSV's
fields; definitions of cohorts, estimands and intervals are in the captions.

`external_manifest/` contains official SatPM URLs/file identities and 404
historical input receipts across the feature, target and thinning manifests.
Receipts document identity; they do not distribute those input files or restore
missing upstream preparation rules. Repeated receipts may refer to the same file
in different analysis contracts. They are not 404 independent datasets.

Put separately acquired inputs in `data/external/`, which is ignored by Git.
Adapt the explicit configuration as described in `docs/ANALYSIS_REPLAY.md`.
Retain the required source/provider attribution and do not apply this
repository's data licence to third-party inputs.

Missing values remain missing. Negative R² is valid and retained. Some summary
files contain exploratory diagnostic or excluded/ineligible rows; use their
status fields and figure registry rather than treating every row as a primary
independent estimate. Summary replay does not regenerate confidence intervals.

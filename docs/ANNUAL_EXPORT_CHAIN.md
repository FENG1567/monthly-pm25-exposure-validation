# Portable annual export consistency package

This package contains the 1,056-row `annual_metrics.csv`, the 704-row
`annual_deltas.csv` distributed in `github_candidate_v2`, a compact file
catalog, and a standard-library Python CLI. After integration, run it from the
repository root:

```bash
python analysis/verify_annual_package.py
```

The CLI checks the included files against their SHA-256 values and column
catalog, checks row counts and key uniqueness, verifies the `qa_variant` and
`estimator_kind` aliases, checks that each saved point delta equals the
corresponding full-arm metric minus its comparator metric, and recomputes all
704 deterministic seed identifiers. It writes
`results/generated/annual_package_validation.json` and prints a one-line pass
record. The checked-in historical receipt is kept separately under
`data/provenance/annual_package_validation.json`. The CLI uses
only the Python standard library, rejects receipt paths that alias an input,
and has no server-specific paths.

## Scope and limitations

This is an export-table consistency check. It does not recover the original
canonical CSV byte streams, compare numerical values to raw station-year
inputs, rebuild annual metrics, or rerun any bootstrap. The recorded hashes of
the original canonical CSVs remain historical identities; their byte streams
are unavailable. The included catalog binds the distributed export bytes. The packaged `annual_deltas.csv` has the
`github_candidate_v2` release hash recorded in `catalog.json`. Its normalized
reference hash is recorded separately and is not substituted for the release
file hash. The 500-replicate confidence intervals are checked only for their
saved support/format constraints and are not recalculated.

The seed check uses the exact three-line `seed_for` algorithm in
`analysis/original/annual/compare_cams.py` from the candidate archive. The
algorithm prefixes the labels with `cams-matched-station-bootstrap-v1`, joins
them with NUL bytes, hashes with SHA-256, and converts the first eight hex
digits to an integer. Each annual seed uses these labels, in order:

```text
fair-totalpm-v1, alias, country, qa, profile, estimator, task,
target_strategy, backtransform, support_scope, V6GL03, comparator
```

This confirms the saved seed values, not a bootstrap replay.

## Provenance and licences

The included annual exports and their distributed SHA-256 values are recorded
in `data/provenance/catalog.json`; column definitions and the
`annual_metrics.csv` export hash come from
`work/submission_delivery_20261115/catalog.json`. The packaged
`annual_deltas.csv` comes from
`work/provenance_closure_20261125/github_candidate_v2/data/summary/source_tables/annual_deltas.csv`.
The candidate release manifest SHA-256 and normalized reference SHA-256 are
recorded separately. Export checks follow
`work/provenance_followup_20261125/verify_annual_export_chain.py`. The recorded
canonical CSV hashes and unavailable-byte-stream status are recorded
separately in `data/provenance/catalog.json`.

The seed helper and verifier code are MIT-licensed; see `LICENSE`. The
author-owned aggregate tables are licensed CC BY 4.0 under `LICENSE-DATA.md`.
Third-party input data are not included or relicensed. Attribute the aggregate
tables to Shuo Feng and contributors, retain the source attributions for any
derived reuse, and indicate changes.

The earlier `data/provenance/annual_export_chain_receipt.json` records the normalized export-stage check. Its `distributed_sha256` labels refer to those normalized tables. Current release bytes are bound by `data/provenance/catalog.json` and `data/provenance/annual_package_validation.json`; the normalized hash is not substituted for the current release hash.

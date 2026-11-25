# EPA target reconstruction

The archived `prepare_epa.py` recipe was applied to the eight original AQS
parameter-88101 yearly ZIPs for 2018–2025. The source ZIP checksums matched the
original input ledgers. Reconstructed site/date and site/year/month keys,
PM₂.₅ values, coordinates and monthly valid-day counts matched the saved
targets: 2,299,990 daily rows and 70,378 monthly rows, with zero maximum absolute
difference in every compared field. No model was fitted and no scientific
estimate or interval was replaced.

`data/provenance/epa_source_manifest.json` provides the original source names,
URLs, sizes and SHA-256 values, the recipe checksum and two saved target
generations. The 2018–2023 historical master and the 2018–2025 extended master
are distinct pinned files. The original six-year reconstruction and the
additional 2024–2025 reconstruction are combined in
`data/provenance/epa_lineage_combined_receipt.json`.

The first six years were checked against the historical master, and the last
two against the extended master. A separate comparison found identical keys
and compared fields for the overlapping six years of the two masters; see
`data/provenance/epa_generation_overlap_receipt.json`. The portable CLI below
was additionally executed for 2025, as recorded in
`data/provenance/epa_portable_cli_receipt.json`. The eight-year command is a
replay instruction, not a claim that this packaged CLI was executed for all
eight years in one run.

## Replay with explicit source files

Install the analysis requirements (including pandas, NumPy and PyArrow). From
the repository root, supply the original yearly ZIPs and the corresponding
saved daily/monthly Parquet generation:

```bash
python analysis/epa_replay.py \
  --raw-dir /path/to/aqs \
  --daily /path/to/station_day_2018_2025.parquet \
  --monthly /path/to/station_month_2018_2025.parquet \
  --years 2018 2019 2020 2021 2022 2023 2024 2025 \
  --out results/generated/epa_target_check.json
```

The paths are placeholders for explicitly supplied files. The ZIPs should use
the names in the manifest. The program rejects mismatched recipe, raw-source
or saved-target checksums. Select a subset with `--years` if desired. To check
the historical master, supply its 2018–2023 files and select years within that
range. This command does not download data or overwrite the inputs. Public
provider files may be revised; a new download with a different checksum must
not be represented as the original frozen input.

## Processing rules and scope

The recipe retains local-condition units and the two full-day durations
`24 HOUR` and `24-HR BLK AVG`, requires at least one observation and 75%
observation completeness, excludes method codes 236/238, and retains valid
dates, coordinates and concentrations in [0, 2000) µg m⁻³. It averages
same-status duplicates within site/date/POC/duration/method/standard keys,
prefers Included over Excluded over unspecified event status, and retains
Excluded-only records. It then averages within POC-days and across POCs to
form site-days. Monthly means require at least 20 distinct valid days;
coordinates are monthly medians.

This closes the original AQS cleaning-to-target reconstruction gap. It does
not recreate the final AOD overlays, all predictor products, model fits or
annual uncertainty intervals. The target files and original AQS observations
are external inputs, not bundled author-owned summaries. Their original terms
remain applicable. Project code is MIT; author-owned metadata and verification
summaries are CC BY 4.0.

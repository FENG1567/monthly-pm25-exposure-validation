# Reproducibility guide

This package distinguishes three levels of reproducibility. The included checkout supports level 1 directly. Levels 2 and 3 depend on external inputs and unresolved upstream provenance.

| Level | What it does | What it does not establish |
|---|---|---|
| **1. Summary replay** | Checks the package manifest and saved scientific summaries; exports the 17 reference tables and re-renders the 14 figures from supplied author-owned summary and panel data. | It does not refit a model, rerun a station bootstrap, recreate upstream monitoring targets, or independently reproduce the scientific analysis from raw measurements. |
| **2. External-input rerun** | Uses adapters to validate an explicit configuration and invoke the archived historical, Portugal, future-year, or LOCO runner with user-supplied inputs. | It does not fetch, license, infer, or silently substitute missing inputs. A successful run only reproduces the configured component with those inputs. |
| **3. Raw-to-result end-to-end rebuild** | Would rebuild the input targets and predictor products from their original raw sources and then rerun all analyses. | This level is incomplete: essential upstream collection/aggregation and product-generation mappings are not available for some inherited inputs. |

## Level 1: replay the supplied summaries

Use Python 3.11 or later and install the pinned/runtime requirements supplied with the repository:

```bash
python -m pip install -r requirements-replay.txt
PYTHONPATH=src python -m pm25_repro verify
PYTHONPATH=src python -m pm25_repro all
```

On PowerShell:

```powershell
python -m pip install -r requirements-replay.txt
$env:PYTHONPATH = "src"
python -m pm25_repro verify
python -m pm25_repro all
```

Run these commands from the repository root. `verify` checks the SHA-256 file manifest. `all` runs the manifest and scientific-summary checks, writes reference-table exports and figure files below `results/generated/`, and writes a reproduction report. The normal output directory is `results/generated`; it must remain inside `results/` and outside `results/reference/`.

The table export uses the included table registry/reference tables. Figure rendering uses the 35 published panel CSVs, including the complete 7,716,800-row compressed S7 panel. S7 is streamed in chunks by the renderer. Thus the replay checks and visualizes saved values; it does not calculate new estimates or uncertainty intervals. `results/reference/` holds the supplied reference artifacts and is not an output target.

## Level 2: run a component with supplied inputs

The adapter prints a validated command plan by default. Review it before execution:

```bash
PYTHONPATH=src python -m pm25_repro analysis --config configs/your-run.json --component historical
```

To launch that archived runner, add `--execute`:

```bash
PYTHONPATH=src python -m pm25_repro analysis --config configs/your-run.json --component historical --execute
```

Replace the placeholder config with a JSON file you prepare. It must declare `version: 1`, a repository-relative `output_root`, and a `components` object containing settings for the selected component. Input files must be explicitly named with paths inside the checkout; optional SHA-256 values are checked when provided. The adapter rejects absolute paths and path traversal, builds an argument vector without a shell, and directs outputs to a component subdirectory under the configured output root.

Supported adapter components are:

- `historical`: the archived historical validation runner with explicit feature/target inputs and settings.
- `portugal`: the archived Portugal validation runner with explicit inputs and settings.
- `future`: the archived external-year runner with explicit training and external features and the original runner arguments.
- `loco`: the archived leave-one-country-out runner with an explicit manifest.

The archive also includes station-thinning, multiplicity, monthly-comparison, and Portugal-tail diagnostic scripts. Generic raw-component configurations must preserve the original runner's argument list, explicitly identify existing input files, and supply one constrained `--out`; their presence does not mean their external inputs are included or that every run has been validated in this public package.

## Level 3: remaining upstream gaps

The package does not claim raw-to-result end-to-end reproducibility. Source recovery now provides original monitoring recipes, linked valid-day summaries for all 17 historical cohorts, original USA/CAN meteorology/population formulas, and the exact annual interval generator. See [Input sources](INPUT_SOURCES.md) and [Annual replay](ANNUAL_REPLAY.md).

The AQS raw-to-target cleaning stage for 2018–2025 has been reconstructed with exact numerical agreement; see [EPA target reconstruction](EPA_LINEAGE.md). This does not close the predictor and annual-interval input gaps.

The remaining gaps concern the complete original annual shared-support pair table, some historical countries' original raster-to-feature mappings, an earlier regional meteorology helper-code hash, and exact fire-download release mappings. Indian/Brazilian selected target masters survive, but their final post-overlay feature files were not numerically reread in this update. These limits prevent a claim of a newly completed raw-to-result rebuild.
Do not describe level 1 as a model/bootstrap replay, and do not describe levels 2 or 3 as end-to-end reproduction. Record the exact external input paths, versions, dates, licenses, and hashes for any rerun. The source and licensing caveats are summarized in [Third-party data](THIRD_PARTY_DATA.md); object-level paper outputs and archived code are listed in [Paper-to-code map](PAPER_CODE_MAP.md).

Annual export-table checks and their narrower scope are documented in [Annual export chain](ANNUAL_EXPORT_CHAIN.md). The original export snapshots in `analysis/original/export/` retain historical dependencies; they are not portable raw-to-result entry points.

# Analysis replay adapter

The adapter validates a repository-local configuration and builds an argument
vector for an archived analysis runner. It does not reimplement the model,
choose transformations from scores, or supply missing inputs. A plan is a
dry-run: it does not create the output directory or start a subprocess.

## Prepare a configuration

Copy `configs/analysis.template.json` to a working JSON file and replace every
`__REPLACE...__` value with a path that exists inside the repository. The
template is intentionally incomplete and is expected to fail until those
values are supplied. Use `sha256` entries to bind feature tables and manifests
to their exact bytes. `sha256` is optional for `feature_files` and auxiliary
input entries; the original LOCO runner also validates the hashes inside its
manifest.

All data and output paths must be repository-relative. Absolute paths and
`..` traversal are rejected. Raw CLI argument arrays may use `@repo/path` to
make a repository-relative input explicit. Output is constrained to a
component-specific directory beneath `output_root`; `output_root` must be
under `results/` and outside `results/reference/`.

The adapter also walks JSON input manifests through nested JSON references.
Known `path` and `*_path` values must be relative, resolve inside the checkout,
and exist. LOCO and thinning references resolve relative to the JSON file that
declares them. Multiplicity registry references resolve from its explicit
`--workspace`, matching that runner's CLI semantics. Keys such as `source_url`
remain metadata and are not treated as paths. `--flag=value` argument forms
are rejected so a later option cannot bypass path and output validation.

The supported configured components are `historical`, `portugal`, `future`,
`loco`, `multiplicity`, and `thinning`. Use the actual runner CLI when
preparing `arguments` arrays; the adapter preserves those arguments and only
resolves repository-local inputs and constrains the output location. It also
requires explicit `--workspace` and `--registry` arguments for multiplicity,
and an explicit `--manifest` for thinning, so those runners cannot silently
use their path defaults.

Example dry-run:

```bash
python -m pip install -e .
python -m pm25_repro analysis --config configs/analysis.json --component historical
```

The command prints JSON containing the resolved runner, working directory,
output directory, and exact argument vector. Missing configuration, runner,
or data inputs fail with CLI exit code 2 and a specific error. Add
`--execute` only when you intend to launch the original runner. Some original
runner argument arrays also have their own `--run` or `--execute` switches;
those are separate controls and remain absent by default.

## Runner snapshots and limits

Historical and Portugal use separate archived runner snapshots:

* `analysis/original/historical/validation/run_validation.py`
* `analysis/original/portugal/validation/run_validation.py`

The Portugal snapshot does not accept `--met-profile`; its `source_declared`
mode uses its own default, and `no_met` maps to its supported `--no-met`
option. The historical snapshot accepts `source_declared`, `era5_six`, and
`no_met`. Use `era5_six` only when supplied inputs contain all six finite ERA5
fields required by that runner; `source_declared` does not assert that data
from different sources are equivalent. The future-year component uses the
historical snapshot's `validation/run_external_year.py` runner.

LOCO runs from `analysis/original/loco/validation`. Keep its paired bootstrap
helper at `analysis/original/loco/fast_cluster_bootstrap.py`; the archived
runner resolves that root-level helper relative to its validation directory.
The adapter does not copy or relocate it.

The `monthly-comparison` and `pt-diagnostic` scripts have fixed paths and no
portable input/output CLI. The adapter refuses to plan or execute them until
they gain an explicit CLI. Passing ignored flags to those scripts would not
make their paths safe.

Station thinning also requires `validation_dir` in its component settings.
Set it to the exact `run_validation.py` and `validation_core.py` snapshot
bound by the thinning manifest. The plan exposes this as the original script's
`CAND1_FINAL_VALIDATION_DIR` environment override; it does not fall back to a
machine-specific workspace path.

This is not a full source-to-result workflow. The upstream source preprocessing
stages M02/M03 are not replayed by this adapter or generated from raw inputs in
this package. Historical, Portugal, and future replay therefore require their
already-prepared feature tables, plus any target/provenance inputs required by
the original runner. Record their source and checksums in the config. Do not
describe the available package as end-to-end reproduction from raw data.

## Self-check

`self_check(repo_root)` verifies the distributed files listed by
`analysis/original/SOURCE_SNAPSHOTS.json` against each entry's published
`sha256`, then runs `--self-check-only` on the historical and Portugal
runners. It does not assert that every packaged file is byte-identical to its
upstream copy: `original_sha256` is historical metadata, while `sha256` is the
expected hash for the current distribution. The returned object includes
`returncode` 0 on success and 2 on verification or runner failure.

Recovered original source recipes and their remaining dependencies are described in [Input sources](INPUT_SOURCES.md). A separate annual-pair replay CLI is documented in [Annual replay](ANNUAL_REPLAY.md). Neither changes the configured model adapters into a complete raw-source workflow.

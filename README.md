# Cross-country PM₂.₅ exposure validation: reproduction package

Code and author-owned summary results for the manuscript **“Limits of monthly PM₂.₅ exposure assignment across space and time and implications for gallbladder and bile duct disease research.”**

**Authors:** Shuo Feng, Chao Shi, Tianming Liu, Haoran Ding, Fenghe Hang, Zhenghao Huang, and Xianzhi Meng.

Repository: https://github.com/FENG1567/monthly-pm25-exposure-validation. The manuscript-associated software version is v0.1.0. No DOI has been assigned.

## What is included

- Archived model and auxiliary analysis source snapshots with distribution checksums, plus portable input adapters.
- Saved author-owned aggregate results: 21 source summary tables, 17 manuscript tables, and 35 figure-panel CSVs for 14 figures. Supplementary Figure S7 is supplied as a losslessly compressed CSV (about 19 MB) containing all 7,716,800 rows.
- Code to replay the supplied tables and render figures from those saved summaries.

The package validates PM₂.₅ exposure estimates. It does not include biliary or pancreatic disease outcome data or code for estimating disease associations.

## Quick start

From the repository root, install the replay dependencies and run the level-1 replay:

```bash
python -m pip install -r requirements-replay.txt
PYTHONPATH=src python -m pm25_repro all
```

On Windows PowerShell, set the source path first:

```powershell
python -m pip install -r requirements-replay.txt
$env:PYTHONPATH = "src"
python -m pm25_repro all
```

The command verifies the file checksums and bundled summary checks, rebuilds the reference tables, and renders the figures under `results/generated/`. It is a deterministic replay of supplied summaries, not a new model fit, station bootstrap, or independent validation. See [Reproducibility](docs/REPRODUCIBILITY.md) for the three levels of replay and their limits.

It also derives pooled conformal coverages/widths from 64 fold summaries,
recomputes 96 monthly point-metric differences, counts directions from 78 saved
simultaneous intervals, and selects the 24 final annual SatPM comparison rows.
These calculations use saved summaries; uncertainty resampling is not rerun.

## Re-running analyses with external inputs

Portable adapters are provided for historical validation, Portugal validation, future-year evaluation, and leave-one-country-out (LOCO) analysis. These require explicitly supplied input files and a JSON run configuration. First inspect the command plan, then add `--execute` to launch the archived runner:

```bash
PYTHONPATH=src python -m pm25_repro analysis --config configs/your-run.json --component historical
PYTHONPATH=src python -m pm25_repro analysis --config configs/your-run.json --component historical --execute
```

`configs/your-run.json` is a placeholder for a configuration you create for your inputs; the command does not imply that external data are bundled. Archived thinning, multiplicity, monthly-comparison, and Portugal-tail diagnostic scripts are preserved for provenance and have their own input requirements. See [Reproducibility](docs/REPRODUCIBILITY.md).

## Annual intervals and source recipes

The original annual bootstrap/seed generator and a portable annual-pair CLI are now included. See [Annual replay](docs/ANNUAL_REPLAY.md) and [Input sources](docs/INPUT_SOURCES.md) for inputs, verified scope and remaining raw-source requirements. These additions do not change the reported estimates or make level 1 a fresh interval replay.

## Additional source checks

The EPA raw-to-target stage has been checked against the original 2018–2025 ZIPs. A portable check requires explicitly supplied source and target files; see [EPA target reconstruction](docs/EPA_LINEAGE.md). The annual metrics table and [annual export checks](docs/ANNUAL_EXPORT_CHAIN.md) support numerical consistency checks for all 704 saved contrasts. These additions do not rerun model fitting or annual bootstrap intervals.

## Data and licensing

Original project code, including archived code snapshots and portable adapters, is licensed under MIT. The author-owned aggregate summaries and reference tables/figures are licensed under CC BY 4.0. Third-party source data retain their own terms and are not relicensed by either grant. No third-party raw datasets are included. See [LICENSE-DATA.md](LICENSE-DATA.md) and [Third-party data](docs/THIRD_PARTY_DATA.md).

## Citation

Please cite the associated manuscript using its final published bibliographic record when available, as well as this software where appropriate. The manuscript has not yet been published; use the final bibliographic record when available. The software version is recorded in CITATION.cff. Author names and software metadata are also recorded in [CITATION.cff](CITATION.cff).

Figure layout and export settings, including the Arial font requirement and 600 dpi TIFF output, are described in [Figure layout](docs/FIGURE_LAYOUT.md).

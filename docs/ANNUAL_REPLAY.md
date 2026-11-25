# Annual paired-comparison replay

The original annual metric generator and its exact bootstrap/seed sources are
archived in `analysis/original/annual/`. `analysis/annual_replay.py` provides a
portable entry point from explicitly supplied annual shared-support pairs.
It does not fit models or fetch external products.

## Method

Observed, model, raw CAMS and V6.GL.03 values are annualized with calendar-day
weights on the same finite station-month intersection. Primary support is
exactly 12 months; sensitivity support is at least 10. Metrics give equal
weight to eligible station-years. The paired bootstrap draws physical
stations with replacement and retains all eligible years of each draw.
There are 500 replicates, with 2.5th/97.5th percentile bounds. Contrasts are
full minus raw CAMS and full minus SatPM. Predictions remain fixed.

Each seed hashes this NUL-delimited label sequence:

```
fair-totalpm-v1, alias, country, qa, profile, estimator, task,
target_strategy, backtransform, support_scope, V6GL03, comparator
```

The prefix is `cams-matched-station-bootstrap-v1` followed by NUL. The seed is
the integer represented by the first eight SHA-256 hexadecimal digits.
`comparator` is `raw_CAMS` or `SatPM`; support labels are
`primary_exact_12_months` and `sensitivity_at_least_10_months`.

## Inputs and command

Supply CSV or Parquet with one row per model-cell/station/year and these columns:

```
alias, qa, profile, estimator, task, arm, target_strategy, backtransform,
country, site_id, year, shared_months, benchmark_version,
observed_annual_pm25, model_annual_pm25, raw_cams_annual_pm25,
satpm_annual_pm25
```

`benchmark_version` must be `V6GL03`. Rows with fewer than 10 shared months
remain excluded support. No eligible 2025 satellite rows can be supplied for
this study. The input file must be bound by its actual SHA-256, and output
must be a fresh directory:

```bash
python analysis/annual_replay.py --annual-pairs /path/annual_pairs.parquet \
  --input-sha256 ACTUAL_SHA256 --out results/annual-rerun
```

Input pairs must already establish the correct common monthly support and
source identity. The CLI cannot infer those facts from annual values.
Original archived drivers keep their historical paths and dependencies;
the portable entry point uses the hash-pinned bootstrap kernel directly.
Use the recorded original analysis runtime for numerical identity claims.

## Verification and limits

All 24 saved selected SatPM comparison seeds matched the original generator.
Tests check repeated station-years against direct row-based cluster draws,
support eligibility, duplicate rejection, missing values, unavailable 2025
support, and insufficient station counts. The shared-support kernel has its
own tests. Run the annual adapter tests from the repository root:

```bash
PYTHONPATH=analysis:analysis/original/annual python -m unittest discover \
  -s tests -p 'test_annual_replay.py'
```

The original complete `annual_shared_support.parquet` is not bundled or
currently recovered. Recovering the generator and matching saved seeds does
not constitute a fresh replay of the 24 reported confidence intervals.
Saved `annual_deltas.csv` remains the published summary; no scientific
estimate or interval was replaced in this update.

The original generator SHA-256 is
`b68b02ba0dc154594f0699c7aae5d4402ed4695c391be9d10f7e20c690182ae5`.
Code is MIT; author-owned aggregate outputs are CC BY 4.0. External pairs
and original third-party products retain their source terms.

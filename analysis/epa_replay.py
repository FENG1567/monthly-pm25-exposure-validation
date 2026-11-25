"""Replay the archived EPA recipe against explicit, hash-pinned inputs.

Original code: MIT. No downloads, model fitting, or changes to input files.
Run from the repository root; see docs/EPA_LINEAGE.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def compare(saved, replayed, keys, fields):
    left, right = saved.copy(), replayed.copy()
    for frame in (left, right):
        frame["site_id"] = frame.site_id.astype(str)
        if "date" in keys:
            frame["date"] = pd.to_datetime(frame.date)
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate target keys")
    joined = left[keys + fields].merge(
        right[keys + fields], on=keys, how="outer", validate="one_to_one",
        suffixes=("_saved", "_replay"), indicator=True,
    )
    shared = joined["_merge"].eq("both")
    differences, missing_equal = {}, {}
    for field in fields:
        x = joined.loc[shared, field + "_saved"].to_numpy(float)
        y = joined.loc[shared, field + "_replay"].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        missing_equal[field] = bool(np.array_equal(np.isfinite(x), np.isfinite(y)))
        differences[field] = float(np.max(np.abs(x[good] - y[good]))) if good.any() else None
    return {
        "passed": bool(shared.all() and all(missing_equal.values()) and
                       all(v is None or v <= 1e-10 for v in differences.values())),
        "saved_rows": len(left), "replayed_rows": len(right),
        "matched_rows": int(shared.sum()),
        "saved_only": int(joined["_merge"].eq("left_only").sum()),
        "replay_only": int(joined["_merge"].eq("right_only").sum()),
        "finite_masks_equal": missing_equal, "max_abs_difference": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/provenance/epa_source_manifest.json"))
    parser.add_argument("--recipe", type=Path,
                        default=Path("analysis/original/input_sources/prepare_epa.py"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inputs = [args.daily, args.monthly, args.manifest, args.recipe]
    inputs += [args.raw_dir / s["filename"] for s in manifest["raw_sources"]]
    for path in inputs:
        if (args.out.resolve() == path.resolve() or
                (args.out.exists() and path.exists() and args.out.samefile(path))):
            parser.error("Output must not replace an input")
    if sha256(args.recipe) != manifest["recipe_sha256"]:
        raise ValueError("Recipe checksum mismatch")
    daily_hash, monthly_hash = sha256(args.daily), sha256(args.monthly)
    generations = [g for g in manifest["target_generations"]
                   if g["daily_sha256"] == daily_hash and g["monthly_sha256"] == monthly_hash]
    if len(generations) != 1:
        raise ValueError("Daily/monthly input checksums do not match a saved target generation")
    generation = generations[0]
    if len(args.years) != len(set(args.years)) or not set(args.years) <= set(generation["years"]):
        raise ValueError("Years must be unique and belong to the saved target generation")
    daily, monthly = pd.read_parquet(args.daily), pd.read_parquet(args.monthly)
    spec = importlib.util.spec_from_file_location("archived_epa", args.recipe)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    details = []
    for year in sorted(args.years):
        source = next(s for s in manifest["raw_sources"] if s["year"] == year)
        raw = args.raw_dir / source["filename"]
        if args.out.resolve() == raw.resolve():
            raise ValueError("Output must not replace a raw input")
        if sha256(raw) != source["sha256"]:
            raise ValueError(f"AQS {year} checksum mismatch; do not substitute a revised download")
        print(f"EPA_YEAR_START {year}", flush=True)
        eligible, counts, legacy, methods = recipe.load_year(raw, year)
        del counts, legacy, methods
        replay_daily, events, selected_methods, included = recipe.new_daily(eligible)
        replay_monthly = recipe.monthly_from_daily(replay_daily)
        day_check = compare(daily.loc[daily.year.eq(year)], replay_daily,
                            ["site_id", "date"], ["pm25", "lat", "lon"])
        month_check = compare(monthly.loc[monthly.year.eq(year)], replay_monthly,
                              ["site_id", "year", "month"], ["pm25", "ndays", "lat", "lon"])
        details.append({"year": year, "raw_sha256": source["sha256"],
                        "daily": day_check, "monthly": month_check,
                        "passed": day_check["passed"] and month_check["passed"]})
        if sha256(raw) != source["sha256"]:
            raise ValueError("Raw input changed during replay")
        print("EPA_YEAR_CHECK " + json.dumps(details[-1]), flush=True)
        del eligible, replay_daily, replay_monthly, events, selected_methods, included
        gc.collect()
    passed = all(row["passed"] for row in details)
    result = {"status": "passed" if passed else "differences_found", "years": details,
              "saved_daily_sha256": daily_hash, "saved_monthly_sha256": monthly_hash,
              "recipe_sha256": sha256(args.recipe), "manifest_sha256": sha256(args.manifest),
              "cli_sha256": sha256(Path(__file__)),
              "model_fitting": False, "saved_source_files_modified": False,
              "pandas_version": pd.__version__, "numpy_version": np.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

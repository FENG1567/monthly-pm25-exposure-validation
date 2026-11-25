"""Render the repository's author-owned summary table registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _table_name(table_id: str) -> str:
    """Return a safe, stable file stem for a registry table identifier."""
    if not table_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in table_id):
        raise ValueError(f"Unsafe table id: {table_id!r}")
    return f"Table_{table_id}"


def _table_groups(registry: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups = []
    for key in ("main_tables", "supplementary_tables"):
        value = registry.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Table registry must contain a {key} array")
        groups.append((key, value))
    return groups


def _validate_table(table: dict[str, Any], group: str) -> tuple[str, list[str], list[list[Any]]]:
    if not isinstance(table, dict):
        raise ValueError(f"Invalid table entry in {group}")
    table_id = table.get("id")
    if not isinstance(table_id, str):
        raise ValueError(f"Table id in {group} must be a string")
    _table_name(table_id)
    caption = table.get("caption")
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(caption, str) or not isinstance(columns, list) or not columns:
        raise ValueError(f"Table {table_id} must have a caption and nonempty columns")
    if not all(isinstance(column, str) for column in columns):
        raise ValueError(f"Table {table_id} has a non-string column name")
    if not isinstance(rows, list):
        raise ValueError(f"Table {table_id} rows must be an array")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError(f"Table {table_id} row {index} does not match its {len(columns)} columns")
    return table_id, columns, rows


def _csv_bytes(columns: list[str], rows: list[list[Any]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _markdown(caption: str, columns: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    lines = [caption, "", "| " + " | ".join(cell(column) for column in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def build_tables(repo_root: str | Path, outdir: str | Path) -> dict[str, Any]:
    """Write one CSV and one Markdown file for each registered table.

    CSV bytes are checked against ``results/reference/tables`` when the
    reference directory is present. The registry remains the source of the
    rendered values; this function does not calculate or alter table data.
    """
    root = Path(repo_root).resolve()
    registry_path = root / "data" / "summary" / "table_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing table registry: {registry_path}") from exc
    if not isinstance(registry, dict):
        raise ValueError("Table registry must contain a JSON object")

    output = Path(outdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for group, tables in _table_groups(registry):
        for table in tables:
            table_id, columns, rows = _validate_table(table, group)
            stem = _table_name(table_id)
            if stem in seen:
                raise ValueError(f"Duplicate table id: {table_id}")
            seen.add(stem)
            csv_path = output / f"{stem}.csv"
            md_path = output / f"{stem}.md"
            csv_payload = _csv_bytes(columns, rows)
            csv_path.write_bytes(csv_payload)
            md_path.write_text(_markdown(table["caption"], columns, rows), encoding="utf-8", newline="")
            records.append({"id": table_id, "group": group, "csv": csv_path.name, "markdown": md_path.name})

    reference_dir = root / "results" / "reference" / "tables"
    reference_check: dict[str, Any]
    if reference_dir.is_dir():
        expected = {f"{stem}.csv" for stem in seen}
        observed = {path.name for path in reference_dir.glob("Table_*.csv")}
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        mismatches = [
            name for name in sorted(expected & observed)
            if (reference_dir / name).read_bytes() != (output / name).read_bytes()
        ]
        reference_check = {
            "status": "passed" if not (missing or extra or mismatches) else "failed",
            "matched": len(expected & observed) - len(mismatches),
            "expected": len(expected), "missing": missing, "extra": extra,
            "content_mismatches": mismatches,
        }
        if reference_check["status"] != "passed":
            raise ValueError(f"Table reference mismatch: {reference_check}")
    else:
        reference_check = {"status": "not_available", "expected": len(records)}

    return {
        "status": "passed", "tables_generated": len(records),
        "csv_files": len(records), "markdown_files": len(records),
        "reference_check": reference_check, "tables": records,
    }

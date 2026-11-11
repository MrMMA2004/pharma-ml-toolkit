"""Explain strict example-data byte drift without changing data or accepting drift.

This runs the original builder in its non-writing --check mode and compares
in-memory generated CSV bytes with the files on disk. Numeric differences are
reported descriptively, with no tolerance and no pass/fail substitution for the
strict builder. PDB content and non-CSV differences remain byte-level only.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import build_example_data as builder
import numpy as np


def compare_csv(actual: bytes, generated: bytes) -> dict[str, Any]:
    """Describe differing cells; never declare unequal CSVs equivalent."""
    try:
        old = list(csv.reader(io.StringIO(actual.decode("utf-8"), newline=""), strict=True))
        new = list(csv.reader(io.StringIO(generated.decode("utf-8"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        return {"parse_error": str(exc)}
    result: dict[str, Any] = {
        "same_shape": len(old) == len(new) and all(len(a) == len(b) for a, b in zip(old, new)),
        "same_header": bool(old and new and old[0] == new[0]),
        "numeric_cells_with_changed_text": 0,
        "non_numeric_cells_with_changed_text": 0,
        "numeric_nonfinite_or_overflow": 0,
        "max_absolute_numeric_difference": 0.0,
    }
    for left, right in zip(old[1:], new[1:]):
        for a, b in zip(left, right):
            if a == b:
                continue
            try:
                x, y = float(a), float(b)
            except ValueError:
                result["non_numeric_cells_with_changed_text"] += 1
                continue
            result["numeric_cells_with_changed_text"] += 1
            difference = abs(x - y)
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(difference):
                result["numeric_nonfinite_or_overflow"] += 1
            else:
                result["max_absolute_numeric_difference"] = max(
                    result["max_absolute_numeric_difference"], difference
                )
    return result


def diagnose() -> dict[str, Any]:
    """Run the strict builder read-only and describe, but do not waive, drift."""
    captured: dict[Path, bytes] = {}
    original = builder._write_if_changed

    def capture(path: Path, data: bytes) -> None:
        captured[path] = data
        original(path, data)

    builder._write_if_changed = capture
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            try:
                builder.main(["--check"])
                exit_code = 0
            except SystemExit as exc:
                if exc.code != 1 or not builder.STATS["drift"]:
                    raise
                exit_code = 1
    finally:
        builder._write_if_changed = original
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_runtime()
    comparisons: list[dict[str, Any]] = []
    for path, generated in sorted(captured.items()):
        if not path.is_file():
            comparisons.append({"path": str(path), "status": "missing"})
            continue
        actual = path.read_bytes()
        if actual == generated:
            continue
        record: dict[str, Any] = {"path": str(path), "status": "byte drift"}
        if path.suffix.lower() == ".csv":
            record["csv_comparison"] = compare_csv(actual, generated)
        comparisons.append(record)
    return {
        "scope": "Diagnostic only; numeric descriptions do not waive strict byte drift",
        "strict_check_exit_code": exit_code,
        "strict_drift": list(builder.STATS["drift"]),
        "generated_files_with_byte_drift": comparisons,
        "numpy_runtime": runtime.getvalue(),
        "builder_output": output.getvalue(),
    }


def main() -> int:
    print(json.dumps(diagnose(), indent=2, allow_nan=False))
    # This tool is diagnostic: preserve the strict check's nonzero exit status.
    return 0 if not builder.STATS["drift"] else 1


if __name__ == "__main__":
    sys.exit(main())

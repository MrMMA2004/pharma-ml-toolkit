"""Isolated, read-only access to checkout-only reproducibility diagnostics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pharmaml.exceptions import DataValidationError


def run_example_drift_diagnostic(script: Path | str, *, timeout_s: float = 120) -> dict[str, Any]:
    """Run the checkout's existing strict drift diagnostic in a child process.

    Its exit 1 is a strict byte mismatch, not an execution error or tolerance
    waiver. Other exit codes, invalid JSON and malformed/mismatched outcomes
    fail closed. The caller should pass the trusted toolkit script, not an
    arbitrary downloaded program. This does not alter files or accept drift;
    it assumes the checkout and generated files are not being concurrently
    modified. It cannot establish numerical or scientific equivalence.
    """
    path = Path(script)
    if not path.is_file():
        raise DataValidationError("example-data diagnostic script is unavailable in this installation")
    try:
        completed = subprocess.run([sys.executable, str(path.resolve())], cwd=path.resolve().parent.parent,
                                   capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DataValidationError(f"example-data diagnostic could not complete: {exc}") from exc
    if completed.returncode not in (0, 1):
        raise DataValidationError(f"example-data diagnostic failed (exit {completed.returncode}): "
                                  f"{completed.stderr.strip()[:1000]}")
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("example-data diagnostic returned invalid JSON") from exc
    if (not isinstance(result, dict) or type(result.get("strict_check_exit_code")) is not int
            or result["strict_check_exit_code"] != completed.returncode
            or not isinstance(result.get("strict_drift"), list)
            or bool(result["strict_drift"]) != bool(completed.returncode)):
        raise DataValidationError("example-data diagnostic returned an inconsistent outcome")
    return result

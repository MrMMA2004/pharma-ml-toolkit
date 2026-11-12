"""Read-only inspection of existing registered checkpoint envelopes."""
from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pharmaml.utils.checkpoint_schema import (
    OPERATION_REGISTRY,
    implementation_fingerprint,
    input_fingerprint,
    validate_checkpoint,
)


def inspect_checkpoint(path: Path | str, *, inputs: Sequence[Path | str] = ()) -> dict[str, Any]:
    """Validate envelope and compare supplied inputs and registered implementation.

    No job is resumed. Unknown/missing inputs remain unchecked, and a match is
    not a complete resume-safety guarantee. Hashing assumes stable files.
    """
    report = validate_checkpoint(path)
    result: dict[str, Any] = {"valid_envelope": report.ok, "legacy": report.legacy,
                              "errors": list(report.errors), "input_status": "not checked",
                              "implementation_status": "not checked",
                              "notice": "Read-only inspection, not a resume or a safety guarantee."}
    if not report.ok:
        return result
    envelope = report.envelope or {}
    op_type = envelope["operation"]["type"]
    result.update({"operation_type": op_type, "resume_unit": envelope["resume_unit"]})
    recorded = envelope.get("inputs")
    if recorded is None:
        result["input_status"] = "no inputs recorded; cannot verify"
    elif inputs:
        current = input_fingerprint(inputs)
        result["input_status"] = ("match" if current["fingerprint"] == recorded["fingerprint"]
                                  else "changed; do not resume without investigation")
    module = importlib.import_module(OPERATION_REGISTRY[op_type].implementation)
    digest = implementation_fingerprint(module)
    result["implementation_status"] = ("match" if digest == envelope["implementation_fingerprint"]
                                        else "changed; do not resume without investigation")
    return result

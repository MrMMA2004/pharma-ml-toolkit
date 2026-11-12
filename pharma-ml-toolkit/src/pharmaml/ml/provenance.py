"""Read-only, user-declared experiment fingerprints and environment metadata.

A fingerprint documents file bytes, not the data actually used by a model.
These records are not signed, and differences do not prove the cause of drift.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

from pharmaml import __version__
from pharmaml.exceptions import DataValidationError


def environment_snapshot() -> dict[str, Any]:
    """Return observed versions and selected numerical-thread settings; change nothing."""
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "rdkit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    settings = ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES")
    return {"toolkit": __version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "numeric_environment": {k: os.environ.get(k) for k in settings}}


def experiment_manifest(data: Path | str, *, parameters: dict[str, str],
                        split_strategy: str) -> dict[str, Any]:
    """Hash a supplied file and record user-declared setup, without executing a run."""
    if not split_strategy.strip():
        raise DataValidationError("split strategy must not be blank")
    path = Path(data)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise DataValidationError(f"cannot read dataset: {exc}") from exc
    return {"schema": "pharmaml-experiment-manifest/1", "dataset": {"name": path.name,
            "sha256": digest.hexdigest(), "size_bytes": size}, "parameters": dict(parameters),
            "split_strategy": split_strategy, "environment": environment_snapshot(),
            "notice": "User-declared metadata, not proof of what was actually trained or evaluated."}


def compare_manifests(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare recorded dataset bytes, declared setup and captured environment."""
    for record in (first, second):
        if (not isinstance(record, dict) or record.get("schema") != "pharmaml-experiment-manifest/1"
                or not isinstance(record.get("dataset"), dict)
                or not isinstance(record["dataset"].get("sha256"), str)
                or not isinstance(record.get("parameters"), dict)
                or not isinstance(record.get("split_strategy"), str)
                or not isinstance(record.get("environment"), dict)):
            raise DataValidationError("invalid or unsupported experiment manifest")
    return {"same_dataset_sha256": first["dataset"]["sha256"] == second["dataset"]["sha256"],
            "same_parameters": first["parameters"] == second["parameters"],
            "same_split_strategy": first["split_strategy"] == second["split_strategy"],
            "same_environment": first["environment"] == second["environment"],
            "notice": "Differences are descriptive; matching records do not prove reproducibility."}

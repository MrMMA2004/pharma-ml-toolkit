"""Opt-in observed CLI training records and read-only record comparison.

Records are unsigned, not a secure audit trail or a reproducibility guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.provenance import environment_snapshot, fingerprint_file


def _json_value(value: Any, *, depth: int = 0) -> Any:
    """Convert parameter values without unsafe deserialization or arbitrary repr.

    Unsupported objects and excessive nesting are tagged, not silently stringified.
    Nested estimator parameters are captured to a bounded depth.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), depth=depth)
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if depth >= 8:
        return {"omitted_type": kind, "reason": "maximum depth"}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _json_value(v, depth=depth + 1) for k, v in value.items()}
        return {"typed_mapping": [{"key": _json_value(k, depth=depth + 1),
                                   "value": _json_value(v, depth=depth + 1)} for k, v in value.items()]}
    if hasattr(value, "get_params") and callable(value.get_params):
        return {"estimator_type": kind,
                "parameters": _json_value(value.get_params(deep=False), depth=depth + 1)}
    return {"omitted_type": kind, "reason": "unsupported parameter value"}


def _index_hash(index: Any) -> str:
    # load_dataset produces a reset range index for the cleaned table. Hash the
    # ordered row positions, not user IDs or the input file's original line numbers.
    values = [int(v) for v in index]
    payload = json.dumps(values, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def prepare_record_path(path: str | Path) -> Path:
    """Check a new JSON path before training; the later exclusive open handles races."""
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise DataValidationError("training run record must end in .json")
    if os.path.lexists(target):
        raise DataValidationError("training run record must be a new, non-existing file")
    if not target.parent.is_dir():
        raise DataValidationError("training run record parent directory must already exist")
    return target


def make_training_record(
    *, dataset: Any, dataset_fingerprint: dict[str, Any], model_path: str | Path,
    result: Any, report: Any, requested_model: str, requested_task: str,
    keep_categorical: bool, test_size: float, cv: int, tune_requested: bool,
    stratified: bool, X_train: Any, X_test: Any,
) -> dict[str, Any]:
    """Capture actual training output and ordered split hashes after a CLI run."""
    from pharmaml.ml.models import MODEL_REGISTRY
    from pharmaml.ml.training import _select_scorer

    tuned = result.cv_scores is not None
    model = result.model
    record = {
        "schema": "pharmaml-cli-training-run/1",
        "dataset": dataset_fingerprint,
        "input_processing": {"target": dataset.target, "task": dataset.task,
                             "requested_task": requested_task, "features": list(dataset.feature_names),
                             "dropped_columns": list(dataset.dropped_columns),
                             "categorical_columns": list(dataset.categorical_columns),
                             "keep_categorical": keep_categorical, "cleaned_rows": dataset.n_samples},
        "split": {"strategy": "row_shuffle", "stratified": stratified, "test_size": test_size,
                  "seed": 42, "train_rows": len(X_train), "test_rows": len(X_test),
                  "train_ordered_clean_row_indices_sha256": _index_hash(X_train.index),
                  "test_ordered_clean_row_indices_sha256": _index_hash(X_test.index),
                  "index_note": "zero-based positions after missing-target rows are dropped, not source line numbers"},
        "training": {"requested_model": requested_model, "resolved_model": result.name,
                     "tune_requested": tune_requested, "grid_search_performed": tuned,
                     "cv_folds": cv if tuned else None,
                     "cv_strategy": ("sklearn classifier stratified k-fold (no shuffle)"
                                     if dataset.task == "classification" else "sklearn k-fold (no shuffle)")
                     if tuned else None,
                     "cv_scorer": _select_scorer(model, dataset.task) if tuned else None,
                     "cv_n_jobs": -1 if tuned else None,
                     "registry_grid": _json_value(MODEL_REGISTRY[result.name].param_grid) if tuned else None,
                     "selected_parameters": _json_value(result.best_params),
                     "fitted_pipeline": _json_value(model),
                     "cv_scores": _json_value(result.cv_scores.tolist()) if tuned else None},
        "held_out_metrics": _json_value(report.as_dict()),
        "model_artifact": fingerprint_file(model_path),
        "environment": environment_snapshot(),
        "notice": ("Observed CLI run metadata is unsigned. File hash assumes stable reads; parameter objects may "
                   "be omitted or depth-limited. Split hashes do not reveal row IDs or prove independence. "
                   "Matching records do not establish scientific validity or reproducibility. Existing model "
                   "save behavior may overwrite an artifact; this record alone is new-file-only."),
    }
    return record


def write_new_record(path: Path, record: dict[str, Any]) -> None:
    """Exclusive write, no replacement. Errors can leave a partial new file."""
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        raise DataValidationError(f"could not write new training run record: {exc}") from exc


def compare_training_records(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare unsigned JSON records only; never unpickle a model."""
    for record in (first, second):
        if (not isinstance(record, dict) or record.get("schema") != "pharmaml-cli-training-run/1"
                or any(not isinstance(record.get(k), dict)
                       for k in ("dataset", "input_processing", "split", "training", "model_artifact", "environment"))
                or not isinstance(record["dataset"].get("sha256"), str)
                or not isinstance(record["model_artifact"].get("sha256"), str)
                or not isinstance(record["split"].get("train_ordered_clean_row_indices_sha256"), str)
                or not isinstance(record["split"].get("test_ordered_clean_row_indices_sha256"), str)):
            raise DataValidationError("invalid or unsupported training run record")
    return {"same_dataset_sha256": first["dataset"]["sha256"] == second["dataset"]["sha256"],
            "same_input_processing": first["input_processing"] == second["input_processing"],
            "same_split": first["split"] == second["split"],
            "same_training": first["training"] == second["training"],
            "same_model_artifact_sha256": first["model_artifact"]["sha256"] == second["model_artifact"]["sha256"],
            "same_environment": first["environment"] == second["environment"],
            "source_status_after_run": {"first": first.get("source_status_after_run"),
                                        "second": second.get("source_status_after_run")},
            "held_out_metrics": {"first": first.get("held_out_metrics"), "second": second.get("held_out_metrics")},
            "notice": "Unsigned records; matching metadata is not proof of reproducibility, independence or validity."}

"""Held-out binary probability recalibration with three disjoint partitions."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.heldout_diagnostics import classification_probability_diagnostics
from pharmaml.ml.preprocessing import build_preprocessor
from pharmaml.ml.uncertainty_workbench import _validate_partitions


def binary_probability_calibration_report(
    train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, *,
    features: list[str], target: str, key: str, positive_label: str,
) -> dict[str, Any]:
    """Fit base model on train, sigmoid on independent calibration, evaluate on test.

    A fixed L2-regularized logistic calibrator (C=1) is trained on base-model
    decision scores. Nothing is selected using test data. Exact leakage-unit
    overlap is rejected, not independence guaranteed. This is NOT conformal
    calibration; any conformal procedure requires a separate independent set.
    No final model or per-row predictions are returned or persisted.
    """
    overlap = _validate_partitions(train, calibration, test, features=features, target=target,
                                   key=key, group=None, classification=True)
    classes = sorted(set(train[target]))
    if len(classes) != 2:
        raise DataValidationError("binary recalibration requires exactly two training classes")
    if not isinstance(positive_label, str) or positive_label not in classes:
        raise DataValidationError("positive label must exactly match a training class")
    if set(calibration[target]) != set(classes):
        raise DataValidationError("probability calibration requires both training classes")
    if not set(test[target]).issubset(classes):
        raise DataValidationError("test contains labels absent from training classes")
    negative_label = next(c for c in classes if c != positive_label)
    base = Pipeline([("prep", build_preprocessor()),
                     ("estimator", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000,
                                                      tol=1e-4, random_state=0))])
    calibrator = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, tol=1e-4, random_state=0)
    # sklearn's binary decision_function refers to the second sorted class.
    direction = 1 if positive_label == classes[1] else -1
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            base.fit(train[features], train[target])
            calibration_scores = direction * base.decision_function(calibration[features])
            calibrator.fit(np.asarray(calibration_scores).reshape(-1, 1),
                           (calibration[target] == positive_label).astype(int))
        except ConvergenceWarning as exc:
            raise DataValidationError("logistic model or recalibrator did not converge") from exc
    test_scores = direction * base.decision_function(test[features])
    base_positive = base.predict_proba(test[features])[:, classes.index(positive_label)]
    calibrated_positive = calibrator.predict_proba(np.asarray(test_scores).reshape(-1, 1))[:, 1]
    truth = (test[target] == positive_label).to_numpy(dtype=int)

    def summarize(p: np.ndarray) -> dict[str, Any]:
        diagnostics = classification_probability_diagnostics(
            np.column_stack((1 - p, p)), test[target].to_numpy(), [negative_label, positive_label])
        return {"binary_brier": float(brier_score_loss(truth, p)),
                "log_loss": float(log_loss(truth, np.column_stack((1 - p, p)), labels=[0, 1])),
                "reliability": diagnostics}

    return {"task": "binary_probability_recalibration", "positive_label": positive_label,
            "negative_label": negative_label, "features": list(features), "target": target, "key": key,
            "n_train": len(train), "n_probability_calibration": len(calibration), "n_test": len(test),
            "partition_class_counts": {name: {c: int((frame[target] == c).sum()) for c in classes}
                                       for name, frame in (("train", train), ("probability_calibration", calibration),
                                                           ("test", test))},
            "duplicate_keys_by_partition": dict(zip(overlap.partition_names, overlap.duplicate_rows, strict=True)),
            "base_model": "logistic regression with training-only median imputation and scaling",
            "calibrator": "L2-regularized logistic sigmoid on independently calibrated decision scores",
            "uncalibrated": summarize(base_positive), "recalibrated": summarize(calibrated_positive),
            "limitations": ("This is not conformal calibration. A subsequent conformal workflow needs another "
                            "independent calibration partition and separate evaluation. Exact-key checks cannot "
                            "prove independence or rule out related compounds. Scores and reliability bins are "
                            "descriptive and may be unstable for small or dependent samples; do not tune on the "
                            "test set. Recalibration can worsen predictions and does not guarantee future calibration. "
                            "This fits a new model; no fitted model is saved. No subgroup guarantees are implied.")}


def binary_probability_calibration_from_files(
    train: str | Path, calibration: str | Path, test: str | Path, *,
    features: list[str], target: str, key: str, positive_label: str,
) -> dict[str, Any]:
    """Read literal ID/label text and fingerprint each input around parsing."""
    from pharmaml.ml.provenance import environment_snapshot, fingerprint_file
    from pharmaml.utils.identifier_table import read_identifier_table

    paths = {"train": Path(train), "probability_calibration": Path(calibration), "test": Path(test)}
    frames, fingerprints = [], {}
    for name, path in paths.items():
        before = fingerprint_file(path)
        frame = read_identifier_table(path, None)
        after = fingerprint_file(path)
        if before != after:
            raise DataValidationError(f"{name} input changed while reading; retry with stable files")
        fingerprints[name] = after
        for column in features:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
        frames.append(frame)
    report = binary_probability_calibration_report(*frames, features=features, target=target,
                                                   key=key, positive_label=positive_label)
    report["provenance"] = {"schema": "pharmaml-binary-probability-recalibration/1", "inputs": fingerprints,
                            "parameters": {"features": list(features), "target": target, "key": key,
                                           "positive_label": positive_label, "base_model_C": 1.0,
                                           "calibrator_C": 1.0, "solver": "lbfgs", "max_iter": 2000,
                                           "tol": 1e-4, "random_state": 0,
                                           "preprocessing": "median imputation and standard scaling",
                                           "partitions": "user-supplied train/probability-calibration/test"},
                            "environment": environment_snapshot(),
                            "notice": "Input hashes assume stable files; unsigned records do not prove independence."}
    return report

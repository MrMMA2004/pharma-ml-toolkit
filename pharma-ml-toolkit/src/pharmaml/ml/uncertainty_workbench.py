"""Explicit held-out regression and classification uncertainty workflows.

Separate files and disjoint exact IDs are necessary checks, not proof of
independence, exchangeability or absence of chemical similarity leakage.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.conformal import calibrate_split_conformal
from pharmaml.ml.conformal_classification import calibrate_split_conformal_classifier
from pharmaml.ml.coverage_diagnostics import coverage_by_group
from pharmaml.ml.heldout_diagnostics import (
    classification_probability_diagnostics,
    distance_domain_diagnostics,
)
from pharmaml.ml.preprocessing import build_preprocessor
from pharmaml.ml.split_audit import audit_split_overlap


def _validate_partitions(
    train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, *,
    features: list[str], target: str, key: str, group: str | None, classification: bool,
):
    """Fail closed for obvious label/key/feature leakage before either workflow."""
    if not features or len(set(features)) != len(features):
        raise DataValidationError("features must be nonempty and unique")
    reserved = {target, key} | ({group} if group is not None else set())
    if not target or not key or target == key or set(features) & reserved:
        raise DataValidationError("features must exclude target, key and subgroup columns")
    if group is not None and group in {target, key}:
        raise DataValidationError("subgroup must differ from target and key")
    partitions = {"train": train, "calibration": calibration, "test": test}
    for name, frame in partitions.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise DataValidationError(f"{name} must be a nonempty table with unique columns")
        needed = [*features, target, key] + ([group] if group and name == "test" else [])
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise DataValidationError(f"{name} is missing columns: {', '.join(missing)}")
        numeric_columns = features if classification else [*features, target]
        for column in numeric_columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise DataValidationError(f"{name} column {column!r} must be numeric")
            values = frame[column].to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any() or (column == target and np.isnan(values).any()):
                raise DataValidationError(f"{name} column {column!r} has invalid values")
        if classification and any(not isinstance(v, str) or not v.strip() for v in frame[target]):
            raise DataValidationError(f"{name} target must contain nonempty string labels")
        if group and name == "test" and any(pd.isna(v) or isinstance(v, str) and not v.strip()
                                            for v in frame[group]):
            raise DataValidationError("test subgroup labels cannot be missing or blank")
        if frame[features].isna().all().any():
            raise DataValidationError(f"{name} contains an entirely missing feature")
    overlap = audit_split_overlap({name: frame[key].tolist() for name, frame in partitions.items()})
    if overlap.has_overlap:
        raise DataValidationError("exact key overlap between train, calibration and test; refuse evaluation")
    return overlap


def regression_uncertainty_report(
    train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, *,
    features: list[str], target: str, key: str, group: str | None = None,
    alpha: float = 0.1, diagnostics: bool = False,
) -> dict[str, Any]:
    """Fit ridge and preprocessing on training only; calibrate and evaluate separately.

    Exact cross-partition key overlap is rejected. Features are explicitly
    listed numeric columns; target, key and group are excluded to reduce obvious
    leakage. The caller is responsible for selecting independent partitions,
    features and prespecified subgroups. Results are empirical; marginal
    conformal coverage assumes exchangeability, not subgroup guarantees.
    """
    overlap = _validate_partitions(train, calibration, test, features=features, target=target,
                                   key=key, group=group, classification=False)
    model = Pipeline([("prep", build_preprocessor()), ("estimator", Ridge())])
    model.fit(train[features], train[target])
    calibrated = calibrate_split_conformal(model, calibration[features], calibration[target], alpha=alpha)
    result = calibrated.predict_interval(test[features], test[target])
    subgroup = None
    if group is not None:
        subgroup = coverage_by_group(result.covered, test[group].to_numpy(),
                                     sizes=result.upper - result.lower)
    report: dict[str, Any] = {
        "task": "regression", "model": "ridge (median imputation and scaling fitted on train only)",
        "features": list(features), "target": target, "key": key, "group": group,
        "alpha": alpha, "n_train": len(train), "n_calibration": len(calibration),
        "n_test": len(test), "radius": result.radius, "mean_width": result.mean_width,
        "coverage": result.coverage, "n_covered": result.n_covered,
        "duplicate_keys_by_partition": dict(zip(overlap.partition_names, overlap.duplicate_rows, strict=True)),
        "subgroups": [asdict(item) for item in subgroup.groups] if subgroup else None,
        "limitations": ("Exact key disjointness does not prove independent partitions or absence "
                        "of related compounds. Marginal coverage assumes exchangeability; subgroup "
                        "coverage is descriptive, with no conditional guarantee. Wilson intervals "
                        "assume independent observations with common within-group coverage."),
    }
    if diagnostics:
        report["applicability"] = distance_domain_diagnostics(model, train, test, features, covered=result.covered)
    return report


def classification_uncertainty_report(
    train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, *,
    features: list[str], target: str, key: str, group: str | None = None,
    alpha: float = 0.1, diagnostics: bool = False,
) -> dict[str, Any]:
    """Fit logistic regression on train, then calibrate sets and evaluate separately.

    Target values must be nonempty strings. All calibration and test labels
    must be present in the fitted training class space. For insufficient
    calibration sample size the underlying conformal method returns full sets.
    Class and subgroup metrics are descriptive, not conditional guarantees.
    """
    overlap = _validate_partitions(train, calibration, test, features=features, target=target,
                                   key=key, group=group, classification=True)
    train_labels = set(train[target])
    if len(train_labels) < 2:
        raise DataValidationError("training requires at least two distinct classes")
    for name, frame in (("calibration", calibration), ("test", test)):
        if not set(frame[target]).issubset(train_labels):
            raise DataValidationError(f"{name} contains labels absent from training classes")
    model = Pipeline([("prep", build_preprocessor()),
                      ("estimator", LogisticRegression(max_iter=1000, random_state=0))])
    model.fit(train[features], train[target])
    calibrated = calibrate_split_conformal_classifier(model, calibration[features],
                                                       calibration[target], alpha=alpha)
    result = calibrated.predict_set(test[features], test[target])
    sizes = result.set_sizes
    classes = coverage_by_group(result.covered, test[target].to_numpy(), sizes=sizes)
    subgroup = (coverage_by_group(result.covered, test[group].to_numpy(), sizes=sizes)
                if group is not None else None)
    report: dict[str, Any] = {
        "task": "classification", "model": "logistic regression (train-only imputation and scaling)",
        "features": list(features), "target": target, "key": key, "group": group,
        "alpha": alpha, "n_train": len(train), "n_calibration": len(calibration), "n_test": len(test),
        "classes": list(result.classes), "threshold": result.threshold, "calibration_rank": calibrated.rank,
        "full_set_fallback": calibrated.rank > len(calibration),
        "coverage": result.coverage, "n_covered": result.n_covered, "mean_set_size": result.mean_set_size,
        "n_empty_sets": int((sizes == 0).sum()), "n_singleton_sets": int((sizes == 1).sum()),
        "n_multiple_sets": int((sizes > 1).sum()),
        "duplicate_keys_by_partition": dict(zip(overlap.partition_names, overlap.duplicate_rows, strict=True)),
        "class_coverage": [asdict(item) for item in classes.groups],
        "subgroups": [asdict(item) for item in subgroup.groups] if subgroup else None,
        "limitations": ("Exact key disjointness does not prove independent partitions or absence "
                        "of related compounds. Marginal coverage assumes exchangeability and a fixed "
                        "complete class space. Class and subgroup coverage are descriptive, not "
                        "conditional guarantees. Empty and multi-label sets are possible. Wilson "
                        "intervals assume independent observations with common within-group coverage."),
    }
    if diagnostics:
        report["probability_diagnostics"] = classification_probability_diagnostics(
            model.predict_proba(test[features]), test[target].to_numpy(), list(result.classes))
        report["applicability"] = distance_domain_diagnostics(model, train, test, features, covered=result.covered)
    return report


def regression_uncertainty_from_files(
    train: str | Path, calibration: str | Path, test: str | Path, *,
    features: list[str], target: str, key: str, group: str | None = None,
    alpha: float = 0.1, diagnostics: bool = False,
) -> dict[str, Any]:
    """Read IDs as literal text, then fit/calibrate/evaluate using separate files."""
    return _uncertainty_from_files(train, calibration, test, features=features, target=target,
                                   key=key, group=group, alpha=alpha, task="regression", diagnostics=diagnostics)


def classification_uncertainty_from_files(
    train: str | Path, calibration: str | Path, test: str | Path, *,
    features: list[str], target: str, key: str, group: str | None = None,
    alpha: float = 0.1, diagnostics: bool = False,
) -> dict[str, Any]:
    """Read IDs and class labels as literal text and run held-out classification."""
    return _uncertainty_from_files(train, calibration, test, features=features, target=target,
                                   key=key, group=group, alpha=alpha, task="classification", diagnostics=diagnostics)


def _uncertainty_from_files(
    train: str | Path, calibration: str | Path, test: str | Path, *,
    features: list[str], target: str, key: str, group: str | None, alpha: float, task: str,
    diagnostics: bool,
) -> dict[str, Any]:
    from pharmaml.ml.provenance import environment_snapshot, fingerprint_file
    from pharmaml.utils.identifier_table import read_identifier_table

    paths = {"train": Path(train), "calibration": Path(calibration), "test": Path(test)}
    # Read and hash around each parse. This catches ordinary modifications, not
    # adversarial changes or mutation that is reverted during the read.
    fingerprints = {}
    frames = []
    for name, path in paths.items():
        before = fingerprint_file(path)
        frame = read_identifier_table(path, None)
        after = fingerprint_file(path)
        if before != after:
            raise DataValidationError(f"{name} input changed while reading; retry with stable files")
        fingerprints[name] = after
        frames.append(frame)
    for frame in frames:
        for column in ([*features, target] if task == "regression" else features):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
    evaluator = regression_uncertainty_report if task == "regression" else classification_uncertainty_report
    report = evaluator(*frames, features=features, target=target, key=key, group=group, alpha=alpha,
                       diagnostics=diagnostics)
    estimator = Ridge() if task == "regression" else LogisticRegression(max_iter=1000, random_state=0)
    report["provenance"] = {
        "schema": "pharmaml-uncertainty-run/1", "inputs": fingerprints,
        "parameters": {"workflow": ("split-conformal-ridge" if task == "regression" else "split-conformal-logistic"),
                       "features": list(features),
                       "target": target, "key": key, "group": group, "alpha": alpha,
                       "partitions": "user-supplied train/calibration/test",
                       **({"ridge_parameters": estimator.get_params(deep=False)} if task == "regression"
                          else {"logistic_parameters": estimator.get_params(deep=False)}),
                       "preprocessing": "median imputation and standard scaling",
                       **({"diagnostics": True} if diagnostics else {})},
        "environment": environment_snapshot(),
        "notice": "Input hashes assume stable files; records are unsigned and do not prove independence."
    }
    return report

"""A small, explicit held-out regression uncertainty workflow.

Separate files and disjoint exact IDs are necessary checks, not proof of
independence, exchangeability or absence of chemical similarity leakage.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.conformal import calibrate_split_conformal
from pharmaml.ml.coverage_diagnostics import coverage_by_group
from pharmaml.ml.preprocessing import build_preprocessor
from pharmaml.ml.split_audit import audit_split_overlap


def regression_uncertainty_report(
    train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, *,
    features: list[str], target: str, key: str, group: str | None = None,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Fit ridge and preprocessing on training only; calibrate and evaluate separately.

    Exact cross-partition key overlap is rejected. Features are explicitly
    listed numeric columns; target, key and group are excluded to reduce obvious
    leakage. The caller is responsible for selecting independent partitions,
    features and prespecified subgroups. Results are empirical; marginal
    conformal coverage assumes exchangeability, not subgroup guarantees.
    """
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
        for column in [*features, target]:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise DataValidationError(f"{name} column {column!r} must be numeric")
            values = frame[column].to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any() or (column == target and np.isnan(values).any()):
                raise DataValidationError(f"{name} column {column!r} has invalid values")
        if frame[features].isna().all().any():
            raise DataValidationError(f"{name} contains an entirely missing feature")
    overlap = audit_split_overlap({name: frame[key].tolist() for name, frame in partitions.items()})
    if overlap.has_overlap:
        raise DataValidationError("exact key overlap between train, calibration and test; refuse evaluation")
    model = Pipeline([("prep", build_preprocessor()), ("estimator", Ridge())])
    model.fit(train[features], train[target])
    calibrated = calibrate_split_conformal(model, calibration[features], calibration[target], alpha=alpha)
    result = calibrated.predict_interval(test[features], test[target])
    subgroup = None
    if group is not None:
        subgroup = coverage_by_group(result.covered, test[group].to_numpy(),
                                     sizes=result.upper - result.lower)
    return {
        "model": "ridge (median imputation and scaling fitted on train only)",
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


def regression_uncertainty_from_files(
    train: str | Path, calibration: str | Path, test: str | Path, *,
    features: list[str], target: str, key: str, group: str | None = None,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Read IDs as literal text, then fit/calibrate/evaluate using separate files."""
    from pharmaml.utils.identifier_table import read_identifier_table

    frames = [read_identifier_table(Path(path), None) for path in (train, calibration, test)]
    for frame in frames:
        for column in [*features, target]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
    return regression_uncertainty_report(*frames, features=features, target=target,
                                         key=key, group=group, alpha=alpha)

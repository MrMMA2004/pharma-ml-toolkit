"""Nested logistic classification with explicit class and group fold checks."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError, JobCancelled
from pharmaml.ml.preprocessing import build_preprocessor


def nested_logistic_validation(
    frame: pd.DataFrame, *, features: list[str], target: str, group: str | None = None,
    outer_folds: int = 3, inner_folds: int = 3, seed: int = 42,
    on_fold: Callable[[int, int], bool | None] | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed, nested logistic tuning procedure, not a final model.

    All outer and inner training AND evaluation folds must contain every class;
    grouped folds are disjoint but not stratified and may be rejected. Imputation
    and scaling are fitted separately inside every training fold. Cancellation is
    cooperative at outer boundaries, with no persisted partial report or resume.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise DataValidationError("data must be a nonempty table with unique columns")
    if (not target or group == "" or not features or len(set(features)) != len(features) or target in features
            or group in features or group == target):
        raise DataValidationError("features must be unique and exclude target and group")
    for count in (outer_folds, inner_folds):
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise DataValidationError("fold counts must be integers of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise DataValidationError("seed must be an integer between 0 and 4294967295")
    missing = [c for c in [*features, target, *([group] if group else [])] if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing columns: {', '.join(missing)}")
    for column in features:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DataValidationError(f"feature {column!r} must be numeric")
        if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise DataValidationError(f"feature {column!r} has infinite values")
    y = frame[target]
    if any(not isinstance(value, str) or not value.strip() for value in y):
        raise DataValidationError("target must contain nonempty string class labels")
    classes = sorted(set(y))
    if len(classes) < 2:
        raise DataValidationError("at least two distinct classes are required")
    groups = frame[group] if group is not None else None
    if groups is not None:
        if groups.isna().any() or groups.map(lambda v: isinstance(v, str) and not v.strip()).any():
            raise DataValidationError("groups cannot be missing or blank")
        if groups.nunique() < outer_folds:
            raise DataValidationError("not enough distinct groups for outer folds")
    elif (y.value_counts() < outer_folds).any():
        raise DataValidationError("each class needs at least one row per outer fold")
    X = frame[features]
    expected = set(classes)

    def checked_splits(indices, x, labels, level):
        pairs = list(indices)
        for train, test in pairs:
            if set(labels.iloc[train]) != expected or set(labels.iloc[test]) != expected:
                raise DataValidationError(f"every {level} training and evaluation fold must contain every class")
            if x.iloc[train].isna().all().any():
                raise DataValidationError(f"a {level} training fold has an entirely missing feature")
        return pairs

    if groups is None:
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    else:
        outer = GroupKFold(n_splits=outer_folds)
    outer_splits = checked_splits(outer.split(X, y, groups), X, y, "outer")
    # Materialize and validate all split boundaries before fitting any models.
    planned = []
    for fold, (training, evaluation) in enumerate(outer_splits, 1):
        x_train, y_train = X.iloc[training], y.iloc[training]
        if groups is None:
            if (y_train.value_counts() < inner_folds).any():
                raise DataValidationError("each class needs at least one row per inner fold")
            inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=(seed + fold) % 2**32)
            inner_groups = None
        else:
            inner_groups = groups.iloc[training]
            if inner_groups.nunique() < inner_folds:
                raise DataValidationError("not enough distinct groups in an outer training fold")
            inner = GroupKFold(n_splits=inner_folds)
        inner_splits = checked_splits(inner.split(x_train, y_train, inner_groups), x_train, y_train, "inner")
        planned.append((training, evaluation, inner_splits))

    records = []
    if on_fold is not None and on_fold(0, outer_folds) is False:
        raise JobCancelled("nested CV cancelled before the first fold")
    for fold, (training, evaluation, inner_splits) in enumerate(planned, 1):
        pipeline = Pipeline([("prep", build_preprocessor()),
                             ("estimator", LogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-4, random_state=seed))])
        search = GridSearchCV(pipeline, {"estimator__C": [0.1, 1.0, 10.0]},
                              scoring="balanced_accuracy", cv=inner_splits, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                search.fit(X.iloc[training], y.iloc[training])
            except ConvergenceWarning as exc:
                raise DataValidationError("logistic regression failed to converge; validation aborted") from exc
        observed = y.iloc[evaluation]
        predictions = search.predict(X.iloc[evaluation])
        records.append({"fold": fold, "train_rows": len(training), "test_rows": len(evaluation),
                        "balanced_accuracy": float(balanced_accuracy_score(observed, predictions)),
                        "accuracy": float(accuracy_score(observed, predictions)),
                        "confusion_matrix": confusion_matrix(observed, predictions, labels=classes).tolist(),
                        "selected_C": float(search.best_params_["estimator__C"])})
        if on_fold is not None and on_fold(fold, outer_folds) is False:
            raise JobCancelled("nested CV cancelled at an outer fold boundary")
    return {"task": "classification", "model": "logistic_regression", "metric": "balanced_accuracy",
            "features": list(features), "target": target, "group": group, "classes": classes,
            "outer_folds": outer_folds, "inner_folds": inner_folds, "seed": seed,
            "split_strategy": "group_kfold" if group else "stratified_shuffled_rows",
            "C_grid": [0.1, 1.0, 10.0],
            "model_parameters": {"solver": "lbfgs", "max_iter": 2000, "tol": 1e-4},
            "preprocessing": "median imputation and standard scaling fitted in each training fold",
            "folds": records,
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in records])),
            "limitations": ("Nested CV estimates this fixed tuning procedure, not a final model or external "
                            "validation. Fold scores may be correlated. Group choice and independence are "
                            "the user's responsibility; stratified rows can leak related compounds. "
                            "Group folds are not stratified. Confusion matrices use the listed class order, "
                            "rows true and columns predicted. No probability calibration is performed.")}


def nested_logistic_from_file(
    data: str | Path, *, features: list[str], target: str, group: str | None = None,
    outer_folds: int = 3, inner_folds: int = 3, seed: int = 42,
    on_fold: Callable[[int, int], bool | None] | None = None,
) -> dict[str, Any]:
    """Preserve class/group text, fingerprint the file, then run nested CV."""
    from pharmaml.ml.provenance import environment_snapshot, fingerprint_file
    from pharmaml.utils.identifier_table import read_identifier_table

    path = Path(data)
    before = fingerprint_file(path)
    frame = read_identifier_table(path, None)
    after = fingerprint_file(path)
    if before != after:
        raise DataValidationError("input changed while reading; retry with a stable file")
    for column in features:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    result = nested_logistic_validation(frame, features=features, target=target, group=group,
                                        outer_folds=outer_folds, inner_folds=inner_folds,
                                        seed=seed, on_fold=on_fold)
    result["provenance"] = {"schema": "pharmaml-nested-logistic-run/1", "input": after,
                            "environment": environment_snapshot(),
                            "notice": "File hash assumes a stable read; records are unsigned and not proof of validity."}
    return result

"""Narrow, nested-CV ridge regression validation with fold-local preprocessing."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.preprocessing import build_preprocessor


def nested_ridge_validation(
    frame: pd.DataFrame, *, features: list[str], target: str, group: str | None = None,
    outer_folds: int = 3, inner_folds: int = 3, seed: int = 42,
) -> dict[str, Any]:
    """Estimate a ridge tuning procedure using nested folds and training-only preprocessing.

    Group mode keeps the supplied groups disjoint in both outer and inner folds.
    Without groups, shuffled row folds do not prevent compound/patient leakage.
    Fixed grid, metric and seed are declared here; this is not a final model.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise DataValidationError("data must be a nonempty table with unique columns")
    if (not features or len(set(features)) != len(features) or target in features
            or group in features or group == target):
        raise DataValidationError("features must be unique and exclude target and group")
    for count in (outer_folds, inner_folds):
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise DataValidationError("fold counts must be integers of at least two")
    missing = [c for c in [*features, target, *([group] if group else [])] if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing columns: {', '.join(missing)}")
    for column in [*features, target]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DataValidationError(f"column {column!r} must be numeric")
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (column == target and np.isnan(values).any()):
            raise DataValidationError(f"column {column!r} has invalid values")
    groups = None
    if group is not None:
        groups = frame[group]
        if groups.isna().any() or groups.map(lambda v: isinstance(v, str) and not v.strip()).any():
            raise DataValidationError("groups cannot be missing or blank")
        if groups.nunique() < outer_folds:
            raise DataValidationError("not enough distinct groups for outer folds")
        outer = GroupKFold(n_splits=outer_folds)
    else:
        outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    X, y = frame[features], frame[target]
    records = []
    for fold, (training, evaluation) in enumerate(outer.split(X, y, groups), 1):
        X_train, y_train = X.iloc[training], y.iloc[training]
        if X_train.isna().all().any():
            raise DataValidationError("an outer training fold has an entirely missing feature")
        inner_groups = groups.iloc[training] if groups is not None else None
        if inner_groups is not None:
            if inner_groups.nunique() < inner_folds:
                raise DataValidationError("not enough distinct groups in an outer training fold")
            inner = GroupKFold(n_splits=inner_folds)
        else:
            inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        pipeline = Pipeline([("prep", build_preprocessor()), ("estimator", Ridge())])
        search = GridSearchCV(pipeline, {"estimator__alpha": [0.1, 1.0, 10.0]},
                              scoring="neg_mean_absolute_error", cv=inner, error_score="raise")
        search.fit(X_train, y_train, groups=inner_groups)
        predictions = search.predict(X.iloc[evaluation])
        records.append({"fold": fold, "train_rows": len(training), "test_rows": len(evaluation),
                        "mae": float(mean_absolute_error(y.iloc[evaluation], predictions)),
                        "selected_alpha": float(search.best_params_["estimator__alpha"])})
    scores = [row["mae"] for row in records]
    return {"model": "ridge", "metric": "mean_absolute_error", "features": features,
            "target": target, "group": group, "outer_folds": outer_folds, "inner_folds": inner_folds,
            "seed": seed, "alpha_grid": [0.1, 1.0, 10.0], "folds": records,
            "mean_mae": float(np.mean(scores)),
            "limitations": ("Nested CV estimates this fixed tuning procedure, not a final fitted model. "
                            "Fold scores can be correlated; group choice and independence are the "
                            "user's responsibility. Row splits do not prevent related-compound leakage.")}

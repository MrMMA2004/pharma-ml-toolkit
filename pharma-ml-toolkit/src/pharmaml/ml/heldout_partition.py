"""Positional holdout partitioning and conservative subgroup row alignment.

These checks cannot prove a model was fitted independently or that supplied
metadata is authentic. The caller must prevent leakage and use stable inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing

from pharmaml.exceptions import DataValidationError


def split_calibration_evaluation(X: Any, y: Any) -> tuple[Any, Any, Any, Any]:
    """Split held-out rows deterministically into disjoint calibration/evaluation sets.

    The two halves are randomly selected by position (not stratified). This
    cannot establish independence from model fitting or prior model selection.
    """
    try:
        n_rows = len(X)
        if len(y) != n_rows or n_rows < 2:
            raise ValueError("held-out feature and label lengths must match and contain at least two rows")
        calibration, evaluation = train_test_split(np.arange(n_rows), test_size=0.5, random_state=42)
        return (_safe_indexing(X, calibration), _safe_indexing(X, evaluation),
                _safe_indexing(y, calibration), _safe_indexing(y, evaluation))
    except (TypeError, ValueError, IndexError) as exc:
        raise DataValidationError(f"cannot partition held-out rows: {exc}") from exc


def aligned_group_values(frame: Any, X: Any, y: Any, target: str, group: str) -> np.ndarray:
    """Map current group metadata to the exact indexed scored rows or refuse.

    The feature and target values are checked in the selected order. Group
    labels themselves are not authenticated and are supplied by the current
    frame. Duplicate/absent indices and array-only features cannot be aligned.
    """
    if (not isinstance(frame, pd.DataFrame) or not isinstance(X, pd.DataFrame)
            or not frame.index.is_unique or not X.index.is_unique
            or not frame.columns.is_unique or not X.columns.is_unique
            or not target or target not in frame.columns or not group or group not in frame.columns
            or X.shape[1] == 0 or not X.columns.isin(frame.columns).all()
            or not X.index.isin(frame.index).all()):
        raise DataValidationError("cannot verify group alignment against indexed held-out features and labels")
    try:
        rows = frame.loc[X.index]
        expected = rows.loc[:, X.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, X.reset_index(drop=True), check_dtype=False, check_exact=True)
        actual_y = pd.Series(np.asarray(y), name=target).reset_index(drop=True)
        expected_y = rows[target].reset_index(drop=True)
        pd.testing.assert_series_equal(expected_y, actual_y, check_dtype=False, check_names=False, check_exact=True)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("current dataset features or target do not match the scored rows; "
                                  "cannot verify group alignment") from exc
    return rows[group].to_numpy(copy=True)

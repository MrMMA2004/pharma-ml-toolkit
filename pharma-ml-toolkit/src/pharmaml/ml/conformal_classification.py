"""Split-conformal prediction sets for fixed probabilistic classifiers.

Uses the true-label nonconformity score ``1 - P(label | X)`` and a held-out
calibration set. Under exchangeability of calibration and future observations,
conditional on an independently trained fixed classifier, and assuming all
possible labels belong to its fixed class space, the resulting sets have
marginal coverage at least ``1 - alpha``. This is not a guarantee for
individual classes, subgroups, or distribution shift. No model fitting occurs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.conformal import _validate_alpha

__all__ = [
    "ConformalPredictionSets",
    "SplitConformalClassifier",
    "calibrate_split_conformal_classifier",
]


def _labels(value: Any, name: str) -> list[Any]:
    """Validate a nonempty vector of finite, scalar string or numeric labels."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a one-dimensional label vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise DataValidationError(f"{name} must be a nonempty one-dimensional label vector")
    labels = array.tolist()
    for label in labels:
        if isinstance(label, (str, bytes, bool, np.bool_)):
            continue
        try:
            valid = isinstance(label, Real) and math.isfinite(float(label))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise DataValidationError(f"{name} must contain finite numeric or string labels")
    return labels


def _model_classes(model: Any) -> tuple[Any, ...]:
    if not callable(getattr(model, "predict_proba", None)):
        raise DataValidationError("model must have a callable predict_proba method")
    classes = _labels(getattr(model, "classes_", None), "model classes_")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise DataValidationError("model classes_ must contain at least two unique labels")
    return tuple(classes)


def _row_count(X: Any) -> int:
    """Accept common matrices, including sparse matrices whose len is ambiguous."""
    try:
        shape = getattr(X, "shape", None)
        count = int(shape[0]) if shape is not None and len(shape) > 0 else len(X)
    except (TypeError, IndexError, ValueError) as exc:
        raise DataValidationError("X must be a sized collection of rows") from exc
    if count < 1:
        raise DataValidationError("X must contain at least one row")
    return count


def _probabilities(model: Any, X: Any, classes: tuple[Any, ...]) -> np.ndarray:
    n = _row_count(X)
    try:
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("predict_proba must return a numeric probability matrix") from exc
    if probabilities.ndim != 2 or probabilities.shape != (n, len(classes)):
        raise DataValidationError("predict_proba must return one row per sample and one column per class")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise DataValidationError("probabilities must be finite, in [0, 1], and sum to 1 per row")
    return probabilities


@dataclass(frozen=True)
class ConformalPredictionSets:
    """Class membership per query row and optional empirical coverage.

    ``membership`` columns correspond to ``classes``; rows follow query input
    order. Sets may be empty or contain multiple labels. ``coverage`` is only
    computed when independent query labels were supplied and is not itself a
    guarantee. Arrays are not immutable despite the frozen record.
    """

    classes: tuple[Any, ...]
    membership: np.ndarray
    alpha: float
    threshold: float
    n_calibration: int
    covered: np.ndarray | None = None

    @property
    def prediction_sets(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(tuple(label for label, member in zip(self.classes, row, strict=True) if member)
                     for row in self.membership)

    @property
    def set_sizes(self) -> np.ndarray:
        return self.membership.sum(axis=1)

    @property
    def n_test(self) -> int:
        return int(self.membership.shape[0])

    @property
    def n_covered(self) -> int | None:
        return None if self.covered is None else int(self.covered.sum())

    @property
    def coverage(self) -> float | None:
        return None if self.covered is None else float(self.covered.mean())

    @property
    def mean_set_size(self) -> float:
        return float(self.set_sizes.mean())

    def to_dataframe(self) -> pd.DataFrame:
        """Return prediction sets, sizes, and (if supplied) coverage by row."""
        frame = pd.DataFrame({"prediction_set": list(self.prediction_sets), "set_size": self.set_sizes})
        if self.covered is not None:
            frame["covered"] = self.covered
        return frame


@dataclass(frozen=True)
class SplitConformalClassifier:
    """Reference to a fixed classifier and its held-out calibration threshold.

    Do not refit or otherwise mutate the classifier or preprocessing after
    calibration. The caller is responsible for independent data splits.
    """

    model: Any
    classes: tuple[Any, ...]
    alpha: float
    threshold: float
    n_calibration: int
    rank: int

    def predict_set(self, X: Any, y_true: Any = None) -> ConformalPredictionSets:
        """Return label sets and optionally evaluate on separate query labels."""
        if _model_classes(self.model) != self.classes:
            raise DataValidationError("model classes_ changed after calibration")
        probabilities = _probabilities(self.model, X, self.classes)
        membership = (1.0 - probabilities) <= self.threshold
        covered = None
        if y_true is not None:
            truth = _labels(y_true, "y_true")
            if len(truth) != len(probabilities):
                raise DataValidationError("y_true and predictions must have the same length")
            column = {label: index for index, label in enumerate(self.classes)}
            # A label outside the fitted class space cannot be in any set.
            covered = np.array([membership[row, column[label]] if label in column else False
                                for row, label in enumerate(truth)], dtype=bool)
        return ConformalPredictionSets(classes=self.classes, membership=membership, alpha=self.alpha,
                                       threshold=self.threshold, n_calibration=self.n_calibration,
                                       covered=covered)


def calibrate_split_conformal_classifier(
    model: Any, X_cal: Any, y_cal: Any, *, alpha: float = 0.1
) -> SplitConformalClassifier:
    """Calibrate split-conformal label sets on held-out classification rows.

    The fitted model must expose ``classes_`` and ``predict_proba``; probability
    columns must be in ``classes_`` order. Calibration labels must all appear
    in ``classes_``. Alignment is positional; pandas indexes are not checked.
    For n rows, use the k-th smallest true-label score ``1 - probability``,
    with k = ceil((n + 1) * (1 - alpha)), using 1-based indexing. If k > n,
    use threshold 1, producing the full class set for every query. This is the
    conservative finite-class equivalent of the infinite conformal quantile.

    The model is held by reference, not copied or fitted. Train and select the
    model using training rows alone; never use calibration or test rows for
    training or tuning. Coverage requires exchangeable calibration and future
    observations, with all possible labels in the fixed ``classes_`` space;
    it does not imply class-conditional or subgroup coverage.
    """
    alpha = _validate_alpha(alpha)
    classes = _model_classes(model)
    truth = _labels(y_cal, "y_cal")
    probabilities = _probabilities(model, X_cal, classes)
    if len(truth) != len(probabilities):
        raise DataValidationError("y_cal and predictions must have the same length")
    column = {label: index for index, label in enumerate(classes)}
    if any(label not in column for label in truth):
        raise DataValidationError("y_cal contains labels absent from model classes_")
    n = len(truth)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        threshold = 1.0
    else:
        scores = 1.0 - probabilities[np.arange(n), [column[label] for label in truth]]
        threshold = float(np.partition(scores, rank - 1)[rank - 1])
    return SplitConformalClassifier(model=model, classes=classes, alpha=alpha, threshold=threshold,
                                    n_calibration=n, rank=rank)

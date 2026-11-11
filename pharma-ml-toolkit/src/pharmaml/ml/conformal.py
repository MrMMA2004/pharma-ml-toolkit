"""Split-conformal prediction intervals for single-output regression.

Calibrate a previously fitted regressor on a separate, untouched calibration
set. The same fitted model must then be used without refitting. The symmetric
radius is the finite-sample order statistic of absolute calibration residuals;
no interpolation or fit on the calibration data is performed.

Under exchangeability of calibration and future observations, conditional on
an independently trained fixed model, split conformal offers marginal coverage
of at least ``1 - alpha``. It does not promise conditional, per-subgroup or
out-of-distribution coverage. Callers are responsible for independent splits,
including fitting all preprocessing on training data only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pharmaml.exceptions import DataValidationError
from pharmaml.ml._estimator_compat import estimator_kind

__all__ = ["ConformalIntervals", "SplitConformalRegressor", "calibrate_split_conformal"]


def _numeric_vector(value: Any, name: str) -> np.ndarray:
    """Validate a nonempty, finite one-dimensional vector."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a numeric one-dimensional vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise DataValidationError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{name} must contain only finite values")
    return array


def _row_count(X: Any, name: str) -> int:
    try:
        shape = getattr(X, "shape", None)
        count = int(shape[0]) if shape is not None and len(shape) > 0 else len(X)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise DataValidationError(f"{name} must expose a sample count") from exc
    if count < 1:
        raise DataValidationError(f"{name} must be nonempty")
    return count


def _predict(model: Any, X: Any) -> Any:
    try:
        return model.predict(X)
    except NotFittedError as exc:
        raise DataValidationError("model must be fitted before prediction") from exc


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise DataValidationError("alpha must be a finite number strictly between 0 and 1")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise DataValidationError("alpha must be a finite number strictly between 0 and 1")
    return value


@dataclass(frozen=True)
class ConformalIntervals:
    """Prediction intervals and optional empirical evaluation on query rows.

    ``covered``, ``coverage`` and ``n_covered`` are computed only when
    independent query labels were supplied; they are empirical, not the
    coverage guarantee. Arrays follow input order and remain mutable.
    """

    prediction: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    radius: float
    n_calibration: int
    coverage: float | None = None
    n_covered: int | None = None
    covered: np.ndarray | None = None

    @property
    def n_test(self) -> int:
        return int(self.prediction.size)

    @property
    def mean_width(self) -> float:
        return 2.0 * self.radius

    def to_dataframe(self) -> pd.DataFrame:
        """Return predictions and interval bounds in input row order."""
        return pd.DataFrame({"prediction": self.prediction, "lower": self.lower, "upper": self.upper})


@dataclass(frozen=True)
class SplitConformalRegressor:
    """A fitted model reference and its split-conformal calibration radius.

    Do not refit or mutate the referenced model after calibration. Training,
    calibration and evaluation data must be separate; this class cannot
    detect leakage or enforce exchangeability.
    """

    model: Any
    alpha: float
    radius: float
    n_calibration: int
    rank: int

    def predict_interval(self, X: Any, y_true: Any = None) -> ConformalIntervals:
        """Predict symmetric intervals on sized rows; optionally evaluate on separate labels."""
        n_rows = _row_count(X, "X")
        prediction = _numeric_vector(_predict(self.model, X), "model predictions")
        if prediction.size != n_rows:
            raise DataValidationError("X and model predictions must have the same length")
        with np.errstate(over="ignore", invalid="ignore"):
            lower = prediction - self.radius
            upper = prediction + self.radius
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise DataValidationError("interval bounds overflowed; check the prediction scale")
        coverage = None
        n_covered = None
        covered = None
        if y_true is not None:
            truth = _numeric_vector(y_true, "y_true")
            if truth.size != prediction.size:
                raise DataValidationError("y_true and predictions must have the same length")
            covered = (truth >= lower) & (truth <= upper)
            n_covered = int(np.count_nonzero(covered))
            coverage = n_covered / int(prediction.size)
        return ConformalIntervals(
            prediction=prediction, lower=lower, upper=upper, alpha=self.alpha,
            radius=self.radius, n_calibration=self.n_calibration,
            coverage=coverage, n_covered=n_covered, covered=covered,
        )


def calibrate_split_conformal(
    model: Any, X_cal: Any, y_cal: Any, *, alpha: float = 0.1
) -> SplitConformalRegressor:
    """Calibrate finite split-conformal intervals on held-out regression data.

    ``model`` must already be fitted and expose ``predict(X)``. Its predictions
    and ``y_cal`` must be finite, nonempty, aligned one-dimensional vectors;
    ``X_cal`` must expose its row count. Alignment is positional and pandas
    indexes are not checked. Sklearn estimators, including Pipelines, must
    identify as regressors and pass sklearn's fitted-state check; unknown or
    invalid estimator tags fail closed. Recognized classifiers and ambiguous
    untagged predictors with ``predict_proba`` are rejected. Lightweight
    untagged predictors without ``predict_proba`` remain supported, but the
    caller must ensure they represent regression and are ready to predict.
    For ``n`` calibration samples, the radius is the ``k``-th smallest
    absolute residual, where ``k = ceil((n + 1) * (1 - alpha))`` (1-based).
    When ``k > n``, a finite interval cannot provide the requested
    finite-sample guarantee; this function raises instead of silently clipping
    the rank. Increase the calibration size or choose a larger alpha.

    This function does not fit or copy the model. The caller must ensure the
    model and all preprocessing were fitted without calibration rows, and
    use a separate test set when reporting empirical coverage.
    """
    alpha = _validate_alpha(alpha)
    if not callable(getattr(model, "predict", None)):
        raise DataValidationError("model must have a callable predict method")
    kind = estimator_kind(model)
    if kind == "classifier":
        raise DataValidationError("model must be a regressor, not a classifier")
    if kind is None and callable(getattr(model, "predict_proba", None)):
        raise DataValidationError("ambiguous predictor with predict_proba may be a classifier")
    if isinstance(model, BaseEstimator) or callable(getattr(model, "__sklearn_tags__", None)):
        try:
            check_is_fitted(model)
        except NotFittedError as exc:
            raise DataValidationError("model must be fitted before calibration") from exc
        except (TypeError, AttributeError) as exc:
            raise DataValidationError("invalid sklearn estimator or estimator tags") from exc
    truth = _numeric_vector(y_cal, "y_cal")
    n = int(truth.size)
    if _row_count(X_cal, "X_cal") != n:
        raise DataValidationError("X_cal and y_cal must have the same length")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise DataValidationError(
            f"{n} calibration samples cannot give finite intervals at alpha={alpha:g}; "
            "increase calibration size or choose a larger alpha"
        )
    prediction = _numeric_vector(_predict(model, X_cal), "calibration predictions")
    if prediction.size != n:
        raise DataValidationError("y_cal and calibration predictions must have the same length")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = np.abs(truth - prediction)
    if not np.isfinite(residuals).all():
        raise DataValidationError("calibration residuals overflowed; check the target scale")
    radius = float(np.partition(residuals, rank - 1)[rank - 1])
    if not math.isfinite(2.0 * radius):
        raise DataValidationError("interval width overflowed; check the target scale")
    return SplitConformalRegressor(model=model, alpha=alpha, radius=radius,
                                   n_calibration=n, rank=rank)

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

from pharmaml.exceptions import DataValidationError, ModelError
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


def _as_1d(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a finite float 1-D array or raise."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise DataValidationError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains non-finite values")
    return array


def _python_labels(classes: Any) -> list[Any]:
    """Model ``classes_`` as native Python labels (not numpy scalars).

    Returned sets must contain plain ``int``/``str`` values so callers can
    compare them with ``==`` against ordinary Python labels — a numpy
    scalar prints as ``np.int64(1)`` and surprises both users and doctests.
    """
    return [value.item() if isinstance(value, np.generic) else value for value in classes]


# --- (r13) the prediction-interval API the ml predict/card/decide paths use ----


def conformal_quantile(scores: Any, alpha: float) -> float:
    """The split-conformal quantile of nonconformity ``scores``.

    Returns the ``ceil((n + 1) * (1 - alpha))``-th smallest score (1-based)
    — the finite-sample-corrected empirical quantile, *not* the plain
    ``numpy.quantile``.  When that rank exceeds ``n`` the quantile is
    ``+inf``: the calibration set is too small to certify ``1 - alpha``
    coverage, so the honest interval is the whole line (regression) or
    every label (classification).

    Parameters
    ----------
    scores:
        Nonconformity scores from the calibration split (finite, 1-D).
    alpha:
        Miscoverage level in ``(0, 1)``; coverage target is ``1 - alpha``.

    Returns
    -------
    float
        The conformal quantile (``math.inf`` when the rank exceeds ``n``).

    Raises
    ------
    DataValidationError
        When ``scores`` is empty/non-finite or ``alpha`` is outside
        ``(0, 1)``.

    Examples
    --------
    >>> from pharmaml.ml.conformal import conformal_quantile
    >>> conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.2)
    5.0
    >>> conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.1)
    inf
    """
    if not 0.0 < alpha < 1.0:
        raise DataValidationError(f"alpha must be in (0, 1), got {alpha!r}")
    array = _as_1d(scores, "scores")
    n = array.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    ordered = np.sort(array)
    return float(ordered[rank - 1])


class ConformalRegressor:
    """Split-conformal prediction intervals around any fitted regressor.

    Parameters
    ----------
    model:
        A fitted estimator exposing ``predict`` (sklearn-compatible).
    alpha:
        Miscoverage level in ``(0, 1)`` (default ``0.1`` → 90 % intervals).

    Attributes
    ----------
    q_:
        The calibration quantile, set by :meth:`calibrate`.
    scores_:
        The calibration nonconformity scores (absolute residuals).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from pharmaml.ml.conformal import ConformalRegressor
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> y = np.array([0.1, 1.9, 4.2, 5.8, 8.1, 9.9])
    >>> reg = ConformalRegressor(LinearRegression().fit(X, y), alpha=0.2)
    >>> _ = reg.calibrate(X, y)
    >>> lower, upper = reg.predict_interval(np.array([[2.5]]))
    >>> bool(lower[0] < 5.0 < upper[0])
    True
    """

    def __init__(self, model: Any, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise DataValidationError(f"alpha must be in (0, 1), got {alpha!r}")
        if not hasattr(model, "predict"):
            raise ModelError(
                "ConformalRegressor needs a fitted estimator with .predict, "
                f"got {type(model).__name__}"
            )
        self.model = model
        self.alpha = float(alpha)
        self.q_: float | None = None
        self.scores_: np.ndarray | None = None

    def calibrate(self, X_cal: Any, y_cal: Any) -> ConformalRegressor:
        """Score the calibration split and set the interval width.

        ``X_cal``/``y_cal`` must be data the model was **not** fitted on,
        or the coverage guarantee is void (the scores would be optimistic).
        """
        y = _as_1d(y_cal, "y_cal")
        if len(X_cal) != y.size:
            raise DataValidationError(
                f"X_cal has {len(X_cal)} row(s) but y_cal has {y.size} value(s)"
            )
        # Pass the caller's object (DataFrame or array) straight through so
        # a fitted sklearn Pipeline keeps the feature names it trained on.
        predictions = _as_1d(self.model.predict(X_cal), "model.predict(X_cal)")
        self.scores_ = np.abs(y - predictions)
        self.q_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict_interval(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Point predictions widened by the calibration quantile.

        Returns ``(lower, upper)`` arrays; with an ``inf`` quantile both
        are infinite (the honest "no guarantee possible" answer).
        """
        if self.q_ is None:
            raise ModelError("call calibrate() before predict_interval()")
        center = _as_1d(self.model.predict(X), "model.predict(X)")
        return center - self.q_, center + self.q_

    def coverage(self, X: Any, y: Any) -> float:
        """Empirical fraction of ``y`` inside the predicted intervals.

        The promise kept on a *new* held-out set — compare it against the
        target ``1 - alpha`` rather than trusting the theory alone.
        """
        if self.q_ is None:
            raise ModelError("call calibrate() before coverage()")
        truth = _as_1d(y, "y")
        lower, upper = self.predict_interval(X)
        if truth.size != lower.size:
            raise DataValidationError(
                f"y has {truth.size} value(s) but X has {lower.size} row(s)"
            )
        return float(np.mean((truth >= lower) & (truth <= upper)))


class ConformalClassifier:
    """Split-conformal prediction sets around any fitted classifier.

    The nonconformity score is ``1 - p(true class)``; a candidate label is
    kept in the set when its own ``1 - p(label)`` does not exceed the
    calibration quantile, so the set shrinks as the model gets confident.

    Parameters
    ----------
    model:
        A fitted classifier exposing ``predict_proba`` and ``classes_``.
    alpha:
        Miscoverage level in ``(0, 1)`` (default ``0.1``).

    Attributes
    ----------
    q_:
        The calibration quantile, set by :meth:`calibrate`.
    scores_:
        The calibration nonconformity scores (``1 - p(true class)``).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from pharmaml.ml.conformal import ConformalClassifier
    >>> X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> clf = ConformalClassifier(LogisticRegression().fit(X, y), alpha=0.3)
    >>> _ = clf.calibrate(X, y)
    >>> sorted(clf.predict_set(np.array([[3.0]]))[0])
    [1]
    """

    def __init__(self, model: Any, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise DataValidationError(f"alpha must be in (0, 1), got {alpha!r}")
        if not hasattr(model, "predict_proba"):
            raise ModelError(
                "ConformalClassifier needs a fitted classifier with "
                f".predict_proba, got {type(model).__name__}"
            )
        self.model = model
        self.alpha = float(alpha)
        self.q_: float | None = None
        self.scores_: np.ndarray | None = None

    def _true_class_probability(self, X: Any, y: np.ndarray) -> np.ndarray:
        proba = np.asarray(self.model.predict_proba(X), dtype=float)
        classes = _python_labels(self.model.classes_)
        index = {label: position for position, label in enumerate(classes)}
        try:
            positions = np.array([index[value] for value in y.tolist()])
        except KeyError as exc:  # a calibration label the model never saw
            raise DataValidationError(
                f"y_cal contains a label the model was not trained on: {exc.args[0]!r}"
            ) from exc
        return proba[np.arange(proba.shape[0]), positions]

    def calibrate(self, X_cal: Any, y_cal: Any) -> ConformalClassifier:
        """Score the calibration split and set the set-membership threshold."""
        y = np.asarray(y_cal).reshape(-1)
        if len(X_cal) != y.size:
            raise DataValidationError(
                f"X_cal has {len(X_cal)} row(s) but y_cal has {y.size} value(s)"
            )
        true_prob = self._true_class_probability(X_cal, y)
        self.scores_ = 1.0 - true_prob
        self.q_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict_set(self, X: Any) -> list[list[Any]]:
        """One label set per row: every label within the conformal quantile.

        An empty set means the model is confidently *wrong* about every
        option (a signal, not a bug); with an ``inf`` quantile every label
        is returned.
        """
        if self.q_ is None:
            raise ModelError("call calibrate() before predict_set()")
        proba = np.asarray(self.model.predict_proba(X), dtype=float)
        classes = _python_labels(self.model.classes_)
        sets: list[list[Any]] = []
        for row in proba:
            if math.isinf(self.q_):
                sets.append(list(classes))
                continue
            keep = [label for label, p in zip(classes, row) if (1.0 - p) <= self.q_]
            sets.append(keep)
        return sets

    def coverage(self, X: Any, y: Any) -> float:
        """Empirical fraction of rows whose true label is in the set."""
        if self.q_ is None:
            raise ModelError("call calibrate() before coverage()")
        truth = np.asarray(y).reshape(-1)
        sets = self.predict_set(X)
        if truth.size != len(sets):
            raise DataValidationError(
                f"y has {truth.size} value(s) but X has {len(sets)} row(s)"
            )
        return float(np.mean([value in label_set for value, label_set in zip(truth, sets)]))

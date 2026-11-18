"""Probability calibration: does a predicted 0.8 actually mean 80%?

Discrimination (ROC-AUC, :mod:`pharmaml.ml.evaluation`) says *whether* a
classifier ranks actives above inactives.  Calibration says whether its
probabilities are honest — whether the compounds it scores at 0.8 really
are positive about 80% of the time.  The two are independent: a model can
rank perfectly and still be wildly over- or under-confident, and it is the
calibration (not the ranking) that a threshold, an expected-cost decision
or a consensus rescore actually consumes.  The toolkit already *draws* a
reliability diagram (:func:`pharmaml.viz.ml_plots.plot_calibration_curve`);
this module is its numeric half — the scores a report or a CI gate needs.

Three readouts, in increasing strictness:

* **Brier score** — mean squared error of the probabilities,
  ``mean((p - y)^2)``; a proper scoring rule, lower is better.  It rewards
  both good ranking and honest probabilities at once.
* **Expected Calibration Error (ECE)** — the bin-weighted mean gap between
  the predicted probability and the observed frequency, ``sum(n_b/N) *
  |mean_predicted - fraction_positive|``.  The headline number.
* **Maximum Calibration Error (MCE)** — the largest single-bin gap: the
  worst place in the probability range to trust the model.

When the numbers say the model is miscalibrated, :func:`recalibrate`
fits a post-hoc map (isotonic or Platt/sigmoid) on held-out data and
returns a fitted estimator with the same API, so the fix travels with the
diagnosis.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.calibration import expected_calibration_error
>>> y = np.array([0, 0, 1, 1])
>>> expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0]))
0.0

A classifier that always says 0.5 on a balanced set is perfectly
calibrated but useless — ECE 0, Brier 0.25:

>>> from pharmaml.ml.calibration import brier_score
>>> round(brier_score(y, np.full(4, 0.5)), 3)
0.25
>>> round(expected_calibration_error(y, np.full(4, 0.5)), 3)
0.0

An over-confident one (says 0.9 on half-positive data) is not:

>>> round(expected_calibration_error(y, np.full(4, 0.9)), 2)
0.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError, ModelError

__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "maximum_calibration_error",
    "recalibrate",
    "reliability_bins",
]


def _as_probability(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a finite 1-D array of probabilities in [0, 1]."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise DataValidationError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains non-finite value(s)")
    if array.min() < 0.0 or array.max() > 1.0:
        raise DataValidationError(
            f"{name} must be probabilities in [0, 1], got range "
            f"[{array.min():.4g}, {array.max():.4g}]"
        )
    return array


def _as_binary(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a 0/1 int array (accepts bool, 0/1, or two labels)."""
    array = np.asarray(values)
    if array.size == 0:
        raise DataValidationError(f"{name} is empty")
    unique = np.unique(array)
    if unique.size > 2:
        raise DataValidationError(
            f"{name} has {unique.size} distinct labels — calibration here is "
            "binary (two classes); encode a one-vs-rest target first"
        )
    if unique.size == 1:
        return (array == unique[0]).astype(int)
    positive = unique[-1]
    return (array == positive).astype(int)


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Mean squared error of the probabilities: ``mean((p - y)^2)``.

    A proper scoring rule for binary problems; ``0`` is perfect.  ``y_true``
    is coerced to 0/1 (the larger label is treated as positive); ``y_prob``
    must be the positive-class probability in ``[0, 1]``.
    """
    y = _as_binary(y_true, "y_true")
    p = _as_probability(y_prob, "y_prob")
    if y.size != p.size:
        raise DataValidationError(
            f"y_true and y_prob must be the same length, got {y.size} and {p.size}"
        )
    return float(np.mean((p - y) ** 2))


def reliability_bins(
    y_true: Any, y_prob: Any, *, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Equal-width reliability bins of ``(count, mean_predicted, observed)``.

    Splits ``[0, 1]`` into ``n_bins`` equal-width bins (the same
    ``strategy="uniform"`` the reliability plot draws), and for each bin
    that has samples records the mean predicted probability and the
    observed positive frequency.  Empty bins are omitted.
    """
    if n_bins < 2:
        raise DataValidationError(f"n_bins must be >= 2, got {n_bins}")
    y = _as_binary(y_true, "y_true")
    p = _as_probability(y_prob, "y_prob")
    if y.size != p.size:
        raise DataValidationError(
            f"y_true and y_prob must be the same length, got {y.size} and {p.size}"
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins, except the last which includes 1.0.
    index = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    out: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_predicted=float(p[mask].mean()),
                fraction_positive=float(y[mask].mean()),
            )
        )
    return out


def expected_calibration_error(y_true: Any, y_prob: Any, *, n_bins: int = 10) -> float:
    """Bin-weighted mean gap between predicted probability and observed rate.

    ``sum(n_b / N) * |mean_predicted_b - fraction_positive_b|`` over the
    equal-width bins.  ``0`` is perfectly calibrated; ``0.05`` is the
    common "good enough" bar.
    """
    y = _as_binary(y_true, "y_true")
    p = _as_probability(y_prob, "y_prob")
    bins = reliability_bins(y, p, n_bins=n_bins)
    total = float(y.size)
    return float(
        sum(b.count / total * abs(b.mean_predicted - b.fraction_positive) for b in bins)
    )


def maximum_calibration_error(y_true: Any, y_prob: Any, *, n_bins: int = 10) -> float:
    """Largest single-bin gap between predicted probability and observed rate."""
    y = _as_binary(y_true, "y_true")
    p = _as_probability(y_prob, "y_prob")
    bins = reliability_bins(y, p, n_bins=n_bins)
    if not bins:
        return 0.0
    return float(max(abs(b.mean_predicted - b.fraction_positive) for b in bins))


@dataclass(slots=True)
class ReliabilityBin:
    """One equal-width bin of the reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    fraction_positive: float

    @property
    def gap(self) -> float:
        """Signed gap ``mean_predicted - fraction_positive``.

        Positive means the model is over-confident in this bin (it says
        more than the data shows); negative means under-confident.
        """
        return self.mean_predicted - self.fraction_positive


@dataclass(slots=True)
class CalibrationReport:
    """Calibration diagnostics for a binary classifier on held-out data.

    Attributes
    ----------
    brier, ece, mce:
        Brier score, expected and maximum calibration error.
    n_bins, n_samples, base_rate:
        Bin count, sample count, and the positive base rate.
    bins:
        The :class:`ReliabilityBin` table behind the scores.
    """

    brier: float
    ece: float
    mce: float
    n_bins: int
    n_samples: int
    base_rate: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def worst_bin(self) -> ReliabilityBin:
        """The bin with the largest absolute gap (the least trustworthy)."""
        if not self.bins:
            raise ValueError("no populated bins")
        return max(self.bins, key=lambda b: abs(b.gap))

    @property
    def verdict(self) -> str:
        """``"well-calibrated"`` / ``"fair"`` / ``"miscalibrated"`` by ECE."""
        if self.ece < 0.05:
            return "well-calibrated"
        if self.ece < 0.10:
            return "fair"
        return "miscalibrated"

    def to_frame(self) -> Any:
        """The reliability table as a pandas DataFrame (one row per bin)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "lower": b.lower,
                    "upper": b.upper,
                    "count": b.count,
                    "mean_predicted": b.mean_predicted,
                    "fraction_positive": b.fraction_positive,
                    "gap": b.gap,
                }
                for b in self.bins
            ]
        )

    def __str__(self) -> str:
        head = (
            f"calibration: brier={self.brier:.4f}  ece={self.ece:.4f}  "
            f"mce={self.mce:.4f}  ({self.n_bins} bins, "
            f"{self.n_samples} rows) — {self.verdict}"
        )
        if not self.bins:
            return head
        w = self.worst_bin
        return (
            head
            + f"\n  worst bin [{w.lower:.1f}, {w.upper:.1f}): predicted "
            f"{w.mean_predicted:.2f} vs actual {w.fraction_positive:.2f} "
            f"(gap {w.gap:+.2f}, n={w.count})"
        )


def _positive_scores(model: Any, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(y_binary, positive_probability)`` for a binary classifier."""
    if not hasattr(model, "predict_proba"):
        raise ModelError(
            "calibration needs a model with predict_proba; fit a probabilistic "
            "classifier (logistic_regression, random_forest, ...)"
        )
    proba = np.asarray(model.predict_proba(X))
    classes = getattr(model, "classes_", None)
    y_arr = np.asarray(y)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ModelError(
            "calibration_report is binary; the model's predict_proba returned "
            f"{proba.shape[1] if proba.ndim == 2 else 'non-2-D'} column(s)"
        )
    # Positive class = the model's own second class_ (sklearn's sorted order);
    # fall back to the larger label when classes_ is absent.
    if classes is not None and len(classes) == 2:
        positive = classes[1]
    else:
        positive = np.unique(y_arr)[-1]
    y_binary = (y_arr == positive).astype(int)
    return y_binary, proba[:, 1]


def calibration_report(
    model: Any, X: Any, y: Any, *, n_bins: int = 10
) -> CalibrationReport:
    """Full calibration diagnostics for a fitted binary classifier.

    Parameters
    ----------
    model:
        Fitted classifier exposing ``predict_proba`` (a Pipeline fits).
    X, y:
        Held-out features and labels (never the training rows — a model is
        optimistic about the data it was fitted on).
    n_bins:
        Equal-width probability bins.

    Returns
    -------
    CalibrationReport
        Brier, ECE, MCE, and the bin table.
    """
    y_binary, scores = _positive_scores(model, X, y)
    return CalibrationReport(
        brier=brier_score(y_binary, scores),
        ece=expected_calibration_error(y_binary, scores, n_bins=n_bins),
        mce=maximum_calibration_error(y_binary, scores, n_bins=n_bins),
        n_bins=n_bins,
        n_samples=int(y_binary.size),
        base_rate=float(y_binary.mean()),
        bins=tuple(reliability_bins(y_binary, scores, n_bins=n_bins)),
    )


def recalibrate(
    model: Any, X_cal: Any, y_cal: Any, *, method: str = "isotonic"
) -> Any:
    """Fit a post-hoc calibrator on held-out data and return the wrapped model.

    Wraps the frozen ``model`` with scikit-learn's
    :class:`~sklearn.calibration.CalibratedClassifierCV`, fitting the
    calibration map on ``(X_cal, y_cal)`` — which must be data the model
    did **not** train on, or the map learns the model's optimism rather
    than its bias.

    Parameters
    ----------
    method:
        ``"isotonic"`` (non-parametric, needs a few hundred rows) or
        ``"sigmoid"`` (Platt scaling, a two-parameter logistic fit that
        behaves better on small calibration sets).

    Returns
    -------
    CalibratedClassifierCV
        A fitted estimator with the same ``predict`` / ``predict_proba``
        API as ``model``.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise DataValidationError(
            f"method must be 'isotonic' or 'sigmoid', got {method!r}"
        )
    if not hasattr(model, "predict_proba"):
        raise ModelError("recalibrate needs a model with predict_proba")
    from sklearn.calibration import CalibratedClassifierCV

    try:  # (r13) sklearn >= 1.6: freeze the fitted model outright
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:  # sklearn 1.5.x (the pinned floor): prefit is the
        # same contract — the inner estimator is treated as already fitted
        # and never refit on the calibration fold.
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, np.asarray(y_cal))
    return calibrated

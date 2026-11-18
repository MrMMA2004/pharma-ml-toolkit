"""Decision-curve analysis: does acting on the model beat the default rules?

The trust layer so far answers two questions.  Drift
(:mod:`pharmaml.ml.drift`) asks whether a new batch has moved away from the
training population.  Calibration (:mod:`pharmaml.ml.calibration`) asks
whether the predicted probabilities are honest — whether a ``0.8`` really
means 80 %.  This module answers the decision-theoretic one that actually
governs whether a model is *useful*: **at a given threshold, does acting on
the model's call beat treating everyone or treating no one?**

That is the **net benefit** — Vickers & Elkin's decision-curve analysis,
the standard readout when a threshold carries a clinical or portfolio cost
(a go/no-go on a costly assay, an enrichment decision, a safety flag).
Net benefit counts true positives and charges false positives at the odds
of the threshold:

``NB = TP/N - FP/N * (p_t / (1 - p_t))``

where ``p_t`` is the threshold probability.  The ratio ``p_t/(1-p_t)`` is
exactly the odds of a false positive relative to a true positive that makes
a clinician indifferent at ``p_t``; ``net_benefit`` exposes it directly as
``cost_ratio`` for the "one missed case is worth k false alarms" framing.
Net benefit is expressed in units of *net true positives per subject*, so
two models (or a model and "treat all") are comparable on one axis, and a
value of zero equals "treat none".  :func:`decision_curve` sweeps a grid of
thresholds and records the model against both defaults, so the useful
region — where the model's curve clears "treat all" — is read off, not
assumed.

:func:`optimal_threshold` is the single-number companion: the threshold
that maximises Youden's J, F1, or net benefit (expected value) on held-out
data.

Examples
--------
A model that separates the classes cleanly beats "treat all" across the
useful threshold range, and "treat none" (net benefit 0) everywhere
positive:

>>> import numpy as np
>>> from pharmaml.ml.decision import net_benefit, treat_all_net_benefit
>>> y = np.array([0, 0, 1, 1])
>>> p = np.array([0.1, 0.2, 0.8, 0.9])
>>> round(net_benefit(y, p, threshold=0.5), 3)
0.5
>>> round(treat_all_net_benefit(y, threshold=0.5), 3)
0.0

At threshold 0.5 (cost ratio 1) the model catches both positives with no
false alarms, while "treat all" on a balanced set is worth nothing:

>>> from pharmaml.ml.decision import decision_curve_from_scores
>>> y = np.array([0, 0, 1, 1])
>>> p = np.array([0.1, 0.4, 0.6, 0.9])
>>> curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
>>> round(curve.max_net_benefit.net_benefit, 3)
0.5
>>> curve.optimal_threshold
0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError, ModelError

__all__ = [
    "DecisionCurve",
    "DecisionPoint",
    "decision_curve",
    "decision_curve_from_scores",
    "net_benefit",
    "optimal_threshold",
    "standardized_net_benefit",
    "treat_all_net_benefit",
]


def _as_binary(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a 0/1 int array (the larger of two labels is positive)."""
    array = np.asarray(values)
    if array.size == 0:
        raise DataValidationError(f"{name} is empty")
    unique = np.unique(array)
    if unique.size < 2:
        raise DataValidationError(
            f"{name} has only one class — decision-curve analysis needs both "
            "positives and negatives to weigh a true positive against a false one"
        )
    if unique.size > 2:
        raise DataValidationError(
            f"{name} has {unique.size} distinct labels — decision-curve "
            "analysis here is binary (two classes)"
        )
    positive = unique[-1]
    return (array == positive).astype(int)


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


def _check_pair(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y = _as_binary(y_true, "y_true")
    p = _as_probability(y_prob, "y_prob")
    if y.size != p.size:
        raise DataValidationError(
            f"y_true and y_prob must be the same length, got {y.size} and {p.size}"
        )
    return y, p


def _odds(threshold: float, cost_ratio: float) -> float:
    """The false-positive : true-positive cost odds at ``threshold``."""
    if not 0.0 <= threshold < 1.0:
        raise DataValidationError(
            f"threshold must be in [0, 1), got {threshold} (a threshold of 1 "
            "treats no one and has no finite odds)"
        )
    return cost_ratio * threshold / (1.0 - threshold)


def net_benefit(
    y_true: Any, y_prob: Any, *, threshold: float, cost_ratio: float = 1.0
) -> float:
    """Net benefit of thresholding ``y_prob`` at ``threshold``.

    ``TP/N - FP/N * cost_ratio * threshold/(1-threshold)`` — net true
    positives per subject.  ``cost_ratio`` (default 1) is how many false
    positives are worth one true positive; the odds term is otherwise the
    threshold probability's own ``p/(1-p)``.  ``0`` equals "treat none";
    negative means the rule is worse than doing nothing.
    """
    if cost_ratio <= 0:
        raise DataValidationError(f"cost_ratio must be positive, got {cost_ratio}")
    y, p = _check_pair(y_true, y_prob)
    n = y.size
    positives = p >= threshold
    tp = float(np.sum(positives & (y == 1)))
    fp = float(np.sum(positives & (y == 0)))
    return tp / n - (fp / n) * _odds(threshold, cost_ratio)


def treat_all_net_benefit(
    y_true: Any, *, threshold: float, cost_ratio: float = 1.0
) -> float:
    """Net benefit of calling every subject positive at ``threshold``.

    ``prevalence - (1 - prevalence) * cost_ratio * threshold/(1-threshold)``:
    the reference line the model must clear to be worth using.  Negative
    when the threshold's cost outweighs the base rate.
    """
    if cost_ratio <= 0:
        raise DataValidationError(f"cost_ratio must be positive, got {cost_ratio}")
    y = _as_binary(y_true, "y_true")
    prevalence = float(y.mean())
    return prevalence - (1.0 - prevalence) * _odds(threshold, cost_ratio)


def standardized_net_benefit(
    y_true: Any, y_prob: Any, *, threshold: float, cost_ratio: float = 1.0
) -> float:
    """Net benefit divided by prevalence (a prevalence-free scale).

    Dividing by the base rate makes curves comparable across cohorts whose
    prevalence differs; ``1.0`` means "captures every positive with no false
    alarms", ``0`` is "no better than treat none".
    """
    y, p = _check_pair(y_true, y_prob)
    prevalence = float(y.mean())
    return net_benefit(y, p, threshold=threshold, cost_ratio=cost_ratio) / prevalence


@dataclass(slots=True)
class DecisionPoint:
    """One threshold on the decision curve."""

    threshold: float
    net_benefit: float
    standardized_net_benefit: float
    treat_all_net_benefit: float

    @property
    def beats_treat_all(self) -> bool:
        """Whether the model is worth using over "treat all" here."""
        return self.net_benefit > self.treat_all_net_benefit

    @property
    def beats_treat_none(self) -> bool:
        """Whether the model is worth using over "treat none" (net benefit > 0)."""
        return self.net_benefit > 0.0


@dataclass(slots=True)
class DecisionCurve:
    """The model's net benefit against both default strategies, over a grid.

    Attributes
    ----------
    cost_ratio, n_samples, base_rate:
        The false-positive cost ratio, sample count, and positive prevalence.
    points:
        The :class:`DecisionPoint` table, one per threshold.
    """

    cost_ratio: float
    n_samples: int
    base_rate: float
    points: tuple[DecisionPoint, ...]

    @property
    def max_net_benefit(self) -> DecisionPoint:
        """The threshold with the highest net benefit on the grid."""
        if not self.points:
            raise ValueError("empty decision curve")
        return max(self.points, key=lambda pt: pt.net_benefit)

    @property
    def optimal_threshold(self) -> float:
        """The grid threshold that maximises net benefit."""
        return self.max_net_benefit.threshold

    @property
    def useful_region(self) -> tuple[float, float] | None:
        """The ``(low, high)`` thresholds where the model beats "treat all".

        ``None`` when the model never clears the default — the honest
        "do not use this model to decide" verdict.
        """
        good = [pt.threshold for pt in self.points if pt.beats_treat_all]
        if not good:
            return None
        return min(good), max(good)

    def net_benefit_at(self, threshold: float) -> float:
        """The curve's net benefit at the grid point nearest ``threshold``."""
        if not self.points:
            raise ValueError("empty decision curve")
        nearest = min(self.points, key=lambda pt: abs(pt.threshold - threshold))
        return nearest.net_benefit

    def to_frame(self) -> Any:
        """The curve as a pandas DataFrame (one row per threshold)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "threshold": pt.threshold,
                    "net_benefit": pt.net_benefit,
                    "standardized_net_benefit": pt.standardized_net_benefit,
                    "treat_all_net_benefit": pt.treat_all_net_benefit,
                }
                for pt in self.points
            ]
        )

    def __str__(self) -> str:
        best = self.max_net_benefit
        region = self.useful_region
        head = (
            f"decision: max net benefit {best.net_benefit:.4f} at "
            f"threshold {best.threshold:.2f} "
            f"(cost ratio {self.cost_ratio:g}, {self.n_samples} rows, "
            f"base rate {self.base_rate:.3f})"
        )
        if region is None:
            return head + "\n  model never beats 'treat all' — do not use it to decide"
        return (
            head
            + f"\n  beats 'treat all' for thresholds in [{region[0]:.2f}, "
            f"{region[1]:.2f}]"
        )


def _curve_points(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, cost_ratio: float
) -> tuple[DecisionPoint, ...]:
    n = y.size
    prevalence = float(y.mean())
    points: list[DecisionPoint] = []
    for t in thresholds:
        positives = p >= t
        tp = float(np.sum(positives & (y == 1)))
        fp = float(np.sum(positives & (y == 0)))
        odds = cost_ratio * float(t) / (1.0 - float(t))
        nb = tp / n - (fp / n) * odds
        ta = prevalence - (1.0 - prevalence) * odds
        points.append(
            DecisionPoint(
                threshold=float(t),
                net_benefit=nb,
                standardized_net_benefit=nb / prevalence if prevalence else float("nan"),
                treat_all_net_benefit=ta,
            )
        )
    return tuple(points)


def _default_thresholds(n_points: int) -> np.ndarray:
    if n_points < 2:
        raise DataValidationError(f"n_points must be >= 2, got {n_points}")
    # Interior grid: 0 and 1 are the degenerate "treat all" / "treat none"
    # ends and carry no odds.
    return np.linspace(0.01, 0.99, n_points)


def decision_curve_from_scores(
    y_true: Any,
    y_prob: Any,
    *,
    thresholds: Any | None = None,
    n_points: int = 100,
    cost_ratio: float = 1.0,
) -> DecisionCurve:
    """Decision curve from precomputed positive-class probabilities.

    Parameters
    ----------
    y_true, y_prob:
        Held-out labels and the model's positive-class probabilities.
    thresholds:
        Explicit threshold grid (each in ``[0, 1)``); defaults to
        ``n_points`` evenly spaced interior values.
    cost_ratio:
        False-positive : true-positive cost ratio (see :func:`net_benefit`).
    """
    if cost_ratio <= 0:
        raise DataValidationError(f"cost_ratio must be positive, got {cost_ratio}")
    y, p = _check_pair(y_true, y_prob)
    if thresholds is None:
        grid = _default_thresholds(n_points)
    else:
        grid = np.asarray(thresholds, dtype=float).reshape(-1)
        if grid.size == 0:
            raise DataValidationError("thresholds is empty")
        if np.any(grid < 0.0) or np.any(grid >= 1.0):
            raise DataValidationError("every threshold must be in [0, 1)")
    return DecisionCurve(
        cost_ratio=cost_ratio,
        n_samples=int(y.size),
        base_rate=float(y.mean()),
        points=_curve_points(y, p, grid, cost_ratio),
    )


def _positive_scores(model: Any, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(y_binary, positive_probability)`` for a fitted binary classifier."""
    if not hasattr(model, "predict_proba"):
        raise ModelError(
            "decision-curve analysis needs a model with predict_proba; fit a "
            "probabilistic classifier (logistic_regression, random_forest, ...)"
        )
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ModelError(
            "decision_curve is binary; the model's predict_proba returned "
            f"{proba.shape[1] if proba.ndim == 2 else 'non-2-D'} column(s)"
        )
    y_arr = np.asarray(y)
    classes = getattr(model, "classes_", None)
    positive = classes[1] if classes is not None and len(classes) == 2 else np.unique(y_arr)[-1]
    return (y_arr == positive).astype(int), proba[:, 1]


def decision_curve(
    model: Any,
    X: Any,
    y: Any,
    *,
    thresholds: Any | None = None,
    n_points: int = 100,
    cost_ratio: float = 1.0,
) -> DecisionCurve:
    """Decision curve for a fitted binary classifier on held-out data.

    See :func:`decision_curve_from_scores` for the arguments; ``model`` must
    expose ``predict_proba`` (a Pipeline fits).
    """
    y_binary, scores = _positive_scores(model, X, y)
    return decision_curve_from_scores(
        y_binary, scores, thresholds=thresholds, n_points=n_points, cost_ratio=cost_ratio
    )


def _sweep_counts(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate ``(thresholds, tp, fp, fn, tn)`` from one descending sort.

    ``O(n log n)``: sort scores once and take cumulative positives/negatives,
    so the optimal-threshold sweep never becomes quadratic on large batches.
    """
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    p_sorted = p[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = int(y.sum())
    total_neg = int(y.size - total_pos)
    fn = total_pos - tp
    tn = total_neg - fp
    return p_sorted, tp, fp, fn, tn


def optimal_threshold(
    y_true: Any,
    y_prob: Any,
    *,
    criterion: str = "youden",
    cost_ratio: float = 1.0,
) -> float:
    """The held-out threshold that maximises ``criterion``.

    Parameters
    ----------
    criterion:
        ``"youden"`` (sensitivity + specificity - 1), ``"f1"`` (the positive
        class's F1), or ``"cost"`` (maximise net benefit — the expected-value
        choice at ``cost_ratio``).
    cost_ratio:
        Used only by ``criterion="cost"``.

    Returns
    -------
    float
        A threshold drawn from the observed scores (the decision boundary
        that realises the optimum).  ``0.5`` when no candidate separates.
    """
    if criterion not in {"youden", "f1", "cost"}:
        raise DataValidationError(
            f"criterion must be 'youden', 'f1' or 'cost', got {criterion!r}"
        )
    y, p = _check_pair(y_true, y_prob)
    total_pos = int(y.sum())
    total_neg = int(y.size - total_pos)
    thresholds, tp, fp, fn, tn = _sweep_counts(y, p)
    tp = tp.astype(float)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tn = tn.astype(float)
    # the three criteria branches produce array-or-scalar numpy maths
    score: Any
    if criterion == "youden":
        score = tp / total_pos + tn / total_neg - 1.0
    elif criterion == "f1":
        denominator = 2.0 * tp + fp + fn
        score = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    else:
        odds = cost_ratio * thresholds / (1.0 - thresholds)
        score = (tp - fp * odds) / y.size
    best = int(np.argmax(score))
    return float(thresholds[best])

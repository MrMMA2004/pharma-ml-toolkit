"""Model & feature diagnostics: the yellowbrick concepts, re-derived.

The ``DistrictDataLabs/yellowbrick`` repository (Apache-2.0) was
reviewed file by file for this round -- ``features/rankd.py``,
``radviz.py``, ``pcoords.py``, ``classifier/threshold.py``,
``regressor/alphas.py`` and ``influence.py``, ``cluster/elbow.py``
(with its vendored kneedle), ``model_selection/validation_curve.py``
and ``importances.py``.  Yellowbrick is a *visualizer* library (fit →
draw → finalize); this toolkit already has its plotting twin in
:mod:`pharmaml.viz.ml_plots`, so the absorption lands as the
**computations** the visualizers wrap, each returning a typed record
the plot functions consume:

* :func:`feature_rank_1d` / :func:`feature_rank_2d` -- the feature
  ranking bars: the Shapiro-Wilk W per column (Rank1D's own
  "normality score") and the pairwise matrix (pearson / covariance /
  spearman / kendalltau, Rank2D's four algorithms).
* :func:`radviz_coordinates` -- the radial visualisation maths
  verbatim: per-column min-max normalisation, the m anchors on the
  unit circle at ``2πi/m``, and each point placed at the
  value-weighted average of the anchors.
* :func:`parallel_coordinates_data` -- the class-coloured lines with
  the five normalisations yellowbrick offers (none / minmax / maxabs
  / standard / l2 row-wise), subsampling included.
* :func:`class_balance_counts` / :func:`class_prediction_error_counts`
  -- the support bars and the true-x-predicted stacked-bar matrix.
* :func:`discrimination_threshold` -- the threshold sweep: n_trials
  shuffle splits, per-trial precision/recall/f-beta/queue-rate
  curves, the uniform threshold grid, and the 10/50/90 percent quantile
  bands with the optimal threshold at the argmax metric (yellowbrick's
  own defaults: 50 trials, f1, ``quantiles=(0.1, 0.5, 0.9)``).
* :func:`alpha_selection` -- the manual alpha path (ManualAlphaSelection):
  a log-spaced alpha grid, cross-validated error per alpha, the best.
* :func:`validation_curve_scores` -- sklearn's ``validation_curve``
  behind the toolkit's typed record (train/test mean ± std).
* :func:`cooks_distance` -- the influence diagnostic: the hat-matrix
  diagonal via the pseudo-inverse, studentized residuals,
  ``D_i = (r²/p)(h/(1-h))``, the F survival p-values and the ``4/n``
  threshold rule with the outlier percentage.
* :func:`distortion_score` / :func:`locate_elbow` -- the elbow
  metric (the *sum* of squared distances to recomputed centroids --
  yellowbrick's own formula, not the mean its docstring claims) and
  the kneedle difference-curve argmax.

The honest scope, stated once: yellowbrick's visualizer *class*
machinery (fit/draw/finalize, the Wrapper proxies, the rcParams
mutation on import) is deliberately not reproduced -- the toolkit's
``plot_*`` functions + records are the standing pattern; the text
visualizers (FreqDist, dispersion, POS), the missing-value
dispersion plots and the ``VisualizerGrid`` stay out.  The plot twins
live in :mod:`pharmaml.viz.ml_plots` (``plot_rank1d`` ...).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from pharmaml.ml.diagnostics import (
...     class_balance_counts, feature_rank_1d, feature_rank_2d,
...     locate_elbow, radviz_coordinates,
... )
>>> rng = np.random.default_rng(42)
>>> frame = pd.DataFrame({
...     "acid": rng.normal(0, 1.0, 90),
...     "base": rng.normal(0, 2.0, 90),
...     "salt": rng.uniform(0, 1, 90),
... })
>>> rank1 = feature_rank_1d(frame)
>>> rank1.features
['acid', 'base', 'salt']
>>> [round(float(value), 3) for value in rank1.scores]
[0.989, 0.98, 0.92]
>>> rank2 = feature_rank_2d(frame, algorithm="pearson")
>>> round(float(rank2.matrix[0, 0]), 6), round(float(rank2.matrix[0, 1]), 3)
(1.0, 0.144)
>>> target = (frame["acid"] + 0.3 * frame["base"] > 0).astype(int)
>>> radviz = radviz_coordinates(frame, target)
>>> radviz.anchors.round(3).tolist()
[[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]]
>>> balance = class_balance_counts([0, 0, 1, 1, 1, 1])
>>> balance.classes, balance.counts.tolist()
([0, 1], [2, 4])
>>> locate_elbow([2, 3, 4, 5, 6, 7, 8], [80.0, 40.0, 25.0, 18.0, 15.0, 13.5, 13.0])
4.0

>>> from pharmaml.ml.diagnostics import distortion_score
>>> tiny = np.array([[0.0, 0.0], [0.0, 0.1], [10.0, 10.0], [10.0, 10.1]])
>>> round(distortion_score(tiny, [0, 0, 1, 1]), 4)   # 4 x 0.05^2
0.01
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger
from pharmaml.ml.stats import RANDOM_STATE, _average_ranks

_LOG = get_logger(__name__)

#: Rank1D's single algorithm: the Shapiro-Wilk W statistic.
RANK1D_ALGORITHMS: tuple[str, ...] = ("shapiro",)

#: Rank2D's four pairwise metrics.
RANK2D_ALGORITHMS: tuple[str, ...] = ("pearson", "covariance", "spearman", "kendalltau")

#: The metrics the discrimination-threshold sweep tracks.
DISCRIMINATION_METRICS: tuple[str, ...] = ("precision", "recall", "fscore", "queue_rate")

#: The quantile bands (lower / median / upper) of the sweep.
DISCRIMINATION_QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)

#: Yellowbrick's default trial count for the threshold sweep.
DISCRIMINATION_TRIALS = 50

#: The validation split size of one threshold-sweep trial.
DISCRIMINATION_TEST_SIZE = 0.1

#: Parallel-coordinates normalisations (yellowbrick's own list).
PCOORDS_NORMALIZATIONS: tuple[str, ...] = (
    "none",
    "minmax",
    "maxabs",
    "standard",
    "l2",
)

#: ManualAlphaSelection's default alpha grid (yellowbrick's own).
ALPHA_GRID = tuple(float(value) for value in np.logspace(-10, -2, 200))

#: Cooks-distance influence threshold rule: ``4/n``.
COOKS_RULE = 4.0


def _as_feature_matrix(X: Any, caller: str) -> tuple[np.ndarray, list[str]]:
    """Validate a numeric 2-D input with column names preserved."""
    if isinstance(X, pd.DataFrame):
        names = [str(name) for name in X.columns]
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        names = [f"feature_{index}" for index in range(matrix.shape[1])]
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise DataValidationError(
            f"{caller}: needs a >= 3x2 numeric matrix, got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError(f"{caller}: the matrix must be finite (no NaN/inf)")
    return matrix, names


# ---------------------------------------------------------------------------
# Rank1D / Rank2D -- the feature ranking bars
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rank1DResult:
    """One Shapiro-Wilk W per feature (1.0 = perfectly normal)."""

    features: list[str]
    scores: np.ndarray

    def line(self) -> str:
        parts = [f"{name} {score:.3f}" for name, score in zip(self.features, self.scores, strict=True)]
        return "shapiro ranking · " + " · ".join(parts)


def feature_rank_1d(X: Any) -> Rank1DResult:
    """Rank1D's own score: the Shapiro-Wilk W statistic per column.

    W in [0, 1]; values near 1 mean the feature is compatible with a
    normal shape -- the ranking the horizontal bars show.

    Raises
    ------
    DataValidationError
        On a bad matrix or a column outside Shapiro's 3..5000 window.
    """
    from scipy import stats as scipy_stats

    matrix, names = _as_feature_matrix(X, caller="feature_rank_1d")
    n = matrix.shape[0]
    if not 3 <= n <= 5000:
        raise DataValidationError(
            f"feature_rank_1d: Shapiro-Wilk needs 3 <= n <= 5000, got n={n}"
        )
    scores = np.array(
        [float(scipy_stats.shapiro(matrix[:, column]).statistic) for column in range(matrix.shape[1])]
    )
    return Rank1DResult(features=names, scores=scores)


@dataclass(frozen=True)
class Rank2DResult:
    """The pairwise feature matrix (correlations or covariance)."""

    features: list[str]
    matrix: np.ndarray
    algorithm: str

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.features, columns=self.features)

    def line(self) -> str:
        strongest = np.unravel_index(np.abs(np.triu(self.matrix, 1)).argmax(), self.matrix.shape)
        value = self.matrix[strongest]
        return (
            f"{self.algorithm} ranking · {len(self.features)} features · "
            f"strongest pair {self.features[strongest[0]]!r}/{self.features[strongest[1]]!r} "
            f"= {value:+.3f}"
        )


def feature_rank_2d(X: Any, *, algorithm: str = "pearson") -> Rank2DResult:
    """Rank2D's pairwise matrix: pearson / covariance / spearman / kendalltau.

    Raises
    ------
    DataValidationError
        On a bad matrix or an unknown algorithm.
    """
    if algorithm not in RANK2D_ALGORITHMS:
        raise DataValidationError(
            f"feature_rank_2d: algorithm must be one of {RANK2D_ALGORITHMS}, got {algorithm!r}"
        )
    matrix, names = _as_feature_matrix(X, caller="feature_rank_2d")
    if algorithm == "pearson":
        result = np.corrcoef(matrix, rowvar=False)
    elif algorithm == "covariance":
        result = np.cov(matrix, rowvar=False)
    elif algorithm == "spearman":
        ranks = np.column_stack([_average_ranks(matrix[:, column]) for column in range(matrix.shape[1])])
        result = np.corrcoef(ranks, rowvar=False)
    else:  # kendalltau
        size = matrix.shape[1]
        result = np.eye(size)
        for i in range(size):
            for j in range(i + 1, size):
                result[i, j] = result[j, i] = _kendall_tau(matrix[:, i], matrix[:, j])
    return Rank2DResult(features=names, matrix=np.asarray(result, dtype=float), algorithm=algorithm)


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-b with tie corrections on both sides."""
    n = a.size
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = a[i] - a[j]
            dy = b[i] - b[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_a += 1
            elif dy == 0:
                ties_b += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (0.5 * n * (n - 1) - ties_a) * (0.5 * n * (n - 1) - ties_b)
    )
    return float((concordant - discordant) / denominator) if denominator > 0 else float("nan")


# ---------------------------------------------------------------------------
# RadViz -- the radial visualisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadVizResult:
    """The RadViz placement: anchors, normalised points, class labels."""

    features: list[str]
    classes: list[Any]
    anchors: np.ndarray
    points: np.ndarray
    labels: np.ndarray

    def line(self) -> str:
        radius = float(np.linalg.norm(self.points, axis=1).max())
        return f"radviz · {len(self.features)} anchors · {len(self.classes)} classes · max radius {radius:.3f}"


def radviz_coordinates(X: Any, y: Any) -> RadVizResult:
    """The radial visualisation placement, yellowbrick's maths verbatim.

    Per-column min-max normalisation; the m feature anchors on the
    unit circle at angles ``2πi/m``; every point placed at
    ``Σ_j (x_j · anchor_j) / Σ_j x_j`` -- the spring analogy.

    Raises
    ------
    DataValidationError
        On a bad matrix, length mismatch, or a degenerate (constant)
        column.
    """
    matrix, names = _as_feature_matrix(X, caller="radviz_coordinates")
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise DataValidationError(
            f"radviz_coordinates: X has {matrix.shape[0]} rows but y has {labels.shape[0] if labels.ndim else 0} values"
        )
    lows = matrix.min(axis=0)
    spans = matrix.max(axis=0) - lows
    if (spans <= 0).any():
        raise DataValidationError(
            "radviz_coordinates: a constant column cannot be min-max normalised"
        )
    normalised = (matrix - lows) / spans
    m = normalised.shape[1]
    angles = 2.0 * math.pi * np.arange(m) / m
    anchors = np.column_stack([np.cos(angles), np.sin(angles)])
    totals = normalised.sum(axis=1)
    safe = np.where(totals > 0, totals, 1.0)
    points = (normalised @ anchors) / safe[:, None]
    classes = list(pd.unique(pd.Series(labels)))
    return RadVizResult(
        features=names,
        classes=classes,
        anchors=anchors,
        points=points,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Parallel coordinates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelCoordsResult:
    """The class-coloured polylines plus their normalisation."""

    features: list[str]
    classes: list[Any]
    data: np.ndarray
    labels: np.ndarray
    normalize: str
    sample_fraction: float

    def line(self) -> str:
        return (
            f"parallel coordinates · {len(self.features)} axes · {len(self.classes)} "
            f"classes · normalize={self.normalize} · {self.data.shape[0]} lines"
        )


def parallel_coordinates_data(
    X: Any,
    y: Any,
    *,
    normalize: str = "none",
    sample: float = 1.0,
    shuffle: bool = False,
    seed: int = RANDOM_STATE,
) -> ParallelCoordsResult:
    """The parallel-coordinates lines, yellowbrick's contract.

    ``normalize`` picks the per-column transform (``'none'`` /
    ``'minmax'`` / ``'maxabs'`` / ``'standard'`` / ``'l2'`` row-wise);
    ``sample`` is an int cap or a float fraction; ``shuffle`` mixes
    which rows survive the sample.

    Raises
    ------
    DataValidationError
        On a bad matrix, a length mismatch, or an unknown normalisation.
    """
    if normalize not in PCOORDS_NORMALIZATIONS:
        raise DataValidationError(
            f"parallel_coordinates_data: normalize must be one of {PCOORDS_NORMALIZATIONS}, got {normalize!r}"
        )
    matrix, names = _as_feature_matrix(X, caller="parallel_coordinates_data")
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise DataValidationError("parallel_coordinates_data: X and y must share rows")
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    if isinstance(sample, bool) or not 0 < sample <= n:
        raise DataValidationError(
            f"parallel_coordinates_data: sample must be in (0, {n}] (a count or a fraction), got {sample!r}"
        )
    limit = int(sample) if sample >= 1 else max(1, int(sample * n))
    limit = min(limit, n)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    indices = np.sort(indices[:limit])
    matrix = matrix[indices]
    labels = labels[indices]
    if normalize == "minmax":
        lows = matrix.min(axis=0)
        spans = matrix.max(axis=0) - lows
        matrix = (matrix - lows) / np.where(spans > 0, spans, 1.0)
    elif normalize == "maxabs":
        scale = np.abs(matrix).max(axis=0)
        matrix = matrix / np.where(scale > 0, scale, 1.0)
    elif normalize == "standard":
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0, ddof=0)
        matrix = (matrix - means) / np.where(stds > 0, stds, 1.0)
    elif normalize == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms > 0, norms, 1.0)
    classes = list(pd.unique(pd.Series(labels)))
    return ParallelCoordsResult(
        features=names,
        classes=classes,
        data=matrix,
        labels=labels,
        normalize=normalize,
        sample_fraction=limit / n,
    )


# ---------------------------------------------------------------------------
# Class balance / class prediction error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassBalanceResult:
    """Per-class support, optionally train vs test."""

    classes: list[Any]
    counts: np.ndarray
    counts2: np.ndarray | None = None

    def line(self) -> str:
        total = int(self.counts.sum())
        if self.counts2 is not None:
            return (
                f"class balance · {total:,} instances · "
                + " · ".join(
                    f"{name!r} {int(first)}/{int(second)}"
                    for name, first, second in zip(
                        self.classes, self.counts, self.counts2, strict=True
                    )
                )
            )
        return (
            f"class balance · {total:,} instances · "
            + " · ".join(f"{name!r} {int(count)}" for name, count in zip(self.classes, self.counts, strict=True))
        )


def _plain(values: Any) -> list[Any]:
    """numpy scalars to plain Python ones (records read better in JSON/GUI)."""
    return [value.item() if hasattr(value, "item") else value for value in values]


def class_balance_counts(y: Any, y2: Any | None = None) -> ClassBalanceResult:
    """The support bars (balance mode, or the train/test compare mode).

    Raises
    ------
    DataValidationError
        On empty inputs or mismatched class sets in compare mode.
    """
    labels = np.asarray(y).ravel()
    if labels.size == 0:
        raise DataValidationError("class_balance_counts: y is empty")
    classes, counts = np.unique(labels, return_counts=True)
    if y2 is None:
        return ClassBalanceResult(classes=_plain(classes), counts=counts.astype(int))
    labels2 = np.asarray(y2).ravel()
    if labels2.size == 0:
        raise DataValidationError("class_balance_counts: y2 is empty")
    classes2, counts2 = np.unique(labels2, return_counts=True)
    if list(classes) != list(classes2):
        raise DataValidationError(
            "class_balance_counts: the train/test class sets differ "
            f"({list(classes)} vs {list(classes2)}) -- yellowbrick's compare mode needs both sides labelled alike"
        )
    return ClassBalanceResult(
        classes=_plain(classes), counts=counts.astype(int), counts2=counts2.astype(int)
    )


@dataclass(frozen=True)
class ClassPredictionErrorResult:
    """Rows = true class, columns = predicted class, values = counts."""

    classes: list[Any]
    matrix: np.ndarray

    def line(self) -> str:
        diagonal = float(np.trace(self.matrix))
        total = float(self.matrix.sum())
        accuracy = diagonal / total if total else 0.0
        return (
            f"class prediction error · {int(total)} predictions · "
            f"{accuracy:.1%} on the diagonal"
        )


def class_prediction_error_counts(y_true: Any, y_pred: Any) -> ClassPredictionErrorResult:
    """The stacked-bar matrix behind ClassPredictionError.

    Raises
    ------
    DataValidationError
        On length mismatches or disjoint label sets.
    """
    truth = np.asarray(y_true).ravel()
    predicted = np.asarray(y_pred).ravel()
    if truth.shape != predicted.shape:
        raise DataValidationError(
            f"class_prediction_error_counts: y_true {truth.shape} and y_pred {predicted.shape} must match"
        )
    classes = sorted(set(truth.tolist()) | set(predicted.tolist()), key=str)
    lookup = {name: index for index, name in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, forecast in zip(truth, predicted, strict=True):
        matrix[lookup[actual], lookup[forecast]] += 1
    return ClassPredictionErrorResult(classes=classes, matrix=matrix)


# ---------------------------------------------------------------------------
# Discrimination threshold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscriminationThresholdResult:
    """The threshold sweep: quantile bands per metric + the optimum."""

    thresholds: np.ndarray
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    optimal_threshold: float
    argmax_metric: str
    n_trials: int
    fbeta: float

    def line(self) -> str:
        return (
            f"discrimination threshold · {self.n_trials} trials · "
            f"optimal {self.argmax_metric} at t={self.optimal_threshold:.3f}"
        )


def discrimination_threshold(
    X: Any,
    y: Any,
    *,
    estimator: Any | None = None,
    n_trials: int = DISCRIMINATION_TRIALS,
    fbeta: float = 1.0,
    quantiles: tuple[float, ...] = DISCRIMINATION_QUANTILES,
    random_state: int = RANDOM_STATE,
) -> DiscriminationThresholdResult:
    """The discrimination-threshold sweep (binary targets only).

    Per trial: a 10% shuffle split, a fresh estimator fit, the score
    column (``predict_proba``[:, 1] or ``decision_function``), then
    precision / recall / f-beta / queue-rate curves.  The trials share
    a uniform threshold grid ``linspace(0, 1, m)`` (m = the shortest
    per-trial threshold count, yellowbrick's own data-dependent rule),
    and each metric aggregates into the (lower, median, upper) quantile
    bands.  The optimum is the argmax metric's median peak.

    Raises
    ------
    DataValidationError
        On a non-binary target, a bad trial count, or bad quantiles.
    ModelError
        When the estimator exposes neither predict_proba nor
        decision_function.
    """
    if not 1 <= int(n_trials) <= 500:
        raise DataValidationError(f"n_trials must be in [1, 500], got {n_trials}")
    if len(quantiles) != 3 or not all(0.0 < q < 1.0 for q in quantiles):
        raise DataValidationError(
            f"quantiles must be three values in (0, 1), got {quantiles!r}"
        )
    if fbeta <= 0:
        raise DataValidationError(f"fbeta must be positive, got {fbeta}")
    matrix, _names = _as_feature_matrix(X, caller="discrimination_threshold")
    labels = np.asarray(y).ravel()
    if labels.shape[0] != matrix.shape[0]:
        raise DataValidationError("discrimination_threshold: X and y must share rows")
    unique = np.unique(labels)
    if unique.size != 2:
        raise DataValidationError(
            f"discrimination_threshold: needs a binary target, got {unique.size} classes"
        )
    positive = unique[1]
    binary = (labels == positive).astype(int)
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import ShuffleSplit

    base = (
        LogisticRegression(max_iter=1000, random_state=random_state)
        if estimator is None
        else clone(estimator)
    )
    threshold_counts: list[int] = []
    per_trial: list[dict[str, np.ndarray]] = []
    for trial in range(int(n_trials)):
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=DISCRIMINATION_TEST_SIZE,
            random_state=random_state + trial,
        )
        train_index, test_index = next(splitter.split(matrix, binary))
        model = clone(base)
        model.fit(matrix[train_index], binary[train_index])
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(matrix[test_index])[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(matrix[test_index])
        else:
            raise ModelError(
                "discrimination_threshold: the estimator exposes neither "
                "predict_proba nor decision_function"
            )
        precision, recall, thresholds = precision_recall_curve(binary[test_index], scores)
        beta_sq = fbeta**2
        with np.errstate(divide="ignore", invalid="ignore"):
            fscore = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        fscore = np.nan_to_num(fscore)
        thresholds = np.append(thresholds, 1.0)
        queue = np.array([float((scores >= threshold).mean()) for threshold in thresholds])
        per_trial.append(
            {
                "thresholds": thresholds,
                "precision": precision,
                "recall": recall,
                "fscore": fscore,
                "queue_rate": queue,
            }
        )
        threshold_counts.append(int(thresholds.size))
    grid_size = min(threshold_counts)
    grid = np.linspace(0.0, 1.0, grid_size)
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for metric in DISCRIMINATION_METRICS:
        values = np.vstack(
            [np.interp(grid, trial["thresholds"], trial[metric]) for trial in per_trial]
        )
        bands = tuple(
            np.quantile(values, q, axis=0) for q in quantiles
        )
        curves[metric] = (  # type: ignore[assignment]
            np.asarray(bands[0]),
            np.asarray(bands[1]),
            np.asarray(bands[2]),
        )
    median_f = curves["fscore"][1]
    optimal = float(grid[int(np.argmax(median_f))])
    _LOG.info(
        "discrimination threshold: %d trials, optimal t=%.3f", int(n_trials), optimal
    )
    return DiscriminationThresholdResult(
        thresholds=grid,
        curves=curves,
        optimal_threshold=optimal,
        argmax_metric="fscore",
        n_trials=int(n_trials),
        fbeta=float(fbeta),
    )


# ---------------------------------------------------------------------------
# Alpha selection / validation curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaSelectionResult:
    """The alpha path: grid, CV error per alpha, the best."""

    alphas: np.ndarray
    errors: np.ndarray
    best_alpha: float
    best_error: float
    model: str

    def line(self) -> str:
        return (
            f"alpha selection ({self.model}) · {len(self.alphas)} alphas · "
            f"best α {self.best_alpha:.4g} (CV error {self.best_error:.4g})"
        )


def alpha_selection(
    X: Any,
    y: Any,
    *,
    alphas: Any | None = None,
    model: str = "ridge",
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> AlphaSelectionResult:
    """The ManualAlphaSelection path: CV error per alpha, the best.

    The default grid is yellowbrick's own ``logspace(-10, -2, 200)``
    (:data:`ALPHA_GRID`); ``model`` picks ``'ridge'`` or ``'lasso'``.

    Raises
    ------
    DataValidationError
        On a bad matrix, an unknown model, or a bad cv count.
    """
    if model not in ("ridge", "lasso"):
        raise DataValidationError(f"model must be 'ridge' or 'lasso', got {model!r}")
    if not 2 <= int(cv) <= 20:
        raise DataValidationError(f"cv must be in [2, 20], got {cv}")
    matrix, _names = _as_feature_matrix(X, caller="alpha_selection")
    target = np.asarray(y, dtype=float).ravel()
    if target.shape[0] != matrix.shape[0]:
        raise DataValidationError("alpha_selection: X and y must share rows")
    grid = (
        np.asarray(alphas, dtype=float)
        if alphas is not None
        else np.asarray(ALPHA_GRID)
    )
    if grid.ndim != 1 or grid.size < 2 or (grid <= 0).any():
        raise DataValidationError(
            f"alpha_selection: alphas must be >= 2 positive values, got shape {grid.shape}"
        )
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.model_selection import cross_val_score

    errors = np.array(
        [
            -float(
                cross_val_score(
                    (Ridge if model == "ridge" else Lasso)(alpha=float(alpha), random_state=random_state),
                    matrix,
                    target,
                    cv=int(cv),
                    scoring="neg_mean_squared_error",
                ).mean()
            )
            for alpha in grid
        ]
    )
    best = int(np.argmin(errors))
    return AlphaSelectionResult(
        alphas=grid,
        errors=errors,
        best_alpha=float(grid[best]),
        best_error=float(errors[best]),
        model=model,
    )


@dataclass(frozen=True)
class ValidationCurveResult:
    """The train/test score bands across a parameter's range."""

    param_name: str
    param_range: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    test_mean: np.ndarray
    test_std: np.ndarray
    scoring: str

    def line(self) -> str:
        best = int(np.argmax(self.test_mean))
        return (
            f"validation curve · {self.param_name} · best {self.param_name}="
            f"{self.param_range[best]:.4g} (CV {self.scoring} {self.test_mean[best]:.4f})"
        )


def validation_curve_scores(
    estimator: Any,
    X: Any,
    y: Any,
    param_name: str,
    param_range: Any,
    *,
    cv: int = 5,
    scoring: str | None = None,
    random_state: int = RANDOM_STATE,
) -> ValidationCurveResult:
    """sklearn's ``validation_curve`` behind the typed record.

    Train and cross-validation scores across the parameter range,
    mean ± std over the folds -- the two band curves the plot twin
    draws.

    Raises
    ------
    DataValidationError
        On a bad matrix, a bad range, or a missing parameter.
    """
    matrix, _names = _as_feature_matrix(X, caller="validation_curve_scores")
    target = np.asarray(y)
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise DataValidationError("validation_curve_scores: X and y must share rows")
    grid = np.asarray(param_range, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise DataValidationError(
            f"validation_curve_scores: param_range needs >= 2 values, got shape {grid.shape}"
        )
    if not hasattr(estimator, "get_params") or param_name not in estimator.get_params(deep=True):
        raise DataValidationError(
            f"validation_curve_scores: the estimator has no parameter {param_name!r}"
        )
    from sklearn.model_selection import validation_curve

    used_scoring = scoring if scoring is not None else (
        "accuracy" if _looks_like_classes(target) else "r2"
    )
    train_scores, test_scores = validation_curve(
        estimator,
        matrix,
        target,
        param_name=param_name,
        param_range=grid,
        cv=int(cv),
        scoring=used_scoring,
    )
    return ValidationCurveResult(
        param_name=str(param_name),
        param_range=grid,
        train_mean=train_scores.mean(axis=1),
        train_std=train_scores.std(axis=1),
        test_mean=test_scores.mean(axis=1),
        test_std=test_scores.std(axis=1),
        scoring=used_scoring,
    )


def _looks_like_classes(target: np.ndarray) -> bool:
    """A crude classifier/regressor sniff: few integer-ish distinct values."""
    unique = np.unique(target)
    return bool(unique.size <= 20) and bool(np.all(np.isclose(unique, np.round(unique))))


# ---------------------------------------------------------------------------
# Cook's distance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooksDistanceResult:
    """Per-observation influence with the 4/n threshold verdict."""

    distances: np.ndarray
    p_values: np.ndarray
    leverage: np.ndarray
    threshold: float
    outlier_pct: float

    def line(self) -> str:
        flagged = int((self.distances > self.threshold).sum())
        return (
            f"cook's distance · {self.distances.size} observations · "
            f"I_t = 4/n = {self.threshold:.3f} · {flagged} flagged ({self.outlier_pct:.1f}%)"
        )


def cooks_distance(X: Any, y: Any) -> CooksDistanceResult:
    """Cook's distance for an OLS design, yellowbrick's chain.

    Leverage ``h_i`` from the hat-matrix diagonal (the pseudo-inverse
    product, taken on the raw matrix exactly as yellowbrick does),
    studentized residuals over the rank-corrected MSE,
    ``D_i = (r_stud²/p)·(h_i/(1−h_i))``, the F survival p-values and
    the ``4/n`` threshold with the flagged percentage.  One
    deliberate deviation, stated once: upstream fits its residuals
    with an intercept while computing leverage without one -- here
    residuals and leverage come from the same raw design, so the
    influence arithmetic is internally consistent.

    Raises
    ------
    DataValidationError
        On a bad matrix or a rank-deficient design.
    """
    matrix, _names = _as_feature_matrix(X, caller="cooks_distance")
    target = np.asarray(y, dtype=float).ravel()
    if target.shape[0] != matrix.shape[0]:
        raise DataValidationError("cooks_distance: X and y must share rows")
    n, p = matrix.shape
    pseudo = np.linalg.pinv(matrix)
    leverage = (matrix * pseudo.T).sum(axis=1)
    if (leverage >= 1.0).any():
        raise DataValidationError(
            "cooks_distance: a leverage of >= 1 (duplicate/overspecified design) breaks the influence ratio"
        )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
    residuals = target - matrix @ coefficients
    dof = n - int(rank)
    if dof <= 0:
        raise DataValidationError("cooks_distance: the design consumes every degree of freedom")
    mse = float(residuals @ residuals) / dof
    if mse <= 0:
        raise DataValidationError("cooks_distance: zero residual variance")
    studentized = residuals / np.sqrt(mse * (1.0 - leverage))
    distances = (studentized**2 / p) * (leverage / (1.0 - leverage))
    from scipy import stats as scipy_stats

    p_values = scipy_stats.f.sf(distances, p, dof)
    threshold = COOKS_RULE / n
    outlier_pct = 100.0 * float((distances > threshold).mean())
    return CooksDistanceResult(
        distances=distances,
        p_values=np.asarray(p_values, dtype=float),
        leverage=leverage,
        threshold=float(threshold),
        outlier_pct=float(outlier_pct),
    )


# ---------------------------------------------------------------------------
# The elbow (kneedle) + distortion
# ---------------------------------------------------------------------------


def distortion_score(X: Any, labels: Any) -> float:
    """The KElbowVisualizer's distortion: the SUM of squared distances.

    Per yellowbrick's code (not its docstring): each cluster's
    centroid is recomputed as the mean, and every member's squared
    euclidean distance is summed -- the total, not the mean.

    Raises
    ------
    DataValidationError
        On a bad matrix or an empty cluster.
    """
    matrix, _names = _as_feature_matrix(X, caller="distortion_score")
    labels_flat = np.asarray(labels).ravel()
    if labels_flat.shape[0] != matrix.shape[0]:
        raise DataValidationError("distortion_score: X and labels must share rows")
    total = 0.0
    for label in np.unique(labels_flat):
        members = matrix[labels_flat == label]
        if members.shape[0] == 0:
            raise DataValidationError(f"distortion_score: cluster {label!r} is empty")
        centroid = members.mean(axis=0)
        total += float(((members - centroid) ** 2).sum())
    return total


def locate_elbow(x_values: Any, y_values: Any, *, curve: str = "decreasing") -> float:
    """The kneedle elbow: the difference-curve argmax.

    Both coordinates are min-max normalised; the difference
    ``y_norm − x_norm`` peaks at the knee for the two curve shapes the
    elbow visualizer sweeps -- convex decreasing (distortion) and
    concave increasing (silhouette / Calinski-Harabasz).  This is the
    vendored ``KneeLocator``'s core without its local-extrema scan.

    Raises
    ------
    DataValidationError
        On length mismatch, fewer than three points, or a bad curve.
    """
    if curve not in ("decreasing", "increasing"):
        raise DataValidationError(
            f"locate_elbow: curve must be 'decreasing' or 'increasing', got {curve!r}"
        )
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise DataValidationError(
            f"locate_elbow: x and y must be equal-length 1-D arrays, got {xs.shape} vs {ys.shape}"
        )
    if xs.size < 3:
        raise DataValidationError(f"locate_elbow: needs >= 3 points, got {xs.size}")
    x_norm = (xs - xs.min()) / (xs.max() - xs.min())
    y_norm = (ys - ys.min()) / (ys.max() - ys.min())
    # kneedle's own reduction: a convex-decreasing curve flips its y axis
    # first, so both shapes peak the difference curve at the knee.
    flipped = 1.0 - y_norm if curve == "decreasing" else y_norm
    difference = flipped - x_norm
    return float(xs[int(np.argmax(difference))])


__all__ = [
    "ALPHA_GRID",
    "COOKS_RULE",
    "DISCRIMINATION_METRICS",
    "DISCRIMINATION_QUANTILES",
    "DISCRIMINATION_TEST_SIZE",
    "DISCRIMINATION_TRIALS",
    "PCOORDS_NORMALIZATIONS",
    "RANK1D_ALGORITHMS",
    "RANK2D_ALGORITHMS",
    "AlphaSelectionResult",
    "ClassBalanceResult",
    "ClassPredictionErrorResult",
    "CooksDistanceResult",
    "DiscriminationThresholdResult",
    "ParallelCoordsResult",
    "RadVizResult",
    "Rank1DResult",
    "Rank2DResult",
    "ValidationCurveResult",
    "alpha_selection",
    "class_balance_counts",
    "class_prediction_error_counts",
    "cooks_distance",
    "discrimination_threshold",
    "distortion_score",
    "feature_rank_1d",
    "feature_rank_2d",
    "locate_elbow",
    "parallel_coordinates_data",
    "radviz_coordinates",
    "validation_curve_scores",
]

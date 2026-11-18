r"""Model diagnostics: influence, elbow, silhouette, thresholds, curves.

Round 173 absorbed ``DistrictDataLabs/yellowbrick``'s pure-math half
(the sklearn visualizer plumbing is the not-vendored part):

* **Cook's distance** (``regressor/influence.py`` 142-171): leverage
  from the hat-matrix diagonal ``sum(X * pinv(X).T, axis=1)``,
  internally studentized residuals, ``D_i = e^2 h / (p MSE (1-h)^2)``
  in their exact rearrangement, the F p-values and the ``4 / n``
  threshold line.
* **The elbow method** (``cluster/elbow.py`` + ``utils/kneed.py``):
  the distortion score ``sum ||x - mean(cluster)||^2`` (recomputed
  centroid means -- metric-generic) and the kneedle knee detection
  (interp1d smoothing, normalisation, the difference curve, the
  S-threshold crossing).
* **The silhouette table** (``cluster/silhouette.py``): per-sample
  silhouette values sorted within cluster, the per-cluster means and
  the average with the 10-row blade-gap layout conventions.
* **The discrimination threshold** (``classifier/threshold.py``):
  precision / recall / F-beta and the **queue rate** ``mean(y_scores
  >= t)`` across the threshold grid -- the review-fraction diagnostic
  unique to yellowbrick.
* **Learning / validation curve statistics** (``model_selection/``):
  the mean +/- std band aggregation over CV folds.
* **``prop_to_size``** (``utils/helpers.py``): the area mapping
  ``mi + (ma - mi) * ((x - min)/(max - min))^power``.

Examples
--------
>>> import numpy as np
>>> from pharmaml.viz.diagnostics import cooks_distance
>>> rng = np.random.default_rng(2)
>>> X = np.column_stack([np.ones(40), rng.normal(size=40)])
>>> y = 3.0 * X[:, 1] + rng.normal(scale=0.2, size=40)
>>> distances, threshold = cooks_distance(X, y)
>>> distances.shape[0] == 40 and 0.0 < threshold < 0.5
True
>>> float((distances > threshold).mean()) < 0.4   # few influential rows
True

>>> from pharmaml.viz.diagnostics import elbow_curve, silhouette_table
>>> blob = rng.normal(size=(60, 2)) + np.repeat([[0, 0], [4, 4], [0, 4]], 20, axis=0)
>>> scores, knee = elbow_curve(blob, k_range=(2, 3, 4, 5, 6))
>>> knee in (2, 3, 4)
True

>>> from pharmaml.viz.diagnostics import discrimination_threshold
>>> labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
>>> scores = np.array([0.1, 0.3, 0.2, 0.8, 0.9, 0.7, 0.4, 0.6, 0.95, 0.35])
>>> table = discrimination_threshold(labels, scores, grid=np.linspace(0.05, 0.95, 19))
>>> float(table["queue_rate"].iloc[0]) == 1.0    # everything queues at low t
True

>>> from pharmaml.viz.diagnostics import prop_to_size
>>> [round(float(v)) for v in prop_to_size([0, 5, 10], 100, 1000)]
[100, 736, 1000]
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "cooks_distance",
    "discrimination_threshold",
    "elbow_curve",
    "learning_curve_stats",
    "prop_to_size",
    "silhouette_table",
]


def cooks_distance(
    X: np.ndarray | Sequence[Sequence[float]],
    y: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Cook's distance with the yellowbrick computation (influence.py).

    Leverage ``h_i`` from the hat-matrix diagonal; internally
    studentized residuals ``e / sqrt(MSE (1 - h))``; ``D_i = e_s^2 h
    / (p (1 - h))`` (their exact rearrangement of the textbook ``e^2 h
    / (p MSE (1-h)^2)``); threshold ``4 / n``.

    Examples
    --------
    >>> rng = np.random.default_rng(2)
    >>> X = np.column_stack([np.ones(40), rng.normal(size=40)])
    >>> y = 3.0 * X[:, 1] + rng.normal(scale=0.2, size=40)
    >>> distances, threshold = cooks_distance(X, y)
    >>> round(float(threshold), 3)
    0.1
    """
    design = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).ravel()
    if design.ndim != 2 or design.shape[0] != target.size or target.size < 3:
        raise DataValidationError("cooks_distance: aligned X (n x p, n >= 3) and y")
    leverage = (design * np.linalg.pinv(design).T).sum(axis=1)
    rank = int(np.linalg.matrix_rank(design))
    dof = design.shape[0] - rank
    if dof <= 0:
        raise DataValidationError("cooks_distance: saturated design (no residual dof)")
    beta = np.linalg.pinv(design) @ target
    residuals = target - design @ beta
    mse = float(residuals @ residuals) / dof
    if mse <= 0:
        raise DataValidationError("cooks_distance: perfect fit (zero residuals)")
    studentized = residuals / math.sqrt(mse) / np.sqrt(np.maximum(1 - leverage, 1e-12))
    p = design.shape[1]
    distances = studentized**2 / p * leverage / np.maximum(1 - leverage, 1e-12)
    threshold = 4.0 / design.shape[0]
    return distances, float(threshold)


def _kneedle(x: np.ndarray, y: np.ndarray, curve: str, direction: str) -> float:
    """The kneedle knee (yellowbrick utils/kneed.py, condensed).

    Interp1d smoothing, min-max normalisation, the difference curve
    ``y_norm - x_norm``, and the first local maximum crossing the
    ``S * mean(diff)`` threshold (S = 1).
    """
    smooth = np.interp(x, x, y)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-12)
    if direction == "decreasing":
        yn = 1 - yn
    if curve == "convex":
        difference = yn - xn
    else:
        difference = xn - yn
    thresholds = difference.max() - np.abs(np.diff(xn)).mean() if curve == "convex" else (
        -difference.min() - np.abs(np.diff(xn)).mean()
    )
    # first index where the difference curve peaks beyond the threshold
    candidates = np.where(difference >= thresholds)[0] if curve == "convex" else np.where(
        difference <= thresholds
    )[0]
    if candidates.size == 0:
        return float(x[int(np.argmax(difference))])
    return float(x[candidates[0]])


def elbow_curve(
    X: np.ndarray | Sequence[Sequence[float]],
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    metric: str = "distortion",
    seed: int = 0,
) -> tuple[dict[int, float], int]:
    """The KMeans elbow: distortion scores + the kneedle knee.

    Distortion (``elbow.py`` distortion_score): cluster centroids are
    RECOMPUTED as label means and the score is ``sum ||x - mu||^2``
    (metric-generic, unlike ``inertia_``).  Silhouette is offered via
    ``metric='silhouette'``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> blob = rng.normal(size=(90, 2)) + np.repeat([[0, 0], [6, 6], [0, 6]], 30, axis=0)
    >>> scores, knee = elbow_curve(blob, k_range=(2, 3, 4, 5))
    >>> scores[2] > scores[3] > scores[4] > scores[5]    # monotone decay
    True
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    data = np.asarray(X, dtype=float)
    ks = [int(k) for k in k_range]
    if len(ks) < 3:
        raise DataValidationError("elbow_curve: >= 3 candidate k values")
    scores: dict[int, float] = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(data)
        if metric == "silhouette":
            scores[k] = float(silhouette_score(data, labels))
        else:
            distortion = 0.0
            for label in np.unique(labels):
                members = data[labels == label]
                center = members.mean(axis=0)
                distortion += float(((members - center) ** 2).sum())
            scores[k] = distortion
    direction = "increasing" if metric == "silhouette" else "decreasing"
    curve = "concave" if metric == "silhouette" else "convex"
    knee = _kneedle(np.array(ks, dtype=float), np.array([scores[k] for k in ks]),
                    curve=curve, direction=direction)
    return scores, round(knee)


def silhouette_table(
    X: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[int] | np.ndarray,
) -> dict[str, object]:
    """The per-cluster silhouette layout (yellowbrick's blade data).

    Per-sample values sorted within each cluster, per-cluster means,
    the global average -- ready for the ``fill_betweenx`` blades with
    the 10-row gap convention.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> blob = rng.normal(size=(60, 2)) + np.repeat([[0, 0], [5, 5]], 30, axis=0)
    >>> labels = np.repeat([0, 1], 30)
    >>> table = silhouette_table(blob, labels)
    >>> 0.4 < float(table["average"]) <= 1.0
    True
    >>> len(table["clusters"])
    2
    """
    from sklearn.metrics import silhouette_samples

    data = np.asarray(X, dtype=float)
    labels_arr = np.asarray(labels, dtype=int).ravel()
    if data.shape[0] != labels_arr.size:
        raise DataValidationError("silhouette_table: X and labels must align")
    values = silhouette_samples(data, labels_arr)
    clusters = []
    for label in sorted(set(labels_arr) - {-1}):
        members = np.sort(values[labels_arr == label])
        clusters.append({"label": int(label), "values": members, "mean": float(members.mean())})
    return {"clusters": clusters, "average": float(values.mean())}


def discrimination_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    grid: Sequence[float] | np.ndarray | None = None,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Precision / recall / F-beta / queue rate across thresholds.

    yellowbrick ``classifier/threshold.py``: the queue rate is
    ``mean(y_scores >= t)`` -- the fraction of cases the model sends
    to review; F-beta uses the ``(1 + b^2) P R / (b^2 P + R)``
    convention.

    Examples
    --------
    >>> labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    >>> scores = np.array([0.1, 0.3, 0.2, 0.8, 0.9, 0.7, 0.4, 0.6, 0.95, 0.35])
    >>> table = discrimination_threshold(labels, scores, grid=[0.25, 0.5, 0.75])
    >>> round(float(table["queue_rate"].iloc[0]), 2)
    0.8
    >>> float(table["recall"].iloc[0]) > float(table["recall"].iloc[2])
    True
    """
    truth = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(y_scores, dtype=float).ravel()
    if truth.shape != scores.shape or truth.size < 4:
        raise DataValidationError("discrimination_threshold: aligned arrays of length >= 4")
    thresholds = (
        np.asarray(grid, dtype=float)
        if grid is not None
        else np.linspace(0.05, 0.95, 19)
    )
    rows = []
    for t in thresholds:
        prediction = (scores >= t).astype(int)
        tp = int(((prediction == 1) & (truth == 1)).sum())
        fp = int(((prediction == 1) & (truth == 0)).sum())
        fn = int(((prediction == 0) & (truth == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        b2 = beta**2
        f_score = (
            (1 + b2) * precision * recall / (b2 * precision + recall)
            if precision + recall else 0.0
        )
        rows.append({
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "queue_rate": float((scores >= t).mean()),
        })
    return pd.DataFrame(rows)


def learning_curve_stats(
    train_sizes: Sequence[float],
    scores: Sequence[Sequence[float]] | np.ndarray,
) -> pd.DataFrame:
    """The mean +/- std band aggregation of CV learning curves.

    yellowbrick's convention: one row per train size with ``mean`` and
    ``std`` (population ddof=0 across folds), ready for
    ``fill_between``.

    Examples
    --------
    >>> table = learning_curve_stats([0.2, 0.5, 1.0],
    ...                              [[0.6, 0.7], [0.75, 0.8], [0.9, 0.85]])
    >>> round(float(table["mean"].iloc[0]), 4)
    0.65
    >>> len(table)
    3
    """
    sizes = np.asarray(train_sizes, dtype=float)
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != sizes.size:
        raise DataValidationError("learning_curve_stats: one score row per train size")
    return pd.DataFrame({
        "train_size": sizes,
        "mean": matrix.mean(axis=1),
        "std": matrix.std(axis=1, ddof=0),
    })


def prop_to_size(
    values: Sequence[float] | np.ndarray,
    min_size: float,
    max_size: float,
    power: float = 0.5,
) -> np.ndarray:
    """The size mapping ``mi + (ma - mi) * ((x - min)/(max - min))^power``.

    yellowbrick ``utils/helpers.py`` -- scatter areas proportional to
    a quantity with a power scale and the degenerate-range guard.

    Examples
    --------
    >>> [round(float(v)) for v in prop_to_size([0, 5, 10], 100, 1000)]
    [100, 736, 1000]
    """
    arr = np.asarray(values, dtype=float)
    span = arr.max() - arr.min()
    if span == 0:
        return np.full(arr.shape, min_size)
    scaled = (arr - arr.min()) / span
    return min_size + (max_size - min_size) * scaled**power

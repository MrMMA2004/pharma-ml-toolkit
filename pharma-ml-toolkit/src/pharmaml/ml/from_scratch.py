"""From-scratch teaching classics: a step-activation perceptron, DBSCAN, mean
shift and item-item collaborative filtering, in numpy and pandas only.

Two teaching repositories were absorbed file-by-file, and this module is
where the four algorithms worth keeping landed as toolkit surface.  Nothing
here imports sklearn, scipy or any other ML library -- the whole point of
the module is that every training loop, every epsilon-neighbour list and
every similarity weight is visible in the source below; the dependency list
of this file is numpy + pandas + the toolkit's own typed errors and logger.

* :class:`PerceptronScratch` -- the "Experiment 1: Training a Simple
  Perceptron" cell of madhurimarawat/Machine-Learning-Using-Python
  (https://github.com/madhurimarawat/Machine-Learning-Using-Python):
  a step-activation neuron (``1 if z >= 0 else 0``) trained by the classic
  error-driven update ``w += lr * (target - prediction) * x`` with the bias
  updated by the same error, and one misclassification count recorded per
  epoch -- exactly the notebook's printed error history.
* :func:`dbscan_cluster` -- madhug-nadig/Machine-Learning-Algorithms-from-Scratch's
  ``DBSCAN.py``
  (https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch):
  pairwise euclidean neighbour lists at ``epsilon`` (each list includes the
  point itself), core points holding at least ``min_points`` neighbours,
  border points that neighbour a core, and the queue-based breadth-first
  expansion that grows each density component -- faithfully
  ``CustomDBSCAN.fit``.
* :func:`mean_shift_cluster` -- the same repo's ``Mean Shift.py``, whose
  checked-in ``CustomMS`` class is an empty ``pass`` shell around the
  classic radial-kernel mean shift the file was lifted from; this module
  rebuilds that algorithm in full: every point starts as a seed, each seed
  climbs the Gaussian-kernel weighted mean
  ``x <- sum(K(||x - xi|| / h) * xi) / sum(K(||x - xi|| / h))`` until it
  stops moving (or ``max_iterations``), converged centers closer than
  ``merge_tolerance`` times the data range merge, and every point takes the
  label of its nearest surviving center.
* :func:`collaborative_filtering` -- the repo's ``Collaborative Filtering.py``:
  item-item adjusted cosine similarity (mean-centered per user, zero when
  two items share no rater), each rating normalised to [-1, 1] and back,
  predictions as the similarity-weighted average of normalised ratings, and
  the threshold-gated recommendation list.

What did NOT land, stated honestly: from the from-scratch repo, the
parallel-multiprocessing KNN (``K Nearest Neighbours - In Parallel.py``)
did not land -- a ``multiprocessing.Pool`` vote has no place inside a GUI
event loop, and the toolkit already ships KNN surfaces (the registry's
``'knn'`` model and :func:`~pharmaml.ml.applicability.knn_applicability_domain`).
The brute-force grid-search SVM (``Support Vector Machine.py``) did not
land: :mod:`pharmaml.ml.tuner` owns hyperparameter search.  A* pathfinding
(``a-star.py``) did not land: graph search is not statistics.  The empty
FP-Growth stub (``FP Growth.py``, whose ``FPGrowth`` class body is a bare
``pass``) did not land: there was no algorithm in the file to port.  From
the teaching repo, only the perceptron experiment landed -- the tanh neuron
of ``Neuron.ipynb`` and the keras feed-forward/CIFAR experiments of the
deep-learning notebooks did not, because :mod:`pharmaml.ml.neural` already
owns real networks.  Determinism: the perceptron's epoch order and the
mean-shift seed order run through ``numpy.random.default_rng(random_state)``
with the toolkit's pinned seed 42; DBSCAN and collaborative filtering
contain no randomness at all.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.ml.from_scratch import PerceptronScratch
>>> rng = np.random.default_rng(42)
>>> X = np.vstack([rng.normal(-2.0, 0.5, size=(25, 2)),
...                rng.normal(2.0, 0.5, size=(25, 2))])
>>> y = np.array([0] * 25 + [1] * 25)
>>> model = PerceptronScratch(learning_rate=0.1, max_epochs=50)
>>> trained = model.fit(X, y)
>>> trained.accuracy >= 0.99                 # the blobs are linearly separable
True
>>> trained.errors_history[-1]               # the early stop fires on a clean epoch
0
>>> bool((model.predict(X) == y).all())
True
>>> print(trained)
Perceptron (from scratch): 2 epochs, final misclassifications 0, training accuracy 100.0%
>>> trained.as_frame()
   epoch  errors
0      1       1
1      2       0

>>> from pharmaml.ml.from_scratch import dbscan_cluster
>>> rng = np.random.default_rng(42)
>>> points = np.vstack([rng.normal(0.0, 0.25, size=(12, 2)),
...                     rng.normal(8.0, 0.25, size=(12, 2)),
...                     [[20.0, 20.0]]])     # the lone outlier
>>> db = dbscan_cluster(points, epsilon=0.8, min_points=3)
>>> db.n_clusters
2
>>> int(db.labels.iloc[-1])                  # the outlier is noise
-1
>>> "clusters" in str(db)                    # the summary leads with the count
True

>>> from pharmaml.ml.from_scratch import mean_shift_cluster
>>> rng = np.random.default_rng(42)
>>> blobs = pd.DataFrame(
...     np.vstack([rng.normal(0.0, 0.3, size=(20, 2)),
...                rng.normal(25.0, 0.3, size=(20, 2))]),
...     columns=["x", "y"])
>>> ms = mean_shift_cluster(blobs)           # bandwidth=None -> NN-distance rule
>>> ms.n_clusters
2
>>> sorted(ms.labels.value_counts().to_list())
[20, 20]

>>> from pharmaml.ml.from_scratch import collaborative_filtering
>>> ratings = pd.DataFrame(
...     {
...         "Dune": [5, 4, np.nan],
...         "Alien": [5, 4, 5],
...         "Arrival": [3, 2, 3],
...         "Contact": [3, 2, np.nan],
...     },
...     index=["alice", "bob", "carol"])
>>> cf = collaborative_filtering(ratings)
>>> cf.recommendations
    user  item  predicted
0  carol  Dune        4.0
>>> round(float(cf.similarity.loc["Dune", "Alien"]), 3)   # agreeing taste
1.0
>>> bool((cf.similarity.to_numpy().diagonal() == 1.0).all())   # self-similarity
True
>>> float(cf.predictions.loc["carol", "Dune"])            # the gap, filled
4.0
>>> float(cf.predictions.loc["alice", "Dune"])            # known kept verbatim
5.0
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "RANDOM_STATE",
    "CollaborativeFilteringResult",
    "DbscanResult",
    "MeanShiftResult",
    "PerceptronScratch",
    "PerceptronScratchResult",
    "collaborative_filtering",
    "dbscan_cluster",
    "mean_shift_cluster",
]

_LOG = get_logger(__name__)

#: Random seed threaded through every stochastic routine (the toolkit's pin).
RANDOM_STATE = 42

#: Convergence tolerance for one mean-shift step, as a share of the data
#: range (a center has converged once a step is smaller than this).
_SHIFT_TOLERANCE = 1e-4


def _as_matrix(X: Any, *, caller: str) -> tuple[np.ndarray, list[str], pd.Index | None]:
    """Validate and convert ``X`` to a complete float matrix.

    Accepts a DataFrame, a one-column Series or a 2-D array-like.  Refuses
    NaN/±inf, non-numeric dtypes and empty tables with the typed error --
    the geometric routines in this module have no target to validate
    against afterwards, so the input must be complete up front (the
    collaborative-filtering frame is the one deliberate exception: there
    NaN is the *meaning*, "unrated").

    Returns the matrix, the column names and the input row index (``None``
    for a plain array, in which case callers fall back to a RangeIndex).
    """
    had_frame = isinstance(X, (pd.Series, pd.DataFrame))
    if isinstance(X, pd.Series):
        frame = X.to_frame()
    elif isinstance(X, pd.DataFrame):
        frame = X
    else:
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise DataValidationError(
                f"{caller}: expected a 2-D table or 1-D series, got array of shape {array.shape}"
            )
        try:
            frame = pd.DataFrame(array.astype(float))
        except (ValueError, TypeError):
            # A ragged / string array: build the object frame so the
            # per-column dtype check below names the culprit honestly.
            frame = pd.DataFrame(array)
    if frame.shape[1] == 0:
        raise DataValidationError(f"{caller}: no feature columns in the input table")
    if frame.shape[0] == 0:
        raise DataValidationError(f"{caller}: no rows in the input table")
    numeric: list[Any] = []
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            numeric.append(frame[column].astype(float))
        elif pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(frame[column])
        else:
            raise DataValidationError(
                f"{caller}: column {column!r} is not numeric "
                f"({frame[column].dtype}); drop or encode it first"
            )
    matrix = pd.concat(numeric, axis=1).to_numpy(dtype=float)
    bad = int((~np.isfinite(matrix)).any(axis=1).sum())
    if bad:
        raise DataValidationError(
            f"{caller}: {bad} row(s) contain NaN or ±inf -- the from-scratch "
            "routines need complete rows (clean_numeric_frame helps)"
        )
    names = [str(column) for column in frame.columns]
    index = frame.index if had_frame else None
    return matrix, names, index


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    """Full euclidean distance matrix (the from-scratch repos' ``norm`` loops)."""
    return np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)


# --------------------------------------------------------------------------- #
# perceptron (repo 1's teaching experiment)
# --------------------------------------------------------------------------- #
@dataclass
class PerceptronScratchResult:
    """Typed report of :meth:`PerceptronScratch.fit`.

    Attributes
    ----------
    weights:
        Feature weights (the bias is kept separate, unlike the notebook's
        one-array trick of parking it at index 0).
    bias:
        The bias term ``b`` in ``w . x + b``.
    errors_history:
        Misclassification count per epoch -- the notebook's printed
        "Epoch k -> Misclassifications: n" log as a list.
    n_epochs:
        Epochs actually run (the early stop ends training once an epoch is
        error-free; the notebook always burned its full fixed budget).
    accuracy:
        Training-set accuracy of the final weights (0..1).
    classes:
        The two label values found in ``y``, sorted -- the smaller maps to
        the negative side of the step function, the larger to the positive.
    """

    weights: np.ndarray
    bias: float
    errors_history: list[int]
    n_epochs: int
    accuracy: float
    classes: tuple[int, int]

    def as_frame(self) -> pd.DataFrame:
        """The per-epoch error history as a DataFrame (epoch, errors)."""
        return pd.DataFrame(
            {
                "epoch": np.arange(1, len(self.errors_history) + 1),
                "errors": list(self.errors_history),
            }
        )

    def __str__(self) -> str:
        return (
            f"Perceptron (from scratch): {self.n_epochs} epochs, "
            f"final misclassifications {self.errors_history[-1]}, "
            f"training accuracy {self.accuracy:.1%}"
        )


class PerceptronScratch:
    """From-scratch perceptron (repo 1's teaching experiment).

    A single step-activation neuron -- ``1 if w . x + b >= 0 else 0``,
    exactly the notebook's ``activation_fn`` -- trained by the perceptron
    learning rule: on every misclassified sample the weights move by
    ``learning_rate * (target - prediction) * x`` and the bias by the same
    error without the input.  Weights start at zero like the notebook's;
    each epoch walks the samples in a shuffled order drawn from
    ``numpy.random.default_rng(random_state)`` (the notebook iterated in
    row order -- the shuffle is the toolkit's determinism requirement, and
    with the seed pinned a rerun reproduces the same history).  Any two
    integer labels are accepted: they are mapped to 0/1 internally and
    mapped back in :meth:`predict`.  Training stops early once an epoch
    finishes with zero misclassifications.

    Examples
    --------
    >>> from pharmaml.ml.from_scratch import PerceptronScratch
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.vstack([rng.normal(-1.0, 0.3, size=(10, 2)),
    ...                rng.normal(1.0, 0.3, size=(10, 2))])
    >>> y = np.array([-1] * 10 + [7] * 10)          # any two integer labels
    >>> model = PerceptronScratch(learning_rate=0.1, max_epochs=50)
    >>> result = model.fit(X, y)
    >>> result.classes
    (-1, 7)
    >>> set(np.unique(model.predict(X))) == {-1, 7}
    True
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if learning_rate <= 0:
            raise DataValidationError(
                f"learning_rate must be > 0, got {learning_rate}"
            )
        if max_epochs < 1:
            raise DataValidationError(f"max_epochs must be >= 1, got {max_epochs}")
        self._learning_rate = float(learning_rate)
        self._max_epochs = int(max_epochs)
        self._random_state = int(random_state)
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._classes: tuple[int, int] | None = None

    def fit(self, X: Any, y: Any) -> PerceptronScratchResult:
        """Train the neuron and return the typed report.

        Parameters
        ----------
        X:
            Numeric DataFrame / Series / 2-D array (complete rows).
        y:
            Binary target with any two distinct integer labels (a pandas
            Series, list or array; its length must match ``X``).

        Returns
        -------
        PerceptronScratchResult
        """
        matrix, _names, _index = _as_matrix(X, caller="PerceptronScratch.fit")
        n_rows, n_features = matrix.shape
        targets = np.asarray(y).reshape(-1)
        if targets.shape[0] != n_rows:
            raise DataValidationError(
                f"PerceptronScratch.fit: X has {n_rows} rows but y has "
                f"{targets.shape[0]} entries"
            )
        unique_values = pd.unique(targets)
        if len(unique_values) != 2:
            raise DataValidationError(
                "PerceptronScratch.fit needs exactly two classes, got "
                f"{len(unique_values)} distinct label(s): {list(unique_values)[:5]}"
            )
        label_a = self._int_label(unique_values[0])
        label_b = self._int_label(unique_values[1])
        if label_a == label_b:
            raise DataValidationError(
                "PerceptronScratch.fit needs two DISTINCT classes, "
                f"both labels coerce to {label_a}"
            )
        # The sorted pair maps to internal 0/1; the larger label is the
        # positive side of the step function.
        classes: tuple[int, int] = (label_a, label_b) if label_a <= label_b else (label_b, label_a)
        mapped = (targets == classes[1]).astype(int)

        # Zero init, exactly the notebook's np.zeros(input_size + 1); the
        # bias lives in its own slot here instead of weights[0].
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0
        rng = np.random.default_rng(self._random_state)
        errors_history: list[int] = []
        for _epoch in range(self._max_epochs):
            errors = 0
            # Seeded shuffled epoch order (the notebook's row order, pinned
            # for the toolkit's determinism contract).
            for i in rng.permutation(n_rows):
                prediction = 1 if float(matrix[i] @ weights + bias) >= 0.0 else 0
                update = self._learning_rate * (mapped[i] - prediction)
                weights += update * matrix[i]
                bias += update
                errors += int(update != 0.0)
            errors_history.append(errors)
            if errors == 0:
                break  # early stop: an error-free epoch ends training

        self._weights = weights
        self._bias = float(bias)
        self._classes = classes
        predictions = self._predict_mapped(matrix)
        accuracy = float(np.mean(predictions == mapped))
        _LOG.info(
            "PerceptronScratch converged in %d epochs (accuracy %.3f)",
            len(errors_history),
            accuracy,
        )
        return PerceptronScratchResult(
            weights=weights,
            bias=float(bias),
            errors_history=errors_history,
            n_epochs=len(errors_history),
            accuracy=accuracy,
            classes=classes,
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the ORIGINAL label values (not the internal 0/1)."""
        if self._weights is None or self._classes is None:
            raise DataValidationError(
                "PerceptronScratch.predict before fit -- call fit first"
            )
        matrix, _names, _index = _as_matrix(X, caller="PerceptronScratch.predict")
        mapped = self._predict_mapped(matrix)
        return np.where(mapped == 1, self._classes[1], self._classes[0])

    def decision_function(self, X: Any) -> np.ndarray:
        """The raw scores ``w . x + b`` (positive side = the larger label)."""
        if self._weights is None:
            raise DataValidationError(
                "PerceptronScratch.decision_function before fit -- call fit first"
            )
        matrix, _names, _index = _as_matrix(
            X, caller="PerceptronScratch.decision_function"
        )
        return matrix @ self._weights + self._bias

    def _predict_mapped(self, matrix: np.ndarray) -> np.ndarray:
        """Step activation ``1 if w . x + b >= 0 else 0`` on validated input."""
        if self._weights is None:
            raise DataValidationError(
                "PerceptronScratch: predict/decision_function before fit -- "
                "call fit first"
            )
        if matrix.shape[1] != self._weights.shape[0]:
            raise DataValidationError(
                f"PerceptronScratch: fit saw {self._weights.shape[0]} feature(s) "
                f"but the input has {matrix.shape[1]}"
            )
        return (matrix @ self._weights + self._bias >= 0.0).astype(int)

    @staticmethod
    def _int_label(value: Any) -> int:
        """Coerce one label to int, refusing non-integer values honestly."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if np.isnan(number) or not float(number).is_integer():
            raise DataValidationError(
                "PerceptronScratch labels must be integers (e.g. 0/1, -1/+1), "
                f"got {value!r}"
            )
        return int(number)


# --------------------------------------------------------------------------- #
# DBSCAN (repo 2's CustomDBSCAN)
# --------------------------------------------------------------------------- #
@dataclass
class DbscanResult:
    """Typed report of :func:`dbscan_cluster`.

    Attributes
    ----------
    labels:
        Cluster id per row (a Series aligned to the input index); ``-1`` is
        noise, ``1..n_clusters`` the density components.
    n_clusters:
        Density components found (the BFS expansions that ran).
    n_noise:
        Points that are neither core nor border.
    n_core:
        Points with at least ``min_points`` neighbours (themselves included).
    n_border:
        Non-core points that neighbour a core point.
    epsilon:
        The neighbourhood radius used.
    min_points:
        The core-point density threshold used.
    """

    labels: pd.Series
    n_clusters: int
    n_noise: int
    n_core: int
    n_border: int
    epsilon: float
    min_points: int

    def as_frame(self) -> pd.DataFrame:
        """One row: the cluster/noise/core/border counts and the parameters."""
        return pd.DataFrame(
            [
                {
                    "n_clusters": self.n_clusters,
                    "n_noise": self.n_noise,
                    "n_core": self.n_core,
                    "n_border": self.n_border,
                    "epsilon": self.epsilon,
                    "min_points": self.min_points,
                }
            ]
        )

    def __str__(self) -> str:
        return (
            f"DBSCAN (from scratch): {self.n_clusters} "
            f"cluster{'' if self.n_clusters == 1 else 's'}, {self.n_noise} noise, "
            f"{self.n_core} core + {self.n_border} border points "
            f"(epsilon={self.epsilon:g}, min_points={self.min_points})"
        )


def dbscan_cluster(
    X: Any,
    epsilon: float = 0.5,
    min_points: int = 5,
) -> DbscanResult:
    """From-scratch DBSCAN (repo 2's CustomDBSCAN): pairwise euclidean
    neighbour lists, core (>= min_points neighbours) / border (neighbour of
    a core) / noise labels, queue-based BFS expansion.  X: DataFrame or array.

    Parameters
    ----------
    X:
        Numeric DataFrame / Series / 2-D array (complete rows).
    epsilon:
        The neighbourhood radius (``Eps`` in the repo's ``fit``).
    min_points:
        The density a point needs to become core (``MinPt``); each point
        counts itself among its own neighbours, exactly
        ``CustomDBSCAN.neighbour_points``.

    Returns
    -------
    DbscanResult
    """
    matrix, _names, index = _as_matrix(X, caller="dbscan_cluster")
    n_rows = matrix.shape[0]
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise DataValidationError(f"epsilon must be > 0, got {epsilon}")
    min_points = int(min_points)
    if min_points < 1:
        raise DataValidationError(f"min_points must be >= 1, got {min_points}")

    distances = _pairwise_distances(matrix)
    # Neighbour lists at epsilon, including the point itself -- exactly
    # CustomDBSCAN.neighbour_points (the repo computes each pair's norm in
    # a Python loop; the vectorised outer difference is the same arithmetic).
    neighbours = [np.flatnonzero(distances[i] <= epsilon) for i in range(n_rows)]

    # Core points: len(neighbours) >= MinPt, exactly the repo's first pass;
    # border points: a non-core point with at least one core neighbour
    # (the repo's second pass over its `border` list).
    core_mask = np.array([len(point_neighbours) >= min_points for point_neighbours in neighbours])
    border_mask = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        if not core_mask[i] and bool(core_mask[neighbours[i]].any()):
            border_mask[i] = True

    # Noise keeps -1 (the toolkit's convention; the repo left it 0).
    labels_array = np.full(n_rows, -1, dtype=int)
    cluster_id = 0
    # The queue-based BFS expansion, exactly CustomDBSCAN.fit: seed at an
    # unclaimed core point, adopt its unclaimed core neighbours into the
    # queue and its unclaimed border neighbours directly, then drain the
    # queue the same way (collections.deque stands in for queue.Queue --
    # identical FIFO semantics, none of the locks).  First claim wins, which
    # is also what the repo's sentinel checks (label == core / == border)
    # resolve to once a point has been claimed.
    for i in range(n_rows):
        if not core_mask[i] or labels_array[i] != -1:
            continue
        cluster_id += 1
        labels_array[i] = cluster_id
        expansion: deque[int] = deque()
        for neighbour in neighbours[i]:
            if labels_array[neighbour] == -1:
                labels_array[neighbour] = cluster_id
                if core_mask[neighbour]:
                    expansion.append(neighbour)
        while expansion:
            for neighbour in neighbours[expansion.popleft()]:
                if labels_array[neighbour] == -1:
                    labels_array[neighbour] = cluster_id
                    if core_mask[neighbour]:
                        expansion.append(neighbour)

    labels = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    result = DbscanResult(
        labels=labels,
        n_clusters=cluster_id,
        n_noise=int((labels_array == -1).sum()),
        n_core=int(core_mask.sum()),
        n_border=int(border_mask.sum()),
        epsilon=epsilon,
        min_points=min_points,
    )
    _LOG.info(
        "DBSCAN found %d clusters (%d core, %d border, %d noise)",
        result.n_clusters,
        result.n_core,
        result.n_border,
        result.n_noise,
    )
    return result


# --------------------------------------------------------------------------- #
# mean shift (repo 2's Mean Shift.py, rebuilt in full)
# --------------------------------------------------------------------------- #
@dataclass
class MeanShiftResult:
    """Typed report of :func:`mean_shift_cluster`.

    Attributes
    ----------
    labels:
        Cluster id per row (a Series aligned to the input index), the index
        of the row's nearest merged center in ``centers``.
    centers:
        The surviving (merged) centers as a DataFrame with the input
        column names.
    n_clusters:
        Number of centers that survived the merge.
    bandwidth:
        The kernel scale actually used (chosen or supplied).
    n_iterations:
        Total shift steps taken across all seeds before every center
        converged or hit ``max_iterations``.
    """

    labels: pd.Series
    centers: pd.DataFrame
    n_clusters: int
    bandwidth: float
    n_iterations: int

    def as_frame(self) -> pd.DataFrame:
        """Labels joined to the merged-center coordinates, one row per point."""
        out = self.labels.to_frame(name="cluster")
        for column in self.centers.columns:
            out[f"center_{column}"] = out["cluster"].map(self.centers[column])
        return out

    def __str__(self) -> str:
        return (
            f"Mean shift (from scratch): {self.n_clusters} "
            f"cluster{'' if self.n_clusters == 1 else 's'}, "
            f"bandwidth {self.bandwidth:.4g}, {self.n_iterations} shift iterations"
        )


def mean_shift_cluster(
    X: Any,
    bandwidth: float | None = None,
    merge_tolerance: float = 0.05,
    max_iterations: int = 60,
    random_state: int = RANDOM_STATE,
) -> MeanShiftResult:
    """From-scratch mean shift: every point a seed, Gaussian kernel shift
    x <- sum(K(||x-xi||/h) xi)/sum(K) until convergence or max_iterations,
    merge centers closer than merge_tolerance * data range, then assign
    labels to the nearest merged center.  bandwidth=None -> the 75th
    percentile of nearest-neighbour distances (pure numpy).

    The 75th-percentile default is a LOCAL scale: on tightly sampled blobs
    it finds several sub-modes per blob, and the merge step (relative to
    the data range) coalesces them; pass an explicit bandwidth on the blob
    scale when the textbook picture is wanted.

    Parameters
    ----------
    X:
        Numeric DataFrame / Series / 2-D array (complete rows).
    bandwidth:
        Gaussian kernel scale ``h``; ``None`` (default) takes the 75th
        percentile of nearest-neighbour distances.
    merge_tolerance:
        Centers within this fraction of the data range merge into one
        (the first-seen center survives, the tutorial's duplicate pop).
    max_iterations:
        Shift steps any one seed may take before giving up.
    random_state:
        Seed (the toolkit's 42 by default) -- it fixes the seed processing
        order; the fixed points themselves are order-independent.

    Returns
    -------
    MeanShiftResult
    """
    matrix, names, index = _as_matrix(X, caller="mean_shift_cluster")
    n_rows = matrix.shape[0]
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise DataValidationError(f"max_iterations must be >= 1, got {max_iterations}")
    merge_tolerance = float(merge_tolerance)
    if merge_tolerance < 0:
        raise DataValidationError(
            f"merge_tolerance must be >= 0, got {merge_tolerance}"
        )

    if bandwidth is None:
        # The nearest-neighbour distance rule, pure numpy: every point's
        # distance to its closest OTHER point, then the 75th percentile.
        distances = _pairwise_distances(matrix)
        np.fill_diagonal(distances, np.inf)
        bandwidth = float(np.percentile(distances.min(axis=1), 75))
        if bandwidth <= 0:
            raise DataValidationError(
                "mean_shift_cluster: the nearest-neighbour heuristic produced "
                "bandwidth 0 -- the input collapses to duplicated points"
            )
        _LOG.info("Mean shift bandwidth from the NN-distance rule: %.4g", bandwidth)
    else:
        bandwidth = float(bandwidth)
        if bandwidth <= 0:
            raise DataValidationError(f"bandwidth must be > 0, got {bandwidth}")

    data_range = float(np.max(matrix.max(axis=0) - matrix.min(axis=0)))
    merge_threshold = merge_tolerance * data_range
    tolerance = _SHIFT_TOLERANCE * max(data_range, 1.0)

    rng = np.random.default_rng(random_state)
    centers: list[np.ndarray] = []
    n_iterations = 0
    # Every point starts as its own seed (the tutorial's per-point centroid
    # dict); the seeded shuffle only fixes the processing order.
    for seed in rng.permutation(n_rows):
        current = matrix[seed].astype(float).copy()
        for _step in range(max_iterations):
            distances = np.linalg.norm(matrix - current, axis=1)
            # The radial (Gaussian) kernel on the scaled distance.
            kernel = np.exp(-0.5 * (distances / bandwidth) ** 2)
            mass = float(kernel.sum())
            if mass <= 0.0:
                break  # numerically flat kernel: the seed is done moving
            shifted = (kernel[:, None] * matrix).sum(axis=0) / mass
            n_iterations += 1
            step = float(np.linalg.norm(shifted - current))
            current = shifted
            if step <= tolerance:
                break
        centers.append(current)

    # Merge converged centers within merge_tolerance x data range; the
    # first center seen survives (the tutorial's pop-the-duplicate pass).
    merged: list[np.ndarray] = []
    for center in centers:
        if not any(float(np.linalg.norm(center - kept)) <= merge_threshold for kept in merged):
            merged.append(center)
    center_matrix = np.vstack(merged)

    # Labels: every point joins its nearest surviving center.
    label_distances = np.linalg.norm(
        matrix[:, None, :] - center_matrix[None, :, :], axis=2
    )
    labels_array = np.argmin(label_distances, axis=1)
    centers_frame = pd.DataFrame(center_matrix, columns=names)
    labels = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    _LOG.info(
        "Mean shift merged %d seeds into %d centers (%d shift iterations)",
        len(centers),
        len(merged),
        n_iterations,
    )
    return MeanShiftResult(
        labels=labels,
        centers=centers_frame,
        n_clusters=len(merged),
        bandwidth=bandwidth,
        n_iterations=n_iterations,
    )


# --------------------------------------------------------------------------- #
# item-item collaborative filtering (repo 2's script)
# --------------------------------------------------------------------------- #
@dataclass
class CollaborativeFilteringResult:
    """Typed report of :func:`collaborative_filtering`.

    Attributes
    ----------
    similarity:
        Item x item adjusted-cosine DataFrame (per-user mean-centered over
        the co-raters, 0 when two items share no rater; the diagonal is
        fixed at 1.0 where the script simply left it absent).
    predictions:
        The full users x items matrix: known ratings verbatim, unrated
        cells filled with the denormalised prediction (0.0 where the
        similarity mass is zero -- the script's bare ``return 0``).
    recommendations:
        Columns ``user``, ``item``, ``predicted`` -- each user's unrated
        items scoring at or above ``threshold``, best first, capped at
        ``top_n`` rows per user.
    min_rating / max_rating:
        The rating scale the normalisation used.
    threshold:
        The recommendation cut-off.
    """

    similarity: pd.DataFrame
    predictions: pd.DataFrame
    recommendations: pd.DataFrame
    min_rating: float
    max_rating: float
    threshold: float

    def as_frame(self) -> pd.DataFrame:
        """The recommendations table (user, item, predicted)."""
        return self.recommendations

    def __str__(self) -> str:
        return (
            f"Collaborative filtering (from scratch): "
            f"{len(self.similarity.index)} users x {len(self.similarity.columns)} items, "
            f"{len(self.recommendations)} recommendation(s) at or above {self.threshold:g} "
            f"(ratings on a {self.min_rating:g}-{self.max_rating:g} scale)"
        )


def collaborative_filtering(
    ratings: pd.DataFrame,
    min_rating: float = 1.0,
    max_rating: float = 5.0,
    threshold: float = 3.5,
    top_n: int = 5,
) -> CollaborativeFilteringResult:
    """Item-item collaborative filtering exactly as repo 2's script:
    adjusted cosine similarity (mean-centered per USER over co-rated items;
    0 when no co-ratings), normalize each rating to [-1, 1], prediction =
    sum(sim * norm_rating)/sum(|sim|), denormalize; recommendations =
    unrated items with prediction >= threshold, sorted per user, capped at
    top_n.  ratings: index = users, columns = items, NaN = unrated.  Known
    ratings are preserved verbatim in `predictions`.

    The per-user mean runs over ALL items each user rated (the script's
    ``averages`` dict, built before the similarity loops), while the
    similarity sums themselves only run over the users who co-rated the
    item pair -- both faithful to ``compute_similarity``.

    Parameters
    ----------
    ratings:
        Users x items DataFrame of numeric ratings; NaN marks an unrated
        cell.
    min_rating / max_rating:
        The rating scale's bounds for the [-1, 1] normalisation (the
        script's ``MINrating`` / ``MAXrating``).
    threshold:
        Minimum predicted rating for an item to be recommended.
    top_n:
        Recommendations kept per user (best predicted first).

    Returns
    -------
    CollaborativeFilteringResult
    """
    if not isinstance(ratings, pd.DataFrame):
        raise DataValidationError(
            "collaborative_filtering needs a ratings DataFrame "
            "(index=users, columns=items, NaN=unrated)"
        )
    if ratings.shape[0] == 0 or ratings.shape[1] == 0:
        raise DataValidationError(
            f"collaborative_filtering: empty ratings table {ratings.shape}"
        )
    for column in ratings.columns:
        if not pd.api.types.is_numeric_dtype(ratings[column]):
            raise DataValidationError(
                f"collaborative_filtering: column {column!r} is not numeric "
                f"({ratings[column].dtype})"
            )
    min_rating = float(min_rating)
    max_rating = float(max_rating)
    if max_rating <= min_rating:
        raise DataValidationError(
            f"max_rating must exceed min_rating, got {min_rating}..{max_rating}"
        )
    threshold = float(threshold)
    top_n = int(top_n)
    if top_n < 1:
        raise DataValidationError(f"top_n must be >= 1, got {top_n}")

    values = ratings.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataValidationError(
            "collaborative_filtering: ratings contain ±inf (NaN means unrated, "
            "inf is not a rating)"
        )
    rated = ~np.isnan(values)
    n_users, n_items = values.shape
    users = ratings.index
    items = list(ratings.columns)

    # Per-user means over all rated items -- the script's averages dict.
    counts = rated.sum(axis=1)
    sums = np.nansum(values, axis=1)
    means = sums / np.maximum(counts, 1)  # an all-NaN row has no co-ratings
    deviations = np.where(rated, values - means[:, None], 0.0)

    # Adjusted cosine, pair by pair -- exactly compute_similarity(item1,
    # item2): sums over co-raters, sqrt(dem1 * dem2), 0 when either
    # denominator sum is zero (or no user co-rated the pair).
    similarity = np.zeros((n_items, n_items), dtype=float)
    np.fill_diagonal(similarity, 1.0)
    for a in range(n_items):
        for b in range(a + 1, n_items):
            co_raters = rated[:, a] & rated[:, b]
            if not bool(co_raters.any()):
                continue
            dev_a = deviations[co_raters, a]
            dev_b = deviations[co_raters, b]
            denominator = np.sqrt(float((dev_a**2).sum()) * float((dev_b**2).sum()))
            value = 0.0 if denominator == 0.0 else float((dev_a * dev_b).sum()) / denominator
            similarity[a, b] = similarity[b, a] = value
    similarity_frame = pd.DataFrame(similarity, index=items, columns=items)

    # The script's normalize(): (2*(r-min) - (max-min)) / (max-min) maps the
    # scale onto [-1, 1]; denormalize() inverts it.
    span = max_rating - min_rating
    normalized = (2.0 * (values - min_rating) - span) / span

    # Prediction = sum(sim * norm_rating) / sum(|sim|) over the items the
    # user rated (the script's prediction()); the diagonal is excluded --
    # the script's sm simply has no self-entry.  Zero similarity mass gives
    # the bare 0.0 of the script's `return 0`, NOT a denormalised value.
    similarity_off = similarity.copy()
    np.fill_diagonal(similarity_off, 0.0)
    numerators = np.where(rated, normalized, 0.0) @ similarity_off.T
    denominators = rated.astype(float) @ np.abs(similarity_off).T
    safe = np.where(denominators == 0.0, 1.0, denominators)
    predicted = np.where(
        denominators == 0.0,
        0.0,
        ((numerators / safe + 1.0) * span) / 2.0 + min_rating,
    )
    full = np.where(rated, values, predicted)
    predictions_frame = pd.DataFrame(full, index=users, columns=items)

    # Recommendations: unrated items at or above the threshold, best first,
    # capped at top_n per user (the script's recommendation() list).
    rows: list[dict[str, Any]] = []
    for user_position, user in enumerate(users):
        candidates: list[tuple[float, Any, int]] = []
        for item_position, item in enumerate(items):
            if not rated[user_position, item_position]:
                score = float(full[user_position, item_position])
                if score >= threshold:
                    candidates.append((-score, str(item), item_position))
        # Highest predicted first; the item name breaks ties deterministically.
        candidates.sort()
        for _neg_score, _name, item_position in candidates[:top_n]:
            rows.append(
                {
                    "user": user,
                    "item": items[item_position],
                    "predicted": float(full[user_position, item_position]),
                }
            )
    recommendations = pd.DataFrame(rows, columns=["user", "item", "predicted"])

    result = CollaborativeFilteringResult(
        similarity=similarity_frame,
        predictions=predictions_frame,
        recommendations=recommendations,
        min_rating=min_rating,
        max_rating=max_rating,
        threshold=threshold,
    )
    _LOG.info(
        "Collaborative filtering: %d users x %d items -> %d recommendation(s)",
        n_users,
        n_items,
        len(recommendations),
    )
    return result

"""Hierarchical Support Vector Regression (HSV / HSVR).

A two-level regression estimator for heterogeneous QSAR data: the
legacy ``Regression/SVM reg.py`` fitted *one* kernel SVR on the whole
dataset, which underfits when different regions of chemical space carry
different input-output structure.  :class:`HierarchicalSVR` keeps that
global SVR as the *upper* level and adds *lower*-level specialists:

1. ``fit`` standardises the features (mean/std fitted on the training
   rows, constant columns keep std 1), runs k-means (seeded) in the
   standardised space, and fits one SVR per cluster plus one global SVR
   on all rows;
2. ``predict`` routes each row to its nearest cluster's local SVR and
   blends it with the global prediction, the local weight fading
   linearly from :attr:`alpha` (at the cluster centroid) to 0 at the
   cluster's trust radius — far-from-training queries fall back to the
   global model instead of extrapolating through an unfamiliar local
   kernel.

The estimator follows the scikit-learn contract (``fit``/``predict``/
``score``, ``get_params``/``set_params``, clonable), so it drops into
:func:`~pharmaml.ml.training.train_model` and
:func:`~pharmaml.ml.training.compare_models` like any registry model —
the registry key is ``'hsvr'`` (aliases ``'hierarchical_svr'``,
``'h_svr'``).

Determinism: k-means is seeded from ``random_state`` and SVR fitting
is deterministic, so the same seed + data always yields identical
predictions.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.hierarchical import HierarchicalSVR

Two clusters with *different* slopes (0.1 vs 8.0) — the regime a single
global SVR smooths over and the hierarchy specialises on:

>>> rng = np.random.default_rng(0)
>>> x = np.concatenate([rng.uniform(0, 5, 60), rng.uniform(15, 20, 60)])
>>> y = np.concatenate([0.1 * x[:60], 8.0 * (x[60:] - 15)])
>>> X = x.reshape(-1, 1)
>>> hsvr = HierarchicalSVR(n_clusters=2, random_state=0).fit(X, y)
>>> from sklearn.svm import SVR
>>> global_only = SVR().fit(X, y)
>>> r2_local = hsvr.score(X, y)
>>> r2_global = global_only.score(X, y)
>>> r2_local > r2_global        # the hierarchy specialises
True

>>> hsvr.cluster_sizes_          # 60 rows per specialist (seed 0)
[60, 60]

>>> flat = HierarchicalSVR(n_clusters=2, alpha=0.0, random_state=0).fit(X, y)
>>> np.allclose(flat.predict(X), global_only.predict(X))   # alpha=0: global
True

>>> HierarchicalSVR().predict([[0.0]])      # sklearn contract: fit first
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: HierarchicalSVR has not been fitted yet — call fit(X, y) first

>>> HierarchicalSVR(n_clusters=8).fit(X[:3], y[:3])
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: n_clusters=8 needs at least 8 training rows, got 3 — lower n_clusters or provide more data

>>> HierarchicalSVR(alpha=1.5).fit(X, y)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: alpha must be in [0, 1], got 1.5
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.svm import SVR

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = ["HierarchicalSVR"]


def _check_matrix(estimator_name: str, X: Any) -> np.ndarray:
    """Validate and convert ``X`` to a finite float64 2-D matrix.

    Raises
    ------
    DataValidationError
        When ``X`` is not 2-dimensional, non-numeric, empty, has no
        feature columns, or contains NaN/infinite values.
    """
    try:
        X_arr = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"features must be numeric — conversion failed: {exc}"
        ) from exc
    if X_arr.ndim != 2:
        raise DataValidationError(
            f"X must be 2-dimensional (n_samples, n_features), got {X_arr.ndim} "
            "dimension(s)"
        )
    if X_arr.shape[0] == 0:
        raise DataValidationError(
            f"X has 0 rows — {estimator_name} needs at least one sample"
        )
    if X_arr.shape[1] == 0:
        raise DataValidationError(
            f"X has 0 feature columns — {estimator_name} needs at least one input "
            "feature"
        )
    if not np.isfinite(X_arr).all():
        raise DataValidationError(
            "X contains NaN or infinite values — impute or drop them first "
            "(pharmaml.ml.build_preprocessor does this inside a Pipeline)"
        )
    return X_arr


class HierarchicalSVR(RegressorMixin, BaseEstimator):
    """Hierarchical Support Vector Regression (global + per-cluster SVRs).

    Parameters
    ----------
    n_clusters:
        Number of k-means clusters (local specialists), >= 1.  Fitting
        requires at least this many training rows.
    C, gamma, epsilon, kernel:
        Forwarded to every :class:`sklearn.svm.SVR` (local and global
        alike): regularisation strength (> 0), kernel coefficient
        (``'scale'``/``'auto'`` or a float), epsilon-insensitive tube
        width (>= 0) and kernel name.
    alpha:
        Maximum weight of the local prediction at the cluster centroid,
        in ``[0, 1]`` — the blend is ``alpha * local + (1 - alpha) *
        global`` at the centroid, fading linearly to pure global at the
        cluster's trust radius.  ``alpha=0`` degenerates to the plain
        global SVR; ``alpha=1`` trusts the local model fully near its
        centroid.
    radius_factor:
        The trust radius of cluster ``c`` is ``radius_factor`` times
        the distance of its farthest training member from the centroid
        (> 0).  Larger values trust local models further out.
    random_state:
        Integer seed for k-means (determinism contract).

    Attributes
    ----------
    n_features_in_:
        Number of input features seen at fit.
    feature_names_in_:
        Column names when ``X`` was a DataFrame (checked against
        predicting DataFrames in names *and* order).
    labels_:
        Cluster label of every training row (``-1`` never occurs —
        k-means leaves no empty clusters).
    cluster_sizes_:
        Training-row count per cluster.
    cluster_centers_:
        Centroids in the *standardised* space (predict-time routing uses
        the same space).
    radii_:
        Per-cluster trust radii (standardised space).

    Raises
    ------
    ModelError
        For invalid hyperparameters, use before ``fit``, or when the
        training set is smaller than ``n_clusters``.
    DataValidationError
        For empty/non-numeric/mismatched/non-finite ``X``/``y``.

    Notes
    -----
    Registry alias: ``build_model('hsvr', 'regression')``.

    See Also
    --------
    pharmaml.ml.models.MODEL_REGISTRY : the full registry.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        C: float = 1.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
        kernel: str = "rbf",
        alpha: float = 0.8,
        radius_factor: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.kernel = kernel
        self.alpha = alpha
        self.radius_factor = radius_factor
        self.random_state = random_state

    # -- validation -----------------------------------------------------
    def _validate_hyperparameters(self) -> None:
        """Validate constructor values before any work.

        Raises
        ------
        ModelError
            When ``n_clusters``/``radius_factor`` are not positive
            numbers, ``C`` is not positive, ``epsilon`` is negative,
            ``alpha`` falls outside ``[0, 1]``, or ``random_state`` is
            not an integer.
        """
        for name, value, minimum in (
            ("n_clusters", self.n_clusters, 1),
            ("radius_factor", self.radius_factor, 1e-12),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or float(value) < minimum
            ):
                raise ModelError(f"{name} must be a number >= {minimum}, got {value!r}")
        if not np.isfinite(self.C) or float(self.C) <= 0:
            raise ModelError(f"C must be > 0, got {self.C!r}")
        if not np.isfinite(self.epsilon) or float(self.epsilon) < 0:
            raise ModelError(f"epsilon must be >= 0, got {self.epsilon!r}")
        if not np.isfinite(self.alpha) or not 0.0 <= float(self.alpha) <= 1.0:
            raise ModelError(f"alpha must be in [0, 1], got {self.alpha!r}")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ModelError(
                f"random_state must be an integer seed (determinism contract), "
                f"got {self.random_state!r}"
            )

    def _check_feature_names(self, X: Any) -> None:
        """Reject DataFrame columns that disagree with ``feature_names_in_``."""
        names = getattr(self, "feature_names_in_", None)
        if names is None or not isinstance(X, pd.DataFrame):
            return
        columns = np.asarray(X.columns, dtype=object)
        if not np.array_equal(columns, names):
            raise DataValidationError(
                f"X has feature names {columns.tolist()} that do not match the "
                f"feature names seen during fit: {names.tolist()} — pass the same "
                "columns in the fit order, or a plain (positional) array"
            )

    def _check_fitted(self) -> None:
        """Raise :class:`ModelError` when the estimator is used before fit."""
        if getattr(self, "_local_models_", None) is None:
            raise ModelError(
                f"{type(self).__name__} has not been fitted yet — call fit(X, y) "
                "first"
            )

    # -- fit --------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> HierarchicalSVR:
        """Fit the global SVR and one local SVR per k-means cluster.

        Parameters
        ----------
        X:
            Feature matrix — DataFrame or 2-D array-like; numeric and
            finite.
        y:
            Numeric regression targets, 1-D, aligned with ``X``.

        Returns
        -------
        self
            Fitted estimator (``labels_``, ``cluster_sizes_``,
            ``cluster_centers_``, ``radii_`` and ``n_features_in_`` are
            now available).
        """
        self._validate_hyperparameters()
        X_arr = _check_matrix(type(self).__name__, X)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise DataValidationError(
                f"y must be 1-dimensional, got {y_arr.ndim} dimension(s)"
            )
        if y_arr.size == 0:
            raise DataValidationError(
                f"y is empty — {type(self).__name__} needs at least one target value"
            )
        if y_arr.shape[0] != X_arr.shape[0]:
            raise DataValidationError(
                f"X and y have inconsistent lengths: {X_arr.shape[0]} rows vs "
                f"{y_arr.shape[0]} targets"
            )
        try:
            y_num = np.asarray(y_arr, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"regression targets must be numeric — conversion failed: {exc}"
            ) from exc
        if not np.isfinite(y_num).all():
            raise DataValidationError(
                "regression targets contain NaN or infinite values — drop or "
                "impute those rows first"
            )

        n = int(X_arr.shape[0])
        n_clusters = int(self.n_clusters)
        if n < n_clusters:
            raise ModelError(
                f"n_clusters={n_clusters} needs at least {n_clusters} training "
                f"rows, got {n} — lower n_clusters or provide more data"
            )

        self.n_features_in_ = int(X_arr.shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        # Standardisation fitted on all training rows (no held-out split
        # here — early stopping is not part of this estimator).
        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self._mean_ = mean
        self._std_ = std
        X_scaled = (X_arr - mean) / std

        kmeans = KMeans(n_clusters=n_clusters, random_state=int(self.random_state), n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        self.labels_ = labels
        self.cluster_sizes_ = [int(np.sum(labels == c)) for c in range(n_clusters)]
        # Copy the centroid array: the fitted KMeans object is dropped
        # after fit (pickles carry plain numpy, not sklearn internals).
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64).copy()
        self.cluster_centers_ = centers
        # k-means leaves no empty clusters, so every max() has >= 1 row.
        radii = [
            float(np.max(np.linalg.norm(X_scaled[labels == c] - centers[c], axis=1)))
            for c in range(n_clusters)
        ]
        self.radii_ = np.asarray(radii, dtype=np.float64) * float(self.radius_factor)

        def make_svr() -> SVR:
            return SVR(C=float(self.C), gamma=self.gamma, epsilon=float(self.epsilon),
                       kernel=self.kernel)

        self._local_models_: list[Any] = []
        for c in range(n_clusters):
            members = labels == c
            local = make_svr()
            if np.unique(y_num[members]).size > 1:
                local.fit(X_scaled[members], y_num[members])
            else:
                # A constant-target cluster: SVR cannot fit a single
                # distinct value — keep the unfitted dummy and answer
                # with the constant at predict time.
                local = _ConstantSVR(float(y_num[members][0]))
            self._local_models_.append(local)
        self._global_model_ = make_svr()
        self._global_model_.fit(X_scaled, y_num)

        _LOG.info(
            "%s fitted on %d samples: %d cluster(s), sizes %s, alpha=%.2f",
            type(self).__name__,
            n,
            n_clusters,
            self.cluster_sizes_,
            float(self.alpha),
        )
        return self

    # -- predict ------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        """Predict targets for ``X`` (local/global blend).

        Each row is routed to its nearest cluster (standardised space);
        the local weight fades linearly from :attr:`alpha` at the
        centroid to 0 at the cluster's trust radius, so distant rows
        fall back to the global SVR.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)``.
        """
        self._check_fitted()
        self._check_feature_names(X)
        X_arr = _check_matrix(type(self).__name__, X)
        if X_arr.shape[1] != self.n_features_in_:
            raise DataValidationError(
                f"X has {X_arr.shape[1]} feature column(s), but "
                f"{type(self).__name__} was fitted with {self.n_features_in_} — "
                "pass the same features as to fit()"
            )
        X_scaled = (X_arr - self._mean_) / self._std_

        # Distance of every row to every centroid -> nearest cluster.
        differences = X_scaled[:, None, :] - self.cluster_centers_[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        nearest = np.argmin(distances, axis=1)
        row_distances = distances[np.arange(X_arr.shape[0]), nearest]

        global_pred = self._global_model_.predict(X_scaled)
        local_pred = np.asarray(
            [self._local_models_[int(c)].predict(row.reshape(1, -1))[0]
             for c, row in zip(nearest, X_scaled)],
            dtype=np.float64,
        )

        radii = self.radii_[nearest]
        with np.errstate(divide="ignore", invalid="ignore"):
            fade = np.clip(1.0 - row_distances / np.where(radii > 0, radii, 1.0), 0.0, 1.0)
        weight = float(self.alpha) * fade
        return weight * local_pred + (1.0 - weight) * global_pred


class _ConstantSVR:
    """Stand-in SVR for a constant-target cluster (predicts the constant).

    SVR with an epsilon-tube cannot represent a target with a single
    distinct value (its dual has no support vectors); the hierarchy
    answers such a cluster with the constant itself — the same value
    the global model averages toward anyway.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: Any) -> np.ndarray:
        """Return the constant for every row of ``X`` (shape-preserving)."""
        rows = np.asarray(X)
        n = rows.shape[0] if rows.ndim == 2 else 1
        return np.full(int(n), self.value, dtype=np.float64)

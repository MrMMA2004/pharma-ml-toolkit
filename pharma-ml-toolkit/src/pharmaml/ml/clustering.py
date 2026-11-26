"""Density, mean-shift and spectral clustering (the part18 §4.3 lane).

The part18 specification asks for the three clustering families the
toolkit honestly documented as stretch candidates in
:mod:`pharmaml.ml.unsupervised` (its "honest scope" note).  This
module closes that gap with the same discipline the rest of the
unsupervised side follows: one typed function per family, a result
report object, honest degenerate-case handling, and the shared
diagnostics block (silhouette, Davies-Bouldin, Calinski-Harabasz,
cluster sizes, noise ratio).

* :func:`dbscan_cluster` -- density-based clustering with explicit
  ``eps`` / ``min_samples`` controls, an optional nearest-neighbour
  k-distance heuristic for ``eps=None``, and first-class noise-point
  reporting (label ``-1``) -- the honest answer "some samples belong
  to no cluster" that k-means cannot express.
* :func:`meanshift_cluster` -- mode-seeking clustering with
  bandwidth either supplied or estimated
  (:func:`sklearn.cluster.estimate_bandwidth`), no k parameter at
  all, and the honest single-cluster / empty-cluster outcomes.
* :func:`spectral_cluster` -- graph-based clustering with the
  affinity choice (``rbf`` with ``gamma`` vs ``nearest_neighbors``
  with ``n_neighbors``) and an explicit cluster count, deliberately
  distinct from the ``spectral`` *embedding* in
  :mod:`pharmaml.ml.manifold` (part18 §4.4's "explicitly
  distinguish" rule).

Every function threads ``random_state=42`` (the toolkit's pinned
seed).  None of these are supervised: no target, no train/test split,
no prediction on new data without an explicit re-fit (the
``transform``-free honesty of the clustering contract).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from pharmaml.ml.clustering import dbscan_cluster
>>> rng = np.random.default_rng(42)
>>> moons = pd.DataFrame({
...     "x": np.concatenate([rng.normal(0, 0.3, 40), rng.normal(4, 0.3, 40)]),
...     "y": np.concatenate([rng.normal(0, 0.3, 40), rng.normal(4, 0.3, 40)]),
... })
>>> result = dbscan_cluster(moons, eps=0.9, min_samples=4)
>>> result.n_clusters
2
>>> result.noise_ratio <= 0.05
True

>>> from pharmaml.ml.clustering import meanshift_cluster
>>> shifted = meanshift_cluster(moons, bandwidth=1.0)
>>> shifted.n_clusters >= 1
True

>>> from pharmaml.ml.clustering import spectral_cluster
>>> spec = spectral_cluster(moons, n_clusters=2, affinity="nearest_neighbors", n_neighbors=8)
>>> spec.n_clusters
2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.ml.unsupervised import RANDOM_STATE, _as_numeric_matrix

_LOG = get_logger(__name__)

#: Affinities :func:`spectral_cluster` accepts (scikit-learn's set
#: for :class:`~sklearn.cluster.SpectralClustering`).
SPECTRAL_AFFINITIES: tuple[str, ...] = ("rbf", "nearest_neighbors")


@dataclass
class ClusterDiagnostics:
    """The shared metric block for every clustering result.

    Attributes
    ----------
    silhouette:
        Mean silhouette coefficient over the non-noise samples, or
        ``None`` when fewer than two clusters exist (honest, not 0).
    davies_bouldin:
        Davies-Bouldin index (lower is better), or ``None`` when only
        one cluster exists.
    calinski_harabasz:
        Calinski-Harabasz index (higher is better), or ``None`` when
        only one cluster exists.
    sizes:
        Samples per cluster (noise counted separately).
    noise_count / noise_ratio:
        Samples labelled ``-1`` (DBSCAN) and their share.  Zero for
        the non-density families.
    n_clusters:
        Distinct non-noise cluster count.
    """

    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    sizes: pd.Series
    noise_count: int
    noise_ratio: float
    n_clusters: int

    def __str__(self) -> str:
        sil = "n/a" if self.silhouette is None else f"{self.silhouette:.3f}"
        dbi = "n/a" if self.davies_bouldin is None else f"{self.davies_bouldin:.3f}"
        chs = "n/a" if self.calinski_harabasz is None else f"{self.calinski_harabasz:.3g}"
        return (
            f"  silhouette: {sil} | Davies-Bouldin: {dbi} | "
            f"Calinski-Harabasz: {chs}\n"
            f"  clusters: {self.n_clusters}, noise: {self.noise_count} "
            f"({self.noise_ratio:.1%})"
        )


def _cluster_diagnostics(
    matrix: np.ndarray, labels: np.ndarray, *, caller: str
) -> ClusterDiagnostics:
    """Compute the shared metric block with honest degenerate answers.

    Noise samples (label ``-1``) are excluded from every score: a
    silhouette over "cluster vs noise" would quietly re-invent a
    cluster the algorithm explicitly refused to assign.
    """
    labels = np.asarray(labels)
    non_noise = labels >= 0
    core = matrix[non_noise]
    core_labels = labels[non_noise]
    distinct = sorted({int(v) for v in core_labels})
    n_clusters = len(distinct)
    noise_count = int((~non_noise).sum())
    sizes = pd.Series(
        [int((core_labels == c).sum()) for c in distinct] + [noise_count],
        index=[str(c) for c in distinct] + ["noise"],
        name="samples",
    )
    silhouette: float | None = None
    davies_bouldin: float | None = None
    calinski_harabasz: float | None = None
    if 2 <= n_clusters <= len(core_labels) - 1:
        try:
            silhouette = float(silhouette_score(core, core_labels))
        except ValueError:
            silhouette = None
    if 2 <= n_clusters and len(core) > n_clusters:
        try:
            davies_bouldin = float(davies_bouldin_score(core, core_labels))
            calinski_harabasz = float(calinski_harabasz_score(core, core_labels))
        except ValueError:
            davies_bouldin = davies_bouldin if davies_bouldin is not None else None
    if n_clusters == 1:
        _LOG.info("%s: single cluster -- similarity indices undefined", caller)
    return ClusterDiagnostics(
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        calinski_harabasz=calinski_harabasz,
        sizes=sizes,
        noise_count=noise_count,
        noise_ratio=float(noise_count / max(1, matrix.shape[0])),
        n_clusters=n_clusters,
    )


@dataclass
class DbscanResult:
    """Typed report of :func:`dbscan_cluster`.

    Attributes
    ----------
    eps, min_samples:
        The parameters used (the chosen eps when estimated).
    eps_source:
        ``'manual'`` or ``'k-distance'`` (the heuristic used when
        ``eps=None``).
    labels:
        Integer membership per sample; ``-1`` is noise.
    diagnostics:
        The shared :class:`ClusterDiagnostics` block.
    n_samples:
        Input row count.
    """

    eps: float
    min_samples: int
    eps_source: str
    labels: Any
    diagnostics: ClusterDiagnostics
    n_samples: int

    @property
    def n_clusters(self) -> int:
        return self.diagnostics.n_clusters

    @property
    def noise_count(self) -> int:
        return self.diagnostics.noise_count

    @property
    def noise_ratio(self) -> float:
        return self.diagnostics.noise_ratio

    def as_frame(self) -> pd.DataFrame:
        """Per-sample membership: the cluster label and a noise flag."""
        if isinstance(self.labels, pd.Series):
            out = self.labels.to_frame(name="cluster")
        else:
            out = pd.DataFrame({"cluster": np.asarray(self.labels)})
        out["is_noise"] = out["cluster"] < 0
        return out

    def __str__(self) -> str:
        head = (
            f"DBSCAN -- eps={self.eps:.3g} ({self.eps_source}), "
            f"min_samples={self.min_samples}, {self.n_samples} rows\n"
        )
        return head + str(self.diagnostics)


def _estimate_eps(matrix: np.ndarray, min_samples: int) -> float:
    """Nearest-neighbour k-distance heuristic for a starting ``eps``.

    The classic elbow-free shortcut: the mean distance to the
    ``min_samples``-th nearest neighbour, floored at a tiny positive
    value so a degenerate all-duplicate matrix still yields a valid
    (if useless) eps.  The result is a *starting point*, and the
    caller is told it was estimated.
    """
    k = max(2, min(min_samples, matrix.shape[0] - 1))
    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=k).fit(matrix)
    distances, _ = neighbours.kneighbors(matrix)
    reach = distances[:, -1]
    eps = float(np.median(reach[reach > 0])) if np.any(reach > 0) else 0.5
    return max(eps, 1e-6)


def dbscan_cluster(
    X: Any,
    eps: float | None = None,
    *,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> DbscanResult:
    """DBSCAN density clustering with explicit controls and noise reporting.

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array (complete rows).
    eps:
        Neighbourhood radius.  ``None`` (default) estimates a starting
        value from the median k-distance -- reported honestly via
        ``eps_source`` so nobody mistakes the heuristic for a tuned
        hyperparameter.
    min_samples:
        Minimum neighbourhood size for a core point.
    metric:
        Distance metric passed to :class:`~sklearn.cluster.DBSCAN`.

    Returns
    -------
    DbscanResult

    Raises
    ------
    DataValidationError
        Empty input, ``min_samples < 1`` or a non-positive ``eps``.
    """
    matrix, _names, index = _as_numeric_matrix(X, caller="dbscan_cluster")
    n_samples = matrix.shape[0]
    if n_samples < 3:
        raise DataValidationError(f"dbscan_cluster needs at least 3 rows, got {n_samples}")
    if min_samples < 1:
        raise DataValidationError(f"min_samples must be >= 1, got {min_samples}")
    if eps is None:
        eps_used = _estimate_eps(matrix, min_samples)
        source = "k-distance"
    else:
        eps_used = float(eps)
        if eps_used <= 0:
            raise DataValidationError(f"eps must be positive, got {eps_used}")
        source = "manual"
    fitted = DBSCAN(eps=eps_used, min_samples=min_samples, metric=metric).fit(matrix)
    labels_array = np.asarray(fitted.labels_)
    if index is not None:
        labels: Any = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    else:
        labels = labels_array
    diagnostics = _cluster_diagnostics(matrix, labels_array, caller="dbscan_cluster")
    if diagnostics.n_clusters == 0:
        _LOG.info("DBSCAN: all %d samples are noise at eps=%.3g", n_samples, eps_used)
    return DbscanResult(
        eps=eps_used,
        min_samples=int(min_samples),
        eps_source=source,
        labels=labels,
        diagnostics=diagnostics,
        n_samples=n_samples,
    )


@dataclass
class MeanShiftClusterResult:
    """Typed report of :func:`meanshift_cluster`.

    Attributes
    ----------
    labels:
        Integer membership per sample (no noise concept -- the honest
        gap versus DBSCAN).
    centers:
        Discovered modes (cluster centroids) as a DataFrame.
    bandwidth:
        The bandwidth used (supplied or estimated).
    bandwidth_source:
        ``'manual'`` or ``'estimated'``.
    diagnostics:
        The shared diagnostics block.
    n_samples:
        Input row count.
    """

    labels: Any
    centers: pd.DataFrame
    bandwidth: float
    bandwidth_source: str
    diagnostics: ClusterDiagnostics
    n_samples: int

    @property
    def n_clusters(self) -> int:
        return self.diagnostics.n_clusters

    def as_frame(self) -> pd.DataFrame:
        if isinstance(self.labels, pd.Series):
            out = self.labels.to_frame(name="cluster")
        else:
            out = pd.DataFrame({"cluster": np.asarray(self.labels)})
        for column in self.centers.columns:
            out[f"mode_{column}"] = out["cluster"].map(self.centers[column])
        return out

    def __str__(self) -> str:
        head = (
            f"MeanShift -- bandwidth={self.bandwidth:.3g} "
            f"({self.bandwidth_source}), {self.n_samples} rows, "
            f"{self.n_clusters} modes\n"
        )
        return head + str(self.diagnostics)


def meanshift_cluster(
    X: Any,
    bandwidth: float | None = None,
    *,
    bin_seeding: bool = False,
    cluster_all: bool = True,
) -> MeanShiftClusterResult:
    """Mean-shift mode-seeking clustering (no k, bandwidth-driven).

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array.
    bandwidth:
        Kernel bandwidth.  ``None`` (default) estimates it with
        :func:`~sklearn.cluster.estimate_bandwidth` (quantile 0.3 --
        reported honestly via ``bandwidth_source``).
    bin_seeding:
        Speed heuristic (coarse initialisation).
    cluster_all:
        When ``True`` every sample is assigned to the nearest mode;
        when ``False`` the far-out samples become orphans labelled
        ``-1`` and enter the noise column of the diagnostics.

    Returns
    -------
    MeanShiftClusterResult
    """
    matrix, names, index = _as_numeric_matrix(X, caller="meanshift_cluster")
    n_samples = matrix.shape[0]
    if n_samples < 3:
        raise DataValidationError(f"meanshift_cluster needs at least 3 rows, got {n_samples}")
    if bandwidth is None:
        bandwidth_used = float(estimate_bandwidth(matrix, quantile=0.3))
        source = "estimated"
        if bandwidth_used <= 0:
            bandwidth_used = 1.0
    else:
        bandwidth_used = float(bandwidth)
        if bandwidth_used <= 0:
            raise DataValidationError(f"bandwidth must be positive, got {bandwidth_used}")
        source = "manual"
    fitted = MeanShift(
        bandwidth=bandwidth_used or None,
        bin_seeding=bin_seeding,
        cluster_all=cluster_all,
        seeds=None,
    ).fit(matrix)
    labels_array = np.asarray(fitted.labels_)
    # MeanShift with cluster_all=False orphans as -1; keep dtype int-safe
    labels_array = labels_array.astype(int)
    centers = pd.DataFrame(np.asarray(fitted.cluster_centers_), columns=names)
    if index is not None:
        labels: Any = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    else:
        labels = labels_array
    diagnostics = _cluster_diagnostics(matrix, labels_array, caller="meanshift_cluster")
    return MeanShiftClusterResult(
        labels=labels,
        centers=centers,
        bandwidth=bandwidth_used,
        bandwidth_source=source,
        diagnostics=diagnostics,
        n_samples=n_samples,
    )


@dataclass
class SpectralResult:
    """Typed report of :func:`spectral_cluster`.

    Attributes
    ----------
    n_clusters, affinity, gamma, n_neighbors:
        The parameters used (``gamma`` only for ``rbf``, ``n_neighbors``
        only for ``nearest_neighbors``).
    labels:
        Integer membership per sample.
    diagnostics:
        The shared diagnostics block.
    n_samples:
        Input row count.
    """

    n_clusters: int
    affinity: str
    gamma: float | None
    n_neighbors: int | None
    labels: Any
    diagnostics: ClusterDiagnostics
    n_samples: int

    def as_frame(self) -> pd.DataFrame:
        if isinstance(self.labels, pd.Series):
            return self.labels.to_frame(name="cluster")
        return pd.DataFrame({"cluster": np.asarray(self.labels)})

    def __str__(self) -> str:
        extra = (
            f"gamma={self.gamma:.3g}" if self.affinity == "rbf"
            else f"n_neighbors={self.n_neighbors}"
        )
        head = (
            f"SpectralClustering -- affinity={self.affinity} ({extra}), "
            f"n_clusters={self.n_clusters}, {self.n_samples} rows\n"
        )
        return head + str(self.diagnostics)


def spectral_cluster(
    X: Any,
    n_clusters: int,
    *,
    affinity: str = "rbf",
    gamma: float = 1.0,
    n_neighbors: int = 10,
    n_init: int = 10,
    random_state: int = RANDOM_STATE,
) -> SpectralResult:
    """Spectral clustering over the sample-similarity graph.

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array.
    n_clusters:
        Explicit cluster count (spectral clustering has no honest
        auto-k; choosing it is the analyst's call).
    affinity:
        ``'rbf'`` (with ``gamma``) or ``'nearest_neighbors'`` (with
        ``n_neighbors``) -- the graph's edge rule.
    gamma:
        RBF kernel scale (``affinity='rbf'`` only).
    n_neighbors:
        k for the mutual-kNN graph (``affinity='nearest_neighbors'``
        only).
    n_init:
        k-means restarts inside the embedding (pinned for stability).
    random_state:
        Seed (the toolkit's 42 by default).

    Returns
    -------
    SpectralResult

    Notes
    -----
    Deliberately distinct from the *embedding* in
    :func:`pharmaml.ml.manifold.manifold_embedding` -- this object
    assigns cluster labels; that one returns coordinates.
    """
    matrix, _names, index = _as_numeric_matrix(X, caller="spectral_cluster")
    n_samples = matrix.shape[0]
    if n_samples < 3:
        raise DataValidationError(f"spectral_cluster needs at least 3 rows, got {n_samples}")
    if affinity not in SPECTRAL_AFFINITIES:
        raise DataValidationError(
            f"affinity must be one of {SPECTRAL_AFFINITIES}, got {affinity!r}"
        )
    n_clusters = int(n_clusters)
    if not 2 <= n_clusters <= n_samples:
        raise DataValidationError(
            f"n_clusters must be in [2, n_samples={n_samples}], got {n_clusters}"
        )
    if gamma <= 0:
        raise DataValidationError(f"gamma must be positive, got {gamma}")
    if n_neighbors < 1:
        raise DataValidationError(f"n_neighbors must be >= 1, got {n_neighbors}")
    if affinity == "nearest_neighbors" and n_neighbors >= n_samples:
        raise DataValidationError(
            f"n_neighbors={n_neighbors} must stay below n_samples={n_samples}"
        )
    estimator = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        gamma=float(gamma) if affinity == "rbf" else 1.0,
        n_neighbors=int(n_neighbors),
        n_init=n_init,
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels_array = np.asarray(estimator.fit_predict(matrix)).astype(int)
    if index is not None:
        labels: Any = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    else:
        labels = labels_array
    diagnostics = _cluster_diagnostics(matrix, labels_array, caller="spectral_cluster")
    return SpectralResult(
        n_clusters=n_clusters,
        affinity=affinity,
        gamma=float(gamma) if affinity == "rbf" else None,
        n_neighbors=int(n_neighbors) if affinity == "nearest_neighbors" else None,
        labels=labels,
        diagnostics=diagnostics,
        n_samples=n_samples,
    )


__all__ = [
    "SPECTRAL_AFFINITIES",
    "ClusterDiagnostics",
    "DbscanResult",
    "MeanShiftClusterResult",
    "SpectralResult",
    "dbscan_cluster",
    "meanshift_cluster",
    "spectral_cluster",
]

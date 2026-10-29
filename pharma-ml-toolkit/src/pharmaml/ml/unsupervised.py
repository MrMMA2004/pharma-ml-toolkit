"""Unsupervised analysis: PCA, k-means, Gaussian mixtures and KDE.

The Python Data Science Handbook's unsupervised chapters (05.09
Principal-Component-Analysis, 05.11 K-Means, 05.12 Gaussian-Mixtures,
05.13 Kernel-Density-Estimation) were the last whole block of standard
data-science practice with no surface in this toolkit: every estimator
in :data:`~pharmaml.ml.models.MODEL_REGISTRY` is supervised.  This
module closes that gap with four typed functions and their record
classes -- the same "fit, return a report object, plot separately"
discipline the supervised side follows:

* :func:`pca_reduce` -- PCA with the explained-variance ladder and the
  loadings table (the handbook's "choosing the number of components"
  and "what do the components mean" sections).
* :func:`kmeans_cluster` -- k-means (or mini-batch k-means) with the
  inertia elbow AND the silhouette scan, because the elbow alone is
  famously subjective; with ``k=None`` the scan picks the k with the
  best silhouette and reports the whole table so the choice stays
  auditable.
* :func:`gmm_cluster` -- Gaussian mixture models with the AIC/BIC
  model-selection sweep the handbook uses (BIC wins the default vote),
  the four covariance types, and the responsibility matrix
  (``predict_proba``) so "which cluster is compound 47 really in?"
  has an honest probabilistic answer.
* :func:`kde_fit` -- kernel density estimation with the bandwidth
  chosen by cross-validated log-likelihood (the handbook's own
  ``GridSearchCV`` over ``bandwidth``), the exact grid it scored, and
  the Silverman fallback for tiny samples where CV cannot be trusted.

The honest scope, stated once: density-based clustering (DBSCAN /
mean-shift) and the Bayesian-GMM / GMM-as-generator halves of chapter
05.12 stay OUT of this module -- they are standing stretch candidates,
not silently-half-implemented features.  Every function threads
``random_state=42`` (the toolkit's pinned seed) so a rerun reproduces
the same labels, the same bandwidth, the same components.

The plotting twins live in :mod:`pharmaml.viz.ml_plots`
(:func:`~pharmaml.viz.ml_plots.plot_pca_variance`,
:func:`~pharmaml.viz.ml_plots.plot_kmeans_diagnostics`,
:func:`~pharmaml.viz.ml_plots.plot_gmm_selection`,
:func:`~pharmaml.viz.ml_plots.plot_kde_density`).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from pharmaml.ml.unsupervised import pca_reduce
>>> rng = np.random.default_rng(42)
>>> frame = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
>>> frame["e"] = 2.0 * frame["a"] - frame["b"]          # one dominant axis
>>> result = pca_reduce(frame, n_components=2)
>>> result.n_components
2
>>> round(float(result.explained_variance_ratio[0]), 2) > 0.4   # the a/b/e axis dominates
True
>>> result.transformed.shape
(80, 2)

>>> from pharmaml.ml.unsupervised import kmeans_cluster
>>> blob = pd.DataFrame(
...     {"x": np.concatenate([rng.normal(0, 0.4, 60), rng.normal(4, 0.4, 60)]),
...      "y": np.concatenate([rng.normal(0, 0.4, 60), rng.normal(4, 0.4, 60)])}
... )
>>> km = kmeans_cluster(blob, k=2)
>>> int(km.labels.nunique())
2
>>> round(float(km.silhouette), 1) > 0.5                  # two clean blobs
True

>>> from pharmaml.ml.unsupervised import kde_fit
>>> density = kde_fit(pd.Series(rng.normal(size=120)))
>>> density.bandwidth > 0
True
>>> len(density.density)
200
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through every estimator (the toolkit's pin).
RANDOM_STATE = 42

#: k-means restart count -- pinned because sklearn >= 1.4's ``auto``
#: resolves differently for k-means++ vs random init; an explicit 10
#: keeps results stable across the supported sklearn range.
_KMEANS_N_INIT = 10

#: Covariance parameterisations :func:`gmm_cluster` accepts (the
#: handbook's "covariance type" section, figure 05.12).
GMM_COVARIANCE_TYPES: tuple[str, ...] = ("full", "tied", "diag", "spherical")

#: Kernels :func:`kde_fit` accepts (scikit-learn's own set).
KDE_KERNELS: tuple[str, ...] = (
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
)


def _as_numeric_matrix(X: Any, *, caller: str) -> tuple[np.ndarray, list[str], pd.Index | None]:
    """Validate and convert ``X`` to a float matrix with column names.

    Accepts a DataFrame, a Series (one column) or a 2-D array-like.
    Refuses NaN/±inf and non-numeric dtypes with the typed error --
    the supervised side imputes inside its pipeline, but an
    unsupervised fit has no target to validate against afterwards, so
    the input must be complete up front (:func:`~pharmaml.ml.preprocessing.clean_numeric_frame`
    or the GUI's dataset cleaner do this before calling).
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
            f"{caller}: {bad} row(s) contain NaN or ±inf -- impute or drop them "
            "before running unsupervised analysis (clean_numeric_frame helps)"
        )
    names = [str(column) for column in frame.columns]
    index = frame.index if had_frame else None
    return matrix, names, index


@dataclass
class PcaResult:
    """Typed report of :func:`pca_reduce`.

    Attributes
    ----------
    n_components:
        Number of components actually fitted.
    explained_variance_ratio:
        Per-component share of the total variance (0-1 each).
    cumulative_variance:
        Running sum of ``explained_variance_ratio``.
    total_variance_explained:
        Cumulative share at the last fitted component.
    transformed:
        The scores (samples x components), columns ``PC1``...
    loadings:
        Features x components rotation -- how each original column
        contributes to each component (the "what does PC1 mean" table).
    column_names:
        The input feature names, in loadings order.
    n_samples / n_features:
        Input shape.
    scaled:
        Whether the input was standardised first (default True -- the
        handbook's own choice whenever column scales differ, which for
        molecular descriptors they always do).
    """

    n_components: int
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    total_variance_explained: float
    transformed: pd.DataFrame
    loadings: pd.DataFrame
    column_names: list[str]
    n_samples: int
    n_features: int
    scaled: bool = True

    def as_frame(self) -> pd.DataFrame:
        """The component scores (samples x ``PC1``...) as a DataFrame."""
        return self.transformed

    def __str__(self) -> str:
        head = ", ".join(
            f"PC{i + 1} {ratio:.1%}" for i, ratio in enumerate(self.explained_variance_ratio[:5])
        )
        more = " ..." if len(self.explained_variance_ratio) > 5 else ""
        return (
            f"PCA({self.n_components} components over {self.n_features} features, "
            f"{self.n_samples} rows, scaled={self.scaled})\n"
            f"  explained variance: {head}{more}\n"
            f"  cumulative at k={self.n_components}: {self.total_variance_explained:.1%}"
        )


def pca_reduce(
    X: Any,
    n_components: int | None = None,
    *,
    scale: bool = True,
) -> PcaResult:
    """Principal component analysis (PDSH 05.09).

    Parameters
    ----------
    X:
        Numeric DataFrame / Series / 2-D array.  Complete (no NaN) --
        see :func:`~pharmaml.ml.preprocessing.clean_numeric_frame`.
    n_components:
        How many components to fit.  ``None`` (default) fits
        ``min(10, n_samples, n_features)`` and leaves the honest
        "how many are enough?" question to the cumulative-variance
        ladder (:func:`~pharmaml.viz.ml_plots.plot_pca_variance`).
    scale:
        Standardise columns first (default).  Keep it True for
        descriptor tables: MolLogP and TPSA live on wildly different
        scales and PCA's variance budget is unit-blind.

    Returns
    -------
    PcaResult
    """
    matrix, names, index = _as_numeric_matrix(X, caller="pca_reduce")
    n_samples, n_features = matrix.shape
    if scale:
        from sklearn.preprocessing import StandardScaler

        matrix = StandardScaler().fit_transform(matrix)
    cap = int(min(10, n_samples, n_features))
    if n_components is None:
        n_components = cap
    n_components = int(n_components)
    if not 1 <= n_components <= min(n_samples, n_features):
        raise DataValidationError(
            f"n_components must be in [1, min(n_samples, n_features)] = "
            f"[1, {min(n_samples, n_features)}], got {n_components}"
        )
    pca = PCA(n_components=n_components, svd_solver="full", random_state=RANDOM_STATE)
    scores = pca.fit_transform(matrix)
    component_names = [f"PC{i + 1}" for i in range(n_components)]
    transformed = pd.DataFrame(scores, index=index, columns=component_names)
    loadings = pd.DataFrame(pca.components_.T, index=names, columns=component_names)
    ratio = [float(value) for value in pca.explained_variance_ratio_]
    cumulative = list(np.cumsum(ratio))
    _LOG.info("PCA fitted %d components over %d features", n_components, n_features)
    return PcaResult(
        n_components=n_components,
        explained_variance_ratio=ratio,
        cumulative_variance=[float(value) for value in cumulative],
        total_variance_explained=float(cumulative[-1]),
        transformed=transformed,
        loadings=loadings,
        column_names=names,
        n_samples=n_samples,
        n_features=n_features,
        scaled=bool(scale),
    )


@dataclass
class KMeansResult:
    """Typed report of :func:`kmeans_cluster`.

    Attributes
    ----------
    k:
        The number of clusters used (chosen manually or by the
        silhouette scan).
    method:
        ``'kmeans'`` or ``'mini_batch'``.
    labels:
        Integer cluster membership per sample (a Series when the input
        was a DataFrame/Series, else a plain array).
    centers:
        Cluster centroids as a DataFrame (k x features).
    inertia:
        Within-cluster sum of squared distances at the chosen k.
    silhouette:
        Mean silhouette coefficient at the chosen k (``None`` when the
        scan could not compute it -- fewer distinct labels than 2).
    scan:
        The k -> inertia/silhouette sweep used for the choice (empty
        when k was supplied manually).
    sizes:
        Samples per cluster.
    n_samples:
        Input row count.
    """

    k: int
    method: str
    labels: Any
    centers: pd.DataFrame
    inertia: float
    silhouette: float | None
    scan: pd.DataFrame
    sizes: pd.Series
    n_samples: int

    def as_frame(self) -> pd.DataFrame:
        """A DataFrame of the labels joined to the centroid coordinates."""
        if isinstance(self.labels, pd.Series):
            out = self.labels.to_frame(name="cluster")
        else:
            out = pd.DataFrame({"cluster": np.asarray(self.labels)})
        for column in self.centers.columns:
            out[f"center_{column}"] = out["cluster"].map(self.centers[column])
        return out

    def __str__(self) -> str:
        chosen = (
            f"{self.k} (manual)"
            if self.scan.empty
            else f"{self.k} (best silhouette {self.silhouette:.3f} in scan)"
            if self.silhouette is not None
            else f"{self.k}"
        )
        sizes = ", ".join(f"{int(count)}" for count in self.sizes.head(8))
        return (
            f"k-means ({self.method}) -- k={chosen}, {self.n_samples} rows\n"
            f"  inertia: {self.inertia:.3g}\n"
            f"  silhouette: {'n/a' if self.silhouette is None else f'{self.silhouette:.3f}'}\n"
            f"  cluster sizes: {sizes}"
        )


def _silhouette_or_none(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean silhouette when computable, ``None`` honestly otherwise."""
    distinct = len({int(label) for label in labels})
    if distinct < 2 or distinct >= len(labels):
        return None
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(matrix, labels))


def kmeans_cluster(
    X: Any,
    k: int | None = None,
    *,
    k_range: tuple[int, int] = (2, 10),
    mini_batch: bool = False,
    n_init: int = _KMEANS_N_INIT,
    random_state: int = RANDOM_STATE,
) -> KMeansResult:
    """k-means clustering (PDSH 05.11), with the elbow AND the silhouette.

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array (complete rows).
    k:
        Fixed cluster count.  ``None`` (default) scans every k in
        ``k_range`` and keeps the one with the best mean silhouette --
        the elbow method's subjective "where does the bend sit" is
        exactly why the scan also reports inertia for plotting.
    k_range:
        Inclusive ``(low, high)`` sweep bounds when ``k=None``.
    mini_batch:
        Use :class:`~sklearn.cluster.MiniBatchKMeans` (the handbook's
        large-sample variant).  Same seed, same scan.
    n_init:
        Restarts per k (pinned at 10 for reproducibility).
    random_state:
        Seed (the toolkit's 42 by default).

    Returns
    -------
    KMeansResult
    """
    matrix, names, index = _as_numeric_matrix(X, caller="kmeans_cluster")
    n_samples = matrix.shape[0]
    if n_samples < 4:
        raise DataValidationError(f"kmeans_cluster needs at least 4 rows, got {n_samples}")

    def _fit(k_value: int) -> Any:
        if mini_batch:
            return MiniBatchKMeans(
                n_clusters=k_value,
                n_init=n_init,
                random_state=random_state,
                batch_size=min(1024, n_samples),
            ).fit(matrix)
        return KMeans(n_clusters=k_value, n_init=n_init, random_state=random_state).fit(matrix)

    scan_rows: list[dict[str, Any]] = []
    if k is None:
        low, high = int(k_range[0]), int(k_range[1])
        if low < 1 or high < low:
            raise DataValidationError(
                f"k_range must be (low, high) with 1 <= low <= high, got {k_range}"
            )
        high = min(high, n_samples - 1)
        if high < low:
            raise DataValidationError(
                f"k_range={k_range} exceeds what {n_samples} rows can support "
                f"(max k = n_samples - 1 = {n_samples - 1})"
            )
        best_k, best_sil = low, None
        for candidate in range(low, high + 1):
            fitted = _fit(candidate)
            sil = _silhouette_or_none(matrix, fitted.labels_)
            scan_rows.append({"k": candidate, "inertia": float(fitted.inertia_), "silhouette": sil})
            if sil is not None and (best_sil is None or sil > best_sil):
                best_k, best_sil = candidate, sil
        chosen_k, chosen_sil = best_k, best_sil
        _LOG.info("k-means scan chose k=%d (silhouette=%s)", best_k, best_sil)
    else:
        chosen_k = int(k)
        if not 1 <= chosen_k <= n_samples:
            raise DataValidationError(f"k must be in [1, n_samples={n_samples}], got {chosen_k}")
        chosen_sil = None

    final = _fit(chosen_k)
    if chosen_sil is None:
        chosen_sil = _silhouette_or_none(matrix, final.labels_)
    centers = pd.DataFrame(final.cluster_centers_, columns=names)
    labels_array = np.asarray(final.labels_)
    counts = np.bincount(labels_array, minlength=chosen_k)
    if index is not None:
        labels: Any = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    else:
        labels = labels_array
    sizes = pd.Series(counts, index=range(chosen_k), name="samples")
    scan = pd.DataFrame(scan_rows, columns=["k", "inertia", "silhouette"])
    return KMeansResult(
        k=chosen_k,
        method="mini_batch" if mini_batch else "kmeans",
        labels=labels,
        centers=centers,
        inertia=float(final.inertia_),
        silhouette=chosen_sil,
        scan=scan,
        sizes=sizes,
        n_samples=n_samples,
    )


@dataclass
class GmmResult:
    """Typed report of :func:`gmm_cluster`.

    Attributes
    ----------
    n_components:
        Mixture components used (manual or the BIC/AIC sweep's pick).
    covariance_type:
        One of :data:`GMM_COVARIANCE_TYPES`.
    chosen_by:
        ``'bic'`` (the sweep's default), ``'aic'`` or ``'manual'``.
    scan:
        The n_components -> AIC/BIC sweep (empty when manual).
    labels:
        Hard assignments (``predict``) per sample.
    probabilities:
        The responsibility matrix (samples x components, rows sum to 1)
        -- the honest "soft membership" the handbook leads with.
    weights:
        Mixing weights per component.
    means:
        Component means as a DataFrame (components x features).
    converged:
        Whether the final EM run converged.
    log_likelihood:
        Total log-likelihood of the data under the fitted mixture.
    n_samples:
        Input row count.
    """

    n_components: int
    covariance_type: str
    chosen_by: str
    scan: pd.DataFrame
    labels: Any
    probabilities: pd.DataFrame
    weights: pd.Series
    means: pd.DataFrame
    converged: bool
    log_likelihood: float
    n_samples: int

    def as_frame(self) -> pd.DataFrame:
        """Labels + responsibilities as one DataFrame."""
        if isinstance(self.labels, pd.Series):
            base = self.labels.to_frame(name="cluster")
        else:
            base = pd.DataFrame({"cluster": np.asarray(self.labels)})
        for column in self.probabilities.columns:
            base[column] = self.probabilities[column].to_numpy()
        return base

    def __str__(self) -> str:
        return (
            f"GaussianMixture({self.n_components} components, "
            f"covariance={self.covariance_type}, chosen_by={self.chosen_by})\n"
            f"  log-likelihood: {self.log_likelihood:.2f}\n"
            f"  converged: {self.converged}\n"
            f"  weights: {', '.join(f'{weight:.2f}' for weight in self.weights)}"
        )


def gmm_cluster(
    X: Any,
    n_components: int | None = None,
    *,
    covariance_type: str = "full",
    component_range: tuple[int, int] = (1, 8),
    criterion: str = "bic",
    random_state: int = RANDOM_STATE,
) -> GmmResult:
    """Gaussian mixture model (PDSH 05.12) with the AIC/BIC sweep.

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array (complete rows).
    n_components:
        Fixed component count.  ``None`` (default) sweeps every count
        in ``component_range`` and keeps the criterion minimum -- the
        handbook's own model-selection figure.
    covariance_type:
        ``'full'`` | ``'tied'`` | ``'diag'`` | ``'spherical'``.
    component_range:
        Inclusive sweep bounds when ``n_components`` is None.
    criterion:
        ``'bic'`` (default; heavier complexity penalty) or ``'aic'``.
    random_state:
        Seed (the toolkit's 42 by default).

    Returns
    -------
    GmmResult
    """
    matrix, names, index = _as_numeric_matrix(X, caller="gmm_cluster")
    n_samples = matrix.shape[0]
    if covariance_type not in GMM_COVARIANCE_TYPES:
        raise DataValidationError(
            f"covariance_type must be one of {GMM_COVARIANCE_TYPES}, got {covariance_type!r}"
        )
    if criterion not in {"aic", "bic"}:
        raise DataValidationError(f"criterion must be 'aic' or 'bic', got {criterion!r}")

    def _fit(count: int) -> GaussianMixture:
        return GaussianMixture(
            n_components=count,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=3,
        ).fit(matrix)

    scan_rows: list[dict[str, Any]] = []
    if n_components is None:
        low, high = int(component_range[0]), int(component_range[1])
        if low < 1 or high < low:
            raise DataValidationError(
                f"component_range must be (low, high) with 1 <= low <= high, got {component_range}"
            )
        high = min(high, n_samples)
        if high < low:
            raise DataValidationError(f"component_range={component_range} exceeds {n_samples} rows")
        best_n, best_score = low, None
        for candidate in range(low, high + 1):
            fitted = _fit(candidate)
            aic, bic = float(fitted.aic(matrix)), float(fitted.bic(matrix))
            scan_rows.append({"n_components": candidate, "aic": aic, "bic": bic})
            score = bic if criterion == "bic" else aic
            if best_score is None or score < best_score:
                best_n, best_score = candidate, score
        chosen_n, chosen_by = best_n, criterion
        _LOG.info("GMM sweep chose %d components by %s", best_n, criterion)
    else:
        chosen_n = int(n_components)
        if not 1 <= chosen_n <= n_samples:
            raise DataValidationError(
                f"n_components must be in [1, n_samples={n_samples}], got {chosen_n}"
            )
        chosen_by = "manual"

    final = _fit(chosen_n)
    labels_array = final.predict(matrix)
    probs = final.predict_proba(matrix)
    component_names = [f"comp_{i + 1}" for i in range(chosen_n)]
    probabilities = pd.DataFrame(probs, index=index, columns=component_names)
    if index is not None:
        labels: Any = pd.Series(labels_array, index=index, name="cluster", dtype=int)
    else:
        labels = labels_array
    means = pd.DataFrame(final.means_, columns=names)
    scan = pd.DataFrame(scan_rows, columns=["n_components", "aic", "bic"])
    return GmmResult(
        n_components=chosen_n,
        covariance_type=covariance_type,
        chosen_by=chosen_by,
        scan=scan,
        labels=labels,
        probabilities=probabilities,
        weights=pd.Series(final.weights_, name="weight"),
        means=means,
        converged=bool(final.converged_),
        log_likelihood=float(final.score(matrix) * n_samples),
        n_samples=n_samples,
    )


@dataclass
class KdeResult:
    """Typed report of :func:`kde_fit`.

    Attributes
    ----------
    column:
        The column (or positional name) whose density was estimated.
    bandwidth:
        The smoothing scale in the data's own units.
    bandwidth_source:
        ``'cv'`` (GridSearchCV log-likelihood), ``'silverman'`` (the
        small-sample fallback) or ``'manual'``.
    kernel:
        Kernel family (default ``'gaussian'``).
    grid:
        The even grid the density was evaluated on.
    density:
        Estimated density at each grid point (same length as grid).
    log_likelihood:
        Mean log-likelihood of the observed points under the fit.
    n_samples:
        Number of observations.
    """

    column: str
    bandwidth: float
    bandwidth_source: str
    kernel: str
    grid: np.ndarray
    density: np.ndarray
    log_likelihood: float
    n_samples: int

    def as_frame(self) -> pd.DataFrame:
        """Grid + density as a DataFrame (the curve, ready to plot)."""
        return pd.DataFrame({"x": self.grid, "density": self.density})

    def __str__(self) -> str:
        return (
            f"KDE({self.column!r}, n={self.n_samples}, kernel={self.kernel})\n"
            f"  bandwidth: {self.bandwidth:.4g} ({self.bandwidth_source})\n"
            f"  mean log-likelihood: {self.log_likelihood:.3f}"
        )


def kde_fit(
    values: Any,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    grid_size: int = 200,
    cv: int = 5,
) -> KdeResult:
    """Kernel density estimation (PDSH 05.13) for ONE numeric column.

    With ``bandwidth=None`` the bandwidth is chosen by cross-validated
    log-likelihood (``GridSearchCV`` over 20 log-spaced candidates),
    the handbook's own selection recipe; below 10 observations CV
    cannot be trusted, so the Silverman rule steps in and the record
    says so via ``bandwidth_source``.

    Parameters
    ----------
    values:
        Series / 1-column DataFrame / 1-D array.
    bandwidth:
        Fixed smoothing scale; ``None`` selects it by CV (default).
    kernel:
        One of :data:`KDE_KERNELS`.
    grid_size:
        Points on the even output grid (data range + 3 % padding).
    cv:
        CV folds for the bandwidth search (clamped to the sample size).

    Returns
    -------
    KdeResult
    """
    if kernel not in KDE_KERNELS:
        raise DataValidationError(f"kernel must be one of {KDE_KERNELS}, got {kernel!r}")
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise DataValidationError(
                f"kde_fit estimates ONE column's density -- got {values.shape[1]} "
                f"({list(values.columns)}); pass a Series or single column"
            )
        column_name = str(values.columns[0])
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        column_name = str(values.name) if values.name is not None else "values"
        series = values
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise DataValidationError(f"kde_fit expects 1-D values, got shape {array.shape}")
        column_name, series = "values", pd.Series(array)
    matrix, _names, _index = _as_numeric_matrix(
        series.to_frame() if isinstance(series, pd.Series) else np.asarray(series).reshape(-1, 1),
        caller="kde_fit",
    )
    column = column_name
    flat = matrix.reshape(-1)
    n_samples = int(flat.size)
    if n_samples < 3:
        raise DataValidationError(f"kde_fit needs at least 3 observations, got {n_samples}")
    if grid_size < 10:
        raise DataValidationError(f"grid_size must be >= 10, got {grid_size}")
    sample_std = float(np.std(flat, ddof=1)) if n_samples > 1 else 0.0
    if sample_std == 0.0:
        raise DataValidationError(
            "kde_fit refuses constant input (std=0): the density is a spike, "
            "not a curve -- drop or jitter the column"
        )

    if bandwidth is not None:
        chosen = float(bandwidth)
        if chosen <= 0:
            raise DataValidationError(f"bandwidth must be > 0, got {chosen}")
        source = "manual"
        fitted = KernelDensity(kernel=kernel, bandwidth=chosen).fit(flat.reshape(-1, 1))
    elif n_samples < 10:
        # Silverman's rule: the honest small-sample fallback (CV below
        # 10 points is noise, and the handbook's own tiny examples
        # never cross this bridge).
        chosen = float(
            0.9
            * min(sample_std, float(np.percentile(flat, 75) - np.percentile(flat, 25)) / 1.349)
            * n_samples ** (-0.2)
        )
        if chosen <= 0:
            chosen = float(max(sample_std, 1e-3) * 0.5)
        source = "silverman"
        fitted = KernelDensity(kernel=kernel, bandwidth=chosen).fit(flat.reshape(-1, 1))
    else:
        search = GridSearchCV(
            KernelDensity(kernel=kernel),
            {"bandwidth": np.logspace(-1, 1, 20)},
            cv=min(int(cv), n_samples),
        )
        search.fit(flat.reshape(-1, 1))
        fitted = search.best_estimator_
        chosen = float(fitted.bandwidth)
        source = "cv"
        _LOG.info("KDE bandwidth chosen by %d-fold CV: %.4g", min(int(cv), n_samples), chosen)

    pad = 0.03 * float(flat.max() - flat.min())
    grid = np.linspace(float(flat.min()) - pad, float(flat.max()) + pad, int(grid_size))
    density = np.exp(fitted.score_samples(grid.reshape(-1, 1)))
    return KdeResult(
        column=column,
        bandwidth=chosen,
        bandwidth_source=source,
        kernel=kernel,
        grid=grid,
        density=density,
        log_likelihood=float(fitted.score(flat.reshape(-1, 1))),
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Agglomerative clustering (the scipy.cluster.hierarchy absorption)
# ---------------------------------------------------------------------------

#: The three linkage rules hierarchical_cluster implements.
LINKAGE_METHODS = ("ward", "single", "average", "complete")


@dataclass(frozen=True)
class HierarchicalResult:
    """The cut of an agglomerative tree: labels, order, merge heights.

    ``labels`` assigns every row to a cluster (0..k-1, deterministic
    by first-appearance order); ``merge_heights`` are the k-1 heights
    at which the cut clusters formed below (scipy's Z[:,2] column);
    ``linkage`` records the rule; ``inertia`` is the within-cluster
    sum of squared Euclidean distances for the cut (Ward's objective,
    reported for every rule for comparability).
    """

    labels: np.ndarray
    k: int
    linkage: str
    merge_heights: tuple[float, ...]
    inertia: float
    n_samples: int


def _euclidean_matrix(matrix: np.ndarray) -> np.ndarray:
    squared = np.sum(matrix**2, axis=1)
    distances_sq = squared[:, None] + squared[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(distances_sq, 0.0, out=distances_sq)
    return np.sqrt(distances_sq)


def hierarchical_cluster(X: Any, k: int = 3, *, linkage: str = "ward") -> HierarchicalResult:
    """Agglomerative clustering with a k-cut -- scipy's linkage + fcluster.

    Lance-Williams agglomeration over the full distance matrix (the
    textbook O(n³) form; honest note: scipy's nn-chain is O(n²) and
    is what production uses -- this is the exact same mathematics at
    teaching scale).  ``ward`` merges by the minimum increase in
    within-cluster SSE (the height *is* that increase); the geometric
    rules (``single`` = nearest pair, ``complete`` = farthest pair,
    ``average`` = mean pair) work on any data, Euclidean distances.

    Raises
    ------
    DataValidationError
        On a malformed matrix, k outside [1, n], or an unknown linkage.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.unsupervised import hierarchical_cluster
    >>> rng = np.random.default_rng(42)
    >>> blobs = np.vstack([rng.normal(size=(30, 2)) + 4.0,
    ...                    rng.normal(size=(30, 2)) - 4.0,
    ...                    rng.normal(size=(30, 2))])
    >>> result = hierarchical_cluster(blobs, k=3)
    >>> result.k
    3
    >>> import collections
    >>> counts = collections.Counter(result.labels.tolist())
    >>> min(counts.values()) > 10        # three real blobs, found
    True
    """
    if linkage not in LINKAGE_METHODS:
        raise DataValidationError(
            f"unknown linkage {linkage!r}; valid: {', '.join(LINKAGE_METHODS)}"
        )
    matrix, _names, _index = _as_numeric_matrix(X, caller="hierarchical_cluster")
    n = matrix.shape[0]
    if not (1 <= k <= n):
        raise DataValidationError(f"k must be in [1, {n}], got {k}")
    if n < 2:
        return HierarchicalResult(
            labels=np.zeros(n, dtype=int),
            k=1,
            linkage=linkage,
            merge_heights=(),
            inertia=0.0,
            n_samples=n,
        )

    distances = _euclidean_matrix(matrix)
    # cluster state: id -> member row indices; lance heights accumulate
    members: dict[int, list[int]] = {index: [index] for index in range(n)}
    active = set(range(n))
    centroid = {index: matrix[index].copy() for index in range(n)}
    count = {index: 1 for index in range(n)}
    next_id = n
    heights: dict[int, float] = {}
    children_pairs: dict[int, tuple[int, int]] = {}

    while len(active) > 1:
        best_pair: tuple[int, int] | None = None
        best_height = math.inf
        for a in sorted(active):
            for b in sorted(active):
                if b <= a:
                    continue
                if linkage == "ward":
                    # the exact Ward height: |A||B|/(|A|+|B|) * ||centroid diff||^2
                    delta = (
                        count[a]
                        * count[b]
                        / (count[a] + count[b])
                        * float(np.sum((centroid[a] - centroid[b]) ** 2))
                    )
                elif linkage == "single":
                    delta = min(distances[i, j] for i in members[a] for j in members[b])
                elif linkage == "complete":
                    delta = max(distances[i, j] for i in members[a] for j in members[b])
                else:  # average
                    delta = float(
                        np.mean([distances[i, j] for i in members[a] for j in members[b]])
                    )
                if delta < best_height:
                    best_height = delta
                    best_pair = (a, b)
        if best_pair is None:  # pragma: no cover - len(active) > 1 guarantees one
            raise DataValidationError("linkage search failed unexpectedly")
        a, b = best_pair
        merged = members[a] + members[b]
        centroid[next_id] = matrix[merged].mean(axis=0)
        count[next_id] = len(merged)
        members[next_id] = merged
        heights[next_id] = best_height
        children_pairs[next_id] = (a, b)
        active.discard(a)
        active.discard(b)
        active.add(next_id)
        next_id += 1

    # the cut: greedy top-down k-clusters (scipy fcluster's maxclust
    # semantics) -- start at the root, repeatedly split the merge with
    # the largest height until k clusters exist
    root = next_id - 1
    clusters: list[int] = [root]
    while len(clusters) < k:
        # the current cluster ids that are internal merges (splittable)
        splittable = [cid for cid in clusters if cid in children_pairs]
        if not splittable:
            break  # every cluster is a singleton already
        target = max(splittable, key=lambda cid: heights[cid])
        child_a, child_b = children_pairs[target]
        clusters.remove(target)
        clusters.extend([child_a, child_b])
    labels = np.zeros(n, dtype=int)
    ordered = sorted(
        (cid for cid in clusters if members.get(cid)),
        key=lambda cid: min(members[cid]),
    )
    for cluster_id, cid in enumerate(ordered):
        for row in members[cid]:
            labels[row] = cluster_id
    inertia = 0.0
    for cluster_id in range(len(set(labels.tolist()))):
        mask = labels == cluster_id
        if mask.any():
            inertia += float(np.sum((matrix[mask] - matrix[mask].mean(axis=0)) ** 2))
    result = HierarchicalResult(
        labels=labels,
        k=int(labels.max() + 1) if n else 0,
        linkage=linkage,
        merge_heights=tuple(
            float(heights[cid])
            for cid in sorted(heights.keys(), key=lambda cid: heights[cid])[-(k - 1) :]
            if cid in children_pairs and cid not in clusters
        ),
        inertia=float(inertia),
        n_samples=int(n),
    )
    _LOG.debug(
        "hierarchical cluster: n=%d k=%d linkage=%s inertia=%.4g",
        n,
        result.k,
        linkage,
        inertia,
    )
    return result


__all__ = [
    "GMM_COVARIANCE_TYPES",
    "KDE_KERNELS",
    "LINKAGE_METHODS",
    "RANDOM_STATE",
    "GmmResult",
    "HierarchicalResult",
    "KMeansResult",
    "KdeResult",
    "PcaResult",
    "gmm_cluster",
    "hierarchical_cluster",
    "kde_fit",
    "kmeans_cluster",
    "pca_reduce",
]

"""UMAP -- lmcinnes/umap, absorbed offline (numpy + scipy + sklearn).

Round 186 checked the umap-learn repository file by file and ported the
algorithm's own machinery, the parts that make UMAP *UMAP* rather than
another neighbour graph projection:

* **``smooth_knn_dist``** (``umap_.py``) -- the binary search for each
  point's normaliser ``sigma_i`` such that the fuzzy k-NN cardinality
  hits ``log2(k)``, with ``rho_i`` (the local-connectivity distance)
  subtracted before the exponent -- Vitter-free, numba-free, the exact
  64-iteration bisection with the doubling lower bracket.
* **``compute_membership_strengths``** (``umap_.py``) -- the local
  fuzzy simplicial set: ``exp(-(d - rho) / sigma)`` per edge, then the
  fuzzy union symmetrisation ``A + A^T - A . A^T``.
* **``find_ab_params``** (``umap_.py``) -- the differentiable curve
  ``1 / (1 + a x^{2b})`` fitted to the offset exponential decay that
  ``min_dist`` and ``spread`` define, via the very same
  ``scipy.optimize.curve_fit`` call.
* **``optimize_layout_euclidean``** (``layouts.py``) -- the SGD itself,
  gradient-for-gradient: attraction ``-2ab d^{2b-1} / (a d^{2b} + 1)``
  and repulsion ``2 gamma b / ((0.001 + d^2)(a d^{2b} + 1))`` on
  squared distances, gradients clipped to +-4, the negative-sampling
  rate, the per-edge ``epochs_per_sample`` firing schedule
  (``make_epochs_per_sample``: ``n_epochs . w_max / w``), and the linear
  learning-rate decay ``alpha (1 - epoch / n_epochs)``.
* **``transform``** (``umap_.py``) -- new points initialised at the
  membership-weighted average of their neighbours' positions, then a
  short SGD with the old embedding frozen.
* **``trustworthiness``** (``validation.py``) -- the Venna-Kaski score,
  rank-based, same normalisation constant ``2 / (n k (2n - 3k - 1))``.

The spectral initialisation delegates to sklearn's
``SpectralEmbedding(affinity="precomputed")`` over the symmetrised
graph (what umap's ``spectral.py`` computes with its own LOBPCG), with
the honest documented fallback to the random init when the graph is too
disconnected for the eigensolver.

The honest scope, stated once: the numba/nndescent acceleration ( ANN
indices, parallel SGD, ``sparse_matrix`` inputs), densMAP,
parametric/aligned UMAP, label propagation and the plot helpers are not
vendored -- brute-force distances keep the whole pipeline numpy-clean
for the sample sizes this toolkit meets in practice.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.umap import umap_embed, find_ab_params, trustworthiness
>>> rng = np.random.default_rng(42)
>>> circle = np.column_stack([
...     np.cos(np.linspace(0, 2 * np.pi, 60, endpoint=False)),
...     np.sin(np.linspace(0, 2 * np.pi, 60, endpoint=False)),
... ])
>>> record = umap_embed(circle, n_neighbors=8, n_epochs=50, random_state=42)
>>> record.embedding.shape
(60, 2)
>>> a, b = find_ab_params(1.0, 0.1)
>>> 0.5 < a < 2.5 and 0.5 < b < 1.0     # the curve fit's stable region
True
>>> 0.0 <= trustworthiness(circle, record.embedding, n_neighbors=8) <= 1.0
True
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from pharmaml.exceptions import DataValidationError

#: The metrics this port computes brute-force (umap's ``named_distances``
#: reduced to the three that matter for feature tables).
UMAP_METRICS = ("euclidean", "manhattan", "cosine")

#: ``smooth_knn_dist``'s bisection tolerance (umap's SMOOTH_K_TOLERANCE).
SMOOTH_K_TOLERANCE = 1e-5

#: The SGD gradient clip (umap's ``clip`` -- a hard +-4).
GRADIENT_CLIP = 4.0

#: Initial learning rate (umap's default ``learning_rate``).
INITIAL_ALPHA = 1.0

#: Off-graph distance floor in the repulsion denominator (0.001 in layouts.py).
REPULSION_EPSILON = 1e-3


def _as_matrix(X: Any, caller: str) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise DataValidationError(
            f"{caller}: needs a 2-D numeric matrix, got shape {array.shape}"
        )
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise DataValidationError(
            f"{caller}: needs at least 2 rows and 1 feature, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise DataValidationError(f"{caller}: input contains NaN or infinity")
    return array


def pairwise_distances(
    X: np.ndarray, metric: str, Y: np.ndarray | None = None
) -> np.ndarray:
    """The distance matrix (brute force, numpy): symmetric when ``Y`` is
    ``None``, cross (rows of ``X`` x rows of ``Y``) otherwise."""
    if metric not in UMAP_METRICS:
        raise DataValidationError(
            f"metric must be one of {UMAP_METRICS}, got {metric!r}"
        )
    other = X if Y is None else Y
    if metric == "euclidean":
        gram = X @ other.T
        x_diag = np.einsum("ij,ij->i", X, X)
        y_diag = np.einsum("ij,ij->i", other, other)
        d2 = x_diag[:, None] + y_diag[None, :] - 2.0 * gram
        if Y is None:
            # the symmetric diagonal is exactly zero (BLAS rounding of
            # a + a - 2a is not); sklearn zeroes it too.
            np.fill_diagonal(d2, 0.0)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)
    if metric == "manhattan":
        return np.abs(X[:, None, :] - other[None, :, :]).sum(axis=2)
    # cosine: 1 - (x . y) / (|x| |y|), NaN-safe on zero vectors
    x_norms = np.linalg.norm(X, axis=1)
    y_norms = np.linalg.norm(other, axis=1)
    x_safe = np.where(x_norms > 0, x_norms, 1.0)
    y_safe = np.where(y_norms > 0, y_norms, 1.0)
    sims = (X @ other.T) / np.outer(x_safe, y_safe)
    return np.clip(1.0 - sims, 0.0, None)


def smooth_knn_dist(
    distances: np.ndarray,
    k: float,
    *,
    n_iter: int = 64,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point sigma and rho -- umap_.py's ``smooth_knn_dist``.

    Binary-searches ``sigma_i`` so the fuzzy cardinality
    ``sum_j exp(-(d_ij - rho_i)/sigma_i)`` reaches ``log2(k) * bandwidth``
    (j running over the neighbours past the first), where ``rho_i`` is
    the ``local_connectivity``-th smallest positive distance.  Returns
    ``(sigmas, rhos)``.
    """
    n_points, n_neighbors = distances.shape
    target = np.log2(k) * bandwidth
    sigmas = np.zeros(n_points, dtype=float)
    rhos = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        row = distances[i]
        non_zero = np.sort(row[row > 0.0])
        if non_zero.size >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho = non_zero[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE and index < non_zero.size:
                    rho += interpolation * (
                        non_zero[index] - non_zero[index - 1]
                    )
            else:
                rho = interpolation * non_zero[0]
        elif non_zero.size > 0:
            rho = non_zero[-1]
        else:
            rho = 0.0
        rhos[i] = rho

        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(int(n_iter)):
            psum = 0.0
            for j in range(1, n_neighbors):
                d = row[j] - rho
                psum += np.exp(-(d / mid)) if d > 0 else 1.0
            if abs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2.0 if hi >= np.inf else (lo + hi) / 2.0
        sigmas[i] = mid
    return sigmas, rhos


def compute_membership_strengths(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The local fuzzy simplicial set -- umap_.py's function of the same
    name: ``exp(-(d - rho) / sigma)`` per edge, 1.0 when ``d <= rho``.

    Returns the COO triple ``(rows, cols, vals)`` over directed edges.
    """
    n_points, n_neighbors = knn_indices.shape
    rows = np.zeros(n_points * n_neighbors, dtype=np.int64)
    cols = np.zeros(n_points * n_neighbors, dtype=np.int64)
    vals = np.zeros(n_points * n_neighbors, dtype=float)
    for i in range(n_points):
        for j in range(n_neighbors):
            neighbour = knn_indices[i, j]
            if neighbour == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = float(np.exp(-((knn_dists[i, j] - rhos[i]) / sigmas[i])))
            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = neighbour
            vals[i * n_neighbors + j] = val
    return rows, cols, vals


def fuzzy_simplicial_set(
    X: np.ndarray,
    n_neighbors: int,
    *,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The symmetrised fuzzy graph -- umap's ``fuzzy_simplicial_set``.

    Local membership strengths (per-row k-NN), then the fuzzy union
    ``A + A^T - A . A^T`` accumulated over unique edges.  Returns the
    COO triple ``(rows, cols, weights)`` with ``rows < cols``.
    """
    n_points = X.shape[0]
    n_neighbors = min(int(n_neighbors), n_points - 1)
    if n_neighbors < 2:
        raise DataValidationError(
            "n_neighbors must be >= 2 (and below the row count), got "
            f"{n_neighbors} with {n_points} rows"
        )
    distances = pairwise_distances(X, metric)
    if metric == "cosine":
        # Ties at 0 (identical rows) still need distinct neighbour slots.
        distances = distances + np.eye(n_points) * 1e-12
    knn_indices = np.argsort(distances, axis=1)[:, : n_neighbors + 1]
    knn_dists = np.take_along_axis(distances, knn_indices, axis=1)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(n_neighbors))
    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    # directed strengths per unordered pair, then the fuzzy union
    # A + A^T - A . A^T:  w_ij + w_ji - w_ij * w_ji  (never > 1).
    # Self-edges (i == i) carry strength 0 and never enter the graph.
    directed: dict[tuple[int, int], list[float]] = {}
    for i, j, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        if i == j or v <= 0.0:
            continue
        key = (i, j) if i < j else (j, i)
        directed.setdefault(key, []).append(v)
    keys = list(directed)
    out_rows = np.array([k[0] for k in keys], dtype=np.int64)
    out_cols = np.array([k[1] for k in keys], dtype=np.int64)
    out_vals = np.array(
        [
            (s[0] + s[1] - s[0] * s[1]) if len(s) == 2 else s[0]
            for s in directed.values()
        ],
        dtype=float,
    )
    return out_rows, out_cols, out_vals


def find_ab_params(spread: float = 1.0, min_dist: float = 0.1) -> tuple[float, float]:
    """Fit the (a, b) curve -- umap_.py's ``find_ab_params`` verbatim.

    ``curve(x) = 1 / (1 + a x^{2b})`` is least-squares-fitted to the
    offset exponential ``exp(-(x - min_dist) / spread)`` past ``min_dist``
    (1.0 before it), over ``x in [0, 3 spread]``.
    """
    if spread <= 0.0 or min_dist < 0.0:
        raise DataValidationError(
            f"spread must be > 0 and min_dist >= 0, got {spread}, {min_dist}"
        )

    def curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _covar = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def _spectral_init(
    rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, n_components: int,
    n_points: int, random_state: int,
) -> np.ndarray | None:
    """sklearn SpectralEmbedding over the fuzzy graph (umap's
    ``spectral.py`` role); ``None`` when the eigensolver refuses a too-
    disconnected graph -- the caller falls back to random honestly."""
    from sklearn.manifold import SpectralEmbedding

    affinity = np.zeros((n_points, n_points), dtype=float)
    affinity[rows, cols] = weights
    affinity[cols, rows] = weights
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedded = SpectralEmbedding(
                n_components=n_components,
                affinity="precomputed",
                random_state=random_state,
            ).fit_transform(affinity)
        return np.asarray(embedded, dtype=float)
    except Exception:  # noqa: BLE001 - honest fallback, documented
        return None


def _optimize_layout_euclidean(
    head: np.ndarray,
    tail: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    weights: np.ndarray,
    *,
    a: float,
    b: float,
    n_epochs: int,
    negative_sample_rate: int,
    initial_alpha: float,
    gamma: float,
    rng: np.random.Generator,
    move_other: bool = True,
    frozen: np.ndarray | None = None,
) -> np.ndarray:
    """The SGD -- layouts.py's ``optimize_layout_euclidean`` in numpy.

    Per positive edge firing (the ``epochs_per_sample`` schedule), one
    attractive step on the squared distance; then ``negative_sample_rate``
    repulsive steps against random vertices.  Gradients clip at +-4;
    alpha decays linearly over epochs.  ``frozen`` (transform mode)
    pins the tail embedding's rows.
    """
    n_vertices = head.shape[0]
    n_edges = weights.shape[0]
    if n_edges == 0:
        return head
    # umap's make_epochs_per_sample: epochs between firings = w_max / w,
    # so an edge fires ~ n_epochs * w / w_max times in total.
    epochs_per_sample = weights.max() / np.maximum(weights, 1e-12)
    epochs_per_negative_sample = epochs_per_sample / float(negative_sample_rate)
    epoch_of_next_sample = epochs_per_sample.copy()
    epoch_of_next_negative = epochs_per_negative_sample.copy()

    for epoch in range(int(n_epochs)):
        alpha = initial_alpha * (1.0 - epoch / float(n_epochs))
        for edge in rng.permutation(n_edges):
            if epoch_of_next_sample[edge] <= epoch + 1:
                i, j = int(edge_rows[edge]), int(edge_cols[edge])
                diff = head[i] - tail[j]
                dist_squared = float(diff @ diff)
                if dist_squared > 0.0:
                    grad_coeff = (
                        -2.0
                        * a
                        * b
                        * dist_squared ** (b - 1.0)
                        / (a * dist_squared**b + 1.0)
                    )
                else:
                    grad_coeff = 0.0
                grad = np.clip(grad_coeff * diff, -GRADIENT_CLIP, GRADIENT_CLIP)
                head[i] += grad * alpha
                if move_other and (frozen is None or not frozen[j]):
                    tail[j] += -grad * alpha

                n_negatives = int(
                    (epoch + 1 - epoch_of_next_negative[edge])
                    / epochs_per_negative_sample[edge]
                )
                for _ in range(max(n_negatives, 0)):
                    k = int(rng.integers(0, n_vertices))
                    if k == i:
                        continue
                    diff = head[i] - tail[k]
                    dist_squared = float(diff @ diff)
                    if dist_squared > 0.0:
                        grad_coeff = (
                            2.0
                            * gamma
                            * b
                            / (
                                (REPULSION_EPSILON + dist_squared)
                                * (a * dist_squared**b + 1.0)
                            )
                        )
                    else:
                        continue
                    if grad_coeff > 0.0:
                        grad = np.clip(
                            grad_coeff * diff, -GRADIENT_CLIP, GRADIENT_CLIP
                        )
                        head[i] += grad * alpha

                epoch_of_next_sample[edge] += epochs_per_sample[edge]
                epoch_of_next_negative[edge] += (
                    max(n_negatives, 0) * epochs_per_negative_sample[edge]
                )
    return head


@dataclass
class UmapRecord:
    """Typed report of :func:`umap_embed` -- the fitted model, kept
    transformable.

    Attributes
    ----------
    embedding:
        The (n, n_components) layout as a DataFrame (columns
        ``UMAP1``, ``UMAP2``...).
    embedding_array:
        The same layout as a plain array (transform's frozen side).
    a / b:
        The fitted curve parameters.
    n_edges / mean_weight:
        The fuzzy graph's shape after symmetrisation.
    sigma_stats:
        ``(min, median, max)`` of the per-point normalisers.
    init:
        The initialisation actually used (spectral may fall back).
    params:
        Every knob as passed, for the record's honest replay.
    """

    embedding: Any
    embedding_array: np.ndarray
    a: float
    b: float
    n_edges: int
    mean_weight: float
    sigma_stats: tuple[float, float, float]
    init: str
    params: dict[str, Any]
    _graph: tuple[np.ndarray, np.ndarray, np.ndarray] = field(
        repr=False, default=(np.array([]), np.array([]), np.array([]))
    )
    _sigmas: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _rhos: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _raw: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _metric: str = "euclidean"
    _n_neighbors: int = 15

    def as_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            self.embedding_array,
            columns=[f"UMAP{i + 1}" for i in range(self.embedding_array.shape[1])],
        )

    def __str__(self) -> str:
        return (
            f"UMAP -- {self.embedding_array.shape[0]} rows x "
            f"{self._raw.shape[1]} features -> "
            f"{self.embedding_array.shape[1]}-D "
            f"({self.n_edges} fuzzy edges, a={self.a:.4f}, b={self.b:.4f}, "
            f"init={self.init})"
        )

    def transform(
        self,
        X_new: Any,
        *,
        n_epochs: int = 100,
        random_state: int = 42,
    ) -> np.ndarray:
        """Project new rows into the fitted embedding (umap's transform).

        Each new row's neighbours among the FITTED data get membership
        strengths from the model's own sigmas/rhos; the initial position
        is the membership-weighted average of their embedding rows; a
        short SGD then moves ONLY the new points (the old embedding is
        frozen -- ``move_other`` tail rows pinned).
        """
        new = _as_matrix(X_new, caller="UmapRecord.transform")
        n_fit = self._raw.shape[0]
        if n_fit < 3:
            raise DataValidationError(
                "transform needs a model fitted on >= 3 rows "
                "(the weighted-average init degenerates below that)"
            )
        if new.shape[1] != self._raw.shape[1]:
            raise DataValidationError(
                f"transform: {new.shape[1]} features vs the fitted "
                f"{self._raw.shape[1]}"
            )
        rng = np.random.default_rng(random_state)
        distances = pairwise_distances(new, self._metric, Y=self._raw)
        k = min(self._n_neighbors, n_fit - 1)
        order = np.argsort(distances, axis=1)[:, : k + 1]
        positions = np.zeros((new.shape[0], self.embedding_array.shape[1]))
        row_weights = np.zeros((new.shape[0], order.shape[1]))
        for r in range(new.shape[0]):
            weights_row = row_weights[r]
            for slot, col in enumerate(order[r]):
                d = distances[r, col]
                rho = self._rhos[col]
                sigma = self._sigmas[col]
                if d - rho <= 0.0 or sigma == 0.0:
                    weights_row[slot] = 1.0
                else:
                    weights_row[slot] = float(np.exp(-((d - rho) / sigma)))
            total = weights_row.sum()
            if total <= 0.0:
                positions[r] = rng.uniform(-10.0, 10.0, positions.shape[1])
            else:
                positions[r] = (
                    weights_row @ self.embedding_array[order[r]] / total
                )
        # cross-edges new->fitted, SGD over the new rows only
        rows_l: list[int] = []
        cols_l: list[int] = []
        w_l: list[float] = []
        for r in range(new.shape[0]):
            for slot, col in enumerate(order[r]):
                w = row_weights[r, slot]
                if w > 0:
                    rows_l.append(r)
                    cols_l.append(int(col))
                    w_l.append(float(w))
        combined = np.vstack([self.embedding_array, positions])
        head = combined
        tail = combined
        frozen = np.zeros(n_fit + new.shape[0], dtype=bool)
        frozen[:n_fit] = True  # the fitted side never moves
        if rows_l:
            _optimize_layout_euclidean(
                head,
                tail,
                np.asarray(rows_l, dtype=np.int64) + n_fit,
                np.asarray(cols_l, dtype=np.int64),
                np.asarray(w_l, dtype=float),
                a=self.a,
                b=self.b,
                n_epochs=int(n_epochs),
                negative_sample_rate=5,
                initial_alpha=INITIAL_ALPHA,
                gamma=1.0,
                rng=rng,
                move_other=True,
                frozen=frozen,
            )
        return combined[n_fit:]


def umap_embed(
    X: Any,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_components: int = 2,
    metric: str = "euclidean",
    n_epochs: int = 200,
    init: str = "spectral",
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    learning_rate: float = INITIAL_ALPHA,
    random_state: int = 42,
) -> UmapRecord:
    """Fit UMAP on ``X`` (the UMAP class's fit path, numpy-only).

    Every parameter keeps umap's own default unless overridden; the
    knobs' semantics are umap's (``min_dist`` the spacing floor,
    ``n_epochs`` the SGD budget, ``negative_sample_rate`` the repulsion
    multiplier).  ``init`` is ``"spectral"`` (sklearn eigensolver over
    the fuzzy graph) or ``"random"`` (uniform +-10, umap's own range).
    """
    matrix = _as_matrix(X, caller="umap_embed")
    if metric not in UMAP_METRICS:
        raise DataValidationError(
            f"metric must be one of {UMAP_METRICS}, got {metric!r}"
        )
    if init not in ("spectral", "random"):
        raise DataValidationError(
            f"init must be 'spectral' or 'random', got {init!r}"
        )
    if min_dist > spread:
        raise DataValidationError(
            f"min_dist ({min_dist}) should not exceed spread ({spread})"
        )
    if n_epochs < 1:
        raise DataValidationError(f"n_epochs must be >= 1, got {n_epochs}")
    n_points = matrix.shape[0]
    n_neighbors = int(n_neighbors)
    if not 2 <= n_neighbors <= n_points - 1:
        raise DataValidationError(
            f"n_neighbors must be in [2, n_rows - 1] = [2, {n_points - 1}], "
            f"got {n_neighbors}"
        )

    rows, cols, weights = fuzzy_simplicial_set(
        matrix, n_neighbors, metric=metric
    )
    a, b = find_ab_params(spread=spread, min_dist=min_dist)

    rng = np.random.default_rng(random_state)
    if init == "spectral":
        spectral = _spectral_init(
            rows, cols, weights, n_components, n_points, random_state
        )
        if spectral is None:
            used_init = "random (spectral fallback)"
            layout = rng.uniform(-10.0, 10.0, (n_points, n_components))
        else:
            used_init = "spectral"
            layout = 10.0 * spectral / max(np.abs(spectral).max(), 1e-12)
    else:
        used_init = "random"
        layout = rng.uniform(-10.0, 10.0, (n_points, n_components))

    if weights.shape[0]:
        _optimize_layout_euclidean(
            layout,
            layout,
            rows,
            cols,
            weights,
            a=a,
            b=b,
            n_epochs=int(n_epochs),
            negative_sample_rate=int(negative_sample_rate),
            initial_alpha=float(learning_rate),
            gamma=float(repulsion_strength),
            rng=rng,
            move_other=True,
        )

    sigmas, rhos = smooth_knn_dist_for_record(matrix, n_neighbors, metric)
    record = UmapRecord(
        embedding=None,
        embedding_array=np.asarray(layout, dtype=float),
        a=a,
        b=b,
        n_edges=int(weights.shape[0]),
        mean_weight=float(weights.mean()) if weights.size else 0.0,
        sigma_stats=(
            float(sigmas.min()),
            float(np.median(sigmas)),
            float(sigmas.max()),
        ),
        init=used_init,
        params={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "spread": spread,
            "n_components": n_components,
            "metric": metric,
            "n_epochs": n_epochs,
            "init": init,
            "negative_sample_rate": negative_sample_rate,
            "repulsion_strength": repulsion_strength,
            "learning_rate": learning_rate,
            "random_state": random_state,
        },
        _graph=(rows, cols, weights),
        _sigmas=sigmas,
        _rhos=rhos,
        _raw=matrix,
        _metric=metric,
        _n_neighbors=n_neighbors,
    )
    record.embedding = record.as_frame()
    return record


def smooth_knn_dist_for_record(
    matrix: np.ndarray, n_neighbors: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive (sigmas, rhos) over the full k-NN matrix -- the model
    state transform reuses."""
    n_points = matrix.shape[0]
    distances = pairwise_distances(matrix, metric)
    if metric == "cosine":
        distances = distances + np.eye(n_points) * 1e-12
    knn = np.argsort(distances, axis=1)[:, : n_neighbors + 1]
    knn_dists = np.take_along_axis(distances, knn, axis=1)
    return smooth_knn_dist(knn_dists, float(n_neighbors))


def trustworthiness(
    X: Any, embedding: Any, *, n_neighbors: int = 5
) -> float:
    """Venna-Kaski trustworthiness at one k -- umap/validation.py's
    formula (sklearn's own ``manifold.trustworthiness`` agrees): each
    embedded k-NN whose SOURCE-space rank (0-based, self excluded)
    reaches past ``k`` contributes ``rank - k`` to the penalty,

    ``T(k) = 1 - 2 / (n k (2n - 3k - 1)) * penalty``.

    1.0 = the neighbourhoods survived the projection intact.
    """
    source = _as_matrix(X, caller="trustworthiness")
    embed = np.asarray(embedding, dtype=float)
    if embed.ndim != 2 or embed.shape[0] != source.shape[0]:
        raise DataValidationError(
            "trustworthiness: embedding rows must match the source rows "
            f"({embed.shape} vs {source.shape})"
        )
    n_samples = source.shape[0]
    k = int(n_neighbors)
    if not 1 <= k < n_samples / 2:
        raise DataValidationError(
            f"n_neighbors must be in [1, n/2) = [1, {n_samples / 2}), got {k}"
        )

    from sklearn.neighbors import NearestNeighbors

    dist_source = pairwise_distances(source, "euclidean")
    np.fill_diagonal(dist_source, np.inf)
    ind_source = np.argsort(dist_source, axis=1)

    inverted = np.zeros((n_samples, n_samples), dtype=int)
    rows = np.arange(n_samples)
    ordered = np.arange(n_samples + 1)
    inverted[rows[:, np.newaxis], ind_source] = ordered[1:]

    ind_embedded = (
        NearestNeighbors(n_neighbors=k)
        .fit(embed)
        .kneighbors(return_distance=False)
    )
    ranks = inverted[rows[:, np.newaxis], ind_embedded] - k
    penalty = int(ranks[ranks > 0].sum())
    denominator = n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0)
    if denominator <= 0:  # pragma: no cover - guarded by the k < n/2 check
        return 1.0
    return float(1.0 - penalty * (2.0 / denominator))

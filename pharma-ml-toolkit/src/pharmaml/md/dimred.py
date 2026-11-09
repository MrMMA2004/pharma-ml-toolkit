"""Essential dynamics — trajectory PCA and tICA, the deeptime core.

``deeptime-ml/deeptime`` (and ``plumed/plumed2``'s collective-variable
machinery) reduce a trajectory to the few coordinates that carry its
variance or its *slowness*:

* :func:`trajectory_pca` — principal component analysis over the
  flattened (optionally aligned) coordinate matrix: the variance
  carriers (the "essential dynamics" of Amadei, Linssen & Berendsen);
* :func:`tica` — time-lagged independent component analysis: the
  eigenproblem of the time-lagged covariance against the instantaneous
  one, solved through the symmetric ``C₀^{-1/2}·C_τ,sym·C₀^{-1/2}``
  form so only ``numpy.linalg.eigh`` is ever needed — the coordinates
  that decorrelate *slowest*, the MSM community's input features.

Both return a :class:`ProjectionResult` (mean, components, explained
variance and ratio) with a ``transform``/``transform_trajectory``
pair, so a trajectory becomes a ``(T, k)`` projection in one call.

Examples
--------
>>> from pharmaml.md.trajectory import synthetic_trajectory
>>> t = synthetic_trajectory(n_frames=40, seed=3, noise=0.35)
>>> res = trajectory_pca(t, n_components=2)
>>> res.explained_variance_ratio[0] > res.explained_variance_ratio[1]
True
>>> res.transform_trajectory(t).shape
(40, 2)
>>> steps = np.arange(400.0)
>>> slow = np.sin(steps * 2 * np.pi / 200.0)
>>> fast = np.sin(steps * 2 * np.pi / 3.7)
>>> tic = tica(np.column_stack([slow, fast]), lag=8, n_components=1)
>>> bool(abs(tic.components[0, 0]) > abs(tic.components[0, 1]))
True
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md.align import superpose
from pharmaml.md.trajectory import (
    Trajectory,
    resolve_atom_indices,
)

__all__ = [
    "ProjectionResult",
    "tica",
    "trajectory_pca",
]


class ProjectionResult:
    """A fitted linear projection: mean, components, explained variance.

    ``components`` rows are the projection axes (unit length, sign
    arbitrary as with every eigenvector); ``explained_variance_ratio``
    is the variance (PCA) or |eigenvalue| (tICA) share of each axis.
    """

    def __init__(
        self,
        mean: np.ndarray,
        components: np.ndarray,
        explained_variance: np.ndarray,
        explained_variance_ratio: np.ndarray,
        *,
        method: str,
    ) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.components = np.asarray(components, dtype=float)
        self.explained_variance = np.asarray(explained_variance, dtype=float)
        self.explained_variance_ratio = np.asarray(
            explained_variance_ratio, dtype=float
        )
        self.method = method
        if self.components.ndim != 2:
            raise MDAnalysisError("components must be a (k, D) matrix")
        if self.mean.shape[0] != self.components.shape[1]:
            raise MDAnalysisError("mean length must match the component width")
        if (
            self.components.shape[0] != self.explained_variance.shape[0]
            or self.components.shape[0] != self.explained_variance_ratio.shape[0]
        ):
            raise MDAnalysisError("explained variance length must match k")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Project a ``(M, D)`` matrix onto the components → ``(M, k)``."""
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.mean.shape[0]:
            raise MDAnalysisError(
                f"data must have shape (n, {self.mean.shape[0]}), got {arr.shape}"
            )
        return (arr - self.mean) @ self.components.T

    def transform_trajectory(
        self,
        traj: Trajectory,
        atom_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Project a trajectory's frames (selection honoured) → ``(T, k)``."""
        idx = resolve_atom_indices(atom_indices, traj.n_atoms)
        flat = traj.coords[:, idx, :].reshape(traj.n_frames, -1)
        return self.transform(flat)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return (
            f"ProjectionResult(method={self.method!r}, k={self.components.shape[0]}, "
            f"D={self.components.shape[1]})"
        )


def _flatten_trajectory(
    traj: Trajectory,
    atom_indices: Sequence[int] | np.ndarray | None,
    *,
    align: bool,
) -> np.ndarray:
    idx = resolve_atom_indices(atom_indices, traj.n_atoms)
    frames = traj.coords[:, idx, :]
    if align:
        reference = frames[0]
        frames = np.array([superpose(frame, reference) for frame in frames])
    return frames.reshape(traj.n_frames, -1)


def trajectory_pca(
    traj: Trajectory,
    n_components: int = 2,
    *,
    atom_indices: Sequence[int] | np.ndarray | None = None,
    align: bool = True,
) -> ProjectionResult:
    """PCA of the trajectory coordinates (essential dynamics).

    Frames are optionally superposed onto the first frame (rigid-body
    motion removed before variance is measured — the MDAnalysis
    ``pca`` convention), flattened to ``(T, 3·n_selected)``, centred,
    and reduced through the SVD of the centred matrix: the principal
    axes are the right singular vectors and the explained variances
    the squared singular values over ``T − 1``.

    >>> from pharmaml.md.trajectory import synthetic_trajectory
    >>> t = synthetic_trajectory(n_frames=20, seed=6, noise=0.3)
    >>> res = trajectory_pca(t, n_components=3)
    >>> bool(np.all(np.diff(res.explained_variance_ratio) <= 0))
    True
    """
    if n_components < 1:
        raise MDAnalysisError(f"n_components must be >= 1, got {n_components}")
    x = _flatten_trajectory(traj, atom_indices, align=align)
    if x.shape[0] < 2:
        raise MDAnalysisError("PCA needs at least two frames")
    mean = x.mean(axis=0)
    if np.allclose(x - mean, 0.0):
        raise MDAnalysisError("the trajectory is degenerate (zero variance)")
    _, singulars, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = min(n_components, vt.shape[0])
    if k < n_components:
        raise MDAnalysisError(
            f"only {k} components exist for {x.shape[0]} frames — asked for "
            f"{n_components}"
        )
    components = vt[:k]
    explained = (singulars[:k] ** 2) / (x.shape[0] - 1)
    total = float(np.sum(singulars**2) / (x.shape[0] - 1))
    ratio = explained / total
    return ProjectionResult(
        mean=mean,
        components=components,
        explained_variance=explained,
        explained_variance_ratio=ratio,
        method="pca",
    )


def tica(
    data: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    *,
    ridge: float | None = None,
) -> ProjectionResult:
    """Time-lagged independent component analysis of a ``(T, D)`` matrix.

    The time-lagged covariance ``C_τ`` (symmetrised, as the MSM
    community standardises it) is diagonalised against the
    instantaneous covariance ``C₀`` through the symmetric matrix
    ``C₀^{-1/2}·C_τ,sym·C₀^{-1/2}``, so the whole solve is
    ``numpy.linalg.eigh``.  Eigenvalues are autocorrelations in
    roughly ``[-1, 1]`` for stationary input; components are ranked by
    eigenvalue (slowest first), and ``explained_variance_ratio``
    reports each axis' |λ| share — the *kinetic* variance.

    ``ridge`` (default ``1e-10·trace(D)``) regularises ``C₀`` for
    degenerate inputs instead of failing the inversion.

    >>> steps = np.arange(200.0)
    >>> data = np.column_stack([np.sin(steps / 20.0), np.sin(steps / 1.3)])
    >>> res = tica(data, lag=5, n_components=1)
    >>> res.transform(data).shape
    (200, 1)
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise MDAnalysisError("tica needs a (T, D) matrix with T >= 2")
    if lag < 1:
        raise MDAnalysisError(f"lag must be >= 1, got {lag}")
    if x.shape[0] <= lag:
        raise MDAnalysisError(
            f"lag ({lag}) must leave at least one time-lagged pair"
        )
    if n_components < 1:
        raise MDAnalysisError(f"n_components must be >= 1, got {n_components}")

    n_pairs = x.shape[0] - lag
    centered = x - x.mean(axis=0)
    left = centered[:-lag]
    right = centered[lag:]
    c00 = (left.T @ left) / (n_pairs - 1)
    c11 = (right.T @ right) / (n_pairs - 1)
    c0 = 0.5 * (c00 + c11)
    c0t = (left.T @ right) / (n_pairs - 1)
    c0t_sym = 0.5 * (c0t + c0t.T)

    scale = float(np.trace(c0)) / c0.shape[0]
    if scale == 0.0:
        raise MDAnalysisError("tica input has zero variance")
    ridge_value = 1e-10 * scale if ridge is None else ridge
    c0_reg = c0 + np.eye(c0.shape[0]) * ridge_value

    # Symmetric square-root inverse of C0: eigh is all we need.
    evals, evecs = np.linalg.eigh(c0_reg)
    evals = np.maximum(evals, ridge_value)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    a_matrix = inv_sqrt @ c0t_sym @ inv_sqrt
    a_matrix = 0.5 * (a_matrix + a_matrix.T)

    w, v = np.linalg.eigh(a_matrix)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]

    k = min(n_components, x.shape[1])
    if k < n_components:
        raise MDAnalysisError(
            f"only {k} components exist for D={x.shape[1]} — asked for {n_components}"
        )
    components = (inv_sqrt @ v[:, :k]).T
    ratio = np.abs(w[:k]) / float(np.sum(np.abs(w))) if np.sum(np.abs(w)) > 0 else (
        np.zeros(k)
    )
    return ProjectionResult(
        mean=x.mean(axis=0),
        components=components,
        explained_variance=w[:k],
        explained_variance_ratio=ratio,
        method="tica",
    )

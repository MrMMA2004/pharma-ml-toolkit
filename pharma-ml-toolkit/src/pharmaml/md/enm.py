"""Elastic network models — ProDy's ANM and GNM in plain NumPy.

``prody/ProDy`` built its name on the anisotropic and Gaussian network
models: a protein's slow global modes, predicted from a *single*
structure, in milliseconds.  Both models are sparse matrices over the
Cα contact graph — exactly the scale plain NumPy owns:

* :func:`kirchhoff_matrix` — the GNM's Kirchhoff (Laplacian) matrix:
  ``−γ`` for contacted pairs, ``γ·degree`` on the diagonal;
* :func:`gnm_fluctuations` — per-atom mean-square fluctuations from
  the pseudo-inverse (the B-factor profile);
* :func:`cross_correlation` — the normalised fluctuation
  cross-correlation matrix (the correlated-domain map);
* :func:`anm_hessian` — the 3N×3N anisotropic Hessian with its
  ``−γ·rrᵀ/‖r‖²`` pair blocks;
* :func:`anm_modes` — the slowest non-trivial modes (the six rigid-body
  zero modes skipped, with an honest error when the network is too
  disconnected to supply the requested count);
* :func:`anm_fluctuations` — the combined-mode fluctuation profile.

Disconnection is handled the ProDy way: the pseudo-inverse carries the
zero modes, and :func:`anm_modes` refuses (with a clear error) rather
than returning rigid-body noise as "slow modes".

Examples
--------
>>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
>>> kirchhoff = kirchhoff_matrix(coords, cutoff=1.5)
>>> bool(np.allclose(kirchhoff.sum(axis=1), 0.0))
True
>>> (float(kirchhoff[0, 1]), kirchhoff[0, 2] == 0)
(-1.0, True)
>>> fl = gnm_fluctuations(coords, cutoff=1.5)
>>> bool(np.all(fl > 0) and fl[1] < fl[0])
True
>>> corr = cross_correlation(coords, cutoff=1.5)
>>> bool(np.allclose(np.diag(corr), 1.0))
True
>>> hessian = anm_hessian(coords, cutoff=1.5)
>>> (hessian.shape, bool(np.allclose(hessian.sum(axis=1), 0.0, atol=1e-10)))
((9, 9), True)
>>> tri = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.7, 1.3, 0.0]])
>>> w, modes = anm_modes(tri, n_modes=3, cutoff=2.0)
>>> (w.shape, modes.shape, bool(np.all(w > 0)))
((3,), (3, 9), True)
"""

from __future__ import annotations

import numpy as np

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md.metrics import pairwise_distances

__all__ = [
    "ANM_CUTOFF_DEFAULT",
    "anm_fluctuations",
    "anm_hessian",
    "anm_modes",
    "cross_correlation",
    "gnm_fluctuations",
    "kirchhoff_matrix",
]

ANM_CUTOFF_DEFAULT = 10.0
"""ProDy's default Cα cutoff (Å) for the elastic network."""


def _as_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise MDAnalysisError(
            f"coords must have shape (n_atoms, 3), got {arr.shape}"
        )
    return arr


def kirchhoff_matrix(
    coords: np.ndarray, cutoff: float = ANM_CUTOFF_DEFAULT, gamma: float = 1.0
) -> np.ndarray:
    """The GNM Kirchhoff matrix over the contact graph.

    Entry ``(i, j)`` is ``−γ`` when atoms *i* and *j* lie within
    ``cutoff`` Å (i ≠ j); the diagonal carries ``γ`` times the contact
    degree.  Row sums are exactly zero.

    >>> chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> g = kirchhoff_matrix(chain, cutoff=1.5, gamma=2.0)
    >>> (float(g[0, 0]), float(g[1, 1]), float(g[0, 1]))
    (2.0, 4.0, -2.0)
    """
    arr = _as_coords(coords)
    if cutoff <= 0:
        raise MDAnalysisError(f"cutoff must be positive, got {cutoff}")
    if gamma <= 0:
        raise MDAnalysisError(f"gamma must be positive, got {gamma}")
    distances = pairwise_distances(arr)
    adjacency = (distances <= cutoff) & ~np.eye(arr.shape[0], dtype=bool)
    matrix = -gamma * adjacency.astype(float)
    np.fill_diagonal(matrix, gamma * adjacency.sum(axis=1))
    return matrix


def gnm_fluctuations(
    coords: np.ndarray,
    cutoff: float = ANM_CUTOFF_DEFAULT,
    gamma: float = 1.0,
) -> np.ndarray:
    """Per-atom GNM mean-square fluctuations (relative, kT/γ units).

    The diagonal of the Kirchhoff matrix's pseudo-inverse: the profile
    ProDy's ``gnm`` plots against B-factors.  Disconnected components
    ride the pseudo-inverse, exactly as ProDy handles them.

    >>> chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> fl = gnm_fluctuations(chain, cutoff=1.5)
    >>> bool(np.all(fl > 0))
    True
    """
    kirchhoff = kirchhoff_matrix(coords, cutoff=cutoff, gamma=gamma)
    inverse = np.linalg.pinv(kirchhoff)
    return np.clip(np.diag(inverse), 0.0, None)


def cross_correlation(
    coords: np.ndarray,
    cutoff: float = ANM_CUTOFF_DEFAULT,
    gamma: float = 1.0,
) -> np.ndarray:
    """Normalised GNM fluctuation cross-correlation matrix.

    ``C_ij / √(C_ii·C_jj)`` from the Kirchhoff pseudo-inverse — the
    correlated-motion map (diagonal exactly 1, entries in ``[-1, 1]``
    up to rounding).

    >>> chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> corr = cross_correlation(chain, cutoff=1.5)
    >>> bool(np.all(np.abs(corr) <= 1.0 + 1e-9))
    True
    """
    kirchhoff = kirchhoff_matrix(coords, cutoff=cutoff, gamma=gamma)
    inverse = np.linalg.pinv(kirchhoff)
    diagonal = np.sqrt(np.clip(np.diag(inverse), 1e-30, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = inverse / np.outer(diagonal, diagonal)
    return np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)


def anm_hessian(
    coords: np.ndarray,
    cutoff: float = ANM_CUTOFF_DEFAULT,
    gamma: float = 1.0,
) -> np.ndarray:
    """The 3N×3N anisotropic network Hessian.

    For every contacted pair ``(i, j)`` the off-diagonal block is
    ``−γ·rᵢⱼrᵢⱼᵀ/‖rᵢⱼ‖²``; diagonal blocks accumulate ``+γ`` times the
    same outer products, so every row sums to zero (the six
    translational/rotational zero modes follow).

    >>> chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> h = anm_hessian(chain, cutoff=1.5)
    >>> bool(np.allclose(h.sum(axis=1), 0.0, atol=1e-12))
    True
    """
    arr = _as_coords(coords)
    if cutoff <= 0:
        raise MDAnalysisError(f"cutoff must be positive, got {cutoff}")
    if gamma <= 0:
        raise MDAnalysisError(f"gamma must be positive, got {gamma}")
    n = arr.shape[0]
    hessian = np.zeros((3 * n, 3 * n))
    cutoff_sq = cutoff * cutoff
    for i in range(n):
        for j in range(i + 1, n):
            delta = arr[i] - arr[j]
            dist_sq = float(delta @ delta)
            if dist_sq == 0.0 or dist_sq > cutoff_sq:
                continue
            block = gamma * np.outer(delta, delta) / dist_sq
            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * j, 3 * j + 3)
            hessian[si, sj] -= block
            hessian[sj, si] -= block
            hessian[si, si] += block
            hessian[sj, sj] += block
    return hessian


def anm_modes(
    coords: np.ndarray,
    n_modes: int = 3,
    *,
    cutoff: float = ANM_CUTOFF_DEFAULT,
    gamma: float = 1.0,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """The slowest non-trivial ANM modes: ``(eigenvalues, mode vectors)``.

    The Hessian's eigenvalues ascend from the six rigid-body zero
    modes; this returns the next ``n_modes`` (rows of the second return
    are the mode vectors, each a flat 3N displacement field).  A
    network too disconnected to supply ``n_modes`` non-trivial modes
    raises rather than smuggling rigid-body motion into the answer
    (a *collinear* system has more than six zero modes and is caught
    by the same guard).

    >>> tri = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.7, 1.3, 0.0]])
    >>> w, modes = anm_modes(tri, n_modes=3, cutoff=2.0)
    >>> (w.shape, modes.shape, bool(np.all(w > 0)))
    ((3,), (3, 9), True)
    """
    if n_modes < 1:
        raise MDAnalysisError(f"n_modes must be >= 1, got {n_modes}")
    hessian = anm_hessian(coords, cutoff=cutoff, gamma=gamma)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    nontrivial = np.flatnonzero(eigenvalues > tolerance)
    if nontrivial.size < n_modes:
        raise MDAnalysisError(
            f"the network supplies only {nontrivial.size} non-trivial modes "
            f"(< {n_modes}) — it is too disconnected; raise the cutoff"
        )
    chosen = nontrivial[:n_modes]
    return eigenvalues[chosen], eigenvectors[:, chosen].T


def anm_fluctuations(
    coords: np.ndarray,
    cutoff: float = ANM_CUTOFF_DEFAULT,
    gamma: float = 1.0,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Per-atom ANM fluctuations summed over every non-trivial mode.

    ``Σ_modes ‖u_mode[3i:3i+3]‖² / λ_mode`` — the anisotropic B-factor
    profile, in relative kT/γ units.

    >>> tri = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.7, 1.3, 0.0]])
    >>> fl = anm_fluctuations(tri, cutoff=2.0)
    >>> (fl.shape, bool(np.all(fl > 0)))
    ((3,), True)
    """
    arr = _as_coords(coords)
    hessian = anm_hessian(arr, cutoff=cutoff, gamma=gamma)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    nontrivial = eigenvalues > tolerance
    if not np.any(nontrivial):
        raise MDAnalysisError(
            "the network has no non-trivial modes — check the cutoff"
        )
    n = arr.shape[0]
    fluctuations = np.zeros(n)
    for mode in np.flatnonzero(nontrivial):
        vector = eigenvectors[:, mode]
        per_atom = (vector.reshape(n, 3) ** 2).sum(axis=1)
        fluctuations += per_atom / eigenvalues[mode]
    return fluctuations

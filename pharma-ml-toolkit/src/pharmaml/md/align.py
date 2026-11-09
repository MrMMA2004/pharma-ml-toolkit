"""Kabsch optimal superposition — the alignment every RMSD needs.

Both ``MDAnalysis/mdanalysis`` (``align.rotation_matrix``) and
``mdtraj/mdtraj`` (``md.rmsd``) bottom out in the same theorem: the
rotation minimising ``‖P·R − Q‖`` over corresponding point sets is
``R = U·diag(1, 1, d)·Vᵀ`` from the SVD of ``PᵀQ`` (Kabsch, 1976 —
with the ``d`` reflection guard for a proper rotation).  This module
is that theorem in plain NumPy, with the translations handled around
it:

* :func:`kabsch_rotation` — the raw optimal rotation;
* :func:`superpose` — a translated+rotated copy of the mobile set
  fitted onto the reference;
* :func:`rmsd` — the root-mean-square deviation *after* optimal
  superposition (rotation- and translation-invariant, the MDAnalysis
  ``rms`` convention).

Examples
--------
>>> pts = np.array(
...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]]
... )
>>> q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
>>> rmsd(pts @ q + np.array([5.0, -2.0, 3.0]), pts) < 1e-10
True
>>> rmsd(pts, pts) < 1e-12
True
"""

from __future__ import annotations

import numpy as np

from pharmaml.exceptions import MDAnalysisError

__all__ = [
    "kabsch_rotation",
    "rmsd",
    "superpose",
]


def _as_coords(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise MDAnalysisError(
            f"{name} must have shape (n_atoms, 3) with n_atoms >= 1, got {arr.shape}"
        )
    return arr


def kabsch_rotation(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Optimal rotation ``R`` with ``(mobile − mean)·R ≈ (reference − mean)``.

    The Kabsch algorithm: SVD of the cross-covariance ``H = PᵀQ`` gives
    ``H = U·Σ·Vᵀ`` and ``R = U·diag(1, 1, d)·Vᵀ`` with
    ``d = sign(det(U·Vᵀ))`` guarding against a reflection.  The
    translation (centring) is the caller's concern — :func:`superpose`
    wraps it.

    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> bool(np.allclose(kabsch_rotation(pts @ q, pts), q.T, atol=1e-10))
    True
    """
    p = _as_coords(mobile, "mobile")
    q = _as_coords(reference, "reference")
    if p.shape != q.shape:
        raise MDAnalysisError(
            f"mobile {p.shape} and reference {q.shape} must match"
        )
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = float(np.sign(np.linalg.det(u @ vt)))
    if d == 0.0:  # pragma: no cover - orthogonal factors cannot be singular
        d = 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def superpose(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the mobile coordinates optimally fitted onto the reference.

    The result is ``(mobile − mean(mobile))·R + mean(reference)`` —
    MDAnalysis' ``rotation_matrix`` + translation applied in one call.
    Degenerate (collinear/small) sets keep the SVD's deterministic
    answer rather than raising.

    >>> a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> b = np.array([[10.0, 5.0, 0.0], [10.0, 7.0, 0.0]])
    >>> fitted = superpose(a, b)
    >>> rmsd(fitted, b) < 1e-10
    True
    """
    p = _as_coords(mobile, "mobile")
    q = _as_coords(reference, "reference")
    if p.shape != q.shape:
        raise MDAnalysisError(
            f"mobile {p.shape} and reference {q.shape} must match"
        )
    rotation = kabsch_rotation(p, q)
    return (p - p.mean(axis=0)) @ rotation + q.mean(axis=0)


def rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """RMSD after optimal superposition (Å, per-atom root mean square).

    >>> a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> rmsd(a, a + np.array([100.0, -3.0, 7.0])) < 1e-10
    True
    """
    p = _as_coords(mobile, "mobile")
    q = _as_coords(reference, "reference")
    if p.shape != q.shape:
        raise MDAnalysisError(
            f"mobile {p.shape} and reference {q.shape} must match"
        )
    diff = superpose(p, q) - q
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

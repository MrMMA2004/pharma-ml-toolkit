"""Trajectory metrics — the GROMACS/cpptraj analysis vocabulary.

The twelve panels ``gmx rms/rmsf/gyrate/sasa/hbond`` and ``cpptraj``
print for every simulation, in plain NumPy over the
:class:`~pharmaml.md.trajectory.Trajectory` stack:

* :func:`rmsd_trajectory` — per-frame RMSD to a reference after Kabsch
  superposition (``gmx rms``);
* :func:`rmsf` — per-atom root-mean-square fluctuations about the
  aligned mean (``gmx rmsf``);
* :func:`radius_of_gyration` / :func:`radius_of_gyration_trajectory`
  (``gmx gyrate``), with optional mass weighting;
* :func:`center_of_mass`;
* :func:`pairwise_distances`, :func:`contact_map` (frequency over the
  trajectory) and :func:`native_contact_fraction` (the Q-value of
  folding analyses);
* :func:`radial_distribution_function` — g(r) with the finite-system
  reference-volume convention;
* :func:`dihedral` — the torsion angle four atoms define (the backbone
  φ/ψ/ω of every structure movie).

Every function honours the shared ``atom_indices`` resolver, so
"backbone-only" is one ``traj.topology.select_atoms("name CA")`` away.

Examples
--------
>>> from pharmaml.md.trajectory import synthetic_trajectory
>>> t = synthetic_trajectory(n_frames=12, seed=1, noise=0.4)
>>> r = rmsd_trajectory(t)
>>> (r.shape[0], r[0] < 1e-12)
(12, True)
>>> fl = rmsf(t)
>>> fl.shape
(30,)
>>> round(radius_of_gyration(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])), 6)
1.0
>>> d = pairwise_distances(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
>>> float(d[0, 1])
3.0
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md.align import rmsd, superpose
from pharmaml.md.trajectory import (
    Trajectory,
    resolve_atom_indices,
)

__all__ = [
    "center_of_mass",
    "contact_map",
    "dihedral",
    "native_contact_fraction",
    "pairwise_distances",
    "radial_distribution_function",
    "radius_of_gyration",
    "radius_of_gyration_trajectory",
    "rmsd_trajectory",
    "rmsf",
]


def _reference_frame(
    traj: Trajectory, reference: np.ndarray | None
) -> np.ndarray:
    if reference is None:
        return traj.coords[0].copy()
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (traj.n_atoms, 3):
        raise MDAnalysisError(
            f"reference must have shape ({traj.n_atoms}, 3), got {ref.shape}"
        )
    return ref


def rmsd_trajectory(
    traj: Trajectory,
    reference: np.ndarray | None = None,
    atom_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame RMSD (Å) to the reference, after optimal superposition.

    ``gmx rms``: the reference defaults to the first frame (so the first
    value is exactly 0.0); pass a reference structure to measure against
    an experimental pose.

    >>> from pharmaml.md.trajectory import synthetic_trajectory
    >>> t = synthetic_trajectory(n_frames=5, seed=2, noise=0.5)
    >>> r = rmsd_trajectory(t, atom_indices=[0, 1, 2])
    >>> (r.shape, r[0] < 1e-12)
    ((5,), True)
    """
    idx = resolve_atom_indices(atom_indices, traj.n_atoms)
    ref = _reference_frame(traj, reference)[idx]
    return np.array([rmsd(frame[idx], ref) for frame in traj.coords], dtype=float)


def rmsf(
    traj: Trajectory,
    reference: np.ndarray | None = None,
    atom_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-atom root-mean-square fluctuation (Å) about the aligned mean.

    ``gmx rmsf``: every frame is first superposed onto the reference
    (first frame by default) so rigid-body drift never masquerades as
    flexibility.

    >>> from pharmaml.md.trajectory import synthetic_trajectory
    >>> t = synthetic_trajectory(n_frames=30, seed=4, noise=0.3)
    >>> fl = rmsf(t)
    >>> fl.shape
    (30,)
    >>> loop = rmsf(t, atom_indices=list(range(12, 21)))
    >>> helix = rmsf(t, atom_indices=[0, 1, 2, 3])
    >>> bool(np.mean(loop) > np.mean(helix))
    True
    """
    idx = resolve_atom_indices(atom_indices, traj.n_atoms)
    ref = _reference_frame(traj, reference)[idx]
    aligned = np.array([superpose(frame[idx], ref) for frame in traj.coords])
    mean = aligned.mean(axis=0)
    diff = aligned - mean
    return np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))


def center_of_mass(
    coords: np.ndarray, masses: np.ndarray | None = None
) -> np.ndarray:
    """Geometric centre, or centre of mass when weights are given.

    >>> center_of_mass(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    array([1., 0., 0.])
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise MDAnalysisError(
            f"coords must have shape (n_atoms, 3), got {arr.shape}"
        )
    if masses is None:
        return arr.mean(axis=0)
    w = np.asarray(masses, dtype=float)
    if w.shape != (arr.shape[0],):
        raise MDAnalysisError(
            f"masses must have shape ({arr.shape[0]},), got {w.shape}"
        )
    return (arr * w[:, None]).sum(axis=0) / w.sum()


def radius_of_gyration(
    coords: np.ndarray, masses: np.ndarray | None = None
) -> float:
    """Radius of gyration (Å): mass-weighted RMS distance to the centre.

    >>> round(radius_of_gyration(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])), 6)
    1.0
    >>> round(
    ...     radius_of_gyration(
    ...         np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([1.0, 3.0])
    ...     ),
    ...     6,
    ... )
    0.866025
    """
    arr = np.asarray(coords, dtype=float)
    com = center_of_mass(arr, masses)
    delta = arr - com
    if masses is None:
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    w = np.asarray(masses, dtype=float)
    return float(np.sqrt(np.sum(w[:, None] * delta * delta) / w.sum()))


def radius_of_gyration_trajectory(
    traj: Trajectory,
    atom_indices: Sequence[int] | np.ndarray | None = None,
    masses: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame radius of gyration (``gmx gyrate``).

    >>> from pharmaml.md.trajectory import synthetic_trajectory
    >>> t = synthetic_trajectory(n_frames=6, seed=5, noise=0.2)
    >>> radius_of_gyration_trajectory(t).shape
    (6,)
    """
    idx = resolve_atom_indices(atom_indices, traj.n_atoms)
    return np.array(
        [radius_of_gyration(frame[idx], masses) for frame in traj.coords],
        dtype=float,
    )


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Full ``(n, n)`` distance matrix for one frame.

    >>> d = pairwise_distances(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    >>> (float(d[0, 1]), float(d[1, 1]))
    (3.0, 0.0)
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise MDAnalysisError(
            f"coords must have shape (n_atoms, 3), got {arr.shape}"
        )
    delta = arr[:, None, :] - arr[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def contact_map(
    traj: Trajectory,
    cutoff: float = 4.5,
    atom_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Symmetric contact-frequency matrix over the trajectory.

    Entry ``(i, j)`` is the fraction of frames in which atoms *i* and *j*
    sit within ``cutoff`` Å of each other (``getcontacts``' contact map
    over a whole run); the diagonal is 0 by convention — an atom is not
    its own contact.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    >>> frames = np.stack([base, base * 2.0])  # second frame: 6/12/6 Å gaps
    >>> topo = Topology(atom_names=["A", "B", "C"])
    >>> cm = contact_map(Trajectory(frames, topo), cutoff=4.5)
    >>> [round(float(v), 3) for v in (cm[0, 1], cm[0, 2], cm[1, 2])]
    [0.5, 0.0, 0.5]
    """
    if cutoff <= 0:
        raise MDAnalysisError(f"cutoff must be positive, got {cutoff}")
    idx = resolve_atom_indices(atom_indices, traj.n_atoms)
    n = idx.shape[0]
    counts = np.zeros((n, n), dtype=float)
    for frame in traj.coords:
        d = pairwise_distances(frame[idx])
        contacts = (d <= cutoff).astype(float)
        np.fill_diagonal(contacts, 0.0)
        counts += contacts
    return counts / float(traj.n_frames)


def native_contact_fraction(
    traj: Trajectory,
    reference: np.ndarray | None = None,
    cutoff: float = 4.5,
    atom_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame fraction of native contacts kept (the Q-value).

    Native contacts are the reference frame's contact set (first frame
    by default); ``Q(t)`` is the fraction still present at frame *t*.
    A reference with no contacts at all yields all-zero Q — folding
    analyses start from a folded reference for exactly this reason.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    >>> apart = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    >>> topo = Topology(atom_names=["A", "B", "C"])
    >>> q = native_contact_fraction(
    ...     Trajectory(np.stack([base, apart, base]), topo), cutoff=4.5
    ... )
    >>> [round(float(v), 3) for v in q]
    [1.0, 0.0, 1.0]
    """
    idx = resolve_atom_indices(atom_indices, traj.n_atoms)
    ref = _reference_frame(traj, reference)[idx]

    def _contact_pairs(coords: np.ndarray) -> set[tuple[int, int]]:
        d = pairwise_distances(coords)
        close = (d <= cutoff) & (d > 0.0)
        return {
            (i, j) for i in range(idx.shape[0]) for j in range(i + 1, idx.shape[0])
            if close[i, j]
        }

    native = _contact_pairs(ref)
    if not native:
        return np.zeros(traj.n_frames, dtype=float)
    return np.array(
        [len(_contact_pairs(frame[idx]) & native) / len(native) for frame in traj.coords],
        dtype=float,
    )


def radial_distribution_function(
    a: np.ndarray,
    b: np.ndarray,
    *,
    r_max: float = 10.0,
    bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair radial distribution function g(r) between two point sets.

    Distances from every point of ``a`` to every point of ``b`` are
    histogrammed and normalised by the ideal-gas expectation at the
    density ``n_b / V`` with ``V`` the reference sphere ``4/3·π·r_max³``
    — the standard finite-system convention when no periodic box is
    known (a documented choice: with a box, use its volume instead).

    Returns ``(bin_centers, g)``.

    >>> r, g = radial_distribution_function(
    ...     np.zeros((1, 3)), np.array([[3.0, 0.0, 0.0]]), r_max=5.0, bins=5
    ... )
    >>> (int(np.argmax(g)), bool(np.all(g[:3] == 0.0)))
    (3, True)
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    for arr, name in ((arr_a, "a"), (arr_b, "b")):
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            raise MDAnalysisError(
                f"{name} must have shape (n_points, 3) with n_points >= 1, "
                f"got {arr.shape}"
            )
    if r_max <= 0:
        raise MDAnalysisError(f"r_max must be positive, got {r_max}")
    if bins < 1:
        raise MDAnalysisError(f"bins must be >= 1, got {bins}")

    delta = arr_a[:, None, :] - arr_b[None, :, :]
    dists = np.sqrt(np.sum(delta * delta, axis=2)).ravel()
    counts, edges = np.histogram(dists, bins=int(bins), range=(0.0, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    shell_volume = 4.0 * np.pi * centers**2 * dr
    volume = (4.0 / 3.0) * np.pi * r_max**3
    density = arr_b.shape[0] / volume
    expected = density * shell_volume * arr_a.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(expected > 0, counts / np.where(expected > 0, expected, 1.0), 0.0)
    return centers, g


def dihedral(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Torsion angle (degrees) of the bond ``p1–p2`` for atoms ``p0..p3``.

    The praxeolitic formulation: project the flanking bonds onto the
    plane perpendicular to the central bond and take the signed angle
    between them — 0° eclipsed (cis), ±180° trans.  Same-side flanking
    atoms give 0°, opposite sides ±180°.

    >>> p1, p2 = np.zeros(3), np.array([1.0, 0.0, 0.0])
    >>> round(dihedral(np.array([0.0, 1.0, 0.0]), p1, p2, np.array([1.0, 1.0, 0.0])), 6)
    0.0
    >>> round(dihedral(np.array([0.0, 1.0, 0.0]), p1, p2, np.array([1.0, -1.0, 0.0])), 6)
    180.0
    """
    a0 = np.asarray(p0, dtype=float)
    a1 = np.asarray(p1, dtype=float)
    a2 = np.asarray(p2, dtype=float)
    a3 = np.asarray(p3, dtype=float)
    for arr in (a0, a1, a2, a3):
        if arr.shape != (3,):
            raise MDAnalysisError("dihedral atoms must be 3-vectors")

    b0 = -(a1 - a0)
    b1 = a2 - a1
    b2 = a3 - a2
    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0.0:
        raise MDAnalysisError("dihedral needs a non-degenerate central bond")
    b1 /= b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))

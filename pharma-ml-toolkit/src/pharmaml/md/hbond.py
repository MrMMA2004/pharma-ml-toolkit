"""Geometric hydrogen bonds and salt bridges — ``gmx hbond`` in plain NumPy.

``gromacs/gromacs``' ``gmx hbond`` and ``getcontacts/getcontacts``
share the same geometric criterion, and it is the one
``Amber-MD/cpptraj``'s default mask uses too: a donor–acceptor pair is
an H-bond when the heavy-atom distance is under a cutoff (3.5 Å by
default) *and* the donor–hydrogen···acceptor angle is wide enough
(130° by default — the H must point at the acceptor).  This module is
that criterion over explicit donor/acceptor indices:

* :func:`find_hbonds` — one frame → the list of (donor, acceptor) pairs;
* :func:`hbond_timeline` — per-frame H-bond counts, the series behind
  the uploaded pipeline's ``hbnum_intermolecular``/``hbnum_intramolecular``
  panels;
* :func:`salt_bridges` — the charged-residue contact pairs
  (ASP/GLU ↔ LYS/ARG) ``getcontacts`` tracks, reduced to a distance
  criterion over caller-supplied indices.

Examples
--------
>>> coords = np.array(
...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [5.0, 0.0, 0.0]]
... )
>>> find_hbonds(coords, donors=[(0, 1)], acceptors=[2, 3, 4])
[(0, 2)]
>>> charged = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
>>> salt_bridges(charged, positive=[0], negative=[1, 2])
[(0, 1)]
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md.trajectory import Trajectory

__all__ = [
    "HBOND_ANGLE_CUTOFF",
    "HBOND_DISTANCE_CUTOFF",
    "SALT_BRIDGE_CUTOFF",
    "find_hbonds",
    "hbond_timeline",
    "salt_bridges",
]

HBOND_DISTANCE_CUTOFF = 3.5
"""Default donor–acceptor heavy-atom distance cutoff (Å)."""

HBOND_ANGLE_CUTOFF = 130.0
"""Default minimum donor–H···acceptor angle (degrees)."""

SALT_BRIDGE_CUTOFF = 4.0
"""Default charged-pair distance for a salt bridge (Å)."""


def _as_frame(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise MDAnalysisError(
            f"coords must have shape (n_atoms, 3), got {arr.shape}"
        )
    return arr


def _validate_donors_acceptors(
    donors: Sequence[tuple[int, int]],
    acceptors: Sequence[int],
    n_atoms: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    donor_pairs: list[tuple[int, int]] = []
    for pair in donors:
        if len(pair) != 2:
            raise MDAnalysisError(
                "each donor must be a (heavy_atom, hydrogen) index pair"
            )
        d_idx, h_idx = int(pair[0]), int(pair[1])
        if d_idx == h_idx:
            raise MDAnalysisError("donor heavy atom and hydrogen indices must differ")
        for idx in (d_idx, h_idx):
            if not 0 <= idx < n_atoms:
                raise MDAnalysisError(
                    f"donor index {idx} out of range for {n_atoms} atoms"
                )
        donor_pairs.append((d_idx, h_idx))
    acceptor_list = [int(a) for a in acceptors]
    for a_idx in acceptor_list:
        if not 0 <= a_idx < n_atoms:
            raise MDAnalysisError(
                f"acceptor index {a_idx} out of range for {n_atoms} atoms"
            )
    return donor_pairs, acceptor_list


def find_hbonds(
    coords: np.ndarray,
    donors: Sequence[tuple[int, int]],
    acceptors: Sequence[int],
    *,
    distance_cutoff: float = HBOND_DISTANCE_CUTOFF,
    angle_cutoff: float = HBOND_ANGLE_CUTOFF,
) -> list[tuple[int, int]]:
    """Detect one frame's H-bonds: (donor, acceptor) index pairs, sorted.

    The criterion is ``gmx hbond``'s: heavy-atom donor–acceptor distance
    ≤ ``distance_cutoff`` AND donor–H···acceptor angle ≥
    ``angle_cutoff`` (angle measured at the hydrogen).  An acceptor that
    *is* the donor or its hydrogen never counts.

    >>> frame = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> find_hbonds(frame, donors=[(0, 1)], acceptors=[2])
    [(0, 2)]
    >>> find_hbonds(frame, donors=[(0, 1)], acceptors=[2], angle_cutoff=170.0)
    [(0, 2)]
    >>> find_hbonds(frame, donors=[(0, 1)], acceptors=[2], distance_cutoff=1.5)
    []
    """
    frame = _as_frame(coords)
    if distance_cutoff <= 0:
        raise MDAnalysisError(f"distance_cutoff must be positive, got {distance_cutoff}")
    if not 0.0 <= angle_cutoff <= 180.0:
        raise MDAnalysisError(f"angle_cutoff must be in [0, 180], got {angle_cutoff}")
    donor_pairs, acceptor_list = _validate_donors_acceptors(
        donors, acceptors, frame.shape[0]
    )

    bonds: list[tuple[int, int]] = []
    for d_idx, h_idx in donor_pairs:
        donor_pos, hyd_pos = frame[d_idx], frame[h_idx]
        for a_idx in acceptor_list:
            if a_idx == d_idx or a_idx == h_idx:
                continue
            acceptor_pos = frame[a_idx]
            heavy_delta = donor_pos - acceptor_pos
            if float(np.linalg.norm(heavy_delta)) > distance_cutoff:
                continue
            v_dh = donor_pos - hyd_pos
            v_ah = acceptor_pos - hyd_pos
            norm = float(np.linalg.norm(v_dh)) * float(np.linalg.norm(v_ah))
            if norm == 0.0:
                continue
            cosine = float(np.clip(np.dot(v_dh, v_ah) / norm, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            if angle >= angle_cutoff:
                bonds.append((d_idx, a_idx))
    return sorted(set(bonds))


def hbond_timeline(
    traj: Trajectory,
    donors: Sequence[tuple[int, int]],
    acceptors: Sequence[int],
    *,
    distance_cutoff: float = HBOND_DISTANCE_CUTOFF,
    angle_cutoff: float = HBOND_ANGLE_CUTOFF,
) -> np.ndarray:
    """Per-frame H-bond counts — the ``hbnum`` series over a trajectory.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> far = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    >>> topo = Topology(atom_names=["D", "H", "A"])
    >>> list(hbond_timeline(Trajectory(np.stack([base, far]), topo),
    ...                     donors=[(0, 1)], acceptors=[2]))
    [1.0, 0.0]
    """
    return np.array(
        [
            float(
                len(
                    find_hbonds(
                        frame,
                        donors,
                        acceptors,
                        distance_cutoff=distance_cutoff,
                        angle_cutoff=angle_cutoff,
                    )
                )
            )
            for frame in traj.coords
        ],
        dtype=float,
    )


def salt_bridges(
    coords: np.ndarray,
    positive: Sequence[int],
    negative: Sequence[int],
    *,
    cutoff: float = SALT_BRIDGE_CUTOFF,
) -> list[tuple[int, int]]:
    """Charged-opposite pairs within ``cutoff`` Å — the salt-bridge list.

    ``getcontacts``' ionic interaction, reduced to the distance
    criterion: caller supplies the indices of positive (LYS/ARG/HIS
    charged nitrogens) and negative (ASP/GLU oxygens) atoms.

    >>> charged = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    >>> salt_bridges(charged, positive=[0], negative=[1, 2])
    [(0, 1)]
    """
    frame = _as_frame(coords)
    if cutoff <= 0:
        raise MDAnalysisError(f"cutoff must be positive, got {cutoff}")
    pos = [int(p) for p in positive]
    neg = [int(n) for n in negative]
    for idx in (*pos, *neg):
        if not 0 <= idx < frame.shape[0]:
            raise MDAnalysisError(
                f"charged-atom index {idx} out of range for {frame.shape[0]} atoms"
            )
    pairs: list[tuple[int, int]] = []
    for p_idx in pos:
        for n_idx in neg:
            if p_idx == n_idx:
                continue
            delta = frame[p_idx] - frame[n_idx]
            if float(np.linalg.norm(delta)) <= cutoff:
                pairs.append((p_idx, n_idx))
    return sorted(set(pairs))

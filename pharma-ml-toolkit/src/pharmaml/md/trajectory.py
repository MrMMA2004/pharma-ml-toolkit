"""Trajectory containers and atom selections — the MDAnalysis/mdtraj core.

``MDAnalysis/mdanalysis`` and ``mdtraj/mdtraj`` define the shared shape
of every Python MD stack: a :class:`Trajectory` is a stack of frames —
an ``(n_frames, n_atoms, 3)`` coordinate array — plus a
:class:`Topology` carrying per-atom metadata, and analysis is selection
+ metric over that stack.  This module is that shape in plain NumPy:

* :class:`Topology` — atom names, residue names/ids and optional masses
  with a real selection mini-language (``"name CA"``, ``"resname ALA
  GLY"``, ``"resid 3:12"``, ``"not name H*"`` combined with ``and`` /
  ``or`` and parentheses, ``*``/``?`` wildcards — the MDAnalysis
  vocabulary's honest subset);
* :class:`Trajectory` — validated ``(T, N, 3)`` coordinates with frame
  slicing, per-frame geometric or mass-weighted centering;
* :func:`synthetic_trajectory` — the deterministic demo system (an
  alpha-helix backbone with a flexible loop, seeded thermal noise and a
  slow rotational drift) that drives the tests, the doctests and the
  web playground's Molecular Dynamics tab;
* :func:`resolve_atom_indices` — the shared ``atom_indices`` resolver
  every metric in this package honours.

Heavy parsers (DCD, XTC, TRR, NetCDF) stay with the reference stacks —
what lands is the in-memory model the toolkit's analyses need, exactly
typed and validated.

Examples
--------
>>> topo = Topology(
...     atom_names=["N", "CA", "C", "H"],
...     residue_names=["ALA", "ALA", "ALA", "ALA"],
...     residue_ids=[1, 1, 1, 1],
... )
>>> list(topo.select_atoms("name CA"))
[1]
>>> list(topo.select_atoms("name N C"))
[0, 2]
>>> list(topo.select_atoms("not name H*"))
[0, 1, 2]
>>> traj = synthetic_trajectory(n_frames=4, seed=0)
>>> (len(traj), traj.n_atoms, traj.topology.atom_names[1])
(4, 30, 'CA')
>>> same = synthetic_trajectory(n_frames=4, seed=0)
>>> bool(np.array_equal(traj.coords, same.coords))
True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase

import numpy as np

from pharmaml.exceptions import MDAnalysisError

__all__ = [
    "Topology",
    "Trajectory",
    "resolve_atom_indices",
    "synthetic_trajectory",
]

_SELECTION_FIELDS = ("name", "resname", "resid")
_SELECTION_KEYWORDS = ("and", "or", "not") + _SELECTION_FIELDS

_RESIDUE_NAMES = ("ALA", "GLY", "SER")
_ATOM_NAMES = ("N", "CA", "C")
_ATOM_MASSES = (14.007, 12.011, 12.011)


def resolve_atom_indices(
    atom_indices: Sequence[int] | np.ndarray | None, n_atoms: int
) -> np.ndarray:
    """Normalise an optional atom selection into a validated index array.

    >>> list(resolve_atom_indices([2, 0], 3))
    [2, 0]
    >>> list(resolve_atom_indices(None, 3))
    [0, 1, 2]
    """
    if atom_indices is None:
        return np.arange(n_atoms, dtype=np.intp)
    idx = np.asarray(atom_indices)
    if idx.ndim != 1:
        raise MDAnalysisError("atom_indices must be a 1-D sequence of integers")
    if idx.size == 0:
        raise MDAnalysisError("atom_indices selects no atoms")
    if not np.issubdtype(idx.dtype, np.integer):
        raise MDAnalysisError("atom_indices must contain integers")
    if np.any(idx < 0) or np.any(idx >= n_atoms):
        raise MDAnalysisError(
            f"atom_indices out of range for {n_atoms} atoms "
            f"(valid: 0..{n_atoms - 1})"
        )
    return idx.astype(np.intp)


class _SelectionParser:
    """Recursive-descent parser for the selection mini-language."""

    def __init__(self, tokens: list[str], topology: Topology) -> None:
        self._tokens = tokens
        self._pos = 0
        self._topology = topology

    def _peek(self) -> str:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _next(self) -> str:
        token = self._peek()
        if token == "":
            raise MDAnalysisError("selection ended mid-expression")
        self._pos += 1
        return token

    def parse(self) -> np.ndarray:
        if not self._tokens:
            raise MDAnalysisError("selection is empty")
        mask = self._expr()
        if self._pos != len(self._tokens):
            raise MDAnalysisError(
                f"unexpected token {self._tokens[self._pos]!r} in selection"
            )
        return np.flatnonzero(mask)

    def _expr(self) -> np.ndarray:
        mask = self._term()
        while self._peek() in ("and", "or"):
            operator = self._next()
            rhs = self._term()
            mask = mask & rhs if operator == "and" else mask | rhs
        return mask

    def _term(self) -> np.ndarray:
        if self._peek() == "not":
            self._next()
            return ~self._term()
        return self._factor()

    def _factor(self) -> np.ndarray:
        token = self._next()
        if token == "(":
            mask = self._expr()
            if self._peek() != ")":
                raise MDAnalysisError("selection parenthesis never closed")
            self._next()
            return mask
        if token not in _SELECTION_FIELDS:
            raise MDAnalysisError(
                f"unknown selection field {token!r} — supported: "
                "name, resname, resid (plus and/or/not/parentheses)"
            )
        values = self._values()
        if token == "name":
            return np.array(
                [
                    any(fnmatchcase(name, pattern) for pattern in values)
                    for name in self._topology.atom_names
                ],
                dtype=bool,
            )
        if token == "resname":
            names = self._topology.residue_names
            if names is None:
                raise MDAnalysisError(
                    "selection by resname needs residue names in the topology"
                )
            return np.array([name in values for name in names], dtype=bool)
        ids = self._topology.residue_ids
        if ids is None:
            raise MDAnalysisError(
                "selection by resid needs residue ids in the topology"
            )
        wanted: set[int] = set()
        for raw in values:
            text = raw.replace("-", ":")
            if ":" in text:
                lo_text, hi_text = text.split(":", 1)
                try:
                    lo, hi = int(lo_text), int(hi_text)
                except ValueError as exc:
                    raise MDAnalysisError(f"invalid resid range {raw!r}") from exc
                if lo > hi:
                    lo, hi = hi, lo
                wanted.update(range(lo, hi + 1))
            else:
                try:
                    wanted.add(int(text))
                except ValueError as exc:
                    raise MDAnalysisError(
                        f"invalid resid {raw!r} (expected an integer or a:b range)"
                    ) from exc
        return np.array([rid in wanted for rid in ids], dtype=bool)

    def _values(self) -> list[str]:
        values: list[str] = []
        while True:
            token = self._peek()
            if token == "" or token in _SELECTION_KEYWORDS or token in ("(", ")"):
                break
            values.append(self._next())
        if not values:
            raise MDAnalysisError(
                "selection field needs at least one value (e.g. 'name CA')"
            )
        return values


@dataclass
class Topology:
    """Per-atom metadata: names, residues and optional masses."""

    atom_names: list[str]
    residue_names: list[str] | None = None
    residue_ids: list[int] | None = None
    masses: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.atom_names)
        if n == 0:
            raise MDAnalysisError("a topology needs at least one atom")
        if any(not name for name in self.atom_names):
            raise MDAnalysisError("atom names must be non-empty strings")
        if self.residue_names is not None and len(self.residue_names) != n:
            raise MDAnalysisError(
                f"residue_names has {len(self.residue_names)} entries "
                f"but there are {n} atoms"
            )
        if self.residue_ids is not None and len(self.residue_ids) != n:
            raise MDAnalysisError(
                f"residue_ids has {len(self.residue_ids)} entries "
                f"but there are {n} atoms"
            )
        if self.masses is not None:
            masses = np.asarray(self.masses, dtype=float)
            if masses.shape != (n,):
                raise MDAnalysisError(
                    f"masses must have shape ({n},), got {masses.shape}"
                )
            if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
                raise MDAnalysisError("masses must be finite and strictly positive")
            self.masses = masses

    def __len__(self) -> int:
        return len(self.atom_names)

    def copy(self) -> Topology:
        return replace(
            self,
            atom_names=list(self.atom_names),
            residue_names=(
                None
                if self.residue_names is None
                else list(self.residue_names)
            ),
            residue_ids=None if self.residue_ids is None else list(self.residue_ids),
            masses=None if self.masses is None else self.masses.copy(),
        )

    def select_atoms(self, selection: str) -> np.ndarray:
        """Resolve a selection string to atom indices (may be empty).

        Supported: ``name`` (with ``*``/``?`` wildcards), ``resname``,
        ``resid`` (single ids or ``a:b`` ranges, also ``a-b``),
        ``not``, ``and``/``or`` and parentheses.  Matching is
        case-sensitive, exactly as MDAnalysis.

        >>> list(Topology(atom_names=["CA", "H"]).select_atoms("name ca"))
        []
        """
        tokens = selection.replace("(", " ( ").replace(")", " ) ").split()
        return _SelectionParser(tokens, self).parse()


@dataclass
class Trajectory:
    """A stack of frames: ``(n_frames, n_atoms, 3)`` coordinates + topology."""

    coords: np.ndarray
    topology: Topology
    time: np.ndarray | None = None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise MDAnalysisError(
                "coords must have shape (n_frames, n_atoms, 3), "
                f"got {coords.shape}"
            )
        if coords.shape[0] == 0 or coords.shape[1] == 0:
            raise MDAnalysisError("a trajectory needs at least one frame and atom")
        if coords.shape[1] != len(self.topology):
            raise MDAnalysisError(
                f"topology has {len(self.topology)} atoms but coords carry "
                f"{coords.shape[1]}"
            )
        if not np.all(np.isfinite(coords)):
            raise MDAnalysisError("trajectory coords contain NaN/inf")
        self.coords = coords
        if self.time is not None:
            time = np.asarray(self.time, dtype=float)
            if time.shape != (coords.shape[0],):
                raise MDAnalysisError(
                    f"time must have shape ({coords.shape[0]},), got {time.shape}"
                )
            if not np.all(np.isfinite(time)):
                raise MDAnalysisError("time axis contains NaN/inf")
            self.time = time

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __getitem__(
        self, key: int | slice | Sequence[int]
    ) -> Trajectory:
        if isinstance(key, int):
            frame = key if key >= 0 else key + self.n_frames
            if not 0 <= frame < self.n_frames:
                raise MDAnalysisError(
                    f"frame index {key} out of range for {self.n_frames} frames"
                )
            index: int | slice | np.ndarray = slice(frame, frame + 1)
        elif isinstance(key, slice):
            index = key
        else:
            arr = np.asarray(key)
            if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
                raise MDAnalysisError(
                    "trajectory fancy indexing needs a 1-D integer sequence"
                )
            index = arr
        frames = self.coords[index]
        if frames.shape[0] == 0:
            raise MDAnalysisError("trajectory slice selected no frames")
        return Trajectory(
            coords=frames,
            topology=self.topology,
            time=None if self.time is None else self.time[index],
        )

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[1])

    def copy(self) -> Trajectory:
        return Trajectory(
            coords=self.coords.copy(),
            topology=self.topology.copy(),
            time=None if self.time is None else self.time.copy(),
        )

    def center(
        self,
        atom_indices: Sequence[int] | np.ndarray | None = None,
        *,
        weighted: bool = False,
    ) -> Trajectory:
        """Return a copy with each frame's selection centre moved to the origin.

        The centre is the selection's geometric mean, or its centre of
        mass with ``weighted=True`` (which requires topology masses).
        The translation is applied to *every* atom of the frame — the
        MDAnalysis ``center()`` behaviour.
        """
        idx = resolve_atom_indices(atom_indices, self.n_atoms)
        subset = self.coords[:, idx, :]
        if weighted:
            if self.topology.masses is None:
                raise MDAnalysisError(
                    "weighted centering needs masses in the topology"
                )
            masses = self.topology.masses[idx]
            centres = np.einsum("tai,a->ti", subset, masses) / masses.sum()
        else:
            centres = subset.mean(axis=1)
        return Trajectory(
            coords=self.coords - centres[:, None, :],
            topology=self.topology,
            time=None if self.time is None else self.time.copy(),
        )


def synthetic_trajectory(
    n_frames: int = 80,
    *,
    seed: int = 0,
    n_residues: int = 10,
    noise: float = 0.25,
    drift_radians: float = 0.05,
    flexible_residues: tuple[int, ...] = (4, 5, 6),
) -> Trajectory:
    """Deterministic demo trajectory: an alpha-helix with a flexible loop.

    The backbone is a canonical 3.6-residue helix (100° per residue,
    1.5 Å rise, 2.3 Å radius) with ``N/CA/C`` atoms per residue.  Every
    frame applies a slow rigid-body rotation about *z* (the drift) and
    seeded per-atom Gaussian noise, amplified for the flexible-loop
    residues — rigid helix, soft loop, the textbook RMSD/RMSF picture
    and the exact dataset behind the web playground's MD tab.

    >>> t = synthetic_trajectory(n_frames=3, n_residues=2, flexible_residues=(1,))
    >>> (len(t), t.n_atoms)
    (3, 6)
    """
    if n_frames < 1:
        raise MDAnalysisError("n_frames must be >= 1")
    if n_residues < 1:
        raise MDAnalysisError("n_residues must be >= 1")
    if noise < 0:
        raise MDAnalysisError("noise must be >= 0")

    n_atoms = n_residues * 3
    angles = np.deg2rad(100.0) * np.arange(n_residues)
    z_rise = 1.5 * np.arange(n_residues)
    atom_offsets = np.deg2rad(np.array([-140.0, 0.0, 40.0]))
    base = np.empty((n_atoms, 3))
    for residue in range(n_residues):
        for atom, offset in enumerate(atom_offsets):
            theta = angles[residue] + offset
            base[3 * residue + atom] = [
                2.3 * np.cos(theta),
                2.3 * np.sin(theta),
                z_rise[residue],
            ]

    flex = np.full(n_atoms, 0.25)
    for residue in flexible_residues:
        if 0 <= residue < n_residues:
            flex[3 * residue : 3 * residue + 3] = 1.0

    rng = np.random.default_rng(seed)
    frames = np.empty((n_frames, n_atoms, 3))
    for frame in range(n_frames):
        theta = drift_radians * frame
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        frames[frame] = base @ rotation.T + rng.normal(
            0.0, 1.0, size=(n_atoms, 3)
        ) * (noise * flex)[:, None]

    topology = Topology(
        atom_names=list(_ATOM_NAMES) * n_residues,
        residue_names=[
            _RESIDUE_NAMES[r % len(_RESIDUE_NAMES)]
            for r in range(n_residues)
            for _ in range(3)
        ],
        residue_ids=[r + 1 for r in range(n_residues) for _ in range(3)],
        masses=np.tile(np.asarray(_ATOM_MASSES, dtype=float), n_residues),
    )
    return Trajectory(
        coords=frames,
        topology=topology,
        time=np.arange(n_frames, dtype=float),
    )


def synthetic_hbond_trajectory(
    n_frames: int = 60,
    seed: int = 0,
    noise: float = 0.03,
) -> Trajectory:
    """Deterministic H-bond demo: two donor–H···acceptor pairs that form
    and break on a designed schedule.

    ``gmx hbond -num``'s picture in six atoms: two independent
    heavy-donor/hydrogen/acceptor triads whose donor···acceptor
    distances oscillate (seeded, phase-shifted) across the
    :data:`~pharmaml.md.hbond.HBOND_DISTANCE_CUTOFF` — bond 2 leads
    bond 1 by a quarter period, so the per-frame count traces an honest
    0 → 1 → 2 → 1 → 0 pattern instead of a flat line.  The hydrogen of
    each triad rides 1.0 Å from its donor *along the donor–acceptor
    axis*, so the donor–H···acceptor angle stays ~180° and the distance
    criterion alone gates the bond — the demo isolates one gate of the
    geometric criterion on purpose (say so when you show it).

    Atoms: ``D1, H1, A1, D2, H2, A2`` — pass ``donors=[(0, 1), (3, 4)]``
    and ``acceptors=[2, 5]`` to :func:`~pharmaml.md.hbond.hbond_timeline`.

    >>> t = synthetic_hbond_trajectory(n_frames=12, seed=0)
    >>> (len(t), t.n_atoms)
    (12, 6)
    >>> import numpy as np
    >>> from pharmaml.md.hbond import hbond_timeline
    >>> counts = hbond_timeline(t, donors=[(0, 1), (3, 4)], acceptors=[2, 5])
    >>> [int(c) for c in counts[[0, 6, 9]]]   # none / bond 2 only / bond 1 only
    [0, 1, 1]
    >>> int((counts > 0).sum()) >= 6         # both bonds work for a living
    True
    """
    if n_frames < 1:
        raise MDAnalysisError("n_frames must be >= 1")
    if noise < 0:
        raise MDAnalysisError("noise must be >= 0")

    phases = (0.0, np.pi / 2.0)
    offsets = (np.zeros(3), np.array([0.0, 8.0, 0.0]))  # far apart: no cross-pairs
    frames = np.empty((n_frames, 6, 3))
    for frame in range(n_frames):
        for bond, (phase, offset) in enumerate(zip(phases, offsets)):
            distance = 3.9 + 1.1 * np.sin(2.0 * np.pi * frame / n_frames + phase)
            base = np.array(
                [
                    [0.0, 0.0, 0.0],          # D_k — the donor heavy atom
                    [1.0, 0.0, 0.0],          # H_k — 1.0 Å along the axis
                    [distance, 0.0, 0.0],     # A_k — at the designed distance
                ]
            )
            frames[frame, 3 * bond : 3 * bond + 3] = base + offset

    rng = np.random.default_rng(seed)
    frames = frames + rng.normal(0.0, noise, size=frames.shape)

    topology = Topology(
        atom_names=["D1", "H1", "A1", "D2", "H2", "A2"],
        residue_names=["HBD"] * 6,
        residue_ids=[1, 1, 1, 2, 2, 2],
        masses=np.array([12.011, 1.008, 15.999] * 2),
    )
    return Trajectory(
        coords=frames,
        topology=topology,
        time=np.arange(n_frames, dtype=float),
    )


def synthetic_salt_bridge_trajectory(
    n_frames: int = 60,
    seed: int = 0,
    noise: float = 0.03,
) -> Trajectory:
    """Deterministic salt-bridge demo: two charged pairs that form and
    break on a designed schedule.

    ``getcontacts``' ionic interaction in four atoms: two independent
    oppositely-charged pairs — Lys Nζ···Glu Cδ and Arg Nη1···Asp Cγ —
    whose cation–anion distances oscillate (seeded, phase-shifted)
    across the :data:`~pharmaml.md.hbond.SALT_BRIDGE_CUTOFF`.  Bridge 2
    leads bridge 1 by a sixth of a period, so the per-frame count holds
    a four-frame two-bridge plateau in the middle of the trace — a
    deliberately different silhouette from the H-bond demo's narrow
    2-frame peak.  Salt bridges are distance-only (no angle gate), so
    nothing is isolated: the demo *is* the whole criterion.

    Atoms: ``NZ, CD, NH1, CG`` — pass ``positive=[0, 2]`` and
    ``negative=[1, 3]`` to
    :func:`~pharmaml.md.hbond.salt_bridge_timeline`.

    >>> t = synthetic_salt_bridge_trajectory(n_frames=12, seed=0)
    >>> (len(t), t.n_atoms)
    (12, 4)
    >>> import numpy as np
    >>> from pharmaml.md.hbond import salt_bridge_timeline
    >>> counts = salt_bridge_timeline(t, positive=[0, 2], negative=[1, 3])
    >>> [int(c) for c in counts[[0, 6, 9]]]   # none / bridge 2 only / both
    [0, 1, 2]
    >>> int((counts > 0).sum()) >= 5          # both bridges work for a living
    True
    """
    if n_frames < 1:
        raise MDAnalysisError("n_frames must be >= 1")
    if noise < 0:
        raise MDAnalysisError("noise must be >= 0")

    phases = (0.0, np.pi / 6.0)
    offsets = (np.zeros(3), np.array([0.0, 8.0, 0.0]))  # far apart: no cross-pairs
    frames = np.empty((n_frames, 4, 3))
    for frame in range(n_frames):
        for bridge, (phase, offset) in enumerate(zip(phases, offsets)):
            distance = 4.4 + 1.1 * np.sin(2.0 * np.pi * frame / n_frames + phase)
            base = np.array(
                [
                    [0.0, 0.0, 0.0],          # the cation (NZ / NH1)
                    [distance, 0.0, 0.0],     # the anion (CD / CG), at the designed distance
                ]
            )
            frames[frame, 2 * bridge : 2 * bridge + 2] = base + offset

    rng = np.random.default_rng(seed)
    frames = frames + rng.normal(0.0, noise, size=frames.shape)

    topology = Topology(
        atom_names=["NZ", "CD", "NH1", "CG"],
        residue_names=["LYS", "GLU", "ARG", "ASP"],
        residue_ids=[1, 2, 3, 4],
        masses=np.array([14.007, 12.011, 14.007, 12.011]),
    )
    return Trajectory(
        coords=frames,
        topology=topology,
        time=np.arange(n_frames, dtype=float),
    )

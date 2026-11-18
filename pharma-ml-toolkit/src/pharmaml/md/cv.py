"""PLUMED-style collective variables — the enhanced-sampling vocabulary.

``plumed/plumed2`` defines how every enhanced-sampling method —
metadynamics, umbrella sampling, steered MD — describes a system: named
*collective variables*, functions of atom coordinates whose time series
is the bias target, declared in a ``plumed.dat`` input
(``d: DISTANCE ATOMS=1,2``) and written to a ``COLVAR`` table.  This
module is that vocabulary in plain NumPy over
:class:`~pharmaml.md.trajectory.Trajectory` frames:

* :func:`cv_distance` / :func:`cv_angle` / :func:`cv_dihedral` — the
  geometric CVs (``DISTANCE`` / ``ANGLE`` / ``DIHEDRAL``);
* :func:`cv_coordination` — ``COORDINATION``: the per-frame fraction of
  groupA–groupB pairs within ``R_0`` (the honest hard-switch subset:
  PLUMED's rational switching keywords are refused, not approximated);
* :func:`cv_rmsd` / :func:`cv_gyration` — ``RMSD`` / ``GYRATION``,
  composed over :mod:`pharmaml.md.metrics` (Kabsch-aligned RMSD, the
  ``gmx gyrate`` radius);
* :func:`cv_table` + :class:`CVTable` — many named CVs on one validated
  time axis, the ``COLVAR`` equivalent (finite-only: a bias target that
  is NaN or ±inf is refused outright — figures may tolerate-and-disclose
  NaN, a CV a metadynamics run steers on may not);
* :func:`parse_plumed_cvs` — the ``plumed.dat`` mini-parser (labels,
  ``ATOMS`` with comma lists *and* ``a-b`` ranges, ``GROUPA``/``GROUPB``/
  ``R_0``, ``MASS``, comments).  PLUMED atom numbering is 1-based; this
  parser honours PLUMED semantics at the boundary and converts to the
  toolkit's 0-based indices internally, so a ``plumed.dat`` copied from a
  real run behaves exactly as it did there.

Examples
--------
>>> from pharmaml.md.trajectory import Topology, Trajectory
>>> import numpy as np
>>> frames = np.stack([
...     np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
...     np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
... ])
>>> traj = Trajectory(frames, Topology(atom_names=["A", "B", "C"]))
>>> list(cv_distance(traj, atoms=[0, 1]))
[3.0, 6.0]
>>> list(cv_angle(traj, atoms=[0, 1, 2]))
[53.13010235415599, 33.690067525979785]
>>> table = parse_plumed_cvs(traj, "d: DISTANCE ATOMS=1,2")
>>> list(table.names)
['d']
>>> table.series.shape
(2, 1)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md.metrics import (
    dihedral,
    radius_of_gyration_trajectory,
    rmsd_trajectory,
)
from pharmaml.md.trajectory import Trajectory
from pharmaml.md.xvg import XvgSeries

__all__ = [
    "CVTable",
    "cv_angle",
    "cv_coordination",
    "cv_dihedral",
    "cv_distance",
    "cv_gyration",
    "cv_rmsd",
    "cv_table",
    "parse_plumed_cvs",
]

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: The parser's supported actions — PLUMED's CV subset this module computes.
PLUMED_ACTIONS: tuple[str, ...] = (
    "DISTANCE",
    "ANGLE",
    "DIHEDRAL",
    "COORDINATION",
    "RMSD",
    "GYRATION",
)


def _require_atoms(
    atoms: Sequence[int], n_atoms: int, count: int, action: str
) -> tuple[int, ...]:
    """Validate an exact-length, distinct, in-range index tuple."""
    idx = [int(a) for a in atoms]
    if len(idx) != count:
        raise MDAnalysisError(
            f"{action} needs exactly {count} atoms, got {len(idx)}"
        )
    for i in idx:
        if not 0 <= i < n_atoms:
            raise MDAnalysisError(
                f"atom index {i} out of range for {n_atoms} atoms "
                f"(valid: 0..{n_atoms - 1})"
            )
    if len(set(idx)) != len(idx):
        raise MDAnalysisError(f"{action} needs {count} distinct atoms")
    return tuple(idx)


def cv_distance(traj: Trajectory, atoms: Sequence[int]) -> np.ndarray:
    """``DISTANCE`` — the two-atom separation per frame (Å).

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B"]))
    >>> list(cv_distance(traj, atoms=[1, 0]))
    [2.0]
    """
    a, b = _require_atoms(atoms, traj.n_atoms, 2, "DISTANCE")
    delta = traj.coords[:, a, :] - traj.coords[:, b, :]
    return np.sqrt(np.sum(delta * delta, axis=1))


def cv_angle(traj: Trajectory, atoms: Sequence[int]) -> np.ndarray:
    """``ANGLE`` — the angle at the middle atom per frame (degrees).

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B", "C"]))
    >>> round(float(cv_angle(traj, atoms=[0, 1, 2])[0]), 6)
    90.0
    """
    i, j, k = _require_atoms(atoms, traj.n_atoms, 3, "ANGLE")
    v1 = traj.coords[:, i, :] - traj.coords[:, j, :]
    v2 = traj.coords[:, k, :] - traj.coords[:, j, :]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    if np.any(n1 == 0.0) or np.any(n2 == 0.0):
        raise MDAnalysisError(
            "ANGLE is undefined: the apex and/or a flanking atom pair is coincident"
        )
    cosine = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def cv_dihedral(traj: Trajectory, atoms: Sequence[int]) -> np.ndarray:
    """``DIHEDRAL`` — the signed torsion per frame (degrees).

    Composed over :func:`pharmaml.md.metrics.dihedral` (the praxeolitic
    formulation): 0° eclipsed, ±180° trans.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([
    ...     np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
    ...               [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ... ])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B", "C", "D"]))
    >>> round(float(cv_dihedral(traj, atoms=[0, 1, 2, 3])[0]), 6)
    0.0
    """
    p0, p1, p2, p3 = _require_atoms(atoms, traj.n_atoms, 4, "DIHEDRAL")
    return np.array(
        [
            dihedral(frame[p0], frame[p1], frame[p2], frame[p3])
            for frame in traj.coords
        ],
        dtype=float,
    )


def cv_coordination(
    traj: Trajectory,
    group_a: Sequence[int],
    group_b: Sequence[int],
    *,
    r_0: float,
) -> np.ndarray:
    """``COORDINATION`` — the fraction of groupA–groupB pairs within ``R_0``.

    The honest hard-switch subset of PLUMED's ``COORDINATION``: a pair
    counts 1 when its distance is ≤ ``r_0``, 0 otherwise, and the value
    is the count over ``|A|·|B|`` pairs.  PLUMED's rational switching
    keywords (``D_0``/``NN``/``MM``/``SWITCH``) are refused with a clear
    message rather than approximated.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([
    ...     np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    ... ])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B", "C"]))
    >>> list(cv_coordination(traj, group_a=[0], group_b=[1, 2], r_0=3.5))
    [0.5]
    """
    if not np.isfinite(r_0) or r_0 <= 0:
        raise MDAnalysisError(f"r_0 must be finite and positive, got {r_0}")
    a = [int(i) for i in group_a]
    b = [int(i) for i in group_b]
    for label, group in (("GROUPA", a), ("GROUPB", b)):
        if not group:
            raise MDAnalysisError(f"{label} selects no atoms")
        for i in group:
            if not 0 <= i < traj.n_atoms:
                raise MDAnalysisError(
                    f"{label} index {i} out of range for {traj.n_atoms} atoms"
                )
    values = np.empty(len(traj), dtype=float)
    total = len(a) * len(b)
    for f, frame in enumerate(traj.coords):
        count = 0
        for i in a:
            for j in b:
                delta = frame[i] - frame[j]
                if float(np.linalg.norm(delta)) <= r_0:
                    count += 1
        values[f] = count / total
    return values


def cv_rmsd(
    traj: Trajectory,
    atoms: Sequence[int],
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """``RMSD`` — per-frame RMSD (Å) after optimal superposition.

    Composed over :func:`pharmaml.md.metrics.rmsd_trajectory` (Kabsch):
    the reference defaults to the trajectory's own first frame, so the
    first value is exactly 0.0.

    >>> from pharmaml.md.trajectory import synthetic_trajectory
    >>> t = synthetic_trajectory(n_frames=4, seed=0)
    >>> r = cv_rmsd(t, atoms=[0, 1, 2])
    >>> (r.shape, r[0] < 1e-12)
    ((4,), True)
    """
    idx = [int(i) for i in atoms]
    if not idx:
        raise MDAnalysisError("RMSD needs at least one atom")
    for i in idx:
        if not 0 <= i < traj.n_atoms:
            raise MDAnalysisError(
                f"atom index {i} out of range for {traj.n_atoms} atoms"
            )
    return rmsd_trajectory(traj, reference=reference, atom_indices=idx)


def cv_gyration(
    traj: Trajectory,
    atoms: Sequence[int],
    masses: np.ndarray | None = None,
) -> np.ndarray:
    """``GYRATION`` — the per-frame radius of gyration (Å).

    Composed over
    :func:`pharmaml.md.metrics.radius_of_gyration_trajectory`
    (``gmx gyrate``): unweighted by default, mass-weighted via ``masses``
    (the parser's ``MASS`` flag supplies the topology's masses).

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B"]))
    >>> round(float(cv_gyration(traj, atoms=[0, 1])[0]), 6)
    1.0
    """
    idx = [int(i) for i in atoms]
    if not idx:
        raise MDAnalysisError("GYRATION needs at least one atom")
    for i in idx:
        if not 0 <= i < traj.n_atoms:
            raise MDAnalysisError(
                f"atom index {i} out of range for {traj.n_atoms} atoms"
            )
    return radius_of_gyration_trajectory(traj, atom_indices=idx, masses=masses)


@dataclass
class CVTable:
    """The ``COLVAR`` equivalent: named CV series on one time axis.

    >>> import numpy as np
    >>> t = CVTable(time=np.array([0.0, 1.0]), names=("d",),
    ...             series=np.array([[3.0], [6.0]]))
    >>> t.to_colvar()
    '#! FIELDS time d\\n0.000000 3.000000\\n1.000000 6.000000\\n'
    """

    time: np.ndarray
    names: tuple[str, ...]
    series: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        series = np.asarray(self.series, dtype=float)
        if time.ndim != 1 or time.shape[0] == 0:
            raise MDAnalysisError("CVTable time must be a non-empty 1-D array")
        if series.ndim != 2 or series.shape[0] != time.shape[0]:
            raise MDAnalysisError(
                "CVTable series must be 2-D with one row per time point "
                f"({time.shape[0]}), got shape {series.shape}"
            )
        if len(self.names) != series.shape[1]:
            raise MDAnalysisError(
                f"{len(self.names)} CV names but the series carries "
                f"{series.shape[1]} columns"
            )
        if not self.names:
            raise MDAnalysisError("a CV table needs at least one CV")
        for name in self.names:
            if not _LABEL_RE.match(name):
                raise MDAnalysisError(f"invalid CV label {name!r}")
        if len(set(self.names)) != len(self.names):
            raise MDAnalysisError(f"duplicate CV labels in {self.names}")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(series)):
            raise MDAnalysisError(
                "CV tables must be finite — NaN/inf is refused outright "
                "(a bias target a metadynamics run steers on may not be NaN)"
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "series", series)

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_cvs(self) -> int:
        """The number of CV columns."""
        return int(self.series.shape[1])

    def to_dict(self) -> dict[str, object]:
        """Plain-Python copy for JSON serialisation (bridge payloads)."""
        return {
            "time": [float(v) for v in self.time],
            "names": list(self.names),
            "series": [[float(v) for v in row] for row in self.series],
            "source": self.source,
        }

    def to_colvar(self) -> str:
        """The ``COLVAR`` text PLUMED's ``PRINT`` would have written.

        ``#! FIELDS`` header, then one row per frame: time first, every
        CV in declaration order, six decimals.
        """
        lines = ["#! FIELDS time " + " ".join(self.names)]
        for t, row in zip(self.time, self.series):
            lines.append(" ".join(f"{v:.6f}" for v in (t, *row)))
        return "\n".join(lines) + "\n"

    def xvg_series(self, name: str) -> XvgSeries:
        """One CV column as an :class:`~pharmaml.md.xvg.XvgSeries`.

        The bridge into the viz twins:
        ``plot_md_series(table.xvg_series("d"))`` renders the CV exactly
        like any other MD metric series.

        >>> t = CVTable(time=np.array([0.0]), names=("d",),
        ...             series=np.array([[3.0]]))
        >>> s = t.xvg_series("d")
        >>> (s.label, list(s.values))
        ('d', [3.0])
        """
        try:
            col = self.names.index(name)
        except ValueError:
            raise MDAnalysisError(
                f"no CV named {name!r} in the table {self.names}"
            ) from None
        return XvgSeries(
            time=self.time,
            values=self.series[:, col],
            source=self.source,
            label=name,
        )


def cv_table(
    traj: Trajectory,
    cvs: Mapping[str, np.ndarray],
    *,
    time: np.ndarray | None = None,
) -> CVTable:
    """Assemble named CV series onto one validated time axis.

    The axes-must-match rule applies: every series must carry exactly the
    trajectory's frame count (the same refusal
    :func:`~pharmaml.viz.plot_md_correlation` makes of misaligned
    abscissas — a COLVAR table with ragged columns is a broken export).
    The time axis defaults to the trajectory's own ``time`` when it
    carries one, else the frame index.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([
    ...     np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ...     np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    ... ])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B"]))
    >>> table = cv_table(traj, {"d": cv_distance(traj, atoms=[0, 1])})
    >>> table.to_colvar()
    '#! FIELDS time d\\n0.000000 2.000000\\n1.000000 3.000000\\n'
    >>> cv_table(traj, {"d": np.zeros(3)})
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.MDAnalysisError: CV 'd' carries 3 values but the trajectory has 2 frames
    """
    if not cvs:
        raise MDAnalysisError("cv_table needs at least one named CV series")
    n_frames = len(traj)
    columns: list[np.ndarray] = []
    for name, values in cvs.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise MDAnalysisError(f"CV {name!r} must be a 1-D array")
        if arr.shape[0] != n_frames:
            raise MDAnalysisError(
                f"CV {name!r} carries {arr.shape[0]} values but the "
                f"trajectory has {n_frames} frames"
            )
        columns.append(arr)
    if time is None:
        time = (
            np.asarray(traj.time, dtype=float)
            if traj.time is not None
            else np.arange(n_frames, dtype=float)
        )
    return CVTable(
        time=time,
        names=tuple(cvs.keys()),
        series=np.column_stack(columns),
    )


def _parse_plumed_indices(
    text: str, n_atoms: int
) -> list[int]:
    """Expand ``ATOMS``/``GROUPA``/``GROUPB`` values: 1-based PLUMED text.

    >>> _parse_plumed_indices("1,3", 5)
    [0, 2]
    >>> _parse_plumed_indices("2-4", 5)
    [1, 2, 3]
    >>> _parse_plumed_indices("1,3-4", 5)
    [0, 2, 3]
    """
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise MDAnalysisError(f"empty atom spec in {text!r}")
        if "-" in part:
            lo_text, _, hi_text = part.partition("-")
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError:
                raise MDAnalysisError(
                    f"bad atom range {part!r} (expected a-b)"
                ) from None
            if lo < 1 or hi < lo:
                raise MDAnalysisError(
                    f"bad atom range {part!r}: PLUMED indices are 1-based "
                    f"and ranges must be ascending"
                )
            indices.extend(range(lo, hi + 1))
        else:
            try:
                value = int(part)
            except ValueError:
                raise MDAnalysisError(
                    f"bad atom index {part!r}: PLUMED indices are 1-based "
                    f"integers"
                ) from None
            if value < 1:
                raise MDAnalysisError(
                    f"atom index {part!r}: PLUMED indices are 1-based "
                    f"(there is no atom 0)"
                )
            indices.append(value)
    zero_based = [i - 1 for i in indices]
    for i in zero_based:
        if not 0 <= i < n_atoms:
            raise MDAnalysisError(
                f"PLUMED atom out of range for {n_atoms} atoms "
                f"(valid: 1..{n_atoms})"
            )
    return zero_based


def _parse_keyword_value(token: str) -> tuple[str, str]:
    """Split ``KEY=VALUE`` (the empty value is an error)."""
    key, sep, value = token.partition("=")
    if not sep:
        raise MDAnalysisError(f"expected KEY=VALUE, got {token!r}")
    if not value:
        raise MDAnalysisError(f"keyword {key!r} has an empty value")
    return key, value


def _compute_cv(
    traj: Trajectory, label: str, action: str, keywords: dict[str, str]
) -> np.ndarray:
    """Compute one declared CV from parsed PLUMED keywords."""
    if action == "DISTANCE":
        if set(keywords) != {"ATOMS"}:
            raise MDAnalysisError(
                f"DISTANCE takes ATOMS only (extras like COMPONENTS are "
                f"not the honest subset), got {sorted(keywords)}"
            )
        return cv_distance(traj, atoms=_parse_plumed_indices(keywords["ATOMS"], traj.n_atoms))
    if action == "ANGLE":
        if set(keywords) != {"ATOMS"}:
            raise MDAnalysisError(
                f"ANGLE takes ATOMS only, got {sorted(keywords)}"
            )
        return cv_angle(traj, atoms=_parse_plumed_indices(keywords["ATOMS"], traj.n_atoms))
    if action == "DIHEDRAL":
        if set(keywords) != {"ATOMS"}:
            raise MDAnalysisError(
                f"DIHEDRAL takes ATOMS only, got {sorted(keywords)}"
            )
        return cv_dihedral(traj, atoms=_parse_plumed_indices(keywords["ATOMS"], traj.n_atoms))
    if action == "COORDINATION":
        if set(keywords) != {"GROUPA", "GROUPB", "R_0"}:
            raise MDAnalysisError(
                "COORDINATION takes GROUPA, GROUPB and R_0 exactly (the "
                "rational switching keywords D_0/NN/MM are not the honest "
                f"subset), got {sorted(keywords)}"
            )
        try:
            r_0 = float(keywords["R_0"])
        except ValueError:
            raise MDAnalysisError(
                f"R_0 must be a number, got {keywords['R_0']!r}"
            ) from None
        return cv_coordination(
            traj,
            group_a=_parse_plumed_indices(keywords["GROUPA"], traj.n_atoms),
            group_b=_parse_plumed_indices(keywords["GROUPB"], traj.n_atoms),
            r_0=r_0,
        )
    if action == "RMSD":
        if "REFERENCE" in keywords:
            raise MDAnalysisError(
                "RMSD REFERENCE= files are not supported — the honest "
                "subset measures against the trajectory's own first frame"
            )
        if set(keywords) != {"ATOMS"}:
            raise MDAnalysisError(f"RMSD takes ATOMS only, got {sorted(keywords)}")
        return cv_rmsd(traj, atoms=_parse_plumed_indices(keywords["ATOMS"], traj.n_atoms))
    if action == "GYRATION":
        wants_mass = "MASS" in keywords
        if not wants_mass and set(keywords) != {"ATOMS"}:
            raise MDAnalysisError(
                f"GYRATION takes ATOMS (and the MASS flag), got {sorted(keywords)}"
            )
        if wants_mass and set(keywords) != {"ATOMS", "MASS"}:
            raise MDAnalysisError(
                f"GYRATION takes ATOMS and the MASS flag only, got {sorted(keywords)}"
            )
        if "MASS" in keywords and keywords["MASS"] != "":
            raise MDAnalysisError("MASS is a flag, not KEY=VALUE")
        masses = None
        if "MASS" in keywords:
            if traj.topology.masses is None:
                raise MDAnalysisError(
                    "GYRATION MASS needs topology masses, but this "
                    "trajectory carries none"
                )
            masses = np.asarray(traj.topology.masses, dtype=float)
        return cv_gyration(
            traj, atoms=_parse_plumed_indices(keywords["ATOMS"], traj.n_atoms),
            masses=masses,
        )
    raise MDAnalysisError(
        f"unsupported PLUMED action {action!r} (supported: "
        f"{', '.join(PLUMED_ACTIONS)})"
    )


def parse_plumed_cvs(traj: Trajectory, source: str) -> CVTable:
    """Parse a ``plumed.dat`` snippet and compute its CVs over ``traj``.

    One declaration per line — ``label: ACTION KEY=VALUE ...`` — with
    ``#`` comments and blank lines skipped.  Labels must be unique and
    identifier-safe; every supported action's keywords are validated
    exactly (unknown keyword → error, never a silent ignore).  Atom
    numbering is PLUMED's 1-based; ``ATOMS=1,3-5`` expands to indices
    1, 3, 4, 5.

    >>> from pharmaml.md.trajectory import Topology, Trajectory
    >>> frames = np.stack([np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])])
    >>> traj = Trajectory(frames, Topology(atom_names=["A", "B"]))
    >>> table = parse_plumed_cvs(
    ...     traj, "# demo\\nd: DISTANCE ATOMS=1,2\\nrg: GYRATION ATOMS=1-2"
    ... )
    >>> list(table.names)
    ['d', 'rg']
    >>> parse_plumed_cvs(traj, "d: DISTANCE ATOMS=1,2\\nd2: DISTANCE ATOMS=1")
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.MDAnalysisError: DISTANCE needs exactly 2 atoms, got 1
    """
    cvs: dict[str, np.ndarray] = {}
    seen_labels: set[str] = set()
    for lineno, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label, sep, rest = line.partition(":")
        if not sep:
            raise MDAnalysisError(
                f"plumed line {lineno}: every CV needs a label — "
                f"'label: ACTION KEY=VALUE ...' (got {line.split()[0]!r})"
            )
        label = label.strip()
        if not _LABEL_RE.match(label):
            raise MDAnalysisError(
                f"plumed line {lineno}: invalid CV label {label!r}"
            )
        if label in seen_labels:
            raise MDAnalysisError(
                f"plumed line {lineno}: duplicate CV label {label!r}"
            )
        tokens = rest.split()
        if not tokens:
            raise MDAnalysisError(f"plumed line {lineno}: {label!r} has no action")
        action = tokens[0]
        if action not in PLUMED_ACTIONS:
            raise MDAnalysisError(
                f"plumed line {lineno}: unsupported action {action!r} "
                f"(supported: {', '.join(PLUMED_ACTIONS)})"
            )
        keywords: dict[str, str] = {}
        for token in tokens[1:]:
            if token == "MASS":
                keywords["MASS"] = ""
                continue
            key, value = _parse_keyword_value(token)
            if key in keywords:
                raise MDAnalysisError(
                    f"plumed line {lineno}: duplicate keyword {key!r}"
                )
            keywords[key] = value
        cvs[label] = _compute_cv(traj, label, action, keywords)
        seen_labels.add(label)
    if not cvs:
        raise MDAnalysisError("no CV declarations in the plumed input")
    return cv_table(traj, cvs)

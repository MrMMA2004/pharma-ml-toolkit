r"""PyMOL's structural-analysis layer, absorbed natively.

``schrodinger/pymol`` (the open-sourced 2015 branch reviewed in this
round's original pass, and re-derived here from the published source)
is a *molecular visualization* system — and the OpenGL engine, the
scene graph, the ray tracer and the movie maker are exactly what this
toolkit does not vendor.  What transfers natively is everything *under*
the rendering: the coordinate model, the measurement suite, the
superposition mathematics and the selection language, and all of it
lands here, in plain NumPy:

* :class:`AtomRecord` + :func:`parse_pdb_block` / :func:`write_pdb_block`
  -- the PDB column contract (ATOM/HETATM fixed-width records, MODEL/
  ENDMDL multi-model frames, the 77-78 element field with the
  atom-name fallback rule: a blank at column 13 means a one-letter
  element symbol at column 14, `` CA `` is C-alpha while ``CA  `` is
  calcium).
* :func:`point_distance`, :func:`bond_angle`, :func:`dihedral_angle`
  -- the measurement trio every structural script starts with (the
  dihedral is the atan2 formulation, signed, −180..180).
* :func:`kabsch_rotation`, :func:`superpose`, :func:`rmsd`,
  :func:`align_structures`, :func:`rmsf` -- the ``super``/``align``/
  ``rmsf`` mathematics: the Kabsch-Weyl SVD rotation with reflection
  correction, CA-based residue matching, iterative outlier rejection
  cycles (``align``'s behaviour), and per-atom fluctuation over a
  coordinate frame stack.
* :func:`compile_selection` / :func:`select_atoms` -- PyMOL's *selection
  algebra* as a real recursive-descent parser: ``and``/``or``/``not``
  (symbolic ``& | !`` too), parentheses, the property selectors
  ``name resn resi chain elem alt b q model hetatm``, the comparison
  operators ``= == != < > <= >= in like``, ranges (``resi 1-57``),
  lists (``chain A+B``) and the spatial ``within 5.0 of (expr)``.
  Selections compile once and evaluate against any atom list — the
  same named-selection statefulness PyMOL's object model has.
* :func:`hydrogen_bond_energies` + :func:`assign_secondary_structure`
  -- the ``dss`` command's engine: Kabsch-Sander backbone H-bond
  energies (the 0.084·(1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN)·332 kcal/mol
  electrostatics with in-plane amide-H placement) feeding a DSSP-style
  assignment (H/G/I helices by i+3/i+4/i+5 turns, E/B by the parallel
  and antiparallel bridge definitions, T turns, S otherwise).
* :func:`build_peptide_backbone` -- an ideal-geometry NeRF backbone
  builder (bond lengths/angles from the standard Engh-Huber set) so
  every example above runs on real helix geometry, not magic arrays.

The honest scope, stated once: no rendering, no scenes, no external
execution of the PyMOL binary; the bridge/ladder bookkeeping follows
the DSSP paper's definitions in their standard simplification (no
chain-boundary smoothing, no β-bulge detection); amide hydrogens are
placed by the in-plane bisector approximation rather than read from
a fully-hydrogenated structure.

Examples
--------
>>> from pharmaml.chem.structure import (
...     assign_secondary_structure, build_peptide_backbone,
...     compile_selection, dihedral_angle, parse_pdb_block,
...     rmsd, select_atoms, structure_summary, superpose,
... )
>>> atoms = build_peptide_backbone(8)          # ideal alpha-helix
>>> len(atoms), atoms[1].name, atoms[1].resname
(32, 'CA', 'ALA')
>>> phi3 = dihedral_angle(*(a.xyz for a in atoms[6:11] if a.name != 'O'))
>>> float(round(phi3, 1))
-57.0
>>> ca = [a for a in atoms if a.name == 'CA']
>>> float(round(superpose(atoms, atoms, 'name CA')['rmsd'], 9))
0.0
>>> ss = assign_secondary_structure(atoms)
>>> ''.join(code for _, code in ss)
'SHHHHHHT'
>>> waters = select_atoms(atoms + _demo_waters(), 'resn HOH')
>>> len(waters)
2
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from pharmaml.exceptions import DataValidationError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "AtomRecord",
    "align_structures",
    "assign_secondary_structure",
    "b_factor_profile",
    "bond_angle",
    "build_peptide_backbone",
    "center_of_mass",
    "compile_selection",
    "contact_pairs",
    "dihedral_angle",
    "distance_matrix",
    "hydrogen_bond_energies",
    "kabsch_rotation",
    "parse_pdb_block",
    "point_distance",
    "residue_groups",
    "rmsd",
    "rmsf",
    "secondary_structure_string",
    "select_atoms",
    "structure_summary",
    "superpose",
    "write_pdb_block",
]


# --------------------------------------------------------------------------
# the coordinate model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomRecord:
    """One ATOM/HETATM record of a PDB file.

    Attributes
    ----------
    serial : int
        The record's serial number (columns 7-11).
    name : str
        Atom name (columns 13-16), e.g. ``"CA"``, ``" OD1"``-style
        spacing already stripped.
    resname : str
        Residue name, three letters (``"ALA"``, ``"HOH"``).
    chain : str
        Chain identifier (one character, possibly blank).
    resseq : int
        Residue sequence number.
    icode : str
        Insertion code (usually blank).
    xyz : np.ndarray
        ``(3,)`` float coordinates.
    occupancy : float
        Occupancy column (default 1.0).
    bfactor : float
        B-value / temperature factor (default 0.0).
    element : str
        Element symbol (``"C"``, ``"N"``, ``"ZN"``), title-cased when
        inferred from the atom name.
    charge : str
        Formal charge column as written (usually blank).
    hetatm : bool
        ``True`` when the record is a HETATM.
    altloc : str
        Alternate-location indicator (usually blank).
    model : int
        MODEL block the record belongs to (1-based; 0 when absent).
    """

    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    xyz: np.ndarray
    occupancy: float = 1.0
    bfactor: float = 0.0
    element: str = ""
    charge: str = ""
    hetatm: bool = False
    altloc: str = ""
    model: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", np.asarray(self.xyz, dtype=float).reshape(3))

    @property
    def residue_key(self) -> tuple[str, int, str, int]:
        """Hashable residue identity ``(chain, resseq, icode, model)``."""

        return (self.chain, self.resseq, self.icode, self.model)

    def from_xyz(self, xyz: np.ndarray) -> AtomRecord:
        """A copy of this record at new coordinates (immutable record)."""

        return replace(self, xyz=np.asarray(xyz, dtype=float).reshape(3))

    def pdb_line(self) -> str:
        """The record rendered back to its 80-column ATOM/HETATM line."""

        record = "HETATM" if self.hetatm else "ATOM  "
        name_field = _format_atom_name(self.name, self.element)
        return (
            f"{record}{self.serial:5d} {name_field}{self.altloc:>1s}"
            f"{self.resname:>3s} {self.chain:>1s}{self.resseq:4d}{self.icode:>1s}   "
            f"{self.xyz[0]:8.3f}{self.xyz[1]:8.3f}{self.xyz[2]:8.3f}"
            f"{self.occupancy:6.2f}{self.bfactor:6.2f}      "
            f"    {self.element:>2s}{self.charge:>2s}"
        )


def _format_atom_name(name: str, element: str) -> str:
    """Format an atom name into PDB columns 13-16.

    The convention: one-letter element symbols start at column 14
    (``" CA "`` is C-alpha), two-letter symbols start at column 13
    (``"CA  "`` is calcium).
    """

    if len(name) >= 4:
        return name[:4]
    if len(element) == 2 and name.upper().startswith(element.upper()):
        return f"{name:<4s}"
    return f" {name:<3s}"


def _infer_element(name_field: str) -> str:
    """Element from the atom-name field when columns 77-78 are blank."""

    stripped = name_field.strip()
    if not stripped:
        raise DataValidationError(f"atom name field {name_field!r} is empty")
    # leading digit (e.g. "1HB") belongs to the numbering, not the element
    if stripped[0].isdigit():
        stripped = stripped[1:].strip()
    if name_field[:1] == " " or len(stripped) == 1:
        return stripped[:1].upper()
    two = stripped[:2]
    if two.upper() in _TWO_LETTER_ELEMENTS:
        return two[0].upper() + two[1].lower()
    return stripped[0].upper()


#: Two-letter element symbols recognised by the name fallback rule.
_TWO_LETTER_ELEMENTS = frozenset(
    {
        "HE", "LI", "BE", "NE", "NA", "MG", "AL", "SI", "CL", "AR", "CA", "SC", "TI",
        "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
        "RB", "SR", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "XE", "CS", "BA", "LA", "CE", "HF", "TA", "RE", "OS", "IR", "PT",
        "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN", "RA", "TH", "PA", "NP", "PU",
    }
)


def parse_pdb_block(text: str) -> list[AtomRecord]:
    """Parse ATOM/HETATM records out of a PDB file's text.

    Multi-model files (``MODEL``/``ENDMDL`` blocks) are flattened in
    order, each record carrying its model number.  Coordinate columns
    must be numeric and present; anything else raises
    :class:`~pharmaml.exceptions.DataValidationError` naming the line.

    Parameters
    ----------
    text : str
        The full text of a PDB file (or a block of it).

    Returns
    -------
    list[AtomRecord]
        Records in file order.

    Examples
    --------
    >>> block = (
    ...     "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
    ...     "  1.00 30.00           N  \\n"
    ...     "HETATM    2  O   HOH A   2       3.000   0.000   0.000"
    ...     "  1.00 55.00           O  \\n"
    ...     "END\\n"
    ... )
    >>> atoms = parse_pdb_block(block)
    >>> (atoms[0].element, atoms[1].hetatm, atoms[1].bfactor)
    ('N', True, 55.0)
    """

    records: list[AtomRecord] = []
    model = 1
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n\r")
        tag = line[:6]
        if tag == "MODEL ":
            digits = line[10:14].strip()
            if digits:
                try:
                    model = int(digits)
                except ValueError as exc:
                    raise DataValidationError(
                        f"PDB line {lineno}: unreadable MODEL number {digits!r}"
                    ) from exc
            continue
        if tag in ("ENDMDL", "END   ", "TER   ", "HEADER", "TITLE ", "COMPND", "CRYST1"):
            continue
        if tag not in ("ATOM  ", "HETATM"):
            continue
        if len(line) < 54:
            raise DataValidationError(
                f"PDB line {lineno}: record too short for coordinates "
                f"({len(line)} columns, 54 needed)"
            )
        line = line.ljust(80)
        try:
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
            )
            occupancy = float(line[54:60] or 1.0)
            bfactor = float(line[60:66] or 0.0)
        except ValueError as exc:
            raise DataValidationError(
                f"PDB line {lineno}: unreadable numeric column in {line!r}"
            ) from exc
        serial_text = line[6:11].strip()
        resseq_text = line[22:26].strip()
        try:
            serial = int(serial_text) if serial_text else len(records) + 1
            resseq = int(resseq_text) if resseq_text else 0
        except ValueError as exc:
            raise DataValidationError(
                f"PDB line {lineno}: unreadable serial/resSeq in {line!r}"
            ) from exc
        name_field = line[12:16]
        element = line[76:78].strip()
        if not element:
            element = _infer_element(name_field)
        else:
            element = element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
        records.append(
            AtomRecord(
                serial=serial,
                name=name_field.strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=resseq,
                icode=line[26:27].strip(),
                xyz=xyz,
                occupancy=occupancy,
                bfactor=bfactor,
                element=element,
                charge=line[78:80].strip(),
                hetatm=tag == "HETATM",
                altloc=line[16:17].strip(),
                model=model,
            )
        )
    if not records:
        raise DataValidationError("PDB block contains no ATOM/HETATM records")
    return records


def write_pdb_block(atoms: Iterable[AtomRecord], *, renumber: bool = True) -> str:
    """Render atom records back to PDB text (plus ``TER``/``END``).

    Parameters
    ----------
    atoms : iterable of AtomRecord
        Records to serialise.
    renumber : bool
        Renumber serials sequentially (PDB serials are 5-digit and
        overflow silently otherwise).

    Returns
    -------
    str
        The block, newline-terminated.

    Examples
    --------
    >>> atoms = parse_pdb_block(_demo_pdb())
    >>> len(write_pdb_block(atoms).splitlines())   # atoms + TER + END
    37
    """

    lines: list[str] = []
    last_chain: str | None = None
    for index, atom in enumerate(atoms, start=1):
        serial = index if renumber else atom.serial % 100000
        if last_chain is not None and atom.chain != last_chain:
            lines.append(f"TER   {index - 1:5d}")
        line = atom.pdb_line()
        if renumber:
            line = line[:6] + f"{serial:5d}" + line[11:]
        lines.append(line)
        last_chain = atom.chain
    lines.append("TER" if last_chain is not None else "END")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _demo_waters() -> list[AtomRecord]:
    """Two stray waters far from the helix (selection-algebra fodder)."""

    return [
        AtomRecord(
            serial=101, name="O", resname="HOH", chain="B", resseq=1, icode="",
            xyz=np.array([20.0, 10.0, 5.0]), occupancy=1.0, bfactor=55.0,
            element="O", hetatm=True,
        ),
        AtomRecord(
            serial=102, name="O", resname="HOH", chain="B", resseq=2, icode="",
            xyz=np.array([21.5, 10.4, 5.9]), occupancy=1.0, bfactor=55.0,
            element="O", hetatm=True,
        ),
    ]


def _demo_pdb() -> str:
    """The 8-residue helix plus waters as a PDB text block."""

    return write_pdb_block(build_peptide_backbone(8) + _demo_waters())


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def point_distance(a: AtomRecord, b: AtomRecord) -> float:
    """Euclidean distance between two atoms (Å)."""

    return float(np.linalg.norm(a.xyz - b.xyz))


def bond_angle(a: AtomRecord, b: AtomRecord, c: AtomRecord) -> float:
    """Angle a-b-c in degrees (b is the vertex)."""

    u = a.xyz - b.xyz
    v = c.xyz - b.xyz
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def dihedral_angle(
    p0: np.ndarray | AtomRecord,
    p1: np.ndarray | AtomRecord,
    p2: np.ndarray | AtomRecord,
    p3: np.ndarray | AtomRecord,
) -> float:
    """Signed dihedral p0-p1-p2-p3 in degrees, −180..180.

    Accepts ``(3,)`` arrays or :class:`AtomRecord` (coordinates used).
    The atan2 formulation keeps the sign and stays accurate near the
    trans planar value (±180°).
    """

    def xyz(point: np.ndarray | AtomRecord) -> np.ndarray:
        return point.xyz if isinstance(point, AtomRecord) else np.asarray(point, dtype=float)

    b0 = xyz(p0) - xyz(p1)
    b1 = xyz(p2) - xyz(p1)
    b2 = xyz(p3) - xyz(p2)
    norm = float(np.linalg.norm(b1))
    if norm == 0.0:
        raise DataValidationError("dihedral_angle: middle points coincide")
    b1n = b1 / norm
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1n, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def distance_matrix(atoms: Sequence[AtomRecord]) -> np.ndarray:
    """The full pairwise distance matrix ``(n, n)`` in Å."""

    coords = np.array([a.xyz for a in atoms], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def contact_pairs(
    atoms: Sequence[AtomRecord], cutoff: float = 4.0, *, intra: bool = True
) -> list[tuple[int, int, float]]:
    """All atom pairs within ``cutoff`` Å as ``(i, j, distance)``.

    Parameters
    ----------
    atoms : sequence of AtomRecord
        The atom stack.
    cutoff : float
        Inclusive distance threshold in Å.
    intra : bool
        When ``False``, only pairs from *different* chains are kept.

    Examples
    --------
    >>> atoms = build_peptide_backbone(4)
    >>> pairs = contact_pairs(atoms, 1.6)
    >>> any(abs(d - 1.458) < 0.02 for _, _, d in pairs)   # the N-CA bond
    True
    """

    matrix = distance_matrix(atoms)
    rows, cols = np.nonzero(np.triu(matrix, k=1) <= cutoff)
    result: list[tuple[int, int, float]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if not intra and atoms[i].chain == atoms[j].chain:
            continue
        result.append((i, j, float(matrix[i, j])))
    return result


#: Covalent radii (Å) for the elements seen in macromolecular files.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05,
    "P": 1.07, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Se": 1.20, "Na": 1.66,
    "Mg": 1.41, "Zn": 1.22, "Fe": 1.32, "Ca": 1.76, "K": 2.03, "Mn": 1.39,
}

#: Atomic masses (u) for the same element set.
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "S": 32.06, "P": 30.974, "Cl": 35.45, "Br": 79.904, "I": 126.904,
    "Se": 78.971, "Na": 22.990, "Mg": 24.305, "Zn": 65.38, "Fe": 55.845,
    "Ca": 40.078, "K": 39.098, "Mn": 54.938,
}


def center_of_mass(atoms: Sequence[AtomRecord]) -> np.ndarray:
    """Mass-weighted centroid (unknown elements weigh as carbon)."""

    masses = np.array(
        [ATOMIC_MASSES.get(a.element, 12.011) for a in atoms], dtype=float
    )
    coords = np.array([a.xyz for a in atoms], dtype=float)
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()


# --------------------------------------------------------------------------
# superposition: the Kabsch machinery behind `super`/`align`
# --------------------------------------------------------------------------


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The optimal rotation aligning ``mobile`` onto ``target``.

    Both arrays are ``(n, 3)`` and are centred here; the returned
    ``R`` satisfies ``target ≈ (mobile − mean) @ R + target_mean``
    in the least-squares sense, with the reflection fix (a proper
    rotation, ``det(R) = +1``) applied through the SVD's sign trick.

    Raises
    ------
    DataValidationError
        If the inputs are empty or shapes disagree.
    """

    p = np.asarray(mobile, dtype=float).reshape(-1, 3)
    q = np.asarray(target, dtype=float).reshape(-1, 3)
    if p.shape[0] == 0 or p.shape != q.shape:
        raise DataValidationError(
            f"kabsch_rotation: need equal non-empty (n,3) arrays, got {p.shape} vs {q.shape}"
        )
    covariance = p.T @ q
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ correction @ u.T


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Root-mean-square deviation between two coordinate stacks."""

    a = np.asarray(coords_a, dtype=float).reshape(-1, 3)
    b = np.asarray(coords_b, dtype=float).reshape(-1, 3)
    if a.shape != b.shape or a.shape[0] == 0:
        raise DataValidationError(
            f"rmsd: need equal non-empty (n,3) arrays, got {a.shape} vs {b.shape}"
        )
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def superpose(
    mobile: Sequence[AtomRecord],
    target: Sequence[AtomRecord],
    selection: str = "",
    *,
    apply_to: str = "mobile",
) -> dict[str, Any]:
    """Least-squares superposition of ``mobile`` onto ``target``.

    The engine behind PyMOL's ``super``: the Kabsch rotation over the
    atoms matched by ``selection`` (default: all), applied to the
    mobile coordinates.

    Parameters
    ----------
    mobile, target : sequence of AtomRecord
        The two structures; matched index-by-index (use the same
        selection on both).
    selection : str
        A selection-algebra expression evaluated on *both* stacks; the
        matched atom pairs feed the Kabsch fit.  Empty = all atoms.
    apply_to : {"mobile", "none"}
        Whether to return transformed mobile coordinates.

    Returns
    -------
    dict
        ``rmsd`` (over the fitted selection), ``rotation`` (3×3),
        ``translation`` (applied after rotation), ``n_fitted``,
        ``fitted_rmsd`` and, when requested, ``transformed``
        (``(n, 3)`` array of the moved mobile coordinates).

    Examples
    --------
    >>> import numpy as np
    >>> atoms = build_peptide_backbone(6)
    >>> moved = [a.from_xyz(a.xyz + np.array([3.0, -2.0, 1.0]))
    ...          for a in atoms]
    >>> fit = superpose(moved, atoms, 'name CA')
    >>> float(round(fit['rmsd'], 6))
    0.0
    """

    if len(mobile) != len(target):
        raise DataValidationError(
            f"superpose: stacks must align index-by-index ({len(mobile)} vs {len(target)})"
        )
    if selection:
        indices = compile_selection(selection).apply(mobile)
        if not indices:
            raise DataValidationError(f"superpose: selection {selection!r} matched no atoms")
    else:
        indices = list(range(len(mobile)))
    p = np.array([mobile[i].xyz for i in indices], dtype=float)
    q = np.array([target[i].xyz for i in indices], dtype=float)
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    rotation = kabsch_rotation(p - pc, q - qc)
    fitted = (p - pc) @ rotation + qc
    result: dict[str, Any] = {
        "rmsd": float(np.sqrt(((fitted - q) ** 2).sum(axis=1).mean())),
        "fitted_rmsd": float(np.sqrt(((fitted - q) ** 2).sum(axis=1).mean())),
        "rotation": rotation,
        "translation": qc - pc @ rotation,
        "n_fitted": len(indices),
    }
    if apply_to == "mobile":
        allp = np.array([a.xyz for a in mobile], dtype=float)
        result["transformed"] = (allp - pc) @ rotation + qc
    return result


def align_structures(
    mobile: Sequence[AtomRecord],
    target: Sequence[AtomRecord],
    *,
    max_cycles: int = 5,
    outlier_sigma: float = 2.0,
) -> dict[str, Any]:
    """CA-based alignment with iterative outlier rejection.

    The engine behind PyMOL's ``align``: residues are matched by
    ``(chain, resseq)`` over CA atoms, fitted by Kabsch, then pairs
    whose squared deviation exceeds ``outlier_sigma`` standard
    deviations are dropped and the fit repeated, up to ``max_cycles``.

    Returns
    -------
    dict
        ``rmsd`` (final, over kept pairs), ``n_aligned``,
        ``n_cycles``, ``dropped`` (the residue keys rejected last
        cycle) and ``transformed`` (mobile coordinates after the
        final transform).
    """

    mobile_ca = {(a.chain, a.resseq): i for i, a in enumerate(mobile) if a.name == "CA"}
    target_ca = {(a.chain, a.resseq): i for i, a in enumerate(target) if a.name == "CA"}
    keys = sorted(set(mobile_ca) & set(target_ca), key=lambda k: (k[0], k[1]))
    if not keys:
        raise DataValidationError("align_structures: no common CA residues to align on")
    pairs = [(mobile_ca[key], target_ca[key]) for key in keys]
    allp = np.array([a.xyz for a in mobile], dtype=float)
    allq = np.array([a.xyz for a in target], dtype=float)
    dropped: list[tuple[str, int]] = []
    cycle = 0
    for cycle in range(1, max_cycles + 1):
        p = allp[[i for i, _ in pairs]]
        q = allq[[j for _, j in pairs]]
        pc, qc = p.mean(axis=0), q.mean(axis=0)
        rotation = kabsch_rotation(p - pc, q - qc)
        fitted = (p - pc) @ rotation + qc
        deviations = ((fitted - q) ** 2).sum(axis=1)
        current_rmsd = float(np.sqrt(deviations.mean()))
        threshold = outlier_sigma**2 * deviations.var() + deviations.mean()
        outlier_mask = deviations > threshold if len(pairs) > 4 else np.zeros(len(pairs), bool)
        if not outlier_mask.any() or cycle == max_cycles:
            dropped = [keys[i] for i in np.nonzero(outlier_mask)[0]]
            break
        dropped = [keys[i] for i in np.nonzero(outlier_mask)[0]]
        pairs = [pair for i, pair in enumerate(pairs) if not outlier_mask[i]]
        keys = [key for i, key in enumerate(keys) if not outlier_mask[i]]
    transformed = (allp - pc) @ rotation + qc
    return {
        "rmsd": current_rmsd,
        "n_aligned": len(pairs),
        "n_cycles": cycle,
        "dropped": dropped,
        "transformed": transformed,
    }


def rmsf(
    frames: np.ndarray | Sequence[np.ndarray],
    *,
    reference: int = 0,
    by_residue: bool = False,
    atoms: Sequence[AtomRecord] | None = None,
) -> np.ndarray:
    """Per-atom RMS fluctuation over a frame stack.

    Each frame is first superposed onto ``frames[reference]`` (all
    atoms, Kabsch) so the fluctuation measures *internal* motion, then
    ``rmsf_i = sqrt(mean_f |x_fi − x̄_i|²)`` — the quantity PyMOL's
    ``rmsf`` and every MD analysis package report.

    Parameters
    ----------
    frames : array-like, ``(n_frames, n_atoms, 3)``
        The trajectory stack.
    reference : int
        The frame everything is superposed onto.
    by_residue : bool
        Average the per-atom values inside each residue instead
        (requires ``atoms``).
    atoms : sequence of AtomRecord, optional
        The atom records matching the frames (needed for
        ``by_residue=True``; also used for the superposition).

    Returns
    -------
    np.ndarray
        ``(n_atoms,)`` — or ``(n_residues,)`` when ``by_residue``.
    """

    stack = np.asarray(frames, dtype=float)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise DataValidationError(
            f"rmsf: need a (n_frames, n_atoms, 3) stack with n_frames >= 2, got {stack.shape}"
        )
    target = stack[reference]
    aligned = np.empty_like(stack)
    for f in range(stack.shape[0]):
        if f == reference:
            aligned[f] = stack[f]
            continue
        pc = stack[f].mean(axis=0)
        qc = target.mean(axis=0)
        rotation = kabsch_rotation(stack[f] - pc, target - qc)
        aligned[f] = (stack[f] - pc) @ rotation + qc
    deviations = aligned - aligned.mean(axis=0)
    per_atom = np.sqrt((deviations**2).sum(axis=2).mean(axis=0))
    if by_residue:
        if atoms is None or len(atoms) != stack.shape[1]:
            raise DataValidationError("rmsf(by_residue=True) needs matching atoms")
        groups: dict[tuple, list[int]] = {}
        for index, atom in enumerate(atoms):
            groups.setdefault(atom.residue_key, []).append(index)
        return np.array([per_atom[idxs].mean() for idxs in groups.values()])
    return per_atom


# --------------------------------------------------------------------------
# the selection algebra
# --------------------------------------------------------------------------


_SELECTOR_WORDS = frozenset(
    {"name", "resn", "resi", "resid", "chain", "elem", "element", "alt", "b", "q", "model"}
)
_FLAG_WORDS = frozenset({"hetatm", "polymer"})


def _tokenize_selection(expression: str) -> list[str]:
    """Split a selection expression into operator/value tokens."""

    tokens: list[str] = []
    position = 0
    length = len(expression)
    while position < length:
        char = expression[position]
        if char.isspace():
            position += 1
            continue
        if char in "()|&!+-":
            tokens.append(char)
            position += 1
            continue
        if char in "<>=:":
            start = position
            while position < length and expression[position] in "<>=":
                position += 1
            tokens.append(expression[start:position])
            continue
        if char in "'\"":
            end = expression.find(char, position + 1)
            if end < 0:
                raise DataValidationError(
                    f"selection: unterminated quote starting at {position!r}"
                )
            tokens.append(expression[position + 1 : end])
            position = end + 1
            continue
        start = position
        while position < length and (
            expression[position].isalnum()
            or expression[position] in "_.*%?@`"
        ):
            position += 1
        if position == start:
            raise DataValidationError(
                f"selection: unexpected character {char!r} at {position}"
            )
        tokens.append(expression[start:position])
    return tokens


class _SelectionParser:
    """Recursive-descent parser for the PyMOL-style selection grammar."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> str | None:
        token = self.peek()
        if token is not None:
            self.pos += 1
        return token

    def expect(self, expected: str) -> str:
        token = self.next()
        if token != expected:
            raise DataValidationError(
                f"selection: expected {expected!r}, got {token!r} "
                f"(tokens: {self.tokens})"
            )
        return token

    def parse(self) -> SelectionNode:
        node = self.parse_or()
        if self.peek() is not None:
            raise DataValidationError(
                f"selection: trailing tokens from {self.peek()!r} onward"
            )
        return node

    def parse_or(self) -> SelectionNode:
        node = self.parse_and()
        while self.peek() in ("or", "|"):
            self.next()
            node = OrNode(node, self.parse_and())
        return node

    def parse_and(self) -> SelectionNode:
        node = self.parse_not()
        while self.peek() in ("and", "&"):
            self.next()
            node = AndNode(node, self.parse_not())
        return node

    def parse_not(self) -> SelectionNode:
        if self.peek() in ("not", "!"):
            self.next()
            return NotNode(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> SelectionNode:
        token = self.peek()
        if token == "(":
            self.next()
            node = self.parse_or()
            self.expect(")")
            return node
        if token is None:
            raise DataValidationError("selection: expression ends where a term was expected")
        lowered = token.lower()
        if lowered == "within":
            self.next()
            distance = self._number("within needs a distance")
            self.expect("of")
            self.expect("(")
            inner = self.parse_or()
            self.expect(")")
            return WithinNode(distance, inner)
        if lowered in _FLAG_WORDS:
            self.next()
            return FlagNode(lowered)
        if lowered in _SELECTOR_WORDS:
            self.next()
            return self.parse_comparison(lowered)
        raise DataValidationError(
            f"selection: unknown selector {token!r} "
            f"(known: {sorted(_SELECTOR_WORDS | _FLAG_WORDS)})"
        )

    def parse_comparison(self, selector: str) -> SelectionNode:
        allow_range = selector in ("resi", "resid")
        operator = self.peek()
        if operator in ("=", "=="):
            self.next()
            return ComparisonNode(selector, "=", self._values())
        if operator == "!=":
            self.next()
            return ComparisonNode(selector, "!=", self._values(allow_range))
        if operator in ("<", ">", "<=", ">="):
            self.next()
            return ComparisonNode(
                selector, operator, [str(self._number(f"{selector} {operator}"))]
            )
        if operator == "in":
            self.next()
            return ComparisonNode(selector, "in", self._values(allow_range))
        if operator == "like":
            self.next()
            return ComparisonNode(selector, "like", self._values(allow_range))
        # bare selector: `name CA` shorthand for `name = CA`
        return ComparisonNode(selector, "=", self._values(allow_range))

    def _values(self, allow_range: bool = False) -> list[str]:
        values: list[str] = [self._atom_token("a selector value")]
        while True:
            token = self.peek()
            if token == "+":  # membership list: chain A+B
                self.next()
                values.append(self._atom_token("another list value"))
            elif (
                token == "-"
                and allow_range
                and self.pos + 1 < len(self.tokens)
                and _looks_numeric(self.tokens[self.pos + 1])
            ):
                self.next()
                values[-1] = f"{values[-1]}-{self._atom_token('a range bound')}"
            else:
                return values

    def _atom_token(self, context: str) -> str:
        token = self.next()
        if token is None or token in ("(", ")", "+", "-", "|", "&", "!", "and", "or"):
            raise DataValidationError(f"selection: {context} expected, got {token!r}")
        return token

    def _number(self, context: str) -> float:
        token = self.next()
        if token is None:
            raise DataValidationError(f"selection: {context}, got end of expression")
        follower = self.peek()
        if token == "-" and follower is not None and _looks_numeric(follower):
            token = f"-{self.next()}"
        try:
            return float(token)
        except ValueError as exc:
            raise DataValidationError(
                f"selection: {context}, got non-numeric {token!r}"
            ) from exc


def _looks_numeric(token: str) -> bool:
    """Whether a token could continue a number (for range detection)."""

    try:
        float(token)
    except ValueError:
        return False
    return True


class SelectionNode:
    """A node of the parsed selection tree (evaluated per atom stack)."""

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        raise NotImplementedError


class OrNode(SelectionNode):
    def __init__(self, left: SelectionNode, right: SelectionNode) -> None:
        self.left, self.right = left, right

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        return sorted(set(self.left.evaluate(atoms)) | set(self.right.evaluate(atoms)))


class AndNode(SelectionNode):
    def __init__(self, left: SelectionNode, right: SelectionNode) -> None:
        self.left, self.right = left, right

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        return sorted(set(self.left.evaluate(atoms)) & set(self.right.evaluate(atoms)))


class NotNode(SelectionNode):
    def __init__(self, inner: SelectionNode) -> None:
        self.inner = inner

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        excluded = set(self.inner.evaluate(atoms))
        return [i for i in range(len(atoms)) if i not in excluded]


class WithinNode(SelectionNode):
    """PyMOL's ``within DIST of (selection)`` — a spatial query."""

    def __init__(self, distance: float, inner: SelectionNode) -> None:
        if distance <= 0:
            raise DataValidationError(f"within: distance must be positive, got {distance}")
        self.distance = distance
        self.inner = inner

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        if not atoms:
            return []
        anchors = np.array(
            [atoms[i].xyz for i in self.inner.evaluate(atoms)], dtype=float
        ).reshape(-1, 3)
        if anchors.size == 0:
            return []
        coords = np.array([a.xyz for a in atoms], dtype=float)
        diff = coords[:, None, :] - anchors[None, :, :]
        distances = np.sqrt((diff * diff).sum(axis=-1))
        return np.nonzero((distances <= self.distance).any(axis=1))[0].tolist()


class FlagNode(SelectionNode):
    """The value-less flags: ``hetatm`` (and ``polymer``, its negation)."""

    def __init__(self, flag: str) -> None:
        self.flag = flag

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        if self.flag == "hetatm":
            return [i for i, a in enumerate(atoms) if a.hetatm]
        return [i for i, a in enumerate(atoms) if not a.hetatm]


class ComparisonNode(SelectionNode):
    """One ``selector operator value-list`` clause."""

    def __init__(self, selector: str, operator: str, values: list[str]) -> None:
        self.selector = selector
        self.operator = operator
        self.values = values

    def _atom_key(self, atom: AtomRecord) -> Any:
        if self.selector in ("name",):
            return atom.name
        if self.selector in ("resn",):
            return atom.resname.upper()
        if self.selector in ("resi", "resid"):
            return atom.resseq
        if self.selector in ("chain",):
            return atom.chain
        if self.selector in ("elem", "element"):
            return atom.element.upper()
        if self.selector in ("alt",):
            return atom.altloc
        if self.selector in ("b",):
            return atom.bfactor
        if self.selector in ("q",):
            return atom.occupancy
        if self.selector in ("model",):
            return atom.model
        raise DataValidationError(f"unknown selector {self.selector!r}")

    def evaluate(self, atoms: Sequence[AtomRecord]) -> list[int]:
        numeric = self.selector in ("resi", "resid", "b", "q", "model")
        if self.operator in ("<", ">", "<=", ">="):
            reference = float(self.values[0])
            hits: list[int] = []
            for i, atom in enumerate(atoms):
                key = self._atom_key(atom)
                try:
                    value = float(key)
                except (TypeError, ValueError):
                    continue
                if self.operator == "<" and value < reference or self.operator == ">" and value > reference or self.operator == "<=" and value <= reference or self.operator == ">=" and value >= reference:
                    hits.append(i)
            return hits
        wanted: set[int | str] = set()
        for raw in self.values:
            if numeric and "-" in raw and not raw.startswith("-"):
                lo_text, _, hi_text = raw.partition("-")
                if lo_text.strip().isdigit() and hi_text.strip().lstrip("+").isdigit():
                    lo, hi = int(lo_text), int(hi_text)
                    wanted.update(range(min(lo, hi), max(lo, hi) + 1))
                    continue
            if numeric:
                try:
                    wanted.add(int(raw))
                except ValueError as exc:
                    raise DataValidationError(
                        f"selection: {self.selector} needs integer values, got {raw!r}"
                    ) from exc
            else:
                wanted.add(raw.upper())
        hits = []
        for i, atom in enumerate(atoms):
            key = self._atom_key(atom)
            if numeric:
                matched = key in wanted
            else:
                key_text = str(key).upper()
                if self.operator == "like":
                    pattern = ".*".join(re.escape(part) for part in self.values[0].split("*"))
                    matched = re.fullmatch(pattern, key_text, flags=re.IGNORECASE) is not None
                else:
                    matched = key_text in wanted
            if self.operator == "!=":
                matched = not matched
            if matched:
                hits.append(i)
        return hits


class Selection:
    """A compiled selection-algebra expression.

    The PyMOL object-model piece that transfers: a selection is a
    first-class, reusable query over *any* atom stack, not a one-off
    filter.

    Examples
    --------
    >>> sel = compile_selection('chain A and name CA and not resi 1-2')
    >>> atoms = build_peptide_backbone(6)
    >>> len(sel(atoms))
    4
    """

    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.root = _SelectionParser(_tokenize_selection(expression)).parse()

    def __repr__(self) -> str:
        return f"Selection({self.expression!r})"

    def apply(self, atoms: Sequence[AtomRecord]) -> list[int]:
        """Matching atom indices, in order."""

        return self.root.evaluate(atoms)

    def __call__(self, atoms: Sequence[AtomRecord]) -> list[AtomRecord]:
        """The matching atoms themselves."""

        return [atoms[i] for i in self.apply(atoms)]


def compile_selection(expression: str) -> Selection:
    """Compile a PyMOL-style selection expression once, reuse forever.

    The grammar (whitespace-insensitive, case-insensitive keywords)::

        expr   := or_expr
        or     := and ( ("or" | "|") and )*
        and    := not ( ("and" | "&") not )*
        not    := ("not" | "!") not | primary
        prim   := "(" expr ")"
               |  "within" DIST "of" "(" expr ")"
               |  flag                                   # hetatm, polymer
               |  selector [op] values                  # name CA, b > 40 ...
        op     := "=" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "in" | "like"
        values := value ( "+" value )*                   # chain A+B, resi 1-57

    Selectors: ``name resn resi/resid chain elem/element alt b q model``.

    Raises
    ------
    DataValidationError
        With the offending token on any syntax or unknown-selector error.

    Examples
    --------
    >>> atoms = build_peptide_backbone(8) + _demo_waters()
    >>> len(select_atoms(atoms, 'name CA'))
    8
    >>> len(select_atoms(atoms, 'resn HOH'))
    2
    >>> len(select_atoms(atoms, 'chain A and resi 3-5 and name CA'))
    3
    >>> len(select_atoms(atoms, 'b > 40'))
    2
    >>> len(select_atoms(atoms, 'not (resn ALA or resn HOH)'))
    0
    >>> len(select_atoms(atoms, 'elem O and chain A'))
    8
    """

    return Selection(expression)


def select_atoms(atoms: Sequence[AtomRecord], expression: str) -> list[AtomRecord]:
    """Evaluate a selection expression against an atom stack."""

    return compile_selection(expression)(atoms)


def select_indices(atoms: Sequence[AtomRecord], expression: str) -> list[int]:
    """``select_atoms``'s index twin (for bookkeeping code)."""

    return compile_selection(expression).apply(atoms)


# --------------------------------------------------------------------------
# secondary structure: Kabsch-Sander energies + DSSP-style assignment
# --------------------------------------------------------------------------


#: The K-S hydrogen-bond acceptance threshold (kcal/mol), as in DSSP.
HBOND_ENERGY_THRESHOLD = -0.5

#: Standard backbone bond lengths (Å) and angles (°) for the builder.
_BUILDER_GEOMETRY = {
    "N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329, "C_O": 1.231,
    "N_CA_C": 111.2, "CA_C_N": 116.2, "C_N_CA": 121.7, "CA_C_O": 120.8,
}


def _place(a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float,
            angle_deg: float, torsion_deg: float) -> np.ndarray:
    """NeRF placement: the point d with |cd|=bond, ∠(b,c,d), τ(a,b,c,d)."""

    angle = math.radians(angle_deg)
    torsion = math.radians(torsion_deg)
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    normal = np.cross(b - a, bc)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 0 else np.array([0.0, 1.0, 0.0])
    m = np.cross(normal, bc)
    local = np.array(
        [
            -bond * math.cos(angle),
            bond * math.sin(angle) * math.cos(torsion),
            bond * math.sin(angle) * math.sin(torsion),
        ]
    )
    return c + np.column_stack([bc, m, normal]) @ local


def build_peptide_backbone(
    n_residues: int = 8,
    phi: float = -57.0,
    psi: float = -47.0,
    *,
    omega: float = 180.0,
    resname: str = "ALA",
    chain: str = "A",
) -> list[AtomRecord]:
    """An ideal-geometry poly-alanine backbone by internal coordinates.

    Standard Engh-Huber bond lengths and angles, placed by the NeRF
    construction with the backbone torsions ``(phi, psi, omega)``
    constant — ``(-57, -47, 180)`` gives the canonical α-helix,
    ``(-139, +135, 180)`` the extended strand.

    Parameters
    ----------
    n_residues : int
        Number of residues (each contributes N, CA, C, O).
    phi, psi, omega : float
        The repeating backbone torsions in degrees.
    resname, chain : str
        Labels written onto every record.

    Returns
    -------
    list[AtomRecord]
        ``4 * n_residues`` records in N/CA/C/O order.

    Examples
    --------
    >>> helix = build_peptide_backbone(3)
    >>> [a.name for a in helix[:5]]
    ['N', 'CA', 'C', 'O', 'N']
    >>> float(round(point_distance(helix[0], helix[1]), 3))   # N-CA
    1.458
    """

    if n_residues < 2:
        raise DataValidationError("build_peptide_backbone: need at least 2 residues")
    n1 = np.zeros(3)
    ca1 = n1 + np.array([_BUILDER_GEOMETRY["N_CA"], 0.0, 0.0])
    dummy = n1 + np.array([0.0, 1.0, 0.0])
    c1 = _place(
        dummy, n1, ca1, _BUILDER_GEOMETRY["CA_C"], _BUILDER_GEOMETRY["N_CA_C"], 180.0
    )
    o1 = _place(
        n1, ca1, c1, _BUILDER_GEOMETRY["C_O"], _BUILDER_GEOMETRY["CA_C_O"], psi + 180.0
    )
    records = [
        AtomRecord(1, "N", resname, chain, 1, "", n1, element="N"),
        AtomRecord(2, "CA", resname, chain, 1, "", ca1, element="C"),
        AtomRecord(3, "C", resname, chain, 1, "", c1, element="C"),
        AtomRecord(4, "O", resname, chain, 1, "", o1, element="O"),
    ]
    prev = [n1, ca1, c1, o1]
    serial = 5
    for resseq in range(2, n_residues + 1):
        n_prev, ca_prev, c_prev, _o_prev = prev
        n_new = _place(
            n_prev, ca_prev, c_prev, _BUILDER_GEOMETRY["C_N"], _BUILDER_GEOMETRY["CA_C_N"], psi
        )
        ca_new = _place(
            ca_prev, c_prev, n_new, _BUILDER_GEOMETRY["N_CA"], _BUILDER_GEOMETRY["C_N_CA"], omega
        )
        c_new = _place(
            c_prev, n_new, ca_new, _BUILDER_GEOMETRY["CA_C"], _BUILDER_GEOMETRY["N_CA_C"], phi
        )
        o_new = _place(
            n_new, ca_new, c_new, _BUILDER_GEOMETRY["C_O"], _BUILDER_GEOMETRY["CA_C_O"],
            psi + 180.0,
        )
        for name, xyz, element in (
            ("N", n_new, "N"), ("CA", ca_new, "C"), ("C", c_new, "C"), ("O", o_new, "O")
        ):
            records.append(
                AtomRecord(serial, name, resname, chain, resseq, "", xyz, element=element)
            )
            serial += 1
        prev = [n_new, ca_new, c_new, o_new]
    return records


def residue_groups(
    atoms: Sequence[AtomRecord],
) -> list[tuple[tuple[str, int, str], list[int]]]:
    """Group atom indices by residue, in first-appearance order.

    Returns
    -------
    list[tuple[tuple[str, int, str], list[int]]]
        ``((chain, resseq, resname), [atom indices])`` per residue.
    """

    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, atom in enumerate(atoms):
        key = (atom.chain, atom.resseq, atom.resname)
        groups.setdefault(key, []).append(index)
    return list(groups.items())


def _backbone_extract(
    atoms: Sequence[AtomRecord],
) -> list[dict[str, Any]]:
    """Per-residue backbone geometry: N/CA/C/O coordinates (+ Cprev)."""

    residues: dict[tuple[str, int, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str]] = []
    for index, atom in enumerate(atoms):
        if atom.name not in ("N", "CA", "C", "O") or atom.altloc not in ("", "A"):
            continue
        key = (atom.chain, atom.resseq, atom.resname)
        if key not in residues:
            residues[key] = {"indices": {}}
            order.append(key)
        residues[key]["indices"][atom.name] = index
    backbone: list[dict[str, Any]] = []
    for position, key in enumerate(order):
        entry = residues[key]
        indices = entry["indices"]
        if not {"N", "C", "O"} <= set(indices):
            continue
        c_prev_index = None
        if position > 0:
            prev_entry = residues[order[position - 1]]
            if prev_entry["indices"].get("C") is not None:
                c_prev_index = prev_entry["indices"]["C"]
        backbone.append(
            {
                "key": key,
                "N": atoms[indices["N"]].xyz,
                "CA": atoms[indices.get("CA", indices["N"])].xyz,
                "C": atoms[indices["C"]].xyz,
                "O": atoms[indices["O"]].xyz,
                "C_prev": atoms[c_prev_index].xyz if c_prev_index is not None else None,
            }
        )
    return backbone


def _amide_h_position(n: np.ndarray, ca: np.ndarray, c_prev: np.ndarray) -> np.ndarray:
    """In-plane amide-H placement (the DSSP source's approximation).

    The hydrogen sits 1.0 Å from N, anti to the average of the Cα and
    previous carbonyl carbon directions — inside the amide plane.
    """

    direction = 2.0 * n - ca - c_prev
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise DataValidationError("degenerate amide geometry: cannot place H")
    return n + direction / norm


def hydrogen_bond_energies(
    atoms: Sequence[AtomRecord], *, threshold: float | None = None
) -> dict[tuple[int, int], float]:
    """Kabsch-Sander backbone hydrogen-bond energies.

    For every ordered residue pair ``(i, j)`` with ``j >= i + 2`` (and
    the reverse direction) where residue ``i`` donates its carbonyl
    O=C(i)···H-N(j), the electrostatic energy::

        E = 0.084 · (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN) · 332  [kcal/mol]

    with the four distances in Å and the amide H placed by the in-plane
    approximation.  Bonds with ``E < -0.5`` (DSSP's threshold, or
    ``threshold`` when given) are returned; everything else is omitted.

    Returns
    -------
    dict[tuple[int, int], float]
        ``{(acceptor_residue_index, donor_residue_index): energy}``,
        residue indices counting backbone-extractable residues from 0.

    Examples
    --------
    >>> atoms = build_peptide_backbone(8)      # alpha-helix: O(i)...N(i+4)
    >>> bonds = hydrogen_bond_energies(atoms)
    >>> bool((0, 4) in bonds and bonds[(0, 4)] < -0.5)
    True
    """

    backbone = _backbone_extract(atoms)
    cutoff = HBOND_ENERGY_THRESHOLD if threshold is None else threshold
    energies: dict[tuple[int, int], float] = {}
    hydrogen: dict[int, np.ndarray | None] = {}
    for j, residue in enumerate(backbone):
        if residue["C_prev"] is None:
            hydrogen[j] = None
        else:
            hydrogen[j] = _amide_h_position(residue["N"], residue["CA"], residue["C_prev"])
    for i, acceptor in enumerate(backbone):
        o = acceptor["O"]
        c = acceptor["C"]
        for j, donor in enumerate(backbone):
            if abs(j - i) < 2 or hydrogen[j] is None:
                continue
            n = donor["N"]
            h = hydrogen[j]
            r_on = float(np.linalg.norm(o - n))
            r_ch = float(np.linalg.norm(c - h))
            r_oh = float(np.linalg.norm(o - h))
            r_cn = float(np.linalg.norm(c - n))
            if min(r_on, r_ch, r_oh, r_cn) <= 0.0:
                continue
            energy = 0.084 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn) * 332.0
            if energy < cutoff:
                energies[(i, j)] = energy
    return energies


def assign_secondary_structure(
    atoms: Sequence[AtomRecord], *, threshold: float | None = None
) -> list[tuple[tuple[str, int, str], str]]:
    """DSSP-style secondary-structure assignment from backbone geometry.

    The pipeline: Kabsch-Sander H-bond energies → turn patterns →
    helices (two consecutive i+4 turns certify ``H`` over their
    span; ``G`` and ``I`` are the i+3 / i+5 equivalents) → bridges
    (the DSSP parallel/antiparallel definitions between
    non-adjacent residues; ladders → ``E``, isolated bridges →
    ``B``) → ``T`` for residues closing an isolated turn, ``S``
    otherwise.

    Returns
    -------
    list[tuple[tuple[str, int, str], str]]
        One ``(residue_key, code)`` per backbone residue, in order —
        the residue key being ``(chain, resseq, resname)``.

    Examples
    --------
    >>> atoms = build_peptide_backbone(8)
    >>> ''.join(code for _, code in assign_secondary_structure(atoms))
    'SHHHHHHT'
    """

    backbone = _backbone_extract(atoms)
    n_res = len(backbone)
    bonds = hydrogen_bond_energies(atoms, threshold=threshold)
    codes = [" "] * n_res

    def bond(acceptor: int, donor: int) -> bool:
        return (acceptor, donor) in bonds

    # -- helices: two consecutive n-turns certify the span between -- #
    for span, letter in ((4, "H"), (3, "G"), (5, "I")):
        streak = 0
        for i in range(n_res):
            streak = streak + 1 if (i + span < n_res and bond(i, i + span)) else 0
            if streak >= 2:
                for k in range(i, min(n_res, i + span)):
                    if letter == "H":
                        codes[k] = "H"  # H outranks the shorter helices
                    elif codes[k] == " ":
                        codes[k] = letter

    # -- bridges: the DSSP parallel / antiparallel definitions ------ #
    bridge_pairs: set[tuple[int, int]] = set()
    for i in range(n_res):
        for j in range(i + 3, n_res):
            parallel = (
                (i >= 1 and j + 1 < n_res and bond(i - 1, j) and bond(j, i + 1))
                or (j >= 1 and i + 1 < n_res and bond(j - 1, i) and bond(i, j + 1))
            )
            antiparallel = (bond(i, j) and bond(j, i)) or (
                i >= 1 and j + 1 < n_res and bond(i - 1, j + 1) and bond(j + 1, i - 1)
            )
            if parallel or antiparallel:
                bridge_pairs.add((i, j))
    laddered: set[int] = set()
    for (i, j) in bridge_pairs:
        if (i + 1, j + 1) in bridge_pairs or (i + 1, j - 1) in bridge_pairs:
            laddered.update((i, j, i + 1))
    for (i, j) in bridge_pairs:
        code = "E" if (i in laddered or j in laddered) else "B"
        for k in (i, j):
            if codes[k] in (" ", "T"):
                codes[k] = code

    # -- turns: a residue closing an n-turn nobody else claimed ---- #
    for i in range(n_res):
        for span in (3, 4, 5):
            if i + span < n_res and bond(i, i + span):
                if codes[i + span] == " ":
                    codes[i + span] = "T"
                break
    codes = ["S" if code == " " else code for code in codes]
    return [(backbone[k]["key"], code) for k, code in enumerate(codes)]


def secondary_structure_string(atoms: Sequence[AtomRecord]) -> str:
    """``assign_secondary_structure`` collapsed to the 1-letter string."""

    return "".join(code for _, code in assign_secondary_structure(atoms))


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------


def structure_summary(atoms: Sequence[AtomRecord]) -> dict[str, Any]:
    """The header-line facts: chains, residues, elements, B-values.

    Examples
    --------
    >>> summary = structure_summary(build_peptide_backbone(8) + _demo_waters())
    >>> (summary['n_atoms'], summary['n_hetatm'], summary['n_residues'])
    (34, 2, 10)
    """

    chains: dict[str, list[int]] = {}
    residues: set[tuple[str, int, str]] = set()
    elements: dict[str, int] = {}
    b_values: list[float] = []
    for atom in atoms:
        chains.setdefault(atom.chain or "_", []).append(atom.resseq)
        residues.add((atom.chain, atom.resseq, atom.resname))
        elements[atom.element] = elements.get(atom.element, 0) + 1
        b_values.append(atom.bfactor)
    return {
        "n_atoms": len(atoms),
        "n_hetatm": sum(1 for a in atoms if a.hetatm),
        "n_residues": len(residues),
        "chains": {
            chain: (min(seq), max(seq)) for chain, seq in chains.items() if seq
        },
        "elements": dict(sorted(elements.items(), key=lambda kv: -kv[1])),
        "mean_b": float(np.mean(b_values)) if b_values else 0.0,
        "max_b": float(np.max(b_values)) if b_values else 0.0,
        "models": sorted({a.model for a in atoms}),
    }


def b_factor_profile(atoms: Sequence[AtomRecord]) -> pd.DataFrame:
    """Per-residue B-factor statistics (the crystallographer's QA view).

    Returns
    -------
    pandas.DataFrame
        Rows ``(chain, resseq, resname)``, columns ``mean_b``, ``min_b``,
        ``max_b``, ``n_atoms``.

    Examples
    --------
    >>> profile = b_factor_profile(build_peptide_backbone(4) + _demo_waters())
    >>> int(profile['n_atoms'].sum())
    18
    """

    import pandas as pd

    rows = []
    for key, indices in residue_groups(atoms):
        b_values = [atoms[i].bfactor for i in indices]
        rows.append(
            {
                "chain": key[0],
                "resseq": key[1],
                "resname": key[2],
                "mean_b": float(np.mean(b_values)),
                "min_b": float(np.min(b_values)),
                "max_b": float(np.max(b_values)),
                "n_atoms": len(indices),
            }
        )
    return pd.DataFrame(rows).set_index(["chain", "resseq", "resname"])

"""Protein-ligand contact geometry and PLIF-style interaction fingerprints.

The archive's docking readers stop at the energy table; this module adds
the structural reading of a docked pose -- *which receptor atoms touch
which ligand atoms, and how*:

* :func:`read_pdb_atoms` -- ATOM/HETATM records of a plain PDB (the
  receptor side; works on the bundled ``lysozyme_1aki.pdb`` unchanged);
* :func:`read_pdbqt_atoms` -- the same records from a PDBQT MODEL block
  (the ligand-pose side, AutoDock atom types included);
* :func:`compute_contacts` -- every ligand-receptor atom pair within a
  cutoff, each classified by a documented distance-geometry rule;
* :func:`contact_matrix` -- the residue x ligand-atom minimum-distance
  grid behind the contact map;
* :func:`plif_matrix` / :func:`plif_bits` -- the residue x
  interaction-class counts (and their 0/1 fingerprint twin) in the
  spirit of Schreyenberg's PLIF, reduced to what pure distance geometry
  supports honestly.

The classification is deliberately conservative -- no force field, no
aromatic perception, no protonation guessing.  Every rule is an
element-and-name distance band, documented on
:func:`classify_interaction`, and anything the rules cannot name falls
through to ``"vdw"`` instead of being silently upgraded.

Examples
--------
>>> from pharmaml.docking.contacts import (
...     ContactAtom, classify_interaction, compute_contacts, contact_matrix,
...     plif_matrix, plif_bits, read_pdb_atoms, read_pdbqt_atoms)

A two-atom ligand meeting a two-atom receptor -- hand-checkable
geometry (an ASP carboxylate oxygen 3.2 A from a ligand nitrogen, a
tryptophan carbon 4.24 A from a ligand carbon):

>>> ligand = [
...     ContactAtom(1, "N1", "N", "LIG", "L", 1, 0.0, 0.0, 0.0, "NA"),
...     ContactAtom(2, "C1", "C", "LIG", "L", 1, 3.0, 0.0, 0.0, "C"),
... ]
>>> receptor = [
...     ContactAtom(1, "OD1", "O", "ASP", "A", 101, 0.0, 3.2, 0.0),
...     ContactAtom(2, "CZ2", "C", "TRP", "A", 62, 6.0, 3.0, 0.0),
... ]
>>> contacts = compute_contacts(ligand, receptor, cutoff=5.0)
>>> [(c.ligand_atom.name, c.receptor_atom.name, round(c.distance, 3), c.interaction)
...  for c in contacts]
[('N1', 'OD1', 3.2, 'salt_bridge'), ('C1', 'CZ2', 4.243, 'hydrophobic'), ('C1', 'OD1', 4.386, 'vdw')]

The N1 x CZ2 pair (6.71 A) never appears -- it is outside the cutoff.
The same reading in matrix form (minimum distance per cell; NaN where
a residue never touches that ligand atom):

>>> mat = contact_matrix(contacts)
>>> list(mat.index)
['A:ASP101', 'A:TRP62']
>>> [round(v, 3) for v in (mat.iloc[0, 0], mat.iloc[1, 0], mat.iloc[0, 1], mat.iloc[1, 1])]
[3.2, nan, 4.386, 4.243]

The PLIF twin -- per-residue interaction-class counts, and the 0/1
fingerprint the pose-comparison workflow bits:

>>> plif = plif_matrix(contacts)
>>> list(plif.columns)
['salt_bridge', 'hbond', 'hydrophobic', 'vdw', 'total']
>>> plif.loc["A:ASP101"].tolist()
[1, 0, 0, 1, 2]
>>> plif_bits(contacts).loc["A:ASP101"].tolist()
[1, 0, 0, 1, 1]

Classification edge cases, each a documented band:

>>> OD1 = ContactAtom(1, "OD1", "O", "ASP", "A", 101, 0.0, 0.0, 0.0)
>>> N1 = ContactAtom(1, "N1", "N", "LIG", "L", 1, 0.0, 0.0, 0.0)
>>> C1 = ContactAtom(1, "C1", "C", "LIG", "L", 1, 0.0, 0.0, 0.0)
>>> classify_interaction(N1, OD1, 3.4)        # N ... carboxylate O within 4.0
'salt_bridge'
>>> classify_interaction(OD1, N1, 3.4)         # direction does not matter
'salt_bridge'
>>> classify_interaction(N1, OD1, 4.1)         # salt band missed
'vdw'
>>> NZ = ContactAtom(1, "NZ", "N", "LYS", "A", 1, 0.0, 0.0, 0.0)
>>> O2 = ContactAtom(1, "O2", "O", "LIG", "L", 1, 0.0, 0.0, 0.0)
>>> classify_interaction(O2, NZ, 3.3)          # plain N/O pair: no carboxylate name
'hbond'
>>> classify_interaction(
...     ContactAtom(1, "OXT", "O", "LIG", "L", 1, 0.0, 0.0, 0.0), NZ, 3.0)
'salt_bridge'
>>> O1 = ContactAtom(1, "O", "O", "HOH", "A", 144, 0.0, 0.0, 0.0)
>>> classify_interaction(N1, O1, 3.0)          # plain N/O pair: the H-bond band
'hbond'
>>> classify_interaction(N1, O1, 3.6)          # N/O but past the H-bond band
'vdw'
>>> CZ2 = ContactAtom(1, "CZ2", "C", "TRP", "A", 62, 0.0, 0.0, 0.0)
>>> classify_interaction(C1, CZ2, 4.4)         # C ... C within 4.5
'hydrophobic'
>>> classify_interaction(C1, CZ2, 4.6)
'vdw'
>>> classify_interaction(N1, CZ2, 3.0)         # cross-element pairs stay generic
'vdw'

Reading a PDB's receptor atoms (fixed-column records; the element
falls back to the name's leading letter when columns 77-78 are blank):

>>> pdb_text = (
...     "ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00 20.78           N\\n"
...     "ATOM      2  CA  MET A   1       2.000   2.000   3.000  1.00 20.78           C\\n"
...     "HETATM    3  O   HOH A 144       5.000   2.000   3.000  1.00 12.00           O\\n"
...     "TER\\n")
>>> atoms = read_pdb_atoms(pdb_text)
>>> [(a.name, a.element, a.residue_key, a.autodock_type) for a in atoms]
[('N', 'N', 'A:MET1', None), ('CA', 'C', 'A:MET1', None), ('O', 'O', 'A:HOH144', None)]
>>> read_pdb_atoms("HEADER    no atoms here")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: no ATOM/HETATM records found in the provided PDB source

Reading a PDBQT pose by MODEL (AutoDock type rides the last column;
records outside the requested model are excluded):

>>> pdbqt_text = (
...     "MODEL 1\\n"
...     "ATOM      1  N1  LIG L   1       0.000   0.000   0.000  1.00  0.00           NA\\n"
...     "ATOM      2  C1  LIG L   1       1.500   0.000   0.000  1.00  0.00            C\\n"
...     "ENDMDL\\n"
...     "MODEL 2\\n"
...     "ATOM      1  N1  LIG L   1      10.000   0.000   0.000  1.00  0.00           NA\\n"
...     "ENDMDL\\n")
>>> pose = read_pdbqt_atoms(pdbqt_text, model=1)
>>> [(a.name, a.autodock_type, a.x) for a in pose]
[('N1', 'NA', 0.0), ('C1', 'C', 1.5)]
>>> read_pdbqt_atoms(pdbqt_text, model=3)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: PDBQT source has 2 model(s); model 3 does not exist

Input validation -- the typed errors instead of silent nonsense:

>>> compute_contacts(ligand, [], cutoff=5.0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: receptor atom list is empty
>>> compute_contacts(ligand, receptor, cutoff=0.0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: cutoff must be a positive finite distance (got 0.0)
>>> read_pdb_atoms("/tmp/pharmaml_doctest_missing_contacts.pdb")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: PDB source not found: /tmp/pharmaml_doctest_missing_contacts.pdb
>>> read_pdbqt_atoms("/tmp/pharmaml_doctest_missing_contacts.pdbqt")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: PDBQT source not found: /tmp/pharmaml_doctest_missing_contacts.pdbqt
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "CARBOXYLATE_OXYGENS",
    "INTERACTION_CLASSES",
    "ContactAtom",
    "ContactRecord",
    "classify_interaction",
    "compute_contacts",
    "contact_matrix",
    "plif_bits",
    "plif_matrix",
    "read_pdb_atoms",
    "read_pdbqt_atoms",
]

_LOG = get_logger(__name__)

#: Receptor-side oxygen names that mark a carboxylate (the salt-bridge rule).
CARBOXYLATE_OXYGENS = frozenset({"OD1", "OD2", "OE1", "OE2", "OXT"})

#: The interaction classes, in classification order (first match wins).
INTERACTION_CLASSES = ("salt_bridge", "hbond", "hydrophobic", "vdw")

_SALT_BRIDGE_MAX = 4.0
_HBOND_MAX = 3.5
_HYDROPHOBIC_MAX = 4.5


@dataclass(frozen=True, slots=True)
class ContactAtom:
    """One ATOM/HETATM record, parsed for contact geometry.

    Attributes
    ----------
    serial:
        The record's atom serial number.
    name:
        Atom name as written (``"OD1"``, ``"N1"``, ...).
    element:
        Single-letter element (``"N"``/``"O"``/``"C"``/``"S"``/...);
        multi-letter AutoDock types (``"NA"``, ``"OA"``, ``"HD"``) are
        normalised to their leading letter.
    resname / chain / resseq:
        Residue identity for grouping; waters ride as ``HOH``.
    x, y, z:
        Cartesian coordinates (A).
    autodock_type:
        The AutoDock atom type column when read from PDBQT (``"NA"``,
        ``"OA"``, ...), ``None`` for plain PDB records.
    """

    serial: int
    name: str
    element: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    autodock_type: str | None = None

    @property
    def residue_key(self) -> str:
        """``"A:ASP101"``-style residue label (chain:resname+number)."""
        return f"{self.chain}:{self.resname}{self.resseq}"

    @property
    def label(self) -> str:
        """Atom label used as a matrix axis (residue + atom name)."""
        return f"{self.residue_key} {self.name}"


@dataclass(frozen=True, slots=True)
class ContactRecord:
    """One ligand-receptor atom pair within the cutoff, classified.

    Attributes
    ----------
    ligand_atom / receptor_atom:
        The pair (:class:`ContactAtom`).
    distance:
        Euclidean distance (A), rounded to 3 decimals.
    interaction:
        One of :data:`INTERACTION_CLASSES`.
    """

    ligand_atom: ContactAtom
    receptor_atom: ContactAtom
    distance: float
    interaction: str


# -- parsing ------------------------------------------------------------------

_ATOM_LINE = ("ATOM  ", "HETATM")


def _read_source(source: str | Path, kind: str) -> str:
    """Accept a path or raw text (the house parser contract)."""
    if isinstance(source, Path):
        if not source.is_file():
            raise DataValidationError(f"{kind} source not found: {source}")
        return source.read_text(encoding="utf-8", errors="ignore")
    if not isinstance(source, str):
        raise DataValidationError(
            f"{kind} source must be text or a path (got {type(source).__name__})"
        )
    # A short single-line string is most likely a path; try to read it
    # and otherwise treat it as literal text (missing paths raise the
    # typed error, single-line record text stays text).
    if "\n" not in source and len(source) < 260:
        candidate = Path(source)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="ignore")
        if not source.startswith(_ATOM_LINE + ("MODEL", "HEADER")):
            raise DataValidationError(f"{kind} source not found: {source}")
    return source


def _normalise_element(raw: str, name: str) -> str:
    """Single-letter element: column 77-78 if usable, else the name."""
    text = raw.strip()
    if text and text[0].isalpha():
        return text[0].upper()
    stripped = name.lstrip("0123456789")
    return (stripped[:1] or "X").upper()


def _parse_atom_line(line: str, is_pdbqt: bool) -> ContactAtom:
    """One fixed-column ATOM/HETATM record -> :class:`ContactAtom`."""
    name = line[12:16].strip()
    resname = line[17:20].strip()
    chain = line[21:22].strip() or " "
    try:
        serial = int(line[6:11])
        resseq = int(line[22:26])
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError as exc:
        raise DataValidationError(
            f"malformed ATOM record (bad numeric columns): {line.rstrip()[:60]}"
        ) from exc
    # PDBQT writes the AutoDock type into the trailing columns 77-79
    # ("NA", "OA", "HD", "C", ...); the element is its leading letter.
    # Plain PDB writes the element there (" N", " C").
    tail = (line[76:79] if is_pdbqt else line[76:78]).strip()
    autodock_type = tail if is_pdbqt else None
    element = _normalise_element(tail[:1] if tail else "", name)
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise DataValidationError(
            f"non-finite coordinate in ATOM record: {line.rstrip()[:60]}"
        )
    return ContactAtom(
        serial=serial,
        name=name or "X",
        element=element,
        resname=resname or "UNK",
        chain=chain,
        resseq=resseq,
        x=x,
        y=y,
        z=z,
        autodock_type=autodock_type,
    )


def read_pdb_atoms(source: str | Path) -> list[ContactAtom]:
    """ATOM/HETATM records of a plain PDB text or file.

    Parameters
    ----------
    source:
        PDB text or a path to a ``.pdb`` file (the receptor side; works
        on the bundled ``lysozyme_1aki.pdb`` unchanged -- waters and
        all).

    Returns
    -------
    list of :class:`ContactAtom`
        In file order; ``autodock_type`` is always ``None``.

    Raises
    ------
    DataValidationError
        If the source is a missing path, holds no ATOM/HETATM records,
        or a record's numeric columns are malformed / non-finite.
    """
    text = _read_source(source, "PDB")
    atoms = [
        _parse_atom_line(line, is_pdbqt=False)
        for line in text.splitlines()
        if line.startswith(_ATOM_LINE)
    ]
    if not atoms:
        raise DataValidationError(
            "no ATOM/HETATM records found in the provided PDB source"
        )
    return atoms


def read_pdbqt_atoms(source: str | Path, model: int = 1) -> list[ContactAtom]:
    """ATOM/HETATM records of a PDBQT text or file, one MODEL at a time.

    Parameters
    ----------
    source:
        PDBQT text or path (a ligand pose, or a receptor PDBQT without
        MODEL blocks -- then every record is returned).
    model:
        1-based MODEL block to read.  MODEL-less sources ignore it.

    Returns
    -------
    list of :class:`ContactAtom`
        In file order, with ``autodock_type`` from the trailing column.

    Raises
    ------
    DataValidationError
        If the source is missing or empty of atom records, ``model``
        is not a positive integer, or the requested MODEL does not
        exist.
    """
    if not isinstance(model, int) or isinstance(model, bool) or model < 1:
        raise DataValidationError(
            f"model must be a positive 1-based integer (got {model!r})"
        )
    text = _read_source(source, "PDBQT")
    atoms: list[ContactAtom] = []
    current: int | None = None
    saw_model = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("MODEL"):
            saw_model = True
            try:
                current = int(stripped.split()[1])
            except (IndexError, ValueError) as exc:
                raise DataValidationError(
                    f"malformed MODEL line: {stripped[:40]}"
                ) from exc
            continue
        if stripped.startswith("ENDMDL"):
            current = None
            continue
        if not line.startswith(_ATOM_LINE):
            continue
        if saw_model and current != model:
            continue
        atoms.append(_parse_atom_line(line, is_pdbqt=True))
    if saw_model and not atoms:
        n_models = sum(1 for ln in text.splitlines() if ln.strip().startswith("MODEL"))
        raise DataValidationError(
            f"PDBQT source has {n_models} model(s); model {model} does not exist"
        )
    if not atoms:
        raise DataValidationError(
            "no ATOM/HETATM records found in the provided PDBQT source"
        )
    return atoms


# -- classification -----------------------------------------------------------

def classify_interaction(ligand: ContactAtom, receptor: ContactAtom,
                         distance: float) -> str:
    """Classify one ligand-receptor atom pair by distance geometry.

    Rules, in order (first match wins; everything else is ``"vdw"``):

    * ``"salt_bridge"`` -- both elements in {N, O}, distance <= 4.0 A,
      and exactly one side is a carboxylate oxygen (name in
      :data:`CARBOXYLATE_OXYGENS`) while the other side's element is N.
      The N side may be the ligand (cationic N into an Asp/Glu pocket)
      or the receptor (Lys/Arg NZ meeting a ligand OXT).
    * ``"hbond"`` -- both elements in {N, O}, distance <= 3.5 A, and
      not a salt bridge.  Donor/acceptor identity is *not* inferred
      (no protonation guessing): any close polar pair counts.
    * ``"hydrophobic"`` -- both elements C, distance <= 4.5 A.
      Aromatic perception (pi-stacking, cation-pi) is out of scope;
      indole/phenyl carbons land here.
    * ``"vdw"`` -- everything else within the caller's cutoff:
      cross-element pairs, N/O pairs past the H-bond band, carbon
      pairs past the hydrophobic band.

    Parameters
    ----------
    ligand, receptor:
        The pair (either side may carry an AutoDock type; only the
        normalised element and the name are consulted).
    distance:
        Their Euclidean distance (A).

    Returns
    -------
    str
        One of :data:`INTERACTION_CLASSES`.

    Raises
    ------
    DataValidationError
        If ``distance`` is not finite or not positive.
    """
    if not isinstance(distance, (int, float)) or isinstance(distance, bool):
        raise DataValidationError(
            f"distance must be a number (got {distance!r})"
        )
    if not math.isfinite(distance) or distance <= 0.0:
        raise DataValidationError(
            f"distance must be positive and finite (got {distance})"
        )
    polar_pair = ligand.element in ("N", "O") and receptor.element in ("N", "O")
    if polar_pair and distance <= _SALT_BRIDGE_MAX:
        lig_carb = ligand.name in CARBOXYLATE_OXYGENS and receptor.element == "N"
        rec_carb = receptor.name in CARBOXYLATE_OXYGENS and ligand.element == "N"
        if lig_carb ^ rec_carb:
            return "salt_bridge"
        if distance <= _HBOND_MAX:
            return "hbond"
        return "vdw"
    if (ligand.element == "C" and receptor.element == "C"
            and distance <= _HYDROPHOBIC_MAX):
        return "hydrophobic"
    return "vdw"


# -- contact computation ------------------------------------------------------

def compute_contacts(ligand: list[ContactAtom], receptor: list[ContactAtom],
                     cutoff: float = 5.0) -> list[ContactRecord]:
    """All ligand-receptor atom pairs within ``cutoff``, classified.

    A cutoff wider than the classic 5.0 A contact shell is allowed
    (PLIF surveys use up to 6.0); the classification bands in
    :func:`classify_interaction` are fixed and independent of it.

    Parameters
    ----------
    ligand:
        Ligand-pose atoms (e.g. :func:`read_pdbqt_atoms` of MODEL 1).
    receptor:
        Receptor atoms (e.g. :func:`read_pdb_atoms` of the PDB).
    cutoff:
        Contact shell in A; positive and finite.  Pairs exactly at the
        cutoff count (``<=``).

    Returns
    -------
    list of :class:`ContactRecord`
        Sorted by (distance, receptor residue key, ligand serial) --
        the closest wall contact first.  An empty list is a valid,
        honest answer ("no atoms within the shell").

    Raises
    ------
    DataValidationError
        If either atom list is empty, ``cutoff`` is not a positive
        finite number, or any coordinate is non-finite.
    """
    if not isinstance(ligand, list) or not isinstance(receptor, list):
        raise DataValidationError(
            "ligand and receptor must be lists of ContactAtom "
            f"(got {type(ligand).__name__} / {type(receptor).__name__})"
        )
    if not ligand:
        raise DataValidationError("ligand atom list is empty")
    if not receptor:
        raise DataValidationError("receptor atom list is empty")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise DataValidationError(
            f"cutoff must be a number (got {cutoff!r})"
        )
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise DataValidationError(
            f"cutoff must be a positive finite distance (got {cutoff})"
        )
    try:
        lig_xyz = np.array([[a.x, a.y, a.z] for a in ligand], dtype=float)
        rec_xyz = np.array([[a.x, a.y, a.z] for a in receptor], dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "ligand/receptor must hold ContactAtom records with numeric "
            f"coordinates ({exc})"
        ) from exc
    if not (np.isfinite(lig_xyz).all() and np.isfinite(rec_xyz).all()):
        raise DataValidationError("non-finite coordinate in ligand/receptor atoms")

    # (n_lig x n_rec) pairwise distances, one vectorised pass.
    delta = lig_xyz[:, None, :] - rec_xyz[None, :, :]
    dist = np.sqrt((delta * delta).sum(axis=2))
    within = np.argwhere(dist <= float(cutoff))

    records: list[ContactRecord] = []
    for lig_i, rec_j in within:
        lig_atom, rec_atom = ligand[int(lig_i)], receptor[int(rec_j)]
        d = round(float(dist[lig_i, rec_j]), 3)
        records.append(
            ContactRecord(
                ligand_atom=lig_atom,
                receptor_atom=rec_atom,
                distance=d,
                interaction=classify_interaction(lig_atom, rec_atom, d),
            )
        )
    records.sort(
        key=lambda c: (c.distance, c.receptor_atom.residue_key, c.ligand_atom.serial)
    )
    _LOG.debug(
        "compute_contacts: %d ligand x %d receptor atoms, cutoff %.2f -> %d contacts",
        len(ligand), len(receptor), cutoff, len(records),
    )
    return records


# -- matrix views -------------------------------------------------------------

def _residue_order(contacts: list[ContactRecord]) -> list[str]:
    """Residue keys ordered by each residue's closest contact."""
    best: dict[str, float] = {}
    for c in contacts:
        key = c.receptor_atom.residue_key
        if key not in best or c.distance < best[key]:
            best[key] = c.distance
    return sorted(best, key=lambda k: (best[k], k))


def contact_matrix(contacts: list[ContactRecord],
                   atom_level: bool = False) -> pd.DataFrame:
    """Minimum-distance grid: receptor residues x ligand atoms.

    Parameters
    ----------
    contacts:
        Output of :func:`compute_contacts`.  An empty list raises --
        a contact map with no contacts is a bug upstream, not a chart.
    atom_level:
        ``False`` (default) groups rows by receptor *residue* (the
        readable map); ``True`` keeps every receptor *atom* its own row.

    Returns
    -------
    pandas.DataFrame
        Float frame; a cell holds the minimum distance over the pairs
        it aggregates, ``NaN`` where that row never touches that
        ligand atom.  Rows are ordered by closest contact; columns by
        ligand atom serial (labelled ``"L:LIG1 N1"``-style).
    """
    if not isinstance(contacts, list) or not contacts:
        raise DataValidationError(
            "contacts must be a non-empty list of ContactRecord (the "
            "contact map has no data -- check the cutoff or the pose)"
        )
    if not isinstance(atom_level, bool):
        raise DataValidationError(
            f"atom_level must be a bool (got {atom_level!r})"
        )
    lig_order: dict[str, int] = {}
    for c in contacts:
        key = f"{c.ligand_atom.residue_key} {c.ligand_atom.name}"
        if key not in lig_order:
            lig_order[key] = c.ligand_atom.serial
    columns = [k for k, _ in sorted(lig_order.items(), key=lambda kv: kv[1])]

    if atom_level:
        row_of = lambda c: c.receptor_atom.label
        row_keys = _atom_row_order(contacts)
    else:
        row_of = lambda c: c.receptor_atom.residue_key
        row_keys = _residue_order(contacts)

    grid = pd.DataFrame(np.nan, index=row_keys, columns=columns, dtype=float)
    for c in contacts:
        row = row_of(c)
        col = f"{c.ligand_atom.residue_key} {c.ligand_atom.name}"
        if math.isnan(grid.at[row, col]) or c.distance < grid.at[row, col]:
            grid.at[row, col] = c.distance
    return grid


def _atom_row_order(contacts: list[ContactRecord]) -> list[str]:
    """Atom labels ordered by (closest contact, residue key, serial)."""
    best: dict[str, tuple[float, str, int]] = {}
    for c in contacts:
        key = c.receptor_atom.label
        serial = c.receptor_atom.serial
        cand = (c.distance, c.receptor_atom.residue_key, serial)
        if key not in best or cand < best[key]:
            best[key] = cand
    return sorted(best, key=lambda k: best[k])


def plif_matrix(contacts: list[ContactRecord]) -> pd.DataFrame:
    """Per-residue interaction-class counts (the PLIF table).

    Rows are receptor residues (closest-contact order), columns are
    :data:`INTERACTION_CLASSES` plus ``"total"``; every cell counts the
    :class:`ContactRecord` entries it aggregates.

    Parameters
    ----------
    contacts:
        Output of :func:`compute_contacts`.

    Returns
    -------
    pandas.DataFrame
        Integer frame.

    Raises
    ------
    DataValidationError
        If ``contacts`` is empty (same contract as
        :func:`contact_matrix`).
    """
    if not isinstance(contacts, list) or not contacts:
        raise DataValidationError(
            "contacts must be a non-empty list of ContactRecord (the "
            "PLIF has no data -- check the cutoff or the pose)"
        )
    rows = _residue_order(contacts)
    columns = list(INTERACTION_CLASSES) + ["total"]
    counts = pd.DataFrame(0, index=rows, columns=columns, dtype=int)
    for c in contacts:
        counts.at[c.receptor_atom.residue_key, c.interaction] += 1
        counts.at[c.receptor_atom.residue_key, "total"] += 1
    return counts


def plif_bits(contacts: list[ContactRecord]) -> pd.DataFrame:
    """The 0/1 fingerprint twin of :func:`plif_matrix`.

    Same shape, but a cell is 1 when the residue has *any* contact of
    that class -- the bit vector a pose-comparison workflow would
    Tanimoto-over (pose similarity = overlapping interaction sets).

    Parameters
    ----------
    contacts:
        Output of :func:`compute_contacts`.

    Returns
    -------
    pandas.DataFrame
        Integer frame of 0/1.
    """
    counts = plif_matrix(contacts)
    return (counts > 0).astype(int)

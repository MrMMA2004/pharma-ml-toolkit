"""Ligand library import (CSV) and batch PDBQT preparation.

Upgrades three scripts from the rar archive that all did the same job
slightly differently:

* ``smilestopdbqt.py`` / ``smiqt.py`` -- read ``ligands.csv`` with a
  hard-coded ``SMILES`` column, created one numbered folder per ligand
  and wrote ``l.pdbqt`` inside it,
* ``ligand_preparation_autodock4.py`` -- the most refined variant:
  per-SMILES error accounting and a ``failed_smiles.csv`` export.

This module unifies them: :func:`load_ligand_csv` imports and validates
the library (column auto-detection included), and
:func:`prepare_ligand_library` turns it into the one-folder-per-ligand
campaign tree every other docking utility expects.  Invalid SMILES are
reported, written to ``failed_smiles.csv`` and never abort the batch.

It also carries the torsion-consistency audit of
``others/Docking/ligand generation/ligprep_match.py`` and
``mismatch2.py``: a prepared PDBQT is only trustworthy when its
TORSDOF equals both the count of active ``REMARK`` torsion definitions
and the count of ``BRANCH`` records.

And it carries the ligand-structure generator of the same folder's
``ligprep.py`` (the 2026-10 "new folder" upload re-delivered it):
:func:`generate_ligand_structures` runs the full tautomer →
multi-conformer ETKDG → MMFF/UFF-lowest-energy pipeline and writes the
one-model PDB (with CONECT records) plus the OpenBabel MOL2 twin, with
the metal-centre skip list and per-row error accounting the original
script had.

Examples
--------
>>> from pathlib import Path
>>> from pharmaml.docking.ligands import check_torsion_consistency, load_ligand_csv
>>> mgl = Path("/tmp/pharmaml_doctest_torsion_mgl.pdbqt")
>>> _ = mgl.write_text(
...     "REMARK  Status 1  A  between atoms: 1_2  and  3_4\\n"
...     "REMARK  Status 2  A  between atoms: 5_6  and  7_8\\n"
...     "BRANCH 1 2\\n"
...     "BRANCH 5 6\\n"
...     "TORSDOF 2\\n"
... )
>>> check = check_torsion_consistency(mgl)   # auto: REMARKs present -> MGLTools rule
>>> (check.torsdof, check.active_remarks, check.branches, check.status())
(2, 2, 2, 'OK')
>>> mgl.unlink(missing_ok=True)

>>> meeko = Path("/tmp/pharmaml_doctest_torsion_meeko.pdbqt")
>>> _ = meeko.write_text("BRANCH 1 2\\nBRANCH 3 4\\nTORSDOF 2\\n")
>>> check_torsion_consistency(meeko).status()          # auto: no REMARKs -> meeko rule
'OK'
>>> check_torsion_consistency(meeko, convention="mgltools").status()
'MISMATCH'
>>> meeko.unlink(missing_ok=True)

>>> library_csv = Path("/tmp/pharmaml_doctest_ligands.csv")
>>> _ = library_csv.write_text("name,smiles\\nethanol,CCO\\nbroken,C1CC\\n")
>>> library = load_ligand_csv(library_csv)   # columns auto-detected (case-insensitive)
>>> len(library), library.validated
(2, True)
>>> library.frame["status"].tolist()
['valid', 'invalid']
>>> library.invalid_frame["error"].tolist()
['RDKit could not parse this SMILES']
>>> library_csv.unlink(missing_ok=True)

>>> check_torsion_consistency("ligand.pdbqt", convention="vina")
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: convention must be 'auto', 'mgltools' or 'meeko', got 'vina'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pharmaml.docking.preparation import smiles_to_pdbqt
from pharmaml.exceptions import DataValidationError, DependencyError, DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table, safe_filename, write_table

_LOG = get_logger(__name__)

__all__ = [
    "NAME_COLUMN_CANDIDATES",
    "SMILES_COLUMN_CANDIDATES",
    "UNSUPPORTED_LIGAND_ATOMS",
    "LigandLibrary",
    "LigandPreparationReport",
    "TorsionCheck",
    "check_torsion_consistency",
    "generate_ligand_structures",
    "load_ligand_csv",
    "prepare_ligand_library",
    "scan_torsion_consistency",
]

# Column names auto-detected when the caller does not pin one
# (case-insensitive; the rar scripts hard-coded "SMILES").
SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "smile",
    "canonical_smiles",
    "smiles_nostereo",
    "smiles_no_stereo",
    "smiles_string",
    "canonical",
)

NAME_COLUMN_CANDIDATES = (
    "name",
    "ligand_name",
    "ligand",
    "compound",
    "compound_name",
    "title",
    "molecule",
    "drug",
    "id",
)


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
    kind: str,
) -> str | None:
    """Find a column by explicit name (case-insensitive) or candidates."""
    lowered = {str(col).strip().lower(): col for col in frame.columns}
    if requested is not None:
        hit = lowered.get(requested.strip().lower())
        if hit is None:
            raise DataValidationError(
                f"{kind} column {requested!r} not found -- available "
                f"columns: {list(frame.columns)}"
            )
        return hit
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _validate_smiles(smiles: object) -> tuple[bool, str]:
    """Return (ok, error) for one SMILES cell."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "empty or non-text SMILES cell"
    try:
        from rdkit import Chem

        # rdkit-stubs does not re-export DisableLog/EnableLog on rdkit.RDLogger,
        # but rdBase (their defining module) is fully typed.
        from rdkit.rdBase import DisableLog, EnableLog
    except ImportError:
        return True, ""  # cannot verify without RDKit; prep will catch it
    # RDKit logs parse errors to stderr before returning None; the
    # message we build below is friendlier than the raw parser trace.
    DisableLog("rdApp.error")
    try:
        ok = Chem.MolFromSmiles(smiles) is not None
    finally:
        EnableLog("rdApp.error")
    return ok, "" if ok else "RDKit could not parse this SMILES"


@dataclass(slots=True)
class LigandLibrary:
    """A validated ligand library loaded from a table file.

    Attributes
    ----------
    frame:
        One row per input ligand with columns ``name``, ``smiles``,
        ``status`` (``"valid"``/``"invalid"``) and ``error``.
    source:
        File the library was loaded from.
    smiles_column / name_column:
        Column names actually used (as found in the source file).
    validated:
        Whether SMILES syntax was checked with RDKit (``False`` when
        RDKit is not installed -- failures then surface at
        :func:`prepare_ligand_library` time instead).
    """

    frame: pd.DataFrame
    source: Path
    smiles_column: str
    name_column: str | None
    validated: bool

    @property
    def valid_frame(self) -> pd.DataFrame:
        """Rows with valid SMILES."""
        return self.frame[self.frame["status"] == "valid"].reset_index(drop=True)

    @property
    def invalid_frame(self) -> pd.DataFrame:
        """Rows that failed validation, with the reason."""
        return self.frame[self.frame["status"] == "invalid"].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __str__(self) -> str:
        n_valid = int((self.frame["status"] == "valid").sum())
        n_invalid = len(self.frame) - n_valid
        unique = self.frame["smiles"].nunique()
        names = "yes" if self.name_column else "index numbers (no name column found)"
        return "\n".join(
            [
                "Ligand library",
                f"  source        : {self.source}",
                (f"  rows          : {len(self.frame)} "
                f"({unique} unique SMILES)"),
                f"  smiles column : {self.smiles_column}",
                f"  names         : {names}",
                f"  valid         : {n_valid}",
                f"  invalid       : {n_invalid}"
                + ("" if self.validated else "  (not pre-validated: RDKit missing)"),
            ]
        )

    def write_failures(self, path: str | Path = "failed_smiles.csv") -> Path | None:
        """Export invalid rows to ``failed_smiles.csv`` (rar parity).

        Returns the written path, or ``None`` when there is nothing to
        report.
        """
        failed = self.invalid_frame
        if failed.empty:
            return None
        out = write_table(failed, path)
        _LOG.info("Wrote %d failed SMILES to %s", len(failed), out)
        return out


@dataclass(slots=True)
class LigandPreparationReport:
    """Audit of one :func:`prepare_ligand_library` batch.

    Attributes
    ----------
    frame:
        One row per ligand with columns ``folder``, ``name``,
        ``smiles``, ``ok``, ``pdbqt``, ``atoms``, ``torsdof`` and
        ``error``.
    root:
        Campaign directory that was populated.
    library:
        The library the batch ran on.
    """

    frame: pd.DataFrame
    root: Path
    library: LigandLibrary

    @property
    def failed_frame(self) -> pd.DataFrame:
        """Rows whose PDBQT preparation failed."""
        return self.frame[~self.frame["ok"]].reset_index(drop=True)

    @property
    def n_prepared(self) -> int:
        return int(self.frame["ok"].sum())

    @property
    def n_failed(self) -> int:
        return int((~self.frame["ok"]).sum())

    def __str__(self) -> str:
        atoms_total = int(self.frame["atoms"].fillna(0).sum())
        tors = int(self.frame["torsdof"].fillna(0).sum())
        return "\n".join(
            [
                "Ligand batch preparation",
                f"  campaign dir : {self.root}",
                (f"  prepared     : {self.n_prepared} ligands "
                f"({atoms_total} atoms, {tors} total rotatable bonds)"),
                f"  failed       : {self.n_failed}"
                + ("  -> failed_smiles.csv" if self.n_failed else ""),
                ("  layout       : one folder per ligand, <name>/l.pdbqt "
                "(scan_docking_directory-ready)"),
            ]
        )

    def write_failures_csv(self, path: str | Path | None = None) -> Path | None:
        """Export failed ligands to CSV (``failed_smiles.csv`` by default)."""
        if self.n_failed == 0:
            return None
        out = Path(path) if path is not None else self.root / "failed_smiles.csv"
        return write_table(self.failed_frame, out)


def load_ligand_csv(
    path: str | Path,
    smiles_col: str | None = None,
    name_col: str | None = None,
) -> LigandLibrary:
    """Load a ligand library from CSV/TSV (or Excel).

    Mirrors the ``ligands.csv`` convention of the original scripts and
    removes its papercuts: the SMILES column is auto-detected when the
    file does not use the canonical ``SMILES`` header, names fall back
    to row numbers, and every SMILES is syntax-checked with RDKit when
    available so bad rows are visible before any 3D embedding happens.

    Parameters
    ----------
    path:
        Table file (``.csv``, ``.tsv``, ``.txt``, ``.xlsx``).
    smiles_col:
        Explicit SMILES column name; ``None`` auto-detects.
    name_col:
        Explicit ligand-name column; ``None`` auto-detects and falls
        back to 1-based row numbers.

    Returns
    -------
    LigandLibrary

    Raises
    ------
    DataValidationError
        Missing/empty file, no usable SMILES column, or no rows.
    """
    src = Path(path)
    if not src.is_file():
        raise DataValidationError(f"ligand file not found: {src}")
    frame = read_table(src)
    if frame.empty:
        raise DataValidationError(f"ligand file has no rows: {src}")

    smiles_column = _resolve_column(frame, smiles_col, SMILES_COLUMN_CANDIDATES, "SMILES")
    if smiles_column is None:
        raise DataValidationError(
            f"no SMILES column found in {src.name} -- columns are "
            f"{list(frame.columns)}; pass smiles_col=... explicitly"
        )
    name_column = _resolve_column(frame, name_col, NAME_COLUMN_CANDIDATES, "name")

    statuses: list[str] = []
    errors: list[str] = []
    for value in frame[smiles_column]:
        ok, why = _validate_smiles(value)
        statuses.append("valid" if ok else "invalid")
        errors.append(why)

    library_frame = pd.DataFrame(
        {
            "name": (
                [_row_name(value, i) for i, value in enumerate(frame[name_column], start=1)]
                if name_column is not None
                else [str(i) for i in range(1, len(frame) + 1)]
            ),
            "smiles": frame[smiles_column],
            "status": statuses,
            "error": errors,
        }
    )
    validated = True
    try:
        import rdkit  # noqa: F401
    except ImportError:
        validated = False
        _LOG.warning(
            "RDKit is not installed: SMILES syntax NOT pre-validated; "
            "failures will surface during prepare_ligand_library"
        )

    n_invalid = int((library_frame["status"] == "invalid").sum())
    _LOG.info(
        "Loaded %d ligands from %s (%d invalid)",
        len(library_frame),
        src,
        n_invalid,
    )
    return LigandLibrary(
        frame=library_frame,
        source=src,
        smiles_column=str(smiles_column),
        name_column=str(name_column) if name_column is not None else None,
        validated=validated,
    )


def _campaign_folder(index: int, name: str, naming: str, width: int) -> str:
    if naming == "index":
        return str(index)  # exact rar convention (1, 2, 3, ...)
    if naming == "name":
        return safe_filename(name)
    return f"{index:0{max(2, width)}d}_{safe_filename(name)}"


def _row_name(raw: object, index: int) -> str:
    """Stringify a ligand-name cell: NaN/blank cells fall back to the
    row number, numeric cells lose their trailing ``.0``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return str(index)
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return str(index)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _pdbqt_stats(pdbqt_path: Path) -> tuple[int, int]:
    """Count atoms and read TORSDOF from a prepared ligand PDBQT."""
    atoms = 0
    torsdof = 0
    for line in pdbqt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms += 1
        elif line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                torsdof = int(parts[1])
    return atoms, torsdof


@dataclass(frozen=True, slots=True)
class TorsionCheck:
    """Torsion-consistency audit of one prepared ligand PDBQT.

    Mirrors the mismatch report columns of
    ``others/Docking/ligand generation/ligprep_match.py`` / ``mismatch2.py``.

    Attributes
    ----------
    ligand:
        Path of the audited PDBQT.
    torsdof:
        Value of the ``TORSDOF`` record (0 when absent).
    active_remarks:
        Number of ``REMARK`` lines marking an active torsion
        (the ``" A "`` convention of the originals).
    branches:
        Number of ``BRANCH`` records.
    ok:
        ``True`` when all three counts agree.
    """

    ligand: str
    torsdof: int
    active_remarks: int
    branches: int
    ok: bool

    def status(self) -> str:
        """``"OK"`` when consistent, ``"MISMATCH"`` otherwise."""
        return "OK" if self.ok else "MISMATCH"


def check_torsion_consistency(
    pdbqt_path: str | Path,
    convention: str = "auto",
) -> TorsionCheck:
    """Compare TORSDOF, active REMARKs and BRANCH counts of a PDBQT.

    AutoDock's ``prepare_ligand4.py`` (MGLTools) writes one ``REMARK ...
    A ...`` torsion definition and one ``BRANCH`` record per rotatable
    bond, and a final ``TORSDOF`` total -- a prepared ligand is only
    valid when all three agree.  The original scripts exported this
    audit to Excel with a red fill on mismatches; pharmaml returns a
    typed :class:`TorsionCheck` instead.

    Meeko-written ligands (what :func:`pharmaml.docking.smiles_to_pdbqt`
    produces) carry no ``REMARK ... A ...`` lines at all; their torsion
    tree invariant is ``TORSDOF == BRANCH`` count.  ``convention``
    selects the rule:

    * ``"auto"`` (default): use the MGLTools three-way rule when the
      file contains at least one active-torsion REMARK, otherwise the
      Meeko two-way rule -- so both lineages audit correctly without
      configuration,
    * ``"mgltools"``: always require TORSDOF == remarks == branches,
    * ``"meeko"``: always require TORSDOF == branches.

    Parameters
    ----------
    pdbqt_path:
        Ligand PDBQT file.
    convention:
        ``"auto"``, ``"mgltools"`` or ``"meeko"``.

    Returns
    -------
    TorsionCheck

    Raises
    ------
    DockingError
        If the PDBQT file is missing or ``convention`` is unknown.
    """
    if convention not in ("auto", "mgltools", "meeko"):
        raise DockingError(
            f"convention must be 'auto', 'mgltools' or 'meeko', got {convention!r}"
        )
    path = Path(pdbqt_path)
    if not path.is_file():
        raise DockingError(f"ligand PDBQT not found: {path}")
    torsdof = 0
    active_remarks = 0
    branches = 0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    torsdof = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith("REMARK") and " A " in line:
            active_remarks += 1
        elif line.startswith("BRANCH"):
            branches += 1
    if convention == "auto":
        rule = "mgltools" if active_remarks > 0 else "meeko"
    else:
        rule = convention
    if rule == "mgltools":
        ok = torsdof == active_remarks == branches
    else:
        ok = torsdof == branches
    return TorsionCheck(
        ligand=str(path),
        torsdof=torsdof,
        active_remarks=active_remarks,
        branches=branches,
        ok=ok,
    )


def scan_torsion_consistency(directory: str | Path) -> pd.DataFrame:
    """Audit every ``.pdbqt`` under a tree for torsion consistency.

    Walks ``directory`` recursively and applies
    :func:`check_torsion_consistency` to each ligand, producing the
    mismatch report of ``ligprep_match.py`` as a DataFrame (unparseable
    files are skipped with a warning instead of aborting the scan).

    Returns
    -------
    pandas.DataFrame
        Columns ``ligand`` (path relative to ``directory``),
        ``torsdof``, ``active_remarks``, ``branches`` and ``status``.

    Raises
    ------
    DockingError
        If ``directory`` is not a directory.
    """
    base = Path(directory)
    if not base.is_dir():
        raise DockingError(f"directory not found: {base}")
    rows: list[dict[str, object]] = []
    for path in sorted(base.rglob("*.pdbqt")):
        try:
            check = check_torsion_consistency(path)
        except DockingError as exc:
            _LOG.warning("Skipping %s: %s", path, exc)
            continue
        rows.append(
            {
                # as_posix (r295): the report key is a cross-platform
                # stable identifier -- ``str()`` would emit backslashes on
                # the Windows runner and break every keyed consumer; on
                # POSIX the two renderings are byte-identical, so the
                # Linux CI's pinned outputs do not drift.
                "ligand": path.relative_to(base).as_posix(),
                "torsdof": check.torsdof,
                "active_remarks": check.active_remarks,
                "branches": check.branches,
                "status": check.status(),
            }
        )
    mismatches = sum(1 for row in rows if row["status"] == "MISMATCH")
    _LOG.info(
        "Torsion audit of %s: %d ligand(s), %d mismatch(es)",
        base,
        len(rows),
        mismatches,
    )
    return pd.DataFrame(
        rows, columns=["ligand", "torsdof", "active_remarks", "branches", "status"]
    )


def prepare_ligand_library(
    library: LigandLibrary | str | Path,
    output_root: str | Path,
    *,
    smiles_col: str | None = None,
    name_col: str | None = None,
    folder_naming: str = "index_name",
    ligand_filename: str = "l.pdbqt",
    seed: int = 0,
) -> LigandPreparationReport:
    """Prepare every ligand of a CSV library into a campaign tree.

    One folder per ligand (the campaign layout ``dpfer.py``,
    ``main.py`` and :func:`~pharmaml.docking.scan_docking_directory`
    all expect), each containing the prepared ``l.pdbqt``.  Invalid
    SMILES are skipped with their reason recorded; failing folders are
    removed again exactly like the original ``smilestopdbqt.py`` did.
    Ligands whose 3D preparation itself fails (e.g. elements without an
    AutoDock4 type, such as boron) are likewise recorded per row with
    ``ok=False`` and their error instead of aborting the batch.

    Parameters
    ----------
    library:
        A :class:`LigandLibrary` or a path to load one from.
    output_root:
        Campaign directory to populate (created if missing).
    smiles_col / name_col:
        Only used when ``library`` is a path.
    folder_naming:
        ``"index_name"`` (``01_aspirin``, default), ``"index"``
        (``1`` -- the rar scripts' convention) or ``"name"``.
    ligand_filename:
        PDBQT filename inside each folder (``l.pdbqt`` matches the
        original scripts and the GPF/DPF generators).
    seed:
        ETKDG embedding seed passed through to
        :func:`~pharmaml.docking.preparation.smiles_to_pdbqt` -- the
        default (0) makes the whole campaign tree deterministic;
        the same seed always rebuilds byte-identical PDBQTs.

    Returns
    -------
    LigandPreparationReport

    Raises
    ------
    DataValidationError
        Unknown ``folder_naming``, or ``"name"`` naming without usable
        ligand names.
    DependencyError
        RDKit/Meeko missing (raised by the first ligand attempt).
    """
    if folder_naming not in {"index", "index_name", "name"}:
        raise DataValidationError(
            f"folder_naming must be 'index', 'index_name' or 'name', "
            f"got {folder_naming!r}"
        )
    if not isinstance(library, LigandLibrary):
        library = load_ligand_csv(library, smiles_col=smiles_col, name_col=name_col)
    if folder_naming == "name" and library.name_column is None:
        raise DataValidationError(
            "folder_naming='name' requires a ligand-name column -- none "
            "was found; use 'index_name' or pass name_col="
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = ligand_filename[:-6] if ligand_filename.lower().endswith(".pdbqt") else ligand_filename

    valid = library.valid_frame
    width = len(str(len(valid)))
    rows: list[dict[str, object]] = []
    used_folders: set[str] = set()
    for i, record in enumerate(valid.itertuples(index=False), start=1):
        name = _row_name(record.name, i)
        folder_name = _campaign_folder(i, name, folder_naming, width)
        while folder_name in used_folders:  # duplicate names: suffix -2, -3...
            folder_name = f"{folder_name}-{i}"
        used_folders.add(folder_name)
        folder = root / folder_name
        error = ""
        pdbqt: Path | None = None
        atoms = 0
        torsdof = 0
        try:
            pdbqt = smiles_to_pdbqt(str(record.smiles), folder, name=stem, seed=seed)
            atoms, torsdof = _pdbqt_stats(pdbqt)
        except DependencyError:
            # Missing RDKit/Meeko cannot get better by retrying: fail the
            # whole batch fast with the actionable install hint.
            raise
        except Exception as exc:  # noqa: BLE001 — convert per-ligand failure to typed error with cleanup
            error = str(exc)
            if folder.exists():
                try:
                    folder.rmdir()
                except OSError:
                    import shutil

                    shutil.rmtree(folder, ignore_errors=True)
            _LOG.warning("Ligand %r failed preparation: %s", name, error)
        rows.append(
            {
                "folder": folder_name,
                "name": name,
                "smiles": record.smiles,
                "ok": pdbqt is not None,
                "pdbqt": str(pdbqt) if pdbqt else "",
                "atoms": atoms,
                "torsdof": torsdof,
                "error": error,
            }
        )

    # Invalid rows ride along so the report covers the whole library.
    for record in library.invalid_frame.itertuples(index=False):
        rows.append(
            {
                "folder": "",
                "name": str(record.name),
                "smiles": record.smiles,
                "ok": False,
                "pdbqt": "",
                "atoms": 0,
                "torsdof": 0,
                "error": str(record.error) or "invalid SMILES",
            }
        )

    report = LigandPreparationReport(
        frame=pd.DataFrame(rows),
        root=root,
        library=library,
    )
    if report.n_failed:
        # rar parity: ligand_preparation_autodock4.py always wrote
        # failed_smiles.csv after the batch.
        report.write_failures_csv()
    _LOG.info(
        "Prepared %d/%d ligands under %s",
        report.n_prepared,
        len(library),
        root,
    )
    return report


#: Elements the ligand-structure generator refuses by default -- the
#: ``unsupported_atoms`` set of ``others/Docking/ligand generation/ligprep.py``
#: (metals without a reliable force-field / charge model for the
#: MMFF/UFF + Gasteiger pipeline; carried as a tuple so the prisma of
#: "no list primitives" stays satisfied -- it is a module constant, not
#: a schema).
UNSUPPORTED_LIGAND_ATOMS = (
    "Fe",
    "Zn",
    "Cu",
    "Mn",
    "Co",
    "Ni",
    "Mo",
    "W",
)


def _pdb_block_with_conect(
    mol: Any,
    name: str,
    conf_id: int = 0,
    default_b_factor: float = 20.0,
) -> str:
    """One-conformer PDB block with HEADER/COMPND and CONECT records.

    Mirrors the ``generate_pdb_block`` helper of ``ligprep.py``: RDKit's
    own ``MolToPDBBlock`` drops bond information MGLTools' ligand
    reader relies on, so the connectivity is written explicitly as
    ``CONECT`` records (one line per atom listing every neighbour).
    """
    from datetime import datetime


    conf = mol.GetConformer(conf_id)
    lines: list[str] = []
    lines.append(
        "HEADER    LIGAND                                  "
        + datetime.now().strftime("%d-%b-%y").upper()
        + "   "
        + name[:3].upper()
    )
    lines.append(f"COMPND    {name}")
    res_name = name[:3].upper()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_name = f"{atom.GetSymbol()}{i + 1}"
        lines.append(
            f"ATOM  {i + 1:5d} {atom_name:>4s} {res_name:3s} A{1:4d}    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 "
            f"{default_b_factor:5.2f}           {atom.GetSymbol():>2s}"
        )
    conect: dict[int, set[int]] = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx() + 1
        a2 = bond.GetEndAtomIdx() + 1
        conect.setdefault(a1, set()).add(a2)
        conect.setdefault(a2, set()).add(a1)
    for atom_idx in sorted(conect):
        lines.append(
            "CONECT{:>5d}{}".format(
                atom_idx,
                "".join(f"{b:>5d}" for b in sorted(conect[atom_idx])),
            )
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _generate_one_structure(
    args: tuple[str, str, str, int, float, bool],
) -> dict[str, object]:
    """Worker body of :func:`generate_ligand_structures` (top-level so a
    process pool can pickle it -- exactly the ``process_ligand`` shape of
    ``ligprep.py``, carrying only plain strings/numbers)."""
    smiles, name, pdb_dir, num_confs, default_b_factor, write_mol2 = args
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    from rdkit.Chem.MolStandardize import rdMolStandardize

    row: dict[str, object] = {
        "ligand": name,
        "smiles": smiles,
        "tautomer": "",
        "pdb": "",
        "mol2": "",
        "forcefield": "",
        "energy": "",
        "conformers": 0,
        "atoms": 0,
        "ok": False,
        "error": "",
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid SMILES string")
        try:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
            row["tautomer"] = Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001 — canonical tautomer is best-effort
            row["error"] = "tautomer canonicalization failed; input form kept"
        unsupported = sorted(
            {atom.GetSymbol() for atom in mol.GetAtoms()}
            & set(UNSUPPORTED_LIGAND_ATOMS)
        )
        if unsupported:
            raise ValueError(
                "unsupported atom(s) " + ", ".join(unsupported)
                + " (no MMFF/UFF parameterisation for the metal centre)"
            )
        mol = Chem.AddHs(mol)
        conf_ids = list(
            rdDistGeom.EmbedMultipleConfs(
                mol, numConfs=num_confs, params=rdDistGeom.ETKDG()
            )
        )
        if not conf_ids:
            raise ValueError("3D embedding returned no conformers")
        row["conformers"] = len(conf_ids)

        forcefield = ""
        lowest_id: int | None = None
        lowest_energy = float("inf")
        if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            for cid in conf_ids:
                try:
                    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                        mol, props, confId=cid
                    )
                    ff.Minimize()
                    energy = ff.CalcEnergy()
                except Exception:  # noqa: BLE001, S112 — per-conformer FF failures are expected
                    continue
                if energy < lowest_energy:
                    lowest_energy, lowest_id = energy, cid
            if lowest_id is not None:
                forcefield = "RDKit MMFF"
        if lowest_id is None:  # MMFF unavailable or all minimisations failed
            for cid in conf_ids:
                try:
                    rdForceFieldHelpers.UFFOptimizeMolecule(mol, confId=cid)
                    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
                        mol, confId=cid
                    )
                    energy = ff.CalcEnergy()
                except Exception:  # noqa: BLE001, S112 — per-conformer FF failures are expected
                    continue
                if energy < lowest_energy:
                    lowest_energy, lowest_id = energy, cid
            if lowest_id is not None:
                forcefield = "RDKit UFF"
        if lowest_id is None:
            raise ValueError("force-field optimisation failed for all conformers")
        row["forcefield"] = forcefield
        row["energy"] = round(lowest_energy, 4)

        best = Chem.Mol(mol)
        best.RemoveAllConformers()
        best.AddConformer(mol.GetConformer(lowest_id), assignId=True)
        row["atoms"] = best.GetNumAtoms()

        safe = safe_filename(name.replace(" ", "_"))
        pdb_path = Path(pdb_dir) / f"{safe}.pdb"
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_text(
            _pdb_block_with_conect(best, safe, conf_id=0, default_b_factor=default_b_factor),
            encoding="utf-8",
        )
        row["pdb"] = str(pdb_path)
        row["ok"] = True

        if write_mol2:
            try:
                from openbabel import pybel

                ob_mol = pybel.readstring("mol", Chem.MolToMolBlock(best))
                ob_mol.addh()
                ob_mol.calccharges(model="gasteiger")
                mol2_path = Path(pdb_dir) / f"{safe}.mol2"
                ob_mol.write("mol2", str(mol2_path), overwrite=True)
                row["mol2"] = str(mol2_path)
            except ImportError:
                row["error"] = (
                    "MOL2 skipped: OpenBabel not installed (pip install openbabel)"
                )
            except Exception as exc:  # noqa: BLE001 — OpenBabel error zoo
                row["error"] = f"MOL2 generation failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — per-ligand failures never abort the batch
        row["error"] = str(exc)
    return row


def generate_ligand_structures(
    library: LigandLibrary | str | Path,
    output_root: str | Path,
    *,
    num_confs: int = 50,
    default_b_factor: float = 20.0,
    write_mol2: bool = True,
    workers: int = 1,
    smiles_col: str | None = None,
    name_col: str | None = None,
) -> pd.DataFrame:
    """Generate 3D ligand structures (PDB + MOL2) from a SMILES library.

    The ``others/Docking/ligand generation/ligprep.py`` pipeline, library
    edition: every valid SMILES is canonicalised to its preferred
    tautomer, embedded as ``num_confs`` ETKDG conformers, minimised
    (MMFF, falling back to UFF when the molecule lacks MMFF
    parameters), and the **lowest-energy conformer** is written as a
    one-model PDB with explicit ``CONECT`` records (the connectivity
    MGLTools' ``prepare_ligand4.py`` reader wants) plus -- when
    OpenBabel is installed -- a Gasteiger-charged MOL2 twin.  Ligands
    carrying metal centres (Fe/Zn/Cu/Mn/Co/Ni/Mo/W by default) are
    skipped with a per-row reason, exactly like the original script's
    ``unsupported_atoms`` filter, and never abort the batch.

    Parameters
    ----------
    library:
        A :class:`LigandLibrary` or a path to load one from.
    output_root:
        Output directory (created if missing); receives ``<name>.pdb``
        and ``<name>.mol2`` per ligand.
    num_confs:
        ETKDG conformers per ligand (the original script used 50).
    default_b_factor:
        B-factor written on every ATOM line (the original used 20.0).
    write_mol2:
        Also write the OpenBabel MOL2 twin; when OpenBabel is missing
        the PDB still lands and the row records the reason.
    workers:
        Parallel worker processes (1 = inline, the deterministic
        default; the original used ``cpu_count() - 1``).

    Returns
    -------
    pandas.DataFrame
        One row per library ligand: ``ligand``, ``smiles``,
        ``tautomer``, ``pdb``, ``mol2``, ``forcefield``, ``energy``,
        ``conformers``, ``atoms``, ``ok``, ``error``.

    Raises
    ------
    DependencyError
        RDKit missing.
    DataValidationError
        ``num_confs`` or ``workers`` out of range.

    Examples
    --------
    >>> from pathlib import Path
    >>> from pharmaml.docking.ligands import generate_ligand_structures
    >>> csv = Path("/tmp/pharmaml_doctest_gen.csv")
    >>> _ = csv.write_text("name,smiles\\naspirin,CC(=O)Oc1ccccc1C(=O)O\\n")
    >>> out = Path("/tmp/pharmaml_doctest_gen_out")
    >>> frame = generate_ligand_structures(csv, out, num_confs=3, write_mol2=False)
    >>> bool(frame["ok"].iloc[0]), (out / "aspirin.pdb").is_file()
    (True, True)
    >>> "CONECT" in (out / "aspirin.pdb").read_text()
    True
    >>> for p in sorted(out.iterdir()): p.unlink()
    >>> csv.unlink(); out.rmdir()
    """
    if num_confs < 1 or num_confs > 1000:
        raise DataValidationError(
            f"num_confs must be between 1 and 1000, got {num_confs}"
        )
    if workers < 1 or workers > 64:
        raise DataValidationError(f"workers must be between 1 and 64, got {workers}")
    if not isinstance(library, LigandLibrary):
        library = load_ligand_csv(library, smiles_col=smiles_col, name_col=name_col)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    valid = library.valid_frame
    jobs: list[tuple[str, str, str, int, float, bool]] = []
    for i, record in enumerate(valid.itertuples(index=False), start=1):
        name = _row_name(getattr(record, "name", None), i)
        jobs.append((str(record.smiles), name, str(root), num_confs, default_b_factor, write_mol2))

    rows: list[dict[str, object]] = []
    if jobs and workers > 1:
        # spawn (not the Linux fork default): the playground runs this
        # inside a worker THREAD, and Python 3.12+ warns -- correctly --
        # that forking a multi-threaded process can deadlock the child.
        # The jobs carry only plain strings/numbers, so spawn's pickle
        # contract is satisfied by construction.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(_generate_one_structure, jobs))
    else:
        rows = [_generate_one_structure(job) for job in jobs]
    for record in library.invalid_frame.itertuples(index=False):
        rows.append(
            {
                "ligand": str(record.name),
                "smiles": record.smiles,
                "tautomer": "",
                "pdb": "",
                "mol2": "",
                "forcefield": "",
                "energy": "",
                "conformers": 0,
                "atoms": 0,
                "ok": False,
                "error": str(record.error) or "invalid SMILES",
            }
        )
    frame = pd.DataFrame(rows)
    n_ok = int(frame["ok"].sum()) if len(frame) else 0
    _LOG.info(
        "Generated %d/%d ligand structures under %s (%d conformers each)",
        n_ok,
        len(frame),
        root,
        num_confs,
    )
    return frame

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
``others/Docking/ligand generation/niggamatch.py`` and
``mismatch2.py``: a prepared PDBQT is only trustworthy when its
TORSDOF equals both the count of active ``REMARK`` torsion definitions
and the count of ``BRANCH`` records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

from pharmaml.docking.preparation import smiles_to_pdbqt
from pharmaml.exceptions import DataValidationError, DependencyError, DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table, safe_filename, write_table

_LOG = get_logger(__name__)

__all__ = [
    "LigandLibrary",
    "LigandPreparationReport",
    "load_ligand_csv",
    "prepare_ligand_library",
    "TorsionCheck",
    "check_torsion_consistency",
    "scan_torsion_consistency",
    "SMILES_COLUMN_CANDIDATES",
    "NAME_COLUMN_CANDIDATES",
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
        from rdkit import Chem, RDLogger
    except ImportError:
        return True, ""  # cannot verify without RDKit; prep will catch it
    # RDKit logs parse errors to stderr before returning None; the
    # message we build below is friendlier than the raw parser trace.
    RDLogger.DisableLog("rdApp.error")
    try:
        ok = Chem.MolFromSmiles(smiles) is not None
    finally:
        RDLogger.EnableLog("rdApp.error")
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
                f"  rows          : {len(self.frame)} "
                f"({unique} unique SMILES)",
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
                f"  prepared     : {self.n_prepared} ligands "
                f"({atoms_total} atoms, {tors} total rotatable bonds)",
                f"  failed       : {self.n_failed}"
                + ("  -> failed_smiles.csv" if self.n_failed else ""),
                f"  layout       : one folder per ligand, <name>/l.pdbqt "
                "(scan_docking_directory-ready)",
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
    ``others/Docking/ligand generation/niggamatch.py`` / ``mismatch2.py``.

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


def check_torsion_consistency(pdbqt_path: str | Path) -> TorsionCheck:
    """Compare TORSDOF, active REMARKs and BRANCH counts of a PDBQT.

    AutoDock's ``prepare_ligand4.py`` writes one ``REMARK ... A ...``
    torsion definition and one ``BRANCH`` record per rotatable bond,
    and a final ``TORSDOF`` total -- a prepared ligand is only valid
    when all three agree.  The original scripts exported this audit to
    Excel with a red fill on mismatches; pharmaml returns a typed
    :class:`TorsionCheck` instead.

    Parameters
    ----------
    pdbqt_path:
        Ligand PDBQT file.

    Returns
    -------
    TorsionCheck

    Raises
    ------
    DockingError
        If the PDBQT file is missing.
    """
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
    ok = torsdof == active_remarks == branches
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
    mismatch report of ``niggamatch.py`` as a DataFrame (unparseable
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
                "ligand": str(path.relative_to(base)),
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
    library: Union[LigandLibrary, str, Path],
    output_root: str | Path,
    *,
    smiles_col: str | None = None,
    name_col: str | None = None,
    folder_naming: str = "index_name",
    ligand_filename: str = "l.pdbqt",
) -> LigandPreparationReport:
    """Prepare every ligand of a CSV library into a campaign tree.

    One folder per ligand (the campaign layout ``dpfer.py``,
    ``main.py`` and :func:`~pharmaml.docking.scan_docking_directory`
    all expect), each containing the prepared ``l.pdbqt``.  Invalid
    SMILES are skipped with their reason recorded; failing folders are
    removed again exactly like the original ``smilestopdbqt.py`` did.

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
            pdbqt = smiles_to_pdbqt(str(record.smiles), folder, name=stem)
            atoms, torsdof = _pdbqt_stats(pdbqt)
        except DependencyError:
            # Missing RDKit/Meeko cannot get better by retrying: fail the
            # whole batch fast with the actionable install hint.
            raise
        except Exception as exc:
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

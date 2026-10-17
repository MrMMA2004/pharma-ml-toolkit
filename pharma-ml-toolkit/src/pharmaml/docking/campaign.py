"""Docking-campaign file management: collect, distribute, clean, report.

Upgrades the campaign housekeeping scripts of the rar archive, which
all operated on the one-folder-per-ligand campaign tree with hardcoded
``os.getcwd()`` roots:

* ``others/General/extaacter.py`` / ``others/dlg_extracter.py`` /
  ``others/Reader for alphabetic folders/dlg_extracter_not_numeral.py``
  -- collect every ``r.dlg`` from the tree into one folder, renamed
  after the parent ligand folder (with a ``--dry-run`` mode),
* ``others/Docking/ligand_pdbqt_extracter.py`` -- the same collection
  driven by a regex (``^\\d+\\.pdbqt$``) instead of an exact name,
* ``others/General/adder.py`` -- distribute the receptor and the
  AutoDock binaries into every ligand folder,
* ``others/General/delete.py`` / ``deleter.py`` -- campaign cleanup
  with a filename allowlist + regex, or deleting one named file
  everywhere,
* ``others/General/Nameextracter.py`` -- list file stems to a CSV,
* ``others/Prettier_extracted_dlgs.py`` -- reindex a DLG summary onto
  the full ligand-number range (0..300) so missing ligands are visible,
* ``others/Docking/gpfreader.py`` -- per-folder ``ligand_types`` audit
  of every ``r.gpf``.

Key safety improvements over the originals:

* every function takes explicit paths (no ``os.chdir``),
* destructive operations support ``dry_run`` and never touch the
  filesystem when it is set,
* reports are pandas DataFrames instead of print statements, and
* collected/derived filenames are sanitised with
  :func:`pharmaml.utils.io.safe_filename`.

Examples
--------
>>> import shutil
>>> from pathlib import Path
>>> from pharmaml.docking.campaign import collect_files
>>> root = Path("/tmp/pharmaml_doctest_campaign")
>>> shutil.rmtree(root, ignore_errors=True)               # deterministic start
>>> for folder in ("ligand_01", "ligand_02"):
...     (root / folder).mkdir(parents=True)
...     _ = (root / folder / "r.dlg").write_text("dlg")
...     _ = (root / folder / "1.pdbqt").write_text("pdbqt")
>>> plan = collect_files(root, filename="r.dlg", dry_run=True)
>>> (plan["folder"].tolist(), plan["copied"].tolist())
(['ligand_01', 'ligand_02'], [False, False])
>>> numbered = collect_files(root, filename=None, pattern=r"^\\d+\\.pdbqt$")
>>> sorted(Path(p).name for p in numbered["destination"])
['ligand_01.pdbqt', 'ligand_02.pdbqt']
>>> _ = collect_files(root, filename="r.dlg")        # the classic r.dlg extractor
>>> sorted(p.name for p in (root / "collected").iterdir())
['ligand_01.dlg', 'ligand_01.pdbqt', 'ligand_02.dlg', 'ligand_02.pdbqt']
>>> collect_files(root, filename=None)
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: collect_files needs a filename or a pattern to match
>>> shutil.rmtree(root)

>>> import pandas as pd
>>> from pharmaml.docking.campaign import reindex_by_number
>>> dlg = pd.DataFrame({"ligand": ["5_result.dlg", "6_result.dlg", "8_result.dlg"],
...                     "energy": [-7.1, -8.2, -6.0]})
>>> wide = reindex_by_number(dlg, "ligand", start=5, stop=7)
>>> (wide.index.tolist(), wide["energy"].tolist())
([5, 6, 7], [-7.1, -8.2, nan])
>>> reindex_by_number(pd.DataFrame({"ligand": ["5_result.dlg"]}), "nope")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: column 'nope' not found -- available columns: ['ligand']

>>> from pharmaml.docking.campaign import (
...     cleanup_tree, delete_named_files, distribute_files, scan_gpf_ligand_types)
>>> root = Path("/tmp/pharmaml_doctest_campaign")
>>> shutil.rmtree(root, ignore_errors=True)
>>> root.mkdir()
>>> for sub in ("lig1", "lig2"):
...     (root / sub).mkdir()
>>> _ = (root / "r.pdbqt").write_text("receptor")
>>> _ = (root / "lig1" / "r.gpf").write_text("ligand_types C HD OA   # comment\\n")
>>> sorted(str(p.relative_to(root)) for p in distribute_files([root / "r.pdbqt"], root))
['lig1/r.pdbqt', 'lig2/r.pdbqt']
>>> scan_gpf_ligand_types(root)["ligand_types"].tolist()
['C HD OA', 'not found']
>>> delete_named_files(root, "r.pdbqt")             # the root copy plus two ligand copies
3
>>> plan = cleanup_tree(root, keep_pattern=r"^\\d+\\.pdbqt$", remove_empty=True, dry_run=True)
>>> sorted(plan["action"].tolist())                 # preview only: nothing is deleted yet
['deleted', 'removed_dir', 'removed_dir']
>>> _ = cleanup_tree(root, keep_pattern=r"^\\d+\\.pdbqt$", remove_empty=True)
>>> sorted(str(p.relative_to(root)) for p in root.rglob("*"))
[]
>>> shutil.rmtree(root)
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pandas as pd

from pharmaml.exceptions import DataValidationError, DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import safe_filename

_LOG = get_logger(__name__)

__all__ = [
    "cleanup_tree",
    "collect_files",
    "delete_named_files",
    "distribute_files",
    "list_file_names",
    "reindex_by_number",
    "scan_gpf_ligand_types",
]


def _compile_pattern(pattern: str, *, argument: str) -> re.Pattern[str]:
    """Compile ``pattern`` or raise a :class:`DockingError` naming it."""
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise DockingError(
            f"invalid regex for {argument}: {pattern!r} ({exc})"
        ) from exc


def collect_files(
    base_dir: str | Path,
    filename: str | None = "r.dlg",
    pattern: str | None = None,
    out_dir: str | Path | None = None,
    rename_by_folder: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Collect matching files from a campaign tree into one folder.

    Walks ``base_dir`` recursively and copies every file whose name
    equals ``filename`` *or* matches the regex ``pattern`` into
    ``out_dir`` (default ``<base_dir>/collected``).  With
    ``rename_by_folder=True`` the destination is named after the
    (sanitised) parent folder -- ``ligand_07/r.dlg`` becomes
    ``ligand_07.dlg`` -- so results from different folders never
    overwrite each other; duplicate names get ``_1``, ``_2`` ... suffixes.

    The two filters combine with OR: a call carrying both ``filename``
    and ``pattern`` collects the union of both match sets, so a pure
    pattern collect requires ``filename=None``.

    Parameters
    ----------
    base_dir:
        Campaign tree to scan.
    filename:
        Exact filename to collect (``r.dlg`` reproduces the DLG
        extractors).
    pattern:
        Regex the filename must match, anchored at the start like
        :func:`re.match` (use ``^...$`` anchors for a full-name match,
        e.g. ``r"^\\d+\\.pdbqt$"`` for the numbered PDBQT collector).
    out_dir:
        Destination folder; defaults to ``<base_dir>/collected``.
    rename_by_folder:
        Name destinations after their parent folder (the DLG-extractor
        convention); ``False`` keeps the original filename.
    dry_run:
        Report the planned copies without touching the filesystem.

    Returns
    -------
    pandas.DataFrame
        One row per match with columns ``source``, ``destination``,
        ``folder`` and ``copied``.

    Raises
    ------
    DockingError
        If ``base_dir`` is not a directory, or neither ``filename`` nor
        ``pattern`` is given.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise DockingError(f"base directory not found: {base}")
    if filename is None and pattern is None:
        raise DockingError("collect_files needs a filename or a pattern to match")
    regex = _compile_pattern(pattern, argument="pattern") if pattern else None
    out = Path(out_dir) if out_dir is not None else base / "collected"
    out_resolved = out.resolve()

    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = {p.name for p in out.iterdir() if p.is_file()} if out.is_dir() else set()

    rows: list[dict[str, object]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        if out_resolved in path.resolve().parents:
            continue  # never collect from the output folder itself
        name = path.name
        if not (
            (filename is not None and name == filename)
            or (regex is not None and regex.match(name) is not None)
        ):
            continue
        if rename_by_folder:
            stem = safe_filename(path.parent.name)
            suffix = path.suffix
        else:
            stem = path.stem
            suffix = path.suffix
        candidate = f"{stem}{suffix}"
        counter = 1
        while candidate in used_names or (not dry_run and (out / candidate).exists()):
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate)
        destination = out / candidate
        copied = False
        if not dry_run:
            shutil.copy2(path, destination)
            copied = True
        rows.append(
            {
                "source": str(path),
                "destination": str(destination),
                "folder": path.parent.name,
                "copied": copied,
            }
        )

    n_matches = len(rows)
    n_copied = sum(1 for row in rows if row["copied"])
    _LOG.info(
        "Collected %d matching file(s) into %s (%d copied%s)",
        n_matches,
        out,
        n_copied,
        " -- dry run" if dry_run else "",
    )
    return pd.DataFrame(rows, columns=["source", "destination", "folder", "copied"])


def list_file_names(
    directory: str | Path,
    extension: str = ".csv",
    out_path: str | Path | None = None,
) -> list[str]:
    """List the stems of all files with ``extension`` in a directory.

    Upgrades ``others/General/Nameextracter.py``: non-recursive,
    alphabetically sorted, and optionally written as a single-column
    CSV with the ``name`` header.

    Parameters
    ----------
    directory:
        Folder whose direct files are listed.
    extension:
        File extension to filter by (case-insensitive; a leading dot is
        added when missing).
    out_path:
        Optional CSV path for the name list.

    Returns
    -------
    list of str
        Sorted file stems.

    Raises
    ------
    DockingError
        If ``directory`` is not a directory.
    """
    base = Path(directory)
    if not base.is_dir():
        raise DockingError(f"directory not found: {base}")
    ext = extension if extension.startswith(".") else f".{extension}"
    names = sorted(
        p.stem for p in base.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()
    )
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"name": names}).to_csv(out, index=False)
        _LOG.info("Wrote %d file names to %s", len(names), out)
    return names


def distribute_files(
    files: list[str | Path],
    target_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Copy each file into every immediate sub-directory of ``target_dir``.

    Upgrades ``others/General/adder.py``, which pushed the AutoDock
    binaries and the receptor PDBQT into every ligand folder of the
    campaign tree.  Existing copies are skipped unless ``overwrite``.

    Parameters
    ----------
    files:
        Source files to distribute.
    target_dir:
        Directory whose *immediate* sub-directories each receive a copy.
    overwrite:
        Replace existing destination files.

    Returns
    -------
    list of Path
        Every destination actually written.

    Raises
    ------
    DockingError
        If ``target_dir`` is not a directory or a source file is missing.
    """
    target = Path(target_dir)
    if not target.is_dir():
        raise DockingError(f"target directory not found: {target}")
    sub_directories = sorted(p for p in target.iterdir() if p.is_dir())
    if not sub_directories:
        _LOG.warning(
            "no sub-directories under %s -- nothing to distribute", target
        )
        return []

    written: list[Path] = []
    for source in files:
        src = Path(source)
        if not src.is_file():
            raise DockingError(f"file to distribute not found: {src}")
        for sub in sub_directories:
            destination = sub / src.name
            if destination.exists() and not overwrite:
                _LOG.debug("Skipping existing %s", destination)
                continue
            shutil.copy2(src, destination)
            written.append(destination)
    _LOG.info(
        "Distributed %d file(s) into %d sub-directory(ies) (%d copies)",
        len(files),
        len(sub_directories),
        len(written),
    )
    return written


def cleanup_tree(
    directory: str | Path,
    keep: set[str] | list[str] | frozenset[str] = frozenset(),
    keep_pattern: str | None = None,
    remove_empty: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Delete every file not on the allowlist, bottom-up.

    Upgrades ``others/General/delete.py``: files survive only when
    their name is in ``keep`` or matches the ``keep_pattern`` regex
    (e.g. ``r"^\\d+\\.pdbqt$"`` keeps numbered ligand PDBQTs); empty
    directories left behind can be removed as well.  With
    ``dry_run=True`` nothing is deleted -- the report simply shows what
    *would* happen.

    Parameters
    ----------
    directory:
        Tree to clean (the root itself is never removed).
    keep:
        Filenames to preserve.
    keep_pattern:
        Regex the filenames must match to be preserved.
    remove_empty:
        Remove directories that are (or become) empty.  With
        ``dry_run=True`` the report *previews* the directories that
        would be removed -- i.e. folders that are empty already or
        whose entire contents are slated for deletion.
    dry_run:
        Report the planned deletions without touching the filesystem.

    Returns
    -------
    pandas.DataFrame
        One row per visited file/directory with columns ``path`` and
        ``action`` (``"kept"``, ``"deleted"`` or ``"removed_dir"``).

    Raises
    ------
    DockingError
        If ``directory`` is not a directory.
    """
    base = Path(directory)
    if not base.is_dir():
        raise DockingError(f"directory not found: {base}")
    keep_names = set(keep)
    regex = _compile_pattern(keep_pattern, argument="keep_pattern") if keep_pattern else None

    rows: list[dict[str, str]] = []
    deleted_count = 0
    removed_dirs = 0
    # Dry-run preview support: files slated for deletion, and directories
    # that would end up empty (filled bottom-up, so every directory is
    # classified before its parent is visited by os.walk).
    planned_deleted: set[Path] = set()
    would_be_empty: set[Path] = set()
    for root, dirs, files in os.walk(base, topdown=False):
        root_path = Path(root)
        for name in sorted(files):
            path = root_path / name
            if name in keep_names or (regex is not None and regex.match(name) is not None):
                rows.append({"path": str(path), "action": "kept"})
                continue
            rows.append({"path": str(path), "action": "deleted"})
            planned_deleted.add(path)
            if dry_run:
                continue
            try:
                path.unlink()
                deleted_count += 1
            except OSError as exc:
                _LOG.warning("Could not delete %s: %s", path, exc)
        if remove_empty:
            if dry_run:
                # Preview: mark this root as would-be-empty when every
                # direct file is slated for deletion and every sub-directory
                # already qualified (bottom-up walk guarantees sub-dirs were
                # visited first).  Already-empty folders qualify trivially.
                if all((root_path / n) in planned_deleted for n in files) and all(
                    (root_path / n) in would_be_empty for n in dirs
                ):
                    would_be_empty.add(root_path)
                for name in sorted(dirs):
                    if (root_path / name) in would_be_empty:
                        rows.append(
                            {"path": str(root_path / name), "action": "removed_dir"}
                        )
            else:
                for name in sorted(dirs):
                    dir_path = root_path / name
                    if not dir_path.is_dir():
                        continue
                    if any(dir_path.iterdir()):
                        continue
                    rows.append({"path": str(dir_path), "action": "removed_dir"})
                    try:
                        dir_path.rmdir()
                        removed_dirs += 1
                    except OSError as exc:
                        _LOG.warning(
                            "Could not remove empty directory %s: %s", dir_path, exc
                        )

    _LOG.info(
        "Cleanup of %s: %d file deletion(s), %d empty directory(ies) removed%s",
        base,
        deleted_count,
        removed_dirs,
        " -- dry run" if dry_run else "",
    )
    return pd.DataFrame(rows, columns=["path", "action"])


def delete_named_files(
    directory: str | Path,
    filename: str,
    dry_run: bool = False,
) -> int:
    """Delete every file named ``filename`` anywhere under ``directory``.

    Upgrades ``others/General/deleter.py`` (which removed stray
    ``r.gpf`` files after grid generation).  Returns the number of
    files deleted -- or, with ``dry_run=True``, the number that *would*
    be deleted.

    Raises
    ------
    DockingError
        If ``directory`` is not a directory.
    """
    base = Path(directory)
    if not base.is_dir():
        raise DockingError(f"directory not found: {base}")
    deleted = 0
    for root, _dirs, _files in os.walk(base):
        path = Path(root) / filename
        if not path.is_file():
            continue
        try:
            if not dry_run:
                path.unlink()
            deleted += 1
        except OSError as exc:
            _LOG.warning("Could not delete %s: %s", path, exc)
    _LOG.info(
        "Deleted %d file(s) named %r under %s%s",
        deleted,
        filename,
        base,
        " -- dry run" if dry_run else "",
    )
    return deleted


def reindex_by_number(
    frame: pd.DataFrame,
    column: str,
    start: int = 0,
    stop: int = 300,
    pattern: str = r"(\d+)",
) -> pd.DataFrame:
    """Reindex a frame onto the full ligand-number range.

    Upgrades ``others/Prettier_extracted_dlgs.py``: the first capture
    group of ``pattern`` is extracted from ``column`` (e.g. the ligand
    number out of ``5_result.dlg``), used as the index, and the frame
    is left-joined onto ``range(start, stop + 1)`` so rows for ligands
    with no result appear as NaN instead of silently missing.  The
    reverse direction is guarded too (27-A): rows whose ``column``
    value does not match ``pattern`` at all, or whose extracted number
    falls outside ``[start, stop]``, are dropped **with a warning**
    naming the count — never silently.

    Parameters
    ----------
    frame:
        Frame to reindex; the original ``column`` is preserved intact.
    column:
        Column holding strings with embedded numbers.
    start, stop:
        Inclusive ends of the target index range.
    pattern:
        Regex with at least one capture group (the first group is used
        when several are present; zero-group patterns raise
        :class:`DataValidationError`).

    Returns
    -------
    pandas.DataFrame
        Reindexed frame (missing rows all-NaN).

    Raises
    ------
    DataValidationError
        If ``column`` is not present in ``frame``, or ``pattern`` is
        invalid or has no capture group.
    """
    if column not in frame.columns:
        raise DataValidationError(
            f"column {column!r} not found -- available columns: {list(frame.columns)}"
        )
    try:
        compiled = _compile_pattern(pattern, argument="pattern")
    except DockingError as exc:
        # reindex_by_number is a table operation -- its contract is
        # DataValidationError, not the filesystem-flavoured DockingError.
        raise DataValidationError(str(exc)) from exc
    if compiled.groups < 1:
        raise DataValidationError(
            f"pattern {pattern!r} has no capture group -- exactly one is "
            f"required (e.g. r'(\\d+)' to extract the ligand number)"
        )
    numbers = frame[column].astype(str).str.extract(pattern, expand=False)
    if isinstance(numbers, pd.DataFrame):  # multi-group pattern
        numbers = numbers[0]
    working = frame.copy()
    working["_number"] = pd.to_numeric(numbers, errors="coerce")
    unparsed = int(working["_number"].isna().sum())
    if unparsed:
        _LOG.warning(
            "%d row(s) in %r do not match pattern %r -- dropped from the "
            "reindexed frame (clean the column or extend the pattern)",
            unparsed,
            column,
            pattern,
        )
    working = working[working["_number"].notna()]
    working["_number"] = working["_number"].astype(int)
    out_of_range = int((~working["_number"].between(start, stop)).sum())
    if out_of_range:
        _LOG.warning(
            "%d row(s) with ligand number(s) outside [%d, %d] -- dropped "
            "from the reindexed frame (widen start/stop to keep them)",
            out_of_range,
            start,
            stop,
        )
    working = working[working["_number"].between(start, stop)]
    duplicates = int(working["_number"].duplicated().sum())
    if duplicates:
        _LOG.warning(
            "%d duplicate ligand number(s) in %r -- keeping the first row",
            duplicates,
            column,
        )
        working = working[~working["_number"].duplicated(keep="first")]
    working = working.set_index("_number")
    working.index.name = None
    result = working.reindex(range(start, stop + 1))
    _LOG.debug(
        "Reindexed %d row(s) of %r onto %d..%d", len(working), column, start, stop
    )
    return result


def scan_gpf_ligand_types(directory: str | Path) -> pd.DataFrame:
    """Report the ``ligand_types`` of every sub-directory's ``r.gpf``.

    Upgrades ``others/Docking/gpfreader.py``: one row per immediate
    sub-directory, with the ``"not found"`` sentinel for folders without
    a GPF (the original printed a blocked-sign emoji; pharmaml keeps
    reports ASCII).  Inline ``#`` comments are stripped from the value,
    and so is an optional ``=`` separator after the keyword -- foreign
    GPF writers emit ``ligand_types = C HD`` while pharmaml's own
    :func:`write_gpf` uses ``ligand_types C HD``.

    Returns
    -------
    pandas.DataFrame
        Columns ``folder`` and ``ligand_types``.

    Raises
    ------
    DockingError
        If ``directory`` is not a directory.
    """
    base = Path(directory)
    if not base.is_dir():
        raise DockingError(f"directory not found: {base}")
    rows: list[dict[str, str]] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        ligand_types = "not found"
        gpf = sub / "r.gpf"
        if gpf.is_file():
            for line in gpf.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                stripped = line.split("#", 1)[0].strip()
                if stripped.startswith("ligand_types"):
                    ligand_types = (
                        stripped.replace("ligand_types", "", 1).strip().removeprefix("=").strip()
                    )
                    break
        rows.append({"folder": sub.name, "ligand_types": ligand_types})
    _LOG.info("Scanned %d sub-directory(ies) of %s for r.gpf files", len(rows), base)
    return pd.DataFrame(rows, columns=["folder", "ligand_types"])

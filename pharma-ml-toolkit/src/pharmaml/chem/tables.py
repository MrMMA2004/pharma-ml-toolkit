"""Ligand-table reconciliation, SMILES annotation and table merging.

Upgrades five rar-era utilities that shuffled ligand spreadsheets around
with hard-coded filenames:

* ``others/compare/notcommon.py`` -- set *differences* of a ligand column
  between two tables (with ``.pdbqt`` extension stripping),
* ``others/General/common_between_two_excel.py`` -- the *intersection*
  counterpart (Persian-commented Excel workflow) that the original
  ``notcommon.py`` never computed,
* ``others/compare/smilesgetter.py`` -- annotate a differences table with
  SMILES looked up from a reference ``Ligand``/``SMILES`` library,
* ``others/General/CSV_merger.py`` -- row-wise concat with a
  "named columns first, remainder alphabetical" reordering,
* ``others/General/filemerger.py`` -- merge every CSV of a folder while
  skipping Excel ``~$`` lock files.

Design changes over the originals: every function accepts DataFrames
*or* csv/xlsx paths (dispatched through :func:`pharmaml.utils.io.read_table`),
comparison results come back as a typed :class:`TableComparison` instead
of being written to a fixed ``differences_output.csv``, and callers'
frames are never mutated in place.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.chem.tables import annotate_with_smiles, compare_tables
>>> left = pd.DataFrame({"Ligand": ["A.pdbqt", "b.pdbqt", "c"]})
>>> right = pd.DataFrame({"Ligand": ["B", "c", "d"]})
>>> comparison = compare_tables(left, right, column="Ligand")
>>> comparison.only_in_a          # names normalised: case + extension
['a']
>>> sorted(comparison.common)
['b', 'c']
>>> library = pd.DataFrame({"Ligand": ["a.pdbqt", "b"], "SMILES": ["CCO", "c1ccccc1"]})
>>> annotated = annotate_with_smiles(
...     pd.DataFrame({"Ligand": ["A.pdbqt", "z"]}), library, name_columns=["Ligand"])
>>> annotated["Ligand_SMILES"].tolist()
['CCO', '']
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table, write_table

_LOG = get_logger(__name__)

__all__ = [
    "TableComparison",
    "annotate_with_smiles",
    "compare_tables",
    "merge_table_folder",
    "merge_tables",
]

#: What a frame-or-path argument may be.
TableSource: TypeAlias = pd.DataFrame | str | Path

#: Default artefact name written by :func:`merge_table_folder`
#: (mirrors the always-write behaviour of the original ``filemerger.py``).
_MERGED_OUTPUT_NAME = "merged_output.csv"


def _resolve_table(data: TableSource, label: str) -> tuple[pd.DataFrame, str]:
    """Coerce a DataFrame or csv/xlsx path to a DataFrame.

    Returns the frame plus a human-readable source label used in error
    messages (the path, or ``label`` for in-memory frames).
    """
    if isinstance(data, pd.DataFrame):
        return data, label
    if isinstance(data, (str, Path)):
        return read_table(data), str(data)
    raise DataValidationError(
        f"{label} must be a pandas DataFrame or a path to a csv/xlsx file, "
        f"got {type(data).__name__}"
    )


def _normalize_name(raw: object, strip_extension: bool, lowercase: bool) -> str:
    """Normalise one ligand name exactly like the rar scripts.

    Whitespace is always stripped; the file extension (``.pdbqt`` etc.)
    and letter case are removed when the corresponding flags are set.
    Any missing-value sentinel (``None``, ``NaN``, ``NA``, ``NaT``)
    collapses to the empty string.
    """
    if _is_missing(raw):
        return ""
    text = str(raw).strip()
    if strip_extension:
        text = os.path.splitext(text)[0]
    if lowercase:
        text = text.lower()
    return text


def _is_missing(raw: object) -> bool:
    """True for ``None`` and any pandas missing-value sentinel.

    Covers ``float('nan')``, ``pd.NA`` (nullable string columns) and
    ``pd.NaT`` (datetime columns) without leaking their ``"<NA>"`` /
    ``"NaT"`` string reprs into comparisons.
    """
    if raw is None:
        return True
    try:
        return bool(pd.isna(raw))
    except (TypeError, ValueError):  # arrays/lists are not scalars
        return False


@dataclass(frozen=True, slots=True)
class TableComparison:
    """Set reconciliation of one ligand column across two tables.

    Attributes
    ----------
    column:
        Name of the compared column.
    common:
        Sorted values present in both tables (the ``common_between_two_
        excel.py`` result).
    only_in_a:
        Sorted values exclusive to the first table (``notcommon.py``'s
        first output column).
    only_in_b:
        Sorted values exclusive to the second table.
    """

    column: str
    common: list[str]
    only_in_a: list[str]
    only_in_b: list[str]

    def summary(self) -> dict[str, int]:
        """Row counts per bucket (the numbers the legacy scripts printed)."""
        return {
            "common": len(self.common),
            "only_in_a": len(self.only_in_a),
            "only_in_b": len(self.only_in_b),
        }


def compare_tables(
    table_a: TableSource,
    table_b: TableSource,
    column: str,
    normalize: bool = True,
    strip_extension: bool = True,
    lowercase: bool = True,
) -> TableComparison:
    """Compare one ligand-name column across two tables.

    Upgrades ``notcommon.py`` (differences) and ``common_between_two_
    excel.py`` (intersection) into a single set reconciliation that
    reports *both* sides plus the overlap.

    Parameters
    ----------
    table_a, table_b:
        DataFrames or paths to csv/xlsx files (read with
        :func:`pharmaml.utils.io.read_table`).
    column:
        Ligand-name column present in both tables.
    normalize:
        When ``True`` (default) names are whitespace-stripped, file
        extensions (``.pdbqt``, ...) removed and letter case folded
        before comparing, so ``LIGAND1.pdbqt`` equals ``ligand1``.
        The two flags below fine-tune which steps run.  Names that
        normalise to the empty string (blank or whitespace-only cells)
        are dropped from the comparison instead of being reported as a
        phantom empty ligand shared by both tables.
    strip_extension:
        Strip the file extension during normalization (default ``True``).
    lowercase:
        Fold names to lowercase during normalization (default ``True``).

    Returns
    -------
    TableComparison
        Sorted ``common`` / ``only_in_a`` / ``only_in_b`` value lists.

    Raises
    ------
    DataValidationError
        If ``column`` is missing from either table (the message names
        the offending table and its available columns), or an input is
        neither a DataFrame nor a readable path.
    """
    frame_a, label_a = _resolve_table(table_a, "table_a")
    frame_b, label_b = _resolve_table(table_b, "table_b")

    for frame, source_label, ordinal in (
        (frame_a, label_a, "table_a"),
        (frame_b, label_b, "table_b"),
    ):
        if column not in frame.columns:
            raise DataValidationError(
                f"Column {column!r} not found in {ordinal} (source: {source_label}); "
                f"available columns: {list(frame.columns)}"
            )

    def _values(frame: pd.DataFrame) -> set[str]:
        series = frame[column].dropna().astype(str)
        if not normalize:
            return set(series)
        normalized = {
            _normalize_name(value, strip_extension=strip_extension, lowercase=lowercase)
            for value in series
        }
        # Blank cells (whitespace-only, or empty strings) normalise to ""
        # -- they are not ligand names and must not surface as a phantom
        # "" ligand shared by both tables.  A bare ".pdbqt" cell keeps
        # splitext dotfile semantics (it stays a literal name, matching
        # the rar-era scripts).
        normalized.discard("")
        return normalized

    set_a = _values(frame_a)
    set_b = _values(frame_b)
    comparison = TableComparison(
        column=column,
        common=sorted(set_a & set_b),
        only_in_a=sorted(set_a - set_b),
        only_in_b=sorted(set_b - set_a),
    )
    _LOG.info(
        "Compared column %r: %d common, %d only in %s, %d only in %s",
        column,
        len(comparison.common),
        len(comparison.only_in_a),
        label_a,
        len(comparison.only_in_b),
        label_b,
    )
    return comparison


def annotate_with_smiles(
    frame: pd.DataFrame,
    reference: TableSource,
    name_columns: list[str],
    reference_name_column: str = "Ligand",
    reference_smiles_column: str = "SMILES",
    suffix: str = "_SMILES",
) -> pd.DataFrame:
    """Annotate ligand-name columns with SMILES from a reference table.

    Upgrades ``smilesgetter.py``: for every column in ``name_columns``
    a ``<column><suffix>`` column is appended holding the SMILES found
    by normalised name lookup in ``reference`` (empty string when the
    name is unknown -- the original behaviour).

    Parameters
    ----------
    frame:
        Table whose ligand-name columns get annotated.
    reference:
        DataFrame or csv/xlsx path holding the lookup columns.
    name_columns:
        Columns of ``frame`` containing ligand names (missing columns
        raise :class:`DataValidationError` listing all of them).
    reference_name_column:
        Ligand-name column in the reference table (default ``'Ligand'``).
    reference_smiles_column:
        SMILES column in the reference table (default ``'SMILES'``).
    suffix:
        Suffix appended to each annotated column name.

    Returns
    -------
    pandas.DataFrame
        A *new* frame; the input is never mutated.

    Raises
    ------
    DataValidationError
        If the reference lacks its name/SMILES columns, or ``frame``
        lacks one of ``name_columns``.
    """
    ref_frame, ref_label = _resolve_table(reference, "reference")
    for ref_col in (reference_name_column, reference_smiles_column):
        if ref_col not in ref_frame.columns:
            raise DataValidationError(
                f"Reference table (source: {ref_label}) lacks the {ref_col!r} "
                f"column; available columns: {list(ref_frame.columns)}"
            )
    missing = [col for col in name_columns if col not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Input frame lacks the name column(s) {missing}; "
            f"available columns: {list(frame.columns)}"
        )

    lookup: dict[str, str] = {}
    for raw_name, raw_smiles in zip(
        ref_frame[reference_name_column], ref_frame[reference_smiles_column]
    ):
        key = _normalize_name(raw_name, strip_extension=True, lowercase=True)
        if not key:
            continue
        smiles = "" if _is_missing(raw_smiles) else str(raw_smiles).strip()
        lookup[key] = smiles

    result = frame.copy()
    for col in name_columns:
        result[f"{col}{suffix}"] = [
            lookup.get(
                _normalize_name(value, strip_extension=True, lowercase=True), ""
            )
            for value in frame[col]
        ]
    _LOG.info(
        "Annotated %d column(s) with SMILES from %d reference entries",
        len(name_columns),
        len(lookup),
    )
    return result


def merge_tables(
    frames: list[TableSource],
    first_columns: list[str] | None = None,
    sort_remaining: bool = True,
) -> pd.DataFrame:
    """Row-wise merge of several tables, ignoring the index.

    Upgrades ``CSV_merger.py``: the tables (DataFrames or csv/xlsx
    paths) are concatenated vertically, and when ``first_columns`` is
    given the merged columns are reordered exactly like the original
    script -- the named columns first (those that exist), the remaining
    columns alphabetically sorted.

    Parameters
    ----------
    frames:
        Tables to concatenate (at least one).
    first_columns:
        Column names to move to the front, in order.  Names that do not
        exist in the merged output are skipped with a warning (the
        original crashed with ``KeyError``).
    sort_remaining:
        Sort the remaining columns alphabetically (default, the
        ``CSV_merger.py`` behaviour); ``False`` keeps their first-seen
        concatenation order.

    Returns
    -------
    pandas.DataFrame
        A new frame; single-table inputs return a copy, never a view.

    Raises
    ------
    DataValidationError
        If ``frames`` is empty or an element is neither a DataFrame nor
        a readable path.
    """
    if not frames:
        raise DataValidationError("merge_tables requires at least one input table")
    resolved = [
        _resolve_table(item, f"frames[{i}]")[0] for i, item in enumerate(frames)
    ]

    if len(resolved) == 1:
        merged = resolved[0].copy()
    else:
        merged = pd.concat(resolved, axis=0, ignore_index=True, sort=False)

    if first_columns:
        present = [col for col in first_columns if col in merged.columns]
        absent = [col for col in first_columns if col not in merged.columns]
        if absent:
            _LOG.warning("first_columns not found in merged table (skipped): %s", absent)
        remaining = [col for col in merged.columns if col not in present]
        if sort_remaining:
            remaining = sorted(remaining)
        merged = merged[present + remaining]

    _LOG.info("Merged %d table(s) into %d rows x %d columns",
              len(frames), len(merged), merged.shape[1])
    return merged


def merge_table_folder(
    directory: str | Path,
    extension: str = ".csv",
    out_path: str | Path | None = None,
    first_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Merge every table of one extension directly inside ``directory``.

    Upgrades ``filemerger.py``: all files with ``extension`` in the
    folder (non-recursive) are merged via :func:`merge_tables` and the
    result is *always* written out -- to ``out_path`` or, by default,
    ``merged_output.csv`` inside the same folder -- mirroring the
    original script's always-write behaviour.

    Temporaries are skipped exactly like the rar script: files starting
    with ``~$`` (Excel lock files) or ``.`` (hidden/partial files).
    The output file itself is also excluded from the scan so repeated
    runs never re-ingest their own result.

    Parameters
    ----------
    directory:
        Folder to scan (not descended into recursively).
    extension:
        File extension to collect, default ``'.csv'``.
    out_path:
        Where the merged table is written; defaults to
        ``<directory>/merged_output.csv``.
    first_columns:
        Passed through to :func:`merge_tables`.

    Returns
    -------
    pandas.DataFrame
        The merged (and written) table.

    Raises
    ------
    DataValidationError
        If ``directory`` is not a directory or contains no matching
        files.
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise DataValidationError(f"merge_table_folder: not a directory: {folder}")

    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    target = Path(out_path) if out_path is not None else folder / _MERGED_OUTPUT_NAME

    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.name.lower().endswith(ext)
        and not path.name.startswith(("~$", "."))
        and path.resolve() != target.resolve()
    )
    if not files:
        raise DataValidationError(
            f"No {ext} files found in {folder} (temp/lock files and the merged "
            f"output are excluded from the scan)"
        )

    merged = merge_tables([str(path) for path in files], first_columns=first_columns)
    written = write_table(merged, target)
    _LOG.info("Merged %d file(s) from %s into %s", len(files), folder, written)
    return merged

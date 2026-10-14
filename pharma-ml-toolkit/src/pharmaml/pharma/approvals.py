"""Regulatory-approval filtering over drug interaction tables.

Upgrades ``others/General/approved.py``: the Persian-commented original
looped over 17 DGIdb TSV exports (``1.tsv`` ... ``17.tsv``), normalised
column names to lowercase/stripped, kept the rows whose ``"regulatory
approval"`` column equalled ``"Approved"`` and dumped the drug names
into a Word document.

The production version:

* accepts a DataFrame, a single csv/tsv path, or a list of paths
  (the 17-file loop becomes one call),
* matches column names case-insensitively and whitespace-tolerantly on
  a *copy* so the caller's frame is never mutated,
* returns sorted unique drug names as ``list[str]`` and optionally
  writes a single-column CSV instead of the ``.docx`` artefact (no
  python-docx dependency),
* raises :class:`~pharmaml.exceptions.DataValidationError` naming the
  file/frame that lacks the required columns instead of printing a
  warning and silently skipping it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.io import write_table

_LOG = get_logger(__name__)

__all__ = ["filter_approved_drugs"]

#: What the ``data`` argument of :func:`filter_approved_drugs` may be.
ApprovalSource = Union[pd.DataFrame, str, Path, list[Union[str, Path]]]

#: Extensions read with a tab separator (everything else is comma).
_TAB_EXTENSIONS = {".tsv", ".tab"}


def _read_approval_table(path: Path) -> pd.DataFrame:
    """Read one csv/tsv export, choosing the separator from the extension."""
    separator = "\t" if path.suffix.lower() in _TAB_EXTENSIONS else ","
    try:
        return pd.read_csv(path, sep=separator)
    except Exception as exc:  # pandas raises a wide variety of errors
        raise DataValidationError(f"Failed to parse '{path}': {exc}") from exc


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase, stripped column names."""
    copy = frame.copy()
    copy.columns = [str(col).strip().lower() for col in copy.columns]
    return copy


def _is_missing(raw: object) -> bool:
    """True for ``None`` and any pandas missing-value sentinel.

    Covers ``float('nan')``, ``pd.NA`` (nullable string columns) and
    ``pd.NaT`` -- without this, their ``"<NA>"`` / ``"NaT"`` string
    reprs would be collected as drug names.
    """
    if raw is None:
        return True
    try:
        return bool(pd.isna(raw))
    except (TypeError, ValueError):  # arrays/lists are not scalars
        return False


def filter_approved_drugs(
    data: ApprovalSource,
    drug_column: str = "drug",
    approval_column: str = "regulatory approval",
    approval_status: str = "approved",
    out_path: str | Path | None = None,
) -> list[str]:
    """Collect the drug names whose regulatory approval matches a status.

    Parameters
    ----------
    data:
        A :class:`pandas.DataFrame`, a single path to a csv/tsv file,
        or a list of such paths (the original script's ``1.tsv`` ...
        ``17.tsv`` batch).  ``.tsv``/``.tab`` files are read with a tab
        separator, everything else with commas.
    drug_column:
        Drug-name column (matched case-insensitively, ignoring
        surrounding whitespace; default ``'drug'``).
    approval_column:
        Regulatory-approval column (same matching rules; default
        ``'regulatory approval'`` -- the DGIdb export header).
    approval_status:
        Status to keep, compared after stripping and lowercasing both
        sides (default ``'approved'``).
    out_path:
        When given, the sorted unique names are also written as a
        single-column CSV with the header ``drug``.

    Returns
    -------
    list[str]
        Sorted unique drug names (stripped).

    Raises
    ------
    DataValidationError
        If a table lacks the drug or approval column (the message names
        the offending file/frame), a list element is not a path, or
        ``data`` has an unsupported type.
    """
    drug_key = drug_column.strip().lower()
    approval_key = approval_column.strip().lower()
    status = approval_status.strip().lower()

    sources: list[tuple[pd.DataFrame, str]] = []
    if isinstance(data, pd.DataFrame):
        sources.append((data, "input DataFrame"))
    elif isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file():
            raise DataValidationError(f"Input file not found: {path}")
        sources.append((_read_approval_table(path), str(path)))
    elif isinstance(data, list):
        if not data:
            raise DataValidationError("filter_approved_drugs got an empty list of paths")
        for item in data:
            if not isinstance(item, (str, Path)):
                raise DataValidationError(
                    "List inputs must contain paths (str or Path), got "
                    f"{type(item).__name__}: {item!r}"
                )
            path = Path(item)
            if not path.is_file():
                raise DataValidationError(f"Input file not found: {path}")
            sources.append((_read_approval_table(path), str(path)))
    else:
        raise DataValidationError(
            "data must be a DataFrame, a path, or a list of paths, got "
            f"{type(data).__name__}"
        )

    names: set[str] = set()
    for raw_frame, label in sources:
        frame = _normalized_columns(raw_frame)
        missing = [col for col in (drug_key, approval_key) if col not in frame.columns]
        if missing:
            raise DataValidationError(
                f"Table {label!r} lacks the column(s) {missing}; "
                f"available columns: {list(frame.columns)}"
            )
        approved = frame[
            frame[approval_key].fillna("").astype(str).str.strip().str.lower() == status
        ]
        for value in approved[drug_key]:
            if _is_missing(value):
                continue
            names.add(str(value).strip())

    result = sorted(names)
    if out_path is not None:
        write_table(pd.DataFrame({"drug": result}), out_path)
    _LOG.info(
        "Collected %d unique %s drug names from %d table(s)",
        len(result),
        approval_status,
        len(sources),
    )
    return result

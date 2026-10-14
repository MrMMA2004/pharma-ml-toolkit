"""Robust tabular and filesystem I/O.

The original scripts called ``pd.read_csv``/``pd.read_excel`` directly
with hardcoded Windows paths scattered across ~30 files.  This module
centralises all persistence so filename handling, extension dispatch
and error reporting live in exactly one place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "read_table",
    "write_table",
    "ensure_directory",
    "safe_filename",
]

_LOG = get_logger(__name__)

_CSV_LIKE = {".csv", ".tsv", ".txt"}
_EXCEL_LIKE = {".xlsx", ".xls", ".xlsm"}


def _read_csv_sniffing(file_path: Path) -> pd.DataFrame:
    """Read a delimited text file, sniffing the separator when needed.

    ``pd.read_csv`` defaults to comma and only raises ``ParserError``
    on *malformed* files — a purely semicolon- or tab-delimited file
    silently parses as one giant column (common with European-locale
    Excel exports).  When that happens, re-read with the separator
    found inside the mangled header.
    """
    try:
        frame = pd.read_csv(file_path)
    except pd.errors.ParserError:
        frame = pd.read_csv(file_path, sep=";")
    if frame.shape[1] == 1:
        header = str(frame.columns[0])
        for sep in (";", "\t", "|"):
            if sep in header:
                try:
                    return pd.read_csv(file_path, sep=sep)
                except pd.errors.ParserError:
                    continue
    return frame


def read_table(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read a CSV/TSV or Excel file into a DataFrame.

    Parameters
    ----------
    path:
        Input file path.  The reader is chosen by extension; CSV files
        are sniffed for their delimiter (comma/semicolon/tab/pipe).
    sheet_name:
        Excel sheet to read (ignored for CSV-like inputs).

    Returns
    -------
    pandas.DataFrame
        Parsed table with stripped column names (the original files
        frequently carried trailing spaces in headers).

    Raises
    ------
    DataValidationError
        If the file does not exist or has an unsupported extension.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataValidationError(f"Input file not found: {file_path}")

    suffix = file_path.suffix.lower()
    try:
        if suffix in _CSV_LIKE:
            frame = _read_csv_sniffing(file_path)
        elif suffix in _EXCEL_LIKE:
            frame = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise DataValidationError(
                f"Unsupported input format '{suffix}' for {file_path}. "
                f"Supported: {sorted(_CSV_LIKE | _EXCEL_LIKE)}"
            )
    except DataValidationError:
        raise
    except Exception as exc:  # pandas raises a wide variety of errors
        raise DataValidationError(f"Failed to parse '{file_path}': {exc}") from exc

    frame.columns = [str(col).strip() for col in frame.columns]
    _LOG.debug("Read %d rows x %d columns from %s", len(frame), frame.shape[1], file_path)
    return frame


def write_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV or Excel honouring the extension.

    Parent directories are created automatically.  Returns the resolved
    path of the written artifact.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()
    if suffix in _CSV_LIKE:
        frame.to_csv(file_path, index=index)
    elif suffix in _EXCEL_LIKE:
        frame.to_excel(file_path, index=index)
    else:
        # Default to CSV for unknown extensions instead of crashing.
        _LOG.warning("Unknown extension '%s'; writing as CSV.", suffix)
        file_path = file_path.with_suffix(".csv")
        frame.to_csv(file_path, index=index)

    _LOG.info("Wrote %d rows to %s", len(frame), file_path)
    return file_path


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing; return the resolved Path."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, replacement: str = "_") -> str:
    """Sanitise a string into a portable filename.

    Replaces whitespace, unicode and reserved characters with
    ``replacement`` and collapses repeats, so artefacts created from
    ligand names or SMILES remain valid across operating systems.
    """
    cleaned = _UNSAFE_CHARS.sub(replacement, name.strip())
    cleaned = replacement.join(part for part in cleaned.split(replacement) if part)
    return cleaned[:200] or "unnamed"

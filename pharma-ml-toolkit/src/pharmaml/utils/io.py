"""Robust tabular and filesystem I/O.

The original scripts called ``pd.read_csv``/``pd.read_excel`` directly
with hardcoded Windows paths scattered across ~30 files.  This module
centralises all persistence so filename handling, extension dispatch
and error reporting live in exactly one place.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.utils.io import read_table, write_table
>>> frame = pd.DataFrame({"ligand": ["aspirin", "ibuprofen"], "ic50": [3.4, 5.6]})
>>> out = write_table(frame, "/tmp/pharmaml_doctest_round.csv")
>>> out
PosixPath('/tmp/pharmaml_doctest_round.csv')
>>> back = read_table(out)
>>> back["ligand"].tolist(), back["ic50"].tolist()
(['aspirin', 'ibuprofen'], [3.4, 5.6])
>>> out.unlink()

>>> from pathlib import Path
>>> from pharmaml.utils.io import read_table
>>> semi = Path("/tmp/pharmaml_doctest_semi.csv")
>>> _ = semi.write_text("ligand;pic50\\naspirin;3.2\\nibuprofen;4.1\\n")
>>> frame = read_table(semi)     # semicolon sniffed, not one mangled column
>>> frame.shape, list(frame.columns)
((2, 2), ['ligand', 'pic50'])
>>> frame["pic50"].tolist()
[3.2, 4.1]
>>> semi.unlink()
>>> read_table(semi)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Input file not found: /tmp/pharmaml_doctest_semi.csv

>>> from pharmaml.utils.io import safe_filename
>>> safe_filename("Ligand 1: active/v2")   # unsafe runs collapse to one "_"
'Ligand_1_active_v2'
>>> safe_filename("CON.csv")               # Windows reserved device name
'_CON.csv'
>>> safe_filename("...")                   # dots-only -> fallback
'unnamed'
>>> safe_filename("ligand (1).sdf")
'ligand_1_.sdf'

>>> import shutil
>>> from pharmaml.utils.io import ensure_directory, write_table
>>> frame = pd.DataFrame({"smiles": ["CCO"]})
>>> out = write_table(frame, "/tmp/pharmaml_doctest_profile.dat")
>>> out                              # unknown extension falls back to CSV
PosixPath('/tmp/pharmaml_doctest_profile.csv')
>>> out.unlink()
>>> runs = ensure_directory("/tmp/pharmaml_doctest_nested/runs/lig1")
>>> runs.is_dir()
True
>>> shutil.rmtree("/tmp/pharmaml_doctest_nested")

>>> from pharmaml.utils.io import read_table_with_progress
>>> prog = Path("/tmp/pharmaml_doctest_progress.csv")
>>> _ = prog.write_text("ligand,pic50\\n" + "".join(f"L{i:03d},{i:03d}.5\\n" for i in range(40)))
>>> seen = []
>>> frame = read_table_with_progress(
...     prog, lambda done, total, note: seen.append((done, total, note)),
...     chunk_rows=10, size_threshold=1,
... )
>>> len(frame), frame["pic50"].tolist()[:3]
(40, [0.5, 1.5, 2.5])
>>> seen[0], seen[-1][0]
((10, 41, ''), 40)
>>> tiny = Path("/tmp/pharmaml_doctest_progress_tiny.csv")
>>> _ = tiny.write_text("ligand,pic50\\naspirin,3.2\\n")
>>> seen.clear()
>>> frame = read_table_with_progress(tiny, seen.append)   # under the threshold
>>> len(frame), seen      # the plain reader -- no reports at all
(1, [])
>>> prog.unlink(); tiny.unlink()
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "ensure_directory",
    "read_table",
    "read_table_with_progress",
    "safe_filename",
    "write_table",
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


def _finalise_columns(frame: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """Strip header whitespace and refuse duplicated names (shared rule).

    Both readers (:func:`read_table` and :func:`read_table_with_progress`)
    funnel through here, so the chunked path can never drift from the
    plain one on column hygiene.
    """
    frame.columns = [str(col).strip() for col in frame.columns]
    duplicated = sorted(
        {col for col in frame.columns if list(frame.columns).count(col) > 1}
    )
    if duplicated:
        # Two headers that differed only in surrounding whitespace ("Ligand"
        # vs "Ligand ") collapse here; pandas would then hand every
        # column access a DataFrame instead of a Series and fail far away
        # from the real cause.
        raise DataValidationError(
            f"Duplicate column name(s) after stripping whitespace in {file_path}: "
            f"{duplicated} -- rename the conflicting header(s) in the source file"
        )
    return frame


def _sniff_separator(file_path: Path) -> str:
    """The separator a comma read betrays in a mangled single-column header.

    Mirrors the retry ladder of :func:`_read_csv_sniffing` -- a purely
    semicolon/tab/pipe-delimited file reads as ONE mangled column whose
    header carries the separator, so the chunked reader can start with
    the right separator without reading the whole file twice.  A header
    that is itself comma-malformed sniffs to ``";"`` (the sync reader's
    own first fallback).
    """
    try:
        head = pd.read_csv(file_path, nrows=0)
    except pd.errors.ParserError:
        return ";"
    if head.shape[1] == 1:
        header = str(head.columns[0])
        for sep in (";", "\t", "|"):
            if sep in header:
                return sep
    return ","


def _estimate_csv_rows(file_path: Path, sample_bytes: int = 64 * 1024) -> int | None:
    """Estimate a CSV's row count from its own bytes-per-line average.

    Samples the first ``sample_bytes``, counts the newlines and scales
    the file size by the sampled average -- an honest estimate for a
    determinate progress bar (the last chunk may overshoot it; the
    reader clamps the fraction).  ``None`` when the sample carries no
    newline at all (a single giant line) -- callers then report
    open-ended counts instead of a fake total.
    """
    total = file_path.stat().st_size
    with file_path.open("rb") as handle:
        sample = handle.read(sample_bytes)
    newlines = sample.count(b"\n")
    if newlines == 0:
        return None
    per_line = len(sample) / newlines
    if per_line <= 0:  # pragma: no cover - impossible for a newline>0 sample
        return None
    return max(round(total / per_line), 1)


def read_table_with_progress(
    path: str | Path,
    report: Callable[[int, int | None, str], None],
    *,
    chunk_rows: int = 50_000,
    size_threshold: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Read a table, reporting row progress for large CSV-like files.

    Small files (below ``size_threshold`` bytes) and Excel inputs go
    straight through :func:`read_table` -- ``report`` never fires and
    the behaviour is byte-for-byte the plain reader's.  CSV-like files
    at/above the threshold are read in ``chunk_rows`` chunks with
    ``report(rows_done, rows_estimated, "")`` pushed after every chunk,
    so a GUI can carry a determinate bar with an ETA while a
    multi-hundred-megabyte table streams in.  The estimated total comes
    from the file's own bytes-per-line average (see
    :func:`_estimate_csv_rows`) -- an estimate that converges as chunks
    land, never a lie: the final chunk may overshoot and the caller's
    fraction simply clamps.

    Parameters
    ----------
    path:
        Input file path (same extensions as :func:`read_table`).
    report:
        Progress callback -- ``(done_rows, estimated_rows_or_None,
        note)`` per chunk, the exact shape the playground's status-bar
        reporter already speaks.
    chunk_rows:
        Rows per chunk on the progressive path.
    size_threshold:
        Files at/above this many bytes read progressively; smaller ones
        take the plain reader (they are fast enough that reports would
        be noise).

    Returns
    -------
    pandas.DataFrame
        The same parsed, column-hygiene-checked table
        :func:`read_table` returns (a parity the tests pin).

    Raises
    ------
    DataValidationError
        If the file does not exist, has an unsupported extension, or
        fails to parse -- the plain reader's own errors, unchanged.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataValidationError(f"Input file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in _CSV_LIKE or file_path.stat().st_size < size_threshold:
        # small file or Excel: the plain reader is fast enough -- no
        # progress reports, no behavioural difference at all
        return read_table(file_path)
    chunk_rows = max(int(chunk_rows), 1)
    separator = _sniff_separator(file_path)
    estimated = _estimate_csv_rows(file_path)
    chunks: list[pd.DataFrame] = []
    done = 0

    def _drain(sep: str) -> None:
        nonlocal done
        with pd.read_csv(file_path, sep=sep, chunksize=chunk_rows) as reader:
            for chunk in reader:
                chunks.append(chunk)
                done += len(chunk)
                if len(chunk):
                    # the header-only file yields one EMPTY chunk -- a
                    # 0-rows "progress" blip would be noise, not news
                    report(done, estimated, "")

    try:
        try:
            _drain(separator)
        except pd.errors.ParserError:
            if separator == ";":
                # the sniffed separator itself chokes on a ragged body:
                # delegate to the sync reader, whose ladder ends by
                # returning its OWN verdict -- a mangled comma frame
                # for this very file (never a different answer than
                # read_table gives the same bytes)
                return read_table(file_path)
            # the sync reader's own fallback: a comma-malformed body
            # re-reads semicolon-delimited from the top
            chunks.clear()
            done = 0
            try:
                _drain(";")
            except pd.errors.ParserError:
                # BOTH separators choke: the sync ladder's verdict is
                # the honest one (a clean DataValidationError for a
                # truly malformed file)
                return read_table(file_path)
    except DataValidationError:
        raise
    except Exception as exc:  # pandas raises a wide variety of errors
        raise DataValidationError(f"Failed to parse '{file_path}': {exc}") from exc
    if not chunks:
        # defensive floor: a reader that yields no chunks at all (pandas
        # today always yields one, even for a header-only file) -- the
        # plain reader's verdict is the honest one
        return read_table(file_path)
    frame = pd.concat(chunks, ignore_index=True)
    _LOG.debug(
        "Progressively read %d rows x %d columns from %s",
        len(frame),
        frame.shape[1],
        file_path,
    )
    return _finalise_columns(frame, file_path)


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

    frame = _finalise_columns(frame, file_path)
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

#: Windows reserved device names -- a file named e.g. ``CON.csv`` cannot be
#: created on Windows (and is a nuisance in several archives/cloud drives).
#: Map every reserved bare name to a safe prefixed variant.
_RESERVED_WINDOWS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename(name: str, replacement: str = "_") -> str:
    """Sanitise a string into a portable filename.

    Replaces whitespace, unicode and reserved characters with
    ``replacement`` and collapses repeats, so artefacts created from
    ligand names or SMILES remain valid across operating systems.

    Portability guards beyond the character whitelist:

    * Windows reserved device names (``CON``, ``PRN``, ``AUX``, ``NUL``,
      ``COM1``..``COM9``, ``LPT1``..``LPT9``) are prefixed with an
      underscore -- with or without an extension (``CON.csv`` counts);
    * trailing dots or spaces are stripped (Windows silently drops them,
      which breaks round-trips on other systems);
    * names consisting only of dots (``.``/``..``/``...``) fall back to
      ``unnamed``.
    """
    cleaned = _UNSAFE_CHARS.sub(replacement, name.strip())
    cleaned = replacement.join(part for part in cleaned.split(replacement) if part)
    cleaned = cleaned[:200].rstrip(". ")
    if len(replacement) == 1:
        cleaned = cleaned.rstrip(replacement)
    stem = cleaned.split(".", 1)[0].upper()
    if stem in _RESERVED_WINDOWS:
        cleaned = f"_{cleaned}"
    return cleaned or "unnamed"

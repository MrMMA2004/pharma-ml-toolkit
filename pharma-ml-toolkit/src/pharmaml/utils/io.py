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
>>> ragged = Path("/tmp/pharmaml_doctest_ragged.csv")    # 100x-145 parity line
>>> _ = ragged.write_text("a;b\\n1;2\\n3;4;5\\n")
>>> frame = read_table_with_progress(                     # sniffed ';' chokes...
...     ragged, lambda *args: None, chunk_rows=10, size_threshold=1)
>>> frame.shape, frame.columns.tolist()                  # ...delegate to sync
((2, 1), ['a;b'])
>>> prog.unlink(); tiny.unlink(); ragged.unlink()

>>> boundary = Path("/tmp/pharmaml_doctest_boundary.csv")   # 100x-146 guard line
>>> _ = boundary.write_text("a,b\\n1,2\\n3,4\\n5,6,7\\n8,9\\n")
>>> read_table_with_progress(                              # the ragged row lands
...     boundary, lambda *args: None, chunk_rows=2, size_threshold=1)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Failed to parse '/tmp/pharmaml_doctest_boundary.csv': Expected 2 fields in line 4, saw 3
>>> boundary.unlink()

>>> from pharmaml.utils.io import read_indexed_table_with_progress
>>> expr = Path("/tmp/pharmaml_doctest_indexed.csv")
>>> _ = expr.write_text(
...     "cell_line,g1,g2\\n" + "".join(f"CL{i:02d},{i},{i * 2}\\n" for i in range(6)))
>>> seen = []
>>> frame = read_indexed_table_with_progress(
...     expr, lambda done, total, note: seen.append((done, total, note)),
...     index_col="cell_line", chunk_rows=2, size_threshold=1,
... )
>>> list(frame.index), frame["g2"].tolist(), frame.index.name
(['CL00', 'CL01', 'CL02', 'CL03', 'CL04', 'CL05'], [0, 2, 4, 6, 8, 10], 'cell_line')
>>> seen[0], seen[-1][0]
((2, 7, ''), 6)
>>> renamed = Path("/tmp/pharmaml_doctest_indexed_renamed.csv")
>>> _ = renamed.write_text("cl,g\\nA,1\\nB,2\\nC,3\\nD,4\\n")
>>> frame = read_indexed_table_with_progress(            # no cell_line column...
...     renamed, lambda *a: None, index_col="cell_line", chunk_rows=2, size_threshold=1,
... )
>>> list(frame.index), frame.index.name                 # ...positional fallback
(['A', 'B', 'C', 'D'], 'cl')
>>> tiny = Path("/tmp/pharmaml_doctest_indexed_tiny.csv")
>>> _ = tiny.write_text("cell_line,g\\nA,1\\nB,2\\n")
>>> seen.clear()
>>> frame = read_indexed_table_with_progress(tiny, seen.append, index_col="cell_line")
>>> list(frame.index), seen                              # under the threshold
(['A', 'B'], [])
>>> boundary = Path("/tmp/pharmaml_doctest_indexed_boundary.csv")
>>> _ = boundary.write_text("cl,g\\nA,1\\nB,2\\nC,3,9\\nD,4\\n")
>>> read_indexed_table_with_progress(                    # ragged row lands on a
...     boundary, lambda *a: None, index_col="cl", chunk_rows=2, size_threshold=1,
... )                                                     # chunk boundary: rejected
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Failed to parse '/tmp/pharmaml_doctest_indexed_boundary.csv': Expected 2 fields in line 4, saw 3
>>> expr.unlink(); renamed.unlink(); tiny.unlink(); boundary.unlink()
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from pharmaml.exceptions import DataValidationError, JobCancelled
from pharmaml.logger import get_logger

__all__ = [
    "ensure_directory",
    "read_indexed_table_with_progress",
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


def _guard_ragged_boundaries(file_path: Path, separator: str, chunk_rows: int) -> None:
    """Reject ragged rows that land exactly on a chunk boundary (100x-146).

    pandas' low-memory chunked C parser silently TRUNCATES a row that
    carries MORE fields than the header and happens to be the first
    row of a chunk -- the plain reader raises ``ParserError`` for the
    same bytes, so without this guard a progressive read could hand
    back different DATA than the sync reader for the same file (the
    one parity hole the round-146 audit caught).  The guard scans the
    physical lines once (bytes only, no parsing) and raises the sync
    reader's own verdict for exactly those rows.

    The scan is only sound where physical lines equal records, so it
    bows out silently (no guard, pandas' own mid-chunk raises still
    apply) when the file carries any double quote -- which could open
    a multi-line quoted field -- or any interior blank line -- which
    ``skip_blank_lines`` drops, shifting every later boundary.  SHORT
    rows need no guard at all: both parsers pad them with ``NaN``.
    """
    sep_bytes = separator.encode("utf-8")
    with file_path.open("rb") as handle:
        header = handle.readline()
        if b'"' in header:
            return
        expected = header.count(sep_bytes) + 1
        pending_blank = False
        lineno = 0  # physical data lines, 1-based (the header excluded)
        for raw in handle:
            lineno += 1
            if not raw.strip():
                pending_blank = True
                continue
            if pending_blank:
                # a blank line with data after it: every later record
                # shifts off its physical line -- the guard bows out
                return
            pending_blank = False
            if b'"' in raw:
                return
            if (
                lineno > 1
                and (lineno - 1) % chunk_rows == 0
                and raw.count(sep_bytes) + 1 > expected
            ):
                raise DataValidationError(
                    f"Failed to parse '{file_path}': Expected {expected} fields "
                    f"in line {lineno + 1}, saw {raw.count(sep_bytes) + 1}"
                )


def read_table_with_progress(
    path: str | Path,
    report: Callable[[int, int | None, str], bool | None],
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

    Notes
    -----
    (100x-146) A ragged row carrying extra fields that lands exactly on
    a chunk boundary would be silently field-truncated by pandas' own
    chunked parser (the plain reader raises for the same bytes);
    :func:`_guard_ragged_boundaries` rejects such files up front so the
    parity promise holds.  Quote-bearing and interior-blank-line files
    skip the guard (their lines cannot be aligned to records); their
    non-boundary ragged rows still raise inside pandas itself, and
    short rows pad with ``NaN`` on both paths.
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
    _guard_ragged_boundaries(file_path, separator, chunk_rows)
    chunks: list[pd.DataFrame] = []
    done = 0

    def _drain(sep: str) -> None:
        nonlocal done
        with pd.read_csv(file_path, sep=sep, chunksize=chunk_rows) as reader:
            for chunk in reader:
                chunks.append(chunk)
                done += len(chunk)
                # the header-only file yields one EMPTY chunk -- a
                # 0-rows "progress" blip would be noise, not news.
                # (round 205) inside the guard: the cooperative stop --
                # the GUI's Stop button / memory watchdog answered False
                # -- the stream halts at this chunk boundary BEFORE the
                # concat would double the memory.
                if len(chunk) and report(done, estimated, "") is False:
                    raise JobCancelled(
                        f"table read cancelled after {done:,} rows -- "
                        "stopped at a checkpoint by the caller"
                    )

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
            _guard_ragged_boundaries(file_path, ";", chunk_rows)
            try:
                _drain(";")
            except pd.errors.ParserError:
                # BOTH separators choke: the sync ladder's verdict is
                # the honest one (a clean DataValidationError for a
                # truly malformed file)
                return read_table(file_path)
    except DataValidationError:
        raise
    except JobCancelled:
        # (round 205) the cooperative stop is the CALLER's verdict,
        # not a parse failure -- it must reach the playground's job
        # poller as itself, never wrapped into a DataValidationError.
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


def read_indexed_table_with_progress(
    path: str | Path,
    report: Callable[[int, int | None, str], bool | None],
    *,
    index_col: str | int,
    fallback_index_col: str | int = 0,
    chunk_rows: int = 50_000,
    size_threshold: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Read an INDEXED table, reporting row progress for large CSVs (100x-146).

    The expression-matrix loader's plain ``pd.read_csv(path,
    index_col=...)`` pair -- preferred label first, positional fallback
    when the label is missing -- with the determinate progress of
    :func:`read_table_with_progress`: CSV-like files at/above the
    threshold stream chunk-by-chunk with ``report(rows_done,
    rows_estimated, "")`` after every chunk, while smaller files take
    the plain pair (``report`` never fires, byte-for-byte parity with
    the synchronous loader).  The index is PRESERVED through the
    chunked concat (no ``ignore_index``), so ``frame.index`` carries
    the cell-line labels exactly as the plain reader hands them back.

    The label-missing fallback keeps its own parity: the ``ValueError``
    pandas raises for an absent label fires at the reader's FIRST chunk
    (before any data lands), and only that empty-chunks case re-drains
    with ``fallback_index_col`` -- a ``ValueError`` raised later (a
    report callback, a mid-file conversion) propagates for its honest
    verdict instead of silently re-reading the file.

    Parameters
    ----------
    path:
        Input file path.  CSV-like extensions take the progressive
        path; anything else (or a small file) takes the plain pair.
    report:
        Progress callback -- ``(done_rows, estimated_rows_or_None,
        note)`` per chunk, the status-bar reporter's own shape.
    index_col:
        The preferred index: a column label or a position.
    fallback_index_col:
        The index to fall back to when a LABEL is absent from the
        header (the plain reader's own ``except ValueError`` twin).
    chunk_rows:
        Rows per chunk on the progressive path.
    size_threshold:
        Files at/above this many bytes read progressively.

    Returns
    -------
    pandas.DataFrame
        The indexed table the plain ``pd.read_csv`` pair returns.

    Raises
    ------
    DataValidationError
        If the file does not exist, or the parse fails on the
        progressive path (the plain pair's own pandas errors, wrapped
        here for one consistent verdict type; the small-file path
        re-raises them raw, exactly as the synchronous loader sees
        them).

    Notes
    -----
    Ragged chunk-boundary rows are rejected up front by
    :func:`_guard_ragged_boundaries` with the same scope notes the
    sibling reader documents (quote-bearing and interior-blank-line
    files skip the guard).  No separator sniffing happens here -- the
    plain pair reads commas only, so the progressive path reads commas
    only; a semicolon-delimited table is one mangled column on BOTH
    paths, never a different answer below the threshold than above it.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataValidationError(f"Input file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in _EXCEL_LIKE:
        # (round 201) the Excel arm -- a workbook takes the plain
        # indexed pair through ``pd.read_excel`` (openpyxl engine),
        # keeping the label-missing fallback of the CSV twin: a sheet
        # without the preferred label index falls back to the first
        # column exactly like the CSV path does.  Workbooks are never
        # chunk-streamed (they load whole into memory by nature).
        try:
            return pd.read_excel(file_path, index_col=index_col)
        except ValueError:
            return pd.read_excel(file_path, index_col=fallback_index_col)
    if suffix not in _CSV_LIKE or file_path.stat().st_size < size_threshold:
        # small file (or non-CSV extension): the plain indexed pair,
        # label-missing fallback included -- byte-for-byte the sync twin
        try:
            return pd.read_csv(file_path, index_col=index_col)
        except ValueError:
            return pd.read_csv(file_path, index_col=fallback_index_col)
    chunk_rows = max(int(chunk_rows), 1)
    estimated = _estimate_csv_rows(file_path)
    _guard_ragged_boundaries(file_path, ",", chunk_rows)
    chunks: list[pd.DataFrame] = []
    done = 0

    def _drain(index: str | int) -> None:
        nonlocal done
        with pd.read_csv(
            file_path, sep=",", chunksize=chunk_rows, index_col=index
        ) as reader:
            for chunk in reader:
                chunks.append(chunk)
                done += len(chunk)
                # the header-only file yields one EMPTY chunk -- a
                # 0-rows "progress" blip would be noise, not news.
                # (round 205) inside the guard: the cooperative stop --
                # same contract as read_table_with_progress's chunk loop.
                if len(chunk) and report(done, estimated, "") is False:
                    raise JobCancelled(
                        f"indexed table read cancelled after {done:,} rows -- "
                        "stopped at a checkpoint by the caller"
                    )

    try:
        try:
            _drain(index_col)
        except ValueError:
            if chunks:
                # data already landed: this is NOT the label-missing
                # error (that one fires at the reader's first chunk) --
                # re-raise for the honest verdict instead of re-reading
                raise
            chunks.clear()
            done = 0
            _drain(fallback_index_col)
    except DataValidationError:
        raise
    except JobCancelled:
        # (round 205) same guard as read_table_with_progress: the stop
        # propagates as itself, never wrapped.
        raise
    except Exception as exc:  # pandas raises a wide variety of errors
        raise DataValidationError(f"Failed to parse '{file_path}': {exc}") from exc
    if not chunks:
        # defensive floor: a reader that yields no chunks at all (pandas
        # today always yields one) -- the plain pair's verdict is honest
        try:
            return pd.read_csv(file_path, index_col=index_col)
        except ValueError:
            return pd.read_csv(file_path, index_col=fallback_index_col)
    frame = pd.concat(chunks)
    _LOG.debug(
        "Progressively read %d indexed rows x %d columns from %s",
        len(frame),
        frame.shape[1],
        file_path,
    )
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

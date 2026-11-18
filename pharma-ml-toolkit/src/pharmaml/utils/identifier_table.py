"""Strict tabular reader preserving identifier text for split-aware workflows."""

from __future__ import annotations

from pathlib import Path


def read_identifier_table(path: Path, delimiter: str | None):
    """Read identifiers as text without inferring NA values or numeric types."""
    import csv

    import pandas as pd

    from pharmaml.exceptions import DataValidationError

    if not path.is_file():
        raise DataValidationError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    separators = {"comma": ",", "tab": "\t", "semicolon": ";"}
    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            sep = separators[delimiter] if delimiter else ("\t" if suffix == ".tsv" else ",")
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=sep, strict=True)
                header = next(reader, [])
                rows = []
                for row in reader:
                    if len(row) != len(header):
                        raise DataValidationError(
                            f"Cannot read split file {path}: inconsistent field count "
                            f"at line {reader.line_num}"
                        )
                    rows.append(row)
            frame = pd.DataFrame(rows, columns=header)
        elif suffix in {".xlsx", ".xlsm"}:
            first = pd.read_excel(path, header=None, nrows=1, dtype=str, keep_default_na=False)
            header = first.iloc[0].tolist() if len(first) else []
            frame = pd.read_excel(path, dtype=str, keep_default_na=False)
        else:
            raise DataValidationError(f"unsupported split file format: {path.suffix}")
    except DataValidationError:
        raise
    except (OSError, UnicodeError, ValueError, ImportError, StopIteration, csv.Error) as exc:
        raise DataValidationError(f"Cannot read split file {path}: {exc}") from exc
    if not header or any(not isinstance(name, str) or not name.strip() for name in header):
        raise DataValidationError(f"Split file {path} has an empty column header")
    if len(set(header)) != len(header):
        raise DataValidationError(f"Split file {path} has duplicate column headers")
    if len(header) != len(frame.columns):
        raise DataValidationError(f"Split file {path} has an inconsistent header")
    frame.columns = header
    return frame

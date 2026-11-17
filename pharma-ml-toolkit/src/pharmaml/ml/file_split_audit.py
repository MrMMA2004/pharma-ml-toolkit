"""Text-preserving, read-only exact identifier audits across partition files."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.split_audit import SplitOverlapReport, audit_split_overlap
from pharmaml.utils.identifier_table import read_identifier_table


def audit_partition_files(
    partitions: Mapping[str, str | Path],
    key_columns: Sequence[str],
    delimiter: str | None = None,
) -> SplitOverlapReport:
    """Read named files as literal text and audit exact single or composite keys.

    The input files must remain stable during reading. A clean audit does not
    establish independence or detect related compounds. Shared identifiers in
    the returned report may be sensitive; callers should avoid exposing them
    by default. No files are written.
    """
    if len(partitions) < 2:
        raise DataValidationError("Select at least two named partition files")
    if (not key_columns or any(not isinstance(key, str) or not key.strip() for key in key_columns)
            or len(set(key_columns)) != len(key_columns)):
        raise DataValidationError("Key columns must be nonempty and unique")
    if delimiter not in (None, "comma", "tab", "semicolon"):
        raise DataValidationError("Unknown delimiter")
    keys_by_partition: dict[str, list[Any]] = {}
    for name, path in partitions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise DataValidationError("Partition names must start with a letter and contain only letters, numbers, _ or -")
        frame = read_identifier_table(Path(path), delimiter)
        missing = [key for key in key_columns if key not in frame.columns]
        if missing:
            raise DataValidationError(f"partition {name!r} is missing key column(s): {', '.join(missing)}")
        selected = frame.loc[:, list(key_columns)]
        keys_by_partition[name] = (selected.iloc[:, 0].tolist() if len(key_columns) == 1
                                   else list(selected.itertuples(index=False, name=None)))
    return audit_split_overlap(keys_by_partition)

"""Data quality and exact split-overlap summaries for research workflows.

This is a descriptive report, not a proof that a dataset or model is safe.
No molecules are standardized, and identifier values are not exported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.split_audit import audit_split_overlap


@dataclass(frozen=True)
class DatasetAudit:
    """Counts over the supplied rows; no raw identifiers are returned."""

    rows: int
    columns: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    duplicate_keys: int | None
    missing_keys: int | None
    class_counts: tuple[tuple[str, int], ...] | None
    partition_rows: dict[str, int] | None
    shared_keys: int | None
    overlap_rows: dict[str, int] | None
    limitations: str = (
        "Exact identifiers only; this cannot rule out related compounds, "
        "patient relationships, label leakage, or model-preprocessing leakage. "
        "SMILES are not normalized."
    )


def audit_dataset(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    key: str | None = None,
    partition: str | None = None,
) -> DatasetAudit:
    """Summarize missing cells, duplicates, class counts, and exact split overlap.

    Duplicate counts exclude the first occurrence. Missing IDs are counted but,
    when a partition audit is requested, rejected: they cannot identify leakage
    units. Partition labels must be nonempty strings; keys use the strict
    validation/equality of :func:`audit_split_overlap`. Target counts include
    missing labels as a separate category, but infer no classification task.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("audit requires a nonempty DataFrame")
    if not frame.columns.is_unique or not all(isinstance(c, str) and c for c in frame.columns):
        raise DataValidationError("column names must be unique nonempty strings")
    for role, name in (("target", target), ("key", key), ("partition", partition)):
        if name is not None and name not in frame.columns:
            raise DataValidationError(f"{role} column {name!r} not found")
    if partition is not None and key is None:
        raise DataValidationError("a key column is required to audit partition overlap")
    try:
        duplicate_rows = int(frame.duplicated().sum())
        duplicate_keys = int(frame[key].duplicated().sum()) if key is not None else None
    except TypeError as exc:
        raise DataValidationError("audit requires scalar, hashable cell values") from exc
    missing = {name: int(frame[name].isna().sum()) for name in frame.columns}
    missing_keys = None
    if key is not None:
        values = frame[key]
        missing_keys = int((values.isna() | values.map(lambda v: isinstance(v, str) and not v.strip())).sum())
    class_counts = None
    if target is not None:
        counts = frame[target].value_counts(dropna=False, sort=False)
        class_counts = tuple(("<missing>" if pd.isna(value) else repr(value), int(count))
                             for value, count in counts.items())
    partition_rows: dict[str, int] | None = None
    overlap_rows: dict[str, int] | None = None
    shared_keys = None
    if partition is not None and key is not None:
        labels = frame[partition].tolist()
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise DataValidationError("partition labels must be nonempty strings on every row")
        if missing_keys:
            raise DataValidationError("partition overlap audit requires a known key on every row")
        groups: dict[str, list[Any]] = {}
        for label, value in zip(labels, frame[key].tolist(), strict=True):
            groups.setdefault(label, []).append(value)
        overlap = audit_split_overlap(groups)
        partition_rows = dict(zip(overlap.partition_names, overlap.partition_rows, strict=True))
        overlap_rows = dict(zip(overlap.partition_names, overlap.overlap_rows, strict=True))
        shared_keys = overlap.n_shared_keys
    return DatasetAudit(rows=len(frame), columns=len(frame.columns), missing_by_column=missing,
                        duplicate_rows=duplicate_rows, duplicate_keys=duplicate_keys,
                        missing_keys=missing_keys, class_counts=class_counts,
                        partition_rows=partition_rows, shared_keys=shared_keys,
                        overlap_rows=overlap_rows)

"""Exact-key overlap audits across data partitions.

Detect shared identifiers without normalizing or interpreting them. Keys can
represent compounds, patients, assay groups or another leakage unit chosen by
the caller. This detects exact overlap only; it cannot establish that partitions
are independent or that differing molecular representations are distinct.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = ["SharedSplitKey", "SplitOverlapReport", "audit_split_overlap"]


@dataclass(frozen=True)
class SharedSplitKey:
    """One cross-partition key and its row counts, in partition order."""

    key: Any
    counts: tuple[int, ...]


@dataclass(frozen=True)
class SplitOverlapReport:
    """Exact-key overlap and duplicate counts for named, nonempty partitions.

    Counts are in ``partition_names`` order; shared keys follow first appearance
    across partitions. ``duplicate_rows`` counts extra rows after the first
    occurrence of each key within each partition. ``overlap_rows`` counts all
    rows whose key appears in at least one other partition. Pairwise counts
    measure unique keys, not row pairs. The pairwise array remains mutable.
    Identifiers can be sensitive: do not share the detailed output unless
    appropriate.
    """

    partition_names: tuple[str, ...]
    partition_rows: tuple[int, ...]
    duplicate_rows: tuple[int, ...]
    overlap_rows: tuple[int, ...]
    shared_keys: tuple[SharedSplitKey, ...]
    pairwise: np.ndarray

    @property
    def has_overlap(self) -> bool:
        return bool(self.shared_keys)

    @property
    def n_shared_keys(self) -> int:
        return len(self.shared_keys)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per shared key and per-partition occurrence counts."""
        columns = ["key", *(f"n_{name}" for name in self.partition_names)]
        return pd.DataFrame([[record.key, *record.counts] for record in self.shared_keys],
                            columns=columns)

    def pairwise_dataframe(self) -> pd.DataFrame:
        """Return unique-key intersections; diagonal entries are unique counts."""
        return pd.DataFrame(self.pairwise, index=self.partition_names, columns=self.partition_names)


def _validate_key(key: Any, partition: str, row: int) -> None:
    """Reject missing or ambiguous identifiers rather than masking overlap."""
    try:
        hash(key)
    except TypeError as exc:
        raise DataValidationError(f"partition {partition!r} row {row} has an unhashable key") from exc
    if isinstance(key, tuple):
        if not key:
            raise DataValidationError(f"partition {partition!r} row {row} has an empty key")
        for item in key:
            _validate_key(item, partition, row)
        return
    if isinstance(key, (str, bytes)):
        if not key.strip():
            raise DataValidationError(f"partition {partition!r} row {row} has an empty key")
        return
    if isinstance(key, (bool, np.bool_, Integral)):
        return
    if isinstance(key, Real):
        try:
            finite = math.isfinite(float(key))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if finite:
            return
    raise DataValidationError(
        f"partition {partition!r} row {row} has an invalid key; use nonempty strings, "
        "finite numeric scalars, or tuples of those"
    )


def audit_split_overlap(partitions: Mapping[str, Iterable[Any]]) -> SplitOverlapReport:
    """Audit exact identifier overlap across two or more named data partitions.

    ``partitions`` maps unique nonempty string names to nonempty one-dimensional
    iterables of keys. Each row must have a nonempty string, finite number, or
    a tuple of those; missing and unsupported keys raise rather than being
    silently ignored. Repeated keys are retained as row counts. Key comparison
    follows Python equality, so numeric values like ``1`` and ``1.0`` match.
    No trimming, SMILES parsing, canonicalization, or chemical similarity
    comparison is performed. Pandas indexes are ignored.

    A clean result rules out only exact overlap for the supplied key definition.
    It cannot detect related compounds, patient relationships, label leakage,
    improperly fitted preprocessing, or distribution shift. Choose keys that
    identify the leakage unit and audit each relevant level independently.
    """
    if not isinstance(partitions, Mapping) or len(partitions) < 2:
        raise DataValidationError("partitions must map at least two names to key sequences")
    names = tuple(partitions.keys())
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise DataValidationError("partition names must be nonempty strings")
    n_parts = len(names)
    occurrences: dict[Any, list[int]] = {}
    rows_per_partition: list[int] = []
    for part, name in enumerate(names):
        keys = partitions[name]
        if (isinstance(keys, (str, bytes, Mapping, pd.DataFrame))
                or not isinstance(keys, Iterable)
                or isinstance(keys, np.ndarray) and keys.ndim != 1):
            raise DataValidationError(f"partition {name!r} must be a one-dimensional key iterable")
        count = 0
        for row, key in enumerate(keys):
            _validate_key(key, name, row)
            occurrences.setdefault(key, [0] * n_parts)[part] += 1
            count += 1
        if count == 0:
            raise DataValidationError(f"partition {name!r} must not be empty")
        rows_per_partition.append(count)
    pairwise = np.zeros((n_parts, n_parts), dtype=np.int64)
    overlap_rows = [0] * n_parts
    unique_counts = [0] * n_parts
    shared: list[SharedSplitKey] = []
    for key, counts in occurrences.items():
        present = [index for index, count in enumerate(counts) if count]
        for i in present:
            unique_counts[i] += 1
            for j in present:
                pairwise[i, j] += 1
        if len(present) > 1:
            shared.append(SharedSplitKey(key=key, counts=tuple(counts)))
            for i in present:
                overlap_rows[i] += counts[i]
    duplicates = tuple(total - unique for total, unique in zip(rows_per_partition, unique_counts, strict=True))
    return SplitOverlapReport(partition_names=names, partition_rows=tuple(rows_per_partition),
                              duplicate_rows=duplicates, overlap_rows=tuple(overlap_rows),
                              shared_keys=tuple(shared), pairwise=pairwise)

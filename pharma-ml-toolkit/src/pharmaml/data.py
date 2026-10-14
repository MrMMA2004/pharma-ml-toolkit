"""Bring your own data: import user files (CSV/TSV/TXT/Excel) into PharmaML.

The legacy workflow hardcoded dataset filenames inside ~30 model scripts
(``1000selected_features_classification.csv``, ``BBB.xlsx``, ...), so
pointing the pipeline at *your own* file meant editing source code.
This module is the single documented entry point for user data and
replaces that workflow entirely:

1. :func:`load_dataset` reads any CSV/TSV/TXT/Excel file, resolves the
   target column (given explicitly or auto-detected by convention),
   separates numeric features from the target, and infers whether the
   endpoint is a **classification label (2 or more classes)** or a
   **continuous regression value**.
2. :func:`summarize_dataset` reports shape, class balance, missing
   values and dropped columns, so data problems surface *before*
   training.
3. The CLI exposes the same flow without writing any code::

       pharmaml data inspect --data my_compounds.csv
       pharmaml ml train --data my_compounds.csv --target activity --task auto

Example
-------
>>> from pharmaml.data import load_dataset
>>> from pharmaml.ml import train_model, evaluate_model
>>>
>>> ds = load_dataset("my_compounds.csv", target="activity")
>>> ds.task                     # 'classification' (inferred)
>>> ds.classes                  # ['active', 'inactive', 'partial'] — 2+ classes OK
>>> result = train_model("random_forest", ds.task, ds.X, ds.y)
>>> report = evaluate_model(result.model, ds.X_test, ds.y_test, ds.task)  # doctest: +SKIP

Multiclass classification is supported natively: any target column
containing between 2 and ``max_class_count`` distinct values (or any
non-numeric target) is treated as a classification problem with that
many classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table

__all__ = [
    "TARGET_CANDIDATES",
    "Dataset",
    "DatasetSummary",
    "load_dataset",
    "summarize_dataset",
    "infer_task",
]

_LOG = get_logger(__name__)

#: Column names tried (in order) when ``target`` is not given explicitly.
TARGET_CANDIDATES: tuple[str, ...] = (
    "Target",
    "target",
    "Class",
    "class",
    "Label",
    "label",
    "Activity",
    "activity",
    "Outcome",
    "outcome",
    "pIC50",
    "PIC50",
    "IC50",
    "y",
    "Y",
)

_VALID_TASKS = {"classification", "regression"}


def infer_task(y: pd.Series | np.ndarray, max_class_count: int = 20) -> str:
    """Decide ``'classification'`` or ``'regression'`` from a target column.

    Rules
    -----
    * Non-numeric targets (strings, categories, booleans) are always
      **classification**.
    * Numeric targets holding between 2 and ``max_class_count`` distinct
      values (e.g. ``0/1/2`` multiclass labels) are **classification**.
    * Numeric targets with more distinct values are **regression**
      (continuous endpoints such as pIC50).

    Parameters
    ----------
    y:
        Target column (any array-like).
    max_class_count:
        A numeric column with at most this many distinct values is
        considered a class label rather than a measurement.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.

    Raises
    ------
    DataValidationError
        If the target has fewer than 2 distinct values (nothing to
        learn) or ``max_class_count`` is not sensible.
    """
    if max_class_count < 2:
        raise DataValidationError(f"max_class_count must be >= 2, got {max_class_count}")

    series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
    n_unique = int(series.nunique(dropna=True))
    if n_unique < 2:
        raise DataValidationError(
            f"Target column has {n_unique} distinct value(s); "
            "classification and regression both need at least 2."
        )

    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return "classification"
    if n_unique <= max_class_count:
        return "classification"
    return "regression"


@dataclass(slots=True)
class Dataset:
    """A user-supplied dataset split into features and target.

    Attributes
    ----------
    frame:
        The full cleaned table (target column included).
    X:
        Feature matrix — numeric columns only; string columns such as
        SMILES or compound IDs are dropped (and reported in
        ``dropped_columns``) exactly like the legacy
        ``Non-numerical data remover.py`` script did, but visibly.
    y:
        Target column.
    target:
        Name of the target column.
    task:
        ``'classification'`` or ``'regression'`` (inferred or explicit).
    path:
        Source file path when loaded from disk (``None`` for frames).
    classes:
        Sorted class labels for classification, else ``None``.
    class_counts:
        ``{label: row_count}`` for classification, else ``None``.
    dropped_columns:
        Non-numeric columns excluded from ``X``.
    """

    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    target: str
    task: str
    path: Path | None = None
    classes: list[Any] | None = None
    class_counts: dict[Any, int] | None = None
    dropped_columns: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return int(len(self.frame))

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.X.shape[1])

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature column names."""
        return list(self.X.columns)

    def summary(self) -> DatasetSummary:
        """Convenience wrapper around :func:`summarize_dataset`."""
        return summarize_dataset(self)


@dataclass(slots=True)
class DatasetSummary:
    """Human-readable audit of a loaded dataset.

    Built by :func:`summarize_dataset`; the ``__str__`` form is what the
    ``pharmaml data inspect`` CLI prints.
    """

    path: str
    n_rows: int
    n_columns: int
    n_features: int
    target: str
    task: str
    classes: list[Any] | None
    class_counts: dict[Any, int] | None
    missing_cells: int
    duplicate_rows: int
    dropped_columns: list[str]
    memory_mb: float

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for JSON/DataFrame consumption."""
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_features": self.n_features,
            "target": self.target,
            "task": self.task,
            "classes": self.classes,
            "class_counts": self.class_counts,
            "missing_cells": self.missing_cells,
            "duplicate_rows": self.duplicate_rows,
            "dropped_columns": self.dropped_columns,
            "memory_mb": round(self.memory_mb, 4),
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lines = [
            f"source        : {self.path}",
            f"shape         : {self.n_rows} rows x {self.n_columns} columns "
            f"({self.n_features} numeric features)",
            f"target        : {self.target}",
            f"task          : {self.task}",
            f"missing cells : {self.missing_cells}",
            f"dup. rows     : {self.duplicate_rows}",
        ]
        if self.task == "classification" and self.class_counts:
            balance = ", ".join(f"{k!r}: {v}" for k, v in self.class_counts.items())
            lines.append(f"classes ({len(self.class_counts)}): {balance}")
        if self.dropped_columns:
            shown = ", ".join(self.dropped_columns[:8])
            extra = len(self.dropped_columns) - 8
            more = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"dropped cols  : {shown}{more}  [non-numeric]")
        return "\n".join(lines)


def _resolve_target(frame: pd.DataFrame, target: str | None) -> str:
    """Return the target column name, auto-detecting when ``None``."""
    if target is not None:
        if target not in frame.columns:
            raise DataValidationError(
                f"Target column {target!r} not found. "
                f"Available columns: {list(frame.columns)[:20]}"
            )
        return target

    for candidate in TARGET_CANDIDATES:
        if candidate in frame.columns:
            _LOG.info("Target column auto-detected: %r", candidate)
            return candidate

    # Case-insensitive second pass (catches 'TARGET', 'Target ', typos).
    lowered = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in TARGET_CANDIDATES:
        actual = lowered.get(candidate.lower())
        if actual is not None:
            _LOG.info("Target column auto-detected (case-insensitive): %r", actual)
            return actual

    raise DataValidationError(
        "No target column found automatically. Pass target=<column name>. "
        f"Available columns: {list(frame.columns)[:20]}"
    )


def load_dataset(
    path: str | Path,
    target: str | None = None,
    task: str | None = None,
    drop_cols: list[str] | None = None,
    max_class_count: int = 20,
    sheet_name: str | int = 0,
) -> Dataset:
    """Load **your own** data file and split it ready for training.

    This is the documented entry point for user data — it replaces the
    legacy habit of editing a hardcoded ``pd.read_csv(...)`` line inside
    whichever model script you wanted to run.

    Parameters
    ----------
    path:
        Path to a CSV/TSV/TXT/Excel file (``.csv``, ``.tsv``, ``.txt``,
        ``.xlsx``, ``.xls``, ``.xlsm``).  Delimiters are sniffed
        automatically (comma / semicolon / tab).
    target:
        Name of the column holding the endpoint (class label or pIC50).
        When ``None``, common names (``Target``, ``class``, ``label``,
        ``activity``, ``pIC50``, ``y`` ...) are tried automatically.
    task:
        ``'classification'`` or ``'regression'``; when ``None`` the task
        is inferred from the target column (see :func:`infer_task`).
        Classification targets may contain **two or more classes**.
    drop_cols:
        Extra columns to exclude from the feature matrix (IDs, SMILES,
        pre-computed splits ...).  Non-numeric columns are always
        excluded automatically.
    max_class_count:
        Numeric targets with at most this many distinct values count as
        class labels (passed to :func:`infer_task`).
    sheet_name:
        Excel sheet to read (ignored for CSV-like inputs).

    Returns
    -------
    Dataset
        With ``X`` (numeric features), ``y`` (target), ``task``,
        ``classes``/``class_counts`` for classification, and the list of
        dropped non-numeric columns.

    Raises
    ------
    DataValidationError
        File missing/unreadable, target column not found, a
        single-class target, or a non-numeric target with
        ``task='regression'``.

    Example
    -------
    >>> ds = load_dataset("qsar.csv", target="activity")   # doctest: +SKIP
    >>> ds.class_counts                                    # doctest: +SKIP
    {'active': 210, 'inactive': 340}
    """
    if task is not None and task not in _VALID_TASKS:
        raise DataValidationError(
            f"task must be one of {sorted(_VALID_TASKS)} or None (auto), got {task!r}"
        )

    file_path = Path(path)
    frame = read_table(file_path, sheet_name=sheet_name)
    if frame.empty:
        raise DataValidationError(f"'{file_path}' contains no data rows")

    target_col = _resolve_target(frame, target)

    # Rows without a target cannot participate in training.
    n_missing_target = int(frame[target_col].isna().sum())
    if n_missing_target:
        _LOG.warning("Dropping %d row(s) with missing target", n_missing_target)
        frame = frame.loc[frame[target_col].notna()].reset_index(drop=True)

    y = frame[target_col]
    resolved_task = task or infer_task(y, max_class_count=max_class_count)

    if resolved_task == "classification":
        n_classes = int(y.nunique(dropna=True))
        if n_classes < 2:
            raise DataValidationError(
                f"Classification target {target_col!r} has {n_classes} class(es); "
                "at least 2 are required."
            )
        classes = sorted(y.unique().tolist())
        class_counts = {label: int((y == label).sum()) for label in classes}
        smallest = min(class_counts.values())
        if smallest < 5:
            _LOG.warning(
                "Class %r has only %d member(s); consider more data or merge "
                "rare classes before training",
                min(class_counts, key=class_counts.get),
                smallest,
            )
    else:  # regression
        if not pd.api.types.is_numeric_dtype(y):
            raise DataValidationError(
                f"task='regression' requested but target {target_col!r} is "
                "non-numeric; use task='classification' instead."
            )
        classes = None
        class_counts = None

    # Feature matrix: numeric columns only, target/drop_cols excluded.
    exclude = {target_col, *(drop_cols or [])}
    numeric = [
        col
        for col in frame.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(frame[col])
    ]
    dropped = [col for col in frame.columns if col not in exclude and col not in numeric]
    if dropped:
        _LOG.warning(
            "Excluding %d non-numeric column(s) from features: %s",
            len(dropped),
            dropped[:10],
        )

    X = frame.loc[:, numeric]
    dataset = Dataset(
        frame=frame,
        X=X,
        y=y.reset_index(drop=True),
        target=target_col,
        task=resolved_task,
        path=file_path,
        classes=classes,
        class_counts=class_counts,
        dropped_columns=dropped,
    )
    _LOG.info(
        "Loaded dataset '%s': %d rows, %d features, task=%s%s",
        file_path,
        dataset.n_samples,
        dataset.n_features,
        resolved_task,
        f" ({len(classes)} classes)" if classes else "",
    )
    return dataset


def summarize_dataset(dataset: Dataset) -> DatasetSummary:
    """Audit a :class:`Dataset` for obvious data-quality problems.

    Reports shape, class balance (classification), missing cells,
    duplicate rows and memory footprint.  Run this before training —
    the legacy scripts discovered these issues as cryptic sklearn
    crashes instead.
    """
    frame = dataset.frame
    return DatasetSummary(
        path=str(dataset.path) if dataset.path is not None else "<in-memory frame>",
        n_rows=dataset.n_samples,
        n_columns=int(frame.shape[1]),
        n_features=dataset.n_features,
        target=dataset.target,
        task=dataset.task,
        classes=dataset.classes,
        class_counts=dataset.class_counts,
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        dropped_columns=list(dataset.dropped_columns),
        memory_mb=float(frame.memory_usage(deep=True).sum() / 1024**2),
    )

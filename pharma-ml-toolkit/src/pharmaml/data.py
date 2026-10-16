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

4. :func:`split_table` upgrades the legacy ``Split CSV file.py`` script
   into a stratified, target-aware splitter that can also carve a
   validation set and write ``<prefix>_train/test/val`` files::

       pharmaml data split --data qsar.csv --test-size 0.2 --val-size 0.1

Example
-------
>>> from pharmaml.data import load_dataset   # usage sketch (needs a real CSV)
>>> ds = load_dataset("my_compounds.csv", target="activity")  # doctest: +SKIP
>>> ds.task                     # 'classification' (inferred)  # doctest: +SKIP
>>> ds.classes                  # ['active', 'inactive', 'partial'] — 2+ classes OK  # doctest: +SKIP
>>> result = train_model("random_forest", ds.task, ds.X, ds.y)  # doctest: +SKIP
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
from pharmaml.utils.validation import looks_continuous

__all__ = [
    "TARGET_CANDIDATES",
    "Dataset",
    "DatasetSummary",
    "SplitResult",
    "infer_task",
    "load_dataset",
    "split_table",
    "summarize_dataset",
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
        Feature matrix — numeric columns only by default; string
        columns such as SMILES or compound IDs are dropped (and
        reported in ``dropped_columns``) exactly like the legacy
        ``Non-numerical data remover.py`` script did, but visibly.
        With ``load_dataset(..., keep_categorical=True)`` string
        columns are kept here and listed in ``categorical_columns``;
        ``train_model`` then one-hot encodes them automatically.
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
        Non-numeric columns excluded from ``X`` (empty when
        ``keep_categorical=True``).
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
    categorical_columns: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return len(self.frame)

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
    categorical_columns: list[str]
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
            "categorical_columns": self.categorical_columns,
            "memory_mb": round(self.memory_mb, 4),
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        feature_note = f"{self.n_features} features"
        if self.categorical_columns:
            feature_note += (
                f" ({self.n_features - len(self.categorical_columns)} numeric + "
                f"{len(self.categorical_columns)} categorical)"
            )
        else:
            feature_note += " (numeric)"
        lines = [
            f"source        : {self.path}",
            f"shape         : {self.n_rows} rows x {self.n_columns} columns ({feature_note})",
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
        if self.categorical_columns:
            shown = ", ".join(self.categorical_columns[:8])
            extra = len(self.categorical_columns) - 8
            more = f" (+{extra} more)" if extra > 0 else ""
            lines.append(
                f"categorical   : {shown}{more}  [kept; one-hot at train time]"
            )
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
    keep_categorical: bool = False,
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
    keep_categorical:
        Keep non-numeric feature columns (assay names, cell lines,
        formulation codes ...) in ``X`` instead of dropping them;
        they are listed in ``categorical_columns`` and one-hot
        encoded automatically at train time by ``train_model`` /
        ``compare_models``.  Metadata like SMILES should still go
        through ``drop_cols`` — one-hot encoding a structure string
        is meaningless.

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
                min(class_counts, key=lambda label: class_counts[label]),
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

    # Feature matrix: numeric columns only (default), target/drop_cols
    # excluded; with keep_categorical the non-numeric feature columns
    # survive in X for one-hot encoding inside train_model.
    exclude = {target_col, *(drop_cols or [])}
    candidate_cols = [col for col in frame.columns if col not in exclude]
    numeric = [
        col for col in candidate_cols if pd.api.types.is_numeric_dtype(frame[col])
    ]
    categorical = [col for col in candidate_cols if col not in numeric]
    if keep_categorical:
        feature_cols = candidate_cols
        dropped: list[str] = []
        if categorical:
            _LOG.info(
                "Keeping %d categorical column(s) as features (one-hot at "
                "train time): %s",
                len(categorical),
                categorical[:10],
            )
    else:
        feature_cols = numeric
        dropped = categorical
        if dropped:
            _LOG.warning(
                "Excluding %d non-numeric column(s) from features: %s",
                len(dropped),
                dropped[:10],
            )

    X = frame.loc[:, feature_cols]
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
        categorical_columns=categorical if keep_categorical else [],
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
        categorical_columns=list(dataset.categorical_columns),
        memory_mb=float(frame.memory_usage(deep=True).sum() / 1024**2),
    )


@dataclass(slots=True)
class SplitResult:
    """Outcome of :func:`split_table` — the split frames plus audit info.

    Attributes
    ----------
    train, test, val:
        The partitioned frames (``val`` is ``None`` unless a validation
        fraction was requested).
    target:
        Column the split was stratified on (``None`` for plain splits).
    stratified:
        Whether stratification was actually applied (``False`` when a
        rare class forced the fallback, with a logged warning).
    paths:
        ``{'train': Path, ...}`` when the frames were written to disk.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame | None
    target: str | None
    stratified: bool
    random_state: int
    paths: dict[str, Path] = field(default_factory=dict)

    def sides(self) -> dict[str, pd.DataFrame]:
        """``{'train': ..., 'test': ..., ('val': ...)}`` without ``None``."""
        out = {"train": self.train, "test": self.test}
        if self.val is not None:
            out["val"] = self.val
        return out

    def write(self, output_dir: str | Path, prefix: str) -> dict[str, Path]:
        """Write each side as ``<prefix>_<side>.csv``; records ``paths``."""
        from pharmaml.utils.io import write_table

        for side, frame in self.sides().items():
            self.paths[side] = write_table(frame, Path(output_dir) / f"{prefix}_{side}.csv")
        return dict(self.paths)

    def _balance(self, frame: pd.DataFrame) -> dict[Any, int] | None:
        """Class counts of the target column when it is label-like."""
        if self.target is None or self.target not in frame.columns:
            return None
        counts = frame[self.target].value_counts()
        if len(counts) > 20:  # continuous endpoint — a balance row is noise
            return None
        return {label: int(n) for label, n in counts.items()}

    def __str__(self) -> str:  # pragma: no cover - presentation only
        mode = "stratified" if self.stratified else "random (unstratified)"
        lines = [f"split        : {mode} on {self.target!r}",
                 f"train        : {len(self.train)} rows"]
        if self.val is not None:
            lines.append(f"val          : {len(self.val)} rows")
        lines.append(f"test         : {len(self.test)} rows")
        for side, frame in self.sides().items():
            balance = self._balance(frame)
            if balance is not None:
                shown = ", ".join(f"{k!r}: {v}" for k, v in sorted(balance.items(), key=str))
                lines.append(f"{side:<12s}: classes: {shown}")
        for side, path in self.paths.items():
            lines.append(f"{side:<12s}: file    : {path}")
        return "\n".join(lines)


def split_table(
    path: str | Path | None = None,
    frame: pd.DataFrame | None = None,
    target: str | None = None,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify: bool = True,
    random_state: int = 42,
) -> SplitResult:
    """Split a dataset table into train / (optional val) / test parts.

    Upgrades the legacy ``Split CSV file.py`` script, which cut a CSV in
    half with no regard for class balance — every rare class could land
    in one side and the later training would crash or silently degrade.

    Parameters
    ----------
    path, frame:
        Either a file path (read via :func:`pharmaml.utils.io.read_table`)
        or an in-memory DataFrame; exactly one must be given.
    target:
        Column to stratify on.  When ``None`` the same auto-detection as
        :func:`load_dataset` applies; pass ``--no-stratify`` (CLI) or
        ``stratify=False`` to split plain rows without a target.
    test_size:
        Fraction of rows held out for testing (0 < test_size < 1).
    val_size:
        Optional fraction of the **original** rows held out for
        validation, carved out of the train side *after* the test split.
    stratify:
        Attempt class-balanced splits (default).  Falls back to a plain
        random split with a warning when a class has a single member.

    Returns
    -------
    SplitResult
        With ``train``/``test`` (and ``val``) frames; call
        ``.write(dir, prefix)`` or let the CLI write the files.

    Raises
    ------
    DataValidationError
        For bad fractions, missing files/columns, or a stratified split
        that is impossible (e.g. a class smaller than the number of
        sides requested).

    Example
    -------
    >>> result = split_table("qsar.csv", test_size=0.2)  # doctest: +SKIP
    >>> result.write("splits", prefix="qsar")             # doctest: +SKIP
    {'train': PosixPath('splits/qsar_train.csv'), ...}
    """
    from sklearn.model_selection import train_test_split

    from pharmaml.utils.io import read_table

    if (path is None) == (frame is None):
        raise DataValidationError("give either path=... or frame=..., not both/neither")
    if not 0.0 < test_size < 1.0:
        raise DataValidationError(f"test_size must be in (0, 1), got {test_size}")
    if val_size is not None and not 0.0 < val_size < 1.0:
        raise DataValidationError(f"val_size must be in (0, 1) or None, got {val_size}")

    # Nested so mypy can narrow: exactly one of path / frame is set here.
    table: pd.DataFrame
    if path is not None:
        if frame is not None:
            raise DataValidationError("give either path=... or frame=..., not both/neither")
        table = read_table(Path(path))
    else:
        if frame is None:
            raise DataValidationError("give either path=... or frame=..., not both/neither")
        table = frame
    if table.empty:
        raise DataValidationError("table contains no data rows")
    if len(table) < 4:
        raise DataValidationError(f"{len(table)} row(s) is too few to split")

    target_col: str | None = target
    if target_col is None and stratify:
        try:
            target_col = _resolve_target(table, None)
        except DataValidationError:
            raise DataValidationError(
                "stratify=True needs a target column (auto-detection found none); "
                "pass target=<column> or stratify=False"
            ) from None
    if target_col is not None and target_col not in table.columns:
        raise DataValidationError(
            f"target column {target_col!r} not found; available: {list(table.columns)[:20]}"
        )

    indices = np.arange(len(table))
    labels = table[target_col] if target_col is not None else None

    use_stratify = stratify and labels is not None
    # ``labels is not None`` spelled in the condition so mypy can narrow
    # the Optional inside the block (``use_stratify`` alone cannot).
    if use_stratify and labels is not None:
        if looks_continuous(labels):
            # Regression-style target column: stratification is not a thing
            # -- skip quietly instead of warning about 1-member "classes".
            _LOG.debug(
                "continuous target column %r -- splitting without stratification",
                str(target_col),
            )
            use_stratify = False
        else:
            counts = labels.value_counts()
            n_sides = 3 if val_size else 2
            # Each side needs at least one member of every class.
            if counts.min() < n_sides:
                _LOG.warning(
                    "Class %r has %d member(s) < %d sides; falling back to an "
                    "unstratified split.",
                    str(counts.idxmin()),
                    int(counts.min()),
                    n_sides,
                )
                use_stratify = False

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if use_stratify else None,
    )

    val: pd.DataFrame | None = None
    if val_size is not None:
        # carve val out of the train side, keeping the overall fraction ≈ val_size
        remaining = 1.0 - test_size
        adjusted = val_size / remaining
        if adjusted >= 1.0:
            raise DataValidationError(
                f"val_size={val_size} leaves no training rows "
                f"(test_size={test_size})"
            )
        sub_labels = labels.iloc[train_idx] if labels is not None and use_stratify else None
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=adjusted,
            random_state=random_state,
            stratify=sub_labels if sub_labels is not None else None,
        )
        val = table.iloc[val_idx].reset_index(drop=True)

    result = SplitResult(
        train=table.iloc[train_idx].reset_index(drop=True),
        test=table.iloc[test_idx].reset_index(drop=True),
        val=val,
        target=target_col,
        stratified=use_stratify,
        random_state=random_state,
    )
    _LOG.info(
        "Split %d rows -> train=%d%s test=%d (%s)",
        len(table),
        len(result.train),
        f" val={len(val)}" if val is not None else "",
        len(result.test),
        "stratified" if use_stratify else "unstratified",
    )
    return result

"""Drug-response screens: GDSC/CCLE-style pharmacology tables.

A *response screen* is a long-format table with one row per
(cell line, compound) pair and one or more numeric metric columns --
the layout popularised by the GDSC and CCLE sensitivity datasets
(``LN_IC50``, ``AUC``, ``RMSE`` ...).  This module loads, audits and
reshapes such tables:

* :func:`load_response_table` accepts a DataFrame or a csv/xlsx path,
  validates the identifier columns, auto-detects the metric columns
  (every remaining numeric column) and coerces stray non-numeric
  entries to NaN *with a warning* (round-11/16 semantics: warn, keep
  NaN -- never drop rows silently),
* :class:`ResponseScreen` is the immutable container carrying the
  frame plus the column configuration, with a ``summary()`` in the
  package's audit style,
* :func:`audit_response_table` reports the classic screen hygiene
  issues (missing identifiers, missing metrics, duplicate pairs, cell
  lines claimed by two tissues, under-tested compounds),
* :func:`collapse_replicates` resolves duplicated pairs by the median
  per metric (rows with a missing identifier are kept as their own
  NaN-keyed group, never silently dropped), optionally carrying each
  metric's replicate CV alongside the collapsed median
  (``add_cv=True``),
* :func:`replicate_variability` reports the pre-collapse spread of
  every duplicated (cell line, compound) pair -- replicate count,
  median, mean, std, spread and the coefficient of variation -- so
  the noise the median collapse discards stays visible,
* :func:`ln_ic50_to_pic50` / :func:`pic50_to_ln_ic50` convert between
  GDSC's ``LN_IC50`` (natural log of the IC50 in micromolar) and the
  chemist's pIC50 scale,
* :func:`response_matrix` pivots one metric into the classic
  cell-line x compound sensitivity matrix (untested pairs stay NaN),
* :func:`summarize_compounds` gives per-compound potency statistics
  across the panel, and
* :func:`call_sensitive_resistant` thresholds one compound's response
  vector into sensitive / intermediate / resistant calls.

Direction conventions matter everywhere here: for ``LN_IC50``-style
metrics *lower* means more sensitive, for pIC50-style metrics *higher*
means more sensitive.  Every direction-sensitive function documents
its assumption and exposes a switch.

Examples
--------
>>> from pharmaml.biology.response import ln_ic50_to_pic50, pic50_to_ln_ic50
>>> ln_ic50_to_pic50(0.0)        # LN_IC50 0 -> IC50 1 uM -> pIC50 6
6.0
>>> pic50_to_ln_ic50(6.0)        # ... and back
0.0
>>> round(ln_ic50_to_pic50(3.0), 3)   # IC50 ~20.1 uM
4.697
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table, read_table_with_progress
from pharmaml.utils.validation import require_columns

_LOG = get_logger(__name__)

__all__ = [
    "ResponseScreen",
    "audit_response_table",
    "call_sensitive_resistant",
    "collapse_replicates",
    "ln_ic50_to_pic50",
    "load_response_table",
    "pic50_to_ln_ic50",
    "replicate_variability",
    "response_matrix",
    "summarize_compounds",
]

#: What a table argument may be: an in-memory frame or a csv/xlsx path.
ResponseTableSource: TypeAlias = pd.DataFrame | str | Path

#: ``ln(10)`` -- the natural-log/decade conversion factor.
_LN10 = math.log(10.0)


def _detect_metric_columns(
    frame: pd.DataFrame, exclude: list[str]
) -> list[str]:
    """Return the remaining columns that hold numeric measurements.

    A column qualifies when its dtype is numeric or when at least one
    value parses as a number (``pd.to_numeric`` with coercion) -- the
    latter keeps a ``LN_IC50`` column that went ``object`` dtype
    because of a few stray strings in an otherwise numeric export.

    Boolean columns never qualify: ``pandas`` counts ``bool`` as a
    numeric dtype, but an ``msi_status``-style flag is an annotation,
    not a measurement.  The rule covers both ``bool`` dtype and an
    ``object`` column holding only ``True``/``False`` (with missing
    entries -- what a csv round-trip of a flag column produces).
    Datetime columns are excluded in the same spirit: a
    ``screened_on`` screening date is an annotation, and
    ``pd.to_numeric`` would otherwise coerce it to ns-epoch integers
    that masquerade as measurements (the ``datetime64`` dtypes pandas'
    Excel/parquet readers emit).
    Pass such a column explicitly via ``metrics=[...]`` to treat its
    ``True``/``False`` values as 1/0.
    """
    metrics: list[str] = []
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            metrics.append(column)
            continue
        non_missing = series.dropna()
        if not non_missing.empty and non_missing.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            metrics.append(column)
    return metrics


def _coerce_metrics(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Coerce metric columns to numeric dtype, warning per column.

    Values that fail to parse become NaN *after* a warning names the
    column and the affected count (round-11/16 semantics: warn, keep
    NaN).  Already-numeric columns pass through unchanged.  Boolean
    metric columns (only reachable through the explicit ``metrics=``
    path -- auto-detection skips them) are converted to 0/1 floats:
    ``pd.to_numeric`` passes ``bool`` dtype through unchanged, but a
    metric column must support arithmetic.
    """
    for metric in metrics:
        series = frame[metric]
        if pd.api.types.is_bool_dtype(series):
            frame[metric] = series.astype("float64")
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        lost = int((series.notna() & coerced.isna()).sum())
        if lost:
            _LOG.warning(
                "%d non-numeric value(s) in metric column %r coerced to NaN",
                lost,
                metric,
            )
        frame[metric] = coerced
    return frame


@dataclass(frozen=True, slots=True)
class ResponseScreen:
    """One loaded drug-response screen.

    Attributes
    ----------
    frame:
        The long-format table: one row per (cell line, compound) pair
        with the metric columns already coerced to numeric dtype.
    cell_col:
        Name of the cell-line identifier column.
    compound_col:
        Name of the compound identifier column.
    tissue_col:
        Name of the optional tissue/lineage annotation column.
    metrics:
        Names of the numeric response-metric columns (``LN_IC50``,
        ``AUC`` ...).  When empty at construction time the numeric
        non-identifier columns are auto-detected.

    Raises
    ------
    DataValidationError
        If no metric column can be auto-detected.
    """

    frame: pd.DataFrame
    cell_col: str
    compound_col: str
    tissue_col: str | None = None
    metrics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.metrics:
            detected = _detect_metric_columns(
                self.frame,
                exclude=[
                    column
                    for column in (self.cell_col, self.compound_col, self.tissue_col)
                    if column is not None
                ],
            )
            if not detected:
                raise DataValidationError(
                    "no metric columns detected in the response table -- "
                    "pass metrics=[...] explicitly. Available columns: "
                    f"{list(self.frame.columns)}"
                )
            object.__setattr__(self, "metrics", detected)

    @property
    def cell_lines(self) -> list[str]:
        """Sorted unique cell-line identifiers (missing ones excluded)."""
        return sorted(self.frame[self.cell_col].dropna().astype(str).unique().tolist())

    @property
    def compounds(self) -> list[str]:
        """Sorted unique compound identifiers (missing ones excluded)."""
        return sorted(
            self.frame[self.compound_col].dropna().astype(str).unique().tolist()
        )

    @property
    def n_pairs(self) -> int:
        """Number of unique (cell line, compound) pairs."""
        return len(self.frame[[self.cell_col, self.compound_col]].drop_duplicates())

    def _duplicate_pairs(self) -> list[tuple[str, str]]:
        """(cell, compound) pairs that occupy more than one row.

        Uses ``dropna=False`` so pairs keyed by a missing identifier
        are counted too -- silently dropping them would hide
        duplicated unusable rows from every audit.
        """
        counts = self.frame.groupby(
            [self.cell_col, self.compound_col], sort=True, dropna=False
        ).size()
        return [pair for pair, count in counts.items() if count > 1]

    def summary(self) -> str:
        """Multi-line audit summary (the package's ``inspect`` style)."""
        lines = [
            (f"screen        : {len(self.cell_lines)} cell line(s) x "
            f"{len(self.compounds)} compound(s) -> {self.n_pairs} (cell, compound) pair(s)")
        ]
        if len(self.frame) != self.n_pairs:
            lines.append(
                f"rows          : {len(self.frame)} "
                f"({len(self.frame) - self.n_pairs} duplicate row(s))"
            )
        if self.tissue_col is not None:
            tissues = sorted(
                self.frame[self.tissue_col].dropna().astype(str).unique().tolist()
            )
            shown = ", ".join(tissues[:8])
            extra = len(tissues) - 8
            more = f" (+{extra} more)" if extra > 0 else ""
            lines.append(
                f"tissue        : {len(tissues)} value(s) in {self.tissue_col!r}: "
                f"{shown}{more}"
            )
        lines.append(f"metrics       : {', '.join(self.metrics) or '(none)'}")
        for metric in self.metrics:
            present = int(self.frame[metric].notna().sum())
            total = len(self.frame)
            missing = total - present
            note = f" ({missing} missing)" if missing else ""
            lines.append(f"  {metric:<13s}: {present}/{total} value(s) present{note}")
        duplicates = self._duplicate_pairs()
        if duplicates:
            shown = ", ".join(f"({cell} + {compound})" for cell, compound in duplicates[:3])
            extra = len(duplicates) - 3
            more = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"dup. pairs    : {len(duplicates)}: {shown}{more}")
        else:
            lines.append("dup. pairs    : 0")
        return "\n".join(lines)


def load_response_table(
    data: ResponseTableSource,
    cell_column: str = "cell_line",
    compound_column: str = "compound",
    tissue_column: str | None = None,
    metrics: list[str] | None = None,
    report: Callable[[int, int | None, str], None] | None = None,
) -> ResponseScreen:
    """Load a drug-response screen from a DataFrame or csv/xlsx path.

    Parameters
    ----------
    data:
        Long-format table (one row per cell line x compound pair) as a
        DataFrame or a path read via
        :func:`pharmaml.utils.io.read_table` (streamed through
        :func:`pharmaml.utils.io.read_table_with_progress` when
        ``report`` is given).
    cell_column, compound_column:
        Names of the identifier columns.
    tissue_column:
        Optional tissue annotation column (kept out of the metrics).
    metrics:
        Response-metric column names.  When ``None`` (the default) all
        remaining numeric columns are used -- including ``object``
        columns that hold at least one parseable number, so a stray
        string in an otherwise numeric export degrades to NaN instead
        of hiding the whole metric.  Boolean columns (flags like
        ``msi_status``) and datetime columns (screening dates) are
        *never* auto-detected; pass one explicitly to have its
        ``True``/``False`` values treated as 1/0 (or its datetimes
        coerced, which is the caller's responsibility).
    report:
        Optional progress callback forwarded to
        :func:`pharmaml.utils.io.read_table_with_progress` when ``data``
        is a path -- ``(done_rows, estimated_rows_or_None, note)`` per
        chunk, so a big screen streams in with a determinate progress
        bar.  DataFrame inputs (and ``None``, the default) read exactly
        as before; the callback never fires for them.

    Returns
    -------
    ResponseScreen
        The validated screen.  Metric columns are coerced to numeric
        dtype on a *copy* of the frame (the caller's frame is never
        mutated); non-numeric values become NaN after a warning that
        names the column and the count.

    Raises
    ------
    DataValidationError
        If an identifier column is missing (all offenders listed at
        once), an explicit metric column is missing, or no metric
        column can be detected at all.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, (str, Path)):
        # (100x-144) a path input with a reporter streams through the
        # progressive reader -- a big screen loads with a determinate
        # bar; without one (and for every DataFrame caller) the plain
        # reader's behaviour is unchanged
        if report is None:
            frame = read_table(data)
        else:
            frame = read_table_with_progress(data, report)
    else:
        raise DataValidationError(
            "data must be a pandas DataFrame or a path to a csv/xlsx file, "
            f"got {type(data).__name__}"
        )

    required = [cell_column, compound_column]
    if tissue_column is not None:
        required.append(tissue_column)
    require_columns(frame, required)

    if metrics is None:
        detected = _detect_metric_columns(frame, exclude=required)
        if not detected:
            raise DataValidationError(
                "no metric columns detected in the response table -- pass "
                "metrics=[...] explicitly. Available columns: "
                f"{list(frame.columns)}"
            )
        metrics = detected
        _LOG.debug("Auto-detected metric column(s): %s", metrics)
    else:
        require_columns(frame, list(metrics))

    frame = _coerce_metrics(frame, list(metrics))
    _LOG.info(
        "Loaded response screen: %d row(s), %d cell line(s), %d compound(s), "
        "metric(s) %s",
        len(frame),
        frame[cell_column].nunique(),
        frame[compound_column].nunique(),
        metrics,
    )
    return ResponseScreen(
        frame=frame,
        cell_col=cell_column,
        compound_col=compound_column,
        tissue_col=tissue_column,
        metrics=list(metrics),
    )


def audit_response_table(screen: ResponseScreen, n_min: int = 3) -> pd.DataFrame:
    """Audit a response screen for the classic hygiene issues.

    One row per concrete issue with columns ``issue``, ``detail`` and
    ``count``:

    * ``missing identifier`` -- per identifier column (cell line /
      compound) holding NaN entries; ``count`` is the number of rows
      without the identifier,
    * ``missing metric values`` -- per metric column with missing
      entries; ``count`` is the number of missing values,
    * ``duplicate pairs`` -- per (cell line, compound) pair occupying
      more than one row (pairs keyed by a missing identifier included);
      ``count`` is the number of rows for the pair,
    * ``ambiguous tissue`` -- per cell line mapped to more than one
      tissue; ``count`` is the number of distinct tissues,
    * ``under-tested compound`` -- per compound measured on fewer than
      ``n_min`` cell lines; ``count`` is the number of cell lines.

    A clean screen returns the single row ``("ok", "no issues found",
    0)``.  ``to_string()`` on the result is print-friendly.

    Parameters
    ----------
    screen:
        The screen to audit.
    n_min:
        Minimum number of cell lines a compound should be measured on
        (GDSC-style panels use 3 as the floor for any per-compound
        statistic).
    """
    rows: list[dict[str, object]] = []

    for column in (screen.cell_col, screen.compound_col):
        missing_ids = int(screen.frame[column].isna().sum())
        if missing_ids:
            rows.append(
                {
                    "issue": "missing identifier",
                    "detail": f"column {column!r}",
                    "count": missing_ids,
                }
            )

    for metric in screen.metrics:
        missing = int(screen.frame[metric].isna().sum())
        if missing:
            rows.append(
                {
                    "issue": "missing metric values",
                    "detail": f"metric {metric!r}",
                    "count": missing,
                }
            )

    pair_counts = screen.frame.groupby(
        [screen.cell_col, screen.compound_col], sort=True, dropna=False
    ).size()
    for (cell, compound), count in pair_counts.items():
        if count > 1:
            rows.append(
                {
                    "issue": "duplicate pairs",
                    "detail": f"{cell} / {compound}",
                    "count": int(count),
                }
            )

    if screen.tissue_col is not None:
        tissues = (
            screen.frame[[screen.cell_col, screen.tissue_col]]
            .dropna(subset=[screen.tissue_col])
            .groupby(screen.cell_col, sort=True, dropna=False)[screen.tissue_col]
            .agg(lambda values: sorted(set(values.astype(str))))
        )
        for cell, tissue_list in tissues.items():
            if len(tissue_list) > 1:
                rows.append(
                    {
                        "issue": "ambiguous tissue",
                        "detail": f"{cell}: {', '.join(tissue_list)}",
                        "count": len(tissue_list),
                    }
                )

    n_cells = screen.frame.groupby(screen.compound_col, sort=True, dropna=False)[
        screen.cell_col
    ].nunique()
    for compound, count in n_cells.items():
        if count < n_min:
            rows.append(
                {
                    "issue": "under-tested compound",
                    "detail": f"{compound}",
                    "count": int(count),
                }
            )

    if not rows:
        rows.append({"issue": "ok", "detail": "no issues found", "count": 0})
    report = pd.DataFrame(rows, columns=["issue", "detail", "count"])
    _LOG.info("Response-table audit: %d issue row(s)", len(report))
    return report


def _replicate_cv(values: pd.Series) -> float:
    """Coefficient of variation (percent) of one group's replicates.

    NaN when fewer than two valid values exist (a single observation
    has no spread) or when the mean is exactly zero (the ratio is
    undefined -- reported as NaN, never as a fabricated ``inf``).
    """
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 2:
        return math.nan
    mean = float(valid.mean())
    if mean == 0.0:
        return math.nan
    return abs(float(valid.std(ddof=1)) / mean) * 100.0


def replicate_variability(screen: ResponseScreen) -> pd.DataFrame:
    """Report the replicate spread of every duplicated pair.

    The median collapse in :func:`collapse_replicates` discards the
    noise information by construction: two replicates ``[1.0, 3.0]``
    and ``[1.9, 2.1]`` both collapse to the *same* median ``2.0``.
    This function keeps that noise visible *before* the collapse --
    one row per duplicated (cell line, compound) pair **per metric**,
    with columns ``metric``, ``n_replicates`` (valid values only; NaN
    entries never inflate the count), ``median``, ``mean``, ``std``
    (sample standard deviation, ``ddof=1``), ``spread`` (``max - min``)
    and ``cv_pct`` -- the coefficient of variation
    ``std / |mean| * 100``.

    ``cv_pct`` is NaN when fewer than two valid replicates exist or
    when the mean is exactly zero (the ratio is undefined).  Pairs
    keyed by a missing identifier are included as (NaN-keyed)
    duplicates, mirroring :func:`audit_response_table`.  A screen
    without duplicated pairs returns an *empty frame with the full
    column set* (never ``None``).

    Parameters
    ----------
    screen:
        The screen to inspect (typically the raw, pre-collapse one).

    Returns
    -------
    pandas.DataFrame
        One row per (duplicate pair, metric), sorted by pair then
        metric.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.biology.response import (
    ...     load_response_table, replicate_variability,
    ... )
    >>> frame = pd.DataFrame({
    ...     "cell_line": ["A", "A", "B"],
    ...     "compound": ["d", "d", "d"],
    ...     "v": [1.0, 3.0, 10.0],
    ... })
    >>> screen = load_response_table(frame, metrics=["v"])
    >>> report = replicate_variability(screen)
    >>> report["n_replicates"].tolist()
    [2]
    >>> round(float(report["cv_pct"].iloc[0]), 1)
    70.7
    """
    columns = [
        screen.cell_col,
        screen.compound_col,
        "metric",
        "n_replicates",
        "median",
        "mean",
        "std",
        "spread",
        "cv_pct",
    ]
    rows: list[dict[str, object]] = []
    grouped = screen.frame.groupby(
        [screen.cell_col, screen.compound_col], sort=True, dropna=False
    )
    for (cell, compound), group in grouped:
        if len(group) < 2:
            continue  # not a duplicated pair
        for metric in screen.metrics:
            valid = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(valid)
            if n == 0:
                median = mean = std = spread = math.nan
            else:
                median = float(valid.median())
                mean = float(valid.mean())
                std = float(valid.std(ddof=1)) if n >= 2 else math.nan
                spread = float(valid.max() - valid.min())
            cv = _replicate_cv(group[metric])
            rows.append(
                {
                    screen.cell_col: cell,
                    screen.compound_col: compound,
                    "metric": metric,
                    "n_replicates": n,
                    "median": median,
                    "mean": mean,
                    "std": std,
                    "spread": spread,
                    "cv_pct": cv,
                }
            )
    report = pd.DataFrame(rows, columns=columns)
    if report.empty:
        _LOG.debug(
            "replicate_variability: no duplicated (cell, compound) pairs"
        )
    else:
        _LOG.info(
            "replicate_variability: %d duplicated pair(s) x %d metric(s)",
            report.groupby(
                [screen.cell_col, screen.compound_col], dropna=False
            ).ngroups,
            len(screen.metrics),
        )
    return report


def collapse_replicates(
    screen: ResponseScreen, add_cv: bool = False
) -> ResponseScreen:
    """Collapse duplicated (cell line, compound) pairs by the median.

    Every pair is reduced to a single row: metric columns take the
    median over the replicates, annotation columns (tissue, study ...)
    keep the first value.  Pairs without duplicates pass through
    untouched.  Rows whose cell-line or compound identifier is missing
    are kept as their own (NaN-keyed) group instead of being dropped
    (pandas' ``groupby`` default would silently discard them); a
    warning names the count of such rows.  The collapsed pairs are
    logged at DEBUG level.

    With ``add_cv=True`` each metric column gains a companion
    ``{metric}_replicate_cv_pct`` column (placed immediately to its
    right) holding the pre-collapse coefficient of variation of that
    pair's replicates -- NaN for singleton pairs, pairs with fewer
    than two valid metric values, or a zero mean (undefined ratio),
    mirroring :func:`replicate_variability`.  The extra columns are
    annotations, *not* metrics: the returned screen's ``metrics``
    list is unchanged, so downstream pivoting and summaries keep
    working.  A pre-existing column with a CV column's name raises
    :class:`DataValidationError` instead of silently overwriting it.

    Parameters
    ----------
    screen:
        The screen to collapse.
    add_cv:
        Append a ``{metric}_replicate_cv_pct`` column next to every
        metric column (default ``False`` -- byte-identical behaviour
        to previous releases).

    Returns
    -------
    ResponseScreen
        A new screen (the input is never mutated) with at most one row
        per pair, preserving first-appearance row order.
    """
    frame = screen.frame
    missing_ids = frame[screen.cell_col].isna() | frame[screen.compound_col].isna()
    if missing_ids.any():
        _LOG.warning(
            "%d row(s) have a missing cell-line/compound identifier -- kept "
            "as their own (NaN) group instead of being dropped "
            "(audit_response_table reports them)",
            int(missing_ids.sum()),
        )
    work = frame
    if add_cv:
        work = frame.copy()
        keys = [screen.cell_col, screen.compound_col]
        for metric in screen.metrics:
            cv_name = f"{metric}_replicate_cv_pct"
            if cv_name in work.columns:
                raise DataValidationError(
                    f"column {cv_name!r} already exists -- collapse_replicates "
                    "cannot attach the replicate CV column (drop or rename "
                    "the existing column first)"
                )
            cv_series = frame.groupby(keys, sort=False, dropna=False)[
                metric
            ].transform(_replicate_cv)
            # position against the *current* work columns: earlier CV
            # inserts have shifted the later metrics right already
            work.insert(
                work.columns.get_loc(metric) + 1, cv_name, cv_series
            )
    aggregations: dict[str, str] = {}
    for column in work.columns:
        aggregations[column] = (
            "median" if column in screen.metrics else "first"
        )
    collapsed = work.groupby(
        [screen.cell_col, screen.compound_col],
        sort=False,
        as_index=False,
        dropna=False,
    ).agg(aggregations)

    duplicate_pairs = screen._duplicate_pairs()
    if duplicate_pairs:
        shown = ", ".join(f"({cell} + {compound})" for cell, compound in duplicate_pairs[:10])
        extra = len(duplicate_pairs) - 10
        more = f" (+{extra} more)" if extra > 0 else ""
        _LOG.debug(
            "Collapsed %d duplicate pair(s) by median: %s%s",
            len(duplicate_pairs),
            shown,
            more,
        )
    if add_cv:
        n_cv = sum(
            1
            for column in collapsed.columns
            if column.endswith("_replicate_cv_pct")
        )
        _LOG.debug(
            "collapse_replicates: attached %d replicate CV column(s)", n_cv
        )
    _LOG.info(
        "collapse_replicates: %d row(s) -> %d row(s)", len(frame), len(collapsed)
    )
    return ResponseScreen(
        frame=collapsed,
        cell_col=screen.cell_col,
        compound_col=screen.compound_col,
        tissue_col=screen.tissue_col,
        metrics=list(screen.metrics),
    )


def ln_ic50_to_pic50(ln_ic50):
    """Convert GDSC ``LN_IC50`` values to pIC50.

    GDSC reports the natural log of the IC50 in micromolar, so
    ``LN_IC50 = 0`` (IC50 = 1 uM) maps to ``pIC50 = 6`` via
    ``pIC50 = 6 - ln(IC50_uM) / ln(10)``.

    Parameters
    ----------
    ln_ic50:
        Scalar, list/array or :class:`pandas.Series` of ``LN_IC50``
        values.  NaN passes through; a Series keeps its index.

    Returns
    -------
    Same kind as the input (float for scalars, ndarray for arrays,
    Series for Series) with the converted values.
    """
    if isinstance(ln_ic50, pd.Series):
        numeric = pd.to_numeric(ln_ic50, errors="coerce")
        return 6.0 - numeric / _LN10
    values = np.asarray(ln_ic50, dtype=float)
    result = 6.0 - values / _LN10
    if result.ndim == 0:
        return float(result)
    return result


def pic50_to_ln_ic50(pic50):
    """Convert pIC50 values back to GDSC ``LN_IC50`` (the exact inverse).

    ``ln(IC50_uM) = (6 - pIC50) * ln(10)``; a 1 uM IC50 (pIC50 6)
    maps back to ``LN_IC50 = 0``.  Accepts the same input kinds as
    :func:`ln_ic50_to_pic50` with NaN passthrough.
    """
    if isinstance(pic50, pd.Series):
        numeric = pd.to_numeric(pic50, errors="coerce")
        return (6.0 - numeric) * _LN10
    values = np.asarray(pic50, dtype=float)
    result = (6.0 - values) * _LN10
    if result.ndim == 0:
        return float(result)
    return result


def response_matrix(screen: ResponseScreen, metric: str) -> pd.DataFrame:
    """Pivot one metric into a cell-line x compound sensitivity matrix.

    Rows are cell lines, columns are compounds (both sorted); untested
    pairs hold NaN.  Duplicated pairs are *not* silently averaged --
    collapse them first with :func:`collapse_replicates`.

    Parameters
    ----------
    screen:
        The screen to pivot.
    metric:
        Metric column to place in the matrix.

    Returns
    -------
    pandas.DataFrame
        The pivoted matrix.

    Raises
    ------
    DataValidationError
        If ``metric`` is not one of the screen's metrics, or duplicate
        (cell line, compound) pairs are present.
    """
    if metric not in screen.metrics:
        raise DataValidationError(
            f"metric {metric!r} is not part of the screen -- available "
            f"metrics: {list(screen.metrics)}"
        )
    duplicates = screen._duplicate_pairs()
    if duplicates:
        shown = ", ".join(f"({cell} + {compound})" for cell, compound in duplicates[:5])
        extra = len(duplicates) - 5
        more = f" (+{extra} more)" if extra > 0 else ""
        raise DataValidationError(
            f"{len(duplicates)} duplicate (cell, compound) pair(s) prevent "
            f"pivoting: {shown}{more} -- collapse_replicates() first"
        )
    matrix = screen.frame.pivot(
        index=screen.cell_col, columns=screen.compound_col, values=metric
    )
    matrix.index.name = screen.cell_col
    matrix.columns.name = screen.compound_col
    return matrix.sort_index().sort_index(axis=1)


def summarize_compounds(
    screen: ResponseScreen,
    metric: str,
    higher_is_more_sensitive: bool = False,
) -> pd.DataFrame:
    """Per-compound potency statistics across the cell-line panel.

    Rows are compounds (sorted) with columns ``n_cell_lines`` (number of
    distinct cell lines measured, replicates counted once -- missing
    metric values are the audit's business), ``median``, ``min``,
    ``max``, ``std`` (sample standard deviation, ddof=1) and
    ``sensitive_spread`` = ``max - min`` -- the potency range across
    the panel, i.e. how selectively the compound hits the panel
    regardless of direction.  Rows whose compound identifier is missing
    form their own trailing ``NaN`` row instead of being dropped
    (:func:`audit_response_table` reports them).

    Parameters
    ----------
    screen:
        The screen to summarise.
    metric:
        Metric column to summarise.
    higher_is_more_sensitive:
        Direction of ``metric``: ``False`` (default) for LN_IC50-style
        metrics where *lower* means more potent; ``True`` for
        pIC50-style metrics where *higher* means more potent.  It only
        affects the ``most_sensitive_cell_line`` column -- the cell
        line with the strongest response under that direction.

    Raises
    ------
    DataValidationError
        If ``metric`` is not one of the screen's metrics, or the frame's
        row index holds duplicated labels -- the positional
        ``idxmin``/``idxmax`` re-alignment would crash there; call
        ``screen.frame.reset_index(drop=True)`` (or reload the table)
        first.
    """
    if metric not in screen.metrics:
        raise DataValidationError(
            f"metric {metric!r} is not part of the screen -- available "
            f"metrics: {list(screen.metrics)}"
        )
    if screen.frame.index.has_duplicates:
        raise DataValidationError(
            "the screen frame has duplicated row index labels -- "
            "summarize_compounds aligns its picks by label; call "
            "screen.frame.reset_index(drop=True) (or reload the table) "
            "first"
        )
    grouped = screen.frame.groupby(
        screen.compound_col, sort=True, dropna=False
    )[metric]
    summary = pd.DataFrame(
        {
            # distinct cell lines (replicates counted once); missing
            # metric values stay the audit's job to report.
            "n_cell_lines": screen.frame.groupby(
                screen.compound_col, sort=True, dropna=False
            )[screen.cell_col].nunique(),
            "median": grouped.median(),
            "min": grouped.min(),
            "max": grouped.max(),
            "std": grouped.std(),
        }
    )
    summary["sensitive_spread"] = summary["max"] - summary["min"]

    # Direction-aware pick of the strongest-responding cell line per
    # compound (NaN metric rows cannot win by construction).
    lookup = screen.frame[
        [screen.compound_col, screen.cell_col, metric]
    ].dropna(subset=[metric])
    if higher_is_more_sensitive:
        positions = lookup.groupby(
            screen.compound_col, sort=True, dropna=False
        )[metric].idxmax()
    else:
        positions = lookup.groupby(
            screen.compound_col, sort=True, dropna=False
        )[metric].idxmin()
    # ``positions`` maps compound -> winning row label; re-align the picked
    # cell lines onto the compound index before assigning into the summary.
    summary["most_sensitive_cell_line"] = pd.Series(
        lookup.loc[positions, screen.cell_col].to_numpy(), index=positions.index
    )
    summary = summary[
        [
            "n_cell_lines",
            "median",
            "min",
            "max",
            "std",
            "sensitive_spread",
            "most_sensitive_cell_line",
        ]
    ]
    summary.index.name = screen.compound_col
    return summary


def call_sensitive_resistant(
    response_series,
    sensitive_quantile: float = 0.25,
    resistant_quantile: float = 0.75,
    lower_is_sensitive: bool = True,
):
    """Threshold a response vector into sensitivity calls.

    The lowest ``sensitive_quantile`` fraction of cells is called
    ``"sensitive"``, the highest ``resistant_quantile`` fraction
    ``"resistant"`` and everything in between ``"intermediate"``
    (thresholds are inclusive, so tied values stay on the boundary
    side).  NaN entries are labelled ``"not tested"``.

    Parameters
    ----------
    response_series:
        Response values, typically one compound's column of a
        :func:`response_matrix` pivot.
    sensitive_quantile, resistant_quantile:
        Quantile cuts; must satisfy
        ``0 < sensitive_quantile < resistant_quantile < 1``.
    lower_is_sensitive:
        Direction of the metric: ``True`` (default) for LN_IC50-style
        metrics where *lower* means more sensitive; ``False`` for
        pIC50-style metrics where *higher* means more sensitive (the
        quantile cuts are mirrored accordingly).

    Returns
    -------
    pandas.Series
        Labels ``"sensitive"`` / ``"resistant"`` / ``"intermediate"``
        / ``"not tested"``, indexed like the input.

    Raises
    ------
    DataValidationError
        If the quantile ordering is violated.
    """
    if not (
        0.0 < sensitive_quantile < resistant_quantile < 1.0
    ):
        raise DataValidationError(
            "quantiles must satisfy 0 < sensitive_quantile < "
            f"resistant_quantile < 1, got {sensitive_quantile} and "
            f"{resistant_quantile}"
        )
    series = (
        response_series
        if isinstance(response_series, pd.Series)
        else pd.Series(response_series)
    )
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series("not tested", index=numeric.index, dtype=object)

    valid = numeric.dropna()
    if valid.empty:
        _LOG.warning(
            "call_sensitive_resistant: no non-NaN values -- every cell line "
            "labelled 'not tested'"
        )
        return labels

    if lower_is_sensitive:
        sensitive_cut = valid.quantile(sensitive_quantile)
        resistant_cut = valid.quantile(resistant_quantile)
        is_sensitive = numeric <= sensitive_cut
        is_resistant = numeric >= resistant_cut
    else:
        # High values are the sensitive ones: mirror the quantile cuts.
        sensitive_cut = valid.quantile(1.0 - sensitive_quantile)
        resistant_cut = valid.quantile(1.0 - resistant_quantile)
        is_sensitive = numeric >= sensitive_cut
        is_resistant = numeric <= resistant_cut

    labels[is_sensitive] = "sensitive"
    labels[is_resistant & ~is_sensitive] = "resistant"
    intermediate = numeric.notna() & ~is_sensitive & ~is_resistant
    labels[intermediate] = "intermediate"
    _LOG.debug(
        "Sensitivity calls: %d sensitive, %d resistant, %d intermediate, "
        "%d not tested",
        int((labels == "sensitive").sum()),
        int((labels == "resistant").sum()),
        int((labels == "intermediate").sum()),
        int((labels == "not tested").sum()),
    )
    return labels

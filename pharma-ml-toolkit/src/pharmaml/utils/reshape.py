"""Validated table reshaping for the Datasets workspace (part8 r9).

The data-handbook audit (``docs/COVERAGE.md`` rows 26-33) documented
seven pandas-reshaping surfaces as STRETCH -- combine, merge, groupby
summary, pivot, string cleaning, resample/rolling and filter
expressions.  This module is their backend: every operation takes
loaded frames, validates its contract up front, returns the new frame
plus a small report so the GUI can show exactly what happened without
re-deriving it.

Design rules (part8 §1):
* No silent behaviour: every operation reports counts, key columns and
  drop/keep decisions; mismatches raise ``ValueError`` with the exact
  problem named rather than producing an empty frame.
* Index identity is preserved: ``concat``/``merge`` keep a stable
  ``source_row``-style integer index unless the user asks for a key
  index; ``groupby``/``pivot`` report their key structure explicitly.
* Expressions (``query``) never ``eval`` arbitrary Python: they go
  through pandas' own ``df.query`` parser with ``engine='python'``
  (no numexpr code paths, no ``local_dict`` escaping).
* Time series (``resample``/``rolling``) are pharma-honest: a
  datetime column is required, the rule string is validated against
  pandas' own offset aliases, and the result carries the window
  description so the GUI can label it.

The functions are deliberately small and composable -- the GUI card
binds one function per button, and the tests pin each contract
independently (``tests/test_reshape.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import keyword
import re

import pandas as pd

__all__ = [
    "ReshapeReport",
    "combine_frames",
    "merge_frames",
    "groupby_summary",
    "pivot_table",
    "clean_strings",
    "resample_timeseries",
    "rolling_window",
    "filter_frame",
    "SUPPORTED_AGGREGATIONS",
]


#: The groupby/pivot aggregations the GUI offers -- exactly the ones
#: tested below; anything else raises at the contract check.
SUPPORTED_AGGREGATIONS: tuple[str, ...] = (
    "mean", "median", "std", "min", "max", "count", "sum", "nunique",
)


@dataclass(frozen=True)
class ReshapeReport:
    """What one reshaping operation did, for the GUI status line.

    Every field is filled by the operation itself -- the GUI never
    guesses counts from the output frame's shape (a merge that drops
    everything and a filter that keeps everything would otherwise look
    identical to a lazy caller).
    """

    operation: str
    rows_in: int
    rows_out: int
    columns_in: int
    columns_out: int
    detail: str = ""
    dropped: tuple[str, ...] = field(default_factory=tuple)

    def summary_line(self) -> str:
        """One-line human summary (the card's status text)."""
        base = (
            f"{self.operation}: {self.rows_in:,}×{self.columns_in} → "
            f"{self.rows_out:,}×{self.columns_out}"
        )
        if self.detail:
            base += f" — {self.detail}"
        return base


def _require_frame(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")
    if frame.empty:
        raise ValueError(f"{name} is empty — load a dataset first")


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing column(s): {', '.join(missing)} — "
            f"available: {', '.join(map(str, frame.columns[:12]))}"
            + (" …" if len(frame.columns) > 12 else "")
        )


# -- 03.06 concat ----------------------------------------------------------


def combine_frames(
    frames: list[pd.DataFrame],
    *,
    axis: Literal[0, 1] = 0,
    keys: list[str] | None = None,
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Stack (``axis=0``) or side-by-side (``axis=1``) loaded frames.

    Rows keep their source identity: stacking adds a ``source_frame``
    column naming which input each row came from (the replicate-assay
    CSV use case), unless ``keys`` is given -- then it carries the
    user's per-frame labels instead.  Side-by-side concatenation
    requires equal row counts (the mismatch is named per frame, never
    silently padded).
    """
    if not frames:
        raise ValueError("no frames to combine — load at least one dataset")
    for i, f in enumerate(frames):
        _require_frame(f, f"frame {i + 1}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 (stack rows) or 1 (side-by-side), got {axis!r}")
    labels = list(keys) if keys is not None else [f"frame_{i + 1}" for i in range(len(frames))]
    if len(labels) != len(frames):
        raise ValueError(
            f"{len(labels)} key labels given for {len(frames)} frames — one per frame"
        )

    rows_in = sum(len(f) for f in frames)
    columns_in = frames[0].shape[1]

    if axis == 0:
        out = pd.concat(
            [f.assign(source_frame=label) for f, label in zip(frames, labels)],
            ignore_index=True,
        )
        detail = (
            f"stacked {len(frames)} frames "
            f"({', '.join(f'{l}: {len(f):,}' for l, f in zip(labels, frames))})"
        )
    else:
        lengths = {label: len(f) for label, f in zip(labels, frames)}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                "side-by-side combine needs equal row counts — "
                + ", ".join(f"{k}: {v:,}" for k, v in lengths.items())
            )
        out = pd.concat(frames, axis=1, ignore_index=True)
        detail = f"joined {len(frames)} frames side-by-side ({rows_in:,} rows)"

    report = ReshapeReport(
        operation="combine",
        rows_in=rows_in,
        rows_out=len(out),
        columns_in=columns_in,
        columns_out=out.shape[1],
        detail=detail,
    )
    return out, report


# -- 03.07 merge -----------------------------------------------------------


def merge_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    how: Literal["inner", "left", "outer"] = "left",
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Merge two loaded tables on one key column.

    ``how`` is deliberately restricted to inner/left/outer -- the three
    joins with unambiguous pharma meaning (shared compounds / keep-A /
    keep-everything).  Duplicate key values are reported, not silently
    multiplied: a many-to-many merge names the fan-out factor.
    """
    _require_frame(left, "left frame")
    _require_frame(right, "right frame")
    _require_columns(left, [on], "left frame")
    _require_columns(right, [on], "right frame")
    if how not in ("inner", "left", "outer"):
        raise ValueError(f"how must be inner, left or outer, got {how!r}")

    left_dups = int(left[on].duplicated().sum())
    right_dups = int(right[on].duplicated().sum())
    out = pd.merge(left, right, on=on, how=how, sort=False)

    parts = [f"{how} merge on '{on}'"]
    matched = int(out[on].notna().sum()) if how != "inner" else len(out)
    parts.append(f"{matched:,} matched rows")
    if left_dups or right_dups:
        parts.append(
            f"warning: many-to-many fan-out (left dups {left_dups:,}, "
            f"right dups {right_dups:,}) — verify the key is unique per table"
        )
    overlap = sorted(set(left[on].dropna()) & set(right[on].dropna()))
    parts.append(f"{len(overlap):,} shared key values")

    report = ReshapeReport(
        operation="merge",
        rows_in=len(left) + len(right),
        rows_out=len(out),
        columns_in=left.shape[1] + right.shape[1],
        columns_out=out.shape[1],
        detail="; ".join(parts),
    )
    return out, report


# -- 03.08 groupby ---------------------------------------------------------


def groupby_summary(
    frame: pd.DataFrame,
    *,
    by: list[str],
    aggregations: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Per-group aggregate statistics (split-apply-combine).

    ``aggregations`` maps column → list of supported aggregation
    names (see :data:`SUPPORTED_AGGREGATIONS`); the default summarizes
    every numeric column with mean/std/min/max/count.  The result's
    index is reset so the GUI table renders key columns as ordinary
    columns, and the report names the group count and columns touched.
    """
    _require_frame(frame, "frame")
    _require_columns(frame, by, "frame")
    if not by:
        raise ValueError("groupby needs at least one key column")

    if aggregations is None:
        numeric = list(frame.select_dtypes("number").columns)
        aggregations = {c: ["mean", "std", "min", "max", "count"] for c in numeric}
    if not aggregations:
        raise ValueError("no columns to aggregate — pass aggregations or include numeric columns")
    for column, aggs in aggregations.items():
        _require_columns(frame, [column], "frame")
        bad = [a for a in aggs if a not in SUPPORTED_AGGREGATIONS]
        if bad:
            raise ValueError(
                f"unsupported aggregation(s) for '{column}': {', '.join(bad)} — "
                f"supported: {', '.join(SUPPORTED_AGGREGATIONS)}"
            )

    grouped = frame.groupby(by, dropna=False).agg(aggregations)
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    out = grouped.reset_index()

    report = ReshapeReport(
        operation="groupby",
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=frame.shape[1],
        columns_out=out.shape[1],
        detail=f"{len(out):,} groups over {', '.join(by)}; "
        f"{sum(len(a) for a in aggregations.values())} statistics",
    )
    return out, report


# -- 03.09 pivot -----------------------------------------------------------


def pivot_table(
    frame: pd.DataFrame,
    *,
    index: list[str],
    columns: str,
    values: str,
    aggfunc: str = "mean",
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Reshape long → wide with a validated aggregation.

    The pivot card's engine: rows are ``index`` keys, one column per
    ``columns`` value, cells aggregate ``values``.  Missing
    combinations stay NaN (reported), never zero-filled -- a zero
    would fabricate a measurement.
    """
    _require_frame(frame, "frame")
    _require_columns(frame, [*index, columns, values], "frame")
    if not index:
        raise ValueError("pivot needs at least one index (row) column")
    if aggfunc not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"unsupported aggregation {aggfunc!r} — "
            f"supported: {', '.join(SUPPORTED_AGGREGATIONS)}"
        )

    out = pd.pivot_table(
        frame, index=index, columns=columns, values=values, aggfunc=aggfunc,
        dropna=False,
    )
    out.columns = [f"{values}_{c}" for c in out.columns]
    out = out.reset_index()
    n_missing = int(out.isna().sum().sum())

    report = ReshapeReport(
        operation="pivot",
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=frame.shape[1],
        columns_out=out.shape[1],
        detail=(
            f"{len(out):,} rows × {out.shape[1] - len(index)} value columns "
            f"({aggfunc} of '{values}' by '{columns}')"
            + (f"; {n_missing:,} empty combinations left NaN" if n_missing else "")
        ),
    )
    return out, report


# -- 03.10 string cleaning ---------------------------------------------------


def clean_strings(
    frame: pd.DataFrame,
    *,
    column: str,
    strip: bool = True,
    lowercase: bool = False,
    uppercase: bool = False,
    collapse_whitespace: bool = True,
    drop_missing: bool = False,
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Vectorized cleaning of one text column (SMILES/categoricals).

    A copy is returned -- the input frame is never mutated (the
    workspace's shared frame stays pristine for other cards).  Reports
    how many values actually changed and how many were missing
    before ``drop_missing`` acted.
    """
    _require_frame(frame, "frame")
    _require_columns(frame, [column], "frame")
    if lowercase and uppercase:
        raise ValueError("lowercase and uppercase cannot both be requested")
    series = frame[column]
    if not (strip or lowercase or uppercase or collapse_whitespace or drop_missing):
        raise ValueError("nothing to do — enable at least one cleaning option")

    n_missing_in = int(series.isna().sum())
    cleaned = series.astype("string")
    changed = 0
    if strip or collapse_whitespace:
        cleaned = cleaned.str.strip()
        if collapse_whitespace:
            cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
        changed = int((cleaned.fillna("") != series.astype("string").fillna("")).sum())
    if lowercase:
        cleaned = cleaned.str.lower()
    if uppercase:
        cleaned = cleaned.str.upper()

    out = frame.copy()
    out[column] = cleaned.astype(object).where(cleaned.notna(), None)
    if drop_missing:
        out = out[out[column].notna()]

    options = [
        name for name, on in [
            ("strip", strip), ("collapse whitespace", collapse_whitespace),
            ("lowercase", lowercase), ("uppercase", uppercase),
        ] if on
    ]
    detail = f"'{column}': {', '.join(options) or 'no transform'}"
    if changed:
        detail += f"; {changed:,} values changed"
    if n_missing_in:
        detail += (
            f"; {n_missing_in:,} missing dropped" if drop_missing
            else f"; {n_missing_in:,} missing values left untouched"
        )

    report = ReshapeReport(
        operation="string-clean",
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=frame.shape[1],
        columns_out=out.shape[1],
        detail=detail,
    )
    return out, report


# -- 03.11 resample / rolling ------------------------------------------------


def resample_timeseries(
    frame: pd.DataFrame,
    *,
    time_column: str,
    value_columns: list[str],
    rule: str,
    aggregation: str = "mean",
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Aggregate a time series onto a fixed frequency grid.

    The time column must be datetime-convertible (the parse failure
    names the offending rows).  The rule is validated through pandas'
    own offset alias machinery -- a typo like ``"5 minn"`` raises
    before any data is touched.
    """
    _require_frame(frame, "frame")
    _require_columns(frame, [time_column, *value_columns], "frame")
    if not value_columns:
        raise ValueError("resample needs at least one value column")
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"unsupported aggregation {aggregation!r} — "
            f"supported: {', '.join(SUPPORTED_AGGREGATIONS)}"
        )
    try:
        pd.tseries.frequencies.to_offset(rule)
    except ValueError as exc:
        raise ValueError(f"invalid resample rule {rule!r}: {exc}") from exc

    times = pd.to_datetime(frame[time_column], errors="coerce")
    bad = frame.index[times.isna()].tolist()
    if bad:
        preview = ", ".join(str(b) for b in bad[:5])
        raise ValueError(
            f"'{time_column}' has non-datetime values at row(s) {preview}"
            + (" …" if len(bad) > 5 else "")
        )

    indexed = frame.assign(__time__=times).set_index("__time__")
    resampled = indexed[value_columns].resample(rule).agg(aggregation)
    out = resampled.reset_index().rename(columns={"__time__": time_column})
    n_empty = int(out[value_columns].isna().all(axis=1).sum())

    report = ReshapeReport(
        operation="resample",
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=frame.shape[1],
        columns_out=out.shape[1],
        detail=(
            f"'{value_columns[0]}'{' …' if len(value_columns) > 1 else ''} "
            f"→ {aggregation} per {rule}"
            + (f"; {n_empty:,} empty windows (no data in period)" if n_empty else "")
        ),
    )
    return out, report


def rolling_window(
    frame: pd.DataFrame,
    *,
    time_column: str,
    value_columns: list[str],
    window: int,
    aggregation: str = "mean",
    min_periods: int = 1,
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Rolling (moving) window statistics, time-sorted first.

    The frame is sorted by the time column before the window slides --
    an unsorted upload must not produce a scrambled smooth.  The
    output carries the original columns plus one ``{col}_rolling_{agg}``
    column per value column; rows are never dropped.
    """
    _require_frame(frame, "frame")
    _require_columns(frame, [time_column, *value_columns], "frame")
    if not value_columns:
        raise ValueError("rolling needs at least one value column")
    if aggregation not in ("mean", "median", "std", "min", "max", "sum"):
        raise ValueError(
            f"unsupported rolling aggregation {aggregation!r} — "
            "supported: mean, median, std, min, max, sum"
        )
    if window < 2:
        raise ValueError(f"window must be ≥ 2, got {window}")
    if min_periods < 1:
        raise ValueError(f"min_periods must be ≥ 1, got {min_periods}")

    times = pd.to_datetime(frame[time_column], errors="coerce")
    bad = frame.index[times.isna()].tolist()
    if bad:
        preview = ", ".join(str(b) for b in bad[:5])
        raise ValueError(
            f"'{time_column}' has non-datetime values at row(s) {preview}"
            + (" …" if len(bad) > 5 else "")
        )

    out = frame.assign(**{time_column: times}).sort_values(time_column).reset_index(drop=True)
    for column in value_columns:
        out[f"{column}_rolling_{aggregation}"] = (
            out[column].rolling(window=window, min_periods=min_periods).agg(aggregation)
        )

    report = ReshapeReport(
        operation="rolling",
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=frame.shape[1],
        columns_out=out.shape[1],
        detail=(
            f"{window}-point {aggregation} window over "
            f"'{value_columns[0]}'{' …' if len(value_columns) > 1 else ''} "
            "(time-sorted, no rows dropped)"
        ),
    )
    return out, report


# -- 03.12 filter -------------------------------------------------------------


def filter_frame(
    frame: pd.DataFrame,
    expression: str,
    *,
    _sentinel: None = None,
) -> tuple[pd.DataFrame, ReshapeReport]:
    """Keep the rows matching a pandas ``query`` expression.

    The expression runs through ``DataFrame.query`` with the python
    engine only -- it is pandas' own expression grammar (column names,
    comparisons, ``and``/``or``/``in``), never arbitrary Python
    ``eval``.  A syntax error surfaces pandas' message verbatim so the
    user can fix the expression; an expression naming an unknown
    column is caught by the same parser.
    """
    _require_frame(frame, "frame")
    if not expression.strip():
        raise ValueError("empty filter expression")

    # Columns named after Python keywords (pharma frames do carry a
    # ``yield`` column) are pandas' documented backtick-escape case:
    # rewrite bare keyword tokens into ```name``` before the query
    # parser sees them.  Only tokens that are BOTH keywords and actual
    # column names are escaped -- ``in``/``and``/``or`` in operator
    # position are grammar, never columns, and mangling them breaks
    # the parse.  Already-backticked names are left alone.
    columns = {str(c) for c in frame.columns}
    keyword_tokens = sorted(
        {
            t
            for t in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression)
            if keyword.iskeyword(t) and t in columns
        }
    )
    prepared = expression
    for token in keyword_tokens:
        prepared = re.sub(rf"(?<!`)\b{re.escape(token)}\b(?!`)", f"`{token}`", prepared)

    try:
        out = frame.query(prepared, engine="python").reset_index(drop=True)
    except Exception as exc:  # pandas raises several syntax-error types
        raise ValueError(f"invalid filter expression {expression!r}: {exc}") from exc
    if len(out) == len(frame):
        detail = "expression matched every row (no rows removed)"
    elif len(out) == 0:
        detail = "expression matched no rows — the output frame is empty"
    else:
        detail = f"kept {len(out):,} of {len(frame):,} rows"

    report = ReshapeReport(
        operation="filter",
        rows_in=len(frame),
        rows_out=len(out),
        columns_in=frame.shape[1],
        columns_out=out.shape[1],
        detail=detail + f"; expression: {expression}",
    )
    return out, report

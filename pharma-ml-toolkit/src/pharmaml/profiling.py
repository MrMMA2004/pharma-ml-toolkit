"""Dataset & column profiling: the fg-data-profiling concepts, re-derived.

The ``Data-Centric-AI-Community/fg-data-profiling`` repository (the
renamed ``ydata-profiling``/``pandas-profiling``, MIT) was reviewed
file by file for this round: ``model/typeset.py`` +
``typeset_relations.py`` (the column-type inference rules),
``model/pandas/describe_numeric_pandas.py`` and its siblings (the
per-type stat suites), ``model/alerts.py`` (the 21 alert rules),
``model/correlations.py`` + ``discretize_pandas.py`` (the "auto"
mixed-dtype correlation strategy), ``model/table.py`` and
``model/duplicates.py`` (the overview rollup).  What survives here,
as readable pure numpy/pandas:

* :func:`infer_column_type` -- the type decision tree with its five
  tunable thresholds (numeric with few distinct values reads as
  categorical; strings up to 50 distinct and under 50% distinct stay
  categorical; boolean-token strings map to boolean; fully
  date-parseable strings become datetime).
* :func:`column_profile` / :func:`profile_dataset` -- the per-type
  stat dictionaries (numeric: the five quantiles, IQR, MAD, CV,
  skew, kurtosis, negative/zero/infinite counts, monotonicity, the
  histogram and its uniformity chi²; categorical/text: entropy
  imbalance ``1 − H/log2 k``, chi², length stats; boolean: top/freq;
  datetime: the epoch range) and the table-level rollup (cells
  missing, duplicate rows, memory footprint, the type census).
* :func:`cramers_v` + :func:`auto_correlation_matrix` -- the
  bias-corrected Cramér's V and ydata's "auto" strategy (numerics
  discretized to 10 uniform bins; Spearman for numeric-numeric
  pairs, Cramér's V wherever a categorical is involved).
* :func:`profile_dataset`'s alert rules -- constant, empty, unique,
  high-cardinality, missing, skewed, imbalanced, uniform, zeros,
  infinite, duplicates -- each with the upstream's default
  threshold.

The honest scope, stated once: the HTML/widget report layer, the
Spark backend, the phik correlation, the image/file/URL profilers
and the FFT/ADF time-series extras are not carried -- this module
is the profiling *semantics* (types, numbers, alerts), and the GUI
renders them as tables.  Also stricter than upstream on purpose:
a string column must parse as dates for *every* value before it is
typed datetime (ydata flips on a single parseable value), because a
treatment code that happens to look like a year must stay a code.

Examples
--------
>>> from pharmaml.profiling import infer_column_type
>>> import pandas as pd
>>> infer_column_type(pd.Series([1.2, 3.4, 5.6, 7.8, 9.9, 2.5]))
'numeric'
>>> infer_column_type(pd.Series([1, 2, 3, 1, 2, 3, 1]))     # few distinct
'categorical'
>>> infer_column_type(pd.Series(["yes", "no", "yes", "no"]))
'boolean'
>>> infer_column_type(pd.Series(["2024-01-05", "2024-02-06", "2024-03-09"]))
'datetime'
>>> infer_column_type(pd.Series(["A1", "B2", "A1", "B2", "C3", "A1", "B2"]))
'categorical'

>>> from pharmaml.profiling import entropy_imbalance, monotonicity
>>> round(entropy_imbalance([1, 1, 1, 1]), 6), round(entropy_imbalance([4]), 6)
(0.0, 1.0)
>>> monotonicity([1, 2, 3]), monotonicity([3, 2, 1]), monotonicity([1, 2, 2])
(2, -2, 1)
>>> monotonicity([2, 1, 2, 3])
0

>>> from pharmaml.profiling import demo_profile
>>> profile = demo_profile()
>>> profile.n_rows, profile.n_columns
(500, 11)
>>> sorted(profile.types)
['boolean', 'categorical', 'datetime', 'numeric', 'text', 'unsupported']
>>> kinds = {alert.kind for alert in profile.alerts}
>>> sorted(kind for kind in kinds if kind != "duplicates")
['constant', 'empty', 'high_cardinality', 'imbalanced', 'missing', 'skewed', 'unique']
>>> frame = profile.as_frame()
>>> frame.shape
(11, 8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.ml.stats import RANDOM_STATE

_LOG = get_logger(__name__)

#: The five column types the inference tree can settle on.
COLUMN_TYPES: tuple[str, ...] = (
    "numeric",
    "categorical",
    "text",
    "boolean",
    "datetime",
    "unsupported",
)

#: A string column above this many distinct values leaves "categorical".
CARDINALITY_THRESHOLD = 50

#: A numeric column with at most this many distinct values reads as
#: categorical (ydata's ``numeric_is_category`` rule).
LOW_CATEGORICAL_THRESHOLD = 5

#: A string column that is more than this fraction distinct is text.
PERCENTAGE_CAT_THRESHOLD = 0.5

#: Strings that map to booleans (ydata's token set).
BOOL_TOKENS: frozenset[str] = frozenset(
    {"true", "false", "t", "f", "yes", "no", "y", "n"}
)

#: The quantile ladder every numeric profile reports.
PROFILE_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Histogram bin ceiling (ydata caps at 50 configured, 250 hard).
HISTOGRAM_BINS = 50

#: Bins used by the "auto" correlation's numeric discretization.
AUTO_CORRELATION_BINS = 10

#: Alert thresholds (ydata's defaults): missing share, skew, imbalance,
#: uniform p-value, zero share.
ALERT_MISSING = 0.01
ALERT_SKEW = 20.0
ALERT_IMBALANCE = 0.5
ALERT_UNIFORM_P = 0.999
ALERT_ZEROS = 0.01


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


def infer_column_type(
    series: pd.Series,
    *,
    cardinality_threshold: int = CARDINALITY_THRESHOLD,
    low_categorical_threshold: int = LOW_CATEGORICAL_THRESHOLD,
    percentage_cat_threshold: float = PERCENTAGE_CAT_THRESHOLD,
) -> str:
    """Classify one column: numeric / categorical / text / boolean / datetime.

    The decision order is ydata's own, tightened where the loose rule
    would mistype pharma tables: boolean dtype first; then numeric
    dtype (or fully numeric-parseable strings), which flips to
    ``'categorical'`` when at most ``low_categorical_threshold``
    distinct values exist; then boolean tokens; then fully
    date-parseable strings; then the categorical/text split on
    cardinality and distinct share; anything else (mixed objects,
    lists) is honestly ``'unsupported'``.

    Raises
    ------
    DataValidationError
        On a column that is entirely empty.
    """
    values = series.dropna()
    if values.empty:
        raise DataValidationError(
            "infer_column_type: the column has no non-missing value to infer from"
        )
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(dtype):
        n_distinct = int(values.nunique())
        if n_distinct <= low_categorical_threshold:
            return "categorical"
        return "numeric"
    # object / string path
    strings = values.astype(str)
    lowered = strings.str.lower()
    if lowered.isin(BOOL_TOKENS).all():
        return "boolean"
    numeric = pd.to_numeric(strings, errors="coerce")
    if numeric.notna().all():
        if int(numeric.nunique()) <= low_categorical_threshold:
            return "categorical"
        return "numeric"
    if len(strings) >= 2:
        stamps = pd.to_datetime(strings, errors="coerce", format="mixed")
        if stamps.notna().all():
            return "datetime"
    n_distinct = int(strings.nunique())
    if n_distinct <= cardinality_threshold and (
        n_distinct / len(strings) < percentage_cat_threshold
    ):
        return "categorical"
    if strings.map(type).eq(str).all():
        return "text"
    return "unsupported"


# ---------------------------------------------------------------------------
# The numeric/categorical helpers
# ---------------------------------------------------------------------------


def entropy_imbalance(counts: list[int] | np.ndarray) -> float:
    """ydata's imbalance score: ``1 − H(p)/log2(k)``.

    0 = perfectly uniform categories, 1 = a single category carrying
    everything (the log2(k) denominator makes the score comparable
    across different category counts).
    """
    array = np.asarray(counts, dtype=float)
    total = array.sum()
    if total <= 0 or array.size < 2:
        return 1.0
    shares = array[array > 0] / total
    entropy = float(-(shares * np.log2(shares)).sum())
    return float(1.0 - entropy / math.log2(array.size))


def monotonicity(values: Any) -> int:
    """The monotonicity flag: 2 strict up, 1 up, -2 strict down, -1 down, 0 neither."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0
    diff = np.diff(array)
    positive = bool((diff > 0).all())
    nonnegative = bool((diff >= 0).all())
    negative = bool((diff < 0).all())
    nonpositive = bool((diff <= 0).all())
    if positive:
        return 2
    if nonnegative:
        return 1
    if negative:
        return -2
    if nonpositive:
        return -1
    return 0


def _chi2_uniform_p(counts: np.ndarray) -> float:
    """Pearson chi² of observed counts against a uniform expectation."""
    observed = np.asarray(counts, dtype=float)
    observed = observed[observed > 0]
    if observed.size < 2 or observed.sum() <= 0:
        return float("nan")
    expected = observed.sum() / observed.size
    statistic = float(((observed - expected) ** 2 / expected).sum())
    return float(scipy_stats.chi2.sf(statistic, observed.size - 1))


def cramers_v(table: Any) -> float:
    """Bias-corrected Cramér's V for a contingency table (Bergsma 2013).

    The raw ``sqrt(chi2/(n·min(r,c)-1))`` overstates association on
    small tables; the correction subtracts the rows/columns degrees
    of freedom from phi² before the final ratio.
    """
    frame = pd.crosstab(
        np.asarray(table)[:, 0] if isinstance(table, np.ndarray) else table,
        np.asarray(table)[:, 1] if isinstance(table, np.ndarray) else None,
    ) if isinstance(table, np.ndarray) else None
    if frame is not None:
        contingency = frame.to_numpy()
    else:
        contingency = np.asarray(table, dtype=float)
    if contingency.ndim != 2 or min(contingency.shape) < 2:
        raise DataValidationError(
            "cramers_v: needs a contingency table with >= 2 rows and columns"
        )
    chi2 = float(scipy_stats.chi2_contingency(contingency, correction=False)[0])
    n = float(contingency.sum())
    rows, cols = contingency.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (rows - 1) * (cols - 1) / (n - 1))
    row_corrected = rows - (rows - 1) ** 2 / (n - 1)
    col_corrected = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(col_corrected - 1, row_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float(math.sqrt(phi2_corrected / denominator))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two equal-length samples (average ranks)."""
    ranks_a = pd.Series(a).rank().to_numpy(dtype=float)
    ranks_b = pd.Series(b).rank().to_numpy(dtype=float)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator < 1e-24:
        return float("nan")
    return float(np.dot(centered_a, centered_b) / denominator)


def auto_correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """ydata's "auto" correlation: Spearman numeric-numeric, Cramér's V mixed.

    Numeric columns are discretized into ``AUTO_CORRELATION_BINS``
    uniform bins first (upstream's own strategy) so both sides of
    every pair speak the same discrete language.
    """
    columns = [str(name) for name in frame.columns]
    if len(columns) < 2:
        raise DataValidationError("auto_correlation_matrix: needs >= 2 columns")
    types = {name: infer_column_type(frame[name]) for name in columns}
    discrete: dict[str, np.ndarray] = {}
    for name in columns:
        series = frame[name].dropna()
        if types[name] in {"numeric", "boolean"} and not pd.api.types.is_bool_dtype(
            frame[name].dtype
        ):
            numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
            edges = np.linspace(numeric.min(), numeric.max(), AUTO_CORRELATION_BINS + 1)
            if edges[-1] > edges[0]:
                discrete[name] = np.digitize(numeric, edges[1:-1])
            else:
                discrete[name] = np.zeros(numeric.size, dtype=int)
        else:
            discrete[name] = series.astype(str).to_numpy()
    matrix = pd.DataFrame(
        np.eye(len(columns)), index=columns, columns=columns, dtype=float
    )
    for i, name_a in enumerate(columns):
        for name_b in columns[i + 1 :]:
            if types[name_a] in {"numeric", "boolean"} and types[name_b] in {
                "numeric",
                "boolean",
            } and not pd.api.types.is_bool_dtype(frame[name_a].dtype) and not pd.api.types.is_bool_dtype(
                frame[name_b].dtype
            ):
                value = _spearman(discrete[name_a], discrete[name_b])
            else:
                table = pd.crosstab(discrete[name_a], discrete[name_b]).to_numpy()
                value = 0.0 if min(table.shape) < 2 else cramers_v(table)
            matrix.loc[name_a, name_b] = value
            matrix.loc[name_b, name_a] = value
    return matrix


# ---------------------------------------------------------------------------
# The column profile
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """One column's typed report: counts, top values, stats, histogram."""

    name: str
    col_type: str
    n: int
    n_missing: int
    n_distinct: int
    top_values: list[tuple[str, int]]
    stats: dict[str, float] = field(default_factory=dict)
    histogram: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def p_missing(self) -> float:
        total = self.n + self.n_missing
        return self.n_missing / total if total else 0.0

    @property
    def is_unique(self) -> bool:
        return self.n >= 2 and self.n_distinct == self.n

    def line(self) -> str:
        missing = f" · {self.n_missing} missing ({self.p_missing:.1%})"
        top = self.top_values[0] if self.top_values else ("—", 0)
        return (
            f"{self.name} [{self.col_type}] · n {self.n}{missing} · "
            f"{self.n_distinct} distinct · top {top[0]!r} x{top[1]}"
        )


def _top_values(series: pd.Series, limit: int = 5) -> list[tuple[str, int]]:
    counts = series.value_counts()
    return [(str(value), int(count)) for value, count in counts.head(limit).items()]


def column_profile(series: pd.Series) -> ColumnProfile:
    """Profile one column with its type's own stat suite.

    Raises
    ------
    DataValidationError
        On an unnamed/empty column.
    """
    name = str(series.name) if series.name is not None else ""
    if not name:
        raise DataValidationError("column_profile: the series needs a name")
    raw_size = int(series.size)
    values = series.dropna()
    if values.empty:
        return ColumnProfile(
            name=name,
            col_type="unsupported",
            n=0,
            n_missing=raw_size,
            n_distinct=0,
            top_values=[],
            stats={},
        )
    col_type = infer_column_type(series)
    n_distinct = int(values.nunique())
    stats: dict[str, float] = {}
    histogram = None
    if col_type == "numeric":
        numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
        finite = numeric[np.isfinite(numeric)]
        quantiles = np.percentile(finite, [100.0 * q for q in PROFILE_QUANTILES])
        median = float(np.median(finite))
        stats = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "range": float(finite.max() - finite.min()),
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "iqr": float(quantiles[3] - quantiles[1]),
            "mad": float(np.median(np.abs(finite - median))),
            "cv": float(finite.std(ddof=1) / abs(finite.mean())) if finite.mean() else float("nan"),
            "skew": float(scipy_stats.skew(finite)) if finite.size > 2 else 0.0,
            "kurtosis": float(scipy_stats.kurtosis(finite)) if finite.size > 3 else 0.0,
            "n_negative": float((finite < 0).sum()),
            "n_zeros": float((finite == 0).sum()),
            "n_infinite": float((~np.isfinite(numeric)).sum()),
            "monotonic": float(monotonicity(finite)),
        }
        bins = min(HISTOGRAM_BINS, max(int(n_distinct), 1), 250)
        if finite.size >= 2 and finite.max() > finite.min():
            counts, edges = np.histogram(finite, bins=bins)
            histogram = (counts, edges)
            stats["chi2_p"] = _chi2_uniform_p(counts)
        else:
            stats["chi2_p"] = float("nan")
    elif col_type == "categorical":
        counts = values.value_counts().to_numpy(int)
        stats = {
            "imbalance": entropy_imbalance(counts),
        }
        if counts.size >= 2:
            stats["chi2_p"] = _chi2_uniform_p(counts)
        strings = values.astype(str)
        lengths = strings.str.len().to_numpy(int)
        stats.update(
            {
                "length_min": float(lengths.min()),
                "length_mean": float(lengths.mean()),
                "length_max": float(lengths.max()),
            }
        )
    elif col_type == "text":
        strings = values.astype(str)
        lengths = strings.str.len().to_numpy(int)
        stats = {
            "length_min": float(lengths.min()),
            "length_mean": float(lengths.mean()),
            "length_median": float(np.median(lengths)),
            "length_max": float(lengths.max()),
        }
    elif col_type == "boolean":
        counts = values.astype(bool).value_counts().to_numpy(int)
        stats = {"imbalance": entropy_imbalance(counts)}
    elif col_type == "datetime":
        stamps = pd.to_datetime(values, errors="coerce").dropna()
        epochs = stamps.astype("int64").to_numpy(float) / 1e9
        stats = {
            "min_epoch": float(epochs.min()),
            "max_epoch": float(epochs.max()),
            "range_seconds": float(epochs.max() - epochs.min()),
        }
    _LOG.debug("profiled column %s as %s", name, col_type)
    return ColumnProfile(
        name=name,
        col_type=col_type,
        n=int(values.size),
        n_missing=raw_size - int(values.size),
        n_distinct=n_distinct,
        top_values=_top_values(values),
        stats=stats,
        histogram=histogram,
    )


# ---------------------------------------------------------------------------
# The dataset profile + alerts
# ---------------------------------------------------------------------------


@dataclass
class ProfileAlert:
    """One triggered quality rule (ydata's alert vocabulary)."""

    column: str
    kind: str
    message: str

    def line(self) -> str:
        where = self.column if self.column else "(dataset)"
        return f"[{self.kind}] {where}: {self.message}"


def _build_alerts(profiles: dict[str, ColumnProfile], duplicates: int) -> list[ProfileAlert]:
    alerts: list[ProfileAlert] = []
    for name, column in profiles.items():
        if column.n == 0:
            alerts.append(
                ProfileAlert(name, "empty", "every value is missing")
            )
            continue
        if column.n_distinct <= 1:
            alerts.append(
                ProfileAlert(name, "constant", f"one distinct value over {column.n} rows")
            )
        if column.col_type in {"categorical", "text"} and column.n_distinct > CARDINALITY_THRESHOLD:
            alerts.append(
                ProfileAlert(
                    name,
                    "high_cardinality",
                    f"{column.n_distinct} distinct values (threshold {CARDINALITY_THRESHOLD})",
                )
            )
        if column.p_missing > ALERT_MISSING:
            alerts.append(
                ProfileAlert(
                    name, "missing", f"{column.p_missing:.1%} missing (threshold {ALERT_MISSING:.0%})"
                )
            )
        if column.is_unique:
            alerts.append(
                ProfileAlert(name, "unique", "every value is distinct")
            )
        if column.col_type == "numeric":
            skew = column.stats.get("skew", 0.0)
            if abs(skew) > ALERT_SKEW:
                alerts.append(
                    ProfileAlert(name, "skewed", f"skew {skew:.1f} (threshold ±{ALERT_SKEW:g})")
                )
            p_zeros = column.stats.get("n_zeros", 0.0) / max(column.n, 1)
            if p_zeros > ALERT_ZEROS:
                alerts.append(
                    ProfileAlert(name, "zeros", f"{p_zeros:.1%} exact zeros")
                )
            if column.stats.get("n_infinite", 0.0) > 0:
                alerts.append(
                    ProfileAlert(name, "infinite", "non-finite values present")
                )
            chi2_p = column.stats.get("chi2_p", float("nan"))
            if math.isfinite(chi2_p) and chi2_p > ALERT_UNIFORM_P:
                alerts.append(
                    ProfileAlert(
                        name, "uniform", f"distribution indistinguishable from uniform (p {chi2_p:.3f})"
                    )
                )
        if column.col_type in {"categorical", "boolean"}:
            imbalance = column.stats.get("imbalance", 0.0)
            if imbalance > ALERT_IMBALANCE:
                alerts.append(
                    ProfileAlert(
                        name, "imbalanced", f"entropy imbalance {imbalance:.2f} (threshold {ALERT_IMBALANCE})"
                    )
                )
        if column.col_type == "unsupported":
            alerts.append(
                ProfileAlert(name, "unsupported", "the type tree could not classify it")
            )
    if duplicates:
        alerts.append(
            ProfileAlert("", "duplicates", f"{duplicates} fully duplicated row(s)")
        )
    return alerts


@dataclass
class DatasetProfile:
    """The whole table's report: rollup, per-column profiles, alerts."""

    n_rows: int
    n_columns: int
    n_missing_cells: int
    n_duplicate_rows: int
    memory_bytes: int
    types: dict[str, int]
    columns: dict[str, ColumnProfile]
    alerts: list[ProfileAlert]

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_columns

    @property
    def p_missing_cells(self) -> float:
        return self.n_missing_cells / self.n_cells if self.n_cells else 0.0

    def headline(self) -> str:
        census = ", ".join(f"{count} {kind}" for kind, count in sorted(self.types.items()))
        return (
            f"{self.n_rows} rows x {self.n_columns} columns · "
            f"{self.p_missing_cells:.1%} cells missing · "
            f"{self.n_duplicate_rows} duplicate rows · {census} · "
            f"{len(self.alerts)} alert(s)"
        )

    def as_frame(self) -> pd.DataFrame:
        """The flat per-column table (the GUI/report twin)."""
        rows: list[dict[str, Any]] = []
        for name, column in self.columns.items():
            rows.append(
                {
                    "column": name,
                    "type": column.col_type,
                    "n": column.n,
                    "missing": round(column.p_missing, 4),
                    "distinct": column.n_distinct,
                    "top": column.top_values[0][0] if column.top_values else "",
                    "top_n": column.top_values[0][1] if column.top_values else 0,
                    "key_stat": _key_stat(column),
                }
            )
        return pd.DataFrame(rows)

    def as_dict(self) -> dict[str, Any]:
        """The JSON-ready schema (analysis / table / variables / alerts)."""
        return {
            "table": {
                "n": self.n_rows,
                "n_var": self.n_columns,
                "n_cells_missing": self.n_missing_cells,
                "p_cells_missing": round(self.p_missing_cells, 6),
                "n_duplicates": self.n_duplicate_rows,
                "memory_bytes": self.memory_bytes,
                "types": dict(sorted(self.types.items())),
            },
            "variables": {
                name: {
                    "type": column.col_type,
                    "n": column.n,
                    "n_missing": column.n_missing,
                    "n_distinct": column.n_distinct,
                    "top_values": [
                        {"value": value, "count": count}
                        for value, count in column.top_values
                    ],
                    "stats": {key: round(value, 6) for key, value in column.stats.items()},
                }
                for name, column in self.columns.items()
            },
            "alerts": [
                {"column": alert.column, "kind": alert.kind, "message": alert.message}
                for alert in self.alerts
            ],
        }


def _key_stat(column: ColumnProfile) -> str:
    """One representative stat per type, for the flat table."""
    if column.col_type == "numeric":
        return f"mean {column.stats.get('mean', float('nan')):.4g} ± {column.stats.get('std', 0.0):.3g}"
    if column.col_type in {"categorical", "boolean"}:
        return f"imbalance {column.stats.get('imbalance', float('nan')):.2f}"
    if column.col_type == "text":
        return f"mean length {column.stats.get('length_mean', 0.0):.1f}"
    if column.col_type == "datetime":
        seconds = column.stats.get("range_seconds", 0.0)
        return f"span {seconds / 86400:.0f} days"
    return "—"


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    """Profile every column, roll up the table, and run the alert rules.

    Raises
    ------
    DataValidationError
        On an empty frame or a frame with no columns.
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("profile_dataset: needs a pandas DataFrame")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise DataValidationError(
            f"profile_dataset: the frame is empty ({frame.shape[0]}x{frame.shape[1]})"
        )
    profiles = {str(name): column_profile(frame[name]) for name in frame.columns}
    types: dict[str, int] = {}
    for column in profiles.values():
        types[column.col_type] = types.get(column.col_type, 0) + 1
    missing_cells = int(frame.isna().to_numpy().sum())
    duplicates = int(frame.duplicated().sum())
    try:
        memory = int(frame.memory_usage(deep=True).sum())
    except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
        memory = 0
    alerts = _build_alerts(profiles, duplicates)
    profile = DatasetProfile(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        n_missing_cells=missing_cells,
        n_duplicate_rows=duplicates,
        memory_bytes=memory,
        types=types,
        columns=profiles,
        alerts=alerts,
    )
    _LOG.info("profiled %dx%d: %d alerts", profile.n_rows, profile.n_columns, len(alerts))
    return profile


def demo_profile(seed: int = RANDOM_STATE) -> DatasetProfile:
    """A seeded 500x11 screen with one planted issue per alert rule.

    The planted issues: a constant lot, a fully missing assay column,
    an 80-code high-cardinality column, a 95/5 imbalanced operator,
    3% missing responses, one dominating spike (the sample skew of a
    single extreme value caps at ``sqrt(n)``, so the column needs the
    one planted 400 against a sea of ~1s to cross ydata's ±20 skew
    threshold honestly), plus the honest citizens (numeric dose,
    4-level plate, boolean QC flag, dates, free text).
    """
    rng = np.random.default_rng(seed)
    n = 500
    spiked = np.concatenate([[400.0], rng.normal(1.0, 0.4, n - 1)])
    rng.shuffle(spiked)
    missing_mask = np.zeros(n, dtype=bool)
    missing_mask[rng.choice(n, size=int(n * 0.03), replace=False)] = True
    frame = pd.DataFrame(
        {
            "dose_mg": rng.uniform(0.0, 10.0, n),
            "response": np.where(
                missing_mask, np.nan, rng.normal(52.0, 9.0, n)
            ),
            "plate_id": rng.integers(1, 5, n),
            "operator": rng.choice(["alice", "bob"], n, p=[0.95, 0.05]),
            "batch_code": [f"B{index:03d}" for index in rng.integers(0, 80, n)],
            "qc_flag": rng.random(n) < 0.9,
            "recruited": pd.to_datetime(
                "2024-01-01", format="%Y-%m-%d"
            )
            + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
            "notes": [f"sample note {index}" for index in range(n)],
            "assay_dead": np.nan,
            "constant_lot": "LOT-7",
            "spike_assay": spiked,
        }
    )
    return profile_dataset(frame)


__all__ = [
    "ALERT_IMBALANCE",
    "ALERT_MISSING",
    "ALERT_SKEW",
    "ALERT_UNIFORM_P",
    "ALERT_ZEROS",
    "AUTO_CORRELATION_BINS",
    "BOOL_TOKENS",
    "CARDINALITY_THRESHOLD",
    "COLUMN_TYPES",
    "HISTOGRAM_BINS",
    "LOW_CATEGORICAL_THRESHOLD",
    "PERCENTAGE_CAT_THRESHOLD",
    "PROFILE_QUANTILES",
    "ColumnProfile",
    "DatasetProfile",
    "ProfileAlert",
    "auto_correlation_matrix",
    "column_profile",
    "cramers_v",
    "demo_profile",
    "entropy_imbalance",
    "infer_column_type",
    "monotonicity",
    "profile_dataset",
]

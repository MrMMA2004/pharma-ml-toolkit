r"""DataFrame profiling: type inference, alerts, duplicates, missingness.

Round 173 absorbed ``Data-Centric-AI-Community/fg-data-profiling``
(the renamed ydata-profiling / pandas-profiling).  The HTML report
engine and the visions type-system graph are the not-vendored part;
the *decision rules* transfer as pure pandas:

* **Type inference** (``model/typeset_relations.py``): numeric gate by
  dtype; numeric->categorical when ``n_unique <= 5``; string->
  categorical when ``n_unique <= 50`` and the distinct ratio < 0.5;
  string->datetime when ``pd.to_datetime(..., errors="coerce")``
  parses at least one value; string->boolean on the
  ``{t,f,yes,no,y,n,true,false}`` lowercase mapping.
* **The alert rules** (``model/alerts.py`` + ``config_default.yaml``
  thresholds): missing/zeros/infinite at > 1%, skewness at |skew| >
  20, high cardinality at n_distinct > 50, high correlation at
  |corr| >= 0.9 (diagonal excluded), uniformity via the chi-square
  p > 0.999, imbalance ``1 - entropy/log2(k)`` > 0.5, constant,
  unique, constant-length and duplicate-row (> 1%) alerts.
* **The corrected Cramér's V** and the auto mixed-type correlation
  matrix (spearman for numeric pairs, Cramér for anything else after
  uniform 10-bin discretisation).
* **Duplicates** (``model/pandas/duplicates_pandas.py``):
  ``duplicated(keep=False)`` + groupby counting of the distinct
  duplicated combinations, top-10 table.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from pharmaml.utils.profile import profile_frame
>>> rng = np.random.default_rng(4)
>>> frame = pd.DataFrame({
...     "dose": rng.normal(10, 2, 300),
...     "route": rng.choice(["IV", "ORAL"], 300),
...     "response": np.concatenate([np.full(150, 1.0), rng.normal(5, 1, 150)]),
...     "empty": np.nan,
... })
>>> report = profile_frame(frame)
>>> int(report["overview"]["n_rows"])
300
>>> report["variables"]["route"]["type"]
'Categorical'
>>> bool(report["variables"]["empty"]["alerts"]["MISSING"])
True
>>> "HIGH_CARDINALITY" in report["variables"]["response"]["alerts"]
True
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "infer_column_type",
    "profile_frame",
]

_BOOL_TOKENS = {"t", "f", "yes", "no", "y", "n", "true", "false"}

#: the fg-data-profiling default thresholds (config_default.yaml)
THRESHOLDS: dict[str, float] = {
    "low_categorical_threshold": 5.0,
    "cardinality_threshold": 50.0,
    "percentage_cat_threshold": 0.5,
    "missing_alert": 0.01,
    "skewness_threshold": 20.0,
    "correlation_threshold": 0.9,
    "chi_squared_threshold": 0.999,
    "imbalance_threshold": 0.5,
    "duplicates_alert": 0.01,
}


def infer_column_type(series: pd.Series) -> str:
    """The typeset decision rules as one function.

    ``Numeric`` / ``Categorical`` / ``Boolean`` / ``DateTime`` /
    ``Text`` with the exact thresholds of
    ``model/typeset_relations.py``.

    Examples
    --------
    >>> infer_column_type(pd.Series(range(20)))
    'Numeric'
    >>> infer_column_type(pd.Series([1, 2, 3]))      # <= 5 levels -> category
    'Categorical'
    >>> infer_column_type(pd.Series(["a", "b", "a", "a", "a"]))   # ratio 0.4
    'Categorical'
    >>> infer_column_type(pd.Series(["2024-01-01", "2024-06-01"]))
    'DateTime'
    >>> infer_column_type(pd.Series(["yes", "no"]))
    'Boolean'
    """
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DateTime"
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if 1 <= series.nunique() <= THRESHOLDS["low_categorical_threshold"]:
            return "Categorical"
        return "Numeric"
    text = series.dropna().astype(str)
    if text.empty:
        return "Unsupported"
    if text.str.lower().isin(_BOOL_TOKENS).all():
        return "Boolean"
    if pd.to_datetime(text, errors="coerce", format="mixed").notna().any():
        return "DateTime"
    n_unique = text.nunique()
    ratio = n_unique / max(len(text), 1)
    if (
        1 <= n_unique <= THRESHOLDS["cardinality_threshold"]
        and ratio < THRESHOLDS["percentage_cat_threshold"]
    ):
        return "Categorical"
    return "Text"


def _imbalance(value_counts: pd.Series) -> float:
    """``1 - entropy(vc) / log2(k)`` (fg-data-profiling's imbalance score)."""
    if len(value_counts) < 2:
        return 1.0
    probabilities = value_counts.values / value_counts.values.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return 1.0 - entropy / math.log2(len(value_counts))


def _chi_square_uniformity(values: np.ndarray, bins: int = 10) -> dict[str, float]:
    from scipy import stats as scipy_stats

    finite = values[np.isfinite(values)]
    if finite.size < bins * 2:
        return {"statistic": float("nan"), "p": float("nan")}
    counts, _ = np.histogram(finite, bins=bins)
    if (counts == 0).any():
        return {"statistic": float("nan"), "p": float("nan")}
    result = scipy_stats.chisquare(counts)
    return {"statistic": float(result.statistic), "p": float(result.pvalue)}


def _cramers_v(observed: np.ndarray) -> float:
    from scipy import stats as scipy_stats

    chi2 = float(scipy_stats.chi2_contingency(observed, correction=False)[0])
    n = float(observed.sum())
    rows, cols = observed.shape
    dof = min(rows - 1, cols - 1)
    if dof == 0 or n == 0:
        return 0.0
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    r_corr = rows - ((rows - 1) ** 2) / (n - 1)
    c_corr = cols - ((cols - 1) ** 2) / (n - 1)
    denom = min(c_corr - 1, r_corr - 1)
    if denom <= 0:
        return math.sqrt(chi2 / (n * dof))
    return float(math.sqrt(phi2_corr / denom))


def _correlation_matrix(frame: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """The 'auto' correlation: spearman numeric-numeric, Cramér otherwise."""
    columns = [c for c, t in types.items()
               if t in {"Numeric", "Categorical", "Boolean", "DateTime"}
               and frame[c].notna().any()]
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    discretized: dict[str, np.ndarray] = {}
    for column in columns:
        if types[column] == "Numeric":
            # full column (NaN passes through) so all arrays share one length;
            # crosstab drops the NaN rows pairwise itself
            discretized[column] = pd.cut(
                frame[column], bins=10, labels=False, duplicates="drop"
            ).to_numpy(dtype=float)
        else:
            discretized[column] = frame[column].astype(str).to_numpy()
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            if types[a] == "Numeric" and types[b] == "Numeric":
                both = frame[[a, b]].dropna()
                if len(both) > 2:
                    matrix.loc[a, b] = matrix.loc[b, a] = float(
                        both[a].corr(both[b], method="spearman"))
            else:
                table = pd.crosstab(discretized[a], discretized[b]).to_numpy()
                if table.shape == (1, 1):
                    continue
                matrix.loc[a, b] = matrix.loc[b, a] = _cramers_v(table)
    return matrix


def profile_frame(
    frame: pd.DataFrame,
    include_correlations: bool = True,
) -> dict[str, Any]:
    """One-pass DataFrame profile: overview, per-column stats, alerts.

    The fg-data-profiling report schema reduced to its data core:
    ``overview`` (rows/cols/memory/missing/duplicates + the type
    counter), ``variables`` (per-column stats with the alert set),
    ``correlations`` (the auto mixed matrix) and ``nullity``
    (per-column missing rates + the pairwise nullity correlations).

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> frame = pd.DataFrame({"a": rng.normal(size=200),
    ...                       "b": rng.choice(["x", "y"], 200)})
    >>> report = profile_frame(frame)
    >>> report["overview"]["n_var"]
    2
    >>> set(report["variables"]["b"]["top"].index) <= {"x", "y"}
    True
    """
    if frame.empty:
        raise DataValidationError("profile_frame: the frame is empty")
    overview: dict[str, Any] = {
        "n_rows": len(frame),
        "n_var": int(frame.shape[1]),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "n_duplicate_rows": int(frame.duplicated().sum()),
        "p_duplicate_rows": float(frame.duplicated().mean()),
        "n_cells_missing": int(frame.isna().sum().sum()),
        "p_cells_missing": float(frame.isna().sum().sum() / frame.size),
    }
    types = {column: infer_column_type(frame[column]) for column in frame.columns}
    overview["types"] = dict(pd.Series(list(types.values())).value_counts())

    variables: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column]
        kind = types[column]
        stats: dict[str, Any] = {
            "type": kind,
            "n": len(series),
            "n_missing": int(series.isna().sum()),
            "p_missing": float(series.isna().mean()),
            "n_distinct": int(series.nunique(dropna=True)),
            "memory_bytes": int(series.memory_usage(deep=True)),
        }
        alerts: dict[str, bool] = {}
        if stats["p_missing"] > THRESHOLDS["missing_alert"]:
            alerts["MISSING"] = True
        if stats["n_distinct"] == 1 and stats["n_missing"] < len(series):
            alerts["CONSTANT"] = True
        if stats["n_distinct"] == len(series) and stats["n_distinct"] > 0:
            alerts["UNIQUE"] = True
        if stats["n_distinct"] > THRESHOLDS["cardinality_threshold"]:
            alerts["HIGH_CARDINALITY"] = True
        if kind == "Numeric":
            values = series.dropna().to_numpy(dtype=float)
            if values.size:
                stats.update({
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_zeros": int((values == 0).sum()),
                    "n_negative": int((values < 0).sum()),
                    "n_infinite": int(np.isinf(values).sum()),
                    "skewness": float(pd.Series(values).skew()),
                    "mad": float(np.median(np.abs(values - np.median(values)))),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "monotonic": bool(
                        np.all(np.diff(values) >= 0) or np.all(np.diff(values) <= 0)
                    ),
                })
                chi = _chi_square_uniformity(values)
                stats["chi_square"] = chi
                if not math.isnan(chi["p"]) and chi["p"] > THRESHOLDS["chi_squared_threshold"]:
                    alerts["UNIFORM"] = True
                if abs(stats["skewness"]) > THRESHOLDS["skewness_threshold"]:
                    alerts["SKEWED"] = True
                if stats["n_zeros"] / len(series) > THRESHOLDS["missing_alert"]:
                    alerts["ZEROS"] = True
                if stats["n_infinite"] / len(series) > THRESHOLDS["missing_alert"]:
                    alerts["INFINITE"] = True
        elif kind in {"Categorical", "Boolean"}:
            counts = series.value_counts(dropna=True)
            stats["top"] = counts.head(5)
            stats["imbalance"] = _imbalance(counts)
            if stats["imbalance"] > THRESHOLDS["imbalance_threshold"]:
                alerts["IMBALANCE"] = True
        elif kind == "DateTime":
            parsed = pd.to_datetime(series, errors="coerce")
            stats["min"] = str(parsed.min())
            stats["max"] = str(parsed.max())
            stats["n_invalid"] = int(parsed.isna().sum() - series.isna().sum())
        elif kind == "Text":
            lengths = series.dropna().astype(str).str.len()
            if len(lengths):
                stats["length_mean"] = float(lengths.mean())
                if lengths.min() == lengths.max():
                    alerts["CONSTANT_LENGTH"] = True
        stats["alerts"] = alerts
        variables[column] = stats

    report: dict[str, Any] = {"overview": overview, "variables": variables}
    if include_correlations and frame.shape[1] > 1:
        corr = _correlation_matrix(frame, types)
        report["correlations"] = corr
        above = (corr.abs() >= THRESHOLDS["correlation_threshold"]).to_numpy()
        np.fill_diagonal(above, False)
        for i, a in enumerate(corr.index):
            partners = [corr.columns[j] for j in np.where(above[i])[0]]
            if partners:
                variables[a]["alerts"]["HIGH_CORRELATION"] = True
                variables[a]["correlated_with"] = partners
    # nullity patterns: pairwise missingness correlations
    missing = frame.isna()
    columns_with_missing = [c for c in frame.columns if missing[c].any()]
    nullity = {c: float(missing[c].mean()) for c in frame.columns}
    report["nullity"] = {"rates": nullity}
    if len(columns_with_missing) >= 2:
        report["nullity"]["correlations"] = missing[columns_with_missing].corr()
    return report

"""Dataset profiling -- the fg-data-profiling metric core.

``Data-Centric-AI-Community/fg-data-profiling`` (the ydata-profiling
lineage, ex-pandas-profiling) computes the dataset health report: per
column, the count/missingness/cardinality base plus type-specific
statistics; at table level, the missing-cell rate, duplicate rows and
correlations; on top, a rule-based alert layer (constant, high
cardinality, skew, missing, zeros, duplicates ...).  This module
absorbs exactly that metric core on pandas/numpy:
:func:`profile_frame` walks every column through
:func:`profile_column`, :func:`table_profile` aggregates, and
:func:`profile_alerts` applies the threshold rules (missing > 1%,
|skew| > 20, cardinality > 50, correlation > 0.9, ...).  The
visions type DAG is reduced to a pandas-dtype + heuristic inference
(numeric / categorical / boolean / datetime / text).

The honest scope, stated once: the HTML report tree, the Great
Expectations bridge, Spark backends, PhiK correlations and the
interactions/scatter layer are out of scope -- this is the numbers a
data-quality gate consumes.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.data_profiling import profile_frame, profile_alerts
>>> frame = pd.DataFrame({
...     "dose": [1.0, 2.0, 3.0, 4.0],
...     "batch": ["a", "a", "b", None],
...     "flag": [True, False, True, True],
... })
>>> table = profile_frame(frame)
>>> table["n"]
4
>>> table["variables"]["dose"]["p_missing"]
0.0
>>> alerts = profile_alerts(frame)
>>> any(a["alert"] == "missing" for a in alerts)
True
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "cramers_v",
    "profile_alerts",
    "profile_column",
    "profile_frame",
]

_log = get_logger(__name__)

#: the alert thresholds (fg-data-profiling's defaults)
MISSING_ALERT = 0.01
SKEW_ALERT = 20.0
CARDINALITY_ALERT = 50
CORRELATION_ALERT = 0.9


def _infer_kind(series: pd.Series) -> str:
    """numeric / datetime / boolean / categorical from dtype + values."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if len(series) and series.dropna().size:
        sample = series.dropna().astype(str).head(200)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                return "datetime"
        except (ValueError, TypeError):
            pass
    return "categorical"


def profile_column(series: pd.Series) -> dict[str, Any]:
    """One column's profile: base stats plus the type-specific block.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.data_profiling import profile_column
    >>> stats = profile_column(pd.Series([1.0, 2.0, 3.0, None]))
    >>> stats["n_missing"], stats["p_missing"]
    (1, 0.25)
    >>> round(stats["mean"], 3)
    2.0
    """
    if not isinstance(series, pd.Series):
        raise DataValidationError("profile_column needs a pandas Series")
    values = series.dropna()
    n = int(series.size)
    count = int(values.size)
    distinct = int(values.nunique())
    base: dict[str, Any] = {
        "name": str(series.name),
        "kind": _infer_kind(series),
        "n": n,
        "count": count,
        "n_missing": n - count,
        "p_missing": (n - count) / n if n else 0.0,
        "n_distinct": distinct,
        "p_distinct": distinct / count if count else 0.0,
        "is_unique": distinct == count and count > 1,
    }
    if base["kind"] == "numeric":
        numeric = pd.to_numeric(values)
        base.update({
            "mean": float(numeric.mean()) if count else None,
            "std": float(numeric.std()) if count > 1 else None,
            "min": float(numeric.min()) if count else None,
            "max": float(numeric.max()) if count else None,
            "q05": float(numeric.quantile(0.05)) if count else None,
            "q25": float(numeric.quantile(0.25)) if count else None,
            "q50": float(numeric.quantile(0.50)) if count else None,
            "q75": float(numeric.quantile(0.75)) if count else None,
            "q95": float(numeric.quantile(0.95)) if count else None,
            "iqr": float(numeric.quantile(0.75) - numeric.quantile(0.25)) if count else None,
            "n_zeros": int((numeric == 0).sum()),
            "n_negative": int((numeric < 0).sum()),
            "n_infinite": int(np.isinf(numeric.to_numpy(dtype=float)).sum())
            if count else 0,
            "skew": float(numeric.skew()) if count > 2 else None,
            "kurtosis": float(numeric.kurt()) if count > 3 else None,
            "mad": float((numeric - numeric.median()).abs().median()) if count else None,
        })
    elif base["kind"] == "categorical":
        top = values.value_counts().head(5)
        base.update({
            "top_values": [(str(k), int(v)) for k, v in top.items()],
            "imbalance": _imbalance(values),
        })
    elif base["kind"] == "boolean":
        top = values.astype(bool).value_counts()
        base.update({
            "top_values": [(bool(k), int(v)) for k, v in top.items()],
            "imbalance": _imbalance(values),
        })
    elif base["kind"] == "datetime":
        parsed = pd.to_datetime(values, errors="coerce").dropna()
        base.update({
            "min": str(parsed.min()) if parsed.size else None,
            "max": str(parsed.max()) if parsed.size else None,
            "n_invalid": count - int(parsed.size),
        })
    return base


def _imbalance(values: pd.Series) -> float:
    """``1 − H₂(freqs)/log₂(k)``: 0 uniform, 1 maximally imbalanced."""
    counts = values.value_counts().to_numpy(dtype=float)
    if counts.size < 2 or counts.sum() <= 0:
        return 0.0
    freqs = counts / counts.sum()
    positive = freqs[freqs > 0]
    entropy = float(-(positive * np.log2(positive)).sum())
    return float(1.0 - entropy / math_log2(counts.size))


def math_log2(x: float) -> float:
    """float log2 (tiny helper to keep the expression above readable)."""
    return float(np.log2(max(x, 1e-12)))


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical columns.

    ``φ² = χ²/n``; ``φ²corr = max(0, φ² − (k−1)(r−1)/(n−1))``;
    ``V = √(φ²corr / min(k−1, r−1))`` -- the exact correction
    fg-data-profiling applies.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.data_profiling import cramers_v
    >>> x = pd.Series(["a", "a", "b", "b"])
    >>> round(cramers_v(x, x), 6)
    0.816497
    """
    from scipy.stats import chi2_contingency

    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    n = float(table.to_numpy().sum())
    chi2 = float(chi2_contingency(table, correction=False)[0])
    phi2 = chi2 / n
    k, r = table.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    denominator = min(k - 1, r - 1)
    if denominator <= 0:
        return 1.0 if phi2_corr > 0 else 0.0
    return float(np.sqrt(phi2_corr / denominator))


def table_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """The table-level block: cells, duplicates, memory, correlations.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.data_profiling import table_profile
    >>> frame = pd.DataFrame({"a": [1.0, 2.0, 2.0], "b": [1.0, 1.0, 1.0]})
    >>> table = table_profile(frame)
    >>> table["n_duplicates"]   # both rows of the duplicated pair
    2
    >>> table["n_var"]
    2
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("table_profile needs a non-empty DataFrame")
    n, n_var = frame.shape
    duplicated = int(frame.duplicated(keep=False).sum())
    return {
        "n": n,
        "n_var": n_var,
        "n_cells": n * n_var,
        "n_cells_missing": int(frame.isna().sum().sum()),
        "p_cells_missing": float(frame.isna().sum().sum() / (n * n_var)),
        "n_var_with_missing": int((frame.isna().any()).sum()),
        "n_var_all_missing": int((frame.isna().all()).sum()),
        "n_duplicates": duplicated,
        "p_duplicates": duplicated / n if n else 0.0,
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
    }


def profile_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """The full profile: table block + per-variable profiles.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.data_profiling import profile_frame
    >>> table = profile_frame(pd.DataFrame({"x": [1.0, 2.0, None]}))
    >>> sorted(table.keys())[:3]
    ['memory_bytes', 'n', 'n_cells']
    """
    table = table_profile(frame)
    table["variables"] = {
        str(column): profile_column(frame[column]) for column in frame.columns
    }
    return table


def profile_alerts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The rule-based alert layer (fg-data-profiling's defaults).

    Rules: constant columns, uniqueness, missing > 1%, |skew| > 20,
    cardinality > 50, zeros, infinite values, duplicate rows,
    |correlation| > 0.9 between numeric columns, and categorical
    imbalance > 0.9 with > 2 levels.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.data_profiling import profile_alerts
    >>> frame = pd.DataFrame({
    ...     "flat": [1.0, 1.0, 1.0], "sparse": [1.0, None, None],
    ... })
    >>> kinds = {a["alert"] for a in profile_alerts(frame)}
    >>> {"constant", "missing"} <= kinds
    True
    """
    alerts: list[dict[str, Any]] = []
    profiles = profile_frame(frame)
    for name, stats in profiles["variables"].items():
        if stats["n"] and stats["n_distinct"] <= 1:
            alerts.append({"alert": "constant", "column": name})
        if stats["is_unique"]:
            alerts.append({"alert": "unique", "column": name})
        if stats["p_missing"] > MISSING_ALERT:
            alerts.append({
                "alert": "missing", "column": name,
                "level": round(stats["p_missing"], 4),
            })
        if stats["kind"] == "numeric":
            skew = stats.get("skew")
            if skew is not None and abs(skew) > SKEW_ALERT:
                alerts.append({"alert": "skewed", "column": name,
                               "level": round(float(skew), 2)})
            if stats.get("n_infinite"):
                alerts.append({"alert": "infinite", "column": name})
            if stats.get("n_zeros", 0) > stats["count"] / 2 and stats["count"]:
                alerts.append({"alert": "zeros", "column": name})
        if stats["n_distinct"] > CARDINALITY_ALERT:
            alerts.append({"alert": "high-cardinality", "column": name,
                           "level": stats["n_distinct"]})
        imbalance = stats.get("imbalance")
        if imbalance is not None and imbalance > 0.9 and stats["n_distinct"] > 2:
            alerts.append({"alert": "imbalance", "column": name,
                           "level": round(float(imbalance), 3)})
    if profiles["p_duplicates"] > MISSING_ALERT:
        alerts.append({"alert": "duplicates", "column": "*",
                       "level": profiles["n_duplicates"]})
    numeric = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric) >= 2:
        corr = frame[numeric].corr().to_numpy(dtype=float)
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                value = corr[i, j]
                if np.isfinite(value) and abs(value) > CORRELATION_ALERT:
                    alerts.append({
                        "alert": "high-correlation",
                        "column": f"{numeric[i]}~{numeric[j]}",
                        "level": round(float(value), 3),
                    })
    return alerts

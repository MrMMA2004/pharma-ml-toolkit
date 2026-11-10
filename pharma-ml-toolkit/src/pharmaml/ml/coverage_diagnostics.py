"""Empirical prediction-coverage summaries, including subgroup diagnostics.

These summaries describe supplied held-out outcomes; they do not establish
conditional conformal coverage or prove a model is safe. Wilson intervals are
approximate binomial intervals for independent evaluation observations with a
common coverage probability within each group, with the fitted predictor and
calibration held fixed. Correlation, shift and adaptive group selection can
invalidate their interpretation. All input alignment is positional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.conformal_classification import _labels

__all__ = ["CoverageGroup", "CoverageReport", "coverage_by_group"]


@dataclass(frozen=True)
class CoverageGroup:
    """Empirical coverage, Wilson bounds, and optional mean size for one group."""

    group: Any
    n_test: int
    n_covered: int
    coverage: float
    lower: float
    upper: float
    mean_size: float | None = None


@dataclass(frozen=True)
class CoverageReport:
    """Overall and first-seen-ordered subgroup coverage on held-out observations."""

    overall: CoverageGroup
    groups: tuple[CoverageGroup, ...]
    confidence: float

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per group; the separate overall record is not included."""
        return pd.DataFrame([
            {"group": row.group, "n_test": row.n_test, "n_covered": row.n_covered,
             "coverage": row.coverage, "lower": row.lower, "upper": row.upper,
             "mean_size": row.mean_size}
            for row in self.groups
        ])


def _booleans(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("covered must be a one-dimensional boolean vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise DataValidationError("covered must be a nonempty one-dimensional boolean vector")
    if not all(isinstance(item, (bool, np.bool_)) for item in array):
        raise DataValidationError("covered must contain only boolean values")
    return np.asarray(array, dtype=bool)


def _sizes(value: Any, n: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("sizes must be a one-dimensional vector aligned with covered") from exc
    if array.ndim != 1 or len(array) != n:
        raise DataValidationError("sizes must be a one-dimensional vector aligned with covered")
    converted: list[float] = []
    for item in array:
        try:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
                raise TypeError("not numeric")
            number = float(item)
            if not math.isfinite(number) or number < 0:
                raise ValueError("not finite and nonnegative")
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError("sizes must contain finite nonnegative numbers") from exc
        converted.append(number)
    return np.asarray(converted, dtype=float)


def _summary(group: Any, covered: np.ndarray, sizes: np.ndarray | None, z: float) -> CoverageGroup:
    n = len(covered)
    count = int(np.count_nonzero(covered))
    p = count / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    size_mean = None
    if sizes is not None:
        # Scaling avoids overflow in the sum for very large finite sizes.
        maximum = float(np.max(sizes))
        size_mean = 0.0 if maximum == 0 else maximum * float(np.mean(sizes / maximum))
    return CoverageGroup(group=group, n_test=n, n_covered=count, coverage=p,
                         lower=max(0.0, center - margin), upper=min(1.0, center + margin),
                         mean_size=size_mean)


def coverage_by_group(
    covered: Any, groups: Any, *, sizes: Any = None, confidence: float = 0.95
) -> CoverageReport:
    """Summarize boolean held-out coverage overall and within observed groups.

    ``covered`` and ``groups`` must be aligned nonempty vectors. Group labels
    are finite numeric or string scalars, with equality following Python's
    normal rules; groups are returned in first-appearance order. Optionally
    provide one finite nonnegative interval width or set size per row. Pandas
    indexes are not used for alignment. ``confidence`` selects two-sided
    Wilson score intervals (strictly between zero and one).

    The intervals describe sampling uncertainty, not a conformal guarantee or
    a correction for subgroup selection/multiple comparisons. Their binomial
    interpretation requires independent observations with a common coverage
    probability within each group, with the fitted model and calibration fixed.
    Correlated test observations, distribution shift, and small groups can
    make comparisons misleading. Use a separate evaluation set and prespecify
    groups; model or threshold tuning on evaluation data causes leakage.
    """
    hits = _booleans(covered)
    labels = _labels(groups, "groups")
    if len(labels) != len(hits):
        raise DataValidationError("groups and covered must have the same length")
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real):
        raise DataValidationError("confidence must be finite and strictly between 0 and 1")
    try:
        level = float(confidence)
    except (ValueError, OverflowError) as exc:
        raise DataValidationError("confidence must be finite and strictly between 0 and 1") from exc
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise DataValidationError("confidence must be finite and strictly between 0 and 1")
    lengths = _sizes(sizes, len(hits)) if sizes is not None else None
    z = -NormalDist().inv_cdf((1.0 - level) / 2.0)
    indices: dict[Any, list[int]] = {}
    for index, label in enumerate(labels):
        indices.setdefault(label, []).append(index)
    groups_out = tuple(
        _summary(label, hits[rows], lengths[rows] if lengths is not None else None, z)
        for label, rows in indices.items()
    )
    return CoverageReport(overall=_summary(None, hits, lengths, z), groups=groups_out,
                          confidence=level)

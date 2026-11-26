"""Isolation-forest anomaly detection (the part18 §4.5 lane).

Anomaly detection is *not* ordinary classification: there is no
balanced class prior, no confusion-matrix verdict, and the "labels"
are a contamination-parameter-driven threshold, not ground truth.
This module keeps that honesty explicit -- the workflow is
:func:`isolation_forest_anomaly`, an unsupervised fit over feature
data that returns per-sample anomaly scores, the outlier verdicts,
and an exportable per-sample report.  It plugs into the same
``random_state=42`` pin as the rest of the toolkit.

Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

Examples
--------
>>> import numpy as np, pandas as pd
>>> from pharmaml.ml.anomaly import isolation_forest_anomaly
>>> rng = np.random.default_rng(42)
>>> frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
>>> frame.iloc[0] = frame.iloc[0] + 12.0        # three planted outliers
>>> frame.iloc[1] = frame.iloc[1] - 14.0
>>> frame.iloc[2] = frame.iloc[2] + 15.0
>>> result = isolation_forest_anomaly(frame, contamination=0.03)
>>> result.n_outliers >= 3          # the three planted, plus at most one false positive
True
>>> sorted(result.outlier_indices[:3])   # the three planted, by anomaly rank
[0, 1, 2]
>>> 0.02 <= result.anomaly_fraction <= 0.05
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.ml.unsupervised import RANDOM_STATE, _as_numeric_matrix

_LOG = get_logger(__name__)

#: Valid ``contamination`` forms (sklearn's own set): a float share in
#: (0, 0.5] or the string ``'auto'`` (the threshold then comes from the
#: forest itself).
_CONTAMINATION_ERROR = (
    "contamination must be 'auto' or a float in (0.0, 0.5] "
    "(the outlier share the forest calibrates to), got {value!r}"
)


@dataclass
class AnomalyResult:
    """Typed report of :func:`isolation_forest_anomaly`.

    Attributes
    ----------
    labels:
        Per-sample verdict: ``1`` inlier, ``-1`` outlier (a Series when
        the input carried an index).
    scores:
        Per-sample anomaly score -- :meth:`IsolationForest.decision_function`
        convention: *lower is more anomalous*, negative samples are the
        flagged outliers.
    threshold:
        The score cutoff under the chosen contamination (the boundary
        between inlier and outlier).
    contamination:
        The contamination parameter actually used.
    n_estimators, seed:
        The forest size and seed (reproducibility evidence).
    n_samples, n_features:
        Shape of the fitted matrix.
    feature_names:
        Column names as fitted.
    """

    labels: Any
    scores: Any
    threshold: float
    contamination: float | str
    n_estimators: int
    seed: int
    n_samples: int
    n_features: int
    feature_names: list[str]
    _frame_index: Any = field(default=None, repr=False)

    @property
    def n_outliers(self) -> int:
        """Flagged outlier count."""
        return int((np.asarray(self.labels) == -1).sum())

    @property
    def anomaly_fraction(self) -> float:
        """Flagged outlier share of all samples."""
        return self.n_outliers / max(1, self.n_samples)

    @property
    def outlier_indices(self) -> list[int]:
        """Positional indices of the flagged outliers (ascending score)."""
        scores = np.asarray(self.scores)
        flagged = np.asarray(self.labels) == -1
        order = np.argsort(scores)
        return [int(i) for i in order if flagged[i]]

    def as_frame(self) -> pd.DataFrame:
        """The per-sample export: verdict, score, rank among anomalies."""
        out = pd.DataFrame(
            {
                "anomaly_score": np.asarray(self.scores),
                "is_outlier": np.asarray(self.labels) == -1,
            }
        )
        ranks: list[int] = [0] * len(out)
        for rank, position in enumerate(self.outlier_indices, start=1):
            ranks[position] = rank
        out["outlier_rank"] = ranks
        if self._frame_index is not None:
            out.index = self._frame_index
        return out

    def feature_report(self, X: Any) -> pd.DataFrame:
        """Feature-level diagnostic: mean per feature for outliers vs inliers.

        The honest caveat: an isolation forest splits on features but
        does not attribute per-sample importance; this contrast table
        (outlier means vs inlier means, the shift in sigma units) is a
        *descriptive* diagnostic, not a causal explanation.
        """
        matrix, names, index = _as_numeric_matrix(X, caller="feature_report")
        flagged = np.asarray(self.labels) == -1
        inliers = ~flagged
        rows: list[dict[str, Any]] = []
        for column in range(matrix.shape[1]):
            out_vals = matrix[flagged, column]
            in_vals = matrix[inliers, column]
            in_mean = float(np.mean(in_vals)) if in_vals.size else float("nan")
            in_std = float(np.std(in_vals)) if in_vals.size else float("nan")
            out_mean = float(np.mean(out_vals)) if out_vals.size else float("nan")
            shift = (out_mean - in_mean) / in_std if in_std and in_std > 0 else float("nan")
            rows.append(
                {
                    "feature": names[column],
                    "inlier_mean": in_mean,
                    "outlier_mean": out_mean,
                    "shift_sigma": shift,
                }
            )
        report = pd.DataFrame(rows)
        if index is not None and len(index) == matrix.shape[0]:
            pass  # feature-level rows do not carry the sample index
        return report

    def __str__(self) -> str:
        contamination = (
            f"{self.contamination:.3g}" if isinstance(self.contamination, float)
            else str(self.contamination)
        )
        return (
            f"IsolationForest -- {self.n_estimators} trees, seed={self.seed}, "
            f"contamination={contamination}\n"
            f"  {self.n_samples} samples x {self.n_features} features "
            f"[{', '.join(self.feature_names[:6])}{'...' if len(self.feature_names) > 6 else ''}]\n"
            f"  flagged {self.n_outliers} outliers "
            f"({self.anomaly_fraction:.1%}); score threshold {self.threshold:.3g}"
        )


def isolation_forest_anomaly(
    X: Any,
    *,
    contamination: float | str = "auto",
    n_estimators: int = 200,
    max_samples: str | float = "auto",
    seed: int = RANDOM_STATE,
) -> AnomalyResult:
    """Unsupervised anomaly detection with :class:`~sklearn.ensemble.IsolationForest`.

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array (complete rows).  No target: the
        "outliers" are defined by isolation depth, not by a label.
    contamination:
        ``'auto'`` (sklearn's own offset) or a float share in
        ``(0.0, 0.5]`` -- the expected outlier proportion the decision
        threshold calibrates to.  This is an analyst prior, not a
        measured rate.
    n_estimators:
        Trees in the forest (200 pinned default).
    max_samples:
        Subsample size per tree (sklearn ``'auto'`` or a float share).
    seed:
        Seed (the toolkit's 42 by default) -- the same data reruns to
        the same verdicts.

    Returns
    -------
    AnomalyResult

    Raises
    ------
    DataValidationError
        Bad ``contamination``, non-positive ``n_estimators`` or too
        few rows to isolate anything.
    """
    matrix, names, index = _as_numeric_matrix(X, caller="isolation_forest_anomaly")
    n_samples = matrix.shape[0]
    if n_samples < 4:
        raise DataValidationError(
            f"isolation_forest_anomaly needs at least 4 rows, got {n_samples}"
        )
    if isinstance(contamination, str):
        if contamination != "auto":
            raise DataValidationError(
                _CONTAMINATION_ERROR.format(value=contamination)
            )
    elif isinstance(contamination, (int, float)):
        if not 0.0 < float(contamination) <= 0.5:
            raise DataValidationError(
                _CONTAMINATION_ERROR.format(value=contamination)
            )
    else:
        raise DataValidationError(_CONTAMINATION_ERROR.format(value=contamination))
    n_estimators = int(n_estimators)
    if n_estimators < 1:
        raise DataValidationError(f"n_estimators must be >= 1, got {n_estimators}")
    forest = IsolationForest(
        contamination=contamination if isinstance(contamination, str) else float(contamination),
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=seed,
        n_jobs=1,  # conservative default: the toolkit never assumes all cores
    ).fit(matrix)
    scores = np.asarray(forest.decision_function(matrix))
    labels_array = np.asarray(forest.predict(matrix)).astype(int)
    if index is not None:
        labels: Any = pd.Series(labels_array, index=index, name="verdict", dtype=int)
        score_series: Any = pd.Series(scores, index=index, name="anomaly_score")
    else:
        labels = labels_array
        score_series = scores
    flagged = labels_array == -1
    # the cutoff: the weakest outlier's score (inliers sit above it,
    # outliers at or below)
    threshold = float(np.max(scores[flagged])) if np.any(flagged) else float(np.min(scores))
    _LOG.info(
        "isolation forest flagged %d/%d outliers (contamination=%r)",
        int(flagged.sum()),
        n_samples,
        contamination,
    )
    return AnomalyResult(
        labels=labels,
        scores=score_series,
        threshold=threshold,
        contamination=contamination,
        n_estimators=n_estimators,
        seed=int(seed),
        n_samples=n_samples,
        n_features=matrix.shape[1],
        feature_names=names,
        _frame_index=index,
    )


__all__ = ["AnomalyResult", "isolation_forest_anomaly"]

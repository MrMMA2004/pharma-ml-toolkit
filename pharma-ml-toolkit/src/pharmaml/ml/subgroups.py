"""Subgroup (stratified) performance audit: where does the model quietly fail?

The trust layer asks a sequence of questions about a fitted model.  Drift
(:mod:`pharmaml.ml.drift`) asks whether a new batch has moved.  Calibration
(:mod:`pharmaml.ml.calibration`) asks whether the probabilities are honest.
Decision curves (:mod:`pharmaml.ml.decision`) ask whether acting on the
model beats the defaults.  All three, however, average over the whole
held-out set — and a model can look fine in aggregate while failing a
subpopulation that matters (an assay, a scaffold class, a chemical series,
a site, a batch).

This module is the slice-it-and-look layer.  :func:`subgroup_report` scores
one held-out table and reports the headline metrics **per subgroup**
alongside the overall row, so the gap is read off rather than assumed.  It
refuses to invent a verdict: a subgroup too small to score is reported as
``insufficient`` (with its ``n``) instead of a number that would be noise,
and a metric that is undefined for a slice (ROC-AUC on a single-class
subgroup) is left ``None`` rather than silently dropped or faked.

The output is a flat, report-friendly table — one row per subgroup plus an
``overall`` row — built on the SAME metrics as
:func:`pharmaml.ml.evaluation.evaluate_model`, so a subgroup number can
never drift from the aggregate number it is meant to be compared against.

Examples
--------
A model that is perfect on the majority series and hopeless on a small one:

>>> import numpy as np
>>> from pharmaml.ml.subgroups import subgroup_report
>>> class EchoClassifier:   # reads the prediction out of the first feature column
...     def predict(self, X):
...         return np.asarray(X)[:, 0].astype(int)
...     def predict_proba(self, X):
...         pos = np.asarray(X)[:, 0].astype(float)
...         return np.column_stack([1 - pos, pos])
>>> y = np.array([0, 1, 0, 1, 0, 1])
>>> series = np.array(["A", "A", "A", "A", "B", "B"])
>>> X = np.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])  # wrong on both B rows
>>> report = subgroup_report(EchoClassifier(), X, y, series,
...                          task="classification", min_group_size=2)
>>> {s.value: round(s.metrics["accuracy"], 3) for s in report.subgroups}
{'A': 1.0, 'B': 0.0}

The worst group is named, not left to the reader:

>>> report.worst_group("accuracy").value
'B'

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.evaluation import (
    ClassificationReport,
    RegressionReport,
    evaluate_model,
)
from pharmaml.ml.multiplicity import CORRECTION_METHODS, correct_p_values
from pharmaml.ml.stats import (
    SignificanceResult,
    mann_whitney_test,
    two_sample_ttest,
)

__all__ = [
    "SubgroupGap",
    "SubgroupMetrics",
    "SubgroupReport",
    "subgroup_gap_scan",
    "subgroup_gap_test",
    "subgroup_report",
]

#: The metrics surfaced per subgroup, by task.  These are read straight off
#: the :func:`evaluate_model` reports so a subgroup number is the same
#: computation as the aggregate one.
_CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "mcc",
    "roc_auc",
)
_REGRESSION_METRICS: tuple[str, ...] = ("rmse", "mae", "r2", "pearson_r")

#: Metrics where a *lower* value is better — the direction :meth:`worst_group`
#: needs to rank groups correctly.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"rmse", "mae"})


@dataclass(frozen=True)
class SubgroupMetrics:
    """One subgroup's headline metrics (plus the aggregate it came from).

    Attributes
    ----------
    value:
        The subgroup label (a ``str`` rendering of the column value).
    n:
        Rows in this subgroup.
    metrics:
        The metric name -> value mapping for this slice.  A metric that is
        undefined for the slice (e.g. ``roc_auc`` on a single-class
        subgroup) is ``None`` — reported, never dropped.
    sufficient:
        ``False`` when ``n`` is below the report's ``min_group_size``; the
        metrics are then still computed but carry no verdict — a slice too
        small to mean anything is labelled, not trusted.
    overall:
        The metric mapping for the WHOLE held-out set, repeated on every
        row so a subgroup is directly comparable to the aggregate in one
        table without a join.
    """

    value: str
    n: int
    metrics: dict[str, float | None]
    sufficient: bool
    overall: dict[str, float | None] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        shown = ", ".join(
            f"{name}={value:.4f}" if value is not None else f"{name}=n/a"
            for name, value in self.metrics.items()
        )
        flag = "" if self.sufficient else "  [insufficient]"
        return f"{self.value} (n={self.n}): {shown}{flag}"


@dataclass(frozen=True)
class SubgroupReport:
    """Stratified performance across the subgroups of one column.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    column:
        The grouping column's name.
    subgroups:
        One :class:`SubgroupMetrics` per subgroup value, in first-seen order.
    overall:
        The metric mapping for the whole held-out set (includes
        ``n_samples``; ``rows_dropped`` when missing labels were skipped).
    metric_names:
        The metric keys present, in report order.
    min_group_size:
        The threshold below which a subgroup is marked ``insufficient``.
    """

    task: str
    column: str
    subgroups: tuple[SubgroupMetrics, ...]
    overall: dict[str, float | None]
    metric_names: tuple[str, ...]
    min_group_size: int = 10

    def __str__(self) -> str:  # pragma: no cover - presentation only
        header = (
            f"subgroups by {self.column!r} — {len(self.subgroups)} group(s), "
            f"overall n={self.overall.get('n_samples')}"
        )
        lines = [header]
        lines.extend("  " + str(group) for group in self.subgroups)
        return "\n".join(lines)

    def group(self, value: Any) -> SubgroupMetrics:
        """The subgroup whose label matches ``value`` (raises if absent)."""
        wanted = str(value)
        for group in self.subgroups:
            if group.value == wanted:
                return group
        raise DataValidationError(
            f"no subgroup {wanted!r}; known: "
            f"{', '.join(g.value for g in self.subgroups)}"
        )

    def worst_group(self, metric: str) -> SubgroupMetrics:
        """The subgroup with the weakest ``metric`` (respecting direction).

        Ties break toward the smaller group (the more fragile slice), so a
        two-row group that matches a two-hundred-row group on the nose is
        the one named.  Only ``sufficient`` groups are ranked; if none
        qualify, the smallest group overall is returned with its
        ``insufficient`` flag intact.

        Raises
        ------
        DataValidationError
            If ``metric`` is not one of the report's metrics, or no
            subgroup carries a numeric value for it.
        """
        if metric not in self.metric_names:
            raise DataValidationError(
                f"metric {metric!r} is not in this report; "
                f"known: {', '.join(self.metric_names)}"
            )
        pool = [
            g for g in self.subgroups
            if g.sufficient and g.metrics.get(metric) is not None
        ]
        if not pool:
            pool = [g for g in self.subgroups if g.metrics.get(metric) is not None]
        if not pool:
            raise DataValidationError(
                f"no subgroup carries a numeric {metric!r} to rank"
            )

        def _value(group: SubgroupMetrics) -> float:
            value = group.metrics[metric]
            assert value is not None
            return value

        if metric in _LOWER_IS_BETTER:
            return min(pool, key=lambda g: (-_value(g), g.n))
        return min(pool, key=lambda g: (_value(g), g.n))

    def to_frame(self) -> pd.DataFrame:
        """One row per subgroup (plus an ``overall`` row) for CSV/report use."""
        columns = ["subgroup", "n", "sufficient", *self.metric_names]
        rows: list[dict[str, Any]] = []
        for group in self.subgroups:
            row: dict[str, Any] = {
                "subgroup": group.value,
                "n": group.n,
                "sufficient": group.sufficient,
            }
            row.update({name: group.metrics.get(name) for name in self.metric_names})
            rows.append(row)
        overall_row: dict[str, Any] = {
            "subgroup": "overall",
            "n": self.overall.get("n_samples"),
            "sufficient": True,
        }
        overall_row.update({name: self.overall.get(name) for name in self.metric_names})
        rows.append(overall_row)
        return pd.DataFrame(rows, columns=columns)


def _finite_or_none(value: Any) -> float | None:
    """A finite float, or ``None`` for a value the slice cannot define.

    :func:`evaluate_model` reports an undefined metric (ROC-AUC on a
    single-class slice) as NaN; the subgroup report's contract is that
    "undefined" is ``None`` — explicit, JSON-safe, and skipped by
    :meth:`SubgroupReport.worst_group` rather than ranked as a number.
    """
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _metric_mapping(
    report: ClassificationReport | RegressionReport, task: str
) -> dict[str, float | None]:
    """Read the headline metrics off an :func:`evaluate_model` report."""
    if task == "classification":
        assert isinstance(report, ClassificationReport)
        return {
            "accuracy": float(report.accuracy),
            "balanced_accuracy": float(report.balanced_accuracy),
            "f1_macro": float(report.f1_macro),
            "precision_macro": float(report.precision_macro),
            "recall_macro": float(report.recall_macro),
            "mcc": float(report.mcc),
            "roc_auc": _finite_or_none(report.roc_auc),
            "n_samples": int(report.n_samples),
        }
    assert isinstance(report, RegressionReport)
    return {
        "rmse": float(report.rmse),
        "mae": float(report.mae),
        "r2": float(report.r2),
        "pearson_r": _finite_or_none(report.pearson_r),
        "n_samples": int(report.n_samples),
    }


def _check_task(task: str) -> str:
    if task not in ("classification", "regression"):
        raise DataValidationError(
            f"task must be 'classification' or 'regression', got {task!r}"
        )
    return task


def subgroup_report(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: Any,
    groups: pd.Series | np.ndarray | list,
    *,
    task: str = "classification",
    min_group_size: int = 10,
    column: str | None = None,
) -> SubgroupReport:
    """Stratified performance: the headline metrics per subgroup and overall.

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator.
    X:
        Held-out features (a DataFrame or an array), sliced positionally.
    y_true:
        Held-out targets, aligned with ``X``.
    groups:
        The grouping labels — one per row.  Every distinct value becomes a
        subgroup; rows with a missing (NaN/None) label are dropped and
        counted, never silently folded into a phantom group.
    task:
        ``'classification'`` or ``'regression'``.
    min_group_size:
        Subgroups smaller than this are reported but flagged
        ``insufficient`` (their metrics are still computed for transparency).
    column:
        The grouping column's name, used in the report header / ``to_frame``.

    Returns
    -------
    SubgroupReport

    Raises
    ------
    DataValidationError
        On an unknown task, a negative ``min_group_size``, a length
        mismatch between ``X`` / ``y_true`` / ``groups``, or a grouping
        column that is entirely missing.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.subgroups import subgroup_report
    >>> class Reg:                       # predicts the constant 0
    ...     def predict(self, X):
    ...         return np.zeros(len(X))
    >>> y = np.array([0.0, 0.0, 5.0, 5.0])
    >>> batch = np.array(["b1", "b1", "b2", "b2"])
    >>> report = subgroup_report(Reg(), np.zeros((4, 1)), y, batch,
    ...                          task="regression", min_group_size=2)
    >>> round(report.group("b2").metrics["rmse"], 3)
    5.0
    >>> round(report.overall["rmse"], 3)
    3.536
    """
    task = _check_task(task)
    if min_group_size < 0:
        raise DataValidationError(
            f"min_group_size must be non-negative, got {min_group_size}"
        )

    y = np.asarray(y_true)
    if y.size == 0:
        raise DataValidationError("y_true is empty")
    if len(X) != y.size:
        raise DataValidationError(f"X has {len(X)} row(s) but y_true has {y.size}")

    labels = np.asarray(groups)
    if labels.size != y.size:
        raise DataValidationError(
            f"groups has {labels.size} row(s) but y_true has {y.size}"
        )
    label_series = pd.Series(labels, index=range(y.size))
    if label_series.isna().all():
        raise DataValidationError(
            "the grouping column is entirely missing — nothing to stratify by"
        )
    kept = ~label_series.isna().to_numpy()
    dropped = int((~kept).sum())

    metric_names = (
        _CLASSIFICATION_METRICS if task == "classification" else _REGRESSION_METRICS
    )

    def _slice(index: np.ndarray) -> Any:
        if isinstance(X, pd.DataFrame):
            return X.iloc[index]
        return np.asarray(X)[index]

    overall_report = evaluate_model(model, _slice(np.flatnonzero(kept)), y[kept], task)
    overall = _metric_mapping(overall_report, task)
    if dropped:
        overall = {**overall, "rows_dropped": dropped}

    subgroups: list[SubgroupMetrics] = []
    for value in pd.unique(label_series.dropna()):
        mask = (labels == value) & kept
        index = np.flatnonzero(mask)
        slice_report = evaluate_model(model, _slice(index), y[index], task)
        slice_metrics = _metric_mapping(slice_report, task)
        subgroups.append(
            SubgroupMetrics(
                value=str(value),
                n=int(index.size),
                metrics={name: slice_metrics.get(name) for name in metric_names},
                sufficient=index.size >= min_group_size,
                overall=overall,
            )
        )

    return SubgroupReport(
        task=task,
        column=column or "groups",
        subgroups=tuple(subgroups),
        overall=overall,
        metric_names=metric_names,
        min_group_size=min_group_size,
    )


@dataclass(frozen=True)
class SubgroupGap:
    """Whether one subgroup's performance gap is real, or could be chance.

    :meth:`SubgroupReport.worst_group` names the weakest slice, but a
    handful of rows can land there by luck — and "the model fails this
    assay" is a claim that deserves a number.  This record pairs the
    descriptive gap (the subgroup's metric minus the rest's) with a
    two-sample test on the *per-row* evidence: a contingency-style
    comparison of right vs wrong for classification, or of absolute error
    for regression.

    Attributes
    ----------
    value:
        The subgroup label tested.
    metric:
        The metric the gap is expressed in.
    n:
        Rows in the subgroup.
    n_rest:
        Rows in the rest of the held-out set (the comparison sample).
    subgroup_value:
        The subgroup's metric (may be ``None`` if undefined for the slice).
    rest_value:
        The same metric computed over the rest.
    delta:
        ``subgroup_value - rest_value`` (negative = worse; ``None`` if
        either side is undefined).
    direction:
        ``'worse'``, ``'better'`` or ``'equal'`` — the sign of ``delta``
        read through the metric's direction (lower is better for RMSE/MAE).
    test:
        The :class:`~pharmaml.ml.stats.SignificanceResult`, or ``None``
        when either side is too small to test (the honest "cannot say").
    """

    value: str
    metric: str
    n: int
    n_rest: int
    subgroup_value: float | None
    rest_value: float | None
    delta: float | None
    direction: str
    test: SignificanceResult | None
    fdr_p_value: float | None = None
    fdr_significant: bool | None = None

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if self.delta is None:
            return f"{self.value!r}: {self.metric} undefined on one side"
        gap = f"{self.metric} {self.delta:+.4f} ({self.direction})"
        if self.test is None:
            return f"{self.value!r}: {gap} — too few rows to test"
        verdict = "significant" if self.test.significant else "not significant"
        line = (
            f"{self.value!r}: {gap} · {self.test.test} p={self.test.p_value:.4g} "
            f"· {verdict} at α {self.test.alpha}"
        )
        if self.fdr_p_value is not None:
            fdr = "significant" if self.fdr_significant else "not significant"
            line += f" · FDR q={self.fdr_p_value:.4g} ({fdr})"
        return line


def _classification_evidence(model: Any, X: Any, y: np.ndarray) -> np.ndarray:
    """Per-row 1/0 for "the prediction was right" — the gap test's sample."""
    pred = np.asarray(model.predict(X))
    return (pred == y).astype(float)


def _regression_evidence(model: Any, X: Any, y: np.ndarray) -> np.ndarray:
    """Per-row absolute error — the regression gap test's sample."""
    pred = np.asarray(model.predict(X), dtype=float)
    return np.abs(pred - np.asarray(y, dtype=float))


def subgroup_gap_test(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: Any,
    groups: pd.Series | np.ndarray | list,
    value: Any,
    *,
    task: str = "classification",
    metric: str | None = None,
    min_group_size: int = 10,
    alpha: float = 0.05,
    nonparametric: bool = False,
) -> SubgroupGap:
    """Is one subgroup's performance gap versus the rest statistically real?

    Scores ``model`` on ``X`` / ``y_true``, splits the rows into the
    ``value`` subgroup and the rest, and compares them on the per-row
    evidence — for classification, whether each prediction was correct; for
    regression, the absolute error.  The evidence samples feed
    :func:`~pharmaml.ml.stats.two_sample_ttest` (Welch by default) or
    :func:`~pharmaml.ml.stats.mann_whitney_test` when ``nonparametric``.

    The default ``metric`` is ``accuracy`` for classification and ``rmse``
    for regression; the reported ``delta`` is subgroup minus rest in that
    metric, read through its direction (``'worse'`` for a lower accuracy or
    a higher RMSE).  If either side is smaller than ``min_group_size`` the
    test is skipped (``test is None``) rather than run on noise — the same
    "cannot say" contract as the rest of the trust layer.

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator.
    X, y_true:
        The held-out features and targets.
    groups:
        One grouping label per row.
    value:
        The subgroup label to test against the rest.
    task:
        ``'classification'`` or ``'regression'``.
    metric:
        The metric the gap is expressed in (default: accuracy / rmse).
    min_group_size:
        Minimum rows per side before a test is attempted.
    alpha:
        Significance level handed to the test.
    nonparametric:
        Use Mann-Whitney U instead of the t-test.

    Returns
    -------
    SubgroupGap

    Raises
    ------
    DataValidationError
        On an unknown task, a length mismatch, or a ``value`` that matches
        no row.

    Examples
    --------
    A subgroup the model gets entirely wrong is flagged worse, and — with
    enough rows — significantly so:

    >>> import numpy as np
    >>> from pharmaml.ml.subgroups import subgroup_gap_test
    >>> class Echo:                      # right when the feature equals y
    ...     def predict(self, X):
    ...         return np.asarray(X)[:, 0].astype(int)
    >>> y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    >>> X = np.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]])
    >>> series = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    >>> gap = subgroup_gap_test(Echo(), X, y, series, "B",
    ...                         task="classification", min_group_size=2)
    >>> gap.direction, gap.delta
    ('worse', -1.0)
    """
    task = _check_task(task)
    if metric is None:
        metric = "accuracy" if task == "classification" else "rmse"
    _check_metric(task, metric)

    y = np.asarray(y_true)
    if len(X) != y.size:
        raise DataValidationError(f"X has {len(X)} row(s) but y_true has {y.size}")
    labels = np.asarray(groups)
    if labels.size != y.size:
        raise DataValidationError(
            f"groups has {labels.size} row(s) but y_true has {y.size}"
        )
    if not (labels == value).any():
        raise DataValidationError(
            f"no row has group {value!r}; cannot test a gap for it"
        )
    return _gap_for_group(
        model, X, y, labels, value,
        task=task, metric=metric, min_group_size=min_group_size,
        alpha=alpha, nonparametric=nonparametric,
    )


def _check_metric(task: str, metric: str) -> None:
    """Reject a metric that does not belong to ``task``'s metric family."""
    known = _CLASSIFICATION_METRICS if task == "classification" else _REGRESSION_METRICS
    if metric not in known:
        raise DataValidationError(
            f"metric {metric!r} is not defined for task {task!r}; "
            f"known: {', '.join(known)}"
        )


def _slice_rows(X: pd.DataFrame | np.ndarray, index: np.ndarray) -> Any:
    """Positionally select rows from a DataFrame or array."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[index]
    return np.asarray(X)[index]


def _gap_for_group(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    value: Any,
    *,
    task: str,
    metric: str,
    min_group_size: int,
    alpha: float,
    nonparametric: bool,
) -> SubgroupGap:
    """One subgroup's gap versus the rest — the shared core of test and scan."""
    in_group = labels == value

    def _metric_on(index: np.ndarray) -> float | None:
        report = evaluate_model(model, _slice_rows(X, index), y[index], task)
        return _metric_mapping(report, task).get(metric)

    group_index = np.flatnonzero(in_group)
    rest_index = np.flatnonzero(~in_group)
    subgroup_value = _metric_on(group_index)
    rest_value = _metric_on(rest_index) if rest_index.size else None
    delta = (
        None
        if subgroup_value is None or rest_value is None
        else float(subgroup_value - rest_value)
    )
    if delta is None:
        direction = "unknown"
    elif delta == 0.0:
        direction = "equal"
    elif (delta < 0) != (metric in _LOWER_IS_BETTER):
        direction = "worse"
    else:
        direction = "better"

    evidence = (
        _classification_evidence if task == "classification" else _regression_evidence
    )
    a = evidence(model, _slice_rows(X, group_index), y[group_index])
    b = evidence(model, _slice_rows(X, rest_index), y[rest_index])
    test: SignificanceResult | None = None
    if a.size >= min_group_size and b.size >= min_group_size:
        runner = mann_whitney_test if nonparametric else two_sample_ttest
        test = runner(a, b, alpha=alpha)

    return SubgroupGap(
        value=str(value),
        metric=metric,
        n=int(group_index.size),
        n_rest=int(rest_index.size),
        subgroup_value=subgroup_value,
        rest_value=rest_value,
        delta=delta,
        direction=direction,
        test=test,
    )


def subgroup_gap_scan(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: Any,
    groups: pd.Series | np.ndarray | list,
    *,
    task: str = "classification",
    metric: str | None = None,
    min_group_size: int = 10,
    alpha: float = 0.05,
    nonparametric: bool = False,
    correction: str = "fdr_bh",
) -> tuple[SubgroupGap, ...]:
    """Test *every* subgroup's gap versus the rest, correcting for multiplicity.

    Testing one subgroup invites cherry-picking; testing all of them at once
    is a family of comparisons, so the raw p-values are corrected before any
    is called significant.  Each subgroup is compared against the rest with
    :func:`subgroup_gap_test`'s machinery; the resulting p-values are then
    adjusted across the family with the chosen method (``'fdr_bh'`` —
    Benjamini-Hochberg, the default — or ``'bonferroni'``/``'holm'`` via
    :func:`statsmodels.stats.multitest.multipletests`) and each
    :class:`SubgroupGap` carries its ``fdr_p_value`` and ``fdr_significant``.

    Subgroups too small to test on either side carry no p-value and are
    simply left out of the correction family (they are reported with
    ``test is None``), so the family size is honest rather than padded.

    Returns the gaps in first-seen subgroup order, each annotated with its
    corrected verdict.

    Parameters
    ----------
    model, X, y_true, groups:
        As for :func:`subgroup_gap_test`.
    task, metric, min_group_size, nonparametric:
        As for :func:`subgroup_gap_test`.
    alpha:
        Family-wise / false-discovery level for the corrected verdict.
    correction:
        ``'fdr_bh'`` (default), ``'bonferroni'`` or ``'holm'``.

    Returns
    -------
    tuple[SubgroupGap, ...]

    Raises
    ------
    DataValidationError
        On an unknown task or metric, a length mismatch, or an unknown
        correction method.

    Examples
    --------
    Five identical good series and one bad one.  Each good series clears
    alpha on its own (its six wrong rows give p ≈ 0.016), but the family-wide
    Bonferroni bar (×6) drops them; only the genuinely bad series survives:

    >>> import numpy as np
    >>> from pharmaml.ml.subgroups import subgroup_gap_scan
    >>> class Echo:                      # right when the feature equals y
    ...     def predict(self, X):
    ...         return np.asarray(X)[:, 0].astype(int)
    >>> n = 80
    >>> y = np.tile([0, 1], 240)                       # 480 rows
    >>> series = np.repeat(["g0", "g1", "g2", "g3", "g4", "bad"], n)
    >>> correct = np.ones(480, dtype=bool)
    >>> for s in range(0, 400, n):                     # 6 wrong rows per good group
    ...     correct[s + n - 6:s + n] = False
    >>> correct[440:] = False                          # 'bad' is half wrong
    >>> X = np.where(correct, y, 1 - y).reshape(-1, 1).astype(float)
    >>> gaps = subgroup_gap_scan(Echo(), X, y, series, min_group_size=10,
    ...                          correction="bonferroni")
    >>> [(g.value, g.test.significant, g.fdr_significant) for g in gaps]
    [('g0', True, False), ('g1', True, False), ('g2', True, False), ('g3', True, False), ('g4', True, False), ('bad', True, True)]
    """
    task = _check_task(task)
    if metric is None:
        metric = "accuracy" if task == "classification" else "rmse"
    _check_metric(task, metric)
    if correction not in CORRECTION_METHODS:
        raise DataValidationError(
            f"correction must be one of {', '.join(sorted(CORRECTION_METHODS))}, "
            f"got {correction!r}"
        )

    y = np.asarray(y_true)
    if y.size == 0:
        raise DataValidationError("y_true is empty")
    if len(X) != y.size:
        raise DataValidationError(f"X has {len(X)} row(s) but y_true has {y.size}")
    labels = np.asarray(groups)
    if labels.size != y.size:
        raise DataValidationError(
            f"groups has {labels.size} row(s) but y_true has {y.size}"
        )
    label_series = pd.Series(labels)
    if label_series.isna().all():
        raise DataValidationError(
            "the grouping column is entirely missing — nothing to stratify by"
        )
    kept = ~label_series.isna().to_numpy()
    X_kept = _slice_rows(X, np.flatnonzero(kept))

    gaps = [
        _gap_for_group(
            model, X_kept, y[kept], labels[kept], value,
            task=task, metric=metric, min_group_size=min_group_size,
            alpha=alpha, nonparametric=nonparametric,
        )
        for value in pd.unique(label_series.dropna())
    ]

    # A t-test on zero-variance evidence returns a NaN p-value; such a
    # subgroup carries no test result, so it stays out of the correction
    # family (its ``fdr_p_value`` is left ``None``) rather than poisoning it.
    family = [
        (i, float(gap.test.p_value))
        for i, gap in enumerate(gaps)
        if gap.test is not None and np.isfinite(gap.test.p_value)
    ]
    if family:
        corrected = correct_p_values(
            np.array([p for _, p in family], dtype=float), correction
        )
        adjusted = list(gaps)
        for (i, _), q in zip(family, corrected):
            adjusted[i] = replace(
                adjusted[i], fdr_p_value=float(q),
                fdr_significant=bool(q < alpha),
            )
        gaps = adjusted
    return tuple(gaps)

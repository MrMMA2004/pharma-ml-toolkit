"""Is the leaderboard's top model *really* ahead?

:func:`pharmaml.ml.cv_results.compare_models_cv` ranks a model panel by
its mean cross-validated score, but a mean is a point estimate: a 0.01
AUROC lead between the top two rows may be fold-to-fold noise.  This
module supplies the missing verdict — a *paired* test of every challenger
against the leader, run on the per-fold scores the two share (the same
folds, because :func:`~pharmaml.ml.cv_results.compare_models_cv` holds the
splitter fixed across the panel).

Two tests are offered, both paired on the fold scores:

* ``paired_t`` (default) — Dietterich's paired t-test, the textbook
  k-fold comparison.  Assumes the fold differences are roughly normal.
* ``wilcoxon`` — the signed-rank test, robust to a skewed difference
  vector (one fold where a model collapses).  Honest caveat: the exact
  two-sided p-value floors at ``2·2^{-n}``, so with the usual 5 folds it
  can never reach α = 0.05 (the floor is 0.0625); it needs at least 6
  folds to have any chance, and more to be useful.

Because every challenger is tested against the same leader, the family of
p-values is corrected for multiplicity (:mod:`pharmaml.ml.multiplicity`)
— the same discipline :func:`~pharmaml.ml.subgroups.subgroup_gap_scan`
applies to subgroups.  A challenger that looked significant raw can be
demoted once its siblings are counted.

The honest reading of an "on par" verdict is *absence of evidence*, not
evidence of equivalence: with few folds a real but small gap is simply
underpowered to detect.  The verdict says "the leaderboard cannot resolve
these two at this fold count", which is the truthful statement.  The
companion question — *how* underpowered, and how many folds it would
take — is answered by :meth:`ModelComparison.power` (see
:mod:`pharmaml.ml.power`).

Examples
--------
>>> from pharmaml.ml.comparison import model_comparison_test
>>> from pharmaml.ml.cv_results import CVResult, MetricSummary
>>> def _result(name, folds):
...     summary = MetricSummary(
...         name="roc_auc_ovo", mean=sum(folds) / len(folds), std=0.0,
...         folds=list(folds), higher_is_better=True, lower_is_better=False)
...     return CVResult(
...         model=name, task="classification", strategy="stratified",
...         n_folds=len(folds), seed=42, metrics={"roc_auc_ovo": summary},
...         primary="roc_auc_ovo", duration_s=0.0)
>>> leader = _result("rf", [0.92, 0.93, 0.91, 0.94, 0.92, 0.93])
>>> close = _result("logistic", [0.915, 0.905, 0.918, 0.902, 0.93, 0.905])
>>> behind = _result("tree", [0.80, 0.83, 0.78, 0.85, 0.79, 0.82])
>>> comparison = model_comparison_test([leader, close, behind])
>>> comparison.leader
'rf'
>>> [(row.model, row.verdict) for row in comparison.rows]
[('logistic', 'on par'), ('tree', 'worse')]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.cv_results import CVResult
from pharmaml.ml.multiplicity import CORRECTION_METHODS, correct_p_values
from pharmaml.ml.power import DEFAULT_TARGET_POWER, PowerReport, power_report
from pharmaml.ml.stats import SignificanceResult, paired_ttest, paired_wilcoxon_test

__all__ = [
    "COMPARISON_TESTS",
    "ModelComparison",
    "ModelComparisonRow",
    "model_comparison_test",
]

#: The paired tests :func:`model_comparison_test` accepts.
COMPARISON_TESTS: tuple[str, ...] = ("paired_t", "wilcoxon")


@dataclass(frozen=True)
class ModelComparisonRow:
    """One challenger's paired verdict against the leader.

    Attributes
    ----------
    model:
        The challenger's registry name.
    mean:
        The challenger's mean primary-metric value across folds.
    leader_mean:
        The leader's mean, repeated here so the row stands alone.
    difference:
        The challenger's mean minus the leader's, on the stored scorer
        scale — a *positive* value favours the challenger.  Every stored
        fold score follows sklearn's convention (higher is better; loss
        metrics carry a ``neg_`` prefix), so no direction flip is needed.
    test:
        The paired :class:`~pharmaml.ml.stats.SignificanceResult`, or
        ``None`` when the pair could not be tested (too few usable folds).
    adjusted_p:
        The multiplicity-adjusted p-value across the challenger family,
        or ``None`` alongside a ``None`` test.
    note:
        Why the pair was left untested, when it was.
    """

    model: str
    mean: float
    leader_mean: float
    difference: float
    test: SignificanceResult | None
    adjusted_p: float | None
    note: str = ""

    @property
    def verdict(self) -> str:
        """``'worse'``, ``'better'``, ``'on par'`` or ``'untested'``.

        ``'on par'`` means the leader's edge over this challenger is not
        resolvable at the tested fold count (absence of evidence); it is
        *not* a claim of equivalence.
        """
        if self.test is None or self.adjusted_p is None:
            return "untested"
        if self.adjusted_p >= self.test.alpha:
            return "on par"
        return "worse" if self.difference < 0.0 else "better"

    def line(self) -> str:
        """A one-line table row for the CLI and reports."""
        p = "—" if self.adjusted_p is None else f"{self.adjusted_p:.4g}"
        return (
            f"{self.model:<22} {self.mean:>10.4f}  {self.difference:>+9.4f}  "
            f"{self.verdict:<8} q={p}"
        )


@dataclass(frozen=True)
class ModelComparison:
    """The leader and every challenger's corrected paired verdict.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    metric:
        The primary scorer name the comparison ran on.
    label:
        The metric's display label (e.g. ``'ROC-AUC (one-vs-one)'``).
    leader:
        The registry name of the model with the best mean.
    leader_mean:
        That model's mean primary-metric value.
    n_folds:
        The fold count the panel was cross-validated at.
    alpha, correction:
        The significance level and the multiplicity method applied.
    test:
        The paired test used (``'paired_t'`` or ``'wilcoxon'``).
    rows:
        One :class:`ModelComparisonRow` per challenger, best mean first.
    notes:
        Honesty notes gathered while testing (untestable pairs,
        degenerate differences, fold-count floors).
    """

    task: str
    metric: str
    label: str
    leader: str
    leader_mean: float
    n_folds: int
    alpha: float
    correction: str
    test: str
    rows: tuple[ModelComparisonRow, ...] = ()
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """The headline: how many challengers the leader *really* beats."""
        tested = [row for row in self.rows if row.test is not None]
        worse = sum(row.verdict == "worse" for row in tested)
        return (
            f"{self.leader} leads on {self.label} ({self.leader_mean:.4f}); "
            f"significantly ahead of {worse}/{len(tested)} tested challenger(s) "
            f"({self.correction} at α {self.alpha:g}, {self.test})"
        )

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lines = [self.summary()]
        header = f"{'model':<22} {'mean':>10}  {'diff':>9}  {'verdict':<8} q"
        lines.append(header)
        lines.extend(row.line() for row in self.rows)
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    def power(
        self,
        *,
        target_power: float = DEFAULT_TARGET_POWER,
    ) -> PowerReport:
        """The fold-count power picture behind these verdicts.

        An ``on par`` row means the leaderboard could not resolve the two
        at this fold count; this says *how* underpowered the panel was.
        The observed effect is the largest standardized fold-difference
        among the tested challengers (Cohen's d, the same effect
        :func:`~pharmaml.ml.stats.paired_ttest` reports) — the strongest
        signal the panel actually carried, so its power is the most
        favourable honest reading.

        Examples
        --------
        >>> from pharmaml.ml.comparison import model_comparison_test
        >>> from pharmaml.ml.cv_results import CVResult, MetricSummary
        >>> def _result(name, folds):
        ...     summary = MetricSummary(
        ...         name="r2", mean=sum(folds) / len(folds), std=0.0,
        ...         folds=list(folds), higher_is_better=True, lower_is_better=False)
        ...     return CVResult(
        ...         model=name, task="regression", strategy="kfold",
        ...         n_folds=len(folds), seed=42, metrics={"r2": summary},
        ...         primary="r2", duration_s=0.0)
        >>> strong = _result("gbr", [0.90, 0.91, 0.89, 0.92, 0.90, 0.91])
        >>> weak = _result("linear", [0.50, 0.55, 0.48, 0.60, 0.52, 0.54])
        >>> report = model_comparison_test([strong, weak]).power()
        >>> report.n_folds, report.sufficient
        (6, True)
        """
        effects = [
            abs(row.test.effect_value)
            for row in self.rows
            if row.test is not None and row.test.effect_value is not None
        ]
        return power_report(
            self.n_folds,
            alpha=self.alpha,
            target_power=target_power,
            observed_effect=max(effects) if effects else None,
        )

    def to_frame(self) -> pd.DataFrame:
        """The verdicts as a tidy frame, one row per challenger.

        Columns: ``model``, ``mean``, ``leader_mean``, ``difference``,
        ``p_value``, ``adjusted_p``, ``verdict`` and ``note`` — the last
        two as text, so the frame exports cleanly to csv/markdown.

        Examples
        --------
        >>> from pharmaml.ml.comparison import model_comparison_test
        >>> from pharmaml.ml.cv_results import CVResult, MetricSummary
        >>> def _result(name, folds):
        ...     summary = MetricSummary(
        ...         name="r2", mean=sum(folds) / len(folds), std=0.0,
        ...         folds=list(folds), higher_is_better=True, lower_is_better=False)
        ...     return CVResult(
        ...         model=name, task="regression", strategy="kfold",
        ...         n_folds=len(folds), seed=42, metrics={"r2": summary},
        ...         primary="r2", duration_s=0.0)
        >>> strong = _result("gbr", [0.90, 0.91, 0.89, 0.92, 0.90, 0.91])
        >>> weak = _result("linear", [0.50, 0.55, 0.48, 0.60, 0.52, 0.54])
        >>> list(model_comparison_test([strong, weak]).to_frame().columns)
        ['model', 'mean', 'leader_mean', 'difference', 'p_value', 'adjusted_p', 'verdict', 'note']
        """
        return pd.DataFrame(
            [
                {
                    "model": row.model,
                    "mean": row.mean,
                    "leader_mean": row.leader_mean,
                    "difference": row.difference,
                    "p_value": None if row.test is None else row.test.p_value,
                    "adjusted_p": row.adjusted_p,
                    "verdict": row.verdict,
                    "note": row.note,
                }
                for row in self.rows
            ]
        )


def _primary_folds(result: CVResult, metric: str) -> np.ndarray:
    """The per-fold primary-metric values for one model, as floats."""
    summary = result.metrics.get(metric)
    if summary is None:
        raise ModelError(
            f"model {result.model!r} has no {metric!r} metric — the panel "
            "must be cross-validated with a common primary scorer"
        )
    return np.asarray(summary.folds, dtype=float)


def model_comparison_test(
    results: list[CVResult],
    *,
    primary: str | None = None,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
    test: str = "paired_t",
) -> ModelComparison:
    """Test every model against the leaderboard leader, correcting the family.

    Parameters
    ----------
    results:
        The per-model :class:`~pharmaml.ml.cv_results.CVResult` objects
        from :func:`~pharmaml.ml.cv_results.compare_models_cv` (one per
        model, sharing the same folds).
    primary:
        The scorer to compare on; defaults to the panel's primary scorer
        (taken from the first result).  Must be present on every model.
    alpha:
        The family-wide significance level.
    correction:
        The multiplicity method: ``'fdr_bh'`` (default), ``'bonferroni'``
        or ``'holm'``.
    test:
        ``'paired_t'`` (default) or ``'wilcoxon'``.

    Returns
    -------
    ModelComparison
        The leader plus a corrected verdict per challenger.

    Raises
    ------
    ModelError
        On an empty panel or a panel whose models disagree on the primary
        scorer.
    DataValidationError
        On a bad alpha, correction or test name.

    Examples
    --------
    >>> from pharmaml.ml.comparison import model_comparison_test
    >>> from pharmaml.ml.cv_results import CVResult, MetricSummary
    >>> def _result(name, folds):
    ...     summary = MetricSummary(
    ...         name="r2", mean=sum(folds) / len(folds), std=0.0,
    ...         folds=list(folds), higher_is_better=True, lower_is_better=False)
    ...     return CVResult(
    ...         model=name, task="regression", strategy="kfold",
    ...         n_folds=len(folds), seed=42, metrics={"r2": summary},
    ...         primary="r2", duration_s=0.0)
    >>> strong = _result("gbr", [0.90, 0.91, 0.89, 0.92, 0.90, 0.91])
    >>> weak = _result("linear", [0.50, 0.55, 0.48, 0.60, 0.52, 0.54])
    >>> model_comparison_test([strong, weak]).rows[0].verdict
    'worse'
    """
    if not results:
        raise ModelError("model_comparison_test needs at least one CVResult")
    if correction not in CORRECTION_METHODS:
        raise DataValidationError(
            f"correction must be one of {', '.join(CORRECTION_METHODS)}, got {correction!r}"
        )
    if test not in COMPARISON_TESTS:
        raise DataValidationError(
            f"test must be one of {', '.join(COMPARISON_TESTS)}, got {test!r}"
        )
    if not 0.0 < alpha < 1.0:
        raise DataValidationError(f"alpha must be in (0, 1), got {alpha}")

    metric = primary or results[0].primary
    metric_summary = results[0].metrics.get(metric)
    if metric_summary is None:
        raise ModelError(f"the panel has no {metric!r} metric to compare on")
    label = metric_summary.label
    task = results[0].task

    folds_by_model = {result.model: _primary_folds(result, metric) for result in results}
    n_folds = len(next(iter(folds_by_model.values())))
    notes: list[str] = []

    # Stored fold scores follow sklearn's convention: higher is always
    # better (loss metrics carry a neg_ prefix, so a larger value is a
    # smaller error).  Rank and diff on that scale directly.
    def _rank_key(result: CVResult) -> float:
        return float(np.nanmean(folds_by_model[result.model]))

    ordered = sorted(results, key=_rank_key, reverse=True)
    leader = ordered[0]
    leader_folds = folds_by_model[leader.model]
    leader_mean = float(np.nanmean(leader_folds))

    if test == "wilcoxon" and n_folds < 6:
        notes.append(
            f"the signed-rank test floors at two-sided p = {2 * 2.0**-n_folds:.4g} "
            f"with {n_folds} folds, so it cannot reach α = {alpha:g} — use "
            "'paired_t' or more folds for a verdict"
        )

    rows: list[ModelComparisonRow] = []
    family: list[float] = []
    pending_indices: list[int] = []
    for challenger in ordered[1:]:
        challenger_folds = folds_by_model[challenger.model]
        usable = np.isfinite(challenger_folds) & np.isfinite(leader_folds)
        a = challenger_folds[usable]
        b = leader_folds[usable]
        difference = float(np.nanmean(challenger_folds) - leader_mean)
        if a.size < 2:
            row = ModelComparisonRow(
                model=challenger.model,
                mean=float(np.nanmean(challenger_folds)),
                leader_mean=leader_mean,
                difference=difference,
                test=None,
                adjusted_p=None,
                note=f"only {a.size} finite fold pair(s) — needs at least 2",
            )
            notes.append(f"{challenger.model}: {row.note}")
            rows.append(row)
            continue
        if np.all(a == b):
            row = ModelComparisonRow(
                model=challenger.model,
                mean=float(np.nanmean(challenger_folds)),
                leader_mean=leader_mean,
                difference=difference,
                test=None,
                adjusted_p=None,
                note="every fold identical to the leader — the paired test is undefined",
            )
            notes.append(f"{challenger.model}: {row.note}")
            rows.append(row)
            continue
        try:
            if test == "wilcoxon":
                outcome = paired_wilcoxon_test(a, b, alpha=alpha)
            else:
                outcome = paired_ttest(a, b, alpha=alpha)
        except DataValidationError as exc:  # zero-variance differences etc.
            row = ModelComparisonRow(
                model=challenger.model,
                mean=float(np.nanmean(challenger_folds)),
                leader_mean=leader_mean,
                difference=difference,
                test=None,
                adjusted_p=None,
                note=str(exc),
            )
            notes.append(f"{challenger.model}: {row.note}")
            rows.append(row)
            continue
        row = ModelComparisonRow(
            model=challenger.model,
            mean=float(np.nanmean(challenger_folds)),
            leader_mean=leader_mean,
            difference=difference,
            test=outcome,
            adjusted_p=outcome.p_value,
        )
        pending_indices.append(len(rows))
        rows.append(row)
        family.append(outcome.p_value)

    if family:
        corrected = correct_p_values(np.asarray(family, dtype=float), correction)
        for index, q in zip(pending_indices, corrected):
            row = rows[index]
            rows[index] = ModelComparisonRow(
                model=row.model,
                mean=row.mean,
                leader_mean=row.leader_mean,
                difference=row.difference,
                test=row.test,
                adjusted_p=float(q),
                note=row.note,
            )

    return ModelComparison(
        task=task,
        metric=metric,
        label=label,
        leader=leader.model,
        leader_mean=leader_mean,
        n_folds=n_folds,
        alpha=float(alpha),
        correction=correction,
        test=test,
        rows=tuple(rows),
        notes=notes,
    )

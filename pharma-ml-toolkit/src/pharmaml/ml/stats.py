"""Statistics for data science: tests, intervals, distributions.

The ``SENATOROVAI/Data-Science-For-Beginners-from-scratch-course``
repository was reviewed file by file for this round.  Its curriculum
outline promises the statistics block every data-science course
teaches -- "Probability, Distributions, Hypothesis testing,
Confidence intervals" -- and the toolkit had none of it as a surface:
the biology modules test *associations* (correlation-style), the QSAR
side reports *model* metrics, and nobody answers the plain question
"are these two groups actually different?".  This module is that
block, the way the course orders it:

* :func:`describe_series` -- the descriptive-statistics lecture:
  n, mean, std, SEM, the five-number summary, IQR, skew, kurtosis,
  missing counts (scipy's moment machinery).
* :func:`mean_confidence_interval` / :func:`bootstrap_interval` -- the
  confidence-interval lecture, both canonical forms: the t-based
  ``mean ± t_{n-1, 1-α/2}·s/√n`` and the percentile bootstrap (seeded
  resampling, the course's "practical" interval for anything
  non-normal).
* :func:`one_sample_ttest`, :func:`paired_ttest`,
  :func:`two_sample_ttest` (Student and Welch), and
  :func:`mann_whitney_test` -- the hypothesis-testing lecture: the
  four t-tests plus the rank-based escape hatch, each returning a
  :class:`SignificanceResult` that carries the statistic, the p-value, the
  degrees of freedom, the decision at α, and an effect size.
* :func:`normality_report` -- Shapiro-Wilk plus D'Agostino-Pearson
  (the two the course leans on), because "which test do I use?" starts
  with "are the data normal?".
* :func:`sample_distribution` -- the probability lecture's sampler:
  normal, uniform, binomial, Poisson and exponential draws from a
  seeded generator, with the theoretical moments attached so the
  histogram card can draw the population against the sample.
* :func:`compare_groups` -- the whole course flow in one call:
  descriptives → normality check → the right test (Welch /
  Mann-Whitney) → both confidence intervals → effect size → the
  verdict sentence, with every branch it took reported so nothing is
  a black box.

The honest scope, stated once: the repository's *other* tracks
(Python fundamentals, pandas wrangling, EDA cleaning, the
from-scratch gradient-descent models) were already covered by the
Data & ML tab's loaders and :mod:`pharmaml.ml.tensors`; its ``docs/git``
tutorials land in the Utilities tab's Git quick-reference card; its
pre-commit tooling is repository hygiene, not data science, and stays
out.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.stats import (
...     bootstrap_interval, compare_groups, describe_series,
...     mean_confidence_interval, sample_distribution,
... )
>>> rng = np.random.default_rng(42)
>>> treated = rng.normal(1.2, 0.4, 40)
>>> control = rng.normal(1.0, 0.4, 40)
>>> summary = describe_series(treated)
>>> round(summary.mean, 1) == 1.2 or round(summary.mean, 1) == 1.3   # near the truth
True
>>> interval = mean_confidence_interval(treated, level=0.95)
>>> interval.lower < interval.estimate < interval.upper
True
>>> boot = bootstrap_interval(treated, level=0.95, n_resamples=1000, seed=42)
>>> boot.lower < interval.estimate < boot.upper
True
>>> report = compare_groups(treated, control, seed=42)
>>> report.result.test in ("Welch t-test", "Student t-test", "Mann-Whitney U")
True
>>> report.result.significant in (True, False)
True
>>> draws = sample_distribution("binomial", n=50, p=0.3, seed=42)
>>> bool(draws.values.min() >= 0 and draws.values.max() <= 50)
True
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through every sampler (the toolkit's pin).
RANDOM_STATE = 42

#: Distribution kinds :func:`sample_distribution` draws from.
DISTRIBUTIONS: tuple[str, ...] = (
    "normal",
    "uniform",
    "binomial",
    "poisson",
    "exponential",
)

#: Default significance level for every test (the course's 5%).
DEFAULT_ALPHA = 0.05


def _as_values(values: Any, *, caller: str, minimum: int = 2) -> np.ndarray:
    """Flatten, validate and return a finite float array."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise DataValidationError(
                f"{caller}: a DataFrame needs exactly one column, got {values.shape[1]}"
            )
        values = values.iloc[:, 0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise DataValidationError(f"{caller}: expected 1-D values, got shape {array.shape}")
    finite = array[np.isfinite(array)]
    if finite.size < minimum:
        raise DataValidationError(
            f"{caller}: needs at least {minimum} finite value(s), got {finite.size}"
        )
    return finite


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptiveStats:
    """The descriptive lecture's full blackboard for one sample."""

    n: int
    n_missing: int
    mean: float
    std: float
    sem: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    iqr: float
    skew: float
    kurtosis: float

    def five_number(self) -> str:
        return (
            f"min {self.minimum:.4g} · Q1 {self.q1:.4g} · median {self.median:.4g} · "
            f"Q3 {self.q3:.4g} · max {self.maximum:.4g}"
        )

    def line(self) -> str:
        return (
            f"n {self.n} ({self.n_missing} missing) · mean {self.mean:.4g} · "
            f"std {self.std:.4g} · SEM {self.sem:.4g} · skew {self.skew:.3g} · "
            f"kurtosis {self.kurtosis:.3g}"
        )


def describe_series(values: Any) -> DescriptiveStats:
    """One sample's descriptive statistics (scipy's moment machinery).

    Raises
    ------
    DataValidationError
        On fewer than two finite values.
    """
    finite = _as_values(values, caller="describe_series")
    n = int(finite.size)
    try:  # the raw size is informational only -- never worth failing on
        raw_size = int(np.asarray(values).size)
    except Exception:  # noqa: BLE001 - exotic containers fall back to n
        raw_size = n
    result = scipy_stats.describe(finite)
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return DescriptiveStats(
        n=n,
        n_missing=max(0, raw_size - n),
        mean=float(result.mean),
        std=float(math.sqrt(result.variance)) if n > 1 else 0.0,
        sem=float(result.variance / n) ** 0.5 if n > 1 else 0.0,
        minimum=float(finite.min()),
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        maximum=float(finite.max()),
        iqr=float(q3 - q1),
        skew=float(result.skewness) if n > 2 else 0.0,
        kurtosis=float(result.kurtosis) if n > 2 else 0.0,
    )


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CiResult:
    """One confidence interval: the estimate and its bounds."""

    estimate: float
    level: float
    lower: float
    upper: float
    method: str
    n_resamples: int | None = None

    def line(self) -> str:
        resamples = f" · {self.n_resamples} resamples" if self.n_resamples else ""
        return (
            f"{self.estimate:.4g}  [{self.lower:.4g}, {self.upper:.4g}] "
            f"({self.method}, {self.level:.0%}{resamples})"
        )


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise DataValidationError(f"level must be in (0, 1), got {level}")
    return float(level)


def mean_confidence_interval(values: Any, *, level: float = 0.95) -> CiResult:
    """The t-based CI for the mean: ``mean ± t·s/√n``.

    Raises
    ------
    DataValidationError
        On fewer than two values or a bad level.
    """
    level = _check_level(level)
    finite = _as_values(values, caller="mean_confidence_interval")
    n = finite.size
    if n < 2:
        raise DataValidationError("the t interval needs at least two values")
    mean = float(finite.mean())
    sem = float(finite.std(ddof=1) / math.sqrt(n))
    critical = float(scipy_stats.t.ppf(0.5 + level / 2.0, df=n - 1))
    return CiResult(
        estimate=mean,
        level=level,
        lower=mean - critical * sem,
        upper=mean + critical * sem,
        method=f"t interval (df={n - 1})",
    )


def bootstrap_interval(
    values: Any,
    *,
    level: float = 0.95,
    n_resamples: int = 2000,
    statistic: Callable[[np.ndarray], float] | None = None,
    seed: int = RANDOM_STATE,
) -> CiResult:
    """The percentile bootstrap CI for any statistic (default: the mean).

    Seeded resampling with replacement, ``n_resamples`` draws, the
    percentile method -- the course's practical interval for the
    anything-non-normal cases the t interval mishandles.

    Raises
    ------
    DataValidationError
        On too few values, a bad level or a non-positive resample count.
    """
    level = _check_level(level)
    if not 1 <= n_resamples <= 1_000_000:
        raise DataValidationError(f"n_resamples must be in [1, 1000000], got {n_resamples}")
    finite = _as_values(values, caller="bootstrap_interval")
    stat = statistic if statistic is not None else (lambda sample: float(np.mean(sample)))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [stat(finite[rng.integers(0, finite.size, finite.size)]) for _ in range(n_resamples)]
    )
    if not np.isfinite(draws).all():
        raise DataValidationError("the statistic produced NaN/inf on a resample")
    lower, upper = np.percentile(draws, [100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0])
    return CiResult(
        estimate=stat(finite),
        level=level,
        lower=float(lower),
        upper=float(upper),
        method="percentile bootstrap",
        n_resamples=int(n_resamples),
    )


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceResult:
    """One test's verdict: statistic, p-value, decision, effect size."""

    test: str
    statistic: float
    p_value: float
    df: float | None
    alpha: float
    significant: bool
    effect_name: str
    effect_value: float | None

    def line(self) -> str:
        degrees = f" · df {self.df:g}" if self.df is not None else ""
        effect = (
            f" · {self.effect_name} {self.effect_value:.3g}" if self.effect_value is not None else ""
        )
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.test} · statistic {self.statistic:.4g} · p {self.p_value:.4g}{degrees} · "
            f"{verdict} at α {self.alpha}{effect}"
        )


def _check_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise DataValidationError(f"alpha must be in (0, 1), got {alpha}")
    return float(alpha)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with the pooled standard deviation."""
    n_a, n_b = a.size, b.size
    pooled = math.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def one_sample_ttest(
    values: Any,
    *,
    mu0: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
) -> SignificanceResult:
    """Is this sample's mean different from ``mu0``?

    Effect size: Cohen's d against the hypothesised mean
    (``(mean − mu0)/s``).

    Raises
    ------
    DataValidationError
        On fewer than two values or a bad alpha (scipy's own n<2 error
        is pre-empted with the toolkit's message).
    """
    alpha = _check_alpha(alpha)
    finite = _as_values(values, caller="one_sample_ttest")
    if not math.isfinite(mu0):
        raise DataValidationError(f"mu0 must be finite, got {mu0!r}")
    result = scipy_stats.ttest_1samp(finite, mu0)
    std = float(finite.std(ddof=1))
    effect = float((finite.mean() - mu0) / std) if std > 0 else 0.0
    return SignificanceResult(
        test=f"one-sample t-test (μ₀ = {mu0:g})",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=float(finite.size - 1),
        alpha=alpha,
        significant=bool(result.pvalue < alpha),
        effect_name="Cohen's d",
        effect_value=effect,
    )


def paired_ttest(
    before: Any,
    after: Any,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> SignificanceResult:
    """Did the paired differences move away from zero?

    Effect size: Cohen's d on the differences.

    Raises
    ------
    DataValidationError
        On length mismatches or fewer than two pairs.
    """
    alpha = _check_alpha(alpha)
    a = _as_values(before, caller="paired_ttest (before)")
    b = _as_values(after, caller="paired_ttest (after)")
    if a.size != b.size:
        raise DataValidationError(f"paired samples disagree ({a.size} vs {b.size})")
    result = scipy_stats.ttest_rel(a, b)
    differences = a - b
    std = float(differences.std(ddof=1))
    effect = float(differences.mean() / std) if std > 0 else 0.0
    return SignificanceResult(
        test="paired t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=float(a.size - 1),
        alpha=alpha,
        significant=bool(result.pvalue < alpha),
        effect_name="Cohen's d (differences)",
        effect_value=effect,
    )


def two_sample_ttest(
    a: Any,
    b: Any,
    *,
    equal_var: bool = False,
    alpha: float = DEFAULT_ALPHA,
) -> SignificanceResult:
    """Two independent samples: Welch's t (default) or Student's t.

    ``equal_var=False`` is Welch's unequal-variance test -- the modern
    default (scipy's own) and the safer choice when group spreads may
    differ.  Effect size: Cohen's d.

    Raises
    ------
    DataValidationError
        On too-small samples.
    """
    alpha = _check_alpha(alpha)
    values_a = _as_values(a, caller="two_sample_ttest (a)")
    values_b = _as_values(b, caller="two_sample_ttest (b)")
    result = scipy_stats.ttest_ind(values_a, values_b, equal_var=equal_var)
    return SignificanceResult(
        test="Student t-test" if equal_var else "Welch t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=float(result.df) if hasattr(result, "df") else None,
        alpha=alpha,
        significant=bool(result.pvalue < alpha),
        effect_name="Cohen's d",
        effect_value=_cohen_d(values_a, values_b),
    )


def mann_whitney_test(
    a: Any,
    b: Any,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> SignificanceResult:
    """The rank-based test for the non-normal cases.

    Effect size: the rank-biserial correlation
    (``2U/(n₁n₂) − 1``; 0 = complete overlap, ±1 = total separation).

    Raises
    ------
    DataValidationError
        On too-small samples.
    """
    alpha = _check_alpha(alpha)
    values_a = _as_values(a, caller="mann_whitney_test (a)")
    values_b = _as_values(b, caller="mann_whitney_test (b)")
    result = scipy_stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    rank_biserial = float(2.0 * result.statistic / (values_a.size * values_b.size) - 1.0)
    return SignificanceResult(
        test="Mann-Whitney U",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=None,
        alpha=alpha,
        significant=bool(result.pvalue < alpha),
        effect_name="rank-biserial r",
        effect_value=rank_biserial,
    )


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalityResult:
    """Shapiro-Wilk and D'Agostino-Pearson on one sample."""

    n: int
    shapiro_statistic: float
    shapiro_p: float
    dagostino_statistic: float
    dagostino_p: float
    alpha: float

    @property
    def is_normal(self) -> bool:
        """Both tests agree the sample is compatible with normality."""
        return bool(self.shapiro_p > self.alpha and self.dagostino_p > self.alpha)

    def line(self) -> str:
        verdict = "compatible with normality" if self.is_normal else "NOT normal"
        return (
            f"Shapiro-Wilk W {self.shapiro_statistic:.4g} (p {self.shapiro_p:.4g}) · "
            f"D'Agostino K² {self.dagostino_statistic:.4g} (p {self.dagostino_p:.4g}) · "
            f"{verdict} at α {self.alpha}"
        )


def normality_report(values: Any, *, alpha: float = DEFAULT_ALPHA) -> NormalityResult:
    """Both normality tests (the course's gate before choosing a test).

    Shapiro-Wilk needs 3 ≤ n ≤ 5000 (scipy's own window); the
    D'Agostino-Pearson K² runs on any n ≥ 8.  Samples outside both
    windows raise -- the honest answer to "is it normal?" on n=2 is
    "cannot say", not a fabricated verdict.

    Raises
    ------
    DataValidationError
        On n < 8 or a bad alpha.
    """
    alpha = _check_alpha(alpha)
    finite = _as_values(values, caller="normality_report", minimum=8)
    shapiro = scipy_stats.shapiro(finite)
    dagostino = scipy_stats.normaltest(finite)
    return NormalityResult(
        n=int(finite.size),
        shapiro_statistic=float(shapiro.statistic),
        shapiro_p=float(shapiro.pvalue),
        dagostino_statistic=float(dagostino.statistic),
        dagostino_p=float(dagostino.pvalue),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionSample:
    """Seeded draws plus the population moments to compare against."""

    kind: str
    n: int
    values: np.ndarray
    theoretical_mean: float
    theoretical_std: float

    def line(self) -> str:
        return (
            f"{self.kind} · n {self.n} · sample mean {float(self.values.mean()):.4g} "
            f"(theory {self.theoretical_mean:.4g}) · sample std {float(self.values.std()):.4g} "
            f"(theory {self.theoretical_std:.4g})"
        )


def sample_distribution(
    kind: str,
    *,
    n: int = 200,
    loc: float = 0.0,
    scale: float = 1.0,
    p: float = 0.5,
    lam: float = 1.0,
    trials: int = 10,
    seed: int = RANDOM_STATE,
) -> DistributionSample:
    """Draw ``n`` seeded samples from one of the course's distributions.

    Parameters
    ----------
    kind:
        One of :data:`DISTRIBUTIONS`.
    n:
        Sample size (>= 1).
    loc / scale:
        Normal's μ/σ and uniform's [loc, loc+scale]; exponential's
        scale (1/λ).
    p:
        Binomial success probability.
    trials:
        Binomial number of trials.
    lam:
        Poisson's λ (the rate).

    Raises
    ------
    DataValidationError
        On an unknown kind or out-of-range parameters.
    """
    if kind not in DISTRIBUTIONS:
        raise DataValidationError(f"unknown distribution {kind!r} (one of {DISTRIBUTIONS})")
    if not 1 <= n <= 1_000_000:
        raise DataValidationError(f"n must be in [1, 1000000], got {n}")
    if scale <= 0:
        raise DataValidationError(f"scale must be positive, got {scale}")
    if not 0.0 < p < 1.0:
        raise DataValidationError(f"p must be in (0, 1), got {p}")
    if lam <= 0:
        raise DataValidationError(f"lambda must be positive, got {lam}")
    if trials < 1:
        raise DataValidationError(f"trials must be >= 1, got {trials}")
    rng = np.random.default_rng(seed)
    if kind == "normal":
        values = rng.normal(loc, scale, n)
        mean, std = loc, scale
    elif kind == "uniform":
        values = rng.uniform(loc, loc + scale, n)
        mean, std = loc + scale / 2.0, scale / math.sqrt(12.0)
    elif kind == "binomial":
        values = rng.binomial(trials, p, n).astype(float)
        mean, std = trials * p, math.sqrt(trials * p * (1.0 - p))
    elif kind == "poisson":
        values = rng.poisson(lam, n).astype(float)
        mean, std = lam, math.sqrt(lam)
    else:  # exponential
        values = rng.exponential(scale, n)
        mean, std = scale, scale
    return DistributionSample(
        kind=kind,
        n=int(n),
        values=values,
        theoretical_mean=float(mean),
        theoretical_std=float(std),
    )


# ---------------------------------------------------------------------------
# The full course flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonReport:
    """``compare_groups``: the whole statistics lecture, one record.

    ``branch`` is the decision path taken ("both normal → Welch" /
    "non-normal → Mann-Whitney"), so the report never hides WHY a
    test was chosen.
    """

    group_a: DescriptiveStats
    group_b: DescriptiveStats
    normality_a: NormalityResult
    normality_b: NormalityResult
    interval_a: CiResult
    interval_b: CiResult
    result: SignificanceResult
    branch: str

    def headline(self) -> str:
        direction = (
            "a > b"
            if self.group_a.mean > self.group_b.mean
            else "a < b" if self.group_a.mean < self.group_b.mean else "a = b"
        )
        verdict = (
            f"REJECT H₀ at α {self.result.alpha} -- the groups differ ({direction})"
            if self.result.significant
            else f"retain H₀ at α {self.result.alpha} -- no detectable difference"
        )
        return f"{self.result.test} · {self.branch} · {verdict}"


def compare_groups(
    a: Any,
    b: Any,
    *,
    alpha: float = DEFAULT_ALPHA,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = RANDOM_STATE,
) -> ComparisonReport:
    """The course's decision flow, end to end, every branch reported.

    Descriptives on both groups → normality on both (both tests must
    agree) → the test: **Welch's t** when both samples look normal
    (the unequal-variance default -- the safer choice when group
    spreads may differ), **Mann-Whitney U** when either does not →
    t-based or bootstrap intervals per group (bootstrap when the
    normality verdict fails) → the effect size rides on the
    :class:`SignificanceResult`.

    Raises
    ------
    DataValidationError
        On too-small groups (normality needs n ≥ 8) or bad parameters.
    """
    alpha = _check_alpha(alpha)
    level = _check_level(level)
    values_a = _as_values(a, caller="compare_groups (a)", minimum=8)
    values_b = _as_values(b, caller="compare_groups (b)", minimum=8)
    summary_a = describe_series(values_a)
    summary_b = describe_series(values_b)
    normality_a = normality_report(values_a, alpha=alpha)
    normality_b = normality_report(values_b, alpha=alpha)
    both_normal = normality_a.is_normal and normality_b.is_normal
    if both_normal:
        result = two_sample_ttest(values_a, values_b, equal_var=False, alpha=alpha)
        branch = "both normal → Welch t (unequal variances)"
        interval_a = mean_confidence_interval(values_a, level=level)
        interval_b = mean_confidence_interval(values_b, level=level)
    else:
        result = mann_whitney_test(values_a, values_b, alpha=alpha)
        why = []
        if not normality_a.is_normal:
            why.append("a")
        if not normality_b.is_normal:
            why.append("b")
        branch = f"non-normal ({'+'.join(why)}) → Mann-Whitney U"
        interval_a = bootstrap_interval(
            values_a, level=level, n_resamples=n_resamples, seed=seed
        )
        interval_b = bootstrap_interval(
            values_b, level=level, n_resamples=n_resamples, seed=seed + 1
        )
    return ComparisonReport(
        group_a=summary_a,
        group_b=summary_b,
        normality_a=normality_a,
        normality_b=normality_b,
        interval_a=interval_a,
        interval_b=interval_b,
        result=result,
        branch=branch,
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DISTRIBUTIONS",
    "RANDOM_STATE",
    "CiResult",
    "ComparisonReport",
    "DescriptiveStats",
    "DistributionSample",
    "NormalityResult",
    "SignificanceResult",
    "bootstrap_interval",
    "compare_groups",
    "describe_series",
    "mann_whitney_test",
    "mean_confidence_interval",
    "normality_report",
    "one_sample_ttest",
    "paired_ttest",
    "sample_distribution",
    "two_sample_ttest",
]

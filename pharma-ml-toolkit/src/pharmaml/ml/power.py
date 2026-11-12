"""How many folds would it take to *resolve* a gap?

:func:`pharmaml.ml.comparison.model_comparison_test` answers "is the
leader's edge real?" and, when the answer is no, returns the verdict
``on par``.  But ``on par`` is *absence of evidence*, not evidence of
absence: with the usual 5 folds a real-but-small gap is simply
underpowered to detect.  This module supplies the missing half of that
sentence — the fold-count power analysis:

* :func:`paired_power` — the achieved power of the paired t-test at a
  given standardized effect (Cohen's d on the fold differences) and
  fold count, from the exact noncentral-t distribution (the normal
  limit takes over once the noncentrality overflows, where it is exact
  to well below a part in 10⁸).
* :func:`minimum_detectable_effect` — the smallest ``|d|`` the panel
  can resolve at a target power (80% by convention) and the current
  fold count.  If the leader's true edge is smaller than this, an
  ``on par`` verdict is the expected outcome — the study was too small,
  not the models equal.
* :func:`folds_for_power` — the fold count that *would* reach the
  target power for a given effect, so "how many folds do I need?" has a
  number rather than a shrug.
* :func:`power_report` — the four numbers above rolled into one
  :class:`PowerReport`, the same way :func:`~pharmaml.ml.stats.compare_groups`
  rolls up the course's hypothesis-testing flow.

The honest scope, stated once: the power figures assume the paired
t-test (Dietterich's k-fold comparison).  The signed-rank test's power
is a different curve — with the usual 5 folds it cannot even reach
α = 0.05 — so :func:`~pharmaml.ml.comparison.model_comparison_test`
only attaches a power reading to a ``paired_t`` panel.

Examples
--------
>>> from pharmaml.ml.power import folds_for_power, minimum_detectable_effect, paired_power
>>> # a large effect is resolvable with few folds
>>> round(paired_power(1.0, 10), 3)
0.803
>>> # a medium effect needs ~34 folds for 80% power (the classic table)
>>> folds_for_power(0.5)
34
>>> # five folds can only see a large effect at 80% power
>>> round(minimum_detectable_effect(5), 3)
1.682
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as scipy_stats
from scipy.optimize import brentq

from pharmaml.exceptions import DataValidationError

#: Target power for the minimum-detectable-effect reading (the convention).
DEFAULT_TARGET_POWER = 0.8

#: Fold counts the toolkit offers (:func:`folds_for_power` never returns
#: a panel larger than this — "you need thousands of folds" is an honest
#: answer, and this is where it starts).
MAX_FOLDS = 10_000

#: Noncentrality above which the noncentral-t CDF loses precision and the
#: normal limit (exact here to < 1e-10) takes over.
_NCT_OVERFLOW = 37.0


def _check_probability(value: float, name: str) -> float:
    if not 0.0 < value < 1.0:
        raise DataValidationError(f"{name} must be in (0, 1), got {value}")
    return float(value)


def _check_folds(n_folds: int) -> int:
    if n_folds < 2:
        raise DataValidationError(f"a paired test needs at least 2 folds, got {n_folds}")
    return int(n_folds)


def _paired_t_power(effect: float, n_folds: int, alpha: float) -> float:
    """Power of the two-sided paired t-test at standardized ``effect``.

    ``effect`` is Cohen's d on the fold differences; the noncentrality
    parameter is ``d·√n`` and the critical value the usual
    ``t_{n-1, 1-α/2}``.
    """
    df = n_folds - 1
    critical = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df))
    noncentrality = abs(effect) * math.sqrt(n_folds)
    if noncentrality > _NCT_OVERFLOW:
        return float(
            scipy_stats.norm.sf(critical - noncentrality)
            + scipy_stats.norm.cdf(-critical - noncentrality)
        )
    power = scipy_stats.nct.sf(critical, df, noncentrality) + scipy_stats.nct.cdf(
        -critical, df, noncentrality
    )
    if not math.isfinite(float(power)):  # pragma: no cover - guard only
        return float(
            scipy_stats.norm.sf(critical - noncentrality)
            + scipy_stats.norm.cdf(-critical - noncentrality)
        )
    return float(power)


def paired_power(
    effect: float,
    n_folds: int,
    *,
    alpha: float = 0.05,
) -> float:
    """The paired t-test's power to detect ``effect`` at ``n_folds`` folds.

    Parameters
    ----------
    effect:
        The standardized effect size (Cohen's d on the fold
        differences).  The sign is irrelevant — power depends on
        ``|effect|``.
    n_folds:
        The number of folds (paired observations) the test runs on.
    alpha:
        The two-sided significance level.

    Returns
    -------
    float
        The power in ``[0, 1]``.

    Raises
    ------
    DataValidationError
        On fewer than two folds or an out-of-range alpha.

    Examples
    --------
    >>> from pharmaml.ml.power import paired_power
    >>> round(paired_power(0.5, 5), 3)   # five folds barely see a medium effect
    0.141
    >>> round(paired_power(0.0, 10), 3)  # no effect -> the false-positive rate
    0.05
    """
    _check_folds(n_folds)
    _check_probability(alpha, "alpha")
    return _paired_t_power(float(effect), int(n_folds), float(alpha))


def minimum_detectable_effect(
    n_folds: int,
    *,
    alpha: float = 0.05,
    target_power: float = DEFAULT_TARGET_POWER,
) -> float:
    """The smallest ``|d|`` resolvable at ``target_power`` with ``n_folds``.

    Solves :func:`paired_power` for the effect.  This is the number that
    makes an ``on par`` verdict honest: if the leader's true edge is
    below it, the panel could not have seen the edge anyway.

    Raises
    ------
    DataValidationError
        On fewer than two folds or out-of-range ``alpha`` / ``target_power``.

    Examples
    --------
    >>> from pharmaml.ml.power import minimum_detectable_effect
    >>> round(minimum_detectable_effect(5), 3)
    1.682
    >>> round(minimum_detectable_effect(20), 3)
    0.66
    """
    n_folds = _check_folds(n_folds)
    _check_probability(alpha, "alpha")
    _check_probability(target_power, "target_power")
    upper = 1.0
    while _paired_t_power(upper, n_folds, alpha) < target_power and upper < 1e6:
        upper *= 2.0
    return float(
        brentq(
            lambda effect: _paired_t_power(effect, n_folds, alpha) - target_power,
            1e-9,
            upper,
            xtol=1e-10,
        )
    )


def folds_for_power(
    effect: float,
    *,
    alpha: float = 0.05,
    target_power: float = DEFAULT_TARGET_POWER,
) -> int | None:
    """The fold count that reaches ``target_power`` for ``effect``.

    Returns the smallest integer fold count whose power meets the
    target, or ``None`` when even :data:`MAX_FOLDS` folds fall short
    (a vanishingly small effect — the honest answer is "you cannot
    power your way there with folds alone").

    Raises
    ------
    DataValidationError
        On a non-positive ``effect`` or out-of-range ``alpha`` /
        ``target_power``.

    Examples
    --------
    >>> from pharmaml.ml.power import folds_for_power
    >>> folds_for_power(0.5)      # the classic "~34 for 80% power"
    34
    >>> folds_for_power(1.0)
    10
    >>> folds_for_power(0.0) is None
    True
    """
    if effect <= 0.0:
        return None
    _check_probability(alpha, "alpha")
    _check_probability(target_power, "target_power")
    upper = 4
    while _paired_t_power(effect, upper, alpha) < target_power and upper < MAX_FOLDS:
        upper *= 2
    if _paired_t_power(effect, upper, alpha) < target_power:
        return None
    lower = 2
    if _paired_t_power(effect, lower, alpha) >= target_power:
        return lower
    # The power rises monotonically in n, so bisect on the integer fold count.
    lo, hi = lower, upper
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _paired_t_power(effect, mid, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class PowerReport:
    """One panel's fold-count power picture.

    Attributes
    ----------
    n_folds:
        The fold count the test runs on.
    alpha:
        The significance level.
    target_power:
        The power the minimum-detectable-effect is read at (0.8 by
        convention).
    mde:
        The smallest standardized effect resolvable at ``target_power``.
    observed_effect:
        The standardized effect actually measured on the panel, when one
        was supplied (``None`` otherwise).
    observed_power:
        The power to detect ``observed_effect`` at this fold count, or
        ``None`` alongside a ``None`` ``observed_effect``.
    sufficient:
        Whether the observed effect clears the MDE — i.e. whether the
        panel had the power to resolve what it saw.  ``None`` when no
        observed effect was supplied.
    """

    n_folds: int
    alpha: float
    target_power: float
    mde: float
    observed_effect: float | None = None
    observed_power: float | None = None
    sufficient: bool | None = None

    def line(self) -> str:
        """A one-line summary for the CLI and reports."""
        text = (
            f"{self.n_folds} folds · MDE |d| {self.mde:.3g} at "
            f"{self.target_power:.0%} power (α {self.alpha:g})"
        )
        if self.observed_effect is not None and self.observed_power is not None:
            verdict = "adequate" if self.sufficient else "underpowered"
            text += (
                f" · observed |d| {abs(self.observed_effect):.3g} "
                f"(power {self.observed_power:.0%}, {verdict})"
            )
        return text


def power_report(
    n_folds: int,
    *,
    alpha: float = 0.05,
    target_power: float = DEFAULT_TARGET_POWER,
    observed_effect: float | None = None,
) -> PowerReport:
    """Assemble the fold-count power picture for one panel.

    Parameters
    ----------
    n_folds:
        The fold count.
    alpha:
        The significance level.
    target_power:
        The power the minimum-detectable-effect is read at.
    observed_effect:
        The standardized effect measured on the panel, when known.

    Raises
    ------
    DataValidationError
        On fewer than two folds or out-of-range ``alpha`` / ``target_power``.

    Examples
    --------
    >>> from pharmaml.ml.power import power_report
    >>> report = power_report(5, observed_effect=1.7)
    >>> round(report.mde, 3), report.sufficient
    (1.682, True)
    >>> power_report(5, observed_effect=0.3).sufficient
    False
    """
    mde = minimum_detectable_effect(n_folds, alpha=alpha, target_power=target_power)
    if observed_effect is None:
        return PowerReport(
            n_folds=int(n_folds),
            alpha=float(alpha),
            target_power=float(target_power),
            mde=mde,
        )
    effect = abs(float(observed_effect))
    achieved = paired_power(effect, n_folds, alpha=alpha)
    return PowerReport(
        n_folds=int(n_folds),
        alpha=float(alpha),
        target_power=float(target_power),
        mde=mde,
        observed_effect=float(observed_effect),
        observed_power=achieved,
        sufficient=achieved >= target_power,
    )


__all__ = [
    "DEFAULT_TARGET_POWER",
    "MAX_FOLDS",
    "PowerReport",
    "folds_for_power",
    "minimum_detectable_effect",
    "paired_power",
    "power_report",
]

"""Differential privacy in plain NumPy: the mechanisms, the budget
arithmetic, and the DP-SGD noise scale.

The best-of-ml-python *Privacy Machine Learning* category (7 projects:
Opacus, diffprivlib, TF-Privacy, PaddleFL, PySyft, OpenDP, snsynth)
was reviewed entry by entry for this module.  What every one of them
ships is the same three pieces of mathematics, and none of the three
needs an engine:

* **The mechanisms** -- the Laplace mechanism (exact (ε, 0)-DP for
  integer-valued sensitivity, the textbook workhorse) and the Gaussian
  mechanism (approximate (ε, δ)-DP, the one every gradient-noising
  trainer uses).  Both are one line of NumPy once the sensitivity is
  known; what the libraries add is plumbing, not math.
* **The budget arithmetic** -- sequential composition (queries on the
  *same* rows: ε's add) versus parallel composition (queries on
  *disjoint* rows: the max, not the sum -- the property that makes a
  DP histogram cost one ε, not one ε per bin).
* **The accountant** -- how (ε, δ, steps) becomes a noise scale σ.
  This module implements the Rényi-DP accountant (Mironov 2017;
  Balle & Wang 2018 for the conversion): each Gaussian step contributes
  ``α/(2σ²)`` Rényi divergence, T steps compose linearly, and the
  ``(ε, δ)`` guarantee comes from minimising
  ``ε(α) = T·α/(2σ²) + ln(1/δ)/(α−1)`` over α > 1.

The honest scope, stated once: Opacus-class trainers hook the
mechanism into backprop on a GPU graph the toolkit deliberately does
not vendor; diffprivlib wraps scikit-learn estimators whose fit loops
would each need the noise injected.  What lands here is the part a
desktop pharma tool actually needs -- privatising a reported statistic
(count, sum, mean, histogram of an assay column) and reading off the
noise price of a training plan.  The accountant is the *plain* RDP
one -- deliberately without subsampling amplification, so its σ is
the conservative ceiling: where Opacus' amplified accountant would
ask σ ≈ 1 for (ε=4, δ=10⁻⁵, 1500 steps), this one asks ≈ 50 -- never
under-noising, always over-pricing the plan (the honest direction
for a tool that cannot run the amplified bookkeeping).

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.privacy import (
...     dp_statistics, dp_sgd_noise_scale, gaussian_mechanism,
...     laplace_mechanism, rdp_epsilon,
... )
>>> rng = np.random.default_rng(42)
>>> values = rng.normal(loc=50.0, scale=5.0, size=400)
>>> report = dp_statistics(values, epsilon=1.0, seed=42)
>>> report.n_values
400
>>> 48.0 < report.dp_mean < 52.0        # the noise price of ε=1
True
>>> 0.0 < report.dp_count_error_pct < 2.0
True
>>> noisy = laplace_mechanism(1234.0, sensitivity=1.0, epsilon=0.5, seed=42)
>>> round(float(noisy.noisy), 1) > 1000.0
True
>>> scale = dp_sgd_noise_scale(epsilon=4.0, delta=1e-5, steps=1500)
>>> 40.0 < scale.sigma < 60.0        # the plain-RDP ceiling
True
>>> rdp = rdp_epsilon(sigma=1.5, steps=1500, delta=1e-5)
>>> round(rdp.epsilon, 1) > 4.0
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through every mechanism (the pin).
RANDOM_STATE = 42

#: The classic analytic-Gaussian δ below which the bound is vacuous.
_GAUSSIAN_DELTA_FLOOR = 1e-9


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_epsilon(epsilon: float) -> float:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise DataValidationError(f"epsilon must be finite and > 0, got {epsilon!r}")
    return epsilon


def _check_delta(delta: float) -> float:
    if not (0.0 < delta < 1.0):
        raise DataValidationError(
            f"delta must be in (0, 1), got {delta!r} "
            "(pure DP is delta=0: use the Laplace mechanism)"
        )
    return delta


def _check_sensitivity(sensitivity: float) -> float:
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise DataValidationError(f"sensitivity must be finite and > 0, got {sensitivity!r}")
    return sensitivity


def _as_array(values: np.ndarray | list[float] | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise DataValidationError("values must not be empty")
    if not np.all(np.isfinite(array)):
        raise DataValidationError("values must all be finite (no NaN/inf)")
    return array


# ---------------------------------------------------------------------------
# The two mechanisms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismResult:
    """One privatised query: the truth, the release, the price.

    ``true_values``/``noisy_values`` keep the input's shape (scalar in,
    scalar out); ``scale`` is the noise's own spread (Laplace b or
    Gaussian σ); ``abs_error``/``rel_error`` are per-element |noisy−true|
    and its ratio to |true| where that is non-zero.
    """

    true_values: np.ndarray
    noisy_values: np.ndarray
    mechanism: str
    epsilon: float
    delta: float
    scale: float
    abs_error: np.ndarray
    rel_error: np.ndarray

    @property
    def noisy(self) -> float:
        """The scalar release (first element), for one-number reports."""
        return float(self.noisy_values.reshape(-1)[0])


def laplace_mechanism(
    values: np.ndarray | list[float] | float,
    sensitivity: float,
    epsilon: float,
    seed: int = RANDOM_STATE,
) -> MechanismResult:
    """The Laplace mechanism: exact (ε, 0)-differential privacy.

    ``noisy = true + Laplace(0, sensitivity/ε)`` -- the scale is the
    sensitivity divided by the budget, which *is* the whole trade-off:
    halving ε doubles the noise.  Laplace is the right mechanism for
    integer-valued or L1-bounded queries (counts, sums of bounded
    values); for everything else the Gaussian below buys smaller noise
    at the cost of a δ.

    Raises
    ------
    DataValidationError
        On non-positive ε/sensitivity or empty/non-finite values.
    """
    _check_sensitivity(sensitivity)
    _check_epsilon(epsilon)
    array = _as_array(values)
    scale = sensitivity / epsilon
    rng = np.random.default_rng(seed)
    noise = rng.laplace(0.0, scale, size=array.shape)
    noisy = array + noise
    abs_error = np.abs(noisy - array)
    nonzero = np.abs(array) > 1e-12
    rel_error = np.where(nonzero, abs_error / np.where(nonzero, np.abs(array), 1.0), np.nan)
    _LOG.debug("laplace mechanism: eps=%.3g scale=%.4g", epsilon, scale)
    return MechanismResult(
        true_values=array,
        noisy_values=noisy,
        mechanism="laplace",
        epsilon=epsilon,
        delta=0.0,
        scale=float(scale),
        abs_error=abs_error,
        rel_error=rel_error,
    )


def gaussian_mechanism(
    values: np.ndarray | list[float] | float,
    sensitivity: float,
    epsilon: float,
    delta: float,
    seed: int = RANDOM_STATE,
) -> MechanismResult:
    """The analytic Gaussian mechanism: (ε, δ)-DP with the classic bound.

    ``σ = sensitivity · sqrt(2·ln(1.25/δ)) / ε`` -- the closed form from
    Dwork & Roth (Algorithm 1 Theorem), valid for the classic regime
    ``ε ≤ 1`` (this module *rejects* larger ε with that citation rather
    than silently stretching the bound; the RDP accountant below is the
    modern tool for bigger budgets).  The noise itself is Gaussian --
    which is why every gradient-noising trainer (Opacus, TF-Privacy)
    builds on exactly this primitive.

    Raises
    ------
    DataValidationError
        On ε > 1 (bound regime), δ outside (0, 1), non-positive
        sensitivity, or empty/non-finite values.
    """
    if epsilon > 1.0:
        raise DataValidationError(
            f"the classic analytic-Gaussian bound holds for epsilon <= 1, got {epsilon!r}; "
            "use rdp_epsilon()/dp_sgd_noise_scale() for larger budgets"
        )
    _check_delta(delta)
    _check_sensitivity(sensitivity)
    _check_epsilon(epsilon)
    array = _as_array(values)
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    rng = np.random.default_rng(seed)
    noisy = array + rng.normal(0.0, sigma, size=array.shape)
    abs_error = np.abs(noisy - array)
    nonzero = np.abs(array) > 1e-12
    rel_error = np.where(nonzero, abs_error / np.where(nonzero, np.abs(array), 1.0), np.nan)
    _LOG.debug("gaussian mechanism: eps=%.3g delta=%.2g sigma=%.4g", epsilon, delta, sigma)
    return MechanismResult(
        true_values=array,
        noisy_values=noisy,
        mechanism="gaussian",
        epsilon=epsilon,
        delta=delta,
        scale=float(sigma),
        abs_error=abs_error,
        rel_error=rel_error,
    )


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionReport:
    """The ε bill for a sequence of queries.

    ``total_epsilon`` is the sequential-composition sum (queries over
    the *same* rows); ``parallel_note`` records that queries over
    *disjoint* rows would have cost the maximum instead -- the property
    the DP histogram below exploits bin by bin.
    """

    epsilons: tuple[float, ...]
    total_epsilon: float
    n_queries: int
    parallel_note: str = (
        "queries over disjoint rows compose to max(eps), not sum(eps) -- "
        "the parallel-composition property"
    )


def composition_report(epsilons: list[float] | tuple[float, ...]) -> CompositionReport:
    """Sequential composition: the ε's of queries on the same rows add.

    Raises
    ------
    DataValidationError
        On an empty sequence or any non-positive ε.
    """
    if not epsilons:
        raise DataValidationError("epsilons must not be empty")
    checked = tuple(_check_epsilon(float(e)) for e in epsilons)
    return CompositionReport(
        epsilons=checked,
        total_epsilon=float(sum(checked)),
        n_queries=len(checked),
    )


# ---------------------------------------------------------------------------
# Privatised statistics (the assay-column demo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistogramBin:
    """One DP histogram bin: edges, true count, DP count."""

    lower: float
    upper: float
    true_count: int
    dp_count: float


@dataclass(frozen=True)
class DPStatisticsReport:
    """A column's headline statistics, privatised under one ε.

    The budget splits three ways (sequential composition -- every query
    sees the same rows): count gets ε/3, sum gets ε/3, the histogram
    gets ε/3 *once* (parallel composition -- each bin counts disjoint
    rows).  ``bounds`` is the clipping range; when derived from the
    data itself the report says so -- production bounds must come from
    public knowledge, not the column, and that caveat is the difference
    between a demo and a deployment.
    """

    n_values: int
    epsilon: float
    bounds: tuple[float, float]
    bounds_from_data: bool
    true_count: float
    dp_count: float
    true_sum: float
    dp_sum: float
    true_mean: float
    dp_mean: float
    count_error: float
    mean_error: float
    histogram: tuple[HistogramBin, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def dp_count_error_pct(self) -> float:
        """The count release's relative error, in percent."""
        if self.true_count == 0:
            return float("nan")
        return abs(self.dp_count - self.true_count) / abs(self.true_count) * 100.0

    @property
    def dp_mean_error_pct(self) -> float:
        """The mean release's relative error, in percent."""
        if abs(self.true_mean) < 1e-12:
            return float("nan")
        return abs(self.dp_mean - self.true_mean) / abs(self.true_mean) * 100.0


def dp_statistics(
    values: np.ndarray | list[float],
    epsilon: float,
    *,
    bounds: tuple[float, float] | None = None,
    n_bins: int = 0,
    seed: int = RANDOM_STATE,
) -> DPStatisticsReport:
    """Count, sum, mean (+ optional histogram) under one ε budget.

    The recipe, exactly as diffprivlib would run it on an assay column:

    1. clip to ``bounds`` (public bounds keep the sensitivities finite:
       count's is 1, sum's is the range);
    2. split ε across the queries that see the same rows
       (sequential composition);
    3. release Laplace-noised values, with the histogram charged
       *once* thanks to parallel composition across its disjoint bins;
    4. divide the DP sum by the DP count for the mean -- the standard
       post-processing, free of further privacy cost.

    When ``bounds`` is ``None`` the data's own min/max are used and the
    report flags it (``bounds_from_data``) -- the demo's honest caveat,
    since real bounds must be public knowledge.

    Raises
    ------
    DataValidationError
        On non-positive ε/bins, inverted bounds or empty values.
    """
    _check_epsilon(epsilon)
    if n_bins < 0:
        raise DataValidationError(f"n_bins must be >= 0, got {n_bins}")
    array = _as_array(values)
    if bounds is None:
        lo, hi = float(array.min()), float(array.max())
        bounds_from_data = True
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
        bounds_from_data = False
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise DataValidationError(f"bounds must be finite with upper > lower, got {(lo, hi)!r}")
    clipped = np.clip(array, lo, hi)
    n_queries = 2 + (1 if n_bins else 0)
    eps_each = epsilon / n_queries

    true_count = float(array.size)
    true_sum = float(clipped.sum())
    true_mean = float(clipped.mean())

    rng = np.random.default_rng(seed)
    dp_count = true_count + rng.laplace(0.0, 1.0 / eps_each)
    dp_sum = true_sum + rng.laplace(0.0, (hi - lo) / eps_each)
    dp_mean = dp_sum / dp_count if abs(dp_count) > 1e-9 else float("nan")

    bins: tuple[HistogramBin, ...] = ()
    if n_bins:
        edges = np.linspace(lo, hi, n_bins + 1)
        true_counts, _ = np.histogram(clipped, bins=edges)
        # parallel composition: one epsilon, not n_bins of them
        noisy_counts = true_counts + rng.laplace(0.0, 1.0 / eps_each, size=n_bins)
        bins = tuple(
            HistogramBin(
                lower=float(edges[i]),
                upper=float(edges[i + 1]),
                true_count=int(true_counts[i]),
                dp_count=float(noisy_counts[i]),
            )
            for i in range(n_bins)
        )

    notes: list[str] = [
        f"budget split {n_queries} ways (count/sum"
        + ("+histogram" if n_bins else "")
        + f"), eps={eps_each:.4g} each (sequential composition)",
        "histogram charged once: bins count disjoint rows (parallel composition)",
    ]
    if bounds_from_data:
        notes.append(
            "bounds were derived from the data itself -- a demo shortcut; "
            "production bounds must be public knowledge"
        )
    _LOG.debug("dp_statistics: n=%d eps=%.3g bins=%d", array.size, epsilon, n_bins)
    return DPStatisticsReport(
        n_values=int(array.size),
        epsilon=float(epsilon),
        bounds=(lo, hi),
        bounds_from_data=bounds_from_data,
        true_count=true_count,
        dp_count=float(dp_count),
        true_sum=true_sum,
        dp_sum=float(dp_sum),
        true_mean=true_mean,
        dp_mean=float(dp_mean),
        count_error=float(dp_count - true_count),
        mean_error=float(dp_mean - true_mean),
        histogram=bins,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The RDP accountant (how (eps, delta, steps) becomes sigma)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RDPReport:
    """The (ε, δ) guarantee of T Gaussian steps at noise multiplier σ.

    ``epsilon`` is the *tightest* guarantee the grid of α's can express:
    ``ε(α) = T·α/(2σ²) + ln(1/δ)/(α−1)`` minimised over α > 1
    (Mironov 2017's Rényi-DP, converted per Balle & Wang 2018).
    """

    epsilon: float
    alpha: float
    sigma: float
    steps: int
    delta: float


def rdp_epsilon(
    sigma: float,
    steps: int,
    delta: float,
    alphas: tuple[float, ...] | None = None,
) -> RDPReport:
    """The RDP accountant: T plain Gaussian steps -> (ε, δ).

    Each step at noise multiplier ``sigma`` (noise std = sigma ×
    sensitivity) contributes ``α/(2σ²)`` Rényi divergence; ``steps``
    of them compose *linearly* (the property that makes RDP the
    accountant of record); the (ε, δ) conversion adds
    ``ln(1/δ)/(α−1)``.  The default α grid spans 1.1–256 logarithmically
    -- the same span Opacus' accountant sweeps.

    Honest conservatism, stated in the report's use: no subsampling
    amplification is applied.  A real DP-SGD run samples a minibatch
    each step, which *tightens* the guarantee (smaller ε, hence less
    noise needed); this function's ε is therefore an upper bound on
    what a production accountant would report for the same σ.  Planning
    against it over-noises, never under-noises.

    Raises
    ------
    DataValidationError
        On σ ≤ 0, steps < 1, δ outside (0, 1).
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise DataValidationError(f"sigma must be finite and > 0, got {sigma!r}")
    if steps < 1:
        raise DataValidationError(f"steps must be >= 1, got {steps}")
    _check_delta(delta)
    if alphas is None:
        alphas = tuple(round(float(a), 4) for a in np.geomspace(1.1, 256.0, 40))
    if not alphas or any(a <= 1.0 for a in alphas):
        raise DataValidationError("alphas must all be > 1")

    log_delta = math.log(1.0 / delta)
    best_epsilon = math.inf
    best_alpha = alphas[0]
    for alpha in alphas:
        candidate = steps * alpha / (2.0 * sigma * sigma) + log_delta / (alpha - 1.0)
        if candidate < best_epsilon:
            best_epsilon = candidate
            best_alpha = alpha
    _LOG.debug(
        "rdp_epsilon: sigma=%.4g steps=%d -> eps=%.4g (alpha=%.2g)",
        sigma,
        steps,
        best_epsilon,
        best_alpha,
    )
    return RDPReport(
        epsilon=float(best_epsilon),
        alpha=float(best_alpha),
        sigma=float(sigma),
        steps=int(steps),
        delta=float(delta),
    )


@dataclass(frozen=True)
class DPNoiseScale:
    """The noise multiplier a training plan needs, by bisection."""

    sigma: float
    epsilon: float
    delta: float
    steps: int
    achieved_epsilon: float
    alpha: float


def dp_sgd_noise_scale(
    epsilon: float,
    delta: float,
    steps: int,
    *,
    sigma_lo: float = 0.01,
    sigma_hi: float = 100.0,
    tolerance: float = 1e-4,
) -> DPNoiseScale:
    """Invert the RDP accountant: the σ that meets (ε, δ) in T steps.

    Bisection over σ: the ε(σ) curve is monotonically decreasing, so
    the bracket converges on the smallest σ whose achieved ε stays
    under the budget.  This is the number a DP-SGD trainer multiplies
    the per-sample gradient norm by.  Plain composition, no
    subsampling amplification: an Opacus-class accountant with a
    realistic sampling rate reports a *smaller* σ for the same budget
    (≈ 1 versus ≈ 50 at ε=4, δ=10⁻⁵, 1500 steps) -- this ceiling
    over-noises, never under-noises.

    Raises
    ------
    DataValidationError
        On non-positive ε, δ outside (0, 1), steps < 1, or an inverted
        bisection bracket.
    """
    _check_epsilon(epsilon)
    _check_delta(delta)
    if steps < 1:
        raise DataValidationError(f"steps must be >= 1, got {steps}")
    if sigma_hi <= sigma_lo or sigma_lo <= 0:
        raise DataValidationError(
            f"bisection bracket must satisfy 0 < sigma_lo < sigma_hi, got {(sigma_lo, sigma_hi)!r}"
        )
    if rdp_epsilon(sigma_lo, steps, delta).epsilon <= epsilon:
        raise DataValidationError(f"sigma_lo={sigma_lo} already meets the budget; lower it")
    if rdp_epsilon(sigma_hi, steps, delta).epsilon > epsilon:
        raise DataValidationError(f"sigma_hi={sigma_hi} cannot meet the budget; raise it")
    lo, hi = sigma_lo, sigma_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        report = rdp_epsilon(mid, steps, delta)
        if report.epsilon > epsilon:
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance:
            break
    final = rdp_epsilon(hi, steps, delta)
    _LOG.debug(
        "dp_sgd_noise_scale: eps=%.3g delta=%.2g steps=%d -> sigma=%.4g",
        epsilon,
        delta,
        steps,
        hi,
    )
    return DPNoiseScale(
        sigma=float(hi),
        epsilon=float(epsilon),
        delta=float(delta),
        steps=int(steps),
        achieved_epsilon=final.epsilon,
        alpha=final.alpha,
    )

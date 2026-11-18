"""Non-compartmental extensions: pkpdutils' NCA beyond the trapezoid.

The toolkit's :mod:`pharmaml.pharma.pk` already ships the classic NCA
(linear/log trapezoid, AUMC, MRT, terminal secant).  This module
absorbs what ``pkpdutils`` (König's PK/PD utilities, 1.x) adds on top,
transcribed natively:

* :func:`auc_linear_log` — the **hybrid linear-up/log-down** rule (the
  PKNCA/WinNonlin ``"linlog"`` default): the logarithmic trapezoid on
  falling positive segments (where it is exact for exponential decay),
  the linear trapezoid on rising or non-positive ones;
* :func:`fit_terminal_phase` — λz by **best adjusted R² over candidate
  windows** (at least ``min_points`` samples, optionally starting after
  Cmax), with the half-life span diagnostic pkpdutils flags at
  ``SPAN_LOW`` when the window covers < 2 terminal half-lives;
* :func:`c0_back_extrapolated` — the IV-bolus C₀ by log-linear
  back-extrapolation through the first two declining samples;
* :func:`observed_lag_time` — the oral absorption lag (last sample
  before the first measurable concentration);
* :func:`steady_state_metrics` — the steady-state interval family:
  AUC_τ, Cavg, Cmax/Cmin/trough, fluctuation, swing and CL_ss;
* :func:`urine_metrics` — urinary recovery NCA (midpoint rate curve,
  amount recovered, renal clearance ``CLr = Ae/AUC``);
* :func:`sparse_auc_bailer` — Bailer's sparse-design AUC with its
  Satterthwaite variance (the mouse-study NCA).

Examples
--------
>>> import numpy as np
>>> from pharmaml.pkpd.nca_ext import (
...     auc_linear_log, fit_terminal_phase, c0_back_extrapolated,
...     observed_lag_time, steady_state_metrics, sparse_auc_bailer,
...     urine_metrics)

Hybrid AUC: linear on the rising first segment (5), log-exact on the
falling second one (``5/ln 2 = 7.2135``):

>>> round(auc_linear_log([0.0, 1.0, 2.0], [0.0, 10.0, 5.0]), 4)
12.2135

Terminal phase on a clean monoexponential (λz = 0.25, t½ = ln2/0.25):

>>> t = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
>>> c = [10.0 * np.exp(-0.25 * x) for x in t]
>>> fit = fit_terminal_phase(t, c)
>>> round(fit.lambda_z, 6), round(fit.half_life, 4)
(0.25, 2.7726)
>>> fit.span_low            # the window spans >= 2 half-lives here
False

C₀ back-extrapolation through two declining points of ``10 e^{-0.3 t}``
recovers the true C₀ = 10:

>>> round(c0_back_extrapolated([0.5, 1.0, 2.0],
...     [10.0 * np.exp(-0.3 * x) for x in [0.5, 1.0, 2.0]]), 6)
10.0

The oral lag: the last non-measurable sample before the first
quantifiable one:

>>> observed_lag_time([0.0, 0.5, 1.0, 2.0], [0.0, 0.0, 0.5, 1.2])
0.5

Urine recovery over three intervals — 70 mg recovered, the midpoint
rate curve integrating to 45, ``CLr = 70/35 = 2`` L/h and 70 % of a
100 mg dose:

>>> from pharmaml.pkpd.nca_ext import urine_metrics
>>> urine = urine_metrics([0, 4, 8], [4, 8, 12], [40.0, 20.0, 10.0],
...                       plasma_auc=35.0, dose=100.0)
>>> (round(urine.amount_recovered, 4), round(urine.aurc, 4),
...  round(urine.renal_clearance or 0.0, 4), round(urine.percent_recovered or 0.0, 4))
(70.0, 45.0, 2.0, 70.0)

Bailer's sparse-design AUC over group means (the trapezoid over the
means with Bailer weights and its Satterthwaite-style variance):

>>> from pharmaml.pkpd.nca_ext import sparse_auc_bailer
>>> sparse = sparse_auc_bailer([0.0, 1.0, 2.0], [0.0, 10.0, 5.0],
...                            [0.0, 2.0, 1.0], n_per_time=10)
>>> (round(sparse.auc, 4), round(sparse.variance, 4),
...  round(sparse.standard_error, 4))
(12.5, 0.425, 0.6519)

One steady-state interval of ``2 e^{-ln2/3 · t}`` (t½ = 3 h, τ = 12 h):

>>> from pharmaml.pkpd.nca_ext import steady_state_metrics
>>> ss = steady_state_metrics(np.linspace(0.0, 12.0, 25),
...     2.0 * np.exp(-np.log(2) / 3.0 * np.linspace(0.0, 12.0, 25)),
...     tau=12.0, dose=100.0, lambda_z=np.log(2) / 3.0)
>>> (round(ss.cl_ss or 0.0, 3), round(ss.accumulation_predicted or 0.0, 4))
(12.309, 1.0667)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma.pk import auc_trapezoid
from pharmaml.utils.validation import validate_concentration_series, validate_positive

__all__ = [
    "TerminalFit",
    "auc_linear_log",
    "c0_back_extrapolated",
    "fit_terminal_phase",
    "observed_lag_time",
    "sparse_auc_bailer",
    "steady_state_metrics",
    "terminal_candidate_table",
    "urine_metrics",
]


def auc_linear_log(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
) -> float:
    """Linear-up / log-down trapezoid AUC (pkpdutils' ``LINEAR_LOG``).

    Segments where the concentration *falls* between two positive
    samples use the logarithmic trapezoid ``Δt·(C₁−C₂)/ln(C₁/C₂)``
    (exact for exponential decline); rising, flat or non-positive
    segments use the plain linear trapezoid.  This is the default AUC
    method of PKNCA and WinNonlin.
    """
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)
    dt = np.diff(t)
    c_start, c_end = c[:-1], c[1:]
    lin = 0.5 * (c_start + c_end)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(c_end) - np.log(c_start)
        quotient = (c_end - c_start) / ratio
    flat = ~np.isfinite(quotient) | (ratio == 0.0)
    quotient = np.where(flat, lin, quotient)
    use_log = (c_start > 0) & (c_end > 0) & (c_end < c_start)
    areas = np.where(use_log, quotient, lin) * dt
    return float(areas.sum())


@dataclass(frozen=True, slots=True)
class TerminalFit:
    """Terminal-phase regression result (pkpdutils ``TerminalFit``)."""

    start_index: int
    end_index: int
    n_points: int
    lambda_z: float
    intercept: float
    r2: float
    r2_adj: float
    half_life: float
    clast_pred: float
    span_half_lives: float
    span_low: bool
    method: str = "best_fit"


def _regress(t: np.ndarray, c: np.ndarray) -> tuple[float, float, float, float]:
    """Log-linear regression of c on t: (slope, intercept, r2, r2_adj)."""
    y = np.log(c)
    n = t.size
    slope, intercept = np.polyfit(t, y, 1)
    predicted = slope * t + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if n > 2 else float("nan")
    return float(slope), float(intercept), float(r2), float(r2_adj)


def terminal_candidate_table(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    *,
    min_points: int = 3,
    exclude_cmax: bool = True,
) -> list[TerminalFit]:
    """Every candidate terminal window with its adjusted R².

    Windows start at each sample index (after Cmax when
    ``exclude_cmax``) and run to the last positive sample; each must
    contain at least ``min_points`` points.  ``best_fit`` selection is
    the maximum ``r2_adj = 1 − (1−r²)(n−1)/(n−2)`` (pkpdutils'
    ``TerminalMethod.BEST_FIT``).
    """
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)
    positive = np.where(c > 0)[0]
    if positive.size < min_points:
        raise DataValidationError(
            f"need at least {min_points} positive concentrations for a "
            f"terminal fit, got {positive.size}"
        )
    last = int(positive[-1])
    first_allowed = int(np.argmax(c)) + 1 if exclude_cmax else 0
    candidates: list[TerminalFit] = []
    for start in range(first_allowed, last - min_points + 2):
        idx = np.arange(start, last + 1)
        if idx.size < min_points:
            continue
        if np.any(c[idx] <= 0):
            continue
        slope, intercept, r2, r2_adj = _regress(t[idx], c[idx])
        lambd = -slope
        if lambd <= 0:
            continue
        half = math.log(2) / lambd
        span = float(t[idx[-1]] - t[idx[0]]) / half
        candidates.append(
            TerminalFit(
                start_index=start,
                end_index=int(idx[-1]),
                n_points=int(idx.size),
                lambda_z=lambd,
                intercept=intercept,
                r2=r2,
                r2_adj=r2_adj,
                half_life=half,
                clast_pred=float(math.exp(intercept - lambd * t[idx[-1]])),
                span_half_lives=span,
                span_low=span < 2.0,
            )
        )
    if not candidates:
        raise DataValidationError("no valid terminal window found")
    return candidates


def fit_terminal_phase(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    *,
    min_points: int = 3,
    exclude_cmax: bool = True,
) -> TerminalFit:
    """Best adjusted-R² terminal window (pkpdutils ``BEST_FIT``).

    The span diagnostic flags windows covering fewer than two
    terminal half-lives (``span_low=True`` — the extrapolated AUC∞
    built on such a λz deserves distrust).
    """
    candidates = terminal_candidate_table(
        times, conc, min_points=min_points, exclude_cmax=exclude_cmax
    )
    best = max(candidates, key=lambda fit: (fit.r2_adj, -fit.start_index))
    return best


def c0_back_extrapolated(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
) -> float:
    """IV-bolus C₀ by log-linear back-extrapolation (pkpdutils C0 method).

    ``C0 = exp(ln C₁ − slope·t₁)`` with the slope through the first two
    declining positive samples — the standard back-extrapolation behind
    WinNonlin's C₀.
    """
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)
    order = np.argsort(t)
    t, c = t[order], c[order]
    idx = np.where(c > 0)[0]
    if idx.size < 2:
        raise DataValidationError("need two positive samples for back-extrapolation")
    i, j = int(idx[0]), int(idx[1])
    if c[j] >= c[i]:
        raise DataValidationError(
            "back-extrapolation needs declining samples "
            f"(got {c[i]} -> {c[j]})"
        )
    slope = (math.log(c[j]) - math.log(c[i])) / (t[j] - t[i])
    return float(math.exp(math.log(c[i]) - slope * t[i]))


def observed_lag_time(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    lloq: float = 0.0,
) -> float:
    """Time of the last sample before the first quantifiable one.

    The observed absorption lag for extravascular administration
    (pkpdutils' ``tlag``): the last non-measurable sample preceding the
    first concentration above ``lloq``.
    """
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)
    measurable = np.where(c > lloq)[0]
    if measurable.size == 0:
        raise DataValidationError("no measurable concentration in the series")
    first = int(measurable[0])
    if first == 0:
        return float("nan")
    return float(t[first - 1])


@dataclass(frozen=True, slots=True)
class SteadyStateMetrics:
    """One steady-state dosing interval's NCA (pkpdutils steady-state)."""

    auc_tau: float
    cavg: float
    cmax_ss: float
    cmin_ss: float
    ctrough: float
    fluctuation: float
    swing: float
    cl_ss: float | None
    accumulation_predicted: float | None

    def as_dict(self) -> dict[str, float | None]:
        """Report-friendly dictionary of the interval metrics."""
        return {
            "AUC_tau": self.auc_tau,
            "Cavg": self.cavg,
            "Cmax_ss": self.cmax_ss,
            "Cmin_ss": self.cmin_ss,
            "Ctrough": self.ctrough,
            "fluctuation": self.fluctuation,
            "swing": self.swing,
            "CL_ss": self.cl_ss,
            "R_accum": self.accumulation_predicted,
        }


def steady_state_metrics(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    tau: float,
    *,
    dose: float | None = None,
    lambda_z: float | None = None,
) -> SteadyStateMetrics:
    """Interval metrics for one steady-state profile (0 … τ).

    ``fluctuation = (Cmax − Cmin)/Cavg`` and ``swing = (Cmax −
    Cmin)/Cmin``; ``CL_ss = dose/AUC_τ`` when the interval dose is
    given; the predicted accumulation ratio ``1/(1 − e^{−λz τ})`` when
    λz is supplied (e.g. from :func:`fit_terminal_phase`).
    """
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)
    tau = validate_positive(tau, "tau")
    if t[-1] < tau - 1e-9:
        raise DataValidationError(
            f"the profile must cover the full interval (last time {t[-1]} < tau {tau})"
        )
    auc = auc_trapezoid(t, c, method="linear")
    cavg = auc / tau
    cmax, cmin = float(c.max()), float(c.min())
    if cavg <= 0:
        raise DataValidationError("AUC over the interval is zero — Cavg undefined")
    if cmin <= 0:
        raise DataValidationError("Cmin must be positive for the swing ratio")
    cl_ss = dose / auc if dose is not None else None
    accum = (
        1.0 / (1.0 - math.exp(-validate_positive(lambda_z, "lambda_z") * tau))
        if lambda_z is not None
        else None
    )
    return SteadyStateMetrics(
        auc_tau=auc,
        cavg=cavg,
        cmax_ss=cmax,
        cmin_ss=cmin,
        ctrough=float(c[int(np.argmin(np.abs(t - tau)))]),
        fluctuation=(cmax - cmin) / cavg,
        swing=(cmax - cmin) / cmin,
        cl_ss=cl_ss,
        accumulation_predicted=accum,
    )


@dataclass(frozen=True, slots=True)
class UrineMetrics:
    """Urinary excretion NCA (pkpdutils ``nca_urine``)."""

    amount_recovered: float
    aurc: float
    renal_clearance: float | None
    percent_recovered: float | None


def urine_metrics(
    interval_starts: np.ndarray | list[float],
    interval_ends: np.ndarray | list[float],
    interval_amounts: np.ndarray | list[float],
    *,
    plasma_auc: float | None = None,
    dose: float | None = None,
) -> UrineMetrics:
    """Renal recovery from collection intervals (midpoint rate curve).

    The excretion rate ``Ȧ_k = A_k / Δt_k`` is plotted against the
    interval midpoint; ``AURC`` is its trapezoid integral,
    ``CLr = ΣA_k / AUC_plasma`` and ``%recovered = ΣA_k / dose``.
    """
    s = np.asarray(interval_starts, dtype=float).reshape(-1)
    e = np.asarray(interval_ends, dtype=float).reshape(-1)
    a = np.asarray(interval_amounts, dtype=float).reshape(-1)
    if not (s.size == e.size == a.size) or s.size == 0:
        raise DataValidationError("starts, ends and amounts must be equal-length")
    if np.any(e <= s) or np.any(a < 0):
        raise DataValidationError("ends must exceed starts; amounts must be >= 0")
    midpoints = 0.5 * (s + e)
    rates = a / (e - s)
    order = np.argsort(midpoints)
    m_sorted, r_sorted = midpoints[order], rates[order]
    aurc = float(
        np.sum(0.5 * (r_sorted[:-1] + r_sorted[1:]) * np.diff(m_sorted))
    )
    total = float(a.sum())
    clr = (
        total / validate_positive(plasma_auc, "plasma_auc")
        if plasma_auc is not None
        else None
    )
    percent = (
        100.0 * total / validate_positive(dose, "dose") if dose is not None else None
    )
    return UrineMetrics(
        amount_recovered=total,
        aurc=aurc,
        renal_clearance=clr,
        percent_recovered=percent,
    )


@dataclass(frozen=True, slots=True)
class SparseAUC:
    """Bailer's sparse-design AUC with Satterthwaite variance."""

    auc: float
    variance: float
    standard_error: float


def sparse_auc_bailer(
    times: np.ndarray | list[float],
    mean_conc: np.ndarray | list[float],
    sd_conc: np.ndarray | list[float],
    n_per_time: np.ndarray | list[float] | int,
) -> SparseAUC:
    """Bailer's method for sparse (serial-sacrifice) designs.

    ``AUC = Σ_j w_j ȳ_j`` with the bin widths ``w_j``; under
    independence ``Var = Σ_j w_j² s_j² / n_j`` (Bailer 1983; the
    Nedelman–Gibiansky Satterthwaite refinement is a degrees-of-
    freedom away).
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    m = np.asarray(mean_conc, dtype=float).reshape(-1)
    sd = np.asarray(sd_conc, dtype=float).reshape(-1)
    if not (t.size == m.size == sd.size) or t.size < 2:
        raise DataValidationError("times, means and sds must be equal-length (>= 2)")
    n = np.asarray(n_per_time, dtype=float)
    if n.ndim == 0:
        n = np.full(t.size, float(n))
    n = n.reshape(-1)
    if n.size != t.size or np.any(n < 1):
        raise DataValidationError("n_per_time must match the bins and be >= 1")
    if np.any(np.diff(t) <= 0):
        raise DataValidationError("times must be strictly increasing")
    # Trapezoid weights: interior points carry half of each neighbour
    # width, the endpoints half of their single width.
    widths = np.diff(t)
    w = np.zeros(t.size)
    w[:-1] += 0.5 * widths
    w[1:] += 0.5 * widths
    auc = float(np.sum(w * m))
    variance = float(np.sum(w**2 * sd**2 / n))
    return SparseAUC(auc=auc, variance=variance, standard_error=math.sqrt(variance))

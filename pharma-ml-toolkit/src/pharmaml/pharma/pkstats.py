"""Trial statistics for PK data: bioequivalence, dose proportionality,
DDI classification, meta-analysis and TOST power.

This module absorbs the ``pkpdutils.stats`` layer (König's PK/PD
utilities -- itself a transcription of Chow & Liu 2009, Schuirmann
1987, PowerTOST, Smith et al. 2000, the FDA 2020 / EMA 2012 DDI
guidance labels, DerSimonian & Laird 1982 and Higgins & Thompson
2002) into the toolkit's numpy/pandas idiom:

* **bioequivalence** -- the two one-sided tests (TOST) on the log
  scale with the 90 % CI / 80.00-125.00 % decision, the 2x2
  crossover's Chow-Liu period-difference decomposition (treatment,
  period, sequence effects and the within-subject CV), and the EMA
  reference-scaled (ABEL) widening ``exp(0.760·s_wR)`` capped at
  69.84-143.19 %.
* **exact TOST power / sample size** -- Owen's Q formulation as used
  by PowerTOST, integrated numerically (``scipy.integrate``), with the
  design table for ``2x2``, ``parallel``, ``2x2x4`` and ``2x2x3``.
* **dose proportionality** -- the Smith power-model slope criterion
  ``1 + ln(θ)/ln(dose ratio)`` on the 0.80-1.25 equivalence band.
* **DDI classification** -- the FDA/EMA AUC-ratio bands for
  inhibitors (>= 1.25 / 2 / 5) and inducers (<= 0.8 / 0.5 / 0.2) plus
  victim sensitivity labels.
* **meta-analysis** -- inverse-variance fixed and random (DerSimonian-
  Laird) pooling with Cochran's Q, I-squared and H-squared
  heterogeneity, and Hedges' g effect sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "ABEL_CAP",
    "ABEL_CV_CAP",
    "ABEL_K",
    "DDI_THRESHOLDS",
    "CrossoverResult",
    "DoseProportionalityResult",
    "MetaResult",
    "abel_limits",
    "bioequivalence_verdict",
    "crossover_2x2",
    "ddi_classify",
    "dose_proportionality",
    "hedges_g",
    "log_ratio_interval",
    "meta_analysis",
    "power_tost",
    "sample_size_tost",
    "substrate_sensitivity",
    "tost_pvalues",
]

# --------------------------------------------------------------------- #
# Ratio machinery (pkpdutils stats.ratio / stats.sample)
# --------------------------------------------------------------------- #


def log_ratio_interval(
    test: np.ndarray | list[float],
    reference: np.ndarray | list[float],
    ci_level: float = 0.90,
    paired: bool = False,
) -> tuple[float, float, float]:
    """Geometric-mean ratio with a t interval on the log scale.

    Paired data pairs by position (the crossover case); unpaired data
    uses the Welch-Satterthwaite degrees of freedom, exactly
    pkpdutils' ``ratio()``: ``ln GMR = mean(ln T) - mean(ln R)``, the
    CI is ``exp(ln GMR ± t_{1-α/2, df} · se)``.

    Examples
    --------
    >>> import numpy as np
    >>> test = np.array([128.0, 121.0, 131.0, 119.0, 125.0])
    >>> ref = np.array([117.0, 110.0, 118.0, 112.0, 109.0])
    >>> gmr, low, high = log_ratio_interval(test, ref, ci_level=0.90)
    >>> round(gmr, 3)
    1.102
    >>> low < gmr < high
    True
    >>> bool(high < 1.25 and low > 0.8)      # bioequivalent band
    True
    """
    test = np.asarray(test, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if test.size < 1 or reference.size < 1:
        raise DataValidationError("both groups need at least one value")
    if not 0.0 < ci_level < 1.0:
        raise DataValidationError("ci_level must lie in (0, 1)")
    alpha = 1.0 - ci_level
    from scipy import stats

    log_t, log_r = np.log(test), np.log(reference)
    if paired:
        if test.size != reference.size:
            raise DataValidationError("paired groups must have equal length")
        diffs = log_t - log_r
        gmr_log = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / math.sqrt(diffs.size)) if diffs.size > 1 else float("nan")
        df = diffs.size - 1
    else:
        gmr_log = float(np.mean(log_t) - np.mean(log_r))
        var_t = float(np.var(log_t, ddof=1)) if test.size > 1 else 0.0
        var_r = float(np.var(log_r, ddof=1)) if reference.size > 1 else 0.0
        se_t = var_t / test.size
        se_r = var_r / reference.size
        se = math.sqrt(se_t + se_r)
        df = (se_t + se_r) ** 2 / (se_t**2 / max(test.size - 1, 1) + se_r**2 / max(reference.size - 1, 1))
    gmr = math.exp(gmr_log)
    if not math.isfinite(se) or se <= 0 or df <= 0:
        return gmr, float("nan"), float("nan")
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return gmr, math.exp(gmr_log - t_crit * se), math.exp(gmr_log + t_crit * se)


def tost_pvalues(gmr: float, se_log: float, df: float, limits: tuple[float, float] = (0.8, 1.25)) -> tuple[float, float, float]:
    """Schuirmann's two one-sided tests: returns ``(p_lower, p_upper, p)``.

    ``t_L = (ln GMR - ln θ_L)/se``, ``t_U = (ln θ_U - ln GMR)/se``;
    each one-sided t test contributes a p-value and TOST's p is their
    maximum (the larger hurdle wins).

    Examples
    --------
    >>> p_low, p_up, p = tost_pvalues(1.02, 0.045, 28)
    >>> round(p, 4) < 0.05          # BE demonstrated at the 90% level
    True
    >>> p_low <= p and p_up <= p
    True
    """
    if se_log <= 0 or df <= 0:
        raise DataValidationError("se_log and df must be positive")
    from scipy import stats

    log_gmr = math.log(gmr)
    t_low = (log_gmr - math.log(limits[0])) / se_log
    t_up = (math.log(limits[1]) - log_gmr) / se_log
    p_low = float(stats.t.sf(t_low, df))
    p_up = float(stats.t.sf(t_up, df))
    return p_low, p_up, max(p_low, p_up)


def bioequivalence_verdict(
    ci_low: float,
    ci_high: float,
    limits: tuple[float, float] = (0.80, 1.25),
) -> bool:
    """The ICH M13A decision: the CI must sit fully inside the limits.

    Examples
    --------
    >>> bioequivalence_verdict(0.95, 1.18)
    True
    >>> bioequivalence_verdict(0.76, 1.10)
    False
    """
    return ci_low >= limits[0] and ci_high <= limits[1]


# --------------------------------------------------------------------- #
# 2x2 crossover (Chow & Liu via pkpdutils)
# --------------------------------------------------------------------- #


@dataclass
class CrossoverResult:
    """The 2x2 crossover decomposition (Chow & Liu chapter 3)."""

    gmr: float
    ci_low: float
    ci_high: float
    cv_intra: float
    se_log: float
    df: int
    p_treatment: float
    p_period: float
    p_sequence: float
    n_subjects: int


def crossover_2x2(frame: pd.DataFrame, ci_level: float = 0.90) -> CrossoverResult:
    """Analyse a 2x2 crossover (RT/TR) study on the log scale.

    Chow & Liu's period-difference method as transcribed by pkpdutils:
    per subject ``d_i = (y_i2 - y_i1)/2`` and ``u_i = y_i1 + y_i2``;
    the treatment effect is ``F = mean(d_A) - mean(d_B)`` (A = the
    sequence dosed test in period 2), the period effect
    ``P = mean(d_A) + mean(d_B)``, the sequence/carryover contrast uses
    the ``u`` sums, and the residual variance is ``σ_e² = 2σ_d²`` --
    giving ``cv_intra = sqrt(exp(σ_e²) - 1)`` and the 90 % CI of the
    GMR from the pooled within-sequence variance.

    The frame needs columns ``subject, sequence, period, formulation,
    value`` (sequence coded ``RT``/``TR``; period ``1``/``2``;
    formulation ``T``/``R`` or names whose T-rows differ from R-rows).

    Examples
    --------
    >>> frame = pd.DataFrame({
    ...     "subject":   [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    ...     "sequence": ["RT", "RT", "RT", "RT", "TR", "TR",
    ...                 "TR", "TR", "RT", "RT", "TR", "TR"],
    ...     "period":   [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ...     "formulation": ["R", "T", "R", "T", "T", "R",
    ...                     "T", "R", "R", "T", "T", "R"],
    ...     "value": [100, 104, 98, 103, 105, 99, 101, 97, 97, 102, 106, 100],
    ... })
    >>> res = crossover_2x2(frame)
    >>> round(res.gmr, 3)
    1.051
    >>> bool(res.ci_low > 0.8 and res.ci_high < 1.25)
    True
    >>> round(res.cv_intra, 3) < 0.10      # tiny within-subject noise here
    True
    """
    required = {"subject", "sequence", "period", "formulation", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"crossover frame misses columns {sorted(missing)}")
    wide = frame.pivot_table(
        index=["subject", "sequence"], columns="period", values="value", aggfunc="first"
    ).reset_index()
    if set(wide.columns) < {1, 2}:
        raise DataValidationError("both periods 1 and 2 must be present")
    wide = wide.dropna(subset=[1, 2])
    wide["d"] = (wide[2] - wide[1]) / 2.0
    wide["u"] = np.log(wide[1]) + np.log(wide[2])
    seq_a = wide[wide["sequence"].astype(str).str.upper().str.endswith("T")]  # TR: T in period 1
    seq_b = wide[wide["sequence"].astype(str).str.upper().str.endswith("R")]  # RT: R in period 1
    if seq_a.empty or seq_b.empty:
        raise DataValidationError("both RT and TR sequences are required")
    # In pkpdutils A = "test in period 2" = the RT sequence.
    a, b = seq_a, seq_b
    log_a = np.log(a[[1, 2]].to_numpy(dtype=float))
    log_b = np.log(b[[1, 2]].to_numpy(dtype=float))
    d_a = (log_a[:, 1] - log_a[:, 0]) / 2.0
    d_b = (log_b[:, 1] - log_b[:, 0]) / 2.0
    u_a = a["u"].to_numpy(dtype=float)
    u_b = b["u"].to_numpy(dtype=float)
    treatment = float(np.mean(d_a) - np.mean(d_b))   # F = mean(d_RT) - mean(d_TR)
    period = float(np.mean(d_a) + np.mean(d_b))
    n_a, n_b = d_a.size, d_b.size
    var_d = (
        (n_a - 1) * float(np.var(d_a, ddof=1)) + (n_b - 1) * float(np.var(d_b, ddof=1))
    ) / (n_a + n_b - 2)
    se = math.sqrt(var_d * (1.0 / n_a + 1.0 / n_b))
    df = n_a + n_b - 2
    var_u_a = float(np.var(u_a, ddof=1))
    var_u_b = float(np.var(u_b, ddof=1))
    from scipy import stats

    se_seq = math.sqrt(var_u_a / n_a + var_u_b / n_b)
    t_crit = float(stats.t.ppf(1.0 - (1.0 - ci_level) / 2.0, df))
    sigma_e_sq = 2.0 * var_d
    return CrossoverResult(
        gmr=math.exp(treatment),
        ci_low=math.exp(treatment - t_crit * se),
        ci_high=math.exp(treatment + t_crit * se),
        cv_intra=math.sqrt(math.expm1(sigma_e_sq)),
        se_log=se,
        df=df,
        p_treatment=float(2 * stats.t.sf(abs(treatment) / se, df)),
        p_period=float(2 * stats.t.sf(abs(period) / se, df)) if se > 0 else float("nan"),
        p_sequence=float(2 * stats.t.sf(abs(np.mean(u_a) - np.mean(u_b)) / se_seq, n_a + n_b - 2))
        if se_seq > 0
        else float("nan"),
        n_subjects=int(n_a + n_b),
    )


# --------------------------------------------------------------------- #
# ABEL widening (EMA reference-scaled average bioequivalence)
# --------------------------------------------------------------------- #

ABEL_K = 0.760
ABEL_CV_CAP = 0.50
ABEL_CAP = (0.6984, 1.4319)


def abel_limits(cv_intra: float) -> tuple[float, float]:
    """EMA ABEL widened limits for a highly variable reference.

    ``s_wR = sqrt(ln(1 + min(CV, 50%)²))`` and ``θ_U = exp(0.760·s_wR)``
    (the EMA cap), clipped to the 69.84-143.19 % window and never
    narrower than 80.00-125.00 %; the point estimate must still fall
    inside 80.00-125.00 % (the caller's job -- pkpdutils splits the
    rule the same way).

    Examples
    --------
    >>> round(abel_limits(0.30)[1], 4)     # at CV 30% the band is 80-125
    1.25
    >>> round(abel_limits(0.40)[1], 4)
    1.3402
    >>> round(abel_limits(0.60)[1], 4)     # capped at CV 50%
    1.4319
    """
    if cv_intra <= 0:
        raise DataValidationError("cv_intra must be positive")
    cv = min(cv_intra, ABEL_CV_CAP)
    if cv <= 0.30:
        return 0.80, 1.25
    s_wr = math.sqrt(math.log1p(cv**2))
    theta_up = math.exp(ABEL_K * s_wr)
    theta_up = min(max(theta_up, 1.25), ABEL_CAP[1])
    return 1.0 / theta_up, theta_up


# --------------------------------------------------------------------- #
# Exact TOST power (Owen's Q, PowerTOST via pkpdutils)
# --------------------------------------------------------------------- #

_DESIGNS: dict[str, dict[str, Any]] = {
    "2x2": {"bk": 2.0, "df": lambda n: n - 2, "step": 2, "start": 4},
    "parallel": {"bk": 4.0, "df": lambda n: n - 2, "step": 1, "start": 3},
    "2x2x4": {"bk": 1.0, "df": lambda n: 3 * n - 4, "step": 2, "start": 4},
    "2x2x3": {"bk": 1.5, "df": lambda n: 2 * n - 3, "step": 2, "start": 3},
}


def _owens_q(nu: float, t: float, delta: float, a: float, b: float) -> float:
    """Owen's Q ``Q_ν(t, δ; a, b)`` by adaptive quadrature (PowerTOST's route)."""
    from scipy import integrate, special, stats

    c = special.gamma(nu / 2.0) * 2.0 ** ((nu - 2.0) / 2.0)

    def integrand(x: float) -> float:
        return stats.norm.cdf(t * x / math.sqrt(nu) - delta) * x ** (nu - 1.0) * math.exp(-x * x / 2.0)

    peak = math.sqrt(max(nu - 1.0, 0.0))
    points = [p for p in (peak, a, b) if a <= p <= b]
    result, _ = integrate.quad(integrand, a, b, points=points or None, limit=200)
    return result / c


def power_tost(
    cv: float,
    n: int,
    gmr: float = 0.95,
    limits: tuple[float, float] = (0.8, 1.25),
    design: str = "2x2",
    alpha: float = 0.05,
) -> float:
    """Exact TOST power via Owen's Q (the PowerTOST algorithm).

    ``power = Q_ν(−t, δ₂; 0, R) − Q_ν(t, δ₁; 0, R)`` with
    ``δ_k = (ln GMR − ln θ_k)/se``, ``R = (δ₁−δ₂)√ν/(2t)`` and the
    design's degrees of freedom (``2x2``: ``ν = n−2``; ``parallel``:
    ``bk = 4``; replicate ``2x2x4``/``2x2x3`` as tabulated).

    Examples
    --------
    >>> round(power_tost(0.25, 28, gmr=0.95, design="2x2"), 3)
    0.807
    >>> round(power_tost(0.25, 12, gmr=0.95, design="2x2"), 3) < 0.4
    True
    >>> power_tost(0.25, 28, gmr=1.20, design="2x2") < 0.2   # off-centre truth
    True
    """
    if cv <= 0 or n < 2:
        raise DataValidationError("cv must be positive and n >= 2")
    if design not in _DESIGNS:
        raise DataValidationError(f"design must be one of {sorted(_DESIGNS)}")
    if not limits[0] < gmr < limits[1]:
        return 0.0
    from scipy import stats

    sigma = math.sqrt(math.log1p(cv**2))
    spec = _DESIGNS[design]
    bk = spec["bk"]
    nu = spec["df"](n)
    half = math.ceil(n / 2.0)
    other = n - half
    se = sigma * math.sqrt(bk / 4.0 * (1.0 / half + 1.0 / other))
    if se <= 0:
        return 1.0
    delta_1 = (math.log(gmr) - math.log(limits[0])) / se
    delta_2 = (math.log(gmr) - math.log(limits[1])) / se
    t = float(stats.t.ppf(1.0 - alpha, nu))
    r = (delta_1 - delta_2) * math.sqrt(nu) / (2.0 * t)
    power = _owens_q(nu, -t, delta_2, 0.0, r) - _owens_q(nu, t, delta_1, 0.0, r)
    return float(min(max(power, 0.0), 1.0))


def sample_size_tost(
    cv: float,
    gmr: float = 0.95,
    target_power: float = 0.8,
    design: str = "2x2",
    alpha: float = 0.05,
    limits: tuple[float, float] = (0.8, 1.25),
    max_n: int = 2000,
) -> int:
    """Smallest ``n`` reaching ``target_power`` (PowerTOST's search).

    Starts from the normal-approximation lower bound
    ``n ≈ bk·ln(1+cv²)·(z_{1-α}+z_{power})² / min|ln GMR − ln θ|²``,
    then walks the design's step (2 for crossovers, 1 for parallel)
    upward and steps back down while the power holds -- the exact
    PowerTOST sample-size dance.

    Examples
    --------
    >>> sample_size_tost(0.25, gmr=0.95, design="2x2")
    28
    >>> sample_size_tost(0.30, gmr=0.95, design="2x2")
    40
    >>> sample_size_tost(0.35, gmr=0.90, design="2x2") > 40
    True
    """
    if not limits[0] < gmr < limits[1]:
        raise DataValidationError("gmr must lie inside the equivalence limits")
    from scipy import stats

    spec = _DESIGNS[design]
    bk = spec["bk"]
    sigma_sq = math.log1p(cv**2)
    z_alpha = stats.norm.ppf(1.0 - alpha)
    z_power = stats.norm.ppf(target_power)
    distance = min(abs(math.log(gmr) - math.log(limits[0])), abs(math.log(limits[1]) - math.log(gmr)))
    n_approx = bk * sigma_sq * (z_alpha + z_power) ** 2 / distance**2
    n = max(spec["start"], math.floor(n_approx) - (math.floor(n_approx) % spec["step"]))
    n = max(n, spec["start"])
    while n <= max_n and power_tost(cv, n, gmr=gmr, limits=limits, design=design, alpha=alpha) < target_power:
        n += spec["step"]
    while (
        n - spec["step"] >= spec["start"]
        and power_tost(cv, n - spec["step"], gmr=gmr, limits=limits, design=design, alpha=alpha)
        >= target_power
    ):
        n -= spec["step"]
    return int(n)


# --------------------------------------------------------------------- #
# Dose proportionality (Smith et al. 2000 via pkpdutils)
# --------------------------------------------------------------------- #


@dataclass
class DoseProportionalityResult:
    """The power-model verdict on dose proportionality."""

    slope: float
    ci_low: float
    ci_high: float
    bounds: tuple[float, float]
    verdict: str
    r2: float


def dose_proportionality(
    doses: np.ndarray | list[float],
    exposures: np.ndarray | list[float],
    criterion: tuple[float, float] = (0.8, 1.25),
    level: float = 0.95,
) -> DoseProportionalityResult:
    """Test dose proportionality with the Smith power-model criterion.

    Fit ``ln AUC = ln a + b·ln D`` (the power model); with the dose
    ratio ``r = D_high/D_low`` the equivalence band for the slope is
    ``[1 + ln(θ_L)/ln r, 1 + ln(θ_H)/ln r]`` -- the slope CI inside it
    means "proportional", overlapping means "inconclusive", outside
    means "not proportional" (Smith et al. 2000, exactly pkpdutils'
    ``proportionality_test``).

    Examples
    --------
    >>> import numpy as np
    >>> doses = np.array([25.0, 50.0, 100.0, 200.0, 400.0])
    >>> # AUC ∝ D^1.0 with a little noise → proportional
    >>> auc = np.array([105.0, 198.0, 410.0, 805.0, 1620.0])
    >>> res = dose_proportionality(doses, auc)
    >>> res.verdict
    'proportional'
    >>> round(res.slope, 2)
    0.99
    >>> # AUC ∝ D^1.3 → clearly super-proportional
    >>> auc2 = np.array([105.0, 260.0, 640.0, 1580.0, 3900.0])
    >>> dose_proportionality(doses, auc2).verdict
    'not proportional'
    """
    doses = np.asarray(doses, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    if doses.size != exposures.size or doses.size < 3:
        raise DataValidationError("need >= 3 paired (dose, exposure) points")
    if np.any(doses <= 0) or np.any(exposures <= 0):
        raise DataValidationError("doses and exposures must be positive")
    from scipy import stats

    log_d, log_e = np.log(doses), np.log(exposures)
    slope, intercept = np.polyfit(log_d, log_e, 1)
    fitted = slope * log_d + intercept
    ss_res = float(np.sum((log_e - fitted) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = doses.size
    resid_var = ss_res / (n - 2)
    se = math.sqrt(resid_var / float(np.sum((log_d - log_d.mean()) ** 2)))
    t_crit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, n - 2))
    ci = (float(slope - t_crit * se), float(slope + t_crit * se))
    ratio = float(np.max(doses) / np.min(doses))
    bounds = (1.0 + math.log(criterion[0]) / math.log(ratio),
              1.0 + math.log(criterion[1]) / math.log(ratio))
    if ci[0] >= bounds[0] and ci[1] <= bounds[1]:
        verdict = "proportional"
    elif ci[1] < bounds[0] or ci[0] > bounds[1]:
        verdict = "not proportional"
    else:
        verdict = "inconclusive"
    return DoseProportionalityResult(
        slope=float(slope), ci_low=ci[0], ci_high=ci[1], bounds=bounds, verdict=verdict, r2=float(r2)
    )


# --------------------------------------------------------------------- #
# DDI classification (FDA 2020 == EMA 2012 via pkpdutils)
# --------------------------------------------------------------------- #

DDI_THRESHOLDS: dict[str, float] = {
    "inhibitor_weak": 1.25,
    "inhibitor_moderate": 2.0,
    "inhibitor_strong": 5.0,
    "inducer_weak": 0.8,
    "inducer_moderate": 0.5,
    "inducer_strong": 0.2,
    "sensitive": 5.0,
    "moderately_sensitive": 2.0,
}


def ddi_classify(auc_ratio: float) -> str:
    """Label an AUC ratio with the FDA/EMA DDI bands.

    Inhibitors: ``>= 5`` strong, ``>= 2`` moderate, ``>= 1.25`` weak;
    inducers: ``<= 0.2`` strong, ``<= 0.5`` moderate, ``<= 0.8`` weak;
    otherwise ``none`` -- the exact pkpdutils/FDA 2020 table.

    Examples
    --------
    >>> ddi_classify(1.05)
    'none'
    >>> ddi_classify(1.4)
    'weak inhibitor'
    >>> ddi_classify(3.2)
    'moderate inhibitor'
    >>> ddi_classify(6.0)
    'strong inhibitor'
    >>> ddi_classify(0.65)
    'weak inducer'
    >>> ddi_classify(0.15)
    'strong inducer'
    """
    if not math.isfinite(auc_ratio) or auc_ratio <= 0:
        raise DataValidationError("auc_ratio must be a positive finite number")
    if auc_ratio >= DDI_THRESHOLDS["inhibitor_strong"]:
        return "strong inhibitor"
    if auc_ratio >= DDI_THRESHOLDS["inhibitor_moderate"]:
        return "moderate inhibitor"
    if auc_ratio >= DDI_THRESHOLDS["inhibitor_weak"]:
        return "weak inhibitor"
    if auc_ratio <= DDI_THRESHOLDS["inducer_strong"]:
        return "strong inducer"
    if auc_ratio <= DDI_THRESHOLDS["inducer_moderate"]:
        return "moderate inducer"
    if auc_ratio <= DDI_THRESHOLDS["inducer_weak"]:
        return "weak inducer"
    return "none"


def substrate_sensitivity(auc_ratio: float) -> str:
    """Victim sensitivity label (FDA): ``>= 5`` sensitive, ``>= 2`` moderate.

    Examples
    --------
    >>> substrate_sensitivity(6.0)
    'sensitive substrate'
    >>> substrate_sensitivity(2.5)
    'moderately sensitive substrate'
    >>> substrate_sensitivity(1.1)
    'none'
    """
    if auc_ratio >= DDI_THRESHOLDS["sensitive"]:
        return "sensitive substrate"
    if auc_ratio >= DDI_THRESHOLDS["moderately_sensitive"]:
        return "moderately sensitive substrate"
    return "none"


# --------------------------------------------------------------------- #
# Meta-analysis (DerSimonian-Laird via pkpdutils)
# --------------------------------------------------------------------- #


def hedges_g(test: np.ndarray | list[float], control: np.ndarray | list[float]) -> tuple[float, float]:
    """Hedges' g effect size and its variance.

    ``d = (mean_T - mean_C)/s_p`` with the small-sample correction
    ``J = 1 - 3/(4N - 9)`` and ``var(d) = N/(n_T n_C) + d²/(2N)``
    scaled by ``J²`` -- the exact pkpdutils/Hedges formulas.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([3.1, 3.4, 2.9, 3.6, 3.2])
    >>> c = np.array([2.4, 2.6, 2.3, 2.8, 2.5])
    >>> g, var = hedges_g(t, c)
    >>> round(g, 2)
    2.77
    >>> var > 0
    True
    """
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    if test.size < 2 or control.size < 2:
        raise DataValidationError("both groups need at least two values")
    n_t, n_c = test.size, control.size
    big_n = n_t + n_c
    pooled_var = (
        (n_t - 1) * float(np.var(test, ddof=1)) + (n_c - 1) * float(np.var(control, ddof=1))
    ) / (big_n - 2)
    pooled_sd = math.sqrt(pooled_var)
    if pooled_sd <= 0:
        raise DataValidationError("pooled SD degenerated to zero")
    d = (float(np.mean(test)) - float(np.mean(control))) / pooled_sd
    j = 1.0 - 3.0 / (4.0 * big_n - 9.0)
    var_d = big_n / (n_t * n_c) + d * d / (2.0 * big_n)
    return j * d, j * j * var_d


@dataclass
class MetaResult:
    """Fixed- and random-effects meta-analysis with heterogeneity."""

    effects: np.ndarray
    variances: np.ndarray
    fixed_effect: float
    fixed_se: float
    random_effect: float
    random_se: float
    tau_sq: float
    q: float
    i_sq: float
    h_sq: float
    k: int


def meta_analysis(
    effects: np.ndarray | list[float],
    variances: np.ndarray | list[float],
    ci_level: float = 0.95,
) -> MetaResult:
    """Inverse-variance pooling: fixed + DerSimonian-Laird random effects.

    Fixed weights ``w = 1/v``; the random model adds
    ``τ² = max(0, (Q - (k-1))/C)`` with ``C = Σw - Σw²/Σw`` and pools
    with ``w = 1/(v + τ²)``.  Heterogeneity: Cochran's ``Q`` with
    ``k-1`` df, ``I² = max(0, (Q-df)/Q)·100`` % and ``H² = Q/df``
    (Higgins & Thompson 2002, exactly pkpdutils' ``meta_analysis``).

    Examples
    --------
    >>> import numpy as np
    >>> # heterogeneous studies: Q > k-1, so tau^2 > 0 and the random
    >>> # estimate shrinks toward the smaller (better-precised) effects
    >>> eff = np.array([0.1, 0.8, 0.15, 0.75])
    >>> var = np.array([0.04, 0.05, 0.03, 0.06])
    >>> res = meta_analysis(eff, var)
    >>> round(res.fixed_effect, 3)
    0.379
    >>> res.random_effect != res.fixed_effect
    True
    >>> res.i_sq > 50 and res.h_sq > 2
    True
    >>> # homogeneous studies → Q small, τ² = 0
    >>> res2 = meta_analysis(np.array([0.4, 0.41, 0.39, 0.40]), var)
    >>> round(res2.tau_sq, 6)
    0.0
    """
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if effects.size != variances.size or effects.size < 2:
        raise DataValidationError("need >= 2 studies with (effect, variance)")
    if np.any(variances <= 0):
        raise DataValidationError("variances must be positive")
    if not 0.0 < ci_level < 1.0:
        raise DataValidationError("ci_level must lie in (0, 1)")
    k = effects.size
    w = 1.0 / variances
    fixed = float(np.sum(w * effects) / np.sum(w))
    q = float(np.sum(w * (effects - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau_sq = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_random = 1.0 / (variances + tau_sq)
    random_eff = float(np.sum(w_random * effects) / np.sum(w_random))
    df = k - 1
    i_sq = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return MetaResult(
        effects=effects,
        variances=variances,
        fixed_effect=fixed,
        fixed_se=1.0 / math.sqrt(float(np.sum(w))),
        random_effect=random_eff,
        random_se=1.0 / math.sqrt(float(np.sum(w_random))),
        tau_sq=float(tau_sq),
        q=q,
        i_sq=float(i_sq),
        h_sq=float(q / df) if df > 0 else float("nan"),
        k=int(k),
    )

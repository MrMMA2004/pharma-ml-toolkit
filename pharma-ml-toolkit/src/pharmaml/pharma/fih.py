r"""The first-in-human dose framework, absorbed natively.

``dcmgithb/Pharmacometrics_Portfolio`` (projects 03 and 05) was
reviewed file by file for this module: the preclinical-to-human
allometric translation, the MABEL and NOAEL-derived starting-dose
logic (ICH M3(R2)'s two independent floors), the operating-
characteristics trial simulation and the estimation diagnostics
(shrinkage, condition number) a popPK workflow reports.  The R/nlmixr2
machinery the portfolio leans on is the not-vendored part; every
formula here is plain NumPy, and the rank-sum test reuses the
toolkit's own :func:`~pharmaml.ml.stats.mann_whitney_test`.

The two floors of a FIH starting dose:

* **MABEL** (Minimum Anticipated Biological Effect Level): the
  concentration that occupies 10% of the receptor (ICH S6's
  convention) -- ``C_free = EC50 · RO / (1 − RO)`` -- scaled to a
  total (bound + unbound) concentration through ``fu``, converted to
  mg/L through the molecular weight, and multiplied by the volume of
  distribution.
* **NOAEL-derived HED**: the animal no-observed-adverse-effect level
  converted to a human equivalent dose by allometric scaling --
  ``HED = NOAEL · BW_animal · (BW_human / BW_animal)^(1 − exponent)``
  -- then divided by the safety factor (10x, ICH M3(R2)).

The starting dose is the *minimum* of the two floors.

Examples
--------
>>> round(receptor_occupancy(9.0, ec50=1.0), 3)      # 90% occupancy
0.9
>>> round(mabel_free_concentration(1.0, target_ro=0.10), 3)
0.111
>>> round(nm_to_mg_per_l(1.0, mw_da=400.0), 4)
0.0004
>>> mabel = mabel_dose(ec50_nm=1.0, mw_da=400.0, fu=0.10, vd_l=100.0)
>>> round(mabel, 4)
0.0444
>>> round(noael_hed(noael_mg_per_kg=100.0, animal_bw=0.25), 1)
102.3
>>> fih = fih_starting_dose(ec50_nm=1.0, mw_da=400.0, fu=0.10, vd_l=100.0,
...                         noael_mg_per_kg=100.0, animal_bw=0.25)
>>> round(fih, 4)                     # the MABEL floor wins here
0.0444

>>> round(allometric_clearance(5.0, 7.0), 3)         # a 7 kg child
0.889
>>> fit = fit_allography([0.25, 10.0], [0.020, 1.20], exponent=0.75)
>>> round(fit.alpha, 4)
0.1099
>>> round(fit.predict(70.0), 3)
2.659
>>> round(fit.human_half_life(v_l=70.0, bw=70.0), 2)
18.25

>>> shrunk = eta_shrinkage([[-0.2, 0.1], [0.3, -0.4], [0.0, 0.3]], [0.3, 0.5])
>>> [round(float(v), 2) for v in shrunk]
[0.16, 0.28]
>>> round(condition_number([[2.0, 0.0], [0.0, 8.0]]), 3)
2.0
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "AllometryFit",
    "TrialPowerReport",
    "allometric_clearance",
    "allometric_volume",
    "condition_number",
    "eta_shrinkage",
    "fih_starting_dose",
    "fit_allography",
    "mabel_dose",
    "mabel_free_concentration",
    "nm_to_mg_per_l",
    "noael_hed",
    "receptor_occupancy",
    "trial_power",
]

logger = get_logger(__name__)

#: The default MABEL receptor occupancy (ICH S6 / S9's 10% rule).
DEFAULT_MABEL_RO: float = 0.10

#: The default interspecies safety factor (ICH M3(R2)'s 10x).
DEFAULT_SAFETY_FACTOR: float = 10.0

#: Reference human body weight for the allometric anchor.
REFERENCE_HUMAN_KG: float = 70.0

#: The default allometric exponents (clearance 0.75, volume 1.0).
CLEARANCE_EXPONENT: float = 0.75
VOLUME_EXPONENT: float = 1.0


# --------------------------------------------------------------------------
# MABEL
# -------------------------------------------------------------------------- #


def receptor_occupancy(concentration: float, ec50: float) -> float:
    """The simple occupancy model ``RO = C / (EC50 + C)``.

    Examples
    --------
    >>> round(receptor_occupancy(1.0, 1.0), 3)
    0.5
    """
    if ec50 <= 0:
        raise DataValidationError(f"ec50 must be positive, got {ec50}")
    if concentration < 0:
        raise DataValidationError(f"concentration must be >= 0, got {concentration}")
    return concentration / (ec50 + concentration)


def mabel_free_concentration(ec50: float, target_ro: float = DEFAULT_MABEL_RO) -> float:
    """The free concentration at a target occupancy, ``C = EC50·RO/(1−RO)``.

    The inverted occupancy model -- the concentration that occupies
    exactly ``target_ro`` of the receptor.  ICH S6's 10% rule makes
    this the MABEL exposure anchor.
    """
    if ec50 <= 0:
        raise DataValidationError(f"ec50 must be positive, got {ec50}")
    if not 0.0 < target_ro < 1.0:
        raise DataValidationError(f"target_ro must lie in (0, 1), got {target_ro}")
    return ec50 * target_ro / (1.0 - target_ro)


def nm_to_mg_per_l(concentration_nm: float, mw_da: float) -> float:
    """Convert a molar concentration (nM) to mg/L via the molecular weight.

    ``1 nM of a 400 Da compound = 0.4 µg/L = 0.0004 mg/L``.
    """
    if mw_da <= 0:
        raise DataValidationError(f"mw_da must be positive, got {mw_da}")
    return concentration_nm * mw_da / 1e6


def mabel_dose(
    ec50_nm: float,
    mw_da: float,
    fu: float,
    vd_l: float,
    target_ro: float = DEFAULT_MABEL_RO,
) -> float:
    """The MABEL starting dose in mg.

    The portfolio's project-03 chain: free target concentration from
    the occupancy inversion, scaled to total by ``1/fu``, converted to
    mg/L by the molecular weight, multiplied by the volume of
    distribution.  Every argument is the study's own (an in-vitro
    EC50, the compound's MW and protein binding, the allometry-derived
    human Vd).

    Examples
    --------
    >>> round(mabel_dose(ec50_nm=1.0, mw_da=400.0, fu=0.10, vd_l=100.0), 4)
    0.0444
    """
    if not 0.0 < fu <= 1.0:
        raise DataValidationError(f"fu must lie in (0, 1], got {fu}")
    if vd_l <= 0:
        raise DataValidationError(f"vd_l must be positive, got {vd_l}")
    free_nm = mabel_free_concentration(ec50_nm, target_ro)
    total_mg_l = nm_to_mg_per_l(free_nm / fu, mw_da)
    return total_mg_l * vd_l


# --------------------------------------------------------------------------
# NOAEL -> HED
# -------------------------------------------------------------------------- #


def noael_hed(
    noael_mg_per_kg: float,
    animal_bw: float,
    human_bw: float = REFERENCE_HUMAN_KG,
    exponent: float = CLEARANCE_EXPONENT,
) -> float:
    """The NOAEL-scaled human equivalent dose, in mg.

    ``HED = NOAEL [mg/kg] · BW_animal · (BW_human / BW_animal)^(1 −
    exponent)`` -- the toxicology convention that dose scales with
    body weight to the clearance exponent's complement (0.25 for the
    0.75 clearance exponent), which is the FDA's default
    kg^(1/4) scaling for small animals.

    Examples
    --------
    >>> round(noael_hed(100.0, 0.25), 1)      # a 100 mg/kg rat NOAEL
    102.3
    """
    if noael_mg_per_kg <= 0:
        raise DataValidationError(f"noael_mg_per_kg must be positive, got {noael_mg_per_kg}")
    if animal_bw <= 0 or human_bw <= 0:
        raise DataValidationError(
            f"body weights must be positive, got {animal_bw} / {human_bw}"
        )
    if not 0.0 < exponent < 1.0:
        raise DataValidationError(f"exponent must lie in (0, 1), got {exponent}")
    return noael_mg_per_kg * animal_bw * (human_bw / animal_bw) ** (1.0 - exponent)


def fih_starting_dose(
    ec50_nm: float,
    mw_da: float,
    fu: float,
    vd_l: float,
    noael_mg_per_kg: float,
    animal_bw: float,
    human_bw: float = REFERENCE_HUMAN_KG,
    target_ro: float = DEFAULT_MABEL_RO,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> float:
    """The FIH starting dose: the minimum of the MABEL and NOAEL floors.

    The portfolio's decision rule: MABEL in mg against the
    safety-factored NOAEL HED, and the smaller of the two is the
    first-in-human starting dose (ICH M3(R2)'s "whichever is lower").

    Examples
    --------
    >>> round(fih_starting_dose(1.0, 400.0, 0.10, 100.0, 100.0, 0.25), 4)
    0.0444
    """
    if safety_factor <= 0:
        raise DataValidationError(f"safety_factor must be positive, got {safety_factor}")
    mabel = mabel_dose(ec50_nm, mw_da, fu, vd_l, target_ro=target_ro)
    hed = noael_hed(noael_mg_per_kg, animal_bw, human_bw=human_bw) / safety_factor
    return min(mabel, hed)


# --------------------------------------------------------------------------
# allometry
# -------------------------------------------------------------------------- #


def allometric_clearance(
    clearance: float,
    body_weight: float,
    exponent: float = CLEARANCE_EXPONENT,
    reference_bw: float = REFERENCE_HUMAN_KG,
) -> float:
    """Scale a clearance to another body weight, ``CL·(BW/ref)^0.75``.

    The exponent is the fixed physiological one (0.75 -- the
    allometric cascade every preclinical translation uses), not an
    estimated one.

    Examples
    --------
    >>> round(allometric_clearance(5.0, 7.0), 3)
    0.889
    """
    if body_weight <= 0 or reference_bw <= 0:
        raise DataValidationError(
            f"body weights must be positive, got {body_weight} / {reference_bw}"
        )
    return clearance * (body_weight / reference_bw) ** exponent


def allometric_volume(
    volume: float,
    body_weight: float,
    exponent: float = VOLUME_EXPONENT,
    reference_bw: float = REFERENCE_HUMAN_KG,
) -> float:
    """Scale a volume to another body weight, ``V·(BW/ref)^1.0``."""
    if body_weight <= 0 or reference_bw <= 0:
        raise DataValidationError(
            f"body weights must be positive, got {body_weight} / {reference_bw}"
        )
    return volume * (body_weight / reference_bw) ** exponent


@dataclass
class AllometryFit:
    """A multi-species allometry fit and its human prediction.

    Attributes
    ----------
    alpha : float
        The allometric coefficient: ``value = alpha · BW^exponent``.
    exponent : float
        The (fixed or fitted) exponent.
    r_squared : float
        R² of the log-log regression when the exponent was fitted.
    fitted : bool
        Whether the exponent was estimated from the data (True) or
        pinned by the caller (False -- the two-species reality).
    """

    alpha: float
    exponent: float
    r_squared: float
    fitted: bool

    def predict(self, body_weight: float) -> float:
        """The predicted value at another body weight."""
        if body_weight <= 0:
            raise DataValidationError(f"body_weight must be positive, got {body_weight}")
        return self.alpha * body_weight**self.exponent

    def human_half_life(self, v_l: float, bw: float) -> float:
        """``t½ = ln2 · V / CL`` at one body weight.

        The portfolio's out-of-sample sanity line: the human half-life
        implied by the predicted clearance and the (separately scaled)
        human volume.
        """
        if v_l <= 0 or bw <= 0:
            raise DataValidationError(f"v_l and bw must be positive, got {v_l} / {bw}")
        clearance = self.predict(bw)
        if clearance <= 0:
            raise DataValidationError("the predicted clearance is not positive")
        return math.log(2.0) * v_l / clearance


def fit_allography(
    body_weights: Sequence[float],
    values: Sequence[float],
    exponent: float | None = None,
) -> AllometryFit:
    """Fit (or pin) the allometric power law across species.

    With ``exponent=None`` the power law is fitted by log-log least
    squares (needs >= 3 species).  With a pinned exponent -- the
    portfolio's two-species reality -- the coefficient is the
    *geometric mean* of the per-species alphas,
    ``alpha = exp(mean(log(value / BW^exponent)))``, exactly the
    portfolio's aggregation.

    Examples
    --------
    >>> fit = fit_allography([0.25, 10.0], [0.020, 1.20], exponent=0.75)
    >>> round(fit.alpha, 4)
    0.1099
    >>> round(fit.predict(70.0), 3)
    2.659
    """
    weights = np.asarray(body_weights, dtype=float)
    observed = np.asarray(values, dtype=float)
    if len(weights) != len(observed) or len(weights) < 2:
        raise DataValidationError(
            "fit_allography: need equal weight/value vectors of at least 2 species"
        )
    if (weights <= 0).any() or (observed <= 0).any():
        raise DataValidationError("fit_allography: weights and values must be positive")
    if exponent is not None:
        if not 0.0 < exponent < 1.5:
            raise DataValidationError(f"exponent out of range, got {exponent}")
        alpha = float(np.exp(np.mean(np.log(observed / weights**exponent))))
        return AllometryFit(alpha=alpha, exponent=float(exponent), r_squared=0.0, fitted=False)
    if len(weights) < 3:
        raise DataValidationError(
            "fit_allography: fitting the exponent needs at least 3 species -- pin it instead"
        )
    slope, intercept = np.polyfit(np.log(weights), np.log(observed), 1)
    predicted = intercept + slope * np.log(weights)
    ss_res = float(((np.log(observed) - predicted) ** 2).sum())
    ss_tot = float(((np.log(observed) - np.log(observed).mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return AllometryFit(
        alpha=float(np.exp(intercept)), exponent=float(slope), r_squared=r_squared, fitted=True
    )


# --------------------------------------------------------------------------
# estimation diagnostics
# -------------------------------------------------------------------------- #


def eta_shrinkage(
    eta: Sequence[Sequence[float]] | np.ndarray,
    omega: Sequence[float],
) -> np.ndarray:
    """The η-shrinkage per random effect, ``1 − SD(η)/ω``.

    The diagnostic every popPK workflow reports: how much of the
    *assumed* variability the data actually identified.  Above ~25-30%
    (EMA/FDA thresholds) the individual estimates collapse toward the
    population mean and covariate searches on that effect are
    untrustworthy.

    Parameters
    ----------
    eta : (n_subjects, n_effects) array-like
        The individual random-effect estimates.
    omega : (n_effects,) sequence
        The assumed random-effect standard deviations (not variances).

    Examples
    --------
    >>> values = eta_shrinkage([[-0.2, 0.1], [0.3, -0.4], [0.0, 0.3]], [0.3, 0.5])
    >>> round(float(values[0]), 2)
    0.16
    >>> round(float(values[1]), 2)
    0.28
    """
    matrix = np.asarray(eta, dtype=float)
    omegas = np.asarray(omega, dtype=float)
    if matrix.ndim != 2:
        raise DataValidationError("eta_shrinkage: eta must be a 2-D (subjects x effects) array")
    if matrix.shape[1] != len(omegas):
        raise DataValidationError(
            f"eta_shrinkage: {matrix.shape[1]} effects in eta vs {len(omegas)} omegas"
        )
    if (omegas <= 0).any():
        raise DataValidationError("eta_shrinkage: every omega must be positive")
    if matrix.shape[0] < 2:
        raise DataValidationError("eta_shrinkage: need at least two subjects")
    sd = matrix.std(axis=0, ddof=1)
    return 1.0 - sd / omegas


def condition_number(matrix: Sequence[Sequence[float]] | np.ndarray) -> float:
    """The popPK covariance condition number, ``sqrt(λmax/λmin)``.

    The parameter-correlation diagnostic: above ~1000 the estimates
    are effectively unidentifiable together.  Uses the symmetric
    eigenvalue decomposition (:func:`numpy.linalg.eigvalsh`), so pass
    a symmetric matrix (a covariance, not its raw data).
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise DataValidationError(
            f"condition_number: need a square matrix >= 2x2, got shape {arr.shape}"
        )
    eigenvalues = np.linalg.eigvalsh(arr)
    if eigenvalues.min() <= 0:
        raise DataValidationError("condition_number: the matrix is not positive definite")
    return float(math.sqrt(eigenvalues.max() / eigenvalues.min()))


# --------------------------------------------------------------------------
# the trial simulation (project 05)
# -------------------------------------------------------------------------- #


@dataclass
class TrialPowerReport:
    """The operating characteristics of one simulated trial design.

    Attributes
    ----------
    power : float
        The fraction of replicates where the active arm beat placebo
        (one-sided Wilcoxon rank-sum, alpha = 0.05).
    power_ci : tuple[float, float]
        The binomial Monte-Carlo 95% interval around ``power``.
    type_one_error : float
        The placebo-vs-placebo rejection rate (the design's honesty check).
    ec50_bias_pct : float
        Mean signed % bias of the re-fitted EC50 across replicates.
    ec50_rmse_pct : float
        The root-mean-square % error of the re-fitted EC50.
    ec50_coverage : float
        The fraction of replicates whose re-fitted EC50 landed within
        ±30% of the truth (the portfolio's success criterion).
    n_replicates : int
        How many Monte-Carlo replicates ran.
    design : str
        One line describing the design that was run.
    """

    power: float
    power_ci: tuple[float, float]
    type_one_error: float
    ec50_bias_pct: float
    ec50_rmse_pct: float
    ec50_coverage: float
    n_replicates: int
    design: str

    def line(self) -> str:
        """The one-line verdict for reports and logs."""
        return (
            f"{self.design}: power {self.power:.0%} "
            f"(95% MC {self.power_ci[0]:.0%}-{self.power_ci[1]:.0%}), "
            f"type-I {self.type_one_error:.1%}, EC50 bias {self.ec50_bias_pct:+.1f}%, "
            f"RMSE {self.ec50_rmse_pct:.1f}%, ±30% coverage {self.ec50_coverage:.0%}"
        )


def trial_power(
    doses: Sequence[float] = (50.0, 100.0, 200.0, 400.0),
    n_per_arm: int = 40,
    replicates: int = 200,
    e0: float = 100.0,
    emax: float = 80.0,
    ec50_auc: float = 2.5,
    hill: float = 1.8,
    cl_l_hr: float = 3.0,
    bioavailability: float = 0.85,
    iiv_cl: float = 0.30,
    iiv_ec50: float = 0.40,
    residual_sd: float = 8.0,
    alpha: float = 0.05,
    seed: int = 2025,
) -> TrialPowerReport:
    """The Monte-Carlo operating characteristics of a dose-ranging design.

    The portfolio's project 05, NumPy-side: every replicate simulates
    ``len(doses)`` active arms + one placebo arm of ``n_per_arm``
    subjects, each with log-normal IIV on clearance and EC50; the
    exposure metric is ``AUC = dose·F/CL``, the effect the sigmoid
    Emax on it, the analysis a one-sided Wilcoxon rank-sum of the best
    dose against placebo plus a 4-parameter Emax re-fit on the arm
    means (the toolkit's own
    :func:`~pharmaml.ml.fitting.curve_fit`).

    The best dose for the test is the one nearest the true EC50 (the
    design intuition the project demonstrates: power peaks where the
    dose-response is steepest).

    Examples
    --------
    >>> report = trial_power(doses=(100.0, 200.0), n_per_arm=30,
    ...                      replicates=60, seed=7)
    >>> 0.5 < report.power <= 1.0
    True
    >>> 0.0 <= report.type_one_error <= 0.2
    True
    >>> "power" in report.line()
    True
    """
    doses_arr = np.asarray(doses, dtype=float)
    if len(doses_arr) < 1 or (doses_arr <= 0).any():
        raise DataValidationError("trial_power: need at least one positive dose")
    if n_per_arm < 2:
        raise DataValidationError(f"n_per_arm must be >= 2, got {n_per_arm}")
    if replicates < 2:
        raise DataValidationError(f"replicates must be >= 2, got {replicates}")
    if not 0.0 < bioavailability <= 1.0:
        raise DataValidationError(f"bioavailability must lie in (0, 1], got {bioavailability}")
    rng = np.random.default_rng(seed)
    best_dose = float(doses_arr[np.argmin(np.abs(doses_arr - ec50_auc))])

    from pharmaml.ml.fitting import curve_fit
    from pharmaml.ml.stats import mann_whitney_test

    def _sigmoid(
        x: object,
        e0_: float,
        emax_: float,
        ec50_: object,
        hill_: float,
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        # the LM probe can wander to a negative EC50 or Hill -- keep the
        # fractional powers real instead of leaking NaN warnings upward
        ec50_arr = np.abs(np.asarray(ec50_, dtype=float))
        hill_abs = abs(hill_)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denominator = ec50_arr**hill_abs + arr**hill_abs
            result = e0_ + emax_ * np.where(
                np.isfinite(denominator) & (denominator > 0),
                arr**hill_abs / denominator,
                0.0,
            )
        return np.asarray(result, dtype=float)

    significant = 0
    false_positive = 0
    ec50_estimates: list[float] = []

    for replicate in range(replicates):
        # one shared subject pool per replicate: the AUCs and the effects
        cl = cl_l_hr * np.exp(rng.normal(0.0, iiv_cl, n_per_arm))
        ec50_subject = ec50_auc * np.exp(rng.normal(0.0, iiv_ec50, n_per_arm))
        arm_effects: dict[float, np.ndarray] = {}
        for dose in doses_arr:
            auc = dose * bioavailability / cl
            effect = _sigmoid(auc, e0, emax, ec50_subject, hill)
            arm_effects[float(dose)] = effect + rng.normal(0.0, residual_sd, n_per_arm)
        placebo = e0 + rng.normal(0.0, residual_sd, n_per_arm)

        # the analysis: rank-sum of the best dose against placebo
        result = mann_whitney_test(arm_effects[best_dose], placebo)
        if float(getattr(result, "p_value", 1.0)) < alpha:
            significant += 1

        # the honesty check: placebo against a fresh placebo
        placebo_b = e0 + rng.normal(0.0, residual_sd, n_per_arm)
        null_result = mann_whitney_test(placebo, placebo_b)
        if float(getattr(null_result, "p_value", 1.0)) < alpha:
            false_positive += 1

        # the Emax re-fit on the arm means (dose -> mean effect)
        means = np.array([arm_effects[float(d)].mean() for d in doses_arr])
        try:
            fitted, _ = curve_fit(
                lambda x, e0_, emax_, ec50_, hill_: _sigmoid(np.asarray(x, dtype=float),
                                                              e0_, emax_, ec50_, hill_),
                doses_arr,
                means,
                p0=[e0 - 5.0, emax - 5.0, ec50_auc, hill],
            )
            ec50_estimates.append(float(fitted[2]))
        except Exception as exc:  # noqa: BLE001 - a degenerate replicate drops out
            logger.debug("trial_power: degenerate replicate dropped (%s)", exc)
            continue

    n = replicates
    power = significant / n
    half_width = 1.96 * math.sqrt(power * (1.0 - power) / n)
    if ec50_estimates:
        estimates = np.asarray(ec50_estimates)
        bias_pct = float(np.mean(100.0 * (estimates / ec50_auc - 1.0)))
        rmse_pct = float(np.sqrt(np.mean((100.0 * (estimates / ec50_auc - 1.0)) ** 2)))
        coverage = float(np.mean(np.abs(estimates / ec50_auc - 1.0) <= 0.30))
    else:
        bias_pct = rmse_pct = float("nan")
        coverage = 0.0
    design = (
        f"{len(doses_arr)} arms x {n_per_arm}/arm "
        f"(doses {', '.join(f'{d:g}' for d in doses_arr)})"
    )
    return TrialPowerReport(
        power=power,
        power_ci=(max(0.0, power - half_width), min(1.0, power + half_width)),
        type_one_error=false_positive / n,
        ec50_bias_pct=bias_pct,
        ec50_rmse_pct=rmse_pct,
        ec50_coverage=coverage,
        n_replicates=n,
        design=design,
    )

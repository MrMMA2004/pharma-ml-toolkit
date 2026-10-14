"""Pharmacokinetic calculations (non-compartmental + 1-compartment).

All functions are pure math over numpy arrays or scalars and are fully
unit-tested.  Conventions used throughout:

* ``dose``  -- amount entering the systemic circulation *after*
  bioavailability (multiply by ``F`` yourself when passing an
  administered dose);
* ``times`` -- sample times in hours;
* ``conc``  -- concentrations in the same unit as ``c0``/``Css``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_concentration_series, validate_positive

__all__ = [
    "half_life",
    "elimination_rate_constant",
    "iv_bolus_concentration",
    "iv_infusion_steady_state",
    "oral_tmax",
    "oral_cmax",
    "auc_trapezoid",
    "aucm_trapezoid",
    "mrt",
    "nca_summary",
    "validate_concentration_series",
    "clearance",
    "volume_of_distribution",
    "vd_from_dose_auc",
    "bioavailability",
    "accumulation_ratio",
    "creatinine_clearance",
    "PKSummary",
]


def half_life(ke: float) -> float:
    """Elimination half-life ``t½ = ln(2)/ke`` (hours)."""
    ke = validate_positive(ke, "ke")
    return math.log(2) / ke


def elimination_rate_constant(t_half: float) -> float:
    """Elimination rate constant ``ke = ln(2)/t½`` (1/hours)."""
    t_half = validate_positive(t_half, "t_half")
    return math.log(2) / t_half


def iv_bolus_concentration(c0: float, ke: float, t: float | np.ndarray) -> float | np.ndarray:
    """Concentration after an IV bolus: ``C(t) = C0 · e^(−ke·t)``.

    One-compartment model, first-order elimination.
    """
    c0 = validate_positive(c0, "c0")
    ke = validate_positive(ke, "ke")
    if np.any(np.asarray(t) < 0):
        raise DataValidationError("time t must be non-negative")
    return c0 * np.exp(-ke * np.asarray(t, dtype=float)) if isinstance(
        t, np.ndarray
    ) else c0 * math.exp(-ke * t)


def iv_infusion_steady_state(rate: float, clearance_: float) -> float:
    """Steady-state concentration for a constant IV infusion ``Css = R/CL``."""
    rate = validate_positive(rate, "rate")
    clearance_ = validate_positive(clearance_, "clearance")
    return rate / clearance_


def oral_tmax(ka: float, ke: float) -> float:
    """Time of peak concentration for 1-compartment oral dosing.

    ``Tmax = ln(ka/ke) / (ka − ke)`` for first-order absorption.

    Raises
    ------
    DataValidationError
        If ``ka <= ke`` (flip-flop kinetics make the formula invalid)
        or either rate is not positive.
    """
    ka = validate_positive(ka, "ka")
    ke = validate_positive(ke, "ke")
    if ka <= ke:
        raise DataValidationError(
            f"ka must exceed ke for the Tmax formula (ka={ka}, ke={ke}); "
            "flip-flop absorption requires numerical optimisation"
        )
    return math.log(ka / ke) / (ka - ke)


def oral_cmax(dose: float, ka: float, ke: float, vd: float, f: float = 1.0) -> float:
    """Peak concentration after a single oral dose (1-compartment).

    ``Cmax = (F·D/Vd) · ka/(ka−ke) · (e^(−ke·Tmax) − e^(−ka·Tmax))``
    """
    dose = validate_positive(dose, "dose")
    ka = validate_positive(ka, "ka")
    ke = validate_positive(ke, "ke")
    vd = validate_positive(vd, "vd")
    f = validate_positive(f, "F")
    t_max = oral_tmax(ka, ke)
    bracket = math.exp(-ke * t_max) - math.exp(-ka * t_max)
    return (f * dose / vd) * (ka / (ka - ke)) * bracket


def auc_trapezoid(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    method: str = "linear",
) -> float:
    """Area under the concentration-time curve (trapezoidal rule).

    Parameters
    ----------
    times, conc:
        Paired sample series (validated by
        :func:`~pharmaml.utils.validation.validate_concentration_series`).
    method:
        ``'linear'`` for the standard trapezoid (safe with zero
        concentrations) or ``'log'`` for the logarithmic trapezoid
        (more accurate during exponential decay; requires positive
        concentrations).

    Returns
    -------
    float
        AUC in ``concentration·time`` units.

    References
    ----------
    .. [1] Purves, R. D. "Optimum numerical integration methods for
           estimation of area-under-the-curve (AUC) and
           area-under-the-moment-curve (AUMC)." J. Pharmacokinet.
           Biopharm. 20 (1992): 211-226.
    """
    t, c = validate_concentration_series(
        times, conc, require_positive_conc=(method == "log")
    )
    dt = np.diff(t)
    if method == "linear":
        areas = 0.5 * (c[:-1] + c[1:]) * dt
    elif method == "log":
        c_mid = np.sqrt(c[:-1] * c[1:])
        areas = c_mid * dt
    else:
        raise DataValidationError(
            f"method must be 'linear' or 'log', got {method!r}"
        )
    return float(areas.sum())


def aucm_trapezoid(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
) -> float:
    """Area under the first-moment curve (AUMC = ∫ t·C dt, linear trapezoid)."""
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)
    tc = t * c
    areas = 0.5 * (tc[:-1] + tc[1:]) * np.diff(t)
    return float(areas.sum())


def mrt(times: np.ndarray | list[float], conc: np.ndarray | list[float]) -> float:
    """Mean residence time ``MRT = AUMC/AUC`` (non-compartmental)."""
    auc = auc_trapezoid(times, conc, method="linear")
    if auc <= 0:
        raise DataValidationError("AUC is zero/undefined; cannot compute MRT")
    return aucm_trapezoid(times, conc) / auc


def clearance(dose: float, auc: float, f: float = 1.0) -> float:
    """Systemic clearance ``CL = F·Dose/AUC``.

    ``dose`` is the *administered* dose and ``f`` the absolute
    bioavailability fraction (default 1 for IV).
    """
    dose = validate_positive(dose, "dose")
    auc = validate_positive(auc, "auc")
    f = validate_positive(f, "F", allow_zero=False)
    return f * dose / auc


def volume_of_distribution(clearance_: float, ke: float) -> float:
    """Apparent volume of distribution ``Vd = CL/ke``."""
    clearance_ = validate_positive(clearance_, "clearance")
    ke = validate_positive(ke, "ke")
    return clearance_ / ke


def vd_from_dose_auc(dose: float, auc: float, mrt: float | None = None) -> float:
    """Dose/AUC ratio — and, with ``mrt``, the steady-state volume.

    Without ``mrt`` this returns the plain ``Dose/AUC`` ratio, which is
    the *clearance* CL (for an IV dose, F = 1) — **not a volume**; the
    name is kept for call-site honesty checks and legacy parity.  The
    honest volume routes are ``Vd = CL/ke``
    (:func:`volume_of_distribution`) or the statistical-moment
    ``Vss = Dose·MRT/AUC`` obtained by passing ``mrt``.
    """
    dose = validate_positive(dose, "dose")
    auc = validate_positive(auc, "auc")
    if mrt is None:
        return dose / auc
    return dose * validate_positive(mrt, "mrt") / auc


def bioavailability(
    dose_ev: float, auc_ev: float, dose_iv: float, auc_iv: float
) -> float:
    """Absolute bioavailability ``F = (AUC_ev·Dose_iv)/(AUC_iv·Dose_ev)``."""
    dose_ev = validate_positive(dose_ev, "dose_ev")
    auc_ev = validate_positive(auc_ev, "auc_ev")
    dose_iv = validate_positive(dose_iv, "dose_iv")
    auc_iv = validate_positive(auc_iv, "auc_iv")
    return (auc_ev * dose_iv) / (auc_iv * dose_ev)


def accumulation_ratio(tau: float, t_half: float) -> float:
    """Accumulation ratio for regular intermittent dosing.

    ``R = 1/(1 − e^(−ke·τ))`` with ``ke = ln2/t½`` -- the classic
    accumulation factor used to convert single-dose to steady-state
    peak/trough expectations.
    """
    tau = validate_positive(tau, "tau")
    t_half = validate_positive(t_half, "t_half")
    ke = math.log(2) / t_half
    return 1.0 / (1.0 - math.exp(-ke * tau))


def creatinine_clearance(
    age_years: float,
    weight_kg: float,
    serum_cr_mg_dl: float,
    sex: str,
) -> float:
    """Estimate creatinine clearance (mL/min) with the Cockcroft-Gault
    equation.

    ``CrCl = (140 − age) · weight / (72 · SCr)`` multiplied by 0.85 for
    female patients.

    Parameters
    ----------
    age_years:
        Patient age in years (1–120).
    weight_kg:
        Body weight in kilograms.
    serum_cr_mg_dl:
        Serum creatinine in mg/dL.
    sex:
        ``'male'``/``'m'`` or ``'female'``/``'f'``.

    References
    ----------
    .. [1] Cockcroft, D.W. and Gault, M.H., Nephron 16 (1976): 31-41.
    """
    if not 1 <= age_years <= 120:
        raise DataValidationError(f"age must be within [1, 120] years, got {age_years}")
    validate_positive(weight_kg, "weight_kg")
    validate_positive(serum_cr_mg_dl, "serum_creatinine")
    key = sex.strip().lower()
    if key in {"male", "m"}:
        factor = 1.0
    elif key in {"female", "f"}:
        factor = 0.85
    else:
        raise DataValidationError(f"sex must be 'male' or 'female', got {sex!r}")
    base = (140.0 - age_years) * weight_kg / (72.0 * serum_cr_mg_dl)
    return base * factor


@dataclass(frozen=True, slots=True)
class PKSummary:
    """Non-compartmental PK summary for one concentration-time profile."""

    auc: float
    aucm: float
    mrt: float
    c_max: float
    t_max: float
    half_life_terminal: float

    def as_dict(self) -> dict[str, float]:
        """Return the summary as a plain dictionary (report friendly)."""
        return {
            "AUC": self.auc,
            "AUMC": self.aucm,
            "MRT": self.mrt,
            "Cmax": self.c_max,
            "Tmax": self.t_max,
            "t_half_terminal": self.half_life_terminal,
        }


def nca_summary(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
) -> PKSummary:
    """Full non-compartmental analysis of one profile.

    The terminal half-life is estimated from the last two (or three)
    positive points by log-linear regression, mirroring the behaviour
    of commercial NCA software.
    """
    t, c = validate_concentration_series(times, conc, require_positive_conc=False)

    positive = c > 0
    c_max = float(np.max(c))
    t_max = float(t[int(np.argmax(c))])

    # MRT is undefined when the profile integrates to zero (blank curve).
    try:
        mrt_value = mrt(t, c)
    except DataValidationError:
        mrt_value = float("nan")

    # Terminal slope from the tail (>=2 positive points), log-linear.
    if positive.sum() >= 2:
        t_tail = t[positive][-2:]
        c_tail = c[positive][-2:]
        slope = (np.log(c_tail[-1]) - np.log(c_tail[0])) / (t_tail[-1] - t_tail[0])
        lambd = -slope if slope < 0 else float("nan")
    else:
        lambd = float("nan")

    return PKSummary(
        auc=auc_trapezoid(t, c),
        aucm=aucm_trapezoid(t, c),
        mrt=mrt_value,
        c_max=c_max,
        t_max=t_max,
        half_life_terminal=(math.log(2) / lambd) if lambd and math.isfinite(lambd) else float("nan"),
    )

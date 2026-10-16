"""Dosage-regimen design and adjustment.

Covers loading/maintenance dose derivation from PK parameters, body
-size scaling (BSA, mg/kg), renal-function adjustment and therapeutic
drug monitoring (TDM) trough correction.

Every function is pure and defensive: parameters are range-checked
with actionable :class:`~pharmaml.exceptions.DataValidationError`
messages, because dosage math is exactly where silent errors become
patient-safety incidents.
"""

from __future__ import annotations

import math

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    "bsa_mosteller",
    "clarks_rule",
    "dose_by_bsa",
    "dose_by_weight",
    "infusion_rate",
    "loading_dose",
    "maintenance_dose",
    "renal_dose_adjustment",
    "tdm_trough_adjustment",
    "youngs_rule",
]


def maintenance_dose(
    clearance_: float,
    target_css: float,
    tau: float,
    f: float = 1.0,
) -> float:
    """Maintenance dose per dosing interval.

    Derived from steady-state averaging:
    ``MD = CL · Css,avg · τ / F``.

    Parameters
    ----------
    clearance_:
        Drug clearance (volume/time, e.g. L/h).
    target_css:
        Desired *average* steady-state concentration.
    tau:
        Dosing interval (time units consistent with clearance).
    f:
        Oral bioavailability fraction (1 for IV).

    Returns
    -------
    float
        Dose (mass units consistent with concentration·volume).
    """
    clearance_ = validate_positive(clearance_, "clearance")
    target_css = validate_positive(target_css, "target_css")
    tau = validate_positive(tau, "tau")
    f = validate_positive(f, "F")
    if f > 1.0 + 1e-9:
        raise DataValidationError(f"bioavailability F cannot exceed 1, got {f}")
    return clearance_ * target_css * tau / f


def loading_dose(
    vd: float,
    target_concentration: float,
    f: float = 1.0,
) -> float:
    """Loading dose to reach target concentration quickly.

    ``LD = Vd · C_target / F``.
    """
    vd = validate_positive(vd, "vd")
    target_concentration = validate_positive(target_concentration, "target_concentration")
    f = validate_positive(f, "F")
    if f > 1.0 + 1e-9:
        raise DataValidationError(f"bioavailability F cannot exceed 1, got {f}")
    return vd * target_concentration / f


def infusion_rate(clearance_: float, target_css: float) -> float:
    """Constant IV infusion rate achieving a steady state: ``R = CL·Css``."""
    clearance_ = validate_positive(clearance_, "clearance")
    target_css = validate_positive(target_css, "target_css")
    return clearance_ * target_css


def dose_by_weight(
    dose_per_kg: float,
    weight_kg: float,
    max_dose: float | None = None,
) -> float:
    """Weight-scaled dose (mg/kg regime) with an optional ceiling.

    Raises
    ------
    DataValidationError
        If the computed dose exceeds ``max_dose`` -- by design the
        caller must handle capping explicitly rather than silently
        clipping an overdose.
    """
    dose_per_kg = validate_positive(dose_per_kg, "dose_per_kg")
    weight_kg = validate_positive(weight_kg, "weight_kg")
    dose = dose_per_kg * weight_kg
    if max_dose is not None and dose > max_dose:
        raise DataValidationError(
            f"computed dose {dose:g} exceeds the maximum allowed {max_dose:g}; "
            "cap the weight or dose_per_kg explicitly"
        )
    return dose


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    """Body surface area (m²) by the Mosteller formula.

    ``BSA = sqrt(height_cm · weight_kg / 3600)``.

    References
    ----------
    .. [1] Mosteller, R. D., N Engl J Med 317 (1987): 1098.
    """
    height_cm = validate_positive(height_cm, "height_cm")
    weight_kg = validate_positive(weight_kg, "weight_kg")
    if not 30 <= height_cm <= 250:
        raise DataValidationError(f"height {height_cm} cm is outside the plausible range")
    if not 0.5 <= weight_kg <= 400:
        raise DataValidationError(f"weight {weight_kg} kg is outside the plausible range")
    return math.sqrt(height_cm * weight_kg / 3600.0)


def dose_by_bsa(
    dose_per_m2: float,
    height_cm: float,
    weight_kg: float,
    max_dose: float | None = None,
) -> float:
    """BSA-scaled dose (Mosteller BSA) with an optional ceiling."""
    dose_per_m2 = validate_positive(dose_per_m2, "dose_per_m2")
    bsa = bsa_mosteller(height_cm, weight_kg)
    dose = dose_per_m2 * bsa
    if max_dose is not None and dose > max_dose:
        raise DataValidationError(
            f"computed dose {dose:.2f} exceeds maximum {max_dose:g} (BSA {bsa:.2f} m²)"
        )
    return dose


def youngs_rule(child_age_years: float, adult_dose: float) -> float:
    """Young's rule for paediatric dose scaling (age 1–12).

    ``dose = adult_dose · age / (age + 12)``.

    .. warning::
       Historical heuristic retained for teaching/comparison purposes;
       modern paediatric dosing prefers mg/kg or BSA scaling.
    """
    if not 1 <= child_age_years <= 12:
        raise DataValidationError(
            f"Young's rule applies to ages 1–12 years, got {child_age_years}"
        )
    validate_positive(adult_dose, "adult_dose")
    return adult_dose * child_age_years / (child_age_years + 12.0)


def clarks_rule(child_weight_lb: float, adult_dose: float) -> float:
    """Clark's rule paediatric scaling by body weight in pounds.

    ``dose = adult_dose · weight_lb / 150``.

    .. warning::
       Historical heuristic; prefer :func:`dose_by_weight`.
    """
    validate_positive(child_weight_lb, "child_weight_lb")
    validate_positive(adult_dose, "adult_dose")
    return adult_dose * child_weight_lb / 150.0


def renal_dose_adjustment(
    normal_dose: float,
    current_creatinine_clearance: float,
    normal_creatinine_clearance: float = 100.0,
    floor_fraction: float = 0.25,
) -> float:
    """Linearly scale a dose with renal function (creatinine clearance).

    ``dose = normal_dose · max(CrCl_current, floor·CrCl_normal)/CrCl_normal``.

    The scaling is bidirectional: impaired function reduces the dose,
    and clearance *above* the normal value increases it proportionally
    (renally-cleared drugs in high-clearance patients).  The
    ``floor_fraction`` guards against absurdly tiny doses for anuric
    patients -- below the floor, clinicians switch modality rather
    than dose.
    """
    validate_positive(normal_dose, "normal_dose")
    validate_positive(
        current_creatinine_clearance, "current_creatinine_clearance", allow_zero=True
    )
    validate_positive(normal_creatinine_clearance, "normal_creatinine_clearance")
    if not 0.0 < floor_fraction <= 1.0:
        raise DataValidationError(f"floor_fraction must be in (0, 1], got {floor_fraction}")
    effective = max(
        current_creatinine_clearance, floor_fraction * normal_creatinine_clearance
    )
    return normal_dose * effective / normal_creatinine_clearance


def tdm_trough_adjustment(
    current_dose: float,
    measured_trough: float,
    target_trough: float,
    min_dose: float | None = None,
) -> float:
    """Proportionally adjust a dose from a measured trough level.

    Linear kinetics assumption: ``new_dose = current_dose · target/measured``.
    Typical use: vancomycin/aminoglycoside trough-guided adjustment.

    Parameters
    ----------
    current_dose:
        Dose the patient was receiving when the trough was drawn.
    measured_trough:
        Observed trough concentration.
    target_trough:
        Desired trough concentration.
    min_dose:
        Optional lower bound for the adjusted dose.
    """
    validate_positive(current_dose, "current_dose")
    measured_trough = validate_positive(measured_trough, "measured_trough")
    target_trough = validate_positive(target_trough, "target_trough")
    ratio = target_trough / measured_trough
    if ratio > 5.0 or ratio < 1.0 / 5.0:
        raise DataValidationError(
            f"target/measured trough ratio {ratio:.2g} lies outside the "
            "factor-5 band accepted for linear kinetics; a change this "
            "large suggests non-linear kinetics, non-steady-state "
            "sampling, or a data-entry error -- refusing to extrapolate "
            "linearly"
        )
    adjusted = current_dose * ratio
    if min_dose is not None and adjusted < min_dose:
        adjusted = min_dose
    return adjusted

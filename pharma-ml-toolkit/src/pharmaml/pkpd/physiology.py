"""PK-Sim's static physiology layer: species, ontogeny, Kp methods, IVIVE.

The round-176 absorption of the Open-Systems-Pharmacology **PK-Sim**
repository.  PK-Sim is a C#/.NET platform, and almost everything that
makes it *PK-Sim* — the UI, the spatial-structure builder, the OSPSuite
formula engine, the dynamic individual-creation optimiser — stays
behind the honest boundary.  What PK-Sim uniquely *ships*, though, is
one queryable SQLite database (``src/Db/PKSimDB.sqlite``, 30 MB) that
carries the whole quantitative physiology layer: ten species' organ
volumes and flows, the ICRP 2002 age curves, twenty enzyme/transporter
ontogenies, the per-organ tissue composition, the five published
partition-coefficient methods as formula strings, and the IVIVE
conversion machinery.  This module is that layer, numpy-native:

* :func:`pma_years`, :func:`create_individual` — the gestational-age
  convention and the ICRP 2002 European reference curves (weight,
  height, hematocrit, liver volume/flow at any age);
* :func:`gfr_maturation_fraction`, :func:`gfr_reference`,
  :func:`bsa_mosteller`, :func:`bsa_dubois` — the renal-maturation Hill
  function (TM50 44.4 weeks, Hill 15, premature floor 0.258), the
  adult GFRspec ``GFRmat / V_kid_std = 0.11704/0.44 = 0.266 L/min/kg``
  that PK-Sim's own integration tests pin, and the two BSA formulas;
* :func:`ontogeny_factor`, :func:`ontogeny_geo_sd`,
  :func:`fu_ontogeny_adjusted` — the 508-point ontogeny tables
  (CYP3A4/3A7 switch-off, UGTs, ABCB1, the plasma proteins AGP/ALB)
  with PK-Sim's linear-in-PMA interpolation and the AGP-driven
  paediatric unbound-fraction correction;
* :func:`kp_pk_sim_standard`, :func:`kp_rodgers_rowland`,
  :func:`kp_poulin_theil`, :func:`kp_schmitt`,
  :func:`kp_berezhkovskiy` — the five ``Cellular partition coefficient
  method`` formula sets transcribed from
  ``tab_calculation_method_rates`` (Willmann 2005, Rodgers & Rowland
  2005-07, Poulin & Theil 2002, Schmitt 2008, Berezhkovskiy 2004) on
  the real per-organ tissue composition — the calculation the
  :mod:`pharmaml.pkpd.pbpk` round explicitly left in PK-Sim;
* :func:`blood_plasma_ratio` — the blood:plasma bridge
  ``B2P = 1 + HCT·(K_rbc − 1)``;
* :func:`fu_mic`, :func:`clint_to_clspec` — the in-vitro→in-vivo
  clearance conversions (microsomes and hepatocytes, half-life and
  residual-fraction bases) with PK-Sim's numeric acceptance tests as
  doctests (``0.009259`` L/µmol/min and ``23.497`` min⁻¹).

Honest boundaries: the dynamic individual-creation algorithm (the
Nelder-Mead volume optimiser that matches a *given* weight/height), the
observer formulas that need the full container topology (peripheral
venous plasma is a flow-weighted sum over six tissues), the Schmitt
K1..K8 charge-state regression (it lives in the OSPSuite.Core .NET
formula evaluator, not in the database), diprotic/strong-base ionic
terms beyond the monoprotic acid/base pair, and every UI/workflow
layer.  Correlated population sampling stays with
:func:`pharmaml.pkpd.population.simulate_population`.

Examples
--------
The ICRP 2002 European reference male at 30 years — the individual the
PK-Sim integration tests pin:

>>> from pharmaml.pkpd.physiology import create_individual
>>> ind = create_individual(age_years=30.0, gender="MALE")
>>> round(ind["Organism|Weight"], 1), round(ind["Organism|Height"] * 10, 1)
(73.0, 176.0)
>>> round(ind["Organism|Hematocrit"], 2)
0.47
>>> round(ind["Liver|Volume"], 4)
2.3578

GFR maturation — a term neonate is at ~39 % of adult capacity, an
infant past six months is essentially mature:

>>> from pharmaml.pkpd.physiology import gfr_maturation_fraction, gfr_reference
>>> round(gfr_maturation_fraction(pma_years=40.0 * 7 / 365.25), 3)
0.386
>>> round(gfr_maturation_fraction(pma_years=2.0), 3)
1.0
>>> round(gfr_reference(age_years=30.0), 3)          # PK-Sim's own test value
0.266

Body-surface area (Mosteller) for the 73 kg / 176 cm reference —
PK-Sim's integration test says 1.89 m²:

>>> from pharmaml.pkpd.physiology import bsa_mosteller
>>> round(bsa_mosteller(weight_kg=73.0, height_cm=176.0), 2)
1.89

CYP3A4 hepatic ontogeny — a 0.8-year-old infant expresses ~14 % of
the adult level; the neonatal enzyme CYP3A7 is switched off by ~2
years:

>>> from pharmaml.pkpd.physiology import ontogeny_factor
>>> round(ontogeny_factor("CYP3A4", pma_years=0.8), 3)
0.136
>>> round(ontogeny_factor("CYP3A7", pma_years=0.8), 2)
0.97
>>> round(ontogeny_factor("CYP3A7", pma_years=2.0), 3)
0.025

Partition coefficients for a lipophilic neutral drug (LogP 2, fu 1) in
human liver, four published methods on identical tissue data:

>>> from pharmaml.pkpd.physiology import (
...     kp_berezhkovskiy, kp_pk_sim_standard, kp_poulin_theil,
...     kp_rodgers_rowland)
>>> round(kp_poulin_theil(log_p=2.0, fu=1.0, tissue="Liver"), 3)
3.669
>>> round(kp_rodgers_rowland(log_p=2.0, fu=1.0, tissue="Liver"), 3)
2.723
>>> round(kp_berezhkovskiy(log_p=2.0, fu=1.0, tissue="Liver"), 3)
3.669

IVIVE — a 1 µL/min/mg microsomal CLint normalised to PK-Sim's default
CYP content (108 pmol/mg) and the residual-fraction conversion PK-Sim's
tests pin:

>>> from pharmaml.pkpd.physiology import clint_to_clspec, fu_mic
>>> round(1.0 / 108.0, 6)                                # CLspec per enzyme
0.009259
>>> round(fu_mic(log_ma=0.0, microsomal_protein_fraction=0.1), 3)
1.0
>>> import math
>>> round(clint_to_clspec(
...     clint_per_protein=math.log(1.0 / 0.8) / 20.0,   # residual basis
...     liver_to_protein=400.0, fu_mic_value=0.28476, f_cell=0.667), 3)
23.497
"""

from __future__ import annotations

import math

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.pkpd._physiology_data import (
    GFR_CONSTANTS,
    HUMAN_AGE_CURVES,
    MOLECULE_DEFAULTS,
    ONTOGENY_TABLE,
    PLASMA_COMPOSITION,
    SPECIES_PHYSIOLOGY,
    TISSUE_COMPOSITION,
    TRANSPORTER_TYPES,
)
from pharmaml.utils.validation import validate_positive

__all__ = [
    "ADULT_GFRSPEC",
    "GFR_HILL",
    "GFR_TM50_WEEKS",
    "MOLECULE_DEFAULTS",
    "ONTOGENY_TABLE",
    "SPECIES_PHYSIOLOGY",
    "TISSUE_COMPOSITION",
    "TRANSPORTER_TYPES",
    "blood_plasma_ratio",
    "bsa_dubois",
    "bsa_mosteller",
    "clint_to_clspec",
    "create_individual",
    "fu_mic",
    "fu_ontogeny_adjusted",
    "gfr_maturation_fraction",
    "gfr_reference",
    "kp_berezhkovskiy",
    "kp_pk_sim_standard",
    "kp_poulin_theil",
    "kp_rodgers_rowland",
    "kp_schmitt",
    "ontogeny_factor",
    "ontogeny_geo_sd",
    "pma_years",
]

# The renal-maturation constants of PK-Sim's GFR_Calculation method
# (tab_calculation_method_rates).  fGFRpremat (0.258), GFRmat (0.11704
# L/min) and the 0.44 L standard-kidney volume are carried verbatim by
# the database; TM50 and Hill are the method's published constants.
GFR_TM50_WEEKS = 44.4
GFR_HILL = 15.0

# Adult reference GFRspec = GFRmat / V_kid_std — exactly the value
# PK-Sim's IndividualSpecs integration test asserts (0.266 L/min/kg,
# ±1e-2) for the standard 30-year-old male.
ADULT_GFRSPEC = (
    GFR_CONSTANTS["GFRmat"] / GFR_CONSTANTS["Volume (standard kidney)"]
)  # 0.266 L/min/kg

_TISSUE_ALIASES = {
    "adipose": "Fat",
    "bone": "Bone",
    "brain": "Brain",
    "fat": "Fat",
    "gonads": "Gonads",
    "heart": "Heart",
    "kidney": "Kidney",
    "liver": "Liver",
    "lung": "Lung",
    "muscle": "Muscle",
    "pancreas": "Pancreas",
    "skin": "Skin",
    "spleen": "Spleen",
    "stomach": "Stomach",
}


def _resolve_tissue(tissue: str) -> dict[str, float]:
    """Map a tissue name (any case, with aliases) to its composition row."""
    key = _TISSUE_ALIASES.get(tissue.strip().lower(), tissue.strip())
    try:
        return TISSUE_COMPOSITION[key]
    except KeyError:
        known = ", ".join(sorted(TISSUE_COMPOSITION))
        raise DataValidationError(
            f"unknown tissue {tissue!r} — the PK-Sim composition table carries: {known}"
        ) from None


def pma_years(age_years: float, gestational_age_weeks: float = 40.0) -> float:
    """Post-menstrual age: ``PMA = age + GA·7/365.25`` (PK-Sim's convention).

    PK-Sim's ``Individual_AgeDependent`` formula — the age axis every
    ontogeny and maturation curve in the database is keyed on, with the
    standard 40-week term default
    (``CoreConstants.NOT_PRETERM_GESTATIONAL_AGE_IN_WEEKS``).
    """
    validate_positive(age_years, "age_years", allow_zero=True)
    validate_positive(gestational_age_weeks, "gestational_age_weeks", allow_zero=True)
    return age_years + gestational_age_weeks * 7.0 / 365.25


def _interp_curve(points: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation on PK-Sim's curve points, clamped at the edges.

    The interpolation registered for ontogeny tables in PK-Sim's Core
    (``Register[IInterpolation, LinearInterpolation]``).
    """
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    xs = np.asarray([p[0] for p in points])
    idx = int(np.searchsorted(xs, x)) - 1
    (x0, y0), (x1, y1) = points[idx], points[idx + 1]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def create_individual(
    age_years: float = 30.0, gender: str = "MALE"
) -> dict[str, float]:
    """The ICRP 2002 European reference individual at ``age_years``.

    Curve means from ``tab_container_parameter_curves``
    (European_ICRP_2002), linearly interpolated — exactly the numbers
    PK-Sim's individual builder starts from before its weight/height
    optimiser runs (the optimiser itself is the honest boundary).  The
    liver's blood flow is the *direct arterial* share
    (``Q = Qspec · tissue weight``); the portal contribution arrives
    from the GI organs in the full topology and is not part of this
    column.

    The returned keys follow the OSP-Suite pipe-path convention so the
    values slot straight into
    :func:`pharmaml.pkpd.sensitivity.create_individual_characteristics`
    table workflows.
    """
    gender = gender.upper()
    if gender not in ("MALE", "FEMALE"):
        raise DataValidationError(
            f"gender must be 'MALE' or 'FEMALE', got {gender!r}"
        )
    validate_positive(age_years, "age_years", allow_zero=True)

    def curve(name: str) -> float:
        points = HUMAN_AGE_CURVES[name][gender]
        return _interp_curve(points, age_years)

    liver_volume = curve("liver_volume")
    liver_q = curve("liver_qspec") * liver_volume  # density 1 kg/L
    return {
        "Organism|Weight": curve("mean_bw"),
        "Organism|Height": curve("mean_height_dm"),  # decimetres
        "Organism|Hematocrit": curve("hematocrit"),
        "Liver|Volume": liver_volume,
        "Liver|Specific blood flow rate": curve("liver_qspec"),
        "Liver|Blood flow": liver_q * 60.0,  # L/hour
    }


def gfr_maturation_fraction(pma_years: float) -> float:
    """Renal maturation fraction of adult GFR — the Hill function.

    ``GFRmat_spec = ((PMA_weeks^Hill / (TM50^Hill + PMA_weeks^Hill))·
    (1 − fGFRpremat) + fGFRpremat)`` with the PK-Sim constants TM50 =
    44.4 weeks, Hill = 15, premature floor = 0.258: a term neonate sits
    at ~39 % of adult capacity, and maturation is essentially complete
    by ~6 months of postnatal age.
    """
    if pma_years < 0:
        raise DataValidationError(
            f"pma_years must be >= 0, got {pma_years}"
        )
    pma_weeks = pma_years * 365.25 / 7.0
    premat = GFR_CONSTANTS["fGFRpremat"]
    ratio = pma_weeks / GFR_TM50_WEEKS
    frac = ratio**GFR_HILL / (1.0 + ratio**GFR_HILL)
    return frac * (1.0 - premat) + premat


def gfr_reference(age_years: float, gestational_age_weeks: float = 40.0) -> float:
    """Reference GFRspec (L/min/kg) at a chronological age.

    ``ADULT_GFRSPEC × maturation(PMA)`` — the adult value
    ``GFRmat/V_kid_std = 0.11704/0.44 = 0.266`` PK-Sim's
    ``IndividualSpecs`` integration test pins, scaled by the Hill
    maturation for paediatric ages (renal *aging* — the decline past
    ~60 years PK-Sim applies through its renal-aging scaling factor —
    is a curve in the database we do not carry; pass adult ages for
    the reference value).
    """
    return ADULT_GFRSPEC * gfr_maturation_fraction(
        pma_years(age_years, gestational_age_weeks)
    )


def bsa_mosteller(weight_kg: float, height_cm: float) -> float:
    """Mosteller BSA (m²): ``sqrt(W·H / 3600)`` — PK-Sim's default."""
    validate_positive(weight_kg, "weight_kg")
    validate_positive(height_cm, "height_cm")
    return math.sqrt(weight_kg * height_cm / 3600.0)


def bsa_dubois(weight_kg: float, height_cm: float) -> float:
    """Du Bois & Du Bois BSA (m²): ``0.007184·W^0.425·H^0.725``."""
    validate_positive(weight_kg, "weight_kg")
    validate_positive(height_cm, "height_cm")
    return 0.007184 * weight_kg**0.425 * height_cm**0.725


def _ontogeny_series(molecule: str, group: str) -> list[tuple[float, float, float]]:
    if molecule not in ONTOGENY_TABLE:
        known = ", ".join(sorted(ONTOGENY_TABLE))
        raise DataValidationError(
            f"unknown ontogeny molecule {molecule!r} — the table carries: {known}"
        )
    groups = ONTOGENY_TABLE[molecule]
    if group == "liver":
        # PK-Sim's container->group fallback: the GI-folded liver series
        # when the molecule has one, the plain liver series otherwise.
        group = (
            "ONTOGENY_LIVER_NO_GI"
            if "ONTOGENY_LIVER_NO_GI" in groups
            else "ONTOGENY_LIVER_ALL"
        )
    elif group == "duodenum":
        group = "ONTOGENY_DUODENUM"
    elif group == "plasma":
        group = "ONTOGENY_PLASMA"
    elif group not in groups:
        known = ", ".join(sorted(groups))
        raise DataValidationError(
            f"{molecule!r} has no ontogeny group {group!r} — it carries: {known}"
        )
    return groups[group]


def ontogeny_factor(
    molecule: str, pma_years: float, group: str = "liver"
) -> float:
    """Enzyme/transporter ontogeny factor at a post-menstrual age.

    PK-Sim's semantics exactly: **linear interpolation of the factor
    against PMA** (not log-age), clamped at the table's ends, group
    fallback liver→``LIVER_NO_GI``→``LIVER_ALL``.  The deviation
    companion (:func:`ontogeny_geo_sd`) is the geometric SD of a
    log-normal — a *population* property, not an error bar on the mean.
    """
    series = _ontogeny_series(molecule, group)
    return _interp_curve([(pma, f) for pma, f, _ in series], pma_years)


def ontogeny_geo_sd(
    molecule: str, pma_years: float, group: str = "liver"
) -> float:
    """The geometric SD of the ontogeny factor's log-normal distribution."""
    series = _ontogeny_series(molecule, group)
    return _interp_curve([(pma, d) for pma, _, d in series], pma_years)


def fu_ontogeny_adjusted(
    fu_reference: float,
    binding_partner: str = "AGP",
    pma_years: float = 30.0 * 7 / 365.25 + 30.0,
    plasma_protein_scale_factor: float = 1.0,
) -> float:
    """Age-dependent unbound fraction — PK-Sim's ``DistributionInVitro`` fu.

    ``fu = min(1 / (1 + ontogeny·(1 − fu_ref)/fu_ref)·PPSF, 1)`` — the
    α1-acid-glycoprotein (or albumin) ontogeny scaling that makes
    highly-bound drugs appear *more* unbound in neonates, where AGP is
    ~25 % of adult levels.  Default ``pma_years`` is the adult
    reference (30 y + 40 weeks gestation), where the AGP factor is 1.0
    and the formula collapses to ``fu_reference``.
    """
    if not 0 < fu_reference <= 1:
        raise DataValidationError(
            f"fu_reference must be in (0, 1], got {fu_reference}"
        )
    partner = binding_partner.upper()
    if partner not in ("AGP", "ALB"):
        raise DataValidationError(
            "binding_partner must be 'AGP' or 'ALB', got "
            f"{binding_partner!r}"
        )
    factor = ontogeny_factor(partner, pma_years, group="plasma")
    value = 1.0 / (1.0 + factor * (1.0 - fu_reference) / fu_reference)
    return min(value * plasma_protein_scale_factor, 1.0)


def fu_mic(log_ma: float, microsomal_protein_fraction: float) -> float:
    """Unbound fraction in a microsomal incubation.

    ``fu_mic = min(1 / ((1 − f_prot) + 10^LogMA·f_prot), 1)`` — the
    binding correction PK-Sim applies before scaling microsomal CLint
    (the incubation's protein fraction, typically 0.1 mg/mL-scale).
    Hydrophilic compounds (LogMA ≈ 0) are essentially unbound in the
    incubation; lipophilic ones (LogMA 3 at f = 0.1) are ~1 % unbound.
    """
    if not 0.0 <= microsomal_protein_fraction < 1.0:
        raise DataValidationError(
            "microsomal_protein_fraction must be in [0, 1), got "
            f"{microsomal_protein_fraction}"
        )
    m = microsomal_protein_fraction
    return min(1.0 / ((1.0 - m) + 10.0**log_ma * m), 1.0)


def clint_to_clspec(
    *,
    clint_per_protein: float,
    liver_to_protein: float,
    fu_mic_value: float,
    f_cell: float,
) -> float:
    """Microsomal/hepatocyte CLint → specific hepatic clearance.

    The four PK-Sim conversion variants share one shape —
    ``CLspec = CLint_per_protein · (liver-to-protein scaling) /
    (fu_mic · f_cell)`` — and differ only in how ``clint_per_protein``
    is *derived* upstream:

    * microsomes, half-life basis: ``0.693 / t½``;
    * microsomes, residual basis: ``ln(1/RES) / t0``;
    * hepatocytes, either basis: same numerators over the
      cells-per-gram scaling (PK-Sim ships 139,000 cells/g liver,
      117,000-215,000 across species — :data:`SPECIES_PHYSIOLOGY`).

    ``liver_to_protein`` is the incubation-to-liver mass scaling
    (``MPPGL / incubation protein concentration`` in matched units —
    40/0.1 = 400 in PK-Sim's acceptance test).  The result is in
    ``min⁻¹·(incubation units)`` — pass per-minute CLint for
    min⁻¹.

    The acceptance doctest below reproduces PK-Sim's
    ``InVitroSpecificClearanceSpecs`` integration test: residual
    fraction 0.8 over t0 = 20 min with fu_mic 0.28476, f_cell 0.667
    and the 400× scaling give **23.497 min⁻¹** (PK-Sim asserts
    23.4972 against its unrounded internals); normalising a 1
    µL/min/mg microsomal CLint to the default 108 pmol CYP/mg gives
    the test's other pin, **0.009259 L/µmol/min**.
    """
    validate_positive(clint_per_protein, "clint_per_protein")
    validate_positive(liver_to_protein, "liver_to_protein")
    if not 0 < fu_mic_value <= 1:
        raise DataValidationError(
            f"fu_mic_value must be in (0, 1], got {fu_mic_value}"
        )
    if not 0 < f_cell <= 1:
        raise DataValidationError(f"f_cell must be in (0, 1], got {f_cell}")
    return clint_per_protein * liver_to_protein / (fu_mic_value * f_cell)


def _kprot(log_ma: float) -> float:
    """Protein-binding partition ratio (PK-Sim's ``LinksCommon``)."""
    return (0.81 + 0.11 * 10.0**log_ma) / 24.92 * 5.0


def kp_pk_sim_standard(
    log_ma: float, fu: float, tissue: str = "Liver"
) -> float:
    """Willmann 2005 (the "PK-Sim Standard" method).

    ``K_cell_pls = (f_water + f_lipid·10^LogMA + f_protein·KProt)·fu``
    with ``KProt = (0.81 + 0.11·10^LogMA)/24.92·5`` — the method the
    PK-Sim UI defaults to, on the base Vf composition columns.
    """
    if not 0 < fu <= 1:
        raise DataValidationError(f"fu must be in (0, 1], got {fu}")
    comp = _resolve_tissue(tissue)
    # Fat's water/lipid fractions are age curves in PK-Sim — absent
    # columns read as 0.0 rather than refusing the tissue.
    return (
        comp.get("f_water", 0.0)
        + comp.get("f_lipid", 0.0) * 10.0**log_ma
        + comp.get("f_protein", 0.0) * _kprot(log_ma)
    ) * fu


def _fu_tissue(fu: float, albumin_ratio: float) -> float:
    """Poulin & Theil's unbound-in-tissue fraction via the albumin ratio."""
    return 1.0 / (1.0 + (1.0 - fu) / fu * albumin_ratio)


def kp_poulin_theil(log_p: float, fu: float, tissue: str = "Liver") -> float:
    """Poulin & Theil 2002 — neutral-lipid / phospholipid volume model.

    ``K = (10^LogP·(V_nlt + 0.3·V_pht) + 0.7·V_pht + V_wt) /
    (10^LogP·(V_nlp + 0.3·V_php) + 0.7·V_php + V_wp) · (fu/fu_t)``
    with ``fu_t = 1/(1 + (1−fu)/fu·RA)`` — exactly the formula string
    in ``tab_calculation_method_rates``.  For an unbound drug the
    tissue binding term vanishes and the method reduces to the
    volume-ratio core.
    """
    if not 0 < fu <= 1:
        raise DataValidationError(f"fu must be in (0, 1], got {fu}")
    comp = _resolve_tissue(tissue)
    fu_t = _fu_tissue(fu, comp.get("albumin_ratio", 0.0))
    num = (
        10.0**log_p * (comp["v_nlt"] + 0.3 * comp["v_pht"])
        + 0.7 * comp["v_pht"]
        + comp["v_wt"]
    )
    den = (
        10.0**log_p * (PLASMA_COMPOSITION["v_nlp"] + 0.3 * PLASMA_COMPOSITION["v_php"])
        + 0.7 * PLASMA_COMPOSITION["v_php"]
        + PLASMA_COMPOSITION["v_wp"]
    )
    return num / den * (fu / fu_t)


def kp_berezhkovskiy(log_p: float, fu: float, tissue: str = "Liver") -> float:
    """Berezhkovskiy 2004 — Poulin & Theil with the fu correction folded in.

    ``K = (10^LogP·(V_nlt + 0.3·V_pht) + 0.7·V_pht + V_wt/fu_t) /
    (10^LogP·(V_nlp + 0.3·V_php) + 0.7·V_php + V_wp/fu)`` — the
    fu/fu_t asymmetry moves *inside* the two sums, so a bound drug
    partitions less than Poulin & Theil predicts at the same inputs.
    """
    if not 0 < fu <= 1:
        raise DataValidationError(f"fu must be in (0, 1], got {fu}")
    comp = _resolve_tissue(tissue)
    fu_t = _fu_tissue(fu, comp.get("albumin_ratio", 0.0))
    num = (
        10.0**log_p * (comp["v_nlt"] + 0.3 * comp["v_pht"])
        + 0.7 * comp["v_pht"]
        + comp["v_wt"] / fu_t
    )
    den = (
        10.0**log_p * (PLASMA_COMPOSITION["v_nlp"] + 0.3 * PLASMA_COMPOSITION["v_php"])
        + 0.7 * PLASMA_COMPOSITION["v_php"]
        + PLASMA_COMPOSITION["v_wp"] / fu
    )
    return num / den


def kp_rodgers_rowland(
    log_p: float,
    fu: float,
    tissue: str = "Liver",
    pka: float | None = None,
    charge: str = "neutral",
) -> float:
    """Rodgers & Rowland 2005-07 — the mechanistic ionic model.

    ``K_cell_pls = ((1+X)/(1+Y)·f_iw + f_ew +
    (10^LogP·f_nl + (0.3·10^LogP+0.7)·f_np)/(1+Y) +
    Ka_PR·(neutral ? lipoprotein ratio : albumin ratio))·fu`` with the
    monoprotic ionic terms ``X = 10^(pKa−pH_IW)`` (bases) /
    ``10^(pH_IW−pKa)`` (acids) and ``Y`` the same at pH 7.4, on the
    pH values PK-Sim ships (intracellular 7.0, plasma 7.4).
    ``Ka_PR = max(0, 1/fu − 1 − (10^LogP·f_nl_p + (0.3·10^LogP+0.7)·
    f_np_p)/(1+Y))``.  Bases gain intracellular accumulation (the
    (1+X)/(1+Y) water term); acids lose it.

    Beyond the honest boundary: diprotic species and the strong-base
    acidic-phospholipid term (``Ka_AP·AP_T·X/(1+Y)``) — the descriptor
    conditions that switch them on live in PK-Sim's formula evaluator.
    """
    if not 0 < fu <= 1:
        raise DataValidationError(f"fu must be in (0, 1], got {fu}")
    charge = charge.lower()
    if charge not in ("neutral", "base", "acid"):
        raise DataValidationError(
            f"charge must be 'neutral', 'base' or 'acid', got {charge!r}"
        )
    if charge != "neutral" and pka is None:
        raise DataValidationError("charged species need a pka")
    comp = _resolve_tissue(tissue)
    ph_iw = PLASMA_COMPOSITION["ph_intracellular"]
    ph_pls = PLASMA_COMPOSITION["ph_plasma"]
    x_iw = 0.0
    y_pls = 0.0
    if charge == "base" and pka is not None:
        x_iw = 10.0 ** (pka - ph_iw)
        y_pls = 10.0 ** (pka - ph_pls)
    elif charge == "acid" and pka is not None:
        x_iw = 10.0 ** (ph_iw - pka)
        y_pls = 10.0 ** (ph_pls - pka)
    f_nl_p = PLASMA_COMPOSITION["f_nl_p"]
    f_np_p = PLASMA_COMPOSITION["f_np_p"]
    ka_pr = max(
        0.0,
        1.0 / fu
        - 1.0
        - (10.0**log_p * f_nl_p + (0.3 * 10.0**log_p + 0.7) * f_np_p)
        / (1.0 + y_pls),
    )
    pr = (
        comp.get("lipoprotein_ratio", 0.0)
        if charge == "neutral"
        else comp.get("albumin_ratio", 0.0)
    )
    return (
        (1.0 + x_iw) / (1.0 + y_pls) * comp["f_iw"]
        + comp["f_ew"]
        + (
            10.0**log_p * comp["f_nl_rr"]
            + (0.3 * 10.0**log_p + 0.7) * comp["f_np_rr"]
        )
        / (1.0 + y_pls)
        + ka_pr * pr
    ) * fu


def kp_schmitt(
    log_d: float,
    log_ma: float,
    fu: float,
    tissue: str = "Liver",
    k_a_pl: float = 0.0,
) -> float:
    """Schmitt 2008 — the microspecies method, neutral form.

    ``K = (f_w + K_n_L·f_nl + K_n_PL·f_np + K_a_PL·AP_T +
    KProt·f_proteins)·fu`` with ``K_n_L = 10^LogD`` and
    ``K_n_PL = 10^LogMA``.  The composition mapping follows PK-Sim's
    descriptor conditions (water/neutral-lipid/neutral-phospholipid
    fractions from the -RR columns, AP_T the RR acidic phospholipids in
    mg/g).  The charge-state regression that turns ``K_a_PL`` into the
    ``K1 + K2·base^CT0 + …`` pH factor is a C# constant set in the
    OSPSuite formula evaluator, not a database row — so ``k_a_pl`` is
    an explicit input (0 for neutral drugs, the honest default).
    """
    if not 0 < fu <= 1:
        raise DataValidationError(f"fu must be in (0, 1], got {fu}")
    if k_a_pl < 0:
        raise DataValidationError(f"k_a_pl must be >= 0, got {k_a_pl}")
    comp = _resolve_tissue(tissue)
    ap_t = comp.get("acidic_phospholipids_rr", 0.0)
    return (
        comp["f_water"]
        + 10.0**log_d * comp.get("f_nl_rr", 0.0)
        + 10.0**log_ma * comp.get("f_np_rr", 0.0)
        + k_a_pl * ap_t
        + _kprot(log_ma) * comp["f_protein"]
    ) * fu


def blood_plasma_ratio(log_ma: float, fu: float, hct: float = 0.47) -> float:
    """Blood:plasma ratio — PK-Sim's ``DistributionInVitro`` B2P bridge.

    ``K_rbc = (f_water_rbc + f_lipid_rbc·10^LogMA +
    f_protein_rbc·KProt)·fu`` on the blood-cell composition columns,
    then ``B2P = 1 + HCT·(K_rbc − 1)`` (algebraically the database's
    ``B2P = K_rbc·HCT − HCT + 1``).  A compound that enters red cells
    (K_rbc > 1) concentrates in whole blood above plasma.
    """
    if not 0 < fu <= 1:
        raise DataValidationError(f"fu must be in (0, 1], got {fu}")
    if not 0 <= hct <= 1:
        raise DataValidationError(f"hct must be in [0, 1], got {hct}")
    k_rbc = (
        PLASMA_COMPOSITION["f_water_rbc"]
        + PLASMA_COMPOSITION["f_lipid_rbc"] * 10.0**log_ma
        + PLASMA_COMPOSITION["f_protein_rbc"] * _kprot(log_ma)
    ) * fu
    return 1.0 + hct * (k_rbc - 1.0)

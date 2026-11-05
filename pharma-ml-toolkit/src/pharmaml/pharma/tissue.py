r"""Tissue distribution, ontogeny and renal maturation — the PK-Sim data stack.

Round 176 absorbed ``Open-Systems-Pharmacology/PK-Sim``'s mathematical
core.  The C# engine, UI and serialization are the not-vendored part;
what transfers is (a) the five cellular partition-coefficient methods
stored as formula strings in ``src/Db/Dump/PKSimDB_dump.txt``
(``tab_calculation_method_rates``, lines 2956-3065) — transcribed
verbatim below; (b) the human tissue-composition and anthropometry
tables extracted from the same dump (TissueComposition_PKSim /
OrganComposition_PKSim / the European_ICRP_2002 male, age-30 curves);
and (c) the ontogeny and GFR machinery
(``OntogenyRepository.cs``, ``tab_ontogenies``, the GFR Hill formula
at dump line 3435 and the renal-aging factor at line 3812).

Methods implemented exactly as the dump writes them:

* **Rodgers & Rowland** — the full X/Y/Z ionization sums (per-protic
  prefix chains, the pure-acid ``Z = 1`` rule), ``Ka_AP`` back-calculated
  from the blood:plasma ratio, ``Ka_PR`` from fu, the strong-base
  acidic-phospholipid branch, and the fat ``LogP_ow = 1.115*LogP -
  1.35`` variant.
* **Berezhkovskiy** / **Poulin & Theil** — the volume-fraction form with
  ``fu_tissue`` and the pKa-corrected ``LogP_ow`` for fat.
* **Schmitt** — the eight charge-state products ``K1..K8`` of the
  neutral fractions ``Fi = 1/(1+10^(CT_i*(pKa_i-pH)))``, ``LogD`` and
  the acidic-phospholipid binding term.
* **PK-Sim Standard (Willmann)** — ``(f_water + f_lipids*10^LogMA +
  f_proteins*KProt)*fu`` with ``KProt = (0.81 + 0.11*10^LogMA)/24.92*5``.

Plus :func:`vss_physchem` — the closed-form Vss of
``PKSim.Core/Services/VSSCalculator.cs`` (``sum_tissue (K_cell_pls +
f_vas*(1-HCT))*V + sum_vascular V*(1-HCT)``, all over body weight) —
and the pediatric/geriatric stack: CYP ontogeny with the
one-percentile-per-subject rule, the GFR Hill maturation
(``TM50 = 44.4 d``, ``Hill = 15``, ``fGFRpremat = 0.258``), the renal
aging factor (onset 30 y, half-time 54 y male / 59 y female, Hill 1.5,
max decrease 0.9) and the plasma-protein-binding ontogeny scaling.

Examples
--------
>>> from pharmaml.pharma.tissue import CompoundProperties, kp_table
>>> compound = CompoundProperties(log_p=2.0, log_ma=1.5, fu=0.5,
...                               pka_bases=(9.0,))
>>> table = kp_table(compound, method="rodgers_rowland")
>>> 2.0 < float(table["Muscle"]) < 3.0          # weak base: ion trapping
True
>>> table.idxmax() == "Kidney"                  # AP-rich kidney binds bases
True

>>> from pharmaml.pharma.tissue import vss_physchem
>>> vss = vss_physchem(table)
>>> 1.0 < vss < 40.0                            # L/kg, physiologic range
True

>>> from pharmaml.pharma.tissue import ontogeny_factor, gfr_maturation_factor
>>> round(float(ontogeny_factor("CYP3A4", age_years=0.18, ga_weeks=0)), 4)  # fetus
0.001
>>> abs(float(ontogeny_factor("CYP3A4", age_years=30.0)) - 1.0) < 1e-9   # adult
True
>>> round(float(gfr_maturation_factor(pma_years=3.0)), 3)
1.0
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from pharmaml.exceptions import DataValidationError

__all__ = [
    "CKD_STAGE_FACTORS",
    "GFR_CONSTANTS",
    "ONTOGENY_TABLES",
    "STANDARD_INDIVIDUAL",
    "TISSUE_COMPOSITION",
    "CompoundProperties",
    "berezhkovskiy_kp",
    "blood_to_plasma_ws",
    "fu_plasma_scaled",
    "gfr_maturation_factor",
    "kp_table",
    "ontogeny_factor",
    "pk_sim_standard_kp",
    "poulin_theil_kp",
    "renal_aging_factor",
    "rodgers_rowland_kp",
    "schmitt_kp",
    "vss_physchem",
]

# ---------------------------------------------------------------------------
# physiology constants (PKSimDB_dump.txt, Human European_ICRP_2002 male 30 y)
# ---------------------------------------------------------------------------

PH_PLASMA = 7.4
PH_INTRACELLULAR = 7.0
PH_RBC = 7.22
HCT = 0.47
BODY_WEIGHT = 73.0  # kg (MeanBW, European male)

# blood cells / plasma composition (Organism rows of TissueComposition_PKSim)
BLOOD_CELLS_WS = {"f_water": 0.625, "f_lipids": 0.005, "f_proteins": 0.325}
PLASMA_WS = {"f_water": 0.926, "f_lipids": 0.007, "f_proteins": 0.067}
PLASMA_PT = {"v_nl": 0.0035, "v_ph": 0.00225, "v_w": 0.945}
INTERSTITIAL_WATER = 0.935
PROTEIN_RATIO_INTERSTITIAL = 0.37

# R Rodgers & Rowland blood-cell / plasma inputs
RBC_RR = {"f_iw": 0.63, "f_nl": 0.0012, "f_np": 0.0033, "ap": 0.57}
PLASMA_RR = {"f_nl": 0.0032, "f_np": 0.0021}

# tissue composition: RR columns (f_iw, f_ew, f_nl, f_np, AP_T g/g,
# albumin ratio, lipoprotein ratio), PT columns (v_nl, v_ph, v_w, RA),
# WS columns (f_water, f_lipids, f_proteins).  Human adult male;
# Fat/Muscle RR + PT + WS values from the age-30 (Fat) / age>=5 (Muscle)
# European_ICRP_2002 curves, the rest from TissueComposition_PKSim.
TISSUE_COMPOSITION: dict[str, dict[str, tuple[float, ...]]] = {
    #              RR:  f_iw   f_ew   f_nl    f_np    AP_T   alb    lipo
    #                  PT:  v_nl   v_ph   v_w     RA
    #                  WS:  water  lipids proteins
    "Fat": {
        "rr": (0.009, 0.135, 0.853, 0.0016, 0.40, 0.049, 0.068),
        "pt": (0.790, 0.002, 0.180, 0.0),
        "ws": (0.030, 0.920, 0.060),
    },
    "Muscle": {
        "rr": (0.666, 0.079, 0.022, 0.0078, 2.42, 0.064, 0.059),
        "pt": (0.0238, 0.0072, 0.8144, 0.5),
        "ws": (0.8144, 0.0049, 0.1932),
    },
    "Brain": {
        "rr": (0.591, 0.162, 0.0391, 0.0015, 0.40, 0.048, 0.041),
        "pt": (0.051, 0.0565, 0.77, 0.5),
        "ws": (0.79, 0.0429, 0.08),
    },
    "Heart": {
        "rr": (0.248, 0.320, 0.0135, 0.0106, 2.25, 0.157, 0.160),
        "pt": (0.011, 0.0173, 0.758, 0.5),
        "ws": (0.70, 0.053, 0.19),
    },
    "Kidney": {
        "rr": (0.399, 0.273, 0.0121, 0.024, 5.03, 0.130, 0.137),
        "pt": (0.021, 0.0164, 0.783, 0.5),
        "ws": (0.73, 0.016, 0.21),
    },
    "Liver": {
        "rr": (0.481, 0.161, 0.0135, 0.0238, 4.56, 0.086, 0.161),
        "pt": (0.0348, 0.0252, 0.751, 0.5),
        "ws": (0.68, 0.0232, 0.21),
    },
    "Lung": {
        "rr": (0.238, 0.336, 0.0215, 0.0123, 3.91, 0.212, 0.168),
        "pt": (0.003, 0.009, 0.811, 0.5),
        "ws": (0.74, 0.02, 0.11),
    },
    "Skin": {
        "rr": (0.276, 0.382, 0.0603, 0.0044, 1.32, 0.277, 0.096),
        "pt": (0.028, 0.011, 0.718, 0.5),
        "ws": (0.47, 0.126, 0.41),
    },
    "Bone": {
        "rr": (0.317, 0.100, 0.0174, 0.0016, 0.67, 0.100, 0.050),
        "pt": (0.074, 0.001, 0.439, 0.5),
        "ws": (0.26, 0.017, 0.21),
    },
    "Spleen": {
        "rr": (0.355, 0.207, 0.0071, 0.0107, 3.18, 0.097, 0.207),
        "pt": (0.020, 0.020, 0.788, 0.5),
        "ws": (0.75, 0.006, 0.23),
    },
    "Pancreas": {
        "rr": (0.521, 0.120, 0.0403, 0.009, 1.67, 0.060, 0.060),
        "pt": (0.040, 0.011, 0.710, 0.5),
        "ws": (0.71, 0.04, 0.13),
    },
    "Gut": {  # small intestine row; SI/LI/stomach share RR values in the dump
        "rr": (0.456, 0.282, 0.0375, 0.0124, 2.41, 0.158, 0.141),
        "pt": (0.049, 0.016, 0.718, 0.5),
        "ws": (0.78, 0.048, 0.15),
    },
}

# standard individual volumes (L) and vascular fractions; gut = SI+LI+stomach
STANDARD_INDIVIDUAL: dict[str, Any] = {
    "volumes": {
        "Fat": 14.868, "Muscle": 32.339, "Brain": 1.5088, "Heart": 0.4172,
        "Kidney": 0.4377, "Liver": 2.3578, "Lung": 1.2943, "Skin": 3.7609,
        "Bone": 11.818, "Spleen": 0.2434, "Pancreas": 0.1903, "Gut": 1.3048,
        "ArterialBlood": 0.41908, "VenousBlood": 0.96388, "PortalVein": 1.0372,
    },
    "fraction_vascular": {
        "Fat": 0.018, "Muscle": 0.025, "Brain": 0.039, "Heart": 0.14,
        "Kidney": 0.23, "Liver": 0.17, "Lung": 0.58, "Skin": 0.046,
        "Bone": 0.034, "Spleen": 0.33, "Pancreas": 0.20, "Gut": 0.024,
        "ArterialBlood": 1.0, "VenousBlood": 1.0, "PortalVein": 1.0,
    },
    "hct": HCT,
    "body_weight": BODY_WEIGHT,
}

# GFR maturation + renal aging (tab_container_parameter_curves, Kidney)
GFR_CONSTANTS: dict[str, float] = {
    "tm50_days": 44.4,
    "hill": 15.0,
    "f_premat": 0.258,
    "gfr_mat": 0.11704,  # L/min/kg kidney
    "age_onset": 30.0,
    "aging_half_time_male": 54.0,
    "aging_half_time_female": 59.0,
    "aging_hill": 1.5,
    "max_decrease": 0.9,
}

# CKD plasma-protein scale factors (CKDDiseaseStateImplementation.cs 177-191)
CKD_STAGE_FACTORS: dict[int, float] = {3: 1.07, 4: 1.16, 5: 1.55}

# ---------------------------------------------------------------------------
# ontogeny tables (tab_ontogenies, Human; PMA years -> (factor, geoSD))
# ---------------------------------------------------------------------------

ONTOGENY_TABLES: dict[str, dict[str, list[tuple[float, float, float]]]] = {
    "CYP3A4": {
        "liver": [
            (0.18, 0.001, 4.63),
            (0.37, 0.011, 3.22),
            (0.5, 0.03, 2.73),
            (0.63, 0.064, 2.46),
            (0.77, 0.12, 2.22),
            (0.92, 0.2, 1.92),
            (1.39, 0.49, 1.45),
            (1.55, 0.58, 1.37),
            (1.73, 0.67, 1.31),
            (1.91, 0.74, 1.26),
            (2.08, 0.79, 1.23),
            (2.26, 0.83, 1.22),
            (2.5, 0.87, 1.19),
            (2.84, 0.91, 1.18),
            (3.19, 0.94, 1.18),
            (3.81, 0.97, 1.18),
            (4.69, 0.98, 1.19),
            (6.48, 0.99, 1.18),
            (11.17, 1, 1.18),
        ],
    },
    "CYP2D6": {
        "liver": [
            (0.2, 0.001, 27.88),
            (0.32, 0.012, 14.29),
            (0.41, 0.046, 8.25),
            (0.49, 0.1, 5.76),
            (0.58, 0.23, 4.13),
            (0.77, 0.58, 2.65),
            (0.86, 0.71, 2.46),
            (0.96, 0.82, 2.38),
            (1.09, 0.9, 2.35),
            (1.26, 0.95, 2.3),
            (1.52, 0.98, 2.38),
            (1.86, 0.99, 2.41),
            (2.61, 1, 2.41),
        ],
    },
    "CYP2C9": {
        "liver": [
            (0.3, 0.00098, 2.91),
            (0.44, 0.019, 2.4),
            (0.51, 0.061, 2.33),
            (0.55, 0.12, 2.22),
            (0.62, 0.24, 2.15),
            (0.77, 0.66, 2.01),
            (0.84, 0.8, 2.03),
            (0.91, 0.89, 2.02),
            (1, 0.94, 2.02),
            (1.15, 0.98, 2.01),
            (1.39, 1, 2.01),
        ],
    },
    "CYP2C19": {
        "liver": [
            (0.33, 0.001, 46.35),
            (0.47, 0.03, 10.04),
            (0.53, 0.09, 5.56),
            (0.58, 0.18, 3.76),
            (0.62, 0.3, 2.96),
            (0.72, 0.64, 1.96),
            (0.77, 0.76, 1.84),
            (0.81, 0.83, 1.75),
            (0.87, 0.91, 1.68),
            (0.94, 0.95, 1.71),
            (1.09, 0.99, 1.69),
            (1.39, 1, 1.72),
            (1.42, 1, 1.75),
        ],
    },
    "CYP1A2": {
        "liver": [
            (0.63, 0.00099, 74.55),
            (0.77, 0.007, 24.93),
            (0.88, 0.026, 11.8),
            (0.98, 0.071, 6.54),
            (1.05, 0.14, 4.42),
            (1.14, 0.26, 3.15),
            (1.34, 0.65, 1.87),
            (1.44, 0.79, 1.64),
            (1.55, 0.88, 1.59),
            (1.61, 0.92, 1.55),
            (1.74, 0.96, 1.53),
            (1.93, 0.99, 1.52),
            (2.52, 1, 1.56),
        ],
    },
    "CYP2E1": {
        "liver": [
            (0.48, 0.00099, 21.07),
            (0.66, 0.021, 9.07),
            (0.77, 0.077, 4.85),
            (0.82, 0.13, 3.67),
            (0.9, 0.26, 2.51),
            (1.05, 0.6, 1.63),
            (1.12, 0.73, 1.47),
            (1.21, 0.85, 1.39),
            (1.3, 0.92, 1.33),
            (1.43, 0.96, 1.32),
            (1.62, 0.99, 1.33),
            (2.12, 1, 1.34),
        ],
    },
    "CYP3A5": {
        "liver": [
            (0, 1, 2.25),
            (0.77, 1, 2.25),
        ],
    },
    "CYP3A7": {
        "liver": [
            (0.72, 1, 1.25),
            (0.77, 0.99, 1.25),
            (0.83, 0.95, 1.26),
            (0.87, 0.85, 1.25),
            (0.9, 0.71, 1.25),
            (0.95, 0.36, 1.25),
            (0.98, 0.2, 1.26),
            (1.02, 0.078, 1.25),
            (1.05, 0.054, 1.26),
            (1.12, 0.03, 1.25),
            (1.83, 0.025, 1.26),
            (3.36, 0.025, 1.25),
            (3.41, 0.025, 1.25),
            (3.56, 0.025, 1.26),
        ],
    },
    "UGT2B7": {
        "liver": [
            (0.48, 0.00099, 17.13),
            (0.66, 0.0073, 12.79),
            (0.77, 0.02, 8.73),
            (0.91, 0.057, 5.68),
            (1.03, 0.12, 3.97),
            (1.15, 0.22, 2.96),
            (1.24, 0.31, 2.48),
            (1.5, 0.62, 1.77),
            (1.64, 0.74, 1.67),
            (1.76, 0.82, 1.61),
            (1.89, 0.88, 1.57),
            (2.07, 0.93, 1.54),
            (2.35, 0.97, 1.52),
            (2.86, 0.99, 1.56),
            (4.01, 1, 1.59),
        ],
    },
    "AGP": {
        "plasma": [
            (0.766712, 0.254564, 1.82915),
            (0.770545, 0.288104, 1.88182),
            (0.774378, 0.320477, 1.84036),
            (0.785877, 0.404893, 1.7157),
            (0.797376, 0.472581, 1.62895),
            (0.805042, 0.510391, 1.58835),
            (0.816541, 0.558547, 1.54513),
            (0.831873, 0.610479, 1.50878),
            (0.847205, 0.652063, 1.48674),
            (0.889368, 0.732511, 1.4591),
            (0.946863, 0.798311, 1.44769),
            (1.03502, 0.855225, 1.44314),
            (1.16918, 0.900406, 1.44166),
            (1.38766, 0.935749, 1.44129),
            (1.72113, 0.96008, 1.44127),
            (3.03201, 0.987428, 1.44142),
            (13.6071, 1.00201, 1.44146),
            (30.7666, 1, 1.44095),
            (90.4586, 0.986597, 1.44347),
        ],
    },
    "ALB": {
        "plasma": [
            (0.05, 0.0011, 4.46),
            (0.1, 0.011, 2.8),
            (0.16, 0.04, 2.2),
            (0.23, 0.13, 1.74),
            (0.46, 0.59, 1.29),
            (0.55, 0.71, 1.24),
            (0.64, 0.81, 1.2),
            (0.77, 0.88, 1.19),
            (0.88, 0.92, 1.2),
            (1.06, 0.96, 1.17),
            (1.42, 0.98, 1.18),
            (2.17, 1, 1.18),
        ],
    },
}


# ---------------------------------------------------------------------------
# compound properties + ionization helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompoundProperties:
    """The physicochemical inputs of the partition methods.

    ``pka_acids`` ascending (first proton off first); ``pka_bases``
    ascending (``Base_0`` is the lowest, protonated last).  ``b2p`` is
    the measured blood:plasma ratio — when ``None`` the PK-Sim Standard
    estimate :func:`blood_to_plasma_ws` is used (the R&R back-calculation
    needs it for ``Ka_AP``).
    """

    log_p: float
    log_ma: float
    fu: float
    pka_acids: tuple[float, ...] = ()
    pka_bases: tuple[float, ...] = ()
    b2p: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.fu <= 1:
            raise DataValidationError("fu must be in (0, 1]")


def _acid_prefix_sum(pkas: Sequence[float], ph: float) -> float:
    """Sum of deprotonated species over the neutral acid (dump X branches)."""
    total = 0.0
    running = 1.0
    for pka in sorted(pkas):
        running *= 10 ** (ph - pka)
        total += running
    return total


def _base_prefix_sum(pkas: Sequence[float], ph: float) -> float:
    """Sum of protonated species over the neutral base (dump X branches)."""
    total = 0.0
    running = 1.0
    for pka in sorted(pkas, reverse=True):
        running *= 10 ** (pka - ph)
        total += running
    return total


def _x_sum(compound: CompoundProperties, ph: float) -> float:
    """The R&R X sum at ``ph`` (intracellular-water ionization ratio)."""
    return _acid_prefix_sum(compound.pka_acids, ph) + _base_prefix_sum(
        compound.pka_bases, ph
    )


def _z_sum(compound: CompoundProperties, ph: float) -> float:
    """The R&R Z sum — 1 for pure acids (the dump rule), full sum otherwise."""
    if compound.pka_bases:
        return _x_sum(compound, ph)
    return 1.0


def _has_strong_base(compound: CompoundProperties) -> bool:
    """Any base group with pKa >= 7 (PK-Sim HasStrongBase)."""
    return any(pka >= 7.0 for pka in compound.pka_bases)


def _kprot(log_ma: float) -> float:
    """``(0.81 + 0.11*10^LogMA)/24.92*5`` (DistributionInVitro_PKSim)."""
    return (0.81 + 0.11 * 10**log_ma) / 24.92 * 5.0


def _logp_ow_rr(log_p: float) -> float:
    """Rodgers & Rowland fat correction (no pKa term)."""
    return 1.115 * log_p - 1.35


def _logp_ow_pt(compound: CompoundProperties, ph: float) -> float:
    """Poulin & Theil / Berezhkovskiy fat correction with pKa terms."""
    correction = 0.0
    for pka in compound.pka_acids:
        correction += math.log10(1 + 10 ** (ph - pka))
    for pka in compound.pka_bases:
        correction += math.log10(1 + 10 ** (pka - ph))
    return 1.115 * compound.log_p - 1.35 - correction


# ---------------------------------------------------------------------------
# partition methods
# ---------------------------------------------------------------------------


def rodgers_rowland_kp(
    compound: CompoundProperties,
    organ: str | None = None,
    hct: float = HCT,
) -> pd.Series | float:
    """Rodgers & Rowland cellular partition coefficients (K_cell_pls).

    The exact dump formula: ``((1+X)/(1+Y)*f_iw + f_ew +
    (10^LogP*f_nl+(0.3*10^LogP+0.7)*f_np)/(1+Y) + (HasStrongBase ?
    Ka_AP*AP_T*X/(1+Y) : Ka_PR*(IsNeutral ? lipo : alb)))*fu`` with
    ``Ka_PR`` and ``Ka_AP`` from fu / the blood:plasma ratio, and fat
    using ``LogP_ow = 1.115*LogP - 1.35``.

    Examples
    --------
    >>> compound = CompoundProperties(log_p=2.0, log_ma=1.5, fu=0.5)
    >>> kp = rodgers_rowland_kp(compound)
    >>> 0.4 < float(kp["Liver"]) < 2.5
    True
    """
    y = _x_sum(compound, PH_PLASMA)
    x = _x_sum(compound, PH_INTRACELLULAR)
    z = _z_sum(compound, PH_RBC)
    neutral = not compound.pka_acids and not compound.pka_bases
    strong_base = _has_strong_base(compound)
    b2p = compound.b2p if compound.b2p is not None else blood_to_plasma_ws(compound)

    kpu_bc = (hct - 1 + b2p) / compound.fu / hct
    ka_pr = max(
        0.0,
        1 / compound.fu
        - 1
        - (10**compound.log_p * PLASMA_RR["f_nl"]
           + (0.3 * 10**compound.log_p + 0.7) * PLASMA_RR["f_np"]) / (1 + y),
    )
    ka_ap = max(
        0.0,
        kpu_bc
        - (1 + z) / (1 + y) * RBC_RR["f_iw"]
        - (10**compound.log_p * RBC_RR["f_nl"]
           + (0.3 * 10**compound.log_p + 0.7) * RBC_RR["f_np"]) / (1 + y),
    ) * (1 + y) / RBC_RR["ap"] / z if z > 0 else 0.0

    results: dict[str, float] = {}
    for name, cols in TISSUE_COMPOSITION.items():
        f_iw, f_ew, f_nl, f_np, ap_t, alb, lipo = cols["rr"]
        logp = _logp_ow_rr(compound.log_p) if name == "Fat" else compound.log_p
        if strong_base:
            binding = ka_ap * ap_t * x / (1 + y)
        else:
            binding = ka_pr * (lipo if neutral else alb)
        kp = (
            (1 + x) / (1 + y) * f_iw
            + f_ew
            + (10**logp * f_nl + (0.3 * 10**logp + 0.7) * f_np) / (1 + y)
            + binding
        ) * compound.fu
        results[name] = kp
    series = pd.Series(results, name="rodgers_rowland")
    if organ is None:
        return series
    if organ not in series.index:
        raise DataValidationError(f"unknown organ {organ!r}")
    return float(series[organ])


def _fu_tissue(fu: float, ra: float) -> float:
    """``1/(1 + (1-fu)/fu * RA)`` — the PT/Berezhkovskiy tissue unbound fraction."""
    return 1 / (1 + (1 - fu) / fu * ra)


def poulin_theil_kp(compound: CompoundProperties) -> pd.Series:
    """Poulin & Theil partition coefficients (dump formula, fat variant).

    ``(10^LogP*(V_nlt+0.3*V_pht) + 0.7*V_pht + V_wt) / (10^LogP*(V_nlp
    + 0.3*V_php) + 0.7*V_php + V_wp) * (fu/fu_t)`` with the pKa-corrected
    fat ``LogP_ow``.
    """
    fu_t = {
        name: _fu_tissue(compound.fu, cols["pt"][3]) for name, cols in TISSUE_COMPOSITION.items()
    }
    results: dict[str, float] = {}
    for name, cols in TISSUE_COMPOSITION.items():
        v_nlt, v_pht, v_wt, _ra = cols["pt"]
        logp = _logp_ow_pt(compound, PH_PLASMA) if name == "Fat" else compound.log_p
        num = 10**logp * (v_nlt + 0.3 * v_pht) + 0.7 * v_pht + v_wt
        den = (
            10**compound.log_p * (PLASMA_PT["v_nl"] + 0.3 * PLASMA_PT["v_ph"])
            + 0.7 * PLASMA_PT["v_ph"]
            + PLASMA_PT["v_w"]
        )
        results[name] = num / den * (compound.fu / fu_t[name])
    return pd.Series(results, name="poulin_theil")


def berezhkovskiy_kp(compound: CompoundProperties) -> pd.Series:
    """Berezhkovskiy partition coefficients (dump formula).

    ``(10^LogP*(V_nlt+0.3*V_pht) + 0.7*V_pht + V_wt/fu_t) / (10^LogP *
    (V_nlp+0.3*V_php) + 0.7*V_php + V_wp/fu)`` with the fat ``LogP_ow``.
    """
    results: dict[str, float] = {}
    for name, cols in TISSUE_COMPOSITION.items():
        v_nlt, v_pht, v_wt, ra = cols["pt"]
        fu_t = _fu_tissue(compound.fu, ra)
        logp = _logp_ow_pt(compound, PH_PLASMA) if name == "Fat" else compound.log_p
        num = 10**logp * (v_nlt + 0.3 * v_pht) + 0.7 * v_pht + v_wt / fu_t
        den = (
            10**compound.log_p * (PLASMA_PT["v_nl"] + 0.3 * PLASMA_PT["v_ph"])
            + 0.7 * PLASMA_PT["v_ph"]
            + PLASMA_PT["v_w"] / compound.fu
        )
        results[name] = num / den
    return pd.Series(results, name="berezhkovskiy")


def schmitt_kp(compound: CompoundProperties) -> pd.Series:
    """Schmitt partition coefficients (dump formula, K1..K8 machinery).

    ``(f_w + K_n_L*f_nl + K_n_PL*f_np + K_a_PL*AP_T + KProt*f_proteins)*fu``
    with ``K_n_L = 10^LogD`` and ``LogD = LogP + log10(logD_Factor)`` —
    up to three ionizable groups (the PK-Sim pKa slot convention).
    """
    groups = [(pka, -1.0) for pka in compound.pka_acids] + [
        (pka, +1.0) for pka in compound.pka_bases
    ]
    groups = groups[:3]
    while len(groups) < 3:
        groups.append((0.0, 0.0))

    factors = []
    for ph in (PH_INTRACELLULAR,):
        fi = [
            1.0 if ct == 0 else 1 / (1 + 10 ** (ct * (pka - ph))) for pka, ct in groups
        ]
        k_vals = []
        for mask in range(8):  # K1..K8 — the (1-F) products
            value = 1.0
            for bit, f in enumerate(fi):
                value *= (1 - f) if mask & (1 << bit) else f
            k_vals.append(value)
        charges = []
        for mask in range(8):
            net = sum(ct for bit, (_pka, ct) in enumerate(groups) if mask & (1 << bit))
            charges.append(10.0**net)
        # logD factor: K1 + (K2+K3+K4)*base^1 + folded charge terms
        folded = 0.0
        pairs = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
        for pair in pairs:
            total_charge = sum(groups[i][1] for i in pair)
            mask = 0
            for i in pair:
                mask |= 1 << i
            folded += k_vals[mask] * 10 ** abs(total_charge)
        factor = k_vals[0] + (k_vals[1] + k_vals[2] + k_vals[3]) * 10.0 + folded
        factors.append(factor)
    log_d = compound.log_p + math.log10(factors[0])

    k_n_l = 10**log_d
    k_n_pl = 10**compound.log_ma
    # K_a_PL pH factor: sum over species of K * 10^(net charge)
    fi = [1.0 if ct == 0 else 1 / (1 + 10 ** (ct * (pka - PH_INTRACELLULAR)))
          for pka, ct in groups]
    k_vals = []
    for mask in range(8):
        value = 1.0
        for bit, f in enumerate(fi):
            value *= (1 - f) if mask & (1 << bit) else f
        k_vals.append(value)
    ph_factor = 0.0
    for mask in range(8):
        net = sum(ct for bit, (_pka, ct) in enumerate(groups) if mask & (1 << bit))
        ph_factor += k_vals[mask] * 10.0**net
    k_a_pl = k_n_pl * ph_factor
    k_prot = _kprot(compound.log_ma)

    results: dict[str, float] = {}
    for name, cols in TISSUE_COMPOSITION.items():
        f_iw, f_ew, f_nl, f_np, ap_t, _alb, _lipo = cols["rr"]
        f_w = f_iw + f_ew
        f_proteins = max(0.0, 1.0 - f_w - f_nl - f_np)
        results[name] = (
            f_w + k_n_l * f_nl + k_n_pl * f_np + k_a_pl * ap_t + k_prot * f_proteins
        ) * compound.fu
    return pd.Series(results, name="schmitt")


def pk_sim_standard_kp(compound: CompoundProperties) -> pd.Series:
    """PK-Sim Standard (Willmann) partition coefficients.

    ``(f_water + f_lipids*10^LogMA + f_proteins*KProt)*fu`` over the WS
    tissue composition.
    """
    k_prot = _kprot(compound.log_ma)
    results: dict[str, float] = {}
    for name, cols in TISSUE_COMPOSITION.items():
        f_water, f_lipids, f_proteins = cols["ws"]
        results[name] = (
            f_water + f_lipids * 10**compound.log_ma + f_proteins * k_prot
        ) * compound.fu
    return pd.Series(results, name="pk_sim_standard")


def blood_to_plasma_ws(compound: CompoundProperties, hct: float = HCT) -> float:
    """The PK-Sim Standard blood:plasma ratio estimate.

    ``(f_water_rbc + f_lipids_rbc*10^LogP + f_proteins_rbc*KProt)*fu*HCT
    - HCT + 1`` (dump line 3313).
    """
    k_prot = _kprot(compound.log_ma)
    return (
        BLOOD_CELLS_WS["f_water"]
        + BLOOD_CELLS_WS["f_lipids"] * 10**compound.log_p
        + BLOOD_CELLS_WS["f_proteins"] * k_prot
    ) * compound.fu * hct - hct + 1


_KP_METHODS = {
    "rodgers_rowland": rodgers_rowland_kp,
    "poulin_theil": poulin_theil_kp,
    "berezhkovskiy": berezhkovskiy_kp,
    "schmitt": schmitt_kp,
    "pk_sim_standard": pk_sim_standard_kp,
}


def kp_table(
    compound: CompoundProperties,
    method: str = "rodgers_rowland",
) -> pd.Series:
    """K_cell_pls for every organ with one method (name -> coefficients).

    Examples
    --------
    >>> compound = CompoundProperties(log_p=2.0, log_ma=1.5, fu=0.5)
    >>> table = kp_table(compound, method="pk_sim_standard")
    >>> 0.4 < float(table["Liver"]) < 2.0
    True
    """
    if method not in _KP_METHODS:
        raise DataValidationError(
            f"unknown method {method!r}; choose from {sorted(_KP_METHODS)}"
        )
    result = _KP_METHODS[method](compound)
    if isinstance(result, float):
        raise DataValidationError("internal error: scalar from a table method")
    return result


# ---------------------------------------------------------------------------
# Vss (the VSSCalculator.cs closed form)
# ---------------------------------------------------------------------------


def vss_physchem(
    kp_cell: pd.Series | Mapping[str, float],
    individual: dict[str, Any] | None = None,
) -> float:
    """Physicochemical Vss (L/kg) without any simulation.

    ``vss = [sum_tissue (K_cell_pls + f_vas*(1-HCT))*V_organ +
    sum_vascular V*(1-HCT)] / BW`` — ``PKSim.Core/Services/
    VSSCalculator.cs`` lines 100-107, evaluated on the standard
    individual.

    Examples
    --------
    >>> compound = CompoundProperties(log_p=2.0, log_ma=1.5, fu=0.5)
    >>> vss = vss_physchem(kp_table(compound, method="pk_sim_standard"))
    >>> 1.0 < vss < 30.0
    True
    """
    indiv = individual if individual is not None else STANDARD_INDIVIDUAL
    volumes = indiv["volumes"]
    f_vas = indiv["fraction_vascular"]
    hct = float(indiv["hct"])
    bw = float(indiv["body_weight"])
    vss = 0.0
    for organ, volume in volumes.items():
        if organ in {"ArterialBlood", "VenousBlood", "PortalVein"}:
            vss += volume * (1 - hct)
            continue
        if organ not in kp_cell:
            raise DataValidationError(f"Kp table missing organ {organ!r}")
        vss += (float(kp_cell[organ]) + f_vas[organ] * (1 - hct)) * volume
    return vss / bw


# ---------------------------------------------------------------------------
# ontogeny / maturation
# ---------------------------------------------------------------------------


def ontogeny_factor(
    name: str,
    age_years: float,
    ga_weeks: float = 40.0,
    container: str = "liver",
    percentile: float | None = None,
) -> float:
    """Ontogeny factor of an enzyme or plasma protein at a given age.

    ``PMA = age + GA_weeks*7/365.25`` then linear interpolation of the
    factor vs PMA (``OntogenyRepository.OntogenyFactorFor``).  With
    ``percentile`` the correlated-variability rule applies: one
    percentile held fixed across the whole table, i.e.
    ``LogNormal(log(factor), log(geoSD)).ppf(percentile)`` — the PK-Sim
    "one percentile per subject" aging correlation.

    Examples
    --------
    >>> round(float(ontogeny_factor("CYP3A4", age_years=0.5)), 4)   # PMA 1.27 y
    0.4139
    >>> bool(abs(ontogeny_factor("CYP2D6", age_years=2.61) - 1.0) < 1e-12)
    True
    """
    tables = ONTOGENY_TABLES.get(name.upper())
    if not tables or container not in tables:
        available = sorted(
            key for table in ONTOGENY_TABLES.values() for key in table
        )
        raise DataValidationError(
            f"no ontogeny table for {name!r} ({container!r}); "
            f"available containers: {available}"
        )
    rows = tables[container]
    pma = age_years + ga_weeks * 7.0 / 365.25
    pmas = np.array([row[0] for row in rows])
    factors = np.array([row[1] for row in rows])
    value = float(np.interp(pma, pmas, factors))
    if percentile is not None:
        if not 0 < percentile < 1:
            raise DataValidationError("percentile must be in (0, 1)")
        # geometric SD interpolated alongside, one percentile for the table
        gsds = np.array([row[2] for row in rows])
        gsd = float(np.interp(pma, pmas, gsds))
        value = math.exp(math.log(value) + statistics_ppf(percentile) * math.log(gsd))
    return value


def statistics_ppf(p: float) -> float:
    """Standard-normal quantile (inverse CDF) via :func:`scipy.special.ndtri`."""
    return float(ndtri(p))


def gfr_maturation_factor(
    pma_years: float,
    tm50_days: float = GFR_CONSTANTS["tm50_days"],
    hill: float = GFR_CONSTANTS["hill"],
    f_premat: float = GFR_CONSTANTS["f_premat"],
) -> float:
    """The GFR maturation Hill factor (dump line 3435, human constants).

    ``((PMA*365.25/7)^Hill/(TM50^Hill+(PMA*365.25/7)^Hill)*(1-fGFRpremat)
    + fGFRpremat)`` — 0.258 at conception-age zero, ~1 after ~3 months
    post-partum (TM50 44.4 days, Hill 15).

    Examples
    --------
    >>> round(float(gfr_maturation_factor(pma_years=0.1)), 3)  # ~3.5 weeks PMA
    0.258
    >>> round(float(gfr_maturation_factor(pma_years=3.0)), 3)
    1.0
    """
    post_conception_weeks = pma_years * 365.25 / 7.0
    ratio = post_conception_weeks**hill / (tm50_days**hill + post_conception_weeks**hill)
    return ratio * (1 - f_premat) + f_premat


def renal_aging_factor(age_years: float, gender: str = "male") -> float:
    """The geriatric renal decline factor (dump line 3812).

    1 before the age of onset (30 y), then
    ``1 - 0.9*(age-30)^1.5/(TA50^1.5 + (age-30)^1.5)`` with TA50 = 54 y
    (male) / 59 y (female).

    Examples
    --------
    >>> round(float(renal_aging_factor(30.0)), 3)
    1.0
    >>> 0.5 < float(renal_aging_factor(85.0)) < 0.75
    True
    """
    if gender not in {"male", "female"}:
        raise DataValidationError("gender must be 'male' or 'female'")
    onset = GFR_CONSTANTS["age_onset"]
    if age_years <= onset:
        return 1.0
    ta50 = (
        GFR_CONSTANTS["aging_half_time_male"]
        if gender == "male"
        else GFR_CONSTANTS["aging_half_time_female"]
    )
    hill = GFR_CONSTANTS["aging_hill"]
    excess = age_years - onset
    decline = (
        GFR_CONSTANTS["max_decrease"]
        * excess**hill
        / (ta50**hill + excess**hill)
    )
    return 1.0 - decline


def fu_plasma_scaled(
    fu_ref: float,
    age_years: float,
    binding_partner: str = "albumin",
    ga_weeks: float = 40.0,
    ckd_stage: int | None = None,
) -> float:
    """Age- and disease-scaled plasma unbound fraction.

    The PK-Sim specific-binding equation (dump line 3314):
    ``fu = min(1/(1 + ontogeny*(1-fu_ref)/fu_ref) * scale, 1)`` with the
    albumin/AGP ontogeny factor and the CKD
    ``PlasmaProteinScaleFactor`` (1.07 / 1.16 / 1.55 for stages 3/4/5).

    Examples
    --------
    >>> fu = fu_plasma_scaled(fu_ref=0.05, age_years=30.0)
    >>> round(float(fu), 4)
    0.05
    >>> fu_baby = fu_plasma_scaled(fu_ref=0.05, age_years=0.5)
    >>> fu_baby > fu        # hypoalbuminemia of infancy frees more drug
    True
    """
    partner = binding_partner.lower()
    if partner not in {"albumin", "agp"}:
        raise DataValidationError("binding_partner must be 'albumin' or 'agp'")
    name = "ALB" if partner == "albumin" else "AGP"
    onto = ontogeny_factor(name, age_years, ga_weeks=ga_weeks, container="plasma")
    scale = 1.0
    if ckd_stage is not None:
        if ckd_stage not in CKD_STAGE_FACTORS:
            raise DataValidationError(f"CKD stage must be one of {sorted(CKD_STAGE_FACTORS)}")
        scale = CKD_STAGE_FACTORS[ckd_stage]
    value = 1 / (1 + onto * (1 - fu_ref) / fu_ref) * scale
    return min(value, 1.0)

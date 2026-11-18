"""PBPK compound parameter sets: digoxin and the itraconazole chain.

Transcribed from the Open-Systems-Pharmacology model snapshots
``Digoxin-Model.json`` (PK-Sim snapshot v79, Hanke et al. 2018 model)
and ``Itraconazole-Model.json`` (snapshot v81) — every number below is
the value in the released model, with its value-origin (fitted /
literature) kept in the comments.  These are *reference parameter
sets* for the lightweight PBPK engine in :mod:`pharmaml.pkpd.pbpk`;
the full differential-equation networks (4-compartment organ
structure, Rodgers & Rowland Kp calculation methods, bile-salt
micellisation, ontogenies) belong to PK-Sim/OSPSuite and are not
re-implemented — see that module's honest boundary note.

Examples
--------
>>> from pharmaml.pkpd.compounds import (
...     DIGOXIN_PARAMS, ITRACONZOLE_COMPOUNDS, P_GP_PROFILE,
...     CYP3A4_PROFILE, ITRACONZOLE_WEIBULL, enzyme_turnover_rate)

Digoxin: MW 780.93 g/mol, fu 71 %, P-gp Km 177 µmol/L:

>>> DIGOXIN_PARAMS["mw_g_mol"]
780.93
>>> DIGOXIN_PARAMS["fraction_unbound_plasma"]
0.71
>>> DIGOXIN_PARAMS["pgp_km_umol_l"]
177.0

The itraconazole chain — four compounds, sequential CYP3A4 metabolism:

>>> [c["label"] for c in ITRACONZOLE_COMPOUNDS]
['Itraconazole', 'Hydroxy-Itraconazole', 'Keto-Itraconazole', 'N-desalkyl-Itraconazole']
>>> ITRACONZOLE_COMPOUNDS[0]["ki_cyp3a4_nmol_l"]       # the fitted Ki
1.3
>>> ITRACONZOLE_COMPOUNDS[0]["ki_pgp_umol_l"]          # 8 nmol/L (Shityakov 2014)
0.008

P-gp expression: small-intestinal mucosa is the maximum, kidney at
0.709 of it (the RT-PCR profile, mucosa boosted 3.57-fold):

>>> P_GP_PROFILE["reference_concentration_umol_l"]
1.41
>>> round(P_GP_PROFILE["relative_expression"]["SmallIntestine_mucosa"], 6), round(
...     P_GP_PROFILE["relative_expression"]["Kidney"], 6)
(1.0, 0.70922)

CYP3A4 turnover: t½ 23 h in gut / 36 h in liver → krec = ln2/t½:

>>> round(enzyme_turnover_rate(23.0), 6), round(enzyme_turnover_rate(36.0), 6)
(0.030137, 0.019254)

The capsule Weibull formulations (fasted vs fed):

>>> (round(ITRACONZOLE_WEIBULL["capsule_fasted"]["t50_min"], 4),
...  round(ITRACONZOLE_WEIBULL["capsule_fasted"]["shape"], 4))
(406.3002, 1.4298)
"""

from __future__ import annotations

import math

from pharmaml.utils.validation import validate_positive

__all__ = [
    "ATP1A2_PROFILE",
    "CYP3A4_PROFILE",
    "DIGOXIN_PARAMS",
    "ITRACONZOLE_COMPOUNDS",
    "ITRACONZOLE_WEIBULL",
    "P_GP_PROFILE",
    "enzyme_turnover_rate",
]

# --- Digoxin (Digoxin-Model.json, PK-Sim snapshot v79) --------------------
# Value origins: lipophilicity / intestinal+organ permeability / liver
# specific clearance / P-gp kcat fitted by parameter identification;
# fu (Neuhoff 2013), Kd+koff (Katz 2010), Km (Troutman 2003 via
# Collett 2004), GFR fraction 1.0 (Steiness 1982).
DIGOXIN_PARAMS: dict[str, float | str] = {
    "molecule": "Digoxin",
    "mw_g_mol": 780.93,                     # DrugBank
    "lipophilicity_log_units": 1.40017663,  # fitted
    "fraction_unbound_plasma": 0.71,        # Neuhoff 2013
    "solubility_mg_l": 64.8,                # at reference pH 7.0
    "specific_intestinal_permeability_dm_min": 2.7627e-07,   # fitted
    "specific_organ_permeability_dm_min": 1.0115e-05,        # fitted
    "plasma_protein_partner": "Albumin",
    # Specific binding to ATP1A2 (Na/K-ATPase) — the receptor
    "atp1a2_kd_nmol_l": 25.6,               # fitted (lit. 0.0256 umol/L, Katz 2010)
    "atp1a2_koff_per_min": 0.00098888,      # Katz 2010
    # Hepatic clearance (pseudo-enzyme "Undefined Liver")
    "liver_specific_clearance_per_min": 0.03758077,           # fitted
    # Active transport: P-gp efflux (Michaelis-Menten)
    "pgp_vmax_umol_l_min": 8.67,
    "pgp_km_umol_l": 177.0,                 # lit. 73/177 (Collett 2004 / Troutman 2003)
    "pgp_kcat_per_min": 71.163,             # fitted
    # Renal: glomerular filtration only
    "gfr_fraction": 1.0,                    # Steiness 1982
}

# --- Itraconazole chain (Itraconazole-Model.json, snapshot v81) -----------
# Hanke et al. 2018 (CPT PSP 7:647-659): sequential CYP3A4 metabolism
# ITZ -> Hydroxy -> Keto -> N-desalkyl, all four competitive CYP3A4
# inhibitors; ITZ also inhibits P-gp, Hydroxy also OATP1B1/1B3.
ITRACONZOLE_COMPOUNDS: list[dict[str, float | str]] = [
    {
        "label": "Itraconazole",
        "mw_g_mol": 705.633,
        "lipophilicity_log_units": 4.624,                 # fitted
        "fraction_unbound_plasma": 0.6016197247 / 100.0,  # fitted (% -> fraction)
        "pka_base": 3.7,
        "intestinal_permeability_dm_min": 5.3261558344e-05,  # fitted
        # CYP3A4 metabolism ITZ -> Hydroxy-ITZ (Isoherranen 2004)
        "vmax_cyp3a4": 0.27,                # pmol/min/pmol enzyme
        "km_nmol_l": 2.0688492598,
        "kcat_per_min": 0.0402937875,
        # Inhibition constants
        "ki_cyp3a4_nmol_l": 1.3,            # fitted (Isoherranen 2004)
        "ki_pgp_umol_l": 0.008,             # Shityakov 2014 (8 nmol/L)
    },
    {
        "label": "Hydroxy-Itraconazole",
        "mw_g_mol": 721.633,
        "lipophilicity_log_units": 3.718,                 # fitted
        "fraction_unbound_plasma": 0.017,                 # fixed (Riccardi)
        "pka_base": 3.7,                                   # assumed
        "vmax_cyp3a4": 0.543,              # nmol/min/pmol -> Hydroxy->Keto
        "km_nmol_l": 4.1716224833,
        "kcat_per_min": 0.0203370845,
        "ki_cyp3a4_nmol_l": 14.4,
        "ki_oatp1b1_umol_l": 0.0177818488,  # fitted (Tuerk 2019)
        "ki_oatp1b3_umol_l": 0.0111606334,  # fitted (Tuerk 2019)
    },
    {
        "label": "Keto-Itraconazole",
        "mw_g_mol": 719.617,
        "lipophilicity_log_units": 4.2109086248,
        "fraction_unbound_plasma": 0.010,                 # fixed
        "pka_base": 3.7,                                   # assumed
        "vmax_cyp3a4": 0.0869,             # pmol/min/pmol -> N-desalkyl
        "km_nmol_l": 2.2214874285,
        "kcat_per_min": 0.3933927416,
        # Ki via Cheng-Prusoff from IC50 7.0 nmol/L (midazolam S=1, Km=2.73)
        "ki_cyp3a4_nmol_l": 5.12,
    },
    {
        "label": "N-desalkyl-Itraconazole",
        "mw_g_mol": 649.527,
        "lipophilicity_log_units": 5.1837535822,
        "fraction_unbound_plasma": 0.011,                 # fixed
        "pka_base": 3.7,                                   # assumed
        "vmax_cyp3a4": 0.0,                 # further metabolism disabled
        "km_nmol_l": 0.6284266369,
        "kcat_per_min": 0.0605873508,
        # Ki via Cheng-Prusoff from IC50 0.44 nmol/L
        "ki_cyp3a4_nmol_l": 0.32,
    },
]

# --- Protein expression profiles (shared digoxin/itraconazole setup) -----
# Reference concentrations and turnover half-lives are identical in both
# model snapshots; relative expressions are x-max normalised.
CYP3A4_PROFILE: dict = {
    "reference_concentration_umol_l": 4.32,   # liver maximum
    "t_half_intestine_h": 23.0,
    "t_half_liver_h": 36.0,
    "relative_expression": {
        "Liver": 1.0,
        "SmallIntestine_mucosa": 0.0727697702,
        "Kidney": 0.0053603428126,
        "Brain": 0.0041682898325,
        "Lung": 0.00042695753798,
        "Gonads": 0.00078691079081,
    },
}

P_GP_PROFILE: dict = {
    "reference_concentration_umol_l": 1.41,   # small-intestinal mucosa max
    "t_half_intestine_h": 23.0,
    "t_half_liver_h": 36.0,
    # RT-PCR profile (Nishimura 2005) with the intestinal mucosa
    # expression increased 3.57-fold during optimisation.
    "relative_expression": {
        "SmallIntestine_mucosa": 1.0,
        "Kidney": 0.7092198582,
        "LargeIntestine_mucosa": 0.3971631206,
        "Liver": 0.1916810428,
        "SmallIntestine_whole": 0.2808543974,
        "LargeIntestine_whole": 0.1108416465,
        "Brain": 0.07608904,
        "Spleen": 0.0742555691,
        "Lung": 0.0657549316,
        "Heart": 0.0419198106,
        "Stomach": 0.0260852897,
        "Bone": 0.0242518189,
        "Pancreas": 0.0142510689,
        "Muscle": 0.0128342959,
        "Gonads": 0.017417973,
    },
}

ATP1A2_PROFILE: dict = {
    "reference_concentration_umol_l": 0.39140774,  # brain maximum
    "t_half_intestine_h": 23.0,
    "t_half_liver_h": 36.0,
    "relative_expression": {
        "Brain": 1.0,
        "Heart": 0.3179,
        "Gonads": 0.0509,
        "Kidney": 0.0159,
    },
}

# --- Formulations (Itraconazole capsules, Weibull dissolution) ------------
ITRACONZOLE_WEIBULL: dict[str, dict[str, float]] = {
    # t50 = time to 50 % dissolved; both fitted 2019-05-15.
    "capsule_fasted": {"t50_min": 406.3001802552, "shape": 1.4297720052},
    "capsule_fed": {"t50_min": 138.95, "shape": 0.82},
}


def enzyme_turnover_rate(t_half_h: float) -> float:
    """First-order degradation rate ``k = ln 2 / t½`` (per hour).

    The protein-synthesis turnover constant behind OSP's
    mechanism-based inactivation / induction: with the shared t½ of
    23 h (intestine) and 36 h (liver), ``k_rec`` is 0.0301 and
    0.0193 1/h — the timescale on which inhibited or induced enzyme
    levels recover after the perpetrator leaves.
    """
    t_half_h = validate_positive(t_half_h, "t_half_h")
    return math.log(2.0) / t_half_h

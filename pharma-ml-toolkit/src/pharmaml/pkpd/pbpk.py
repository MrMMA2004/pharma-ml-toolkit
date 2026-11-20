"""Minimal PBPK: five compartments, IVIVE, well-stirred liver, DDI math.

A numpy-native absorption of the pharmacometrics-portfolio's
``python/pbpk_helpers.py`` (the MinimalPBPK engine) plus the
translational and DDI mathematics its projects exercise:

* :class:`MinimalPBPK` — the five-compartment minimal PBPK model
  (venous blood → lung → {liver, kidney, rest}) with ICRP 2002
  reference physiology, integrated by the toolkit's own
  :func:`pharmaml.pkpd.ode.solve_ode` engine instead of SciPy's
  LSODA;
* :func:`ivive_clint_invivo` — the microsomal intrinsic-clearance
  scaling ``CLint,invivo = CLint · MPPGL · V_liver · 60/1e6``;
* :func:`well_stirred_clearance` — the well-stirred hepatic model
  ``CLh = QH·fuB·CLint/(QH + fuB·CLint)`` with ``fuB = fu/BP``;
* :func:`hepatic_ddi_auc_ratio` — the competitive-inhibition AUC
  ratio through the well-stirred model (``CLint/(1 + I/Ki)``), the
  closed-form companion of :func:`pharmaml.pkpd.ddi`'s machinery;
* :func:`allometric_scale` — volumes × BW, flows × BW^0.74,
  CLint × BW^0.75;
* :func:`weibull_dissolution` — PK-Sim's Weibull formulation
  (``F = 1 − exp(−ln2·(t/t50)^shape)`` so ``t50`` really is the 50 %
  dissolved time);
* :func:`mabel_dose` / :func:`noael_hed` / :func:`first_in_human_dose`
  — the preclinical-to-human translation chain (10 % receptor
  occupancy MABEL, the ``(70/BW)^0.25`` HED conversion, FIH = min).

Honest boundary: the partition coefficients Kp are *inputs* here (as
in the portfolio's engine).  PK-Sim derives them at runtime with the
Rodgers & Rowland method from logP/fu/pKa and its tissue-composition
tables — that calculation is not re-implemented; the ``CalculationMethods``
stayed in PK-Sim when this round was absorbed.

Examples
--------
>>> import numpy as np
>>> from pharmaml.pkpd.pbpk import (
...     ICRP_ADULT, MinimalPBPK, allometric_scale, first_in_human_dose,
...     hepatic_ddi_auc_ratio, ivive_clint_invivo, mabel_dose,
...     noael_hed, weibull_dissolution, well_stirred_clearance)

IVIVE: a 50 µL/min/mg microsomal CLint in a 70-kg ICRP liver
(0.026·70·1000 = 1820 g, MPPGL 40 mg/g) is 218.4 L/hr in vivo:

>>> round(ivive_clint_invivo(clint_ul_min_mg=50.0, liver_weight_g=1820.0), 2)
218.4

Well-stirred hepatic clearance with QH = 90.75 L/hr (25 % of the ICRP
cardiac output 15·70^0.74 ≈ 348 L/hr), fu 0.10, BP 1.2:

>>> round(well_stirred_clearance(q_h=90.75, fu_b=0.10 / 1.2, clint=218.4), 2)
15.16

A perpetrator holding I/Ki = 10 at the enzyme collapses that to a
strong-inhibition AUC ratio (FDA ≥ 5):

>>> round(hepatic_ddi_auc_ratio(
...     q_h=90.75, fu_b=0.10 / 1.2, clint=218.4, inhibitor_conc=10.0, ki=1.0), 2)
9.33

The minimal PBPK model itself — one IV bolus of 70 mg (C0 =
dose / V_blood / BP = 70 / 5.53 / 1.2):

>>> params = MinimalPBPK.build_params(bw=70.0, clint=50.0, fu=0.10)
>>> model = MinimalPBPK(params)
>>> times = np.linspace(0.0, 48.0, 97)
>>> frame = model.solve_dataframe(times, dose_mg=70.0, route="iv")
>>> round(float(frame["C_plasma"].iloc[0]), 4)
10.5485

Allometric scaling to a 25-kg paediatric patient (flows scale by
``(25/70)^0.74``):

>>> scaled = allometric_scale(params, ref_bw=70.0, target_bw=25.0)
>>> (round(scaled["QC"], 1),
...  round(params["QC"] * (25.0 / 70.0) ** 0.74, 1) == round(scaled["QC"], 1))
(162.4, True)

PK-Sim's Weibull: exactly half dissolved at t50:

>>> round(weibull_dissolution(406.3001802552, t50=406.3001802552, shape=1.43), 6)
0.5

Translation chain — MABEL at 10 % occupancy of a 0.5 mg/L EC50
(fu 0.1, V 20 L), rat NOAEL 100 mg/kg at 0.25 kg, FIH = min:

>>> round(mabel_dose(ec50=0.5, fu_plasma=0.1, v_volume=20.0), 3)
11.111
>>> round(noael_hed(noael_mgkg=100.0, animal_bw=0.25), 1)
102.3
>>> round(first_in_human_dose(mabel=11.111, noael=102.3), 3)
11.111
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.pkpd.ode import solve_ode
from pharmaml.utils.validation import validate_positive

__all__ = [
    "ICRP_ADULT",
    "MinimalPBPK",
    "allometric_scale",
    "first_in_human_dose",
    "hepatic_ddi_auc_ratio",
    "ivive_clint_invivo",
    "mabel_dose",
    "noael_hed",
    "weibull_dissolution",
    "well_stirred_clearance",
]

# ICRP 2002 reference adult (70 kg) — transcribed from the portfolio's
# python/pbk_helpers.py.
ICRP_ADULT: dict[str, float] = {
    "fV_lung": 0.0076,
    "fV_liver": 0.0260,
    "fV_kidney": 0.0044,
    "fV_adipose": 0.0214,
    "fV_blood": 0.0790,
    "QC_coeff": 15.0,       # L/hr — cardiac output = 15 · BW^0.74
    "QC_exp": 0.74,
    "fQ_liver": 0.250,
    "fQ_kidney": 0.190,
    "fQ_adipose": 0.050,
}

MPPGL_DEFAULT = 40.0  # mg microsomal protein per gram liver


def ivive_clint_invivo(
    *,
    clint_ul_min_mg: float,
    liver_weight_g: float,
    mppgl: float = MPPGL_DEFAULT,
) -> float:
    """Scale a microsomal intrinsic clearance to in-vivo L/hr.

    ``CLint_invivo = CLint [µL/min/mg] · MPPGL [mg/g] · V_liver [g] ·
    60/1e6 [L/hr per µL/min]`` — the standard IVIVE step (the
    portfolio's formula, MPPGL default 40 mg/g).
    """
    clint_ul_min_mg = validate_positive(clint_ul_min_mg, "clint_ul_min_mg")
    liver_weight_g = validate_positive(liver_weight_g, "liver_weight_g")
    mppgl = validate_positive(mppgl, "mppgl")
    return clint_ul_min_mg * mppgl * liver_weight_g * 60.0 / 1e6


def well_stirred_clearance(*, q_h: float, fu_b: float, clint: float) -> float:
    """Well-stirred hepatic clearance ``QH·fuB·CLint/(QH + fuB·CLint)``.

    ``fu_b`` is the free fraction in *blood* (``fu_plasma / BP``).
    """
    q_h = validate_positive(q_h, "q_h")
    fu_b = validate_positive(fu_b, "fu_b")
    clint = validate_positive(clint, "clint")
    return q_h * fu_b * clint / (q_h + fu_b * clint)


def hepatic_ddi_auc_ratio(
    *,
    q_h: float,
    fu_b: float,
    clint: float,
    inhibitor_conc: float,
    ki: float,
) -> float:
    """AUC ratio of a hepatically cleared victim under inhibition.

    Competitive inhibition divides the intrinsic clearance by
    ``1 + I/Ki``; for a victim whose only clearance is the well-stirred
    liver, ``AUCR = CLh(control) / CLh(inhibited)`` — the closed form
    the DDI qualification workflow's simulations are judged against.
    """
    clint_inhibited = clint / (1.0 + validate_positive(inhibitor_conc, "inhibitor_conc")
                               / validate_positive(ki, "ki"))
    control = well_stirred_clearance(q_h=q_h, fu_b=fu_b, clint=clint)
    inhibited = well_stirred_clearance(q_h=q_h, fu_b=fu_b, clint=clint_inhibited)
    return control / inhibited


def allometric_scale(
    params: dict,
    ref_bw: float,
    target_bw: float,
    *,
    clint_exponent: float = 0.75,
) -> dict:
    """Scale physiology from a reference body weight to a target.

    Volumes × ``BW`` (exponent 1), blood flows × ``BW^0.74`` (matching
    the cardiac-output allometry), CLint × ``BW^0.75`` (liver-mass
    scaling); drug properties (Kp, fu, BP) are untouched — they belong
    to the compound, not the organism.
    """
    ref_bw = validate_positive(ref_bw, "ref_bw")
    target_bw = validate_positive(target_bw, "target_bw")
    ratio = target_bw / ref_bw
    scaled = dict(params)
    for key in ("V_lung", "V_liver", "V_kidney", "V_adipose", "V_blood", "V_rest"):
        if key in scaled:
            scaled[key] = params[key] * ratio
    for key in ("QC", "Q_liver", "Q_kidney", "Q_adipose", "Q_rest"):
        if key in scaled:
            scaled[key] = params[key] * ratio**ICRP_ADULT["QC_exp"]
    if "CLint" in scaled:
        scaled["CLint"] = params["CLint"] * ratio**clint_exponent
    scaled["BW"] = target_bw
    return scaled


def weibull_dissolution(
    t: float | np.ndarray, *, t50: float, shape: float
) -> float | np.ndarray:
    """PK-Sim's Weibull dissolution fraction.

    ``F(t) = 1 − exp(−ln2 · (t/t50)^shape)`` — parameterised so that
    ``t50`` is by construction the 50 % dissolved time (the
    "Dissolution time (50 % dissolved)" knob of PK-Sim's Weibull
    formulation; the itraconazole capsule's fitted values live in
    :mod:`pharmaml.pkpd.compounds`).
    """
    t50 = validate_positive(t50, "t50")
    shape = validate_positive(shape, "shape")
    times = np.asarray(t, dtype=float)
    values = 1.0 - np.exp(-math.log(2.0) * (times / t50) ** shape)
    if values.ndim == 0:
        return float(values)
    return values


class MinimalPBPK:
    """Five-compartment minimal PBPK model (portfolio-exact structure).

    Compartments (state = *concentration* in each):

    ============  ==========================================
    index 0       venous blood (the sampling compartment)
    index 1       lung (arterialises the blood)
    index 2       liver (well-stirred elimination)
    index 3       kidney (distribution only)
    index 4       rest (all remaining tissues lumped)
    ============  ==========================================

    ``params`` comes from :meth:`build_params`; the ODE right-hand
    side is the portfolio's transcription (arterial concentration =
    lung concentration / Kp_lung; hepatic extraction through the
    well-stirred CLh).  Integration rides the toolkit's own adaptive
    RK45 (:func:`pharmaml.pkpd.ode.solve_ode`).
    """

    COMPARTMENTS = ("venous", "lung", "liver", "kidney", "rest")

    def __init__(self, params: dict) -> None:
        self.params = params
        required = (
            "V_blood", "V_lung", "V_liver", "V_kidney", "V_rest",
            "QC", "Q_liver", "Q_kidney", "Q_rest",
            "Kp_lung", "Kp_liver", "Kp_kidney", "Kp_rest",
            "CLint", "fu", "BP",
        )
        missing = [key for key in required if key not in params]
        if missing:
            raise DataValidationError(f"MinimalPBPK: missing parameters: {missing}")

    @staticmethod
    def build_params(
        *,
        bw: float = 70.0,
        clint: float = 50.0,
        fu: float = 0.10,
        bp: float = 1.20,
        kp_lung: float = 2.0,
        kp_liver: float = 5.0,
        kp_kidney: float = 3.0,
        kp_rest: float = 1.5,
        mppgl: float = MPPGL_DEFAULT,
        icrp: dict[str, float] = ICRP_ADULT,
    ) -> dict:
        """Parameter dictionary for a body weight from ICRP fractions.

        ``clint`` is the *microsomal* intrinsic clearance in
        µL/min/mg protein; it is scaled in vivo through
        :func:`ivive_clint_invivo` with ``mppgl``.
        """
        bw = validate_positive(bw, "bw")
        qc = icrp["QC_coeff"] * bw ** icrp["QC_exp"]
        q_liver = icrp["fQ_liver"] * qc
        q_kidney = icrp["fQ_kidney"] * qc
        q_adipose = icrp["fQ_adipose"] * qc
        v_liver_l = icrp["fV_liver"] * bw
        clint_invivo = ivive_clint_invivo(
            clint_ul_min_mg=clint,
            liver_weight_g=v_liver_l * 1000.0,
            mppgl=mppgl,
        )
        return {
            "BW": bw,
            "QC": qc,
            "Q_liver": q_liver,
            "Q_kidney": q_kidney,
            "Q_adipose": q_adipose,
            "Q_rest": qc - q_liver - q_kidney - q_adipose,
            "V_blood": icrp["fV_blood"] * bw,
            "V_lung": icrp["fV_lung"] * bw,
            "V_liver": v_liver_l,
            "V_kidney": icrp["fV_kidney"] * bw,
            "V_adipose": icrp["fV_adipose"] * bw,
            "V_rest": bw
            * (
                1.0
                - sum(
                    icrp[key]
                    for key in (
                        "fV_lung", "fV_liver", "fV_kidney", "fV_adipose", "fV_blood",
                    )
                )
            ),
            "Kp_lung": kp_lung,
            "Kp_liver": kp_liver,
            "Kp_kidney": kp_kidney,
            "Kp_rest": kp_rest,
            "CLint": clint_invivo,
            "fu": fu,
            "BP": bp,
        }

    def odes(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side (portfolio transcription, concentrations)."""
        p = self.params
        c_ven, c_lung, c_liver, c_kidney, c_rest = y
        c_art = c_lung / p["Kp_lung"]
        fu_b = p["fu"] / p["BP"]
        cl_h = well_stirred_clearance(
            q_h=p["Q_liver"], fu_b=fu_b, clint=p["CLint"]
        )
        d_ven = (
            p["Q_liver"] * c_liver / p["Kp_liver"]
            + p["Q_kidney"] * c_kidney / p["Kp_kidney"]
            + p["Q_rest"] * c_rest / p["Kp_rest"]
            - p["QC"] * c_ven
        ) / p["V_blood"]
        d_lung = (p["QC"] * c_ven - p["QC"] * c_lung / p["Kp_lung"]) / p["V_lung"]
        d_liver = (
            p["Q_liver"] * c_art
            - p["Q_liver"] * c_liver / p["Kp_liver"]
            - cl_h * c_liver / p["Kp_liver"]
        ) / p["V_liver"]
        d_kidney = (
            p["Q_kidney"] * c_art - p["Q_kidney"] * c_kidney / p["Kp_kidney"]
        ) / p["V_kidney"]
        d_rest = (p["Q_rest"] * c_art - p["Q_rest"] * c_rest / p["Kp_rest"]) / p["V_rest"]
        return np.array([d_ven, d_lung, d_liver, d_kidney, d_rest])

    def solve(
        self,
        times: np.ndarray | list[float],
        dose_mg: float,
        *,
        route: str = "iv",
        ka: float = 1.0,
        f: float = 0.85,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> np.ndarray:
        """Solve and return the five-compartment state matrix (rows = times).

        IV: the dose lands in the venous pool at the first time.  Oral:
        a depot compartment is prepended (``ka`` first-order, ``f``
        bioavailability) and stripped from the returned matrix — the
        portfolio's structure, on the toolkit's own integrator.
        """
        if route not in {"iv", "oral"}:
            raise DataValidationError(f"route must be 'iv' or 'oral', got {route!r}")
        dose_mg = validate_positive(dose_mg, "dose_mg")
        t = np.asarray(times, dtype=float).reshape(-1)
        p = self.params
        if route == "iv":
            y0 = np.array([dose_mg / p["V_blood"], 0.0, 0.0, 0.0, 0.0])
            sol = solve_ode(self.odes, t, y0, rtol=rtol, atol=atol)
            return sol.states
        # Oral: depot amount state + the five PBPK concentrations.
        def oral_rhs(_t: float, y: np.ndarray) -> np.ndarray:
            depot = y[0]
            state = y[1:]
            dy = np.empty(6)
            dy[0] = -ka * depot
            pbpk = self.odes(_t, state)
            pbpk[0] = pbpk[0] + ka * depot * f / p["V_blood"]
            dy[1:] = pbpk
            return dy

        y0 = np.array([dose_mg, 0.0, 0.0, 0.0, 0.0, 0.0])
        sol = solve_ode(oral_rhs, t, y0, rtol=rtol, atol=atol)
        return sol.states[:, 1:]

    def solve_dataframe(
        self,
        times: np.ndarray | list[float],
        dose_mg: float,
        *,
        route: str = "iv",
        ka: float = 1.0,
        f: float = 0.85,
    ) -> pd.DataFrame:
        """Tidy frame: time, the five compartments, and C_plasma."""
        states = self.solve(times, dose_mg, route=route, ka=ka, f=f)
        frame = pd.DataFrame(states, columns=list(self.COMPARTMENTS))
        frame.insert(0, "time", np.asarray(times, dtype=float).reshape(-1))
        frame["C_plasma"] = frame["venous"] / self.params["BP"]
        return frame


def mabel_dose(*, ec50: float, fu_plasma: float, v_volume: float) -> float:
    """Minimal-anticipated-biological-effect dose (10 % occupancy).

    Duplicate-symbol map (round 304, part12 A-6): ``mabel_dose`` also
    exists at :mod:`pharmaml.pharma.fih` (the canonical mg chain —
    the occupancy inversion, MW conversion, configurable ``target_ro``)
    and at :mod:`pharmaml.pharma.pbpk` (the mg-chain twin, math-parity
    with fih pinned by tests). THIS one is the unitless variant the
    capability registry references (``pkpd.pbpk`` — dose in the
    input's own units, no MW conversion); the family's definition
    sites and parity are pinned by
    ``tests/test_architecture_hygiene.py``.

    ``C_mabel_free = EC50 · 0.10/0.90``, total = free/fu, dose =
    total × V — the portfolio's translation, the standard
    first-in-human conservative starting dose.
    """
    ec50 = validate_positive(ec50, "ec50")
    fu_plasma = validate_positive(fu_plasma, "fu_plasma")
    v_volume = validate_positive(v_volume, "v_volume")
    c_free = ec50 * 0.10 / 0.90
    return c_free / fu_plasma * v_volume


def noael_hed(
    *,
    noael_mgkg: float,
    animal_bw: float,
    human_bw: float = 70.0,
) -> float:
    """NOAEL human-equivalent dose in mg (the ``(70/BW)^0.25`` form).

    Duplicate-symbol map (round 304, part12 A-6): ``noael_hed`` also
    exists at :mod:`pharmaml.pharma.fih` (the configurable-exponent
    canonical) and :mod:`pharmaml.pharma.pbpk` (its math-parity twin);
    THIS one is the unitless fixed-exponent variant of the
    registry-referenced ``pkpd.pbpk`` family — same formula at the
    shared defaults, pinned by ``tests/test_architecture_hygiene.py``.

    ``HED = NOAEL · BW_animal · (BW_human/BW_animal)^(1−0.75)`` —
    dose-per-body-area conversion by allometric exponent difference.
    """
    noael_mgkg = validate_positive(noael_mgkg, "noael_mgkg")
    animal_bw = validate_positive(animal_bw, "animal_bw")
    human_bw = validate_positive(human_bw, "human_bw")
    return noael_mgkg * animal_bw * (human_bw / animal_bw) ** 0.25


def first_in_human_dose(*, mabel: float, noael: float) -> float:
    """FIH starting dose = min(MABEL, NOAEL-HED), the safety rule."""
    mabel = validate_positive(mabel, "mabel")
    noael = validate_positive(noael, "noael")
    return min(mabel, noael)


@dataclass(frozen=True, slots=True)
class PBPKSolution:
    """Compatibility alias kept for readers of the portfolio engine."""

    times: np.ndarray
    states: np.ndarray

    def to_dataframe(self, bp: float) -> pd.DataFrame:
        """Frame with time, compartments and C_plasma = venous/BP."""
        frame = pd.DataFrame(self.states, columns=list(MinimalPBPK.COMPARTMENTS))
        frame.insert(0, "time", self.times)
        frame["C_plasma"] = frame["venous"] / bp
        return frame

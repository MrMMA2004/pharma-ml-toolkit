"""Maternal--fetal PK: the OSP pregnancy models' transferable core.

``Open-Systems-Pharmacology/Pregnancy-Models`` distributes 27-compartment
whole-body MoBi/PK-Sim projects, not a small Python library.  Its README and
``PassiveTransport_Pregnancy_with-fetal-fu.pkml`` were inspected directly.
The refined building block's key correction is explicit: maternal and fetal
plasma can have DIFFERENT fractions unbound, and placental passive diffusion
uses the unbound concentration on each side (the PKML formula
``P*SA*C_pls*fu - P*SA*fu*C_cell/K_cell_pls``).

This module preserves that semantics in a disclosed two-compartment model.
It does NOT claim parity with the 27-compartment physiology, nine pregnancy-
specific compartments, fetal liver, cotyledon perfusion model or released
compound fittings.  The 16 distributed compound names are an index, not
validated parameters for this lightweight model.  Users must supply volumes,
clearances and placental transfer clearance appropriate to their analysis.

For maternal/fetal amounts ``A_m, A_f`` (mg), volumes ``V_m, V_f`` (L),
placental clearance ``Q_p`` (L/h) and fractions unbound ``fu_m, fu_f``:

``transfer = Q_p * (fu_m*A_m/V_m - fu_f*A_f/V_f)``
``dA_m/dt = -CL_m*A_m/V_m - transfer``
``dA_f/dt = -CL_f*A_f/V_f + transfer``

The initial dose is a maternal IV bolus.  The existing adaptive Dormand--
Prince PK/PD solver integrates the system; no compiled OSP dependency,
network request or model download is needed.

Examples
--------
With elimination and transfer disabled, amount is conserved exactly:

>>> from pharmaml.pkpd.pregnancy import (
...     MaternalFetalParameters, simulate_maternal_fetal)
>>> params = MaternalFetalParameters(
...     maternal_volume_l=50.0, fetal_volume_l=3.0,
...     maternal_clearance_l_h=0.0, fetal_clearance_l_h=0.0,
...     placental_clearance_l_h=0.0)
>>> result = simulate_maternal_fetal(params, dose_mg=100.0, times=[0, 1, 2])
>>> result.frame["fetal_amount_mg"].max()
0.0
>>> result.frame["maternal_amount_mg"].tolist()
[100.0, 100.0, 100.0]

PK-Sim encodes fertilization age as an offset from age 30:

>>> from pharmaml.pkpd.pregnancy import pk_sim_age_years, fertilization_age_weeks
>>> round(pk_sim_age_years(38.0), 2)
30.73
>>> round(fertilization_age_weeks(30.75), 1)
39.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "PREGNANCY_MODELS",
    "MaternalFetalParameters",
    "MaternalFetalSolution",
    "fertilization_age_weeks",
    "pk_sim_age_years",
    "pregnancy_model_frame",
    "simulate_maternal_fetal",
]

#: Compound model directories distributed by the upstream repository.
PREGNANCY_MODELS: tuple[str, ...] = (
    "Acetaminophen",
    "Acyclovir",
    "Caffeine",
    "Cefazolin",
    "Cefradine",
    "Cefuroxime",
    "Diazepam",
    "Dolutegravir",
    "Emtricitabine",
    "Granisetron",
    "Metoprolol",
    "Metronidazole",
    "Midazolam",
    "Nifedipine",
    "Ondansetron",
    "Raltegravir",
)


def pregnancy_model_frame() -> pd.DataFrame:
    """Return the model-name index with an explicit lightweight-scope note."""
    return pd.DataFrame(
        {
            "compound": PREGNANCY_MODELS,
            "upstream": "Open-Systems-Pharmacology/Pregnancy-Models",
            "scope": "27-compartment source project; not a fitted lightweight parameter set",
        }
    )


def fertilization_age_weeks(age_years: float) -> float:
    """Decode PK-Sim's pregnancy age convention (30 years = FA zero)."""
    try:
        age = float(age_years)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("age_years must be numeric") from exc
    if not np.isfinite(age) or age < 30.0:
        raise DataValidationError("PK-Sim pregnancy age_years must be finite and >= 30")
    return (age - 30.0) * 52.0


def pk_sim_age_years(fertilization_age_weeks_value: float) -> float:
    """Encode fertilization age in weeks using PK-Sim's 52-week/year offset."""
    try:
        weeks = float(fertilization_age_weeks_value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("fertilization age must be numeric") from exc
    if not np.isfinite(weeks) or not 0.0 <= weeks <= 42.0:
        raise DataValidationError("fertilization age must be between 0 and 42 weeks")
    return 30.0 + weeks / 52.0


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be numeric") from exc
    valid = np.isfinite(number) and (number >= 0.0 if allow_zero else number > 0.0)
    if not valid:
        comparator = ">= 0" if allow_zero else "> 0"
        raise DataValidationError(f"{name} must be finite and {comparator}")
    return number


def _fraction(value: Any, name: str) -> float:
    number = _positive(value, name)
    if number > 1.0:
        raise DataValidationError(f"{name} must be in (0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class MaternalFetalParameters:
    """Inputs for the lightweight maternal/fetal amount ODE."""

    maternal_volume_l: float
    fetal_volume_l: float
    maternal_clearance_l_h: float
    fetal_clearance_l_h: float
    placental_clearance_l_h: float
    maternal_fu: float = 1.0
    fetal_fu: float = 1.0

    def validated(self) -> MaternalFetalParameters:
        return MaternalFetalParameters(
            maternal_volume_l=_positive(self.maternal_volume_l, "maternal_volume_l"),
            fetal_volume_l=_positive(self.fetal_volume_l, "fetal_volume_l"),
            maternal_clearance_l_h=_positive(
                self.maternal_clearance_l_h, "maternal_clearance_l_h", allow_zero=True
            ),
            fetal_clearance_l_h=_positive(
                self.fetal_clearance_l_h, "fetal_clearance_l_h", allow_zero=True
            ),
            placental_clearance_l_h=_positive(
                self.placental_clearance_l_h, "placental_clearance_l_h", allow_zero=True
            ),
            maternal_fu=_fraction(self.maternal_fu, "maternal_fu"),
            fetal_fu=_fraction(self.fetal_fu, "fetal_fu"),
        )


@dataclass(slots=True)
class MaternalFetalSolution:
    """Simulation frame and exposure summaries (total plasma concentrations)."""

    frame: pd.DataFrame
    fetal_maternal_auc_ratio: float
    dose_mg: float
    parameters: MaternalFetalParameters
    n_steps: int
    n_rejected: int


def simulate_maternal_fetal(
    parameters: MaternalFetalParameters,
    *,
    dose_mg: float,
    times: Any,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> MaternalFetalSolution:
    """Simulate one maternal IV bolus with bidirectional unbound transfer."""
    params = parameters.validated()
    dose = _positive(dose_mg, "dose_mg")
    grid = np.asarray(times, dtype=float).reshape(-1)
    if grid.size < 2 or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise DataValidationError("times must contain at least two strictly increasing finite values")
    if grid[0] != 0.0:
        raise DataValidationError("times must start at 0 for the maternal IV-bolus initial condition")
    rtol_value = _positive(rtol, "rtol")
    atol_value = _positive(atol, "atol")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        maternal, fetal = state
        maternal_concentration = maternal / params.maternal_volume_l
        fetal_concentration = fetal / params.fetal_volume_l
        transfer = params.placental_clearance_l_h * (
            params.maternal_fu * maternal_concentration
            - params.fetal_fu * fetal_concentration
        )
        return np.array(
            [
                -params.maternal_clearance_l_h * maternal_concentration - transfer,
                -params.fetal_clearance_l_h * fetal_concentration + transfer,
            ]
        )

    from pharmaml.pkpd.ode import solve_ode

    solved = solve_ode(
        rhs,
        grid,
        [dose, 0.0],
        method="rk45",
        rtol=rtol_value,
        atol=atol_value,
    )
    amounts = solved.states
    maternal_concentration = amounts[:, 0] / params.maternal_volume_l
    fetal_concentration = amounts[:, 1] / params.fetal_volume_l
    frame = pd.DataFrame(
        {
            "time_h": grid,
            "maternal_amount_mg": amounts[:, 0],
            "fetal_amount_mg": amounts[:, 1],
            "maternal_concentration_mg_l": maternal_concentration,
            "fetal_concentration_mg_l": fetal_concentration,
            "maternal_unbound_mg_l": params.maternal_fu * maternal_concentration,
            "fetal_unbound_mg_l": params.fetal_fu * fetal_concentration,
        }
    )
    widths = np.diff(grid)
    maternal_auc = float(
        np.sum((maternal_concentration[:-1] + maternal_concentration[1:]) * 0.5 * widths)
    )
    fetal_auc = float(
        np.sum((fetal_concentration[:-1] + fetal_concentration[1:]) * 0.5 * widths)
    )
    if maternal_auc <= 0.0:
        raise DataValidationError("maternal AUC is zero; exposure ratio is undefined")
    return MaternalFetalSolution(
        frame=frame,
        fetal_maternal_auc_ratio=fetal_auc / maternal_auc,
        dose_mg=dose,
        parameters=params,
        n_steps=solved.n_steps,
        n_rejected=solved.n_rejected,
    )

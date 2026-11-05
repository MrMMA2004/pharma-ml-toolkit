"""Nonlinear mixed-effects population PK -- the nlmixr2 concepts.

Absorption of ``gunjansubedi02/pkpd_nlmixr2_r_project`` (reviewed file
by file): the 164-line ``nlmixr2_pk_model.R`` with its
``one_cmt_oral`` function-UI -- the ``ini({})`` block (``tka <-
log(1.0)``, ``tcl <- log(5.0)``, ``tv <- log(50.0)``, ``eta.ka ~
0.16``, ``eta.cl ~ 0.09``, ``eta.v ~ 0.0625``, ``add.err <- 0.05``,
``prop.err <- 0.12``), the ``model({})`` block's ODEs (``d/dt(depot)
<- -ka*depot``, ``d/dt(center) <- ka*depot - (cl/v)*center``,
``cp <- center/v``), the allometric covariates (``cl <- exp(tcl +
eta.cl) * (WT/70)^0.75``, ``v <- exp(tv + eta.v) * (WT/70)``, no
covariate on ka), the combined residual ``cp ~ prop(prop.err) +
add(add.err)``, the ``est="focei"`` call with ``foceiControl(print =
5)``, the NONMEM-style reshaping (dose rows EVID=1/CMT=1 at TIME=0,
observations EVID=0/CMT=2, ``arrange(ID, TIME, desc(EVID))``), the
``vpc`` package call (``n = 200``, ``bins = "jenks"`` -- Fisher-Jenks
breaks implemented here, since the R script passed ``sim = NULL`` and
never actually simulated), and the closing comparison of FOCEI's
population CL against the Python Standard Two-Stage geometric mean.

The estimation engine is the toolkit's own reverse-mode autograd
(:mod:`pharmaml.ml.autograd`, Adam over every theta, log-Omega,
log-Sigma and per-subject eta jointly).  Honest boundaries:

* The objective is the **exact joint -2 log-likelihood** with every
  eta optimized jointly with the fixed effects (conditional
  estimation *without* the FO linearization FOCEI applies; the
  "interaction" half is honored -- the proportional error uses the
  individual prediction, as FOCEI's ``I`` does).  No Laplace
  second-order term, no covariance step, no ``print = 5`` iteration
  log of inner/outer cycles.
* Single-dose designs only (the source project's data is one 100 mg
  oral dose per subject); no steady-state, no infusion, no PD arm.
* Residual error: the combined additive + proportional model, both
  on the SD scale exactly as nlmixr2 reads ``add()``/``prop()``.
* BLQ censoring is not modeled (the source data shows floored 0.001
  values that the R script also ignores).

Examples
--------
>>> import math
>>> from pharmaml.pharma.nlme import NLMIXR2_INI, individual_params
>>> round(NLMIXR2_INI["tcl"], 6)                 # log(5.0), the R ini block
1.609438
>>> ka, cl, v = individual_params(0.0, math.log(5.0), math.log(50.0), 0.0, 0.0, 0.0, 70.0)
>>> (round(ka, 6), round(cl, 6), round(v, 6))    # the typical subject at 70 kg
(1.0, 5.0, 50.0)
>>> ka, cl, v = individual_params(0.0, math.log(5.0), math.log(50.0), 0.0, 0.0, 0.0, 140.0)
>>> round(cl, 4), round(v, 4)                    # allometry: 2^0.75 and 2^1.0
(8.409, 100.0)
>>> from pharmaml.pharma.nlme import one_cmt_oral_conc
>>> round(one_cmt_oral_conc(100.0, 1.0, 5.0, 50.0, 0.0), 9)   # nothing absorbed yet
0.0
>>> peak_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
>>> values = [one_cmt_oral_conc(100.0, 1.0, 5.0, 50.0, t) for t in peak_grid]
>>> max(range(len(values)), key=lambda i: values[i])   # Tmax = ln(10)/0.9 = 2.56 h
3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "ALLOMETRIC_CL_EXPONENT",
    "ALLOMETRIC_V_EXPONENT",
    "DEFAULT_SAMPLE_TIMES",
    "NLMIXR2_INI",
    "REFERENCE_WEIGHT_KG",
    "NLMEFit",
    "PopulationData",
    "build_nonmem_records",
    "compare_focei_sts",
    "demo_population",
    "eta_shrinkage",
    "fit_population",
    "individual_params",
    "jenks_breaks",
    "one_cmt_oral_conc",
    "one_cmt_oral_ode",
    "simulate_population",
    "standard_two_stage",
    "vpc_percentiles",
    "vpc_simulate",
]

#: The reference weight of the R model's allometry (kg).
REFERENCE_WEIGHT_KG = 70.0
#: ``cl <- exp(tcl + eta.cl) * (WT / 70) ^ 0.75`` -- clearance exponent.
ALLOMETRIC_CL_EXPONENT = 0.75
#: ``v <- exp(tv + eta.v) * (WT / 70)`` -- volume exponent (exactly 1).
ALLOMETRIC_V_EXPONENT = 1.0
#: The 10-sample grid of the source project's simulated Phase-1 data (h).
DEFAULT_SAMPLE_TIMES = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0)

#: The ``ini({})`` block of ``one_cmt_oral``, verbatim values: the
#: three fixed effects on the log scale (typical ka = 1.0 /h, CL =
#: 5.0 L/h, V = 50 L at 70 kg), the diagonal OMEGA variances
#: (``(0.4)^2``, ``(0.3)^2``, ``(0.25)^2``) and the combined residual
#: error SDs (additive 0.05 mg/L + proportional 12 %).
NLMIXR2_INI: dict[str, Any] = {
    "tka": math.log(1.0),
    "tcl": math.log(5.0),
    "tv": math.log(50.0),
    "omega_ka": 0.16,
    "omega_cl": 0.09,
    "omega_v": 0.0625,
    "add_err": 0.05,
    "prop_err": 0.12,
}


# --------------------------------------------------------------------- #
# the structural model
# --------------------------------------------------------------------- #

def individual_params(
    tka: float,
    tcl: float,
    tv: float,
    eta_ka: float,
    eta_cl: float,
    eta_v: float,
    weight_kg: float,
    reference_weight: float = REFERENCE_WEIGHT_KG,
) -> tuple[float, float, float]:
    """Individual (ka, CL, V) from the log-scale thetas, etas and WT.

    Exactly the R ``model({})`` block: log-normal between-subject
    variability on every parameter, allometric weight on CL (exponent
    0.75) and V (exponent 1.0) against the 70 kg reference, no
    covariate on ka.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import individual_params
    >>> import math
    >>> ka, cl, v = individual_params(math.log(1.0), math.log(5.0),
    ...                               math.log(50.0), 0.4, 0.3, 0.25, 70.0)
    >>> round(ka, 4), round(cl, 4), round(v, 4)   # exp(eta) multipliers
    (1.4918, 6.7493, 64.2013)
    """
    weight_kg = float(weight_kg)
    if weight_kg <= 0:
        raise DataValidationError(f"weight_kg must be positive, got {weight_kg}")
    factor = weight_kg / float(reference_weight)
    ka = math.exp(float(tka) + float(eta_ka))
    cl = math.exp(float(tcl) + float(eta_cl)) * factor**ALLOMETRIC_CL_EXPONENT
    v = math.exp(float(tv) + float(eta_v)) * factor**ALLOMETRIC_V_EXPONENT
    return ka, cl, v


def one_cmt_oral_conc(
    dose: float,
    ka: float,
    cl: float,
    v: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Central concentration of the 1-cmt first-order absorption model.

    The analytic solution of ``d/dt(depot) = -ka*depot``,
    ``d/dt(center) = ka*depot - (cl/v)*center`` after a single oral
    dose (what the R model integrates numerically with RxODE; the
    closed form is used here for speed and exact gradients)::

        C(t) = (Dose * ka) / (V * (ka - ke)) * (e^(-ke*t) - e^(-ka*t))

    with ``ke = cl/v``; the ``ka == ke`` flip-flop limit collapses to
    ``C(t) = (Dose/V) * ka * t * e^(-ka*t)``.  ``t`` may be scalar or
    array.

    Raises
    ------
    DataValidationError
        If any parameter is non-positive or ``t`` is negative.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import one_cmt_oral_conc
    >>> conc = one_cmt_oral_conc(100.0, 1.0, 5.0, 50.0, [0.25, 0.5, 1.0, 2.0])
    >>> [round(float(value), 4) for value in conc]
    [0.4367, 0.766, 1.1932, 1.5187]
    >>> flip = one_cmt_oral_conc(100.0, 0.2, 10.0, 50.0, 1.0)   # ka == ke
    >>> round(flip, 6)
    0.327492
    """
    dose = float(dose)
    ka = float(ka)
    cl = float(cl)
    v = float(v)
    if dose < 0:
        raise DataValidationError(f"dose must be non-negative, got {dose}")
    if ka <= 0 or cl <= 0 or v <= 0:
        raise DataValidationError(
            f"ka, cl and v must be positive (got ka={ka}, cl={cl}, v={v})"
        )
    times = np.asarray(t, dtype=float)
    if np.any(times < 0):
        raise DataValidationError("time t must be non-negative")
    ke = cl / v
    scalar = np.ndim(t) == 0
    grid = times.reshape(-1) if not scalar else np.asarray([float(times)])
    if abs(ka - ke) < 1e-12:
        conc = (dose / v) * ka * grid * np.exp(-ka * grid)
    else:
        conc = (dose * ka / (v * (ka - ke))) * (np.exp(-ke * grid) - np.exp(-ka * grid))
    if scalar:
        return float(conc[0])
    return conc.reshape(times.shape)


def one_cmt_oral_ode(
    dose: float,
    ka: float,
    cl: float,
    v: float,
    times: np.ndarray | list[float],
    step: float = 0.01,
) -> np.ndarray:
    """The same model integrated by RK4 (the RxODE route).

    The R project writes the structural model as ODEs and lets RxODE
    integrate them; this is the honest twin -- a classical
    fourth-order Runge-Kutta march over ``d(depot)/dt`` and
    ``d(center)/dt`` on a grid fine enough to hit every requested
    observation time exactly (the observation times are spliced into
    the integration grid, so no interpolation error exists).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.pharma.nlme import one_cmt_oral_conc, one_cmt_oral_ode
    >>> grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    >>> analytic = np.asarray(one_cmt_oral_conc(100.0, 1.0, 5.0, 50.0, grid))
    >>> numeric = one_cmt_oral_ode(100.0, 1.0, 5.0, 50.0, grid)
    >>> float(np.max(np.abs(analytic - numeric))) < 5e-4   # parity
    True
    """
    times_array = np.asarray(sorted(np.asarray(times, dtype=float)), dtype=float)
    if times_array.size == 0:
        raise DataValidationError("times must not be empty")
    if np.any(times_array < 0):
        raise DataValidationError("times must be non-negative")
    dose = float(dose)
    ka = float(ka)
    cl = float(cl)
    v = float(v)
    if dose < 0 or ka <= 0 or cl <= 0 or v <= 0:
        raise DataValidationError("dose >= 0 and ka, cl, v > 0 are required")
    if step <= 0:
        raise DataValidationError(f"step must be positive, got {step}")

    # one grid: every observation time plus <= step-spaced fillers
    spans = np.diff(np.concatenate(([0.0], times_array)))
    fills = [max(1, math.ceil(span / step)) for span in spans]
    grid_points = [0.0]
    for end, fill in zip(times_array, fills, strict=True):
        start = grid_points[-1]
        grid_points.extend(
            start + (end - start) * (i / fill) for i in range(1, fill + 1)
        )
    grid_array = np.asarray(grid_points, dtype=float)
    lookup = {round(float(value), 9): index for index, value in enumerate(times_array)}

    def derivatives(state: np.ndarray) -> np.ndarray:
        depot, center = float(state[0]), float(state[1])
        return np.array([-ka * depot, ka * depot - (cl / v) * center])

    state = np.array([dose, 0.0])
    recorded: dict[float, float] = {}
    if 0.0 in lookup:
        recorded[0.0] = 0.0
    for index in range(len(grid_array) - 1):
        h = grid_array[index + 1] - grid_array[index]
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * h * k1)
        k3 = derivatives(state + 0.5 * h * k2)
        k4 = derivatives(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        stamp = round(float(grid_array[index + 1]), 9)
        if stamp in lookup:
            recorded[stamp] = float(state[1]) / v
    return np.asarray([recorded[round(float(t), 9)] for t in times_array], dtype=float)


# --------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------- #

_RECORD_COLUMNS = ["ID", "TIME", "AMT", "EVID", "DV", "CMT", "WT"]


def build_nonmem_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape a raw observation table into NONMEM-style records.

    The R script's ``dplyr`` recipe exactly: one dosing record per
    subject (``TIME = 0``, ``AMT = DOSE_MG``, ``EVID = 1``, ``DV =
    NA``, ``CMT = 1`` -- the depot), one record per observation
    (``EVID = 0``, ``CMT = 2`` -- the central compartment), stacked
    and sorted by ``ID, TIME, desc(EVID)`` so the dose precedes any
    observation sharing its timestamp.

    Parameters
    ----------
    raw:
        A frame with the source project's columns ``ID``,
        ``TIME_H``, ``CONC_MG_L``, ``DOSE_MG`` and ``WEIGHT_KG``
        (extra columns are ignored).

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.pharma.nlme import build_nonmem_records
    >>> raw = pd.DataFrame({
    ...     "ID": [1, 1, 2, 2],
    ...     "TIME_H": [0.5, 2.0, 0.5, 2.0],
    ...     "CONC_MG_L": [0.6, 1.1, 0.4, 0.9],
    ...     "DOSE_MG": [100.0, 100.0, 100.0, 100.0],
    ...     "WEIGHT_KG": [70.0, 70.0, 55.0, 55.0],
    ... })
    >>> records = build_nonmem_records(raw)
    >>> list(records.columns)
    ['ID', 'TIME', 'AMT', 'EVID', 'DV', 'CMT', 'WT']
    >>> first = records.head(2)
    >>> (int(first.iloc[0]["ID"]), float(first.iloc[0]["TIME"]),
    ...  float(first.iloc[0]["AMT"]), int(first.iloc[0]["EVID"]), int(first.iloc[0]["CMT"]))
    (1, 0.0, 100.0, 1, 1)
    >>> import math
    >>> math.isnan(float(first.iloc[0]["DV"]))       # the dose row carries no DV
    True
    >>> (int(first.iloc[1]["EVID"]), float(first.iloc[1]["DV"]), int(first.iloc[1]["CMT"]))
    (0, 0.6, 2)
    >>> records[records.EVID == 1].shape[0]      # one dose row per subject
    2
    """
    required = ["ID", "TIME_H", "CONC_MG_L", "DOSE_MG", "WEIGHT_KG"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise DataValidationError(
            f"build_nonmem_records needs columns {required}, missing {missing}"
        )
    observations = pd.DataFrame(
        {
            "ID": raw["ID"].to_numpy(),
            "TIME": raw["TIME_H"].to_numpy(dtype=float),
            "AMT": 0.0,
            "EVID": 0,
            "DV": raw["CONC_MG_L"].to_numpy(dtype=float),
            "CMT": 2,
            "WT": raw["WEIGHT_KG"].to_numpy(dtype=float),
        }
    )
    per_subject = raw.groupby("ID", sort=True).first().reset_index()
    doses = pd.DataFrame(
        {
            "ID": per_subject["ID"].to_numpy(),
            "TIME": 0.0,
            "AMT": per_subject["DOSE_MG"].to_numpy(dtype=float),
            "EVID": 1,
            "DV": np.nan,
            "CMT": 1,
            "WT": per_subject["WEIGHT_KG"].to_numpy(dtype=float),
        }
    )
    records = pd.concat([doses, observations], ignore_index=True)
    records = records.sort_values(
        ["ID", "TIME", "EVID"], ascending=[True, True, False], kind="stable"
    ).reset_index(drop=True)
    return records[_RECORD_COLUMNS]


@dataclass
class PopulationData:
    """A simulated population: NONMEM records plus the truths.

    ``records`` is the fitting frame (``build_nonmem_records``
    layout); ``subjects`` carries the generating per-subject
    parameters (WT, the three etas and the resulting ka/CL/V);
    ``truth`` is the parameter dict used to simulate.
    """

    records: pd.DataFrame
    subjects: pd.DataFrame
    truth: dict[str, float] = field(default_factory=dict)


def simulate_population(
    n_subjects: int,
    *,
    theta: dict[str, float] | None = None,
    omega: dict[str, float] | None = None,
    sigma: dict[str, float] | None = None,
    dose: float = 100.0,
    weight_range: tuple[float, float] = (45.0, 90.0),
    sample_times: list[float] | tuple[float, ...] = DEFAULT_SAMPLE_TIMES,
    seed: int | None = None,
) -> PopulationData:
    """Draw one Phase-1-style study from the one-cmt oral model.

    Weights uniform in ``weight_range`` (the source data spans
    45-90 kg), etas iid ``N(0, Omega)`` on the log-scale parameters,
    concentrations from :func:`one_cmt_oral_conc` at ``sample_times``
    (the 10-sample grid by default) after a single oral ``dose``,
    observations perturbed by the combined error model ``DV ~
    N(cp, sqrt(add^2 + (prop*cp)^2))`` -- nlmixr2's ``prop() +
    add()`` on the SD scale.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import simulate_population
    >>> study = simulate_population(8, seed=42)
    >>> study.records.shape, study.subjects.shape
    ((88, 7), (8, 8))
    >>> sorted(study.subjects.columns)
    ['CL', 'ID', 'KA', 'V', 'WT', 'eta_cl', 'eta_ka', 'eta_v']
    >>> bool(study.records[study.records.EVID == 0]["DV"].min() < 1.0)
    True
    """
    if n_subjects < 1:
        raise DataValidationError(f"n_subjects must be >= 1, got {n_subjects}")
    theta = {
        "tka": NLMIXR2_INI["tka"],
        "tcl": NLMIXR2_INI["tcl"],
        "tv": NLMIXR2_INI["tv"],
        **(theta or {}),
    }
    omega = {
        "ka": NLMIXR2_INI["omega_ka"],
        "cl": NLMIXR2_INI["omega_cl"],
        "v": NLMIXR2_INI["omega_v"],
        **(omega or {}),
    }
    sigma = {
        "add": NLMIXR2_INI["add_err"],
        "prop": NLMIXR2_INI["prop_err"],
        **(sigma or {}),
    }
    generator = np.random.default_rng(seed)
    weights = generator.uniform(weight_range[0], weight_range[1], n_subjects)
    eta_ka = generator.normal(0.0, math.sqrt(omega["ka"]), n_subjects)
    eta_cl = generator.normal(0.0, math.sqrt(omega["cl"]), n_subjects)
    eta_v = generator.normal(0.0, math.sqrt(omega["v"]), n_subjects)
    times = np.asarray(sample_times, dtype=float)

    rows: list[dict[str, float]] = []
    subjects: list[dict[str, float]] = []
    for index in range(n_subjects):
        ka, cl, v = individual_params(
            theta["tka"], theta["tcl"], theta["tv"],
            eta_ka[index], eta_cl[index], eta_v[index], weights[index],
        )
        conc = np.asarray(one_cmt_oral_conc(dose, ka, cl, v, times), dtype=float)
        sd = np.sqrt(sigma["add"] ** 2 + (sigma["prop"] * conc) ** 2)
        dv = generator.normal(conc, sd)
        rows.append(
            {
                "ID": index + 1, "TIME": 0.0, "AMT": float(dose),
                "EVID": 1, "DV": float("nan"), "CMT": 1, "WT": weights[index],
            }
        )
        rows.extend(
            {
                "ID": index + 1, "TIME": float(time), "AMT": 0.0,
                "EVID": 0, "DV": float(value), "CMT": 2, "WT": weights[index],
            }
            for time, value in zip(times, dv, strict=True)
        )
        subjects.append(
            {
                "ID": index + 1, "WT": weights[index],
                "eta_ka": eta_ka[index], "eta_cl": eta_cl[index],
                "eta_v": eta_v[index], "KA": ka, "CL": cl, "V": v,
            }
        )
    truth = {
        "tka": theta["tka"], "tcl": theta["tcl"], "tv": theta["tv"],
        **{f"omega_{key}": value for key, value in omega.items()},
        **{f"sigma_{key}": value for key, value in sigma.items()},
    }
    return PopulationData(
        records=pd.DataFrame(rows, columns=_RECORD_COLUMNS),
        subjects=pd.DataFrame(subjects),
        truth=truth,
    )


def demo_population(seed: int = 42, n_subjects: int = 8) -> PopulationData:
    """The seeded demo study at the R project's true parameters."""
    return simulate_population(n_subjects, seed=seed)


# --------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------- #

@dataclass
class NLMEFit:
    """The result of :func:`fit_population` -- one FOCEI-flavored fit.

    Attributes
    ----------
    theta:
        ``{"tka", "tcl", "tv"}`` -- fixed effects on the log scale.
    omega:
        ``{"ka", "cl", "v"}`` -- between-subject variances (diagonal).
    sigma:
        ``{"add", "prop"}`` -- residual error SDs.
    etas:
        Per-subject empirical Bayes etas (columns ``eta_ka``,
        ``eta_cl``, ``eta_v``).
    ofv:
        The final joint ``-2 log likelihood`` (constants included).
    history:
        The objective every few iterations (the ``print = 5`` analog).
    """

    theta: dict[str, float]
    omega: dict[str, float]
    sigma: dict[str, float]
    etas: pd.DataFrame
    ofv: float
    history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    records: pd.DataFrame | None = None

    # -- diagnostics ------------------------------------------------- #

    def shrinkage(self) -> dict[str, float]:
        """Eta shrinkage in percent, ``1 - sd(eta)/sqrt(omega)``.

        The diagnostic nlmixr2 reports per random effect: the closer
        the empirical eta spread stays to the modelled Omega, the
        more the data actually informed each subject's etas (large
        shrinkage means the etas collapsed toward the population
        mean -- sparse data, or an over-shrunk model).
        """
        return eta_shrinkage(
            {name: self.etas[name].to_numpy(dtype=float) for name in
             ("eta_ka", "eta_cl", "eta_v")},
            {"eta_ka": self.omega["ka"], "eta_cl": self.omega["cl"],
             "eta_v": self.omega["v"]},
        )

    def predict(self, records: pd.DataFrame | None = None) -> pd.DataFrame:
        """PRED / IPRED / CWRES for every observation record.

        ``PRED`` uses the population parameters (eta = 0), ``IPRED``
        the subject's fitted etas; ``CWRES`` divides the residual by
        the subject's own combined-error SD (a conditional weighted
        residual in spirit -- the ``I`` of FOCEI in the error model,
        without the full FO linearization the real CWRES carries).
        """
        frame = self.records if records is None else records
        if frame is None:
            raise DataValidationError("the fit carries no records to predict")
        observations = frame[frame["EVID"] == 0].reset_index(drop=True)
        eta_lookup = {
            int(row.ID): (row.eta_ka, row.eta_cl, row.eta_v)
            for row in self.etas.itertuples(index=False)
        }
        pred_rows: list[dict[str, float]] = []
        for row in observations.itertuples(index=False):
            subject = int(row.ID)
            wt = float(row.WT)
            t = float(row.TIME)
            dose = float(row.AMT) if float(row.AMT) > 0 else _dose_of(frame, subject)
            eta_ka, eta_cl, eta_v = eta_lookup.get(subject, (0.0, 0.0, 0.0))
            ka_p, cl_p, v_p = individual_params(
                self.theta["tka"], self.theta["tcl"], self.theta["tv"],
                0.0, 0.0, 0.0, wt,
            )
            ka_i, cl_i, v_i = individual_params(
                self.theta["tka"], self.theta["tcl"], self.theta["tv"],
                eta_ka, eta_cl, eta_v, wt,
            )
            pred = float(one_cmt_oral_conc(dose, ka_p, cl_p, v_p, t))
            ipred = float(one_cmt_oral_conc(dose, ka_i, cl_i, v_i, t))
            sd = math.sqrt(
                self.sigma["add"] ** 2 + (self.sigma["prop"] * ipred) ** 2
            )
            pred_rows.append(
                {
                    "ID": subject, "TIME": t, "DV": float(row.DV),
                    "PRED": pred, "IPRED": ipred,
                    "CWRES": (float(row.DV) - ipred) / sd,
                }
            )
        return pd.DataFrame(pred_rows)

    def summary(self) -> str:
        """The fit table: thetas, omega, sigma, shrinkage, OFV."""
        lines = ["population PK fit (the nlmixr2 one_cmt_oral model)"]
        labels = (
            ("tka", "ka", math.exp(self.theta["tka"])),
            ("tcl", "CL", math.exp(self.theta["tcl"])),
            ("tv", "V", math.exp(self.theta["tv"])),
        )
        for name, plain, value in labels:
            lines.append(f"  {name:<5}= {self.theta[name]:+.4f}   ({plain} = {value:.3f})")
        lines.append(
            "  omega = ("
            + ", ".join(f"{self.omega[key]:.4f}" for key in ("ka", "cl", "v"))
            + ")"
        )
        lines.append(
            f"  sigma = (add {self.sigma['add']:.4f}, prop {self.sigma['prop']:.4f})"
        )
        shrinkage = self.shrinkage()
        lines.append(
            "  eta shrinkage = ("
            + ", ".join(
                f"{shrinkage[f'eta_{key}']:.1f}%" for key in ("ka", "cl", "v")
            )
            + ")"
        )
        lines.append(f"  objective (-2LL) = {self.ofv:.2f}")
        return "\n".join(lines)


def _dose_of(frame: pd.DataFrame, subject: int) -> float:
    doses = frame[(frame["ID"] == subject) & (frame["EVID"] == 1)]["AMT"]
    if doses.empty:
        raise DataValidationError(f"no dosing record for subject {subject}")
    return float(doses.iloc[0])


def eta_shrinkage(
    etas: dict[str, np.ndarray | list[float]], omega: dict[str, float]
) -> dict[str, float]:
    """``100 * (1 - sd(eta) / sqrt(omega))`` per random effect.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import eta_shrinkage
    >>> import math
    >>> shrinkage = eta_shrinkage(
    ...     {"eta_cl": [0.3, -0.3, 0.15, -0.15]}, {"eta_cl": 0.09})
    >>> round(shrinkage["eta_cl"])          # sd(eta) = 0.2739, sqrt(0.09) = 0.3
    9
    """
    result: dict[str, float] = {}
    for name, values in etas.items():
        array = np.asarray(values, dtype=float)
        if name not in omega:
            raise DataValidationError(f"omega is missing the variance for {name!r}")
        if array.size < 2:
            result[name] = float("nan")
            continue
        spread = float(np.std(array, ddof=1))
        result[name] = max(
            0.0, 100.0 * (1.0 - spread / math.sqrt(float(omega[name])))
        )
    return result


def fit_population(
    records: pd.DataFrame,
    *,
    init: dict[str, float] | None = None,
    iterations: int = 500,
    lr: float = 0.06,
    history_every: int = 25,
) -> NLMEFit:
    """Fit the population model by joint likelihood on the autograd tape.

    The objective is the exact joint ``-2 log likelihood`` of
    ``(y, eta)`` -- every per-subject eta is a free parameter
    optimized jointly with the fixed effects (conditional estimation
    without FOCEI's first-order linearization; the *interaction*
    half is honored because the proportional residual uses the
    individual prediction).  Minimized with the toolkit's own Adam
    (:mod:`pharmaml.ml.autograd`) over ``tka, tcl, tv`` (log-scale
    thetas), ``log(omega_k)`` and ``log(sigma)`` -- positivity by
    construction, exactly why nlmixr2's UI works on the log scale
    too.

    Parameters
    ----------
    records:
        A NONMEM-style frame (``build_nonmem_records`` layout or
        :attr:`PopulationData.records`): single oral dose per subject.
    init:
        Starting values keyed ``tka``/``tcl``/``tv``/``omega_ka``/
        ``omega_cl``/``omega_v``/``add_err``/``prop_err`` (defaults:
        the R project's ``ini({})`` block values, slightly perturbed
        so the fit has real work to do).
    iterations, lr:
        Adam's budget.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import demo_population, fit_population
    >>> study = demo_population(seed=42, n_subjects=8)
    >>> fit = fit_population(study.records, iterations=300, lr=0.08)
    >>> abs(fit.theta["tcl"] - study.truth["tcl"]) < 0.30   # CL recovered ~5 L/h
    True
    >>> 0.0 <= fit.omega["cl"] <= 0.6
    True
    >>> 0.01 <= fit.sigma["prop"] <= 0.4
    True
    >>> predictions = fit.predict()
    >>> predictions.shape[0]
    80
    >>> abs(float(predictions["CWRES"].mean())) < 0.5      # ~unbiased residuals
    True
    """
    from pharmaml.ml.autograd import Adam, Parameter, Tensor

    for column in _RECORD_COLUMNS:
        if column not in records.columns:
            raise DataValidationError(
                f"records needs column {column!r} (the NONMEM layout)"
            )
    observations = records[records["EVID"] == 0]
    if observations.empty:
        raise DataValidationError("records carries no observations (EVID == 0)")
    subject_ids = sorted(int(value) for value in records["ID"].unique())
    subject_index = {subject: index for index, subject in enumerate(subject_ids)}
    n_subjects = len(subject_ids)
    obs_subject = observations["ID"].map(subject_index).to_numpy(dtype=int)
    obs_time = observations["TIME"].to_numpy(dtype=float)
    obs_dv = observations["DV"].to_numpy(dtype=float)
    if np.any(np.isnan(obs_dv)):
        raise DataValidationError("observation rows must carry numeric DV values")

    doses = np.asarray(
        [_dose_of(records, subject) for subject in subject_ids], dtype=float
    )
    weights = np.asarray(
        [
            float(
                records[(records["ID"] == subject) & (records["EVID"] == 0)]["WT"].iloc[0]
            )
            for subject in subject_ids
        ],
        dtype=float,
    )

    # the perturbed start: the fit has real work to do even when the
    # data was generated at the ini-block truths (pass ``init`` to
    # start elsewhere; the R script itself starts AT the truths).
    start = {
        "tka": math.log(0.8), "tcl": math.log(4.0), "tv": math.log(60.0),
        "omega_ka": 0.25, "omega_cl": 0.25, "omega_v": 0.25,
        "add_err": 0.1, "prop_err": 0.2,
    }
    if init:
        start.update(init)

    # -- the selection matrix: one row per observation --------------- #
    selection = np.zeros((len(observations), n_subjects))
    selection[np.arange(len(observations)), obs_subject] = 1.0
    selection_t = Tensor(selection)
    dose_obs = selection @ doses  # (n_obs,)
    wt_cl = selection @ ((weights / REFERENCE_WEIGHT_KG) ** ALLOMETRIC_CL_EXPONENT)
    wt_v = selection @ ((weights / REFERENCE_WEIGHT_KG) ** ALLOMETRIC_V_EXPONENT)
    time_t = Tensor(obs_time.reshape(-1, 1))
    dose_t = Tensor(dose_obs.reshape(-1, 1))
    wt_cl_t = Tensor(wt_cl.reshape(-1, 1))
    wt_v_t = Tensor(wt_v.reshape(-1, 1))
    y_t = Tensor(obs_dv.reshape(-1, 1))

    tka = Parameter([[start["tka"]]])
    tcl = Parameter([[start["tcl"]]])
    tv = Parameter([[start["tv"]]])
    log_om_ka = Parameter([[math.log(start["omega_ka"])]])
    log_om_cl = Parameter([[math.log(start["omega_cl"])]])
    log_om_v = Parameter([[math.log(start["omega_v"])]])
    log_add = Parameter([[math.log(start["add_err"])]])
    log_prop = Parameter([[math.log(start["prop_err"])]])
    eta_ka = Parameter(np.zeros((n_subjects, 1)))
    eta_cl = Parameter(np.zeros((n_subjects, 1)))
    eta_v = Parameter(np.zeros((n_subjects, 1)))
    parameters = [
        tka, tcl, tv, log_om_ka, log_om_cl, log_om_v, log_add, log_prop,
        eta_ka, eta_cl, eta_v,
    ]
    optimizer = Adam(parameters, lr=lr)
    log_2pi = math.log(2.0 * math.pi)

    def objective() -> Any:
        ka = (tka + selection_t @ eta_ka).exp()
        cl = (tcl + selection_t @ eta_cl).exp() * wt_cl_t
        v = (tv + selection_t @ eta_v).exp() * wt_v_t
        ke = cl / v
        conc = dose_t * ka / (v * (ka - ke)) * (
            (-ke * time_t).exp() - (-ka * time_t).exp()
        )
        add = log_add.exp()
        prop = log_prop.exp()
        variance = add * add + (prop * conc) * (prop * conc)
        resid = y_t - conc
        data_term = variance.log().sum() + (resid * resid / variance).sum()
        data_term = data_term + float(len(observations)) * log_2pi
        prior_terms = []
        for eta_param, log_om in (
            (eta_ka, log_om_ka), (eta_cl, log_om_cl), (eta_v, log_om_v),
        ):
            omega = log_om.exp()
            prior_terms.append(
                (eta_param * eta_param / omega).sum() + n_subjects * omega.log()
            )
        return data_term + sum(prior_terms) + 3.0 * n_subjects * log_2pi

    history: list[float] = []
    completed = 0
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad()
        loss = objective()
        value = float(loss.item())
        if not math.isfinite(value):
            _LOG.warning("objective went non-finite at iteration %d; stopping", iteration)
            break
        loss.backward()
        optimizer.step()
        completed = iteration
        if iteration % max(1, history_every) == 0 or iteration == 1:
            history.append(value)

    def scalar(parameter: Parameter) -> float:
        return float(np.asarray(parameter.data).reshape(-1)[0])

    theta = {"tka": scalar(tka), "tcl": scalar(tcl), "tv": scalar(tv)}
    omega = {
        "ka": math.exp(scalar(log_om_ka)),
        "cl": math.exp(scalar(log_om_cl)),
        "v": math.exp(scalar(log_om_v)),
    }
    sigma = {"add": math.exp(scalar(log_add)), "prop": math.exp(scalar(log_prop))}
    etas = pd.DataFrame(
        {
            "ID": subject_ids,
            "eta_ka": np.asarray(eta_ka.data).reshape(-1),
            "eta_cl": np.asarray(eta_cl.data).reshape(-1),
            "eta_v": np.asarray(eta_v.data).reshape(-1),
        }
    )
    final_ofv = float(np.asarray(objective().detach().data).reshape(-1)[0])
    return NLMEFit(
        theta=theta, omega=omega, sigma=sigma, etas=etas,
        ofv=final_ofv, history=history, n_iterations=completed,
        records=records,
    )


# --------------------------------------------------------------------- #
# the Standard Two-Stage twin
# --------------------------------------------------------------------- #

def standard_two_stage(
    records: pd.DataFrame, *, iterations: int = 300, lr: float = 0.08
) -> pd.DataFrame:
    """Fit every subject separately (the project's Python twin).

    The source repo exists to compare nlmixr2's FOCEI against a
    simplified Standard Two-Stage analysis: per-subject nonlinear
    least squares on ``(ka, CL, V)`` (Adam on the log parameters,
    same autograd engine), then population moments from the sample.
    Returns one row per subject: ``ID, KA, CL, V, WT`` -- the layout
    of the committed ``individual_parameter_estimates.csv``.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import demo_population, standard_two_stage
    >>> study = demo_population(seed=42, n_subjects=8)
    >>> sts = standard_two_stage(study.records, iterations=200, lr=0.1)
    >>> list(sts.columns)
    ['ID', 'KA', 'CL', 'V', 'WT']
    >>> 2.0 < float(sts["CL"].mean()) < 9.0       # subjects near ~5 L/h
    True
    """
    from pharmaml.ml.autograd import Adam, Parameter, Tensor

    rows: list[dict[str, float]] = []
    for subject in sorted(int(value) for value in records["ID"].unique()):
        block = records[records["ID"] == subject]
        observations = block[block["EVID"] == 0]
        dose = _dose_of(records, subject)
        times = observations["TIME"].to_numpy(dtype=float)
        values = observations["DV"].to_numpy(dtype=float)
        weight = float(observations["WT"].iloc[0])
        time_t = Tensor(times.reshape(-1, 1))
        y_t = Tensor(values.reshape(-1, 1))
        lka = Parameter([[math.log(0.8)]])
        lcl = Parameter([[math.log(4.0)]])
        lv = Parameter([[math.log(60.0)]])
        optimizer = Adam([lka, lcl, lv], lr=lr)
        for _ in range(iterations):
            optimizer.zero_grad()
            ka, cl, v = lka.exp(), lcl.exp(), lv.exp()
            ke = cl / v
            conc = (
                Tensor(dose)
                * ka / (v * (ka - ke))
                * ((-ke * time_t).exp() - (-ka * time_t).exp())
            )
            residual = y_t - conc
            (residual * residual).sum().backward()
            optimizer.step()
        rows.append(
            {
                "ID": subject,
                "KA": math.exp(float(lka.data.reshape(-1)[0])),
                "CL": math.exp(float(lcl.data.reshape(-1)[0])),
                "V": math.exp(float(lv.data.reshape(-1)[0])),
                "WT": weight,
            }
        )
    return pd.DataFrame(rows)


def compare_focei_sts(fit: NLMEFit, sts: pd.DataFrame) -> dict[str, float]:
    """The R script's closing comparison, as numbers.

    ``focei_cl`` is ``exp(theta["tcl"])`` (the population CL at the
    70 kg reference), ``sts_cl`` the geometric mean of the STS
    individual CLs (``exp(mean(log(CL_EST)))`` -- the exact R line),
    plus the percent difference between them.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import compare_focei_sts, demo_population
    >>> from pharmaml.pharma.nlme import fit_population, standard_two_stage
    >>> study = demo_population(seed=42, n_subjects=8)
    >>> fit = fit_population(study.records, iterations=250, lr=0.08)
    >>> sts = standard_two_stage(study.records, iterations=150, lr=0.1)
    >>> comparison = compare_focei_sts(fit, sts)
    >>> sorted(comparison)
    ['difference_percent', 'focei_cl', 'sts_cl']
    >>> abs(comparison["difference_percent"]) < 60.0   # same ballpark
    True
    """
    focei_cl = math.exp(fit.theta["tcl"])
    sts_cl = float(np.exp(np.mean(np.log(sts["CL"].to_numpy(dtype=float)))))
    difference = 100.0 * (focei_cl - sts_cl) / sts_cl
    return {"focei_cl": focei_cl, "sts_cl": sts_cl, "difference_percent": difference}


# --------------------------------------------------------------------- #
# VPC
# --------------------------------------------------------------------- #

def jenks_breaks(values: list[float] | np.ndarray, n_classes: int) -> list[float]:
    """Fisher-Jenks optimal class boundaries (the ``vpc`` binning).

    The dynamic-programming exact solution: boundaries minimizing the
    within-class sum of squared deviations.  The R project's ``vpc``
    call asked for ``bins = "jenks"`` -- and then passed ``sim =
    NULL`` so no VPC data ever reached it; this implementation is
    the honest completion.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import jenks_breaks
    >>> jenks_breaks([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], 2)
    [3.0]
    >>> jenks_breaks([1.0, 2.0, 3.0], 1)
    []
    >>> jenks_breaks([1.0, 5.0], 5)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: n_classes (5) cannot exceed the number of distinct values (2)
    """
    data = sorted(float(value) for value in np.asarray(values, dtype=float).reshape(-1))
    unique = sorted(set(data))
    n = len(data)
    if n_classes < 1:
        raise DataValidationError(f"n_classes must be >= 1, got {n_classes}")
    if n_classes > len(unique):
        raise DataValidationError(
            f"n_classes ({n_classes}) cannot exceed the number of distinct "
            f"values ({len(unique)})"
        )
    if n_classes == 1:
        return []
    if n_classes == len(unique):
        return unique[1:-1] if len(unique) > 2 else []
    prefix = [0.0] * (n + 1)
    prefix_sq = [0.0] * (n + 1)
    for index, value in enumerate(data):
        prefix[index + 1] = prefix[index] + value
        prefix_sq[index + 1] = prefix_sq[index] + value * value

    def ssq(start: int, stop: int) -> float:
        count = stop - start
        total = prefix[stop] - prefix[start]
        return prefix_sq[stop] - prefix_sq[start] - total * total / count

    infinity = float("inf")
    cost = [[infinity] * (n_classes + 1) for _ in range(n + 1)]
    back = [[0] * (n_classes + 1) for _ in range(n + 1)]
    for classes in range(1, n_classes + 1):
        cost[0][classes] = 0.0
    for index in range(1, n + 1):
        cost[index][1] = ssq(0, index)
    for classes in range(2, n_classes + 1):
        for index in range(1, n + 1):
            for split in range(classes - 1, index):
                candidate = cost[split][classes - 1] + ssq(split, index)
                if candidate < cost[index][classes]:
                    cost[index][classes] = candidate
                    back[index][classes] = split
    boundaries: list[float] = []
    index, classes = n, n_classes
    while classes > 1:
        split = back[index][classes]
        boundaries.append(data[split - 1])
        index, classes = split, classes - 1
    return sorted(boundaries)


def vpc_simulate(
    fit: NLMEFit,
    *,
    n_replicates: int = 200,
    sample_times: list[float] | tuple[float, ...] = DEFAULT_SAMPLE_TIMES,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate VPC replicates from a fitted model.

    The step the R script left unwired (``vpc(sim = NULL, ...)``):
    ``n_replicates`` copies of the study design, every subject redrawn
    from the fitted thetas/omegas/sigmas -- etas ``N(0, Omega)``,
    combined residual error on top of the individual profiles.  The
    long frame carries ``REP, TIME, DV`` for :func:`vpc_percentiles`.

    Examples
    --------
    >>> from pharmaml.pharma.nlme import demo_population, fit_population, vpc_simulate
    >>> study = demo_population(seed=42, n_subjects=6)
    >>> fit = fit_population(study.records, iterations=150, lr=0.08)
    >>> simulated = vpc_simulate(fit, n_replicates=5, seed=1)
    >>> simulated.shape[0] == 5 * 6 * 10
    True
    >>> sorted(simulated.columns)
    ['DV', 'REP', 'TIME']
    """
    if fit.records is None:
        raise DataValidationError("the fit carries no records to simulate from")
    generator = np.random.default_rng(seed)
    design = fit.records[fit.records["EVID"] == 0]
    subjects = sorted(int(value) for value in design["ID"].unique())
    rows: list[dict[str, float]] = []
    times = np.asarray(sample_times, dtype=float)
    for replicate in range(1, n_replicates + 1):
        for subject in subjects:
            block = design[design["ID"] == subject]
            weight = float(block["WT"].iloc[0])
            dose = _dose_of(fit.records, subject)
            eta = (
                generator.normal(0.0, math.sqrt(fit.omega["ka"])),
                generator.normal(0.0, math.sqrt(fit.omega["cl"])),
                generator.normal(0.0, math.sqrt(fit.omega["v"])),
            )
            ka, cl, v = individual_params(
                fit.theta["tka"], fit.theta["tcl"], fit.theta["tv"],
                eta[0], eta[1], eta[2], weight,
            )
            conc = np.asarray(one_cmt_oral_conc(dose, ka, cl, v, times), dtype=float)
            sd = np.sqrt(fit.sigma["add"] ** 2 + (fit.sigma["prop"] * conc) ** 2)
            dv = generator.normal(conc, sd)
            rows.extend(
                {"REP": float(replicate), "TIME": float(time), "DV": float(value)}
                for time, value in zip(times, dv, strict=True)
            )
    return pd.DataFrame(rows)


def vpc_percentiles(simulated: pd.DataFrame) -> pd.DataFrame:
    """Per-time 5th / 50th / 95th percentiles of simulated DVs.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.pharma.nlme import vpc_percentiles
    >>> simulated = pd.DataFrame({
    ...     "REP": [1] * 4 + [2] * 4,
    ...     "TIME": [0.5, 0.5, 2.0, 2.0] * 2,
    ...     "DV": [1.0, 2.0, 3.0, 4.0] * 2,
    ... })
    >>> table = vpc_percentiles(simulated)
    >>> [round(row["p50"], 3) for row in table.to_dict("records")]
    [1.5, 3.5]
    """
    grouped = simulated.groupby("TIME", sort=True)
    rows = []
    for time, block in grouped:
        values = block["DV"].to_numpy(dtype=float)
        rows.append(
            {
                "TIME": float(time),
                "p5": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    return pd.DataFrame(rows)

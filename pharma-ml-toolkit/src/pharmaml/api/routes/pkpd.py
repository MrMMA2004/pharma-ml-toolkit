"""``/api/v1/pkpd`` — the PK/PD workspace's API surface over the REAL
backend (part10's row four).

All calculations are the toolkit's own pharmacometrics layer:

* ``GET /api/v1/pkpd/catalog`` — the route inventory (groups, the
  backend functions each endpoint wraps).
* ``POST /api/v1/pkpd/nca`` — :func:`pharmaml.pharma.pk.nca_summary` +
  :func:`auc_trapezoid` (linear + log) + the terminal-phase fit
  (:func:`pharmaml.pkpd.nca_ext.fit_terminal_phase`), inputs validated
  by :func:`pharmaml.utils.validation.validate_concentration_series`.
* ``POST /api/v1/pkpd/steady-state`` —
  :func:`pharmaml.pkpd.nca_ext.steady_state_metrics`.
* ``POST /api/v1/pkpd/iv-bolus`` —
  :func:`pharmaml.pharma.pk.iv_bolus_concentration`.
* ``POST /api/v1/pkpd/bateman`` —
  :func:`pharmaml.pkpd.compartments.bateman` (oral one-compartment).
* ``POST /api/v1/pkpd/superposition`` —
  :func:`pharmaml.pkpd.compartments.superposition` over a causal
  bateman unit response.
* ``POST /api/v1/pkpd/pbpk`` — :class:`pharmaml.pkpd.pbpk.MinimalPBPK`
  (``build_params`` + ``solve_dataframe``; iv or oral).
* ``POST /api/v1/pkpd/pd`` —
  :func:`pharmaml.pharma.pd.effect_table` (sigmoid Emax).
* ``POST /api/v1/pkpd/population`` —
  :func:`pharmaml.pkpd.population.simulate_population` as a managed
  job over a bateman model with IIV on ``ka``: the route batches the
  subjects (50 per batch, each batch a REAL seeded simulation) and
  reports honest ``job.progress`` per batch; cancelling between
  batches lands the job CANCELLED, never FAILED.  The payload returns
  population summary curves (mean / p5 / p95) plus the first few
  subject profiles.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma.pd import effect_table
from pharmaml.pharma.pk import auc_trapezoid, iv_bolus_concentration, nca_summary
from pharmaml.pkpd.compartments import bateman, superposition
from pharmaml.pkpd.nca_ext import fit_terminal_phase, steady_state_metrics
from pharmaml.pkpd.pbpk import MinimalPBPK
from pharmaml.pkpd.population import simulate_population
from pharmaml.utils.validation import validate_concentration_series

from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/pkpd", tags=["pkpd"])

_MAX_POINTS = 10000
_POP_BATCH = 50


class TimesConcRequest(BaseModel):
    times: list[float] = Field(min_length=2, max_length=_MAX_POINTS)
    conc: list[float] = Field(min_length=2, max_length=_MAX_POINTS)

    def _validated(self) -> tuple[np.ndarray, np.ndarray]:
        return validate_concentration_series(self.times, self.conc)


class NcaRequest(TimesConcRequest):
    # part16 §8 (Phase 5, surfaced by the demo lane): the NCA lane's
    # scientific core (nca_summary / fit_terminal_phase / the log
    # trapezoid in nca_ext.py) deliberately accepts intentional zeros
    # — a pre-dose t=0 row is standard PK practice, and the shipped
    # examples/data/pk_profile.csv starts with one.  The blanket
    # positive-only validation belongs to the lanes where a zero
    # concentration is a real input error; NCA mirrors the
    # steady-state precedent (require_positive_conc=False) instead,
    # and the route keeps its own auc_log guard (np.all(conc > 0)).
    def _validated(self) -> tuple[np.ndarray, np.ndarray]:
        return validate_concentration_series(
            self.times, self.conc, require_positive_conc=False
        )


class SteadyStateRequest(TimesConcRequest):
    tau: float = Field(gt=0.0, le=10000.0)
    dose: float | None = Field(default=None, gt=0.0)
    lambda_z: float | None = Field(default=None, gt=0.0)

    def _validated(self) -> tuple[np.ndarray, np.ndarray]:
        # steady-state profiles legitimately carry zero concentrations
        # (troughs / not-yet-absorbed segments): only negatives are a
        # client error here
        return validate_concentration_series(
            self.times, self.conc, require_positive_conc=False
        )


class IvBolusRequest(BaseModel):
    c0: float = Field(ge=0.0)
    ke: float = Field(gt=0.0, le=10.0)
    times: list[float] = Field(min_length=1, max_length=_MAX_POINTS)


class BatemanRequest(BaseModel):
    dose_per_volume: float = Field(ge=0.0)
    ka: float = Field(gt=0.0, le=100.0)
    ke: float = Field(gt=0.0, le=100.0)
    times: list[float] = Field(min_length=1, max_length=_MAX_POINTS)


class SuperpositionRequest(BatemanRequest):
    tau: float = Field(gt=0.0, le=10000.0)
    n_doses: int = Field(default=8, ge=1, le=1000)


class PbpkRequest(BaseModel):
    times: list[float] = Field(min_length=1, max_length=_MAX_POINTS)
    dose_mg: float = Field(gt=0.0, le=100000.0)
    route: Literal["iv", "oral"] = "iv"
    ka: float = Field(default=1.0, gt=0.0, le=100.0)
    f: float = Field(default=0.85, gt=0.0, le=1.0)
    bw: float = Field(default=70.0, gt=0.0, le=500.0)
    clint: float = Field(default=50.0, gt=0.0, le=100000.0)
    fu: float = Field(default=0.10, gt=0.0, le=1.0)
    bp: float = Field(default=1.20, gt=0.0, le=100.0)
    kp_lung: float = Field(default=2.0, gt=0.0, le=100.0)
    kp_liver: float = Field(default=5.0, gt=0.0, le=100.0)
    kp_kidney: float = Field(default=3.0, gt=0.0, le=100.0)
    kp_rest: float = Field(default=1.5, gt=0.0, le=100.0)


class PdRequest(BaseModel):
    concentrations: list[float] = Field(
        default_factory=lambda: [0.1, 1.0, 10.0, 100.0],
        min_length=1, max_length=2000,
    )
    e0: float = Field(default=0.0)
    emax: float
    ec50: float = Field(gt=0.0)
    hill: float = Field(default=1.0, gt=0.0, le=20.0)


class PopulationRequest(BaseModel):
    theta: list[float] = Field(min_length=3, max_length=3)
    omega_diag: list[float] = Field(min_length=1, max_length=1)
    times: list[float] = Field(min_length=2, max_length=2000)
    n_subjects: int = Field(default=100, ge=10, le=20000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    sigma_add: float = Field(default=0.0, ge=0.0, le=1000.0)
    sigma_prop: float = Field(default=0.0, ge=0.0, le=10.0)
    n_profiles: int = Field(default=3, ge=0, le=10)
    mode: Literal["job", "sync"] = "job"


@router.get("/catalog")
def catalog(request: Request) -> dict:
    return {
        "groups": [
            {"group": "nca", "routes": ["/nca", "/steady-state"],
             "backend": "pharmaml.pharma.pk + pharmaml.pkpd.nca_ext"},
            {"group": "models", "routes": ["/iv-bolus", "/bateman",
                                           "/superposition", "/pbpk"],
             "backend": "pharmaml.pharma.pk + pharmaml.pkpd.compartments + pbpk"},
            {"group": "pd", "routes": ["/pd"],
             "backend": "pharmaml.pharma.pd"},
            {"group": "population", "routes": ["/population"],
             "backend": "pharmaml.pkpd.population.simulate_population"},
        ]
    }


@router.post("/nca")
def nca(req: NcaRequest, request: Request) -> dict:
    times, conc = req._validated()
    summary = nca_summary(times, conc)
    auc_linear = auc_trapezoid(times, conc, method="linear")
    auc_log: float | None = None
    if bool(np.all(conc > 0.0)):
        auc_log = float(auc_trapezoid(times, conc, method="log"))
    terminal = fit_terminal_phase(times, conc)
    return {
        "summary": summary.as_dict(),
        "auc_trapezoid_linear": float(auc_linear),
        "auc_trapezoid_log": auc_log,
        "terminal": {
            "start_index": int(terminal.start_index),
            "end_index": int(terminal.end_index),
            "n_points": int(terminal.n_points),
            "lambda_z": float(terminal.lambda_z),
            "r2": float(terminal.r2),
            "r2_adj": float(terminal.r2_adj),
            "half_life": float(terminal.half_life),
            "clast_pred": float(terminal.clast_pred),
            "span_half_lives": float(terminal.span_half_lives),
            "span_low": bool(terminal.span_low),
            "method": terminal.method,
        },
        "n_points": len(times),
    }


@router.post("/steady-state")
def steady_state(req: SteadyStateRequest, request: Request) -> dict:
    times, conc = req._validated()
    try:
        metrics = steady_state_metrics(
            times, conc, req.tau, dose=req.dose, lambda_z=req.lambda_z
        )
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"tau": req.tau, "metrics": metrics.as_dict(), "n_points": len(times)}


@router.post("/iv-bolus")
def iv_bolus(req: IvBolusRequest, request: Request) -> dict:
    times = np.asarray(req.times, dtype=float)
    values = iv_bolus_concentration(req.c0, req.ke, times)
    return {
        "c0": req.c0,
        "ke": req.ke,
        "n_points": len(times),
        "concentration": json_safe([float(v) for v in np.atleast_1d(values)]),
    }


@router.post("/bateman")
def bateman_route(req: BatemanRequest, request: Request) -> dict:
    times = np.asarray(req.times, dtype=float)
    values = bateman(req.dose_per_volume, req.ka, req.ke, times)
    return {
        "dose_per_volume": req.dose_per_volume,
        "ka": req.ka,
        "ke": req.ke,
        "n_points": len(times),
        "concentration": json_safe([float(v) for v in np.atleast_1d(values)]),
    }


@router.post("/superposition")
def superposition_route(req: SuperpositionRequest, request: Request) -> dict:
    times = np.asarray(req.times, dtype=float)

    def unit_response(t: np.ndarray) -> np.ndarray:
        # causal: bateman(0) == 0 exactly, so clipping negative
        # time-since-dose to zero is the honest causal continuation
        clamped = np.clip(np.asarray(t, dtype=float), 0.0, None)
        return np.asarray(
            bateman(req.dose_per_volume, req.ka, req.ke, clamped)
        )

    values = superposition(times, unit_response, req.tau, req.n_doses)
    return {
        "tau": req.tau,
        "n_doses": req.n_doses,
        "n_points": len(times),
        "concentration": json_safe([float(v) for v in np.asarray(values)]),
    }


@router.post("/pbpk")
def pbpk(req: PbpkRequest, request: Request) -> dict:
    times = np.asarray(req.times, dtype=float)
    params = MinimalPBPK.build_params(
        bw=req.bw, clint=req.clint, fu=req.fu, bp=req.bp,
        kp_lung=req.kp_lung, kp_liver=req.kp_liver,
        kp_kidney=req.kp_kidney, kp_rest=req.kp_rest,
    )
    model = MinimalPBPK(params)
    frame = model.solve_dataframe(
        times, req.dose_mg, route=req.route, ka=req.ka, f=req.f
    )
    rows = frame.to_dict(orient="records")
    return {
        "route": req.route,
        "dose_mg": req.dose_mg,
        "compartments": MinimalPBPK.COMPARTMENTS,
        "n_points": len(rows),
        "rows": json_safe(rows),
    }


@router.post("/pd")
def pd_route(req: PdRequest, request: Request) -> dict:
    table = effect_table(
        req.concentrations, req.e0, req.emax, req.ec50, hill=req.hill
    )
    return {
        "e0": req.e0, "emax": req.emax, "ec50": req.ec50, "hill": req.hill,
        "n_points": len(req.concentrations),
        **{k: json_safe(v) for k, v in table.items()},
    }


def _population_predict(theta: list[float]):
    """predict_fn(theta, eta, times): bateman with IIV on ka."""
    def predict(theta_: Any, eta: Any, times: Any) -> np.ndarray:
        ka = float(theta_[1]) * float(np.exp(np.asarray(eta)[0]))
        return np.asarray(
            bateman(float(theta_[0]), ka, float(theta_[2]), times)
        )
    return predict


def _population_body(
    theta: list[float],
    omega_diag: list[float],
    times: list[float],
    n_subjects: int,
    seed: int,
    sigma_add: float,
    sigma_prop: float,
    n_profiles: int,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: REAL simulate_population, batched for telemetry."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the population stage.")
    predict_fn = _population_predict(theta)
    grid = np.asarray(times, dtype=float)
    profiles: list[np.ndarray] = []
    done = 0
    while done < n_subjects:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJob(
                f"population simulation cancelled after {done}/{n_subjects} subjects"
            )
        batch = min(_POP_BATCH, n_subjects - done)
        profiles.extend(
            simulate_population(
                predict_fn, theta, omega_diag, grid, batch,
                seed=seed + done, sigma_add=sigma_add, sigma_prop=sigma_prop,
            )
        )
        done += batch
        if manager is not None and job_id is not None:
            manager.report_progress(
                job_id,
                min(99, int(100.0 * done / n_subjects)),
                f"simulated {done}/{n_subjects} subjects",
            )
    matrix = np.vstack([np.asarray(p).reshape(1, -1) for p in profiles])
    mean = matrix.mean(axis=0)
    p5 = np.percentile(matrix, 5, axis=0)
    p95 = np.percentile(matrix, 95, axis=0)
    return {
        "n_subjects": n_subjects,
        "n_times": len(grid),
        "seed": seed,
        "summary": {
            "mean": json_safe([float(v) for v in mean]),
            "p5": json_safe([float(v) for v in p5]),
            "p95": json_safe([float(v) for v in p95]),
        },
        "subject_profiles": [
            json_safe([float(v) for v in profiles[i]])
            for i in range(min(n_profiles, len(profiles)))
        ],
    }


@router.post("/population")
def population(req: PopulationRequest, request: Request) -> dict:
    if float(req.omega_diag[0]) < 0.0:
        raise HTTPException(status_code=422, detail="omega_diag must be >= 0")
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_population_body(
                req.theta, req.omega_diag, req.times, req.n_subjects,
                req.seed, req.sigma_add, req.sigma_prop, req.n_profiles,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "pkpd.population",
        "pkpd",
        _population_body,
        configuration={
            "theta": req.theta,
            "omega_diag": req.omega_diag,
            "n_subjects": req.n_subjects,
            "n_times": len(req.times),
            "seed": req.seed,
            "sigma_add": req.sigma_add,
            "sigma_prop": req.sigma_prop,
        },
        theta=req.theta,
        omega_diag=req.omega_diag,
        times=req.times,
        n_subjects=req.n_subjects,
        seed=req.seed,
        sigma_add=req.sigma_add,
        sigma_prop=req.sigma_prop,
        n_profiles=req.n_profiles,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}

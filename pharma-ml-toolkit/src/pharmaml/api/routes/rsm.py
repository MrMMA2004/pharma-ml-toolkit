"""``/api/v1/rsm`` — the Response Surface Methodology lane (part17
§6, the campaign queue's R1 lane) over the REAL in-tree backend
(:mod:`pharmaml.ml.rsm`: the design generators, the second-order fit
with its ANOVA / diagnostics / hierarchy / aliasing disclosure, the
zone-vocabulary prediction lane, the contour / desirability grids,
and the Derringer-Suich multi-response optimiser).

* ``GET /api/v1/rsm/registry`` — the vocabulary verbatim: the three
  design generators (each with its runs formula, its factor bounds,
  and its honest notes — the rotatable CCD's axial runs sit outside
  the declared ranges, disclosed), the desirability goals, the
  second-order term registry formula, the bounds, the seeded
  SYNTHETIC demo experiment descriptor, and the scope statement (the
  mandate's own distinction table).
* ``POST /api/v1/rsm/design`` — generate a design (CCD rotatable /
  faced / custom alpha, Box-Behnken, full 2^k factorial): the run
  table in coded AND actual units with run order, blocks (the CCD
  two-block split), run types, the run-type census, and the honest
  assumptions.  Validation is typed (factor ranges, run-count bound
  512, blocking refused for the designs whose block structure the
  module honestly does not implement).  The export rides the existing
  generic table lane on the frontend — NO duplicate export endpoint.
* ``POST /api/v1/rsm/fit`` — import the observed results and fit the
  full second-order model for EVERY response column: the request's own
  factor ranges + response columns over a registered typed dataset,
  or the route's own seeded SYNTHETIC demo experiment (labelled
  synthetic wherever it travels).  The response carries the
  coefficient tables (estimate / std_error / t / p / CI), the ANOVA
  partition (regression / residual / pure error / lack-of-fit — nulls
  where no replicated design points exist), the diagnostics block
  (R2 / adj R2 / pred R2 / PRESS / RMSE / leverage / studentised
  residuals / Cook's distance with STATED thresholds), the
  hierarchy notes (flagged, never auto-removed), the aliasing
  disclosure (rank deficiency honestly reported), and the
  fitted-vs-observed rows.  The fitted ``ResponseSurfaceBundle``
  registers in the session model store and the run records in the
  §15.2 lineage as family ``ml.rsm``.
* ``POST /api/v1/rsm/predict`` — point predictions in ACTUAL units
  with the mean CI and the new-observation PI, and the honest zone
  vocabulary per point (inside_ranges / axial_region /
  outside_design — extrapolation flagged, never silent).
* ``POST /api/v1/rsm/contour`` — the two-factor grid payload for the
  Plotly lanes: a chosen response's prediction OR the combined
  desirability under the request's goals, with every other factor
  held at explicitly displayed fixed values.
* ``POST /api/v1/rsm/optimize`` — the Derringer-Suich
  multi-response optimisation over the fitted bundle: the goals
  (maximize / minimize / target / range) with weights and exponents,
  the combined geometric desirability, optional linear coded-factor
  constraints (SLSQP-honoured), the deterministic multi-start
  candidate confined to the declared factor ranges, the per-factor
  sensitivity sweep, the curated Pareto alternatives, and the
  feasible-region mask for the requested pair.
* ``GET /api/v1/rsm/models`` — the session's RSM models (the
  comparison surface: every fitted surface's statistics side by
  side).

The honest notes: the optimiser is confined to the coded box so it
cannot extrapolate by construction (the prediction lane flags
extrapolated points separately); lack-of-fit needs genuinely
replicated design points (nulls otherwise, never invented); the
desirability bounds the request did not set fall back to the observed
design min/max (stated per goal, never silent); and every demo
number is a model estimate over KNOWN synthetic surfaces, never an
experimental fact.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pharmaml import __version__ as _toolkit_version
from pharmaml.api.runs import ExperimentRunRecord
from pharmaml.api.stores import TrainedModelRecord
from pharmaml.ml.rsm import (
    DEMO_FACTORS,
    DEMO_RESPONSES,
    GOALS,
    MAX_CONSTRAINTS,
    MAX_FACTORS,
    MAX_GRID_POINTS,
    MAX_RESPONSES,
    MIN_FACTORS,
    MIN_GRID_POINTS,
    FactorSpec,
    build_design,
    contour_grid,
    demo_rsm_dataset,
    environment_metadata,
    fit_response_surface,
    optimize_desirability,
    predict_surface,
    rsm_registry_payload,
)

router = APIRouter(prefix="/rsm", tags=["rsm"])

_MAX_SEED = 2147483647
_MAX_CONSTRAINT_LIST = MAX_CONSTRAINTS


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _uuid12() -> str:
    return uuid.uuid4().hex[:12]


class FactorSpecModel(BaseModel):
    """One declared factor: name, range, optional unit."""

    name: str = Field(min_length=1, max_length=64)
    low: float
    high: float
    unit: str | None = Field(default=None, max_length=32)


class DesignRsmRequest(BaseModel):
    """The §6.3 design-generation contract."""

    factors: list[FactorSpecModel] = Field(
        min_length=MIN_FACTORS, max_length=MAX_FACTORS
    )
    design: Literal["ccd", "box_behnken", "full_factorial"] = "ccd"
    alpha_type: Literal["rotatable", "faced", "custom"] = "rotatable"
    alpha: float | None = Field(default=None, gt=0.5, le=4.0)
    center_points: int = Field(default=3, ge=0, le=50)
    replicates: int = Field(default=1, ge=1, le=20)
    randomize: bool = True
    seed: int = Field(default=42, ge=0, le=_MAX_SEED)
    blocks: bool = False


class FitRsmRequest(BaseModel):
    """The §6.4 fit contract over an observed frame.

    ``dataset_id`` selects a registered typed dataset whose factor
    columns carry the OBSERVED actual levels and whose response
    columns carry the measured results; leaving it None rides the
    seeded SYNTHETIC demo experiment (labelled).  The factor ranges
    are the request's own declared ranges (the coding transform rides
    them); the optional design block records which design produced
    the rows (the zone vocabulary's axial region needs the CCD's
    alpha).
    """

    dataset_id: str | None = None
    factors: list[FactorSpecModel] | None = None
    response_columns: list[str] = Field(
        default_factory=lambda: list(DEMO_RESPONSES),
        min_length=1,
        max_length=MAX_RESPONSES,
    )
    design: Literal["ccd", "box_behnken", "full_factorial"] | None = None
    alpha_type: Literal["rotatable", "faced"] | None = None
    alpha: float | None = Field(default=None, gt=0.5, le=4.0)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)


class PredictRsmRequest(BaseModel):
    """Point rows (actual units) -> per-response predictions + CI/PI."""

    model_id: str
    points: list[dict[str, float]] = Field(min_length=1, max_length=200)
    confidence: float | None = Field(default=None, gt=0.5, lt=1.0)


class GoalSpec(BaseModel):
    """One response's desirability goal (the §6.6 contract)."""

    response: str = Field(min_length=1, max_length=64)
    goal: Literal["maximize", "minimize", "target", "range"]
    low: float | None = None
    high: float | None = None
    target: float | None = None
    weight: float = Field(default=1.0, gt=0.0, le=10.0)
    exponent: float = Field(default=1.0, ge=0.1, le=10.0)


class ContourRsmRequest(BaseModel):
    """The §6.5 grid contract: one factor pair + fixed values."""

    model_id: str
    factor_x: str = Field(min_length=1, max_length=64)
    factor_y: str = Field(min_length=1, max_length=64)
    fixed: dict[str, float] | None = Field(default=None, max_length=10)
    grid_points: int = Field(default=25, ge=MIN_GRID_POINTS, le=MAX_GRID_POINTS)
    mode: Literal["response", "desirability"] = "response"
    response: str | None = None
    goals: list[GoalSpec] | None = Field(default=None, min_length=1, max_length=MAX_RESPONSES)


class ConstraintSpec(BaseModel):
    """One linear coded-factor constraint: a . x <= / >= b."""

    coefficients: list[float] = Field(min_length=1, max_length=MAX_FACTORS)
    sense: Literal["<=", ">="] = "<="
    bound: float


class OptimizeRsmRequest(BaseModel):
    """The §6.6 optimisation contract."""

    model_id: str
    goals: list[GoalSpec] = Field(min_length=1, max_length=MAX_RESPONSES)
    constraints: list[ConstraintSpec] | None = Field(
        default=None, max_length=_MAX_CONSTRAINT_LIST
    )
    pair: tuple[str, str] | None = None
    grid_points: int = Field(default=25, ge=MIN_GRID_POINTS, le=MAX_GRID_POINTS)


def _specs(models: list[FactorSpecModel]) -> list[FactorSpec]:
    return [
        FactorSpec(m.name, float(m.low), float(m.high), m.unit)
        for m in models
    ]


def _demo_specs() -> list[FactorSpec]:
    return list(DEMO_FACTORS)


def _get_rsm_model(request: Request, model_id: str) -> TrainedModelRecord:
    record = request.app.state.model_store.get(model_id)
    if record is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Unknown model_id.")
    if record.task != "rsm-fit":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail=(
                f"model {model_id} is not an RSM surface model "
                f"(task={record.task!r}); use the owning lane for it"
            ),
        )
    return record


def _model_view_block(record: TrainedModelRecord) -> dict[str, Any]:
    """The standard view + the RSM contract block."""
    view = record.view()
    bundle = record.model
    contract = bundle.contract
    return {
        **view,
        "rsm": {
            "kind": contract["kind"],
            "factors": [f["name"] for f in contract["factors"]],
            "responses": contract["responses"],
            "n_terms": len(contract["terms"]),
            "design_meta": contract["design_meta"],
            "df_residual": contract["df_residual"],
            "rank": contract["rank"],
        },
    }


@router.get("/registry")
def registry() -> dict:
    payload = rsm_registry_payload()
    return {
        "designs": payload["designs"],
        "goals": payload["goals"],
        "model": payload["model"],
        "bounds": payload["bounds"],
        "demo_dataset": payload["demo_dataset"],
        "scope_statement": payload["scope_statement"],
    }


@router.post("/design")
def design(req: DesignRsmRequest) -> dict:
    try:
        result = build_design(
            _specs(req.factors),
            req.design,
            alpha_type=req.alpha_type,
            alpha=req.alpha,
            center_points=req.center_points,
            replicates=req.replicates,
            randomize=req.randomize,
            seed=req.seed,
            blocks=req.blocks,
        )
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **result,
        "export_note": (
            "the design table exports through the EXISTING generic table "
            "lane (POST /api/v1/visualization/table) on the frontend -- "
            "NO duplicate export endpoint exists here"
        ),
    }


@router.post("/fit")
def fit(req: FitRsmRequest, request: Request) -> dict:
    if req.dataset_id is not None:
        record = request.app.state.dataset_store.get(req.dataset_id)
        if record is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Unknown dataset_id.")
        frame = record.frame
        factors = req.factors
        if factors is None:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=422,
                detail=(
                    "a dataset fit needs the declared factor ranges "
                    "(factors: [{name, low, high}] -- the coding and the "
                    "design-region vocabulary ride them)"
                ),
            )
        provenance = {
            "source": "dataset",
            "dataset_id": req.dataset_id,
            "dataset_name": record.name,
            "synthetic": False,
        }
    else:
        frame = demo_rsm_dataset()
        factors = [
            FactorSpecModel(name=f.name, low=f.low, high=f.high, unit=f.unit)
            for f in _demo_specs()
        ]
        provenance = {
            "source": "demo",
            "dataset_id": None,
            "dataset_name": "synthetic-rsm-demo",
            "synthetic": True,
            "note": (
                "the seeded SYNTHETIC demonstration experiment -- every "
                "number it produces is a model estimate over KNOWN "
                "generated surfaces, never an experimental fact"
            ),
        }

    design_meta: dict[str, Any] = {}
    if req.design is not None:
        design_meta["type"] = req.design
    if req.alpha_type is not None:
        design_meta["alpha_type"] = req.alpha_type
    if req.alpha is not None:
        design_meta["alpha"] = float(req.alpha)
    if req.dataset_id is None:
        design_meta.update(
            {"type": "ccd", "alpha_type": "rotatable", "alpha": 1.681793}
        )

    started_at = _now_iso()
    try:
        result = fit_response_surface(
            frame,
            _specs(factors),
            list(req.response_columns),
            design_meta=design_meta,
            confidence=req.confidence,
            provenance_extra={
                **provenance,
                "toolkit_version": _toolkit_version,
            },
        )
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc

    bundle = result["bundle"]
    record = TrainedModelRecord(
        model_id=f"rsm-{_uuid12()}",
        model=bundle,
        name=(
            f"rsm-{'-'.join(r[:12] for r in bundle.response_names[:3])}"
        ),
        task="rsm-fit",
        dataset_id=req.dataset_id,
        target="|".join(bundle.response_names),
        features=[f.name for f in bundle.factors],
        n_train_rows=result["n_runs"],
        best_params={},
        cv_scores=None,
        cv_mean=None,
        cv_std=None,
        fit_seconds=result["fit_seconds"],
    )
    request.app.state.model_store.register(record)

    dataset_view = (
        request.app.state.dataset_store.get(req.dataset_id).view()
        if req.dataset_id is not None
        else {
            "dataset_id": None,
            "name": provenance["dataset_name"],
            "synthetic": True,
        }
    )
    run = ExperimentRunRecord(
        run_id=f"run-{_uuid12()}",
        family="ml.rsm",
        mode="sync",
        status="completed",
        job_id=None,
        dataset={
            "dataset_id": dataset_view.get("dataset_id"),
            "name": dataset_view.get("name"),
            "source_filename": dataset_view.get("source_filename"),
            "size_bytes": dataset_view.get("size_bytes"),
            "shape": dataset_view.get("shape"),
            "synthetic": provenance["synthetic"],
        },
        feature_configuration={
            "factors": [f.name for f in bundle.factors],
            "responses": bundle.response_names,
            "term_count": len(bundle.terms),
        },
        preprocessing={
            "enabled": False,
            "note": "the second-order fit rides coded factors -- no scaling lane",
        },
        algorithm={
            "name": "rsm",
            "label": "Response Surface Methodology (second-order)",
            "task": "rsm-fit",
            "toolkit_version": _toolkit_version,
        },
        hyperparameters={
            "design_meta": design_meta,
            "confidence": req.confidence,
        },
        random_seed=0,  # the designed-experiment fit is deterministic -- no seed lane
        cross_validation={
            "cv_folds": None,
            "strategy": "not_applicable (the designed-experiment fit "
            "has no k-fold lane -- the fit statistics ride the ANOVA)",
        },
        evaluation={
            "per_response_statistics": {
                r: {
                    "r2": result["fit_statistics"][r]["r2"],
                    "adj_r2": result["fit_statistics"][r]["adj_r2"],
                    "pred_r2": result["fit_statistics"][r]["pred_r2"],
                    "rmse": result["fit_statistics"][r]["rmse"],
                }
                for r in bundle.response_names
            },
            "primary_metric": "r2",
        },
        model_artifact={
            "model_id": record.model_id,
            "name": record.name,
            "n_train_rows": result["n_runs"],
        },
        environment={
            **environment_metadata(),
            "toolkit_version": _toolkit_version,
        },
        logs={"fit_seconds": result["fit_seconds"]},
        started_at=started_at,
        completed_at=_now_iso(),
    )
    request.app.state.run_store.register(run)

    return {
        "model": _model_view_block(record),
        "run_id": run.run_id,
        "provenance": provenance,
        "configuration": {
            "factors": [f.view() for f in bundle.factors],
            "response_columns": list(req.response_columns),
            "design_meta": design_meta,
            "confidence": req.confidence,
        },
        "data": {
            "n_runs": result["n_runs"],
            "dropped_rows": result["dropped_rows"],
            "terms": result["terms"],
            "hierarchy_notes": result["hierarchy_notes"],
            "aliasing": result["aliasing"],
        },
        "coefficients": result["coefficients"],
        "anova": result["anova"],
        "diagnostics": result["diagnostics"],
        "fit_statistics": result["fit_statistics"],
        "fit_seconds": result["fit_seconds"],
    }


@router.post("/predict")
def predict(req: PredictRsmRequest, request: Request) -> dict:
    record = _get_rsm_model(request, req.model_id)
    points = pd.DataFrame(req.points)
    try:
        result = predict_surface(record.model, points, confidence=req.confidence)
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_id": req.model_id,
        **result,
        "note": (
            "predictions are fitted model values with the mean CI and the "
            "new-observation PI at the request's confidence; the zone "
            "vocabulary flags extrapolated points (outside the declared "
            "factor ranges), never silently extrapolated"
        ),
    }


@router.post("/contour")
def contour(req: ContourRsmRequest, request: Request) -> dict:
    record = _get_rsm_model(request, req.model_id)
    goals_payload = (
        [g.model_dump() for g in req.goals] if req.goals is not None else None
    )
    try:
        result = contour_grid(
            record.model,
            factor_x=req.factor_x,
            factor_y=req.factor_y,
            fixed_actual=req.fixed,
            grid_points=req.grid_points,
            mode=req.mode,
            response=req.response,
            goals=goals_payload,
        )
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"model_id": req.model_id, **result}


@router.post("/optimize")
def optimize(req: OptimizeRsmRequest, request: Request) -> dict:
    record = _get_rsm_model(request, req.model_id)
    constraints_payload = (
        [c.model_dump() for c in req.constraints]
        if req.constraints is not None
        else None
    )
    try:
        result = optimize_desirability(
            record.model,
            [g.model_dump() for g in req.goals],
            constraints=constraints_payload,
            pair=(req.pair[0], req.pair[1]) if req.pair is not None else None,
            grid_points=req.grid_points,
        )
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_id": req.model_id,
        **result,
        "goals_vocabulary": list(GOALS),
    }


@router.get("/models")
def models(request: Request) -> dict:
    """The session's RSM models — the comparison surface."""
    views = []
    for record in request.app.state.model_store.list():
        got = request.app.state.model_store.get(record["model_id"])
        if got is not None and got.task == "rsm-fit":
            views.append(_model_view_block(got))
    return {"models": views}

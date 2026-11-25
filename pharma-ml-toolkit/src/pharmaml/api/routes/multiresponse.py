"""``/api/v1/multiresponse`` — the generic multi-response ML lane
(part17 §5, the campaign queue's M1 lane) over the REAL in-tree
backend (:mod:`pharmaml.ml.multiresponse`: the strategy registry with
its stated mechanisms, the leakage-safe split-first trainer, the
per-target evaluation surface, and the full §5.6 persistence
contract).

* ``GET /api/v1/multiresponse/registry`` — the registry vocabulary
  verbatim: the strategies (each with its mechanism — native /
  wrapper / chain — the sample-weight flag, the notes, the
  wrapper/chain base choices), the missing-value strategies
  (complete_case always; per_target the wrapper lane's masked
  multi-task), the metric names, the demo dataset descriptor, and the
  honest scope statement.
* ``POST /api/v1/multiresponse/train`` — the full round trip over a
  registered typed dataset (``dataset_id`` + the request's own feature
  and >=2 target column selections) or the route's own seeded
  SYNTHETIC demo frame (labelled synthetic wherever it travels).  The
  response carries the per-target metrics (R2/RMSE/MAE/MSE/explained
  variance per target, each with its own test-row count), the uniform
  and user-weighted aggregates, the residual analysis sample, the
  target correlation matrix, the k-fold CV summary, the usable-rows
  census, the split report, and the COMPLETE §5.6 training contract.
  The fitted bundle registers in the session model store
  (``MultiResponseBundle`` — predictable, exportable through the
  existing artifact lane) and the run records in the §15.2 run
  lineage.  Sync-only light compute (bounded rows x features x
  targets — the bounds 422'd up front, never silently clamped).
* ``POST /api/v1/multiresponse/predict`` — multi-column predictions
  in the STABLE contract target order with per-target inverse
  scaling, from inline rows (feature columns by name, any order).
* ``GET /api/v1/multiresponse/models`` — the session's multi-response
  models (the comparison surface: every trained model's per-target
  metric block side by side).

The honest notes: the mechanism is stated per strategy, never
implied; ``per_target`` missing handling rides the wrapper mechanism
only (a native/chain estimator needs the complete Y block — a
complete_case-only request for those is accepted, a per_target request
is a typed 422); metrics are computed per target on raw-unit
predictions after inverse scaling; the aggregate is reported
ALONGSIDE the per-target block, never instead (a weak target can not
hide behind a strong aggregate); and the demo dataset is synthetic —
every number it produces is a model estimate over generated data,
never an experimental fact.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml import __version__ as _toolkit_version
from pharmaml.api.runs import ExperimentRunRecord
from pharmaml.api.stores import TrainedModelRecord
from pharmaml.ml.multiresponse import (
    DEMO_FEATURES,
    DEMO_TARGETS,
    MAX_SAMPLE_ROWS,
    METRIC_NAMES,
    demo_multiresponse_dataset,
    environment_metadata,
    predict_multi_response,
    strategy_registry_payload,
    train_multi_response,
)

router = APIRouter(prefix="/multiresponse", tags=["multiresponse"])

_MAX_SEED = 2147483647
_MAX_ROWS = 5000
_MAX_FEATURES = 100
_MAX_TARGETS = 32
_MAX_PARAM_KEYS = 12
_MAX_ROWS_PREDICT = 500


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _uuid12() -> str:
    return uuid.uuid4().hex[:12]


class TrainMultiResponseRequest(BaseModel):
    """The §5.4 input contract over a real frame.

    ``dataset_id`` selects a registered typed dataset; leaving it None
    rides the seeded SYNTHETIC demo frame (labelled).  Features and
    >=2 numeric targets are the request's own column selections.  The
    missing-value strategy, the split/CV configuration, per-target
    scaling, preprocessing, the strategy key + its hyperparameters,
    the sample-weight column, and the target-weight vector all ride
    this one body -- each validated by the backend's own typed errors.
    """

    dataset_id: str | None = None
    feature_columns: list[str] = Field(
        default_factory=lambda: list(DEMO_FEATURES),
        min_length=1,
        max_length=_MAX_FEATURES,
    )
    target_columns: list[str] = Field(
        default_factory=lambda: list(DEMO_TARGETS),
        min_length=2,
        max_length=_MAX_TARGETS,
    )
    strategy: str = "ridge"
    params: dict[str, Any] | None = Field(
        default=None, max_length=_MAX_PARAM_KEYS
    )
    preprocess: bool = True
    target_scaling: bool = False
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)
    missing: Literal["complete_case", "per_target"] = "complete_case"
    group_column: str | None = None
    sample_weight_column: str | None = None
    target_weights: list[float] | None = Field(
        default=None, min_length=2, max_length=_MAX_TARGETS
    )
    random_state: int = Field(default=42, ge=0, le=_MAX_SEED)
    target_units: dict[str, str] | None = None


class PredictMultiResponseRequest(BaseModel):
    """Inline rows -> multi-column predictions in the contract order."""

    model_id: str
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=_MAX_ROWS_PREDICT)


def _resolve_frame(
    request: Request, req: TrainMultiResponseRequest
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The frame source: a registered typed dataset or the demo frame."""
    if req.dataset_id is not None:
        record = request.app.state.dataset_store.get(req.dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown dataset_id.")
        frame = record.frame
        provenance = {
            "source": "dataset",
            "dataset_id": req.dataset_id,
            "dataset_name": record.name,
            "synthetic": False,
        }
        return frame, provenance
    frame = demo_multiresponse_dataset()
    provenance = {
        "source": "demo",
        "dataset_id": None,
        "dataset_name": "synthetic-multiresponse-demo",
        "synthetic": True,
        "note": (
            "the seeded synthetic demonstration frame -- every number "
            "it produces is a model estimate over generated data, "
            "never an experimental fact"
        ),
    }
    return frame, provenance


def _model_view_block(record: TrainedModelRecord) -> dict[str, Any]:
    """The standard view + the multiresponse contract block."""
    view = record.view()
    bundle = record.model
    contract = getattr(bundle, "contract", None) or {}
    return {
        **view,
        "multiresponse": {
            "target_names": contract.get("target_names"),
            "target_order": contract.get("target_order"),
            "mechanism": contract.get("estimator_strategy", {}).get("mechanism"),
            "missing_strategy": contract.get("missing_strategy"),
            "per_target_metrics": contract.get("per_target_metrics"),
            "aggregate_metrics": contract.get("aggregate_metrics"),
            "n_test_rows": contract.get("n_test_rows"),
        },
    }


@router.get("/registry")
def registry() -> dict:
    payload = strategy_registry_payload()
    return {
        "strategies": payload["strategies"],
        "missing_strategies": payload["missing_strategies"],
        "metrics": payload["metrics"],
        "max_sample_rows": payload["max_sample_rows"],
        "demo_dataset": payload["demo_dataset"],
        "bounds": {
            "max_rows": _MAX_ROWS,
            "max_feature_columns": _MAX_FEATURES,
            "max_target_columns": _MAX_TARGETS,
            "max_predict_rows": _MAX_ROWS_PREDICT,
            "cv_folds": [2, 10],
        },
        "scope_statement": payload["scope_statement"],
    }


@router.post("/train")
def train(req: TrainMultiResponseRequest, request: Request) -> dict:
    frame, provenance = _resolve_frame(request, req)
    if len(frame) > _MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"dataset too large for the sync multiresponse lane "
                f"({len(frame)} rows > {_MAX_ROWS}); the bounds are "
                "stated, never silently clamped"
            ),
        )
    started_at = _now_iso()
    try:
        result = train_multi_response(
            frame,
            feature_columns=list(req.feature_columns),
            target_columns=list(req.target_columns),
            strategy_key=req.strategy,
            params=req.params,
            preprocess=req.preprocess,
            target_scaling=req.target_scaling,
            test_size=req.test_size,
            cv_folds=req.cv_folds,
            missing=req.missing,
            group_column=req.group_column,
            sample_weight_column=req.sample_weight_column,
            target_weights=req.target_weights,
            random_state=req.random_state,
            target_units=req.target_units,
            environment_extra={"toolkit_version": _toolkit_version},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    contract = result["contract"]
    bundle = result["bundle"]
    record = TrainedModelRecord(
        model_id=f"mresp-{_uuid12()}",
        model=bundle,
        name=(
            f"multiresponse-{req.strategy}"
            f"-{'-'.join(t[:12] for t in bundle.target_names[:3])}"
        ),
        task="multiresponse-regression",
        dataset_id=req.dataset_id,
        target="|".join(bundle.target_names),
        features=list(bundle.feature_names),
        n_train_rows=contract["n_train_rows"],
        best_params=dict(req.params or {}),
        cv_scores=result["cv"]["folds_table"],
        cv_mean=result["aggregate_metrics"]["uniform"]["r2"],
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
        family="ml.multiresponse",
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
            "feature_names": contract["feature_names"],
            "target_names": contract["target_names"],
            "target_order": contract["target_order"],
            "group_column": req.group_column,
        },
        preprocessing={
            "enabled": req.preprocess,
            "impute": contract["preprocessing"]["impute"],
            "feature_scaling": contract["preprocessing"]["feature_scaling"],
            "target_scaling": req.target_scaling,
        },
        algorithm={
            "name": req.strategy,
            "label": contract["estimator_strategy"]["label"],
            "mechanism": contract["estimator_strategy"]["mechanism"],
            "task": "multiresponse-regression",
            "toolkit_version": _toolkit_version,
        },
        hyperparameters={
            "requested": dict(req.params or {}),
            "missing_strategy": req.missing,
            "sample_weight_column": req.sample_weight_column,
        },
        random_seed=req.random_state,
        cross_validation={
            "cv_folds": result["cv"]["folds"],
            "strategy": "kfold(shuffle)" if req.group_column is None else "groupkfold",
        },
        evaluation={
            "per_target": contract["per_target_metrics"],
            "aggregate_uniform": contract["aggregate_metrics"]["uniform"],
            "aggregate_weighted": contract["aggregate_metrics"]["weighted"],
            "primary_metric": "r2",
            "metrics": list(METRIC_NAMES),
        },
        model_artifact={
            "model_id": record.model_id,
            "name": record.name,
            "n_train_rows": contract["n_train_rows"],
            "n_test_rows": contract["n_test_rows"],
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
            "strategy": req.strategy,
            "mechanism": contract["estimator_strategy"]["mechanism"],
            "params": dict(req.params or {}),
            "preprocess": req.preprocess,
            "target_scaling": req.target_scaling,
            "test_size": req.test_size,
            "cv_folds": req.cv_folds,
            "missing": req.missing,
            "group_column": req.group_column,
            "sample_weight_column": req.sample_weight_column,
            "target_weights": req.target_weights,
            "random_state": req.random_state,
        },
        "data": result["data"],
        "per_target_metrics": result["per_target_metrics"],
        "aggregate_metrics": result["aggregate_metrics"],
        "residuals": result["residuals"],
        "target_correlation_matrix": result["target_correlation_matrix"],
        "cv": result["cv"],
        "contract": contract,
        "fit_seconds": result["fit_seconds"],
        "max_sample_rows": MAX_SAMPLE_ROWS,
    }


@router.post("/predict")
def predict(req: PredictMultiResponseRequest, request: Request) -> dict:
    record = request.app.state.model_store.get(req.model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    if record.task != "multiresponse-regression":
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is not a multi-response model "
                f"(task={record.task!r}); use the ml predict lane for it"
            ),
        )
    rows = pd.DataFrame(req.rows)
    try:
        result = predict_multi_response(record.model, rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_id": req.model_id,
        **result,
        "note": (
            "predictions arrive in the model's STABLE contract target "
            "order (target_names), in raw target units (per-target "
            "inverse scaling applied)"
        ),
    }


@router.get("/models")
def models(request: Request) -> dict:
    """The session's multi-response models — the comparison surface."""
    views = []
    for record in request.app.state.model_store.list():
        got = request.app.state.model_store.get(record["model_id"])
        if got is not None and got.task == "multiresponse-regression":
            views.append(_model_view_block(got))
    return {"models": views, "metrics": list(METRIC_NAMES)}

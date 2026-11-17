"""``/api/v1/ml`` — the ML workspace's API surface over the REAL
training backend (part10's queued row).

* ``GET /api/v1/ml/models`` — the registry catalog projection
  (:func:`pharmaml.ml.models.list_models`, one source of truth).
* ``POST /api/v1/ml/train`` — one registry model trained by the real
  :func:`pharmaml.ml.training.train_model` on a registered dataset
  (sync or managed job), the fitted estimator kept in the session
  model store for prediction and artifact export.
* ``POST /api/v1/ml/predict`` — predictions (and optional
  probabilities) from a session-trained model on inline rows or on a
  registered dataset's feature matrix.

``n_jobs`` passes through verbatim, including ``-1`` (all cores) —
the Phase-Eight passthrough clause, previewed here.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.models import list_models, resolve_model_name
from pharmaml.ml.training import train_model

from ..jobs import CancelledJob
from ..stores import TrainedModelRecord, json_safe

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    dataset_id: str
    model: str = Field(min_length=1)
    task: Literal["classification", "regression"] | None = None
    tune: bool = True
    cv: int = Field(default=5, ge=2, le=10)
    n_jobs: int = Field(default=-1, ge=-1, le=64)
    preprocess: bool = True
    random_state: int = 42
    model_params: dict[str, Any] | None = None
    mode: Literal["job", "sync"] = "job"


class PredictRequest(BaseModel):
    model_id: str
    rows: list[dict[str, Any]] | None = Field(default=None, max_length=500)
    dataset_id: str | None = None
    predict_proba: bool = False
    mode: Literal["job", "sync"] = "sync"

    def _exactly_one_source(self) -> None:
        if (self.rows is None) == (self.dataset_id is None):
            raise ValueError(
                "give either rows=[...] or dataset_id=..., not both/neither"
            )


@router.get("/models")
def model_catalog(
    request: Request,
    task: Literal["classification", "regression", "both"] | None = None,
    only_available: bool = True,
) -> dict:
    return {
        "models": list_models(task=task, only_available=only_available)
    }


@router.get("/trained")
def trained_models(request: Request) -> dict:
    return {"models": request.app.state.model_store.list()}


@router.get("/trained/{model_id}")
def trained_model_view(model_id: str, request: Request) -> dict:
    record = request.app.state.model_store.get(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    return record.view()


def _train_body(
    dataset_id: str,
    dataset_X: pd.DataFrame,
    dataset_y: pd.Series,
    target: str,
    model_name: str,
    task: str,
    tune: bool,
    cv: int,
    n_jobs: int,
    preprocess: bool,
    random_state: int,
    model_params: dict | None,
    model_store,
    cancel_event=None,
    manager=None,
    job_id: str | None = None,
) -> dict:
    """Job/sync body: the REAL train_model call + store registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the fit stage.")
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 10, "preparing features")
    result = train_model(
        model_name,
        task=task,
        X_train=dataset_X,
        y_train=dataset_y,
        tune=tune,
        cv=cv,
        n_jobs=n_jobs,
        preprocess=preprocess,
        random_state=random_state,
        model_params=model_params,
    )
    record = TrainedModelRecord(
        model_id=f"mdl-{uuid.uuid4().hex[:12]}",
        model=result.model,
        name=result.name,
        task=result.task,
        dataset_id=dataset_id,
        target=target,
        features=[str(c) for c in dataset_X.columns],
        n_train_rows=len(dataset_X),
        best_params=result.best_params,
        cv_scores=(
            [float(s) for s in result.cv_scores.tolist()]
            if result.cv_scores is not None else None
        ),
        cv_mean=None if result.cv_mean is None else float(result.cv_mean),
        cv_std=None if result.cv_std is None else float(result.cv_std),
        fit_seconds=float(result.fit_seconds),
    )
    model_store.register(record)
    return {"model": record.view()}


@router.post("/train")
def train_model_route(req: TrainRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    task = req.task or dataset.task
    # pre-validate the registry key up front (both modes): an unknown
    # model name is a client error, not an asynchronous job failure
    resolve_model_name(req.model)
    manager = request.app.state.job_manager
    model_store = request.app.state.model_store
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_train_body(
                req.dataset_id, dataset.X, dataset.y, dataset.target,
                req.model, task, req.tune, req.cv, req.n_jobs,
                req.preprocess, req.random_state, req.model_params,
                model_store,
            ),
        }
    record = manager.submit(
        "ml.train",
        "ml",
        _train_body,
        configuration={
            "dataset_id": req.dataset_id,
            "model": req.model,
            "task": task,
            "tune": req.tune,
            "cv": req.cv,
            "n_jobs": req.n_jobs,
            "preprocess": req.preprocess,
            "random_state": req.random_state,
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        dataset_y=dataset.y,
        target=dataset.target,
        model_name=req.model,
        task=task,
        tune=req.tune,
        cv=req.cv,
        n_jobs=req.n_jobs,
        preprocess=req.preprocess,
        random_state=req.random_state,
        model_params=req.model_params,
        model_store=model_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


def _predict_body(
    model_record: TrainedModelRecord,
    X: pd.DataFrame,
    want_proba: bool,
    cancel_event=None,
) -> dict:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the predict stage.")
    estimator = model_record.model
    predictions = estimator.predict(X)
    payload: dict[str, Any] = {
        "model_id": model_record.model_id,
        "n_rows": len(X),
        "predictions": [json_safe(p) for p in predictions],
    }
    if want_proba:
        if not hasattr(estimator, "predict_proba"):
            raise ModelError(
                f"model {model_record.name!r} does not expose "
                "predict_proba; ask for predictions only"
            )
        # sklearn's Pipeline proxies predict_proba; the class order
        # lives on the final step (or the estimator itself).
        final_step = getattr(estimator, "steps", None)
        classes = None
        if final_step is not None:
            classes = getattr(final_step[-1][1], "classes_", None)
        if classes is None:
            classes = getattr(estimator, "classes_", None)
        proba = estimator.predict_proba(X)
        payload["probabilities"] = {
            "classes": json_safe(list(classes) if classes is not None else []),
            "rows": [[json_safe(v) for v in row] for row in proba],
        }
    return payload


@router.post("/predict")
def predict(req: PredictRequest, request: Request) -> dict:
    try:
        req._exactly_one_source()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model_store = request.app.state.model_store
    model_record = model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    if req.rows is not None:
        X = pd.DataFrame(req.rows)
    else:
        dataset = request.app.state.dataset_store.get(
            req.dataset_id or ""
        )
        if dataset is None:
            raise HTTPException(
                status_code=404, detail="Unknown dataset_id."
            )
        X = dataset.X
    if req.mode == "sync":
        try:
            return {"mode": "sync", **_predict_body(
                model_record, X, req.predict_proba
            )}
        except (DataValidationError, ModelError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    manager = request.app.state.job_manager
    record = manager.submit(
        "ml.predict",
        "ml",
        _predict_body,
        configuration={
            "model_id": req.model_id,
            "n_rows": len(X),
            "predict_proba": req.predict_proba,
        },
        model_record=model_record,
        X=X,
        want_proba=req.predict_proba,
    )
    return {"mode": "job", "job": record.public()}

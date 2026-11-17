"""``/api/v1/applicability-domain`` — the AD analysis surface over the
REAL backend (part10's row four).

* ``GET /api/v1/applicability-domain/catalog`` — the two entry
  points + their defaults (one source: the backend signatures).
* ``POST /api/v1/applicability-domain/knn`` —
  :func:`pharmaml.ml.applicability.knn_applicability_domain` (the
  distance-only half: StandardScaler + kNN, self-match excluded).
* ``POST /api/v1/applicability-domain/analyze`` —
  :func:`pharmaml.ml.applicability.applicability_domain` over a
  session-trained model (``/api/v1/ml/train``) with the REAL
  bootstrap-uncertainty half: the backend's own
  ``progress(done, total)`` callback (one call per fitted ensemble
  member) drives ``training.progress`` events and the job record;
  ``uncertainty=False`` skips the bootstrap half honestly.
  ``figure=true`` renders
  :func:`pharmaml.viz.ml_plots.plot_applicability_domain` and
  registers it as a downloadable artifact.
* ``POST /api/v1/applicability-domain/domain-box`` —
  :func:`pharmaml.ml.applicability.check_domain_box` (the persisted
  feature-range box checked against new rows).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.ml.applicability import (
    applicability_domain,
    check_domain_box,
    knn_applicability_domain,
)
from pharmaml.viz._compat import close_figure
from pharmaml.viz.export import export_figure
from pharmaml.viz.ml_plots import plot_applicability_domain

from ..events import EventType
from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/applicability-domain", tags=["applicability-domain"])

_MAX_TRAIN_ROWS = 2000
_MAX_TEST_ROWS = 1000
_MAX_FEATURES = 50


class KnnRequest(BaseModel):
    x_train: list[list[float]] = Field(
        min_length=10, max_length=_MAX_TRAIN_ROWS
    )
    x_test: list[list[float]] = Field(min_length=1, max_length=_MAX_TEST_ROWS)
    k: int = Field(default=5, ge=1, le=50)
    percentile: float = Field(default=95.0, ge=50.0, lt=100.0)
    metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean"


class AnalyzeRequest(BaseModel):
    model_id: str
    dataset_id: str | None = None
    x_test: list[list[float]] = Field(min_length=1, max_length=_MAX_TEST_ROWS)
    k: int = Field(default=5, ge=1, le=50)
    percentile: float = Field(default=95.0, ge=50.0, lt=100.0)
    n_models: int = Field(default=30, ge=1, le=200)
    uncertainty: bool = True
    random_state: int = Field(default=0, ge=0, le=2**31 - 1)
    figure: bool = False
    mode: Literal["job", "sync"] = "job"


class DomainBoxRequest(BaseModel):
    metadata: dict[str, Any]
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=2000)


@router.get("/catalog")
def catalog(request: Request) -> dict:
    return {
        "entry_points": [
            {
                "route": "/knn",
                "backend": "knn_applicability_domain",
                "defaults": {"k": 5, "percentile": 95.0, "metric": "euclidean"},
            },
            {
                "route": "/analyze",
                "backend": "applicability_domain",
                "defaults": {
                    "k": 5, "percentile": 95.0, "n_models": 30,
                    "uncertainty": True, "random_state": 0,
                },
            },
            {
                "route": "/domain-box",
                "backend": "check_domain_box (persisted feature_ranges)",
            },
        ],
        "result_columns": [
            "name", "predicted", "prediction_sd", "knn_distance",
            "inside_ad", "low_uncertainty", "reliable",
        ],
    }


def _matrix(rows: list[list[float]], what: str) -> np.ndarray:
    try:
        matrix = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{what} must be numeric rows: {exc}"
        ) from exc
    if matrix.ndim != 2:
        raise HTTPException(
            status_code=422, detail=f"{what} must be a 2-D rows x features array"
        )
    if matrix.shape[1] > _MAX_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"{what} has {matrix.shape[1]} features; the limit is {_MAX_FEATURES}",
        )
    return matrix


@router.post("/knn")
def knn(req: KnnRequest, request: Request) -> dict:
    train = _matrix(req.x_train, "x_train")
    test = _matrix(req.x_test, "x_test")
    if train.shape[1] != test.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"x_train has {train.shape[1]} features; x_test has "
                f"{test.shape[1]} (they must match)"
            ),
        )
    try:
        result = knn_applicability_domain(
            train, test, k=req.k, percentile=req.percentile, metric=req.metric
        )
    except (DataValidationError, ModelError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    frame = result.to_frame()
    return {
        "k": int(result.k),
        "percentile": float(result.percentile),
        "distance_threshold": float(result.distance_threshold),
        "train_knn_mean": float(np.mean(result.train_knn_mean)),
        "n_train": len(result.train_knn_mean),
        "n_test": len(result.test_knn_mean),
        "n_inside": int(np.count_nonzero(result.inside_ad)),
        "frame": json_safe(frame.to_dict(orient="records")),
    }


def _analyze_body(
    model_record,
    X_train,
    y_train,
    x_test: np.ndarray,
    k: int,
    percentile: float,
    n_models: int,
    uncertainty: bool,
    random_state: int,
    figure: bool,
    artifact_store=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL applicability_domain analysis."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the bootstrap stage.")

    def progress_adapter(done: int, total: int) -> None:
        # the backend fires report(done, total) once per fitted
        # ensemble member -- REAL bootstrap telemetry
        if manager is not None and job_id is not None:
            pct = min(99, int(100.0 * done / max(1, total)))
            manager.report_progress(
                job_id, pct, f"bootstrap member {done}/{total}",
                event=EventType.TRAINING_PROGRESS,
            )

    try:
        result = applicability_domain(
            model_record.model,
            X_train,
            y_train,
            x_test,
            k=k,
            percentile=percentile,
            n_models=n_models,
            random_state=random_state,
            uncertainty=uncertainty,
            progress=progress_adapter,
        )
    except (DataValidationError, ModelError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = result.as_dict()
    frame = result.to_frame()
    artifact: dict | None = None
    if figure:
        try:
            fig = plot_applicability_domain(result)
        except (DataValidationError, DependencyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if artifact_store is not None:
            with tempfile.TemporaryDirectory(prefix="pharmaml-ad-") as tmp:
                target = Path(tmp) / "applicability-domain.png"
                export_figure(fig, target)
                data = target.read_bytes()
            record = artifact_store.register(
                kind="figure",
                filename="applicability-domain.png",
                content_type="image/png",
                data=data,
                meta={"model_id": model_record.model_id, "k": k},
            )
            artifact = record.view()
        close_figure(fig)
    return {
        "analysis": {
            **json_safe(payload),
            "frame": json_safe(frame.to_dict(orient="records")),
        },
        "artifact": artifact,
    }


@router.post("/analyze")
def analyze(req: AnalyzeRequest, request: Request) -> dict:
    model_store = request.app.state.model_store
    model_record = model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    dataset_store = request.app.state.dataset_store
    dataset_id = req.dataset_id or model_record.dataset_id
    dataset = dataset_store.get(dataset_id) if dataset_id else None
    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "the model's training dataset is not registered in this "
                "session; re-upload it (POST /api/v1/datasets) and pass "
                "dataset_id"
            ),
        )
    X_train = dataset.X
    y_train = dataset.y
    x_test = _matrix(req.x_test, "x_test")
    if X_train.shape[1] != x_test.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the model was trained on {X_train.shape[1]} features "
                f"({list(dataset.X.columns)}); x_test has "
                f"{x_test.shape[1]}"
            ),
        )
    # bind nothing else: the body imports CancelledJob at module level
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_analyze_body(
                model_record, X_train, y_train, x_test, req.k,
                req.percentile, req.n_models, req.uncertainty,
                req.random_state, req.figure,
                artifact_store=request.app.state.artifact_store,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "ad.analyze",
        "machine-learning",
        _analyze_body,
        configuration={
            "model_id": req.model_id,
            "dataset_id": dataset_id,
            "n_test": len(req.x_test),
            "k": req.k,
            "percentile": req.percentile,
            "n_models": req.n_models,
            "uncertainty": req.uncertainty,
            "figure": req.figure,
        },
        model_record=model_record,
        X_train=X_train,
        y_train=y_train,
        x_test=x_test,
        k=req.k,
        percentile=req.percentile,
        n_models=req.n_models,
        uncertainty=req.uncertainty,
        random_state=req.random_state,
        figure=req.figure,
        artifact_store=request.app.state.artifact_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/domain-box")
def domain_box(req: DomainBoxRequest, request: Request) -> dict:
    import pandas as pd

    frame = pd.DataFrame(req.rows)
    report = check_domain_box(req.metadata, frame)
    if report is None:
        return {
            "checked": False,
            "reason": "the metadata carries no feature_ranges box",
            "n_rows": len(frame),
        }
    return {
        "checked": True,
        "n_rows": int(report.n_rows),
        "n_out": int(report.n_out),
        "fraction_out": float(report.fraction_out),
        "below": {k: int(v) for k, v in report.below.items()},
        "above": {k: int(v) for k, v in report.above.items()},
        "out": [bool(v) for v in report.out_mask.tolist()],
    }

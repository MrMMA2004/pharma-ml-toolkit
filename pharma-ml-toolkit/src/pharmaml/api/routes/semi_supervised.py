"""``/api/v1/semi-supervised`` — the semi-supervised workspace's API
surface over the REAL backend (part10's queued row).

* ``POST /api/v1/semi-supervised/train`` —
  :func:`pharmaml.ml.semi_supervised.train_semi_supervised` on a
  registered dataset.  Unlabeled rows arrive the way the backend
  understands them: the target column already carries the marker
  (``-1`` by default), or the request carves a seeded fraction of
  labels out (``mask_fraction``) to simulate the unlabeled pool —
  the same data prep the Studio's workflow performs.  The fitted
  model registers in the session model store (task:
  classification) so predictions run through the single
  ``/api/v1/ml/predict`` path.
* ``POST /api/v1/semi-supervised/evaluate`` —
  :func:`pharmaml.ml.semi_supervised.evaluate_semi_supervised`: the
  held-out comparison of the semi-supervised method against its
  fully-labeled baseline, n repeats, honest metrics.

Methods: ``self_training`` / ``label_propagation`` /
``label_spreading`` — the backend's own Literal, never a re-declared
enum.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.ml.models import build_model, resolve_model_name
from pharmaml.ml.semi_supervised import (
    SemiSupervisedResult,
    evaluate_semi_supervised,
    train_semi_supervised,
    validate_unlabeled,
)

from ..jobs import CancelledJob
from ..stores import TrainedModelRecord, json_safe

router = APIRouter(
    prefix="/semi-supervised", tags=["semi-supervised"]
)

_Method = Literal["self_training", "label_propagation", "label_spreading"]


class SemiSupervisedTrainRequest(BaseModel):
    dataset_id: str
    method: _Method = "self_training"
    unlabeled_marker: Any = -1
    mask_fraction: float | None = Field(
        default=None, ge=0.05, le=0.9,
        description=(
            "carve this fraction of labels out as unlabeled (seeded); "
            "omit when the target column already carries the marker"
        ),
    )
    random_state: int = 42
    base_estimator: str | None = Field(
        default=None,
        description=(
            "registry classification model backing self-training "
            "(must expose predict_proba; default LogisticRegression)"
        ),
    )
    threshold: float = Field(default=0.75, gt=0.0, lt=1.0)
    k_best: int = Field(default=10, ge=1, le=100)
    max_iter: int = Field(default=10, ge=1, le=100)
    kernel: Literal["knn", "rbf"] = "knn"
    n_neighbors: int = Field(default=7, ge=1, le=50)
    gamma: float = Field(default=20.0, gt=0.0)
    mode: Literal["job", "sync"] = "job"


class SemiSupervisedEvaluateRequest(BaseModel):
    dataset_id: str
    method: _Method = "self_training"
    test_size: float = Field(default=0.25, gt=0.0, lt=0.9)
    n_repeats: int = Field(default=3, ge=1, le=10)
    random_state: int = 42
    mode: Literal["job", "sync"] = "sync"


def _masked_labels(y, mask_fraction: float | None, marker: Any,
                   random_state: int) -> tuple[Any, dict]:
    """Marker-native labels, or a seeded synthetic unlabeled carve-out."""
    if mask_fraction is None:
        return y, {}
    y_arr = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = round(len(y_arr) * float(mask_fraction))
    n = min(n, len(y_arr) - 2)  # keep at least two labeled rows
    if n < 1:
        raise HTTPException(
            status_code=422,
            detail="dataset too small for the requested mask_fraction",
        )
    indices = rng.choice(len(y_arr), size=n, replace=False)
    masked = y_arr.copy()
    masked[indices] = marker
    return masked, {"masked_indices": [int(i) for i in sorted(indices)]}


def _semi_view(result: SemiSupervisedResult, dataset_id: str,
               target: str, model_id: str | None) -> dict:
    """The JSON-safe result view (labels preview, honest counts)."""
    labels = np.asarray(result.labels_)
    confidence = (
        np.asarray(result.confidence)
        if result.confidence is not None else None
    )
    payload = {
        "method": result.method,
        "params": json_safe(result.params),
        "dataset_id": dataset_id,
        "target": target,
        "n_labeled": int(result.n_labeled),
        "n_unlabeled": int(result.n_unlabeled),
        "n_pseudo_labeled": int(
            np.count_nonzero(np.asarray(result.pseudo_label_mask))
        ),
        "classes": json_safe(list(result.classes_)),
        "fit_seconds": float(result.fit_seconds),
        "labels_preview": json_safe(labels[:10].tolist()),
        "n_labels": int(labels.shape[0]),
    }
    if confidence is not None:
        payload["confidence_summary"] = {
            "min": json_safe(float(confidence.min())),
            "mean": json_safe(float(confidence.mean())),
            "max": json_safe(float(confidence.max())),
        }
    if model_id is not None:
        payload["model_id"] = model_id
        payload["predict_url"] = "/api/v1/ml/predict"
    return payload


def _train_body(
    dataset_id: str,
    dataset_X,
    dataset_y,
    dataset_target: str,
    method: _Method,
    masked_y,
    train_kwargs: dict,
    base_estimator,
    model_store,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL semi-supervised fit + registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the semi-supervised fit.")
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 10, "validating unlabeled pool")
    result = train_semi_supervised(
        dataset_X, masked_y, method, base_estimator=base_estimator,
        **train_kwargs,
    )
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 80, "registering the model")
    model_id = None
    if hasattr(result.model, "predict"):
        record = TrainedModelRecord(
            model_id=f"mdl-{uuid.uuid4().hex[:12]}",
            model=result.model,
            name=f"semi:{method}",
            task="classification",
            dataset_id=dataset_id,
            target=dataset_target,
            features=[str(c) for c in dataset_X.columns],
            n_train_rows=len(dataset_X),
            best_params={"method": method, **json_safe(result.params)},
            cv_scores=None,
            cv_mean=None,
            cv_std=None,
            fit_seconds=float(result.fit_seconds),
        )
        model_store.register(record)
        model_id = record.model_id
    return {"semi_supervised": _semi_view(
        result, dataset_id, dataset_target, model_id
    )}


@router.post("/train")
def train(req: SemiSupervisedTrainRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    if dataset.task != "classification":
        raise HTTPException(
            status_code=422,
            detail=(
                "semi-supervised training is a classification surface; "
                f"dataset {req.dataset_id!r} is a {dataset.task} dataset"
            ),
        )
    base_estimator = None
    if req.base_estimator is not None:
        # a registry name is resolved through the REAL factory; the
        # backend itself then demands predict_proba (honest 422 if not)
        name = resolve_model_name(req.base_estimator)
        base_estimator = build_model(name, "classification")
    masked_y, mask_info = _masked_labels(
        dataset.y, req.mask_fraction, req.unlabeled_marker,
        req.random_state,
    )
    if req.mask_fraction is None:
        _, validation = validate_unlabeled(masked_y, req.unlabeled_marker)
        if validation.get("n_unlabeled", 0) == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "the target column carries no unlabeled marker rows; "
                    "either upload marker-encoded labels or set "
                    "mask_fraction to carve out a labeled pool"
                ),
            )
    train_kwargs = {
        "threshold": req.threshold,
        "k_best": req.k_best,
        "max_iter": req.max_iter,
        "kernel": req.kernel,
        "n_neighbors": req.n_neighbors,
        "gamma": req.gamma,
        "random_state": req.random_state,
    }
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_train_body(
                req.dataset_id, dataset.X, dataset.y, dataset.target,
                req.method, masked_y, train_kwargs, base_estimator,
                request.app.state.model_store,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "semi_supervised.train",
        "machine-learning",
        _train_body,
        configuration={
            "dataset_id": req.dataset_id,
            "method": req.method,
            "mask_fraction": req.mask_fraction,
            "base_estimator": req.base_estimator,
            **mask_info,
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        dataset_y=dataset.y,
        dataset_target=dataset.target,
        method=req.method,
        masked_y=masked_y,
        train_kwargs=train_kwargs,
        base_estimator=base_estimator,
        model_store=request.app.state.model_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


def _evaluate_body(
    dataset_id: str,
    dataset_X,
    dataset_y,
    method: _Method,
    eval_kwargs: dict,
    n_repeats_requested: int,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL held-out evaluation."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the evaluation stage.")
    evaluation = evaluate_semi_supervised(
        dataset_X, dataset_y, method, **eval_kwargs
    )
    payload = {
        k: json_safe(getattr(evaluation, k))
        for k in (
            "heldout_accuracy", "heldout_f1_macro", "baseline_accuracy",
            "baseline_f1_macro", "train_accuracy_on_labeled", "method",
        )
    }
    payload["dataset_id"] = dataset_id
    payload["n_repeats"] = n_repeats_requested
    if payload.get("heldout_accuracy") is not None:
        payload["delta_vs_baseline"] = json_safe(
            (evaluation.heldout_accuracy or 0.0)
            - (evaluation.baseline_accuracy or 0.0)
        )
    return {"evaluation": payload}


@router.post("/evaluate")
def evaluate(req: SemiSupervisedEvaluateRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    if dataset.task != "classification":
        raise HTTPException(
            status_code=422,
            detail=(
                "semi-supervised evaluation is a classification surface; "
                f"dataset {req.dataset_id!r} is a {dataset.task} dataset"
            ),
        )
    eval_kwargs = {
        "test_size": req.test_size,
        "n_repeats": req.n_repeats,
        "random_state": req.random_state,
    }
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_evaluate_body(
                req.dataset_id, dataset.X, dataset.y, req.method,
                eval_kwargs, req.n_repeats,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "semi_supervised.evaluate",
        "machine-learning",
        _evaluate_body,
        configuration={
            "dataset_id": req.dataset_id,
            "method": req.method,
            **eval_kwargs,
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        dataset_y=dataset.y,
        method=req.method,
        eval_kwargs=eval_kwargs,
        n_repeats_requested=req.n_repeats,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}

"""``/api/v1/dl`` — the Deep Learning workspace's API surface over the
REAL torch backend (part10's queued row).

* ``GET /api/v1/dl/networks`` — the tabular torch families from
  :mod:`pharmaml.ml.neural` (DeepMLP / ResidualMLP, classifier and
  regressor), each with its live default hyperparameters read off
  the class signature (one source of truth, never re-declared).
* ``POST /api/v1/dl/train`` — one torch network trained by its REAL
  ``fit(X, y, on_epoch=...)`` on a registered dataset (sync or
  managed job).  The estimator's own per-epoch hook drives the
  **training-progress events**: every epoch publishes
  ``training.progress`` (progress %, stage, epoch counter) and
  ``training.metrics`` (train/val losses, best val, staleness) on
  the session bus — in job mode the job record's progress tracks the
  epoch counter too.  This is the §9.3 training-progress vocabulary
  running on REAL telemetry, not a stub.

Predictions for a DL-trained model flow through
``POST /api/v1/ml/predict`` — the model registers in the same
session model store, so there is exactly one prediction path for
every trainer, never two.
"""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pharmaml.ml.neural import (
    DeepMLPClassifier,
    DeepMLPRegressor,
    ResidualMLPClassifier,
    ResidualMLPRegressor,
)

from ..events import EventType
from ..jobs import CancelledJob
from ..stores import TrainedModelRecord, json_safe

router = APIRouter(prefix="/dl", tags=["dl"])

_NETWORKS: dict[str, tuple[type, str]] = {
    "deep_mlp_classifier": (DeepMLPClassifier, "classification"),
    "deep_mlp_regressor": (DeepMLPRegressor, "regression"),
    "residual_mlp_classifier": (ResidualMLPClassifier, "classification"),
    "residual_mlp_regressor": (ResidualMLPRegressor, "regression"),
}


def _network_catalog() -> list[dict[str, Any]]:
    """The live catalog: defaults read off the class signatures."""
    out: list[dict[str, Any]] = []
    for name, (cls, task) in _NETWORKS.items():
        defaults: dict[str, Any] = {}
        try:
            sig = inspect.signature(cls)
            for pname, param in sig.parameters.items():
                if pname == "self" or param.default is inspect.Parameter.empty:
                    continue
                defaults[pname] = json_safe(param.default)
        except (TypeError, ValueError):  # pragma: no cover - signature drift
            pass
        out.append(
            {
                "network": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "task": task,
                "engine": "torch",
                "defaults": defaults,
                "train_url": "/api/v1/dl/train",
            }
        )
    return out


class DLTrainRequest(BaseModel):
    dataset_id: str
    network: Literal[
        "deep_mlp_classifier",
        "deep_mlp_regressor",
        "residual_mlp_classifier",
        "residual_mlp_regressor",
    ]
    hyperparams: dict[str, Any] | None = None
    mode: Literal["job", "sync"] = "job"


_HYPERPARAM_SPEC = {
    "hidden_layers": list[int],
    "dropout": float,
    "activation": str,
    "batchnorm": bool,
    "epochs": int,
    "batch_size": int,
    "learning_rate": float,
    "weight_decay": float,
    "patience": int,
    "val_split": float,
    "random_state": int,
    "device": str,
}


def _resolve_hyperparams(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the client hyperparameters against the network signature."""
    params: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if key not in _HYPERPARAM_SPEC:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown hyperparameter {key!r}; the network accepts "
                    f"{sorted(_HYPERPARAM_SPEC)}"
                ),
            )
        params[key] = value
    return params


@router.get("/networks")
def networks(request: Request) -> dict:
    return {
        "networks": _network_catalog(),
        "predict_url": "/api/v1/ml/predict",
        "note": (
            "predictions for a DL-trained model run through the single "
            "session prediction path /api/v1/ml/predict"
        ),
    }


def _training_progress_hook(
    bus,
    manager=None,
    job_id: str | None = None,
) -> Any:
    """The per-epoch adapter: REAL ``on_epoch`` dicts become typed events."""

    def on_epoch(info: dict[str, Any]) -> None:
        epoch = int(info.get("epoch", 0))
        epochs = max(1, int(info.get("epochs", 1)))
        metrics = {
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": json_safe(info.get("train_loss")),
            "val_loss": json_safe(info.get("val_loss")),
            "best_val": json_safe(info.get("best_val")),
            "stale": int(info.get("stale", 0)),
        }
        progress = min(100, int(100.0 * epoch / epochs))
        stage = f"epoch {epoch}/{epochs}"
        if manager is not None and job_id is not None:
            manager.report_progress(
                job_id, progress, stage, event=EventType.TRAINING_PROGRESS
            )
        else:
            bus.publish(
                EventType.TRAINING_PROGRESS,
                {"progress": progress, "stage": stage},
                job_id=job_id,
            )
        bus.publish(EventType.TRAINING_METRICS, metrics, job_id=job_id)

    return on_epoch


def _train_body(
    dataset_id: str,
    dataset_X,
    dataset_y,
    target: str,
    network_name: str,
    hyperparams: dict[str, Any] | None,
    model_store,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL torch fit + store registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the network build stage.")
    cls, task = _NETWORKS[network_name]
    params = _resolve_hyperparams(hyperparams)
    if manager is not None and job_id is not None:
        manager.report_progress(
            job_id, 5, "building the torch network"
        )
    estimator = cls(**params)
    on_epoch = _training_progress_hook(bus, manager, job_id)
    started = time.monotonic()
    estimator.fit(dataset_X, dataset_y, on_epoch=on_epoch)
    fit_seconds = time.monotonic() - started
    record = TrainedModelRecord(
        model_id=f"mdl-{uuid.uuid4().hex[:12]}",
        model=estimator,
        name=network_name,
        task=task,
        dataset_id=dataset_id,
        target=target,
        features=[str(c) for c in dataset_X.columns],
        n_train_rows=len(dataset_X),
        best_params={"network": network_name, **params},
        cv_scores=None,  # no CV in this path — honest, not faked
        cv_mean=None,
        cv_std=None,
        fit_seconds=fit_seconds,
    )
    model_store.register(record)
    view = record.view()
    view["engine"] = "torch"
    view["predict_url"] = "/api/v1/ml/predict"
    return {"model": view}


@router.post("/train")
def train(req: DLTrainRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    _, natural_task = _NETWORKS[req.network]
    if dataset.task != natural_task:
        raise HTTPException(
            status_code=422,
            detail=(
                f"network {req.network!r} is a {natural_task} model but "
                f"dataset {req.dataset_id!r} is a {dataset.task} dataset"
            ),
        )
    # validate hyperparameters up front (both modes): a bad key is a
    # client error, not an asynchronous job failure
    _resolve_hyperparams(req.hyperparams)
    manager = request.app.state.job_manager
    model_store = request.app.state.model_store
    bus = request.app.state.bus
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_train_body(
                req.dataset_id, dataset.X, dataset.y, dataset.target,
                req.network, req.hyperparams, model_store,
                bus=bus,
            ),
        }
    record = manager.submit(
        "dl.train",
        "deep-learning",
        _train_body,
        configuration={
            "dataset_id": req.dataset_id,
            "network": req.network,
            "hyperparams": req.hyperparams or {},
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        dataset_y=dataset.y,
        target=dataset.target,
        network_name=req.network,
        hyperparams=req.hyperparams,
        model_store=model_store,
        bus=bus,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}

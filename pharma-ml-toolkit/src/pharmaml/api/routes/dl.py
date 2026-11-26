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

The GNN/MPNN lane (part13 Phase 5's first item — the API leg):

* ``GET /api/v1/dl/networks`` — the tabular families + the GNN
  vocabulary (``CONV_NAMES`` / ``POOL_NAMES`` read off the backend —
  one source of truth).
* ``POST /api/v1/dl/gnn-train`` — :class:`pharmaml.ml.gnn.GraphClassifier`
  (conv stack → pool → linear head) trained on inline graph specs
  (node features + the 2×E endpoint list per graph) or on the
  backend's own ``synthetic_motif_graphs`` fixture (sync or managed
  job).  The torch GNN trainer exposes ``loss_history_`` after the
  fit, not a live per-epoch callback — the job's progress is the
  honest stage-level telemetry (assembling → training), never a
  fabricated epoch stream.  The fitted estimator registers in the
  session model store as a GRAPH model with its own predict path.
* ``POST /api/v1/dl/gnn-predict`` — the graph models' prediction
  path: label sets (and optional class probabilities) for inline
  graph rows from a session-trained GNN.  ``POST /api/v1/ml/predict``
  refuses graph models up front with the pointer here — one path
  per model class, never a tabular predict over graph weights.
"""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.autoencoder import TabularAutoencoder
from pharmaml.ml.cnn import (
    SequenceCNNClassifier,
    SequenceCNNRegressor,
    synthetic_motif_sequences,
)
from pharmaml.ml.gnn import (
    CONV_NAMES,
    POOL_NAMES,
    Data,
    GraphClassifier,
    synthetic_motif_graphs,
)
from pharmaml.ml.neural import (
    DeepMLPClassifier,
    DeepMLPRegressor,
    ResidualMLPClassifier,
    ResidualMLPRegressor,
)
from pharmaml.ml.transformer import (
    TabularTransformerClassifier,
    TabularTransformerRegressor,
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
    # part18 §5.4 — the tabular Transformer lane (from scratch; no
    # pretrained molecular transformer is claimed anywhere)
    "tabular_transformer_classifier": (
        TabularTransformerClassifier, "classification",
    ),
    "tabular_transformer_regressor": (
        TabularTransformerRegressor, "regression",
    ),
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
        "tabular_transformer_classifier",
        "tabular_transformer_regressor",
    ]
    hyperparams: dict[str, Any] | None = None
    mode: Literal["job", "sync"] = "job"


# the tabular families' constructor-key vocabulary, one source: the
# union of the live class signatures (validated per-network below)
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
    # the tabular Transformer family (part18 §5.4)
    "d_model": int,
    "heads": int,
    "n_blocks": int,
    "dim_feedforward": int,
}


def _resolve_hyperparams(
    raw: dict[str, Any] | None, network_name: str | None = None
) -> dict[str, Any]:
    """Validate the client hyperparameters against the network signature.

    With ``network_name`` the accepted keys are the live constructor
    parameters of THAT network (unknown key -> 422, never an
    asynchronous TypeError); without it the union vocabulary above
    gates the up-front check only.
    """
    accepted = set(_HYPERPARAM_SPEC)
    if network_name is not None:
        cls, _task = _NETWORKS[network_name]
        accepted = {
            name for name, param in inspect.signature(cls).parameters.items()
            if name != "self"
        }
    params: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if key not in accepted:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown hyperparameter {key!r}; network {network_name!r} "
                    f"accepts {sorted(accepted)}"
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
        "gnn": {
            "convs": list(CONV_NAMES),
            "pools": list(POOL_NAMES),
            "defaults": {"hidden": 16, "layers": 2, "lr": 0.05, "epochs": 200},
            "train_url": "/api/v1/dl/gnn-train",
            "predict_url": "/api/v1/dl/gnn-predict",
            "note": (
                "graph models predict through their own path — never the "
                "tabular one"
            ),
        },
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
    params = _resolve_hyperparams(hyperparams, network_name)
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
    _resolve_hyperparams(req.hyperparams, req.network)
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


# -- the GNN/MPNN lane (part13 Phase 5, the API leg) -----------------------

class GnnGraphSpec(BaseModel):
    """One graph: node features + the 2×E source→destination endpoints."""

    x: list[list[float]] = Field(min_length=1)
    edge_index: list[list[int]] = Field(min_length=0)

    def to_data(self) -> Data:
        try:
            return Data(
                np.asarray(self.x, dtype=float),
                np.asarray(self.edge_index, dtype=int),
            )
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


class GnnTrainRequest(BaseModel):
    # unique OpenAPI component names (the family's anti-collision rule)
    graphs: list[GnnGraphSpec] | None = Field(
        default=None, min_length=2, max_length=500
    )
    labels: list[int] | None = Field(
        default=None, min_length=2, max_length=500
    )
    synthetic_n: int | None = Field(default=None, ge=10, le=500)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    conv: Literal["gcn", "sage", "gin", "gat", "edge"] = "gcn"
    pool: Literal["mean", "sum", "max", "topk", "sag"] = "mean"
    hidden: int = Field(default=16, ge=1, le=128)
    layers: int = Field(default=2, ge=1, le=6)
    lr: float = Field(default=0.05, gt=0.0, le=1.0)
    epochs: int = Field(default=200, ge=1, le=2000)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    heads: int = Field(default=1, ge=1, le=8)
    ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    mode: Literal["job", "sync"] = "job"


class GnnPredictRequest(BaseModel):
    model_id: str
    graphs: list[GnnGraphSpec] = Field(min_length=1, max_length=500)
    probabilities: bool = False


def _gnn_train_body(
    graphs: list[Data],
    labels: list[int],
    hyperparams: dict[str, Any],
    model_store,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL GraphClassifier fit + store registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the graph assembly stage.")
    if manager is not None and job_id is not None:
        # stage-level telemetry only — the torch GNN trainer has no
        # live per-epoch callback (loss_history_ lands AFTER the fit);
        # this lane never fabricates an epoch stream
        manager.report_progress(job_id, 5, "assembling the graph batch")
    estimator = GraphClassifier(**hyperparams)
    if manager is not None and job_id is not None:
        manager.report_progress(
            job_id, 50, "training the conv stack (history lands in loss_history_)"
        )
    started = time.monotonic()
    estimator.fit(graphs, labels)
    fit_seconds = time.monotonic() - started
    train_accuracy = float(estimator.score(graphs, labels))
    record = TrainedModelRecord(
        model_id=f"mdl-{uuid.uuid4().hex[:12]}",
        model=estimator,
        name="gnn_graph_classifier",
        task="classification",
        dataset_id=None,  # no tabular dataset behind a graph model — honest
        target=None,
        features=[],  # node features are per-graph matrices, not columns
        n_train_rows=len(graphs),
        best_params={"network": "gnn", **hyperparams},
        cv_scores=None,  # no CV in this path — honest, not faked
        cv_mean=None,
        cv_std=None,
        fit_seconds=fit_seconds,
    )
    model_store.register(record)
    view = record.view()
    view["engine"] = "torch"
    view["graph_model"] = True
    view["predict_url"] = "/api/v1/dl/gnn-predict"
    return {
        "model": view,
        "train_accuracy": train_accuracy,
        "loss_history_tail": estimator.loss_history_[-5:],
        "n_graphs": len(graphs),
        "classes": [json_safe(c) for c in estimator.classes_],
        "conv": hyperparams.get("conv"),
        "pool": hyperparams.get("pool"),
    }


@router.post("/gnn-train")
def gnn_train(req: GnnTrainRequest, request: Request) -> dict:
    specs = req.graphs
    synthetic_n = req.synthetic_n
    if specs is None:
        if synthetic_n is None:
            raise HTTPException(
                status_code=422,
                detail="the trainer needs inline graphs+labels or synthetic_n",
            )
        fixture, y = synthetic_motif_graphs(synthetic_n, seed=req.seed)
        graphs = list(fixture)
        labels = [int(v) for v in y]
    else:
        if synthetic_n is not None:
            raise HTTPException(
                status_code=422,
                detail="pass either inline graphs or synthetic_n, not both",
            )
        if req.labels is None:
            raise HTTPException(
                status_code=422,
                detail="inline graphs need their labels",
            )
        if len(specs) != len(req.labels):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"graphs has {len(specs)} rows; labels has "
                    f"{len(req.labels)} (one label per graph)"
                ),
            )
        graphs = [spec.to_data() for spec in specs]
        labels = list(req.labels)
    widths = {int(g.x.shape[1]) for g in graphs}
    if len(widths) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "all graphs must share one node-feature width; got "
                f"{sorted(widths)}"
            ),
        )
    hyperparams = {
        "hidden": req.hidden,
        "layers": req.layers,
        "conv": req.conv,
        "pool": req.pool,
        "lr": req.lr,
        "epochs": req.epochs,
        "weight_decay": req.weight_decay,
        "heads": req.heads,
        "ratio": req.ratio,
        "seed": req.seed,
    }
    manager = request.app.state.job_manager
    model_store = request.app.state.model_store
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_gnn_train_body(graphs, labels, hyperparams, model_store),
        }
    record = manager.submit(
        "dl.gnn-train",
        "deep-learning",
        _gnn_train_body,
        configuration={
            "n_graphs": len(graphs),
            "conv": req.conv,
            "pool": req.pool,
            "epochs": req.epochs,
            "synthetic": req.synthetic_n is not None,
        },
        graphs=graphs,
        labels=labels,
        hyperparams=hyperparams,
        model_store=model_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/gnn-predict")
def gnn_predict(req: GnnPredictRequest, request: Request) -> dict:
    model_record = request.app.state.model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    if not isinstance(model_record.model, GraphClassifier):
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is not a graph model; graph "
                "prediction needs a GNN trained through /api/v1/dl/gnn-train"
            ),
        )
    graphs = [spec.to_data() for spec in req.graphs]
    width = int(getattr(model_record.model, "n_features_in_", -1))
    bad = [
        i for i, g in enumerate(graphs)
        if width >= 0 and int(g.x.shape[1]) != width
    ]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the GNN was trained on {width}-wide node features; graph "
                f"{bad[0]} has {int(graphs[bad[0]].x.shape[1])}"
            ),
        )
    try:
        predictions = model_record.model.predict(graphs)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload: dict[str, Any] = {
        "model_id": req.model_id,
        "n_graphs": len(graphs),
        "predictions": [json_safe(p) for p in predictions],
        "classes": [
            json_safe(c) for c in model_record.model.classes_
        ],
    }
    if req.probabilities:
        probabilities = model_record.model.predict_proba(graphs)
        payload["probabilities"] = json_safe(
            [list(row) for row in probabilities]
        )
    return payload


# -- the sequence-CNN lane (part18 §5.2, the API leg) -----------------------

class CnnSequenceSpec(BaseModel):
    """One sequence: a list of feature-vector time steps."""

    steps: list[list[float]] = Field(min_length=1, max_length=512)


class CnnTrainRequest(BaseModel):
    # unique OpenAPI component names (the family's anti-collision rule)
    sequences: list[CnnSequenceSpec] | None = Field(
        default=None, min_length=2, max_length=500
    )
    labels: list[float] | None = Field(default=None, min_length=2, max_length=500)
    synthetic_n: int | None = Field(default=None, ge=10, le=500)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    task: Literal["classification", "regression"] = "classification"
    channels: int = Field(default=16, ge=1, le=128)
    kernel: int = Field(default=3, ge=1, le=31)
    n_layers: int = Field(default=2, ge=1, le=6)
    epochs: int = Field(default=120, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    patience: int = Field(default=15, ge=1, le=100)
    val_split: float = Field(default=0.15, ge=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0, le=2**31 - 1)
    mode: Literal["job", "sync"] = "job"


class CnnPredictRequest(BaseModel):
    model_id: str
    sequences: list[CnnSequenceSpec] = Field(min_length=1, max_length=500)
    probabilities: bool = False


def _cnn_train_body(
    x: np.ndarray,
    y: np.ndarray,
    hyperparams: dict[str, Any],
    model_store,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL SequenceCNN fit + store registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the sequence assembly stage.")
    task = hyperparams.pop("_task", "classification")
    cls = SequenceCNNClassifier if task == "classification" else SequenceCNNRegressor
    estimator = cls(**hyperparams)
    on_epoch = _training_progress_hook(bus, manager, job_id)
    started = time.monotonic()
    estimator.fit(x, y, on_epoch=on_epoch)
    fit_seconds = time.monotonic() - started
    record = TrainedModelRecord(
        model_id=f"mdl-{uuid.uuid4().hex[:12]}",
        model=estimator,
        name=f"sequence_cnn_{task}",
        task=task,
        dataset_id=None,  # no tabular dataset behind a sequence model — honest
        target=None,
        features=[],  # time-step channels are per-sequence matrices, not columns
        n_train_rows=len(x),
        best_params={"network": "sequence_cnn", **hyperparams},
        cv_scores=None,  # no CV in this path — honest, not faked
        cv_mean=None,
        cv_std=None,
        fit_seconds=fit_seconds,
    )
    model_store.register(record)
    view = record.view()
    view["engine"] = "torch"
    view["sequence_model"] = True
    view["predict_url"] = "/api/v1/dl/cnn-predict"
    out = {
        "model": view,
        "seq_len": int(x.shape[1]),
        "n_channels": int(x.shape[2]),
        "n_iter": int(estimator.n_iter_),
    }
    if task == "classification":
        out["classes"] = [json_safe(c) for c in estimator.classes_]
        out["train_accuracy"] = float(
            np.mean(estimator.predict(x) == y)
        )
    else:
        out["train_mae"] = float(np.mean(np.abs(estimator.predict(x) - y)))
    return out


@router.post("/cnn-train")
def cnn_train(req: CnnTrainRequest, request: Request) -> dict:
    if req.sequences is None and req.synthetic_n is None:
        raise HTTPException(
            status_code=422,
            detail="the trainer needs inline sequences+labels or synthetic_n",
        )
    if req.sequences is not None and req.synthetic_n is not None:
        raise HTTPException(
            status_code=422,
            detail="pass either inline sequences or synthetic_n, not both",
        )
    if req.synthetic_n is not None:
        # the deterministic motif fixture (classification only)
        if req.task != "classification":
            raise HTTPException(
                status_code=422,
                detail="synthetic_n trains the binary motif fixture (classification)",
            )
        seed = int(req.seed) if req.seed is not None else 42
        x, y = synthetic_motif_sequences(
            int(req.synthetic_n), seed=seed
        )
    else:
        assert req.sequences is not None and req.labels is not None  # narrow types
        if len(req.sequences) != len(req.labels):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"sequences has {len(req.sequences)} rows; labels has "
                    f"{len(req.labels)} (one label per sequence)"
                ),
            )
        try:
            x = np.asarray(
                [s.steps for s in req.sequences], dtype=np.float64
            )
            y = np.asarray(req.labels, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"sequences must be rectangular numeric steps: {exc}",
            ) from exc
    hyperparams = {
        "channels": req.channels,
        "kernel": req.kernel,
        "n_layers": req.n_layers,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "weight_decay": req.weight_decay,
        "patience": req.patience,
        "val_split": req.val_split,
        "random_state": req.seed if req.seed is not None else req.random_state,
    }
    manager = request.app.state.job_manager
    model_store = request.app.state.model_store
    bus = request.app.state.bus
    if req.mode == "sync":
        try:
            params = dict(hyperparams)
            params["_task"] = req.task
            return {
                "mode": "sync",
                **_cnn_train_body(x, y, params, model_store, bus=bus),
            }
        except (DataValidationError, ModelError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    params = dict(hyperparams)
    params["_task"] = req.task
    record = manager.submit(
        "dl.cnn-train",
        "deep-learning",
        _cnn_train_body,
        configuration={
            "n_sequences": int(x.shape[0]),
            "task": req.task,
            "epochs": req.epochs,
            "synthetic": req.synthetic_n is not None,
        },
        x=x,
        y=y,
        hyperparams=params,
        model_store=model_store,
        bus=bus,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/cnn-predict")
def cnn_predict(req: CnnPredictRequest, request: Request) -> dict:
    model_record = request.app.state.model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    model = model_record.model
    if not isinstance(model, (SequenceCNNClassifier, SequenceCNNRegressor)):
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is not a sequence model; sequence "
                "prediction needs a CNN trained through /api/v1/dl/cnn-train"
            ),
        )
    try:
        x = np.asarray([s.steps for s in req.sequences], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"sequences must be rectangular numeric steps: {exc}",
        ) from exc
    try:
        predictions = model.predict(x)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload: dict[str, Any] = {
        "model_id": req.model_id,
        "n_sequences": len(req.sequences),
        "predictions": [json_safe(p) for p in predictions],
    }
    if req.probabilities and isinstance(model, SequenceCNNClassifier):
        probabilities = model.predict_proba(x)
        payload["probabilities"] = json_safe([list(row) for row in probabilities])
        payload["classes"] = [json_safe(c) for c in model.classes_]
    return payload


# -- the tabular-autoencoder lane (part18 §5.5, the API leg) ------------------

class AutoencoderTrainRequest(BaseModel):
    dataset_id: str
    hidden_layers: list[int] = Field(default=[64], min_length=0, max_length=8)
    latent_dim: int = Field(default=8, ge=1, le=64)
    epochs: int = Field(default=120, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    patience: int = Field(default=15, ge=1, le=100)
    val_split: float = Field(default=0.15, ge=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0, le=2**31 - 1)
    mode: Literal["job", "sync"] = "job"


class AutoencoderEncodeRequest(BaseModel):
    model_id: str
    dataset_id: str | None = None
    rows: list[dict[str, Any]] | None = Field(default=None, min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


def _autoencoder_train_body(
    dataset_id: str,
    dataset_X,
    hyperparams: dict[str, Any],
    model_store,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL TabularAutoencoder fit + registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the autoencoder build stage.")
    estimator = TabularAutoencoder(**hyperparams)
    on_epoch = _training_progress_hook(bus, manager, job_id)
    started = time.monotonic()
    estimator.fit(dataset_X, on_epoch=on_epoch)
    fit_seconds = time.monotonic() - started
    record = TrainedModelRecord(
        model_id=f"mdl-{uuid.uuid4().hex[:12]}",
        model=estimator,
        name="tabular_autoencoder",
        task="unsupervised",  # honest: representation/reconstruction lane
        dataset_id=dataset_id,
        target=None,
        features=[str(c) for c in dataset_X.columns],
        n_train_rows=len(dataset_X),
        best_params={"network": "autoencoder", **hyperparams},
        cv_scores=None,  # unsupervised — no CV metric exists, not faked
        cv_mean=None,
        cv_std=None,
        fit_seconds=fit_seconds,
    )
    model_store.register(record)
    view = record.view()
    view["engine"] = "torch"
    view["encode_url"] = "/api/v1/dl/autoencoder-encode"
    return {
        "model": view,
        "latent_dim": int(estimator.latent_dim),
        "n_iter": int(estimator.n_iter_),
        "mean_reconstruction_error": float(
            estimator.reconstruction_errors(dataset_X.to_numpy()).mean()
        ),
    }


@router.post("/autoencoder-train")
def autoencoder_train(req: AutoencoderTrainRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    ae_params = {
        "hidden_layers": tuple(req.hidden_layers),
        "latent_dim": req.latent_dim,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "weight_decay": req.weight_decay,
        "patience": req.patience,
        "val_split": req.val_split,
        "random_state": req.random_state,
    }
    manager = request.app.state.job_manager
    model_store = request.app.state.model_store
    bus = request.app.state.bus
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_autoencoder_train_body(
                    req.dataset_id, dataset.X, ae_params, model_store, bus=bus
                ),
            }
        except (DataValidationError, ModelError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = manager.submit(
        "dl.autoencoder-train",
        "deep-learning",
        _autoencoder_train_body,
        configuration={
            "dataset_id": req.dataset_id,
            "latent_dim": req.latent_dim,
            "epochs": req.epochs,
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        hyperparams=ae_params,
        model_store=model_store,
        bus=bus,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/autoencoder-encode")
def autoencoder_encode(req: AutoencoderEncodeRequest, request: Request) -> dict:
    model_record = request.app.state.model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    model = model_record.model
    if not isinstance(model, TabularAutoencoder):
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is not a tabular autoencoder; the "
                "encode lane needs one trained through /api/v1/dl/autoencoder-train"
            ),
        )
    if (req.dataset_id is None) == (req.rows is None):
        raise HTTPException(
            status_code=422,
            detail="pass either dataset_id or inline rows, not both",
        )
    if req.dataset_id is not None:
        dataset = request.app.state.dataset_store.get(req.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Unknown dataset_id.")
        X = dataset.X
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    else:
        try:
            X_arr = np.asarray(
                [[float(v) for v in row.values()] for row in (req.rows or [])],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"rows must be numeric feature vectors: {exc}",
            ) from exc
    try:
        latent = model.encode(X_arr)
        errors = model.reconstruction_errors(X_arr)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    top = np.argsort(errors)[::-1][: req.top_k].tolist()
    return {
        "model_id": req.model_id,
        "n_rows": int(X_arr.shape[0]),
        "latent_dim": int(latent.shape[1]),
        "latent": json_safe(latent.tolist()),
        "reconstruction_errors": json_safe(errors.tolist()),
        "mean_reconstruction_error": float(np.mean(errors)),
        "top_outliers": [
            {"index": int(i), "error": float(errors[i])} for i in top
        ],
        "note": (
            "reconstruction error is a heuristic outlier ranking — not "
            "ground truth (the dedicated anomaly lane owns Isolation Forest)"
        ),
    }

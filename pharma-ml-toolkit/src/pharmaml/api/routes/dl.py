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
* ``POST /api/v1/dl/train`` with ``tune: true`` — DL-1 (the roadmap's
  row 13): the search lane over the SAME route.  A bounded per-key
  search block (``choices``, or ``low``/``high`` — int specs when
  both bounds are ints, float otherwise with optional ``log``
  sampling) over the SAME constructor vocabulary the route validates;
  the trials run through the in-house TPE/Random engine
  (:class:`pharmaml.ml.hpo.HPOStudy` — the same engine the classical
  ML lane's ``tune`` flag rides): each candidate is built and fit on
  a seeded train side and scored on the held-out val side (the
  sklearn mixin semantics — accuracy / R², negated inside the study,
  the optuna rule).  Cancellation lands BETWEEN trials (the
  population-route precedent); the winner is refit on the full data
  with the full per-epoch telemetry and registers like any trained
  model — one prediction path for every trainer, never two.  The
  report mirrors the ML lane's TuneReport surface (the trial table +
  the best configuration + the best score) plus the honest extras:
  the refit's comparable score on the SAME carve, and the searched
  block echoed.

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
* ``POST /api/v1/dl/gnn-explain`` — DL-2 (the roadmap's row 14): the
  graph models' explainability lane over the REAL
  ``pharmaml.ml.gnnxai`` engine: the per-edge importance table for
  inline graph rows through the chosen explainer (the learned mask
  of ``GNNExplainer``, the Shapley-guided ``SubgraphX``, the GAT
  attention coefficients, or the parametric ``PGExplainer`` — the
  one that trains its scorer over the request's own labeled graphs),
  each row carrying the fidelity_plus / fidelity_minus / sparsity
  verdicts and the deterministic circular node layout the backend
  computes for the sized-edge scatter.  A non-graph model is refused
  up front; the attention lane refuses non-GAT models with the
  engine's own message.  Sync by design (masked forward passes,
  seconds per explained graph).
"""

from __future__ import annotations

import inspect
import math
import time
import uuid
from collections.abc import Callable
from functools import partial
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
    GnnError,
    GraphClassifier,
    synthetic_motif_graphs,
)
from pharmaml.ml.gnnxai import (
    AttentionExplainer,
    EdgeExplanation,
    GNNExplainer,
    PGExplainer,
    SubgraphX,
    fidelity_minus,
    fidelity_plus,
    sparsity,
)
from pharmaml.ml.hpo import (
    CategoricalParam,
    FloatParam,
    HPOStudy,
    IntParam,
    SearchSpace,
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
    # DL-1 (the roadmap's row 13) — the search lane: `tune: true`
    # rides the same route with a bounded per-key search space over
    # the SAME constructor vocabulary (choices, or low/high — int
    # specs when both bounds are ints, float otherwise with optional
    # log sampling), through the in-house TPE/Hyperband engine
    tune: bool = False
    search: dict[str, dict[str, Any]] | None = None
    n_trials: int = Field(default=12, ge=1, le=50)
    sampler: Literal["tpe", "random"] = "tpe"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
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


def _accepted_keys(network_name: str) -> set[str]:
    """The live constructor vocabulary of one network."""
    cls, _task = _NETWORKS[network_name]
    return {
        name for name in inspect.signature(cls).parameters
        if name != "self"
    }


def _build_search_space(
    search: dict[str, dict[str, Any]] | None, network_name: str
) -> SearchSpace:
    """DL-1 — validate the per-key search block and build the space.

    Every key must live in the network's own constructor vocabulary
    (the same rule `_resolve_hyperparams` enforces); each entry is a
    ``choices`` list (categorical — the only lane for list-valued
    keys like ``hidden_layers``) or a ``low``/``high`` pair (ints
    when both bounds are ints, floats otherwise, ``log`` optional on
    the float lane).  Everything else is a 422 — a doomed search is
    a client error, never an asynchronous job failure.
    """
    if not search:
        raise HTTPException(
            status_code=422,
            detail=(
                "the tune lane needs a search block: {key: {choices: [...]} "
                "or {key: {low: ..., high: ..., log: bool}}} over the "
                f"network {network_name!r} constructor vocabulary"
            ),
        )
    accepted = _accepted_keys(network_name)
    space = SearchSpace()
    for key, spec in search.items():
        if key not in accepted:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"unknown search key {key!r}; network {network_name!r} "
                    f"accepts {sorted(accepted)}"
                ),
            )
        if not isinstance(spec, dict) or not spec:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the search entry for {key!r} must be an object with "
                    "either choices: [...] or low/high bounds"
                ),
            )
        if "choices" in spec:
            choices = spec["choices"]
            if (
                not isinstance(choices, list)
                or len(choices) == 0
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"the search entry for {key!r} carries an empty "
                        "or non-list choices block"
                    ),
                )
            space.add(CategoricalParam(name=key, choices=tuple(choices)))
        elif "low" in spec and "high" in spec:
            low, high = spec["low"], spec["high"]
            for bound in (low, high):
                if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"the search bounds for {key!r} must be numbers "
                            f"(got {bound!r})"
                        ),
                    )
            if not float(low) < float(high):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"the search bounds for {key!r} must satisfy "
                        f"low < high (got {low!r} .. {high!r})"
                    ),
                )
            log = spec.get("log", False)
            if not isinstance(log, bool):
                raise HTTPException(
                    status_code=422,
                    detail=(f"the search log flag for {key!r} must be a boolean"),
                )
            if isinstance(low, int) and isinstance(high, int):
                space.add(IntParam(name=key, low=low, high=high))
            else:
                if float(low) <= 0.0 and log:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"a log search for {key!r} needs a positive "
                            f"low bound (got {low!r})"
                        ),
                    )
                space.add(
                    FloatParam(
                        name=key, low=float(low), high=float(high), log=log
                    )
                )
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the search entry for {key!r} carries neither choices "
                    "nor low/high bounds"
                ),
            )
    return space


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
    fit_seconds = time.perf_counter() - started
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
    # DL-1: validate the search block up front too (a doomed search is
    # a client error, never an asynchronous job failure), and gate the
    # lane's own data minimum (the objective's carve needs both sides)
    space = None
    if req.tune:
        space = _build_search_space(req.search, req.network)
        if len(dataset.X) < 12:
            raise HTTPException(
                status_code=422,
                detail=(
                    "the tune lane carves a validation side for the "
                    "objective; the dataset carries "
                    f"{len(dataset.X)} rows and the route's own minimum is 12"
                ),
            )
    manager = request.app.state.job_manager
    model_store = request.app.state.model_store
    bus = request.app.state.bus
    if req.tune:
        if req.mode == "sync":
            assert space is not None  # built in the guard above
            return {
                "mode": "sync",
                **_dl_tune_body(
                    req.dataset_id, dataset.X, dataset.y, dataset.target,
                    req.network, req.hyperparams, space,
                    req.n_trials, req.sampler, req.seed,
                    model_store, search_echo=req.search, bus=bus,
                ),
            }
        record = manager.submit(
            "dl.train",
            "deep-learning",
            _dl_tune_body,
            configuration={
                "dataset_id": req.dataset_id,
                "network": req.network,
                "hyperparams": req.hyperparams or {},
                "tune": True,
                "search": req.search or {},
                "n_trials": req.n_trials,
                "sampler": req.sampler,
                "seed": req.seed,
            },
            dataset_id=req.dataset_id,
            dataset_X=dataset.X,
            dataset_y=dataset.y,
            target=dataset.target,
            network_name=req.network,
            hyperparams=req.hyperparams,
            space=space,
            n_trials=req.n_trials,
            sampler=req.sampler,
            seed=req.seed,
            model_store=model_store,
            search_echo=req.search,
            bus=bus,
            manager=manager,
        )
        return {"mode": "job", "job": record.public()}
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


# -- DL-1: the search lane body (the roadmap's row 13) ---------------------

_MIN_TUNE_ROWS = 12


def _dl_tune_body(
    dataset_id: str,
    dataset_X,
    dataset_y,
    target: str,
    network_name: str,
    hyperparams: dict[str, Any] | None,
    space: SearchSpace,
    n_trials: int,
    sampler: str,
    seed: int,
    model_store,
    search_echo: dict[str, dict[str, Any]] | None = None,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """DL-1 — the search body: trials through the REAL TPE engine.

    The objective is the honest one the roadmap names: each candidate
    configuration is BUILT and FIT on a seeded train side of the
    dataset, then scored on the held-out val side (accuracy for the
    classifiers, R² for the regressors — the sklearn mixin score,
    the same semantics the ML lane's TuneReport carries), negated so
    smaller-is-better inside the study (the optuna rule the engine
    documents).  Cancellation lands between trials (the
    population-route precedent); the winner is refit on the FULL data
    through the same `_train_body` fit path and registers like any
    trained model — one prediction path for every trainer, never two.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the search stage.")
    cls, task = _NETWORKS[network_name]
    base = _resolve_hyperparams(hyperparams, network_name)

    # the seeded objective carve (train on one side, score on the other)
    rng = np.random.default_rng(seed)
    n = len(dataset_X)
    if n < _MIN_TUNE_ROWS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the tune lane needs >= {_MIN_TUNE_ROWS} rows for its "
                f"objective carve; the dataset carries {n}"
            ),
        )
    perm = rng.permutation(n)
    val_n = max(2, round(0.2 * n))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    X_train = dataset_X.iloc[train_idx]
    y_train = dataset_y.iloc[train_idx]
    X_val = dataset_X.iloc[val_idx]
    y_val = dataset_y.iloc[val_idx]

    def objective(params) -> float:
        estimator = cls(**{**base, **dict(params)})
        estimator.fit(X_train, y_train)
        return -float(estimator.score(X_val, y_val))

    study = HPOStudy(
        objective=objective, space=space, sampler=sampler, seed=seed
    )
    completed = 0
    for _ in range(n_trials):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJob(
                f"Cancelled between trials (after {completed} of {n_trials})"
            )
        study.run(1)
        completed += 1
        if manager is not None and job_id is not None:
            manager.report_progress(
                job_id,
                min(90, int(100.0 * completed / n_trials)),
                f"trial {completed}/{n_trials}",
                event=EventType.TRAINING_PROGRESS,
            )
    if not study.trials:
        raise HTTPException(  # pragma: no cover - n_trials >= 1 always
            status_code=500,
            detail="the search produced no trials",
        )
    best = study.best()
    if best.value is None:  # pragma: no cover - objective always returns
        raise HTTPException(
            status_code=500,
            detail="the search produced no completed trial",
        )
    best_params = {**base, **dict(best.params)}
    if manager is not None and job_id is not None:
        manager.report_progress(
            job_id, 95, "refitting the best configuration on the full data"
        )

    # the winner refit on the FULL data — the same fit + telemetry path
    # the fixed-config lane rides (per-epoch events included)
    estimator = cls(**best_params)
    on_epoch = _training_progress_hook(bus, manager, job_id)
    started = time.monotonic()
    estimator.fit(dataset_X, dataset_y, on_epoch=on_epoch)
    fit_seconds = time.perf_counter() - started
    record = TrainedModelRecord(
        model_id=f"mdl-{uuid.uuid4().hex[:12]}",
        model=estimator,
        name=network_name,
        task=task,
        dataset_id=dataset_id,
        target=target,
        features=[str(c) for c in dataset_X.columns],
        n_train_rows=len(dataset_X),
        best_params={"network": network_name, **best_params},
        cv_scores=None,  # no CV in this path — honest, not faked
        cv_mean=None,
        cv_std=None,
        fit_seconds=fit_seconds,
    )
    model_store.register(record)
    view = record.view()
    view["engine"] = "torch"
    view["predict_url"] = "/api/v1/ml/predict"

    # the refit's comparable score on the SAME carve — the acceptance
    # clause the roadmap names (the refit metric vs the fixed-config
    # baseline, one scale, never a second evaluation authority)
    refit_score = float(estimator.score(X_val, y_val))
    return {
        "model": view,
        "tune": {
            "best_params": dict(best.params),
            "best_score": -float(best.value),
            "n_trials": completed,
            "task": task,
            "trials": [
                {
                    "number": t.number,
                    "params": dict(t.params),
                    "value": t.value,
                }
                for t in study.trials
            ],
            "refit_score": refit_score,
            "sampler": sampler,
            "seed": seed,
            "searched": {
                name: dict(spec)
                for name, spec in (search_echo or {}).items()
            },
        },
    }


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
    fit_seconds = time.perf_counter() - started
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
    fit_seconds = time.perf_counter() - started
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
            sync_params: dict[str, Any] = dict(hyperparams)
            sync_params["_task"] = req.task
            return {
                "mode": "sync",
                **_cnn_train_body(x, y, sync_params, model_store, bus=bus),
            }
        except (DataValidationError, ModelError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    params: dict[str, Any] = dict(hyperparams)
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
    fit_seconds = time.perf_counter() - started
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


# -- DL-2: the GNN explainability lane (the roadmap's row 14) -------------


def _circular_layout(n_nodes: int) -> list[list[float]]:
    """The deterministic node positions for the sized-edge scatter.

    One node per angle on the unit circle — dependency-free and stable
    (the same graph always draws the same way; no spring iterations,
    no networkx, nothing random).
    """
    if n_nodes <= 0:
        return []
    if n_nodes == 1:
        return [[0.0, 0.0]]
    return [
        [
            round(math.cos(2.0 * math.pi * i / n_nodes), 6),
            round(math.sin(2.0 * math.pi * i / n_nodes), 6),
        ]
        for i in range(n_nodes)
    ]


class GnnExplainRequest(BaseModel):
    """POST /api/v1/dl/gnn-explain — DL-2's edge-attribution body."""

    model_id: str
    # the honest sync bound: masked forward passes are seconds per
    # explained graph — 20 keeps the worst request in the tens of
    # seconds, never the minutes a job would deserve
    graphs: list[GnnGraphSpec] = Field(min_length=1, max_length=20)
    # one label per graph — the PGExplainer's OWN training signal (its
    # scorer fits over the request's graphs); unused by the other
    # three methods (a label with them is a 422, never a silent skip)
    labels: list[int] | None = Field(default=None, min_length=1, max_length=20)
    method: Literal[
        "gnnexplainer", "pgexplainer", "attention", "subgraphx"
    ] = "gnnexplainer"
    top_k: int = Field(default=8, ge=1, le=100)
    # the class to explain (stamped onto every graph's y so every
    # explainer and both fidelity passes agree on ONE target); null =
    # the model's own argmax prediction per graph
    target: int | None = None
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    # the optimizer iters knob (gnnexplainer/pgexplainer only — the
    # teaching default 200/150 lands when null; subgraphx/attention
    # carry their own sampling scales)
    iters: int | None = Field(default=None, ge=10, le=2000)


@router.post("/gnn-explain")
def gnn_explain(req: GnnExplainRequest, request: Request) -> dict:
    """DL-2 — the graph models' edge attribution (sync by design).

    The REAL `pharmaml.ml.gnnxai` engine over a session-trained GNN:
    the chosen explainer's per-edge importance (aligned with each
    graph's own `edge_index` columns), the fidelity_plus /
    fidelity_minus drops, the sparsity, and the deterministic
    circular layout the frontend renders the sized-edge scatter
    over.  The explained subgraph never exceeds the full graph —
    `top_edges` is a subset of the edge columns by construction.
    """
    model_record = request.app.state.model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    model = model_record.model
    if not isinstance(model, GraphClassifier):
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is not a graph model; the "
                "explainability lane needs one trained through "
                "POST /api/v1/dl/gnn-train"
            ),
        )
    if req.labels is not None and req.method != "pgexplainer":
        raise HTTPException(
            status_code=422,
            detail=(
                "labels ride the pgexplainer lane only (its scorer "
                "trains over the request's own labeled graphs); the "
                f"{req.method!r} explainer takes graphs alone"
            ),
        )
    if req.method == "pgexplainer" and req.labels is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "the pgexplainer lane needs labels: one per graph (its "
                "parametric scorer trains over the request's own graphs)"
            ),
        )
    if (
        req.labels is not None
        and len(req.labels) != len(req.graphs)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"labels carries {len(req.labels)} entries for "
                f"{len(req.graphs)} graphs — exactly one per graph"
            ),
        )
    if req.iters is not None and req.method not in (
        "gnnexplainer",
        "pgexplainer",
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"the iters knob rides the optimizer explainers only "
                f"(gnnexplainer/pgexplainer); {req.method!r} carries its "
                "own sampling scales"
            ),
        )
    # the graphs (and the target stamp so every explainer + both
    # fidelity passes agree on ONE target class)
    datas: list[Data] = [spec.to_data() for spec in req.graphs]
    if req.target is not None:
        for data in datas:
            data.y = np.asarray(req.target, dtype=int)
    elif req.method == "pgexplainer":
        # the PG scorer's own training targets: the per-graph labels
        for data, label in zip(datas, req.labels or [], strict=False):
            data.y = np.asarray(int(label), dtype=int)

    explainer: GNNExplainer | PGExplainer | AttentionExplainer | SubgraphX
    # the per-method callable: SubgraphX's explain carries no top_k
    # (its own sampling scales) — the other three do
    explain_fn: Callable[[GraphClassifier, Data], EdgeExplanation]
    if req.method == "gnnexplainer":
        explainer = GNNExplainer(
            iters=(req.iters if req.iters is not None else 200),
            seed=req.seed,
        )
        explain_fn = partial(explainer.explain, top_k=req.top_k)
    elif req.method == "subgraphx":
        explainer = SubgraphX(seed=req.seed)
        explain_fn = explainer.explain
    elif req.method == "attention":
        explainer = AttentionExplainer()
        explain_fn = partial(explainer.explain, top_k=req.top_k)
    else:
        explainer = PGExplainer(
            iters=(req.iters if req.iters is not None else 150),
            seed=req.seed,
        )
        try:
            explainer.fit(model, datas, np.asarray(req.labels or [], dtype=int))
        except GnnError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        explain_fn = partial(explainer.explain, top_k=req.top_k)

    rows: list[dict[str, Any]] = []
    for index, data in enumerate(datas):
        try:
            explanation = explain_fn(model, data)
            fid_plus = fidelity_plus(model, data, explanation)
            fid_minus = fidelity_minus(model, data, explanation)
            spar = sparsity(explanation)
        except GnnError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        n_nodes = int(np.asarray(data.x).shape[0])
        rows.append(
            {
                "index": index,
                "n_nodes": n_nodes,
                "n_edges": int(explanation.scores.size),
                "target_class": int(explanation.target_class),
                "method": explanation.method,
                "scores": [float(s) for s in explanation.scores],
                "top_edges": [int(e) for e in explanation.top_edges],
                "fidelity_plus": float(fid_plus),
                "fidelity_minus": float(fid_minus),
                "sparsity": float(spar),
                "extra": json_safe(explanation.extra),
                "edge_index": [
                    [int(data.edge_index[0, c]), int(data.edge_index[1, c])]
                    for c in range(data.edge_index.shape[1])
                ],
                "layout": {
                    "kind": "circular",
                    "nodes": _circular_layout(n_nodes),
                },
            }
        )
    return {
        "mode": "sync",
        "model_id": req.model_id,
        "method": req.method,
        "top_k": req.top_k,
        "seed": req.seed,
        "graphs": rows,
        "note": (
            "the per-edge scores are aligned with each graph's own "
            "edge_index columns; fidelity_plus is the drop when the top "
            "edges are removed, fidelity_minus the drop when only they "
            "remain; the layout is the deterministic circular node "
            "position (the sized-edge scatter's canvas)"
        ),
    }

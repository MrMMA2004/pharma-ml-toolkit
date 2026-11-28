"""``/api/v1/ml`` — the ML workspace's API surface over the REAL
training backend (part10's queued row).

* ``GET /api/v1/ml/models`` — the registry catalog projection
  (:func:`pharmaml.ml.models.list_models`, one source of truth).
* ``POST /api/v1/ml/train`` — one registry model trained by the real
  :func:`pharmaml.ml.training.train_model` on a registered dataset
  (sync or managed job), the fitted estimator kept in the session
  model store for prediction and artifact export.
* ``DELETE /api/v1/ml/trained/{model_id}`` — the registry's deletion
  lane (part16 §15.3): one session-trained model evicted from the
  bounded store (the in-memory record, the persisted row and its
  vault payload when the durability lane is on).  The eviction is
  honest about scope: experiment-run records keep their
  model-artifact references (the lineage outlives the model — the
  record's fetch pointer degrades to the honest 404), exported
  artifacts are independent copies and survive, datasets are never
  touched, and an unknown id is a 404, never a silent no-op.
* ``POST /api/v1/ml/predict`` — predictions (and optional
  probabilities) from a session-trained model on inline rows or on a
  registered dataset's feature matrix. Dataset mode additionally
  carries the aligned ``evaluation`` block (part11 §22.5's point
  metadata lane): the bounded sample rows with the stable frame-index
  name, the actual target value, the prediction, the regression
  residual, and the SMILES string whenever the frame carries a
  detectable SMILES column — the raw the parity plot and the sample
  inspector in the GUI plot and click through.
* ``POST /api/v1/ml/explain`` — the SHAP summary of a session-trained
  model over (a bounded sample of) its dataset's feature matrix
  (:func:`pharmaml.ml.explain.shap_summary` — part11 §11.2's
  Explainability/Uncertainty stage, the r289 queue row closed here):
  the per-feature mean |SHAP| ranking, the bounded values preview for
  the beeswarm, and the optional importance + beeswarm figure
  artifacts (``plot_shap_importance`` / ``plot_shap_beeswarm``).
  Non-tree estimators are refused honestly with the typed error that
  names the permutation-importance sibling — never a silent fallback.
* ``POST /api/v1/ml/drift`` — ML-5 (the roadmap's row 12): the batch-
  level distribution-shift verdict
  (:func:`pharmaml.ml.drift.detect_drift` — the same engine the
  batch-training CLI treats as mandatory): a reference frame (a
  session-trained model's own training dataset, or a registered
  dataset) compared against a registered current dataset, per-feature
  PSI + KS with the credit-risk verdicts (``< 0.1`` stable, ``0.1 -
  0.2`` moderate, ``>= 0.2`` shifted — the engine's own exported
  constants). Sync by design (histogram sketches, sub-second at
  session sizes); a current frame sharing no numeric reference
  features is the honest ``n_features: 0`` answer, never an error;
  an empty comparison never fabricates a verdict.

``n_jobs`` passes through verbatim, including ``-1`` (all cores) —
the Phase-Eight passthrough clause, previewed here — but the DEFAULT
is the conservative ``1`` (part16 §5.2: a default all-cores spawn is
exactly the TerminatedWorkerError mechanism on constrained machines;
all-cores stays an explicit user choice). The resource-aware lane
(§5.3) estimates the working set before the fit and reduces
CONCURRENCY -- never the scientific workflow -- when the estimate
does not fit available memory, reporting the decision in the
``parallelism`` payload block (never silent).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml import __version__ as _toolkit_version
from pharmaml.chem import detect_smiles_column
from pharmaml.exceptions import (
    DataValidationError,
    DependencyError,
    ModelError,
)
from pharmaml.ml.autoencoder import TabularAutoencoder
from pharmaml.ml.cnn import SequenceCNNClassifier, SequenceCNNRegressor
from pharmaml.ml.drift import detect_drift
from pharmaml.ml.explain import shap_summary
from pharmaml.ml.gnn import GraphClassifier
from pharmaml.ml.models import list_models, resolve_model_name
from pharmaml.ml.provenance import environment_snapshot
from pharmaml.ml.splits import scaffold_groups, split_indices
from pharmaml.ml.training import train_model
from pharmaml.viz._compat import close_figure
from pharmaml.viz.export import export_figure
from pharmaml.viz.ml_plots import plot_shap_beeswarm, plot_shap_importance

from ..evaluation_curves import enrich_evaluation_block
from ..jobs import CancelledJob
from ..runs import build_ml_train_failed_run, build_ml_train_run
from ..stores import TrainedModelRecord, json_safe

router = APIRouter(prefix="/ml", tags=["ml"])

# the SHAP preview bounds (§17.3: explicit, never silent): the rows
# sampled for the explanation, and the preview matrices' feature
# columns (the top-ranked ones; the ranking itself carries all)
_MAX_EXPLAIN_ROWS = 500
_MAX_PREVIEW_FEATURES = 50

# the evaluation block's own wire bound (§17.3 again: the parity
# plot's sample is bounded, the bound is reported, never silent)
_MAX_EVALUATION_ROWS = 500


class TrainSplitSpec(BaseModel):
    """ML-3 -- the Train lane's split section (the roadmap's row 2).

    The same engine vocabulary DS-1 exposed at the Datasets stage
    (``pharmaml.ml.splits``), now riding ``POST /ml/train``: with
    ``split`` set, the route carves a held-out side through
    :func:`pharmaml.ml.splits.split_indices` BEFORE the fit -- the
    training frame the estimator sees is the carved train side, and the
    carve itself (the two sides, the per-side group census and the
    ``shared_groups`` leakage verdict) is reported in the train
    response and in the §15.2 run record's ``cross_validation.split``
    block at both exits.  ``smiles_column`` left null = the honest
    auto-detection (scaffold); ``group_column`` is the group lane's
    required input, never guessed (the DS-1 prerequisites verbatim).
    """

    strategy: Literal[
        "random", "stratified", "scaffold", "group", "kennard_stone", "temporal"
    ]
    smiles_column: str | None = None
    group_column: str | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class TrainRequest(BaseModel):
    dataset_id: str
    model: str = Field(min_length=1)
    task: Literal["classification", "regression"] | None = None
    tune: bool = True
    cv: int = Field(default=5, ge=2, le=10)
    # part16 §5.2 -- the conservative default: -1 (all cores) stays
    # explicitly requestable (ge=-1, le=64 passthrough preserved), but
    # the default parallelism is sequential -- the all-cores default
    # is the TerminatedWorkerError mechanism on constrained machines.
    n_jobs: int = Field(default=1, ge=-1, le=64)
    preprocess: bool = True
    random_state: int = 42
    model_params: dict[str, Any] | None = None
    # ML-3 -- the optional strategy-aware held-out carve before the fit
    # (null = the legacy full-frame lane, byte-identical behavior).
    split: TrainSplitSpec | None = None
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


class ExplainRequest(BaseModel):
    """POST /api/v1/ml/explain — the §11.2 Explainability stage body."""

    model_id: str
    dataset_id: str | None = None
    max_rows: int = Field(default=200, ge=1, le=_MAX_EXPLAIN_ROWS)
    figure: bool = False
    mode: Literal["job", "sync"] = "sync"


class DriftRequest(BaseModel):
    """POST /api/v1/ml/drift — ML-5's distribution-shift body.

    Exactly one reference source: the training dataset of a
    session-trained model (``model_id`` — the re-scoring lane the
    roadmap names) or a registered dataset (``dataset_id``).  The
    current batch is always a registered dataset (``new_dataset_id``).
    """

    model_id: str | None = None
    dataset_id: str | None = None
    new_dataset_id: str
    features: list[str] | None = Field(default=None, max_length=200)

    def _exactly_one_reference(self) -> None:
        if (self.model_id is None) == (self.dataset_id is None):
            raise ValueError(
                "give either model_id=... or dataset_id=... as the drift "
                "reference, not both/neither"
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


@router.delete("/trained/{model_id}")
def evict_trained_model(model_id: str, request: Request) -> dict:
    """part16 §15.3 (Phase 12) — the model registry's deletion lane.

    Evicts one session-trained model from the bounded registry (the
    in-memory record plus, through ``ModelStore._forget``, the
    persisted ``trained_models`` row and its vault payload when the
    durability lane is on — the same channel the artifacts' evict
    lane rides, never a bespoke second deletion path).  The response
    carries the evicted id and the remaining count so the GUI can
    refresh honestly without a second listing call.

    Scope, per the directive's own rails: the experiment-run records
    keep their ``model_artifact`` references (the §15.2 lineage
    outlives the model — the record's ``fetch`` pointer degrades to
    the honest 404, never rewritten), exported joblib artifacts are
    independent copies that survive, and the dataset the model was
    trained on is untouched.  An unknown id is a 404, never a silent
    no-op.
    """
    if not request.app.state.model_store.evict(model_id):
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    return {
        "evicted": model_id,
        "remaining": len(request.app.state.model_store.list()),
    }


def _resolve_parallelism(
    n_jobs: int, X: pd.DataFrame
) -> tuple[int, dict]:
    """part16 §5.3 -- the resource-aware parallelism lane.

    Estimates the training working set BEFORE the fit spawns workers
    and reduces concurrency -- never the scientific workflow -- when
    the estimate does not fit available memory, reporting the clamp
    in the returned advisory (``requested``/``executed``/``reason``/
    ``estimated_bytes``/``available_bytes`` -- never silent).

    The estimate is deliberately conservative and documented: the raw
    feature matrix is pickled once per worker (``n_workers + 1``
    copies), each fold fits a full pipeline (imputer + scaler + the
    estimator's own buffers), so a 4x safety factor over the raw
    matrix bytes per worker lane is the honest floor, not hype.  When
    ``psutil`` is unavailable the lane degrades to the requested
    parallelism with a best-effort advisory (``reason`` names the
    missing instrument) -- the clause is "where practical", never a
    hard dependency.
    """
    if n_jobs == 0:
        workers = 1
    else:
        workers = n_jobs
    # the ESTIMATE worker count: -1 (all cores) resolves to the real
    # core count for the memory math only -- the passthrough value is
    # never silently rewritten (the no-clamp path returns it verbatim).
    if n_jobs == -1:
        import os

        est_workers = os.cpu_count() or 1
    else:
        est_workers = max(workers, 1)
    advisory: dict[str, Any] = {
        "requested": n_jobs,
        "executed": est_workers,
        "reason": None,
        "estimated_bytes": None,
        "available_bytes": None,
    }
    try:
        import psutil
    except ImportError:
        advisory["reason"] = (
            "psutil unavailable -- parallelism unclamped (best-effort)"
        )
        return (n_jobs if n_jobs != 0 else 1), advisory
    available = int(psutil.virtual_memory().available)
    advisory["available_bytes"] = available
    matrix_bytes = int(
        X.memory_usage(deep=True).sum()
        if isinstance(X, pd.DataFrame) else int(getattr(X, "nbytes", 0))
    )
    # the per-worker working set: the pickled frame + the fitted
    # pipeline buffers (the documented 4x floor).
    per_worker = matrix_bytes * 4
    estimated = per_worker * (est_workers + 1)
    advisory["estimated_bytes"] = estimated
    if estimated <= int(available * 0.75):
        # no clamp: the requested value passes through VERBATIM (the
        # Phase-Eight passthrough clause, preserved) -- only the
        # reported executed count tells the user what -1 means.
        return (n_jobs if n_jobs != 0 else 1), advisory
    # reduce concurrency (never the science): drop workers until the
    # estimate fits, floor at 1 -- sequential execution always fits if
    # the matrix itself does.
    clamped = est_workers
    while clamped > 1 and per_worker * (clamped + 1) > int(available * 0.75):
        clamped -= 1
    advisory["executed"] = clamped
    advisory["reason"] = (
        f"memory guard: the estimated working set ({estimated} bytes over "
        f"{est_workers} workers) exceeded the safe fraction of "
        f"available memory ({available} bytes); concurrency reduced to "
        f"{clamped} worker(s) -- the training itself is unchanged, only "
        "the wall-clock is longer"
    )
    return clamped, advisory


def _train_split_slice(
    spec: TrainSplitSpec,
    frame: pd.DataFrame,
    y_full: pd.Series,
    target: str | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """ML-3: one strategy carve of the registered frame before the fit.

    The carve runs over the FULL registered frame (the strategy inputs
    -- SMILES, group labels -- are columns the loader's own feature
    matrix legitimately excludes), through the same
    :func:`pharmaml.ml.splits.split_indices` engine the CLI, the
    in-backend templates and the Datasets split route ride -- never a
    second splitter.  The prerequisites are the DS-1 lane's own honest
    ``DataValidationError`` 422s (the scaffold smiles auto-pick, the
    never-guessed group column, kennard_stone's numeric matrix with
    the target column excluded).  Returns the positional train/heldout
    index arrays plus the response/record block: the two sides' row
    counts and, for the group-carrying strategies, the per-side group
    census + the ``shared_groups`` leakage verdict (which the engine's
    whole-group assignment guarantees is empty -- the census states
    it, never assumes it).
    """
    strategy = spec.strategy
    smiles_col: str | None = None
    if strategy == "scaffold":
        smiles_col = spec.smiles_column
        if smiles_col is None:
            # the same auto-pick the §15.4 quality card and DS-1 use;
            # raises DataValidationError naming every column it tried
            smiles_col = detect_smiles_column(frame)
        elif smiles_col not in frame.columns:
            raise DataValidationError(
                f"smiles column {smiles_col!r} not found; available: "
                f"{[str(c) for c in frame.columns][:20]}"
            )
    group_col: str | None = None
    if strategy == "group":
        group_col = spec.group_column
        if group_col is None:
            raise DataValidationError(
                "the 'group' strategy needs an explicit group_column "
                "(group labels are never guessed)"
            )
        if group_col not in frame.columns:
            raise DataValidationError(
                f"group column {group_col!r} not found; available: "
                f"{[str(c) for c in frame.columns][:20]}"
            )
    feature_cols: list[str] = []
    if strategy == "kennard_stone":
        feature_cols = [
            str(c) for c in frame.columns
            if str(c) != target and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if not feature_cols:
            raise DataValidationError(
                "kennard_stone needs at least one numeric feature column "
                "(distances are computed on the numeric matrix)"
            )
    X_arg: object = (
        frame[feature_cols].to_numpy(dtype=float)
        if strategy == "kennard_stone"
        else frame
    )
    y_arg: object = y_full if strategy == "stratified" else None
    smiles = (
        [str(v) for v in frame[smiles_col].tolist()]
        if strategy == "scaffold"
        else None
    )
    groups = frame[group_col].tolist() if strategy == "group" else None
    train_idx, heldout_idx = split_indices(
        X=X_arg,
        y=y_arg,
        strategy=strategy,
        test_size=spec.test_size,
        random_state=random_state,
        smiles=smiles,
        groups=groups,
    )
    n = len(frame)
    block: dict[str, Any] = {
        "strategy": strategy,
        "smiles_column": smiles_col if strategy == "scaffold" else None,
        "group_column": group_col if strategy == "group" else None,
        "test_size": spec.test_size,
        "random_state": random_state,
        "n_rows": n,
        "sides": {
            "train": {"rows": len(train_idx)},
            "heldout": {"rows": len(heldout_idx)},
        },
        "group_census": None,
    }
    if strategy in ("scaffold", "group"):
        if strategy == "scaffold":
            keys = scaffold_groups([str(v) for v in frame[smiles_col].tolist()])
        else:
            keys = [str(v) for v in frame[group_col].tolist()]
        train_keys = [keys[int(i)] for i in train_idx]
        heldout_keys = [keys[int(i)] for i in heldout_idx]

        def _counts(side_keys: list[str]) -> dict[str, int]:
            side_counts: dict[str, int] = {}
            for key in side_keys:
                side_counts[key] = side_counts.get(key, 0) + 1
            return side_counts

        train_counts = _counts(train_keys)
        heldout_counts = _counts(heldout_keys)

        def _top(counts: dict[str, int]) -> list[list[object]]:
            return [
                [key, count]
                for key, count in sorted(
                    counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[:8]
            ]

        block["group_census"] = {
            "train": {
                "n_groups": len(train_counts),
                "top_groups": _top(train_counts),
            },
            "heldout": {
                "n_groups": len(heldout_counts),
                "top_groups": _top(heldout_counts),
            },
            "shared_groups": sorted(
                set(train_counts) & set(heldout_counts)
            ),
        }
    return train_idx, heldout_idx, block


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _record_run(
    run_store,
    status: str,
    error: str,
    mode: str,
    job_id,
    dataset_view,
    model_name: str,
    task: str,
    tune: bool,
    cv: int,
    n_jobs: int,
    preprocess: bool,
    random_state: int,
    model_params,
    started_at: str,
    split: dict | None = None,
) -> None:
    """The failure/cancellation exit's run record (§15.2's honest lane).

    ``split`` is the ML-3 carve block as it stood at the failure: the
    computed block when the carve had already run (the fit itself
    failed), or the requested echo plus its ``error`` when the carve
    was refused (a doomed split configuration is traceable "do not
    re-run this" information, exactly like a doomed model_params).
    """
    if run_store is None:
        return
    run_store.register(build_ml_train_failed_run(
        mode=mode,
        job_id=job_id,
        dataset_view=dataset_view or {},
        model_name=model_name,
        task=task,
        tune=tune,
        cv=cv,
        n_jobs_requested=n_jobs,
        preprocess=preprocess,
        random_state=random_state,
        model_params=model_params,
        split=split,
        environment=environment_snapshot(),
        toolkit_version=_toolkit_version,
        started_at=started_at,
        completed_at=_now_iso(),
        error=error,
        status=status,
    ))


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
    dataset_view: dict | None = None,
    run_store=None,
    mode: str = "job",
    cancel_event=None,
    manager=None,
    job_id: str | None = None,
    split: TrainSplitSpec | None = None,
    dataset_frame: pd.DataFrame | None = None,
) -> dict:
    """Job/sync body: the REAL train_model call + store registration.

    part16 §15.2: the body records the experiment run at both exits —
    the completed run (the full §15.2 lineage view) and the
    failed/cancelled run (the config that did not land, traceable so
    it is not re-run blind).  The records are built from the very
    objects this body already holds — no second source, no
    re-derivation.

    ML-3: with a ``split`` section the body carves the held-out side
    through the shared ``ml/splits`` engine BEFORE the fit (the carve
    itself can fail honestly — its 422 exits record the refused echo,
    the same lane every other doomed configuration rides), and the
    carve block rides the response and the §15.2 record's
    ``cross_validation.split`` at both exits.
    """
    started_at = _now_iso()
    split_echo: dict[str, Any] | None = None
    if split is not None:
        split_echo = {
            "strategy": split.strategy,
            "smiles_column": (
                split.smiles_column if split.strategy == "scaffold" else None
            ),
            "group_column": (
                split.group_column if split.strategy == "group" else None
            ),
            "test_size": split.test_size,
            "random_state": random_state,
            "n_rows": len(dataset_X),
            "sides": None,
            "group_census": None,
            "error": None,
        }
    if cancel_event is not None and cancel_event.is_set():
        _record_run(
            run_store, "cancelled", "Cancelled before the fit stage.",
            mode, job_id, dataset_view, model_name, task, tune, cv,
            n_jobs, preprocess, random_state, model_params,
            started_at, split=split_echo,
        )
        raise CancelledJob("Cancelled before the fit stage.")
    split_block: dict[str, Any] | None = split_echo
    if split is not None:
        # the strategy inputs (SMILES / group labels) live in the FULL
        # registered frame, not the loader's numeric feature matrix; the
        # positional indices the engine returns slice both the same.
        frame = dataset_frame if dataset_frame is not None else dataset_X
        try:
            train_idx, _heldout_idx, split_block = _train_split_slice(
                split, frame, dataset_y, target, random_state
            )
            dataset_X = dataset_X.iloc[train_idx]
            dataset_y = dataset_y.iloc[train_idx]
        except DataValidationError as exc:
            refused = dict(split_echo or {})
            refused["error"] = str(exc)
            _record_run(
                run_store, "failed", f"{type(exc).__name__}: {exc}",
                mode, job_id, dataset_view, model_name, task, tune, cv,
                n_jobs, preprocess, random_state, model_params,
                started_at, split=refused,
            )
            raise
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 10, "preparing features")
    # part16 §5.3 -- resolve the effective parallelism BEFORE the fit
    # (the memory-aware clamp; concurrency may reduce, the science
    # never does, the decision is always reported).  The estimate runs
    # over the carved train side -- the frame the fit actually sees.
    effective_n_jobs, parallelism_advisory = _resolve_parallelism(
        n_jobs, dataset_X
    )
    try:
        result = train_model(
            model_name,
            task=task,
            X_train=dataset_X,
            y_train=dataset_y,
            tune=tune,
            cv=cv,
            n_jobs=effective_n_jobs,
            preprocess=preprocess,
            random_state=random_state,
            model_params=model_params,
        )
    except CancelledJob:
        _record_run(
            run_store, "cancelled", "Cancelled during the fit stage.",
            mode, job_id, dataset_view, model_name, task, tune, cv,
            n_jobs, preprocess, random_state, model_params,
            started_at, split=split_block,
        )
        raise
    except Exception as exc:
        _record_run(
            run_store, "failed", f"{type(exc).__name__}: {exc}",
            mode, job_id, dataset_view, model_name, task, tune, cv,
            n_jobs, preprocess, random_state, model_params,
            started_at, split=split_block,
        )
        raise
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
    if run_store is not None:
        run_store.register(build_ml_train_run(
            mode=mode,
            job_id=job_id,
            dataset_view=dataset_view or {},
            model_name=model_name,
            task=task,
            tune=tune,
            cv=cv,
            n_jobs_requested=n_jobs,
            n_jobs_effective=effective_n_jobs,
            parallelism_advisory=parallelism_advisory,
            preprocess=preprocess,
            random_state=random_state,
            model_params=model_params,
            split=split_block,
            training_result=result,
            model_view=record.view(),
            environment=environment_snapshot(),
            toolkit_version=_toolkit_version,
            started_at=started_at,
            completed_at=_now_iso(),
        ))
    payload: dict[str, Any] = {
        "model": record.view(),
        "parallelism": parallelism_advisory,
    }
    if split_block is not None:
        payload["split"] = split_block
    return payload


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
                dataset_view=dataset.view(),
                run_store=request.app.state.run_store,
                mode="sync",
                split=req.split,
                dataset_frame=dataset.frame,
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
            "split": (
                req.split.model_dump() if req.split is not None else None
            ),
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
        dataset_view=dataset.view(),
        run_store=request.app.state.run_store,
        mode="job",
        manager=manager,
        split=req.split,
        dataset_frame=dataset.frame,
    )
    return {"mode": "job", "job": record.public()}


def _evaluation_block(dataset: Any, predictions: Any) -> dict:
    """§22.5 — the aligned actual/predicted rows for the parity lane.

    Dataset mode only: the frame's own index is the stable row
    identity, ``y`` carries the actuals, the SMILES column (when the
    frame has one, by ``detect_smiles_column``'s two-pass probe)
    carries the structures, and the regression residual is
    ``actual - predicted`` over the numeric pairs. The block is
    bounded to the first ``_MAX_EVALUATION_ROWS`` rows with both
    counts reported — a display bound never a truncation of the
    model's authoritative prediction set (the Export stage registers
    the full table).
    """
    frame = dataset.frame
    y = dataset.y
    smiles_col: str | None = None
    try:
        smiles_col = detect_smiles_column(frame)
    except DataValidationError:
        smiles_col = None  # a numeric-only frame simply has no structures
    regression = dataset.task == "regression"
    n_available = len(predictions)
    bounded = min(n_available, _MAX_EVALUATION_ROWS)
    rows: list[dict[str, Any]] = []
    for i in range(bounded):
        actual = y.iloc[i]
        predicted = predictions[i]
        residual = None
        if regression and isinstance(
            actual, (int, float, np.number)
        ) and isinstance(predicted, (int, float, np.number)):
            residual = json_safe(float(actual) - float(predicted))
        smiles = None
        if smiles_col is not None:
            value = frame.iloc[i][smiles_col]
            if value is not None and not (
                isinstance(value, float) and np.isnan(value)
            ):
                smiles = str(value)
        rows.append(
            {
                "name": json_safe(frame.index[i]),
                "actual": json_safe(actual),
                "predicted": json_safe(predicted),
                "residual": residual,
                "smiles": smiles,
            }
        )
    return {
        "target": dataset.target,
        "task": dataset.task,
        "smiles_col": smiles_col,
        "n_rows_available": n_available,
        "n_rows_evaluated": bounded,
        "rows": rows,
    }


def _predict_body(
    model_record: TrainedModelRecord,
    X: pd.DataFrame,
    want_proba: bool,
    dataset: Any | None = None,
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
    if dataset is not None and len(dataset.y) == len(X):
        payload["evaluation"] = _evaluation_block(dataset, predictions)
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
    # part21 §3 — the evaluation-sample diagnostics ride the same
    # pass: regression rows gain the residual histogram/stats/Q–Q,
    # classification rows gain per-class metrics and (when the pass
    # carried probabilities) the ROC/PR curves + the binary
    # reliability diagram. Every attachment is conditional on its
    # real data being present (never fabricated, never a second
    # evaluation of the model's authoritative set).
    if "evaluation" in payload:
        enrich_evaluation_block(
            payload["evaluation"], payload.get("probabilities")
        )
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
    if isinstance(model_record.model, GraphClassifier):
        # graph models predict through their own lane — never a tabular
        # predict over node-feature matrices (one path per model class)
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is a graph model; predict it "
                "through POST /api/v1/dl/gnn-predict with graph rows"
            ),
        )
    if isinstance(model_record.model, (SequenceCNNClassifier, SequenceCNNRegressor)):
        # sequence CNN models predict through their own lane — never a
        # tabular predict over time-step matrices (one path per class)
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is a sequence model; predict it "
                "through POST /api/v1/dl/cnn-predict with sequence rows"
            ),
        )
    if isinstance(model_record.model, TabularAutoencoder):
        # the autoencoder is unsupervised — its encode lane, never a
        # supervised predict (one path per model class)
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is an unsupervised autoencoder; its "
                "lane is POST /api/v1/dl/autoencoder-encode (latent + "
                "reconstruction errors)"
            ),
        )
    if req.rows is not None:
        X = pd.DataFrame(req.rows)
        evaluation_dataset = None  # inline rows carry no aligned actuals
    else:
        dataset = request.app.state.dataset_store.get(
            req.dataset_id or ""
        )
        if dataset is None:
            raise HTTPException(
                status_code=404, detail="Unknown dataset_id."
            )
        X = dataset.X
        evaluation_dataset = dataset
    if req.mode == "sync":
        try:
            return {"mode": "sync", **_predict_body(
                model_record, X, req.predict_proba,
                dataset=evaluation_dataset,
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
        dataset=evaluation_dataset,
    )
    return {"mode": "job", "job": record.public()}


# ------------------------------------------------------------------ explain


def _explain_body(
    model_record: TrainedModelRecord,
    X: pd.DataFrame,
    max_rows: int,
    figure: bool,
    artifact_store=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL ``shap_summary`` engine, bounded preview.

    The explanation runs over the first ``max_rows`` rows of the
    feature matrix (a deterministic, explicitly-bounded sample — the
    payload reports both ``n_rows_available`` and the explained
    ``n_samples`` so the bound is never silent). Non-tree estimators
    surface the engine's own typed refusal (which names the
    permutation-importance sibling) as a 422 — no silent fallback.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the SHAP stage.")
    if manager is not None and job_id is not None:
        manager.report_progress(
            job_id, 10, "sampling the explanation frame"
        )
    X_sample = X.head(max_rows)
    try:
        summary = shap_summary(model_record.model, X_sample)
    except (
        DataValidationError, DependencyError, ModelError, ValueError
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 55, "assembling the SHAP summary")

    ranking = [
        {"feature": name, "mean_abs_shap": float(value)}
        for name, value in summary.ranking()
    ]
    # the preview matrices carry only the top-ranked feature columns
    # (the ranking above carries every feature — the bound is on the
    # wire size, never on the honest inventory)
    keep = [
        str(row["feature"])
        for row in ranking[:_MAX_PREVIEW_FEATURES]
    ]
    indices = [summary.names.index(name) for name in keep]
    values = np.asarray(summary.values, dtype=float)
    feature_values = np.asarray(summary.feature_values, dtype=float)
    preview = {
        "max_features": _MAX_PREVIEW_FEATURES,
        "features": keep,
        "values": values[:, indices].tolist() if indices else [],
        "feature_values": (
            feature_values[:, indices].tolist() if indices else []
        ),
    }

    artifacts: list[dict] = []
    if figure:
        if manager is not None and job_id is not None:
            manager.report_progress(
                job_id, 80, "rendering the SHAP figures"
            )
        for filename, plot in (
            ("shap-importance.png", plot_shap_importance),
            ("shap-beeswarm.png", plot_shap_beeswarm),
        ):
            if artifact_store is None:
                break
            try:
                fig = plot(summary)
            except (DataValidationError, DependencyError) as exc:
                raise HTTPException(
                    status_code=422, detail=str(exc)
                ) from exc
            with tempfile.TemporaryDirectory(prefix="pharmaml-shap-") as tmp:
                target = Path(tmp) / filename
                export_figure(fig, target)
                data = target.read_bytes()
            record = artifact_store.register(
                kind="figure",
                filename=filename,
                content_type="image/png",
                data=data,
                meta={"model_id": model_record.model_id},
            )
            artifacts.append(record.view())
            close_figure(fig)

    return {
        "explain": {
            "model_id": model_record.model_id,
            "model_name": summary.model_name,
            "task": summary.task,
            "engine": "shap_summary",
            "n_rows_available": len(X),
            "n_samples": int(summary.n_samples),
            "n_features": len(summary.names),
            "base_value": (
                None if summary.base_value is None
                else float(summary.base_value)
            ),
            "ranking": ranking,
            "preview": preview,
            "artifacts": artifacts,
        }
    }


@router.post("/explain")
def explain(req: ExplainRequest, request: Request) -> dict:
    model_store = request.app.state.model_store
    model_record = model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    if isinstance(model_record.model, GraphClassifier):
        # the SHAP surface is tabular; graph models are refused up front
        # with the GNN explainability pointer instead of a misleading
        # dataset miss
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is a graph model; the SHAP summary "
                "covers tabular models — graph explainability rides "
                "pharmaml.ml.gnnxai in the backend library"
            ),
        )
    if isinstance(
        model_record.model, (SequenceCNNClassifier, SequenceCNNRegressor)
    ):
        # the SHAP surface is tabular; sequence models have their own lane
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is a sequence model; the SHAP summary "
                "covers tabular models — sequence models predict through "
                "POST /api/v1/dl/cnn-predict"
            ),
        )
    if isinstance(model_record.model, TabularAutoencoder):
        # the SHAP surface is supervised-tabular; an unsupervised
        # autoencoder has no SHAP path at all
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {req.model_id} is an unsupervised autoencoder; the "
                "SHAP summary covers supervised tabular models — the AE "
                "lane is POST /api/v1/dl/autoencoder-encode"
            ),
        )
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
    X = dataset.X
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_explain_body(
                model_record,
                X,
                req.max_rows,
                req.figure,
                artifact_store=request.app.state.artifact_store,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "ml.explain",
        "ml",
        _explain_body,
        configuration={
            "model_id": req.model_id,
            "dataset_id": dataset_id,
            "n_rows_available": len(X),
            "max_rows": req.max_rows,
            "figure": req.figure,
        },
        model_record=model_record,
        X=X,
        max_rows=req.max_rows,
        figure=req.figure,
        artifact_store=request.app.state.artifact_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


# ------------------------------------------------------------- drift (ML-5)


def _feature_drift_row(feature: Any) -> dict[str, Any]:
    """One per-feature drift row, JSON-safe, verdict verbatim."""
    return {
        "feature": feature.name,
        "psi": float(feature.psi),
        "ks_statistic": (
            None if feature.ks_statistic is None else float(feature.ks_statistic)
        ),
        "ks_pvalue": (
            None if feature.ks_pvalue is None else float(feature.ks_pvalue)
        ),
        "reference_mean": float(feature.reference_mean),
        "current_mean": float(feature.current_mean),
        "mean_direction": feature.mean_direction,
        "verdict": feature.verdict,
    }


@router.post("/drift")
def drift(req: DriftRequest, request: Request) -> dict:
    """ML-5 — the batch-level distribution-shift verdict (sync).

    ``detect_drift`` over the stored frames: the reference (the
    model's own training dataset restricted to its feature
    vocabulary, or a registered dataset) vs the current registered
    dataset.  Per-feature PSI + KS, the credit-risk verdicts, the
    drifted list worst-first — the engine's own exported constants
    ride the payload so the GUI never re-encodes a threshold.
    """
    try:
        req._exactly_one_reference()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dataset_store = request.app.state.dataset_store
    if req.model_id is not None:
        model_record = request.app.state.model_store.get(req.model_id)
        if model_record is None:
            raise HTTPException(status_code=404, detail="Unknown model_id.")
        reference_dataset_id = model_record.dataset_id
        if reference_dataset_id is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "model " + req.model_id + " carries no training "
                    "dataset_id (a persisted-bundle reload); pass "
                    "dataset_id=... as the drift reference instead"
                ),
            )
        reference_dataset = dataset_store.get(reference_dataset_id)
        if reference_dataset is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "the model's training dataset " + reference_dataset_id
                    + " is no longer registered in this session; "
                    "re-upload it or pass dataset_id=... explicitly"
                ),
            )
        reference = reference_dataset.X
        # the model's own feature vocabulary, in its own order (the
        # engine skips features the current frame lacks — the AD
        # box's honesty, never a 422 over a missing column)
        features: list[str] | None = (
            list(model_record.features)
            if req.features is None
            else req.features
        )
        reference_label = {
            "kind": "model",
            "model_id": model_record.model_id,
            "name": model_record.name,
            "dataset_id": reference_dataset_id,
        }
    else:
        reference_dataset = dataset_store.get(req.dataset_id or "")
        if reference_dataset is None:
            raise HTTPException(status_code=404, detail="Unknown dataset_id.")
        reference = reference_dataset.X
        features = req.features
        reference_label = {
            "kind": "dataset",
            "dataset_id": reference_dataset.dataset_id,
            "name": reference_dataset.name,
        }
    current = dataset_store.get(req.new_dataset_id)
    if current is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown new_dataset_id.",
        )
    try:
        report = detect_drift(reference, current.X, features=features)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "mode": "sync",
        "reference": {
            **reference_label,
            "n_reference_rows": len(reference),
        },
        "current": {
            "dataset_id": current.dataset_id,
            "name": current.name,
            "n_rows": int(report.n_rows),
        },
        "warn_psi": float(report.warn_psi),
        "shift_psi": float(report.shift_psi),
        "n_features": len(report.features),
        "n_drifted": report.n_drifted,
        "drift_share": float(report.drift_share),
        "features": [_feature_drift_row(f) for f in report.features],
        "drifted": [_feature_drift_row(f) for f in report.drifted],
        "summary": str(report),
    }

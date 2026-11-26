"""``/api/v1/generative`` — the Generative AI workspace's API surface
over the REAL backend (part10's queued row).

* ``GET /api/v1/generative/objectives`` — the scoring-oracle catalog
  (the :mod:`pharmaml.generative.scoring` vocabulary: QED, SAS,
  band objectives, similarity, substructure, novelty, and the
  weighted multi-objective) with each objective's parameter schema.
* ``POST /api/v1/generative/rnn/train`` —
  :meth:`pharmaml.generative.rnn.SmilesRnn.train_prior` over an
  inline corpus or a registered dataset's SMILES column (sync or
  managed job).  The prior registers in the session prior store
  with its REAL per-epoch loss curve; job mode publishes the loss
  curve as a ``training.metrics`` event when training ends.
* ``POST /api/v1/generative/rnn/sample`` —
  :meth:`SmilesRnn.sample` from a session prior (temperature, top-k,
  seed, max length — all passed through to the backend verbatim).
* ``POST /api/v1/generative/optimize`` —
  :class:`pharmaml.generative.genetic.GeneticOptimizer` over a
  built objective spec, seed SMILES and/or corpus.  The optimizer's
  OWN progress callback drives ``iteration.update`` events per
  generation; returning ``False`` from the adapter when the job's
  cancel flag is set triggers the backend's native
  ``JobCancelled`` — the evolution itself stops at the checkpoint,
  and the job lands in CANCELLED (never FAILED).
* ``POST /api/v1/generative/gan/train`` + ``/gan/sample`` —
  :class:`pharmaml.generative.gan.FeatureSpaceGAN` over a
  registered dataset's numeric feature matrix (part18 §5.6's
  adversarial lane, integrated HERE in Generative AI per the spec,
  never the supervised registry).  FEATURE-space, not molecules:
  the responses say so, and sampling carries the honest
  distributional parity report.
* ``POST /api/v1/generative/diffusion/train`` +
  ``/diffusion/sample`` —
  :class:`pharmaml.generative.diffusion.FeatureSpaceDiffusion`, the
  same contract over the DDPM machinery (linear schedule, noise
  prediction, the full reverse chain).  Same honest scope: feature
  rows, not molecules.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import (
    DataValidationError,
    ModelError,
)
from pharmaml.exceptions import (
    JobCancelled as BackendJobCancelled,
)
from pharmaml.generative.diffusion import FeatureSpaceDiffusion
from pharmaml.generative.gan import FeatureSpaceGAN
from pharmaml.generative.genetic import GeneticOptimizer
from pharmaml.generative.rnn import SmilesRnn
from pharmaml.generative.scoring import (
    LogPBandObjective,
    MolecularWeightBandObjective,
    MultiObjective,
    NoveltyObjective,
    QedObjective,
    SAScoreObjective,
    SimilarityObjective,
    SubstructureObjective,
    TpsaBandObjective,
)

from ..events import EventType
from ..jobs import CancelledJob
from ..stores import FeaturePriorRecord, PriorRecord, json_safe

router = APIRouter(prefix="/generative", tags=["generative"])

_MAX_INLINE_SMILES = 5000
_MAX_SAMPLE = 500

_OBJECTIVE_CATALOG: list[dict[str, Any]] = [
    {
        "kind": "qed", "class": "QedObjective",
        "params": {},
        "description": "quantified drug-likeness (0..1, higher better)",
    },
    {
        "kind": "sas", "class": "SASScoreObjective",
        "params": {"easy": 3.0, "hard": 7.0},
        "description": "synthetic accessibility score (lower better)",
    },
    {
        "kind": "logp_band", "class": "LogPBandObjective",
        "params": {"lo": None, "hi": None},
        "description": "LogP inside [lo, hi]",
    },
    {
        "kind": "mw_band", "class": "MolecularWeightBandObjective",
        "params": {"lo": None, "hi": None},
        "description": "molecular weight inside [lo, hi] Da",
    },
    {
        "kind": "tpsa_band", "class": "TpsaBandObjective",
        "params": {"lo": None, "hi": None},
        "description": "topological polar surface area inside [lo, hi]",
    },
    {
        "kind": "similarity", "class": "SimilarityObjective",
        "params": {"seed_smiles": None, "radius": 2, "n_bits": 2048},
        "description": "Tanimoto similarity to a seed molecule",
    },
    {
        "kind": "substructure", "class": "SubstructureObjective",
        "params": {"smarts": None},
        "description": "contains the SMARTS substructure (0/1)",
    },
    {
        "kind": "novelty", "class": "NoveltyObjective",
        "params": {"corpus": None, "radius": 2, "n_bits": 2048},
        "description": "dissimilarity from a reference corpus",
    },
    {
        "kind": "multi", "class": "MultiObjective",
        "params": {
            "objectives": ["nested objective specs"],
            "weights": "optional per-objective weights",
        },
        "description": "weighted combination of nested objectives",
    },
]


def _number(spec: dict, key: str, kind: str) -> float:
    value = spec.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail=f"objective {kind!r} needs numeric {key!r}",
        )
    return float(value)


def _string(spec: dict, key: str, kind: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=422,
            detail=f"objective {kind!r} needs string {key!r}",
        )
    return value


def _build_objective(spec: Any) -> Any:
    """Build the REAL scoring objective from a request spec."""
    if not isinstance(spec, dict):
        raise HTTPException(
            status_code=422, detail="objective must be an object spec"
        )
    kind = spec.get("kind")
    if kind == "qed":
        return QedObjective()
    if kind == "sas":
        return SAScoreObjective(
            easy=_number(spec, "easy", "sas"),
            hard=_number(spec, "hard", "sas"),
        )
    if kind == "logp_band":
        return LogPBandObjective(
            lo=_number(spec, "lo", "logp_band"),
            hi=_number(spec, "hi", "logp_band"),
        )
    if kind == "mw_band":
        return MolecularWeightBandObjective(
            lo=_number(spec, "lo", "mw_band"),
            hi=_number(spec, "hi", "mw_band"),
        )
    if kind == "tpsa_band":
        return TpsaBandObjective(
            lo=_number(spec, "lo", "tpsa_band"),
            hi=_number(spec, "hi", "tpsa_band"),
        )
    if kind == "similarity":
        return SimilarityObjective(
            seed_smiles=_string(spec, "seed_smiles", "similarity"),
            radius=int(spec.get("radius", 2)),
            n_bits=int(spec.get("n_bits", 2048)),
        )
    if kind == "substructure":
        return SubstructureObjective(
            smarts=_string(spec, "smarts", "substructure")
        )
    if kind == "novelty":
        corpus = spec.get("corpus")
        if corpus is not None and not isinstance(corpus, list):
            raise HTTPException(
                status_code=422,
                detail="novelty corpus must be a list of SMILES",
            )
        return NoveltyObjective(
            corpus=corpus,
            radius=int(spec.get("radius", 2)),
            n_bits=int(spec.get("n_bits", 2048)),
        )
    if kind == "multi":
        nested = spec.get("objectives")
        if not isinstance(nested, list) or not nested:
            raise HTTPException(
                status_code=422,
                detail="multi objective needs a non-empty objectives list",
            )
        weights = spec.get("weights")
        if weights is not None and (
            not isinstance(weights, list)
            or len(weights) != len(nested)
        ):
            raise HTTPException(
                status_code=422,
                detail="weights must align with the objectives list",
            )
        return MultiObjective(
            [_build_objective(s) for s in nested], weights=weights
        )
    raise HTTPException(
        status_code=422,
        detail=(
            f"unknown objective kind {kind!r}; see "
            "GET /api/v1/generative/objectives"
        ),
    )


class PriorTrainRequest(BaseModel):
    smiles: list[str] | None = Field(
        default=None, max_length=_MAX_INLINE_SMILES
    )
    dataset_id: str | None = None
    column: str | None = None
    name: str = "smiles-prior"
    embedding_dim: int = Field(default=48, ge=8, le=256)
    hidden_dim: int = Field(default=96, ge=8, le=512)
    n_layers: int = Field(default=1, ge=1, le=8)
    epochs: int = Field(default=8, ge=1, le=200)
    batch_size: int = Field(default=32, ge=4, le=512)
    lr: float = Field(default=0.005, gt=0.0, le=1.0)
    grad_clip: float = Field(default=5.0, gt=0.0, le=100.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    mode: Literal["job", "sync"] = "job"


class SampleRequest(BaseModel):
    prior_id: str
    n: int = Field(default=10, ge=1, le=_MAX_SAMPLE)
    temperature: float = Field(default=1.0, gt=0.05, le=3.0)
    top_k: int | None = Field(default=None, ge=1, le=256)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    max_length: int = Field(default=120, ge=10, le=256)


class OptimizeRequest(BaseModel):
    objective: dict[str, Any]
    seed_smiles: list[str] | None = Field(
        default=None, max_length=200
    )
    dataset_id: str | None = None
    column: str | None = None
    population_size: int = Field(default=64, ge=4, le=512)
    generations: int = Field(default=20, ge=1, le=500)
    mutation_rate: float = Field(default=0.08, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    elitism: int = Field(default=4, ge=0, le=64)
    tournament_k: int = Field(default=3, ge=2, le=16)
    max_smiles_length: int = Field(default=120, ge=20, le=256)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    top_n: int = Field(default=10, ge=1, le=50)
    mode: Literal["job", "sync"] = "job"


def _resolve_corpus(
    request: Request,
    req: PriorTrainRequest | OptimizeRequest,
) -> tuple[list[str], dict[str, Any]]:
    """Inline smiles or a dataset SMILES column (validated, counted)."""
    inline = getattr(req, "smiles", None)
    if inline:
        corpus = [s for s in inline if isinstance(s, str) and s.strip()]
        if len(corpus) < 5:
            raise HTTPException(
                status_code=422,
                detail="a prior corpus needs at least 5 SMILES strings",
            )
        return corpus, {"source": "inline", "n": len(corpus)}
    if req.dataset_id is not None:
        dataset = request.app.state.dataset_store.get(req.dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=404, detail="Unknown dataset_id."
            )
        column = req.column or dataset.target
        if column not in dataset.frame.columns:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"column {column!r} not in dataset "
                    f"{req.dataset_id!r} "
                    f"(columns: {[str(c) for c in dataset.frame.columns]})"
                ),
            )
        series = dataset.frame[column].dropna()
        corpus = [str(v) for v in series.tolist()]
        if len(corpus) < 5:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"column {column!r} has only {len(corpus)} usable "
                    "rows; a prior corpus needs at least 5"
                ),
            )
        return corpus, {
            "source": "dataset",
            "dataset_id": req.dataset_id,
            "column": column,
            "n": len(corpus),
        }
    raise HTTPException(
        status_code=422,
        detail="give either smiles=[...] or dataset_id (+ column)",
    )


@router.get("/objectives")
def objectives(request: Request) -> dict:
    return {"objectives": _OBJECTIVE_CATALOG}


@router.get("/priors")
def priors(request: Request) -> dict:
    return {"priors": request.app.state.prior_store.list()}


@router.get("/priors/{prior_id}")
def prior_view(prior_id: str, request: Request) -> dict:
    record = request.app.state.prior_store.get(prior_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown prior_id.")
    return record.view()


def _prior_train_body(
    name: str,
    corpus: list[str],
    corpus_source: str,
    build_params: dict[str, Any],
    train_params: dict[str, Any],
    prior_store,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL SmilesRnn.train_prior + registration."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the prior build stage.")
    if manager is not None and job_id is not None:
        manager.report_progress(
            job_id, 5, f"tokenizing {len(corpus)} SMILES"
        )
    rnn = SmilesRnn(**build_params)
    if manager is not None and job_id is not None:
        manager.report_progress(
            job_id, 10, "training the prior"
        )
    losses = rnn.train_prior(corpus, **train_params)
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 90, "registering the prior")
    record = PriorRecord(
        prior_id=f"prior-{uuid.uuid4().hex[:12]}",
        model=rnn,
        name=name,
        params={**build_params, **train_params},
        n_train_smiles=len(corpus),
        losses=[float(x) for x in losses],
        corpus_source=corpus_source,
    )
    prior_store.register(record)
    # the loss curve is REAL telemetry: publish it as training.metrics
    if bus is not None:
        bus.publish(
            EventType.TRAINING_METRICS,
            {
                "stage": "prior trained",
                "epochs": len(losses),
                "losses": json_safe([float(x) for x in losses]),
                "final_loss": float(losses[-1]) if losses else None,
            },
            job_id=job_id,
        )
    return {"prior": record.view()}


@router.post("/rnn/train")
def train_prior(req: PriorTrainRequest, request: Request) -> dict:
    corpus, source = _resolve_corpus(request, req)
    build_params = {
        "embedding_dim": req.embedding_dim,
        "hidden_dim": req.hidden_dim,
        "n_layers": req.n_layers,
        "seed": req.seed,
    }
    train_params = {
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "lr": req.lr,
        "grad_clip": req.grad_clip,
    }
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_prior_train_body(
                req.name, corpus, json_safe(source), build_params,
                train_params, request.app.state.prior_store,
                bus=request.app.state.bus,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "generative.prior_train",
        "generative-ai",
        _prior_train_body,
        configuration={
            "name": req.name,
            "corpus": source,
            **build_params,
            **train_params,
        },
        name=req.name,
        corpus=corpus,
        corpus_source=json_safe(source),
        build_params=build_params,
        train_params=train_params,
        prior_store=request.app.state.prior_store,
        bus=request.app.state.bus,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/rnn/sample")
def sample(req: SampleRequest, request: Request) -> dict:
    record = request.app.state.prior_store.get(req.prior_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown prior_id.")
    samples = record.model.sample(
        req.n,
        temperature=req.temperature,
        seed=req.seed,
        max_length=req.max_length,
        top_k=req.top_k,
    )
    return {
        "prior_id": req.prior_id,
        "n_requested": req.n,
        "n_sampled": len(samples),
        "unique": len(set(samples)),
        "samples": samples,
        "params": {
            "temperature": req.temperature,
            "top_k": req.top_k,
            "seed": req.seed,
            "max_length": req.max_length,
        },
    }


def _optimize_body(
    objective_spec: dict[str, Any],
    seed_smiles: list[str] | None,
    corpus: list[str] | None,
    ga_kwargs: dict[str, Any],
    top_n: int,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL evolution with per-generation telemetry."""
    objective = _build_objective(objective_spec)
    optimizer = GeneticOptimizer(objective, **ga_kwargs)

    def progress_adapter(generation: int, generations: int,
                         stage: str) -> bool:
        # iteration telemetry on the bus + the job record
        if manager is not None and job_id is not None:
            pct = min(100, int(100.0 * generation / max(1, generations)))
            manager.report_progress(
                job_id, pct, f"{stage} {generation}/{generations}",
                event=EventType.ITERATION_UPDATE,
            )
        # returning False stops the evolution at its own checkpoint:
        # the backend raises JobCancelled itself (the real cancel path)
        return not (cancel_event is not None and cancel_event.is_set())

    try:
        result = optimizer.optimize(
            seed_smiles=seed_smiles, corpus=corpus,
            progress=progress_adapter,
        )
    except BackendJobCancelled as exc:  # the backend's own signal
        # translate to the JobManager's cancellation state machine
        raise CancelledJob(str(exc)) from exc
    ranked = sorted(
        zip(result.smiles, result.scores, result.breakdown),
        key=lambda t: t[1],
        reverse=True,
    )
    top = [
        {"smiles": s, "score": json_safe(sc),
         "breakdown": json_safe(br)}
        for s, sc, br in ranked[:top_n]
    ]
    return {
        "optimization": {
            "best": result.best,
            "best_score": json_safe(result.best_score),
            "generations": len(result.history),
            "history": json_safe(result.history),
            "metadata": json_safe(result.metadata),
            "top": top,
        }
    }


@router.post("/optimize")
def optimize(req: OptimizeRequest, request: Request) -> dict:
    # validate the objective spec up front (both modes): an unknown
    # kind is a client error, not an asynchronous job failure
    _build_objective(req.objective)
    seed_smiles = req.seed_smiles
    corpus: list[str] | None = None
    if req.dataset_id is not None:
        corpus, _source = _resolve_corpus(request, req)
    if seed_smiles is None and corpus is None:
        # no seeds at all: give the optimizer the corpus itself
        if corpus is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "give seed_smiles and/or dataset_id (+column); the "
                    "evolution needs at least one starting structure"
                ),
            )
        seed_smiles = list(corpus)
    ga_kwargs = {
        "population_size": req.population_size,
        "generations": req.generations,
        "mutation_rate": req.mutation_rate,
        "crossover_rate": req.crossover_rate,
        "elitism": req.elitism,
        "tournament_k": req.tournament_k,
        "max_smiles_length": req.max_smiles_length,
        "seed": req.seed,
    }
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_optimize_body(
                req.objective, seed_smiles, corpus, ga_kwargs, req.top_n,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "generative.optimize",
        "generative-ai",
        _optimize_body,
        configuration={
            "objective": req.objective,
            "n_seed_smiles": len(seed_smiles or []),
            "corpus": bool(corpus),
            **ga_kwargs,
        },
        objective_spec=req.objective,
        seed_smiles=seed_smiles,
        corpus=corpus,
        ga_kwargs=ga_kwargs,
        top_n=req.top_n,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


# -- the feature-space GAN / Diffusion lanes (part18 §5.6, the API leg) ------
#
# FEATURE-space generative models over a registered dataset's numeric
# feature matrix — the honest scope line rides every response: these
# are synthetic FEATURE ROWS, not molecules (the SMILES lanes above own
# molecular generation), and nothing sampled is experimentally
# validated data.  They register in the session prior store like the
# RNN prior (the tabular analogue of the same registry).

class GanTrainRequest(BaseModel):
    dataset_id: str
    name: str = "feature-gan"
    latent_dim: int = Field(default=16, ge=1, le=128)
    hidden_layers: list[int] = Field(default=[64], min_length=0, max_length=8)
    d_steps: int = Field(default=1, ge=1, le=8)
    epochs: int = Field(default=100, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    random_state: int = Field(default=42, ge=0, le=2**31 - 1)
    mode: Literal["job", "sync"] = "job"


class GanSampleRequest(BaseModel):
    prior_id: str
    n: int = Field(default=32, ge=1, le=500)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


def _gan_train_body(
    name: str,
    X,
    build_params: dict[str, Any],
    dataset_source: str,
    prior_store,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL FeatureSpaceGAN adversarial fit."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the GAN build stage.")
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 5, "building the adversarial pair")
    gan = FeatureSpaceGAN(**build_params)
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 10, "training the adversarial pair")
    gan.fit(X.to_numpy())
    record = FeaturePriorRecord(
        prior_id=f"prior-{uuid.uuid4().hex[:12]}",
        model=gan,
        name=name,
        kind="feature_gan",
        params=build_params,
        n_train_rows=len(X),
        losses=[h["g_loss"] for h in gan.history_],
        dataset_source=dataset_source,
    )
    prior_store.register(record)
    if bus is not None:
        bus.publish(
            EventType.TRAINING_METRICS,
            {
                "stage": "feature GAN trained",
                "epochs": len(gan.history_),
                "final_g_loss": gan.history_[-1]["g_loss"] if gan.history_ else None,
                "final_d_loss": gan.history_[-1]["d_loss"] if gan.history_ else None,
            },
            job_id=job_id,
        )
    view = record.view()
    view["final_d_loss"] = (
        gan.history_[-1]["d_loss"] if gan.history_ else None
    )
    view["note"] = (
        "feature-space generative model — synthetic feature rows, NOT "
        "molecules (the SMILES lanes own molecular generation)"
    )
    return {"prior": view}


@router.post("/gan/train")
def gan_train(req: GanTrainRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    X = dataset.X
    build_params = {
        "latent_dim": req.latent_dim,
        "hidden_layers": tuple(req.hidden_layers),
        "d_steps": req.d_steps,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "weight_decay": req.weight_decay,
        "random_state": req.random_state,
    }
    dataset_source = f"dataset:{req.dataset_id}"
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_gan_train_body(
                    req.name, X, build_params, dataset_source,
                    request.app.state.prior_store,
                    bus=request.app.state.bus,
                ),
            }
        except (DataValidationError, ModelError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    manager = request.app.state.job_manager
    record = manager.submit(
        "generative.gan_train",
        "generative-ai",
        _gan_train_body,
        configuration={
            "name": req.name,
            "dataset_id": req.dataset_id,
            **build_params,
        },
        name=req.name,
        X=X,
        build_params=build_params,
        dataset_source=dataset_source,
        prior_store=request.app.state.prior_store,
        bus=request.app.state.bus,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/gan/sample")
def gan_sample(req: GanSampleRequest, request: Request) -> dict:
    record = request.app.state.prior_store.get(req.prior_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown prior_id.")
    if not isinstance(record.model, FeatureSpaceGAN):
        raise HTTPException(
            status_code=422,
            detail=(
                f"prior {req.prior_id} is not a feature GAN; the GAN sampler "
                "needs one trained through POST /api/v1/generative/gan/train"
            ),
        )
    try:
        rows = record.model.sample(req.n, seed=req.seed)
    except ModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    report = record.model.quality_report()
    return {
        "prior_id": req.prior_id,
        "n_requested": req.n,
        "n_sampled": int(rows.shape[0]),
        "rows": json_safe(rows.tolist()),
        "quality": report,
        "params": {"seed": req.seed},
        "note": (
            "synthetic feature rows — a distributional parity report, not "
            "real data; feature-space generation, not molecules"
        ),
    }


class DiffusionTrainRequest(BaseModel):
    dataset_id: str
    name: str = "feature-diffusion"
    timesteps: int = Field(default=100, ge=2, le=1000)
    hidden_layers: list[int] = Field(default=[64], min_length=0, max_length=8)
    epochs: int = Field(default=120, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    random_state: int = Field(default=42, ge=0, le=2**31 - 1)
    mode: Literal["job", "sync"] = "job"


class DiffusionSampleRequest(BaseModel):
    prior_id: str
    n: int = Field(default=32, ge=1, le=500)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


def _diffusion_train_body(
    name: str,
    X,
    build_params: dict[str, Any],
    dataset_source: str,
    prior_store,
    bus=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL FeatureSpaceDiffusion DDPM fit."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the denoiser build stage.")
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 5, "building the denoiser")
    dpm = FeatureSpaceDiffusion(**build_params)
    if manager is not None and job_id is not None:
        manager.report_progress(job_id, 10, "training the noise predictor")
    dpm.fit(X.to_numpy())
    record = FeaturePriorRecord(
        prior_id=f"prior-{uuid.uuid4().hex[:12]}",
        model=dpm,
        name=name,
        kind="feature_diffusion",
        params=build_params,
        n_train_rows=len(X),
        losses=[h["train_loss"] for h in dpm.history_],
        dataset_source=dataset_source,
    )
    prior_store.register(record)
    if bus is not None:
        bus.publish(
            EventType.TRAINING_METRICS,
            {
                "stage": "feature diffusion trained",
                "epochs": len(dpm.history_),
                "final_train_loss": (
                    dpm.history_[-1]["train_loss"] if dpm.history_ else None
                ),
            },
            job_id=job_id,
        )
    view = record.view()
    view["note"] = (
        "feature-space generative model — synthetic feature rows, NOT "
        "molecules (the SMILES lanes own molecular generation)"
    )
    return {"prior": view}


@router.post("/diffusion/train")
def diffusion_train(req: DiffusionTrainRequest, request: Request) -> dict:
    store = request.app.state.dataset_store
    dataset = store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    X = dataset.X
    build_params = {
        "timesteps": req.timesteps,
        "hidden_layers": tuple(req.hidden_layers),
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "weight_decay": req.weight_decay,
        "random_state": req.random_state,
    }
    dataset_source = f"dataset:{req.dataset_id}"
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_diffusion_train_body(
                    req.name, X, build_params, dataset_source,
                    request.app.state.prior_store,
                    bus=request.app.state.bus,
                ),
            }
        except (DataValidationError, ModelError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    manager = request.app.state.job_manager
    record = manager.submit(
        "generative.diffusion_train",
        "generative-ai",
        _diffusion_train_body,
        configuration={
            "name": req.name,
            "dataset_id": req.dataset_id,
            **build_params,
        },
        name=req.name,
        X=X,
        build_params=build_params,
        dataset_source=dataset_source,
        prior_store=request.app.state.prior_store,
        bus=request.app.state.bus,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/diffusion/sample")
def diffusion_sample(req: DiffusionSampleRequest, request: Request) -> dict:
    record = request.app.state.prior_store.get(req.prior_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown prior_id.")
    if not isinstance(record.model, FeatureSpaceDiffusion):
        raise HTTPException(
            status_code=422,
            detail=(
                f"prior {req.prior_id} is not a feature diffusion model; the "
                "diffusion sampler needs one trained through POST "
                "/api/v1/generative/diffusion/train"
            ),
        )
    try:
        rows = record.model.sample(req.n, seed=req.seed)
    except ModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    report = record.model.quality_report()
    return {
        "prior_id": req.prior_id,
        "n_requested": req.n,
        "n_sampled": int(rows.shape[0]),
        "rows": json_safe(rows.tolist()),
        "quality": report,
        "params": {"seed": req.seed},
        "note": (
            "synthetic feature rows — a distributional parity report, not "
            "real data; feature-space generation, not molecules"
        ),
    }

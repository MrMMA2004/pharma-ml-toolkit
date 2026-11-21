"""``/api/v1/multiobjective`` — the multi-objective HPO lane over the
REAL in-tree backend (part13 Phase 5's LAST remainder item — the API
lane over :mod:`pharmaml.ml.multiobjective`).

* ``GET /api/v1/multiobjective/registry`` — the module's own
  vocabulary verbatim: the optuna minimize convention and the Pareto
  order (``dominates`` / ``pareto_indices`` — mutual non-domination is
  the definition, duplicates survive together, the honest
  ``O(n^2 * k)`` pairwise cost at study scale), the exact 2-D
  hypervolume indicator (the staircase sweep, the WFG
  reference-dominance contract, the report's nadir + 10 % margin),
  NSGA-II's crowding distance (boundary members ``+inf``), the MOEA/D
  Tchebycheff decomposition that turns BOTH existing samplers into
  front-chasers, the study's weight supply (Das & Dennis's
  simplex-lattice spread cycling corner-first, or seeded uniform
  simplex draws), the tunable-objective ledger (``score`` = the
  negated mean CV score, ``fit_seconds`` / ``score_seconds`` = the
  mean fold times straight off ``cross_validate``), the report's
  schema (the front in study order, the aligned crowding, the
  bi-objective-only indicator with the monotone non-decreasing
  hypervolume curve, ``best_for`` endpoints, the one-shot
  full-front-crowding ``prune``), and the honest scope statement:
  optuna/Ax/nevergrad's leaders reviewed, what transfers is the order
  theory + the indicator + the decomposition + the study arithmetic
  (pure NumPy); EHVI's GP over a *vector* objective, MOTPE's per-side
  Parzen estimators, NSGA-II's genetic operators and the exact
  hypervolume beyond two objectives stay upstream.
* ``POST /api/v1/multiobjective/optimize`` — the full honest round
  trip over the real backend: the route's own seeded SYNTHETIC dataset
  (80x6, the seed the only generative dial), the search space built
  from the request's dimension specs, and ``tune_model_multi``'s real
  cross-validated study over the sklearn registry model — the
  accuracy-vs-cost front, its crowding, its 2-D indicator + curve, the
  per-objective ``best_for`` endpoints, and the crowding-pruned
  shortlist.  Sync-only light compute (bounded trials over a small
  dense frame); every objective value is a statistic over the
  synthetic dataset — a model estimate, never an experimental fact.

The honest notes everywhere: no real dataset ships with this route;
the boundary members' ``+inf`` crowding reads as ``null`` in JSON (the
crowding note states the convention); a 3-objective study reports
``null`` hypervolume (the exact sweep is 2-D only), never a fabricated
indicator.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.hpo import FloatParam, IntParam, SearchSpace
from pharmaml.ml.multiobjective import _SUPPORTED_OBJECTIVES, tune_model_multi

router = APIRouter(prefix="/multiobjective", tags=["multiobjective"])

_MAX_SEED = 2147483647
_MAX_TRIALS = 60
_MAX_CV = 5
_MAX_DIMENSIONS = 4
_MAX_NAME_LEN = 64


class SpaceDimension(BaseModel):
    """One dimension of the search space, in the backend's own terms.

    ``uniform`` / ``loguniform`` build a ``FloatParam`` on
    ``[low, high]`` (the log sampling inside); ``randint`` builds an
    ``IntParam`` on the inclusive integer range.  The name must be one
    of the model's own hyperparameters — the backend's
    ``build_model`` raises its typed ``ModelError`` otherwise (a 422,
    never a 500).
    """

    name: str = Field(min_length=1, max_length=_MAX_NAME_LEN)
    kind: Literal["uniform", "loguniform", "randint"]
    low: float
    high: float


class MultiObjectiveOptimizeRequest(BaseModel):
    """One seeded synthetic study: dataset + space + model + objectives.

    The dataset is the route's own seeded 80x6 SYNTHETIC stand-in (the
    seed is the only generative dial); the space is the request's own
    dimension specs (1..4); the model is any sklearn registry name the
    backend itself validates; the objectives are 2..3 distinct entries
    from the backend's own ledger.  ``weight_mode`` /
    ``n_weight_vectors`` are the study's own spread dials, forwarded
    verbatim.  The default study (random_forest over n_estimators x
    max_depth) carries a real trade-off: accuracy climbs with depth
    while fit seconds climb with the tree count.
    """

    model: str = "random_forest"
    task: Literal["classification", "regression"] = "classification"
    objectives: list[str] = Field(default=["score", "fit_seconds"])
    dimensions: list[SpaceDimension] = Field(
        default_factory=lambda: [
            SpaceDimension(name="n_estimators", kind="randint", low=10, high=80),
            SpaceDimension(name="max_depth", kind="randint", low=2, high=10),
        ],
        min_length=1,
        max_length=_MAX_DIMENSIONS,
    )
    n_trials: int = Field(default=20, ge=1, le=_MAX_TRIALS)
    cv: int = Field(default=3, ge=2, le=_MAX_CV)
    sampler: Literal["random", "tpe"] = "tpe"
    seed: int = Field(default=42, ge=0, le=_MAX_SEED)
    weight_mode: Literal["spread", "random"] = "spread"
    n_weight_vectors: int = Field(default=8, ge=2, le=32)
    shortlist_keep: int = Field(default=5, ge=1, le=10)


def _provenance_block() -> dict[str, Any]:
    """The provenance + limitations block (the honest-scope state)."""
    return {
        "backend": "pharmaml.ml.multiobjective",
        "source": (
            "the best-of-ml-python HPO/AutoML leaders reviewed file by "
            "file — optuna/optuna (the multi-objective study: vector "
            "values, the ParetoFront pruner, MOTPE, the seeded uniform "
            "simplex weight draws); facebook/Ax + BoTorch (the expected "
            "hypervolume improvement over a GP); facebookresearch/"
            "nevergrad (the portfolio of gradient-free optimizers per "
            "objective)"
        ),
        "n_repositories": 3,
        "transfers_exactly": (
            "the Pareto order (dominates/pareto_indices), the exact 2-D "
            "hypervolume (the staircase sweep), NSGA-II's crowding "
            "distance, the MOEA/D Tchebycheff decomposition, and the "
            "study arithmetic that drives BOTH existing samplers "
            "(random + TPE) as front-chasers — pure NumPy"
        ),
        "stays_upstream": (
            "EHVI needs a GP posterior over a vector objective (this "
            "house's GP is scalar); MOTPE's per-side multivariate Parzen "
            "estators and NSGA-II's genetic operators stay upstream (the "
            "generative GA carries the latter for molecules); the exact "
            "hypervolume beyond two objectives (WFG's sweep) is not "
            "vendored — the 2-D sweep is the indicator reported"
        ),
        "shipped_checkpoints": 0,
        "shipped_checkpoints_note": (
            "no pretrained anything ships — the study runs in-tree over "
            "the seeded synthetic dataset on every request; there is no "
            "learned-model surface to checkpoint beyond the trained "
            "models the ML workspace itself registers"
        ),
        "dataset": (
            "the served dataset is the route's own SYNTHETIC seeded "
            "stand-in (80 rows x 6 features, default_rng(seed)) — the "
            "demo frame a real pharma dataset would stand in for; no "
            "real dataset ships with this route"
        ),
        "disclaimer": (
            "every objective value is a cross-validated statistic over "
            "the synthetic dataset — front members are model estimates "
            "for triage, never experimental facts and never a "
            "deployment recommendation"
        ),
    }


def _finite(value: float) -> float | None:
    """JSON-safe float (NaN/inf read as null, never as lies)."""
    number = float(value)
    return number if math.isfinite(number) else None


def _synthetic_dataset(seed: int, task: str) -> tuple[np.ndarray, np.ndarray]:
    """The route's own seeded 80x6 stand-in (the seed the only dial)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 6))
    if task == "classification":
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:
        y = 1.5 * X[:, 0] + 0.1 * rng.normal(size=80)
    return X, y


def _build_space(dimensions: list[SpaceDimension]) -> SearchSpace:
    """Translate the request's dimension specs into the backend's space.

    The honest failure modes raise the backend's own typed
    ``DataValidationError`` (a 422): a non-strictly-increasing range, a
    non-positive loguniform bound, and non-integral randint bounds.
    """
    specs = []
    for dim in dimensions:
        if not dim.low < dim.high:
            raise DataValidationError(
                f"dimension {dim.name!r} needs low < high, got "
                f"{dim.low} !< {dim.high}"
            )
        if dim.kind == "randint":
            if dim.low != int(dim.low) or dim.high != int(dim.high):
                raise DataValidationError(
                    f"dimension {dim.name!r} needs integral randint "
                    f"bounds, got {dim.low}..{dim.high}"
                )
            specs.append(IntParam(dim.name, int(dim.low), int(dim.high)))
        else:
            if dim.kind == "loguniform" and dim.low <= 0.0:
                raise DataValidationError(
                    f"dimension {dim.name!r} needs a positive loguniform "
                    f"lower bound, got {dim.low}"
                )
            specs.append(
                FloatParam(dim.name, dim.low, dim.high, log=dim.kind == "loguniform")
            )
    return SearchSpace(*specs)


@router.get("/registry")
def registry(request: Request) -> dict:
    return {
        "order": {
            "convention": (
                "the optuna convention — every objective is MINIMIZED; "
                "'score' rides the negated mean CV score"
            ),
            "dominates": (
                "a dominates b iff a is at least as good on every "
                "objective and strictly better on at least one"
            ),
            "pareto_indices": (
                "the non-dominated set, ascending input order — "
                "objective-space duplicates survive together (mutual "
                "non-domination is the definition)"
            ),
            "complexity": "pairwise O(n^2 * k) at study scale — the honest cost",
        },
        "indicator": {
            "family": "the exact 2-D dominated hypervolume (the staircase sweep)",
            "reference_contract": (
                "the WFG contract — the reference point must dominate "
                "every point; a non-dominating reference raises, never "
                "silently clips"
            ),
            "report_reference": (
                "the observed nadir padded by 10 % of the "
                "ideal-to-nadir range per axis (the documented margin)"
            ),
            "curve": (
                "the per-trial indicator under that same FIXED "
                "reference — monotonically non-decreasing by "
                "construction (a growing point set grows the union)"
            ),
            "two_d_only": (
                "the exact sweep is 2-D only — a 3-objective study "
                "reports null, never a fabricated indicator"
            ),
        },
        "spread": {
            "family": "NSGA-II's crowding distance (boundaries +inf)",
            "rule": (
                "the two extremes of every objective gain +inf and "
                "each interior point gains the normalised neighbour gap; "
                "a degenerate objective (zero span) contributes nothing"
            ),
            "json_note": (
                "boundary members' +inf reads as null in JSON — the "
                "crowding note on every answer states the convention"
            ),
        },
        "decomposition": {
            "family": "the MOEA/D Tchebycheff decomposition",
            "form": "max_j w_j |v_j - ideal_j|, minimized",
            "rule": (
                "a zero weight is the corner subproblem that optimizes "
                "its objective alone (the front's endpoint); at least "
                "one weight must be positive, negative weights are "
                "meaningless"
            ),
        },
        "study": {
            "samplers": ["random", "tpe"],
            "samplers_note": (
                "the decomposition drives BOTH existing samplers as "
                "front-chasers — each trial's Tchebycheff distance to "
                "the running ideal point is the single number the "
                "underlying engine is told"
            ),
            "weight_modes": {
                "spread": (
                    "Das & Dennis's simplex-lattice weight set, "
                    "deterministic spread, corner-heavy prefix, cycled"
                ),
                "random": (
                    "seeded uniform simplex draws — exponential "
                    "variates normalised to sum 1 (the Dirichlet(1,...,1) "
                    "shortcut, Optuna's own approach)"
                ),
            },
            "n_objectives": "a Pareto study needs at least two objectives",
            "drift_note": (
                "the ideal point advances as the front does, so the "
                "scalar landscape is not strictly stationary across "
                "trials — the hypervolume curve is the convergence "
                "check that the drift still descends"
            ),
        },
        "tunable_objectives": {
            "ledger": {
                label: {
                    "cross_validate_key": key,
                    "sign": sign,
                }
                for label, (key, sign) in _SUPPORTED_OBJECTIVES.items()
            },
            "note": (
                "'score' = the negated mean CV score (bigger is better "
                "-> minimize the negation); 'fit_seconds' / "
                "'score_seconds' = the mean fold times — the same "
                "numbers pharmaml.ml.cv_results records"
            ),
            "bounds": "2..3 distinct entries per request",
        },
        "report": {
            "schema": [
                "objective_names", "front (the non-dominated MOTrial "
                "members in study order)", "crowding (aligned with the "
                "front)", "ref", "hypervolume", "hypervolume_curve",
                "n_trials",
            ],
            "best_for": (
                "the front member minimizing one named objective (its "
                "endpoint); an unknown name or an empty front raises"
            ),
            "prune": (
                "up to keep front members by full-front crowding, "
                "largest first, ties in front order — a one-shot "
                "selection; the shortlist never claims the full study's "
                "hypervolume as its own"
            ),
        },
        "request": {
            "defaults": {
                "model": "random_forest",
                "task": "classification",
                "objectives": ["score", "fit_seconds"],
                "dimensions": [
                    {"name": "n_estimators", "kind": "randint", "low": 10, "high": 80},
                    {"name": "max_depth", "kind": "randint", "low": 2, "high": 10},
                ],
                "n_trials": 20,
                "cv": 3,
                "sampler": "tpe",
                "seed": 42,
                "weight_mode": "spread",
                "n_weight_vectors": 8,
                "shortlist_keep": 5,
            },
            "bounds": {
                "n_trials": f"1..{_MAX_TRIALS}",
                "cv": f"2..{_MAX_CV}",
                "dimensions": f"1..{_MAX_DIMENSIONS}",
                "n_weight_vectors": "2..32",
                "shortlist_keep": "1..10",
                "seed": f"0..{_MAX_SEED}",
            },
            "dimension_kinds": ["uniform", "loguniform", "randint"],
            "model_note": (
                "the model is any sklearn registry name — the backend's "
                "own build_model validates it (unknown names and "
                "unknown hyperparameter names raise its typed ModelError, "
                "a 422, never a 500)"
            ),
        },
        "dataset": {
            "geometry": {"n_samples": 80, "n_features": 6},
            "label": (
                "SYNTHETIC — the route's own seeded stand-in "
                "(default_rng(seed)); no real dataset ships"
            ),
        },
        "provenance": _provenance_block(),
        "optimize_url": "/api/v1/multiobjective/optimize",
        "note": (
            "sync-only light compute — bounded trials over a small dense "
            "synthetic frame; the seed fixes the dataset and (under the "
            "random sampler) the parameterization sequence, while the "
            "fit/score-seconds objectives are wall-clock fold timings "
            "that vary run-to-run and the TPE sampler conditions on the "
            "observed scalars — the honest drift, stated; every "
            "objective value is a cross-validated statistic over the "
            "synthetic dataset"
        ),
    }


@router.post("/optimize")
def optimize(req: MultiObjectiveOptimizeRequest, request: Request) -> dict:
    # the real backend round trip: the seeded synthetic dataset -> the
    # request's own search space -> the real cross-validated study ->
    # the front + crowding + the 2-D indicator + the endpoints + the
    # crowding-pruned shortlist
    space = _build_space(req.dimensions)
    X, y = _synthetic_dataset(req.seed, req.task)

    report = tune_model_multi(
        req.model,
        space,
        X,
        y,
        task=req.task,
        cv=req.cv,
        n_trials=req.n_trials,
        objectives=req.objectives,
        sampler=req.sampler,
        seed=req.seed,
        weight_mode=req.weight_mode,
        n_weight_vectors=req.n_weight_vectors,
    )
    shortlist = report.prune(req.shortlist_keep)
    endpoints = {
        name: report.best_for(name) for name in report.objective_names
    }

    def trial_row(trial: Any, crowding: float) -> dict[str, Any]:
        return {
            "number": int(trial.number),
            "params": {k: v for k, v in trial.params.items()},
            "values": [float(v) for v in trial.values],
            "weights": [float(w) for w in trial.weights],
            "scalar": _finite(trial.scalar),
            "crowding": _finite(crowding),
            "crowding_is_boundary": not math.isfinite(crowding),
        }

    def endpoint_row(trial: Any) -> dict[str, Any]:
        # the trade-off corners carry no crowding claim — that metric
        # belongs to the front rows only
        return {
            "number": int(trial.number),
            "params": {k: v for k, v in trial.params.items()},
            "values": [float(v) for v in trial.values],
            "weights": [float(w) for w in trial.weights],
            "scalar": _finite(trial.scalar),
        }

    front_rows = [
        trial_row(trial, crowd)
        for trial, crowd in zip(report.trials, report.crowding)
    ]
    shortlist_rows = [
        trial_row(trial, crowd)
        for trial, crowd in zip(shortlist.trials, shortlist.crowding)
    ]
    hypervolume_curve = (
        [_finite(v) for v in report.hypervolume_curve]
        if report.hypervolume_curve
        else []
    )

    return {
        "request": {
            "model": req.model,
            "task": req.task,
            "objectives": list(req.objectives),
            "dimensions": [
                {"name": d.name, "kind": d.kind, "low": d.low, "high": d.high}
                for d in req.dimensions
            ],
            "n_trials": req.n_trials,
            "cv": req.cv,
            "sampler": req.sampler,
            "seed": req.seed,
            "weight_mode": req.weight_mode,
            "n_weight_vectors": req.n_weight_vectors,
            "shortlist_keep": req.shortlist_keep,
        },
        "dataset": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "task": req.task,
            "label": "SYNTHETIC — the route's own seeded stand-in, not a real dataset",
        },
        "study": {
            "n_trials": int(report.n_trials),
            "n_objectives": len(report.objective_names),
            "objective_names": list(report.objective_names),
            "sampler": req.sampler,
            "weight_mode": req.weight_mode,
            "n_weight_vectors": req.n_weight_vectors,
            "n_front": len(report.trials),
            "objectives_note": (
                "every objective is minimized — 'score' is the negated "
                "mean CV score, so a lower score value means a MORE "
                "accurate model"
            ),
        },
        "front": front_rows,
        "crowding_note": (
            "NSGA-II crowding aligned with the front; boundary members "
            "carry +inf and read as null here (crowding_is_boundary "
            "marks them) — null is never a zero spread"
        ),
        "indicator": {
            "ref": [float(v) for v in report.ref] if report.ref else None,
            "hypervolume": _finite(report.hypervolume)
            if report.hypervolume is not None
            else None,
            "hypervolume_curve": hypervolume_curve,
            "two_d_only": (
                "the exact sweep is 2-D only — null here means a "
                "3-objective study, never a fabricated indicator"
            ),
            "curve_note": (
                "the per-trial indicator under the FIXED reference — "
                "monotonically non-decreasing; the convergence check "
                "for the drifting ideal point"
            ),
        },
        "best_for": {
            name: endpoint_row(trial) for name, trial in endpoints.items()
        },
        "best_for_note": (
            "the front endpoint minimizing each named objective — the "
            "trade-off corners of the accuracy-vs-cost menu"
        ),
        "shortlist": {
            "keep": req.shortlist_keep,
            "n_front": int(shortlist.n_front),
            "n_trials": int(shortlist.n_trials),
            "trials": shortlist_rows,
            "note": (
                "selected by full-front crowding, largest first — a "
                "one-shot selection that never claims the full study's "
                "hypervolume as its own"
            ),
        },
        "provenance": _provenance_block(),
        "timings_note": (
            "the seed fixes the dataset and (under the random sampler) "
            "the parameterization sequence; the fit/score-seconds "
            "columns are wall-clock fold timings that vary run-to-run "
            "and the TPE sampler conditions on the observed scalars — "
            "the honest drift, stated where the timings ride"
        ),
        "note": (
            "every objective value is a cross-validated statistic over "
            "the synthetic dataset — front members are model estimates "
            "for triage, never experimental facts; no real dataset ships "
            "with this route"
        ),
    }

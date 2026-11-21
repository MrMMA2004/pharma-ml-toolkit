"""``/api/v1/drugresponse`` — the cancer cell-line drug-response lane
over the REAL in-tree backend (part13 Phase 5's next remainder item —
the drug-response API lane over :mod:`pharmaml.ml.drugresponse`).

* ``GET /api/v1/drugresponse/registry`` — the module's own vocabulary
  verbatim: the CaDRReS bilinear factorisation with its sigmoid link
  (the gamma layer ``W`` over cell-line features, per-drug weight
  vectors ``w_d``, score = sigmoid of the bilinear form), the fit
  hyperparameters with the route's sync-light caps, the five-key
  metric contract (:func:`pharmaml.ml.drugresponse.response_metrics`
  — "the four numbers every response paper reports" plus the surviving
  pair count ``n``), the sensitivity-ranking schema (``cell_line,
  drug, predicted_score, rank`` — most-responsive first), the
  per-drug biomarker readout (:meth:`DrugResponseModel.explain_drug`
  — the composed ``W w_d`` logit weights, sorted |weight| descending),
  the cross-cell-line validation contract (folds split *cell lines*,
  never (cell line, drug) pairs), and the honest provenance block: no
  pretrained checkpoints ship, no real CCLE/GDSC/CTRP/DepMap table
  ships, and the served screen is the route's own SYNTHETIC seeded
  demo.
* ``POST /api/v1/drugresponse/analyze`` — the full honest round trip
  over the real backend: the seeded synthetic screen, the CaDRReS fit
  (full-batch gradient descent over the observed pairs only), the
  predicted response-score matrix, the metric contract on the
  observed entries, the per-cell sensitivity ranking, the per-drug
  biomarker weights, and the cross-cell-line validation folds with
  their across-fold summary.  Sync-only light compute (tiny dense
  matrices, bounded epochs); every predicted value is labelled a
  model estimate over the synthetic screen — one response scale only
  (the unit sigmoid scale, §11.4's own harmonisation rule), never a
  clinical recommendation.

The honest notes everywhere: the rankings and biomarker weights are
triage hypotheses on a synthetic stand-in screen; no real cell line,
expression panel, mutation panel or drug-response table ships with
this route.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pharmaml.ml.drugresponse import (
    DrugResponseModel,
    cross_validate_response,
    response_metrics,
)

router = APIRouter(prefix="/drugresponse", tags=["drugresponse"])

_MAX_CELLS = 40
_MAX_DRUGS = 30
_MAX_FEATURES = 20
_MAX_FACTORS = 10
_MAX_EPOCHS = 1000
_MAX_TOP_K = 10
_MAX_PROFILES = 8
_MAX_FOLDS = 5


def _seeded_screen(
    n_cells: int,
    n_drugs: int,
    n_features: int,
    noise: float,
    observed: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The parameterised mirror of the Studio ResponseAI demo screen.

    A seeded ``(cell line x gene feature)`` matrix with a per-cell
    latent factor, ``n_drugs`` drugs with graded potency, responses
    generated through a sigmoid of the potency-weighted latent, then
    a NaN observation mask — the CaDRReS data contract in miniature
    (NaN = the pair was never screened).  Deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0.0, 1.0, (n_cells, n_features)),
        index=[f"CELL_{i:02d}" for i in range(n_cells)],
        columns=[f"gene_{j:02d}" for j in range(n_features)],
    )
    true_factor = rng.normal(0.0, 1.0, n_cells)
    drugs = [f"DRUG_{k:02d}" for k in range(n_drugs)]
    potency = np.linspace(0.9, 0.3, n_drugs) if n_drugs > 1 else np.array([0.9])
    table = pd.DataFrame(
        {
            d: 1.0
            / (
                1.0
                + np.exp(-(potency[k] * true_factor + rng.normal(0.0, noise, n_cells)))
            )
            for k, d in enumerate(drugs)
        },
        index=X.index,
    )
    mask = rng.random(table.shape) < (1.0 - observed)
    return X, table.mask(mask)


class DrugResponseAnalyzeRequest(BaseModel):
    """One seeded synthetic screen + fit + ranking request.

    The geometry bounds mirror the backend's own rules exactly
    (``>= 4`` cells, ``>= 2`` drugs, ``>= 2`` features, ``2 <=
    n_folds <= n_cells`` — the last one enforced by the backend
    itself as its own 422); the remaining caps are this route's
    sync-light budget, stated in the registry.
    """

    n_cells: int = Field(default=12, ge=4, le=_MAX_CELLS)
    n_drugs: int = Field(default=4, ge=2, le=_MAX_DRUGS)
    n_features: int = Field(default=8, ge=2, le=_MAX_FEATURES)
    n_factors: int = Field(default=3, ge=1, le=_MAX_FACTORS)
    lr: float = Field(default=0.1, ge=0.001, le=0.5)
    epochs: int = Field(default=200, ge=1, le=_MAX_EPOCHS)
    l2: float = Field(default=1e-3, ge=0.0, le=1.0)
    noise: float = Field(default=0.05, ge=0.0, le=0.5)
    observed: float = Field(default=0.85, ge=0.5, le=1.0)
    seed: int = Field(default=42, ge=0, le=2147483647)
    top_k: int = Field(default=5, ge=1, le=_MAX_TOP_K)
    profiles: int = Field(default=3, ge=1, le=_MAX_PROFILES)
    n_folds: int = Field(default=3, ge=2, le=_MAX_FOLDS)


def _provenance_block() -> dict[str, Any]:
    """The provenance + limitations block (the §11.1-honest state)."""
    return {
        "backend": "pharmaml.ml.drugresponse",
        "source": (
            "CSOgroup/CaDRReS — the bilinear drug-response model with "
            "its sigmoid link absorbed in plain NumPy, training "
            "included (the gamma layer W over cell-line features, "
            "per-drug weight vectors w_d, full-batch gradient descent "
            "over the observed pairs only); SmritiChawla/Precily — the "
            "multi-omics block-fusion discipline (MultiOmicsFusion, "
            "real in-tree); burakcan-izmirli/DeepResponse — the "
            "variational companion (ResponseVAE, real in-tree)"
        ),
        "n_repositories": 3,
        "shipped_checkpoints": 0,
        "shipped_checkpoints_note": (
            "no pretrained checkpoints ship — DeepCDR's multi-omics "
            "CNN and the external VAE weights stay external; this "
            "lane ships the CaDRReS-style model that trains in-tree "
            "over the seeded screen, never a pretrained predictor"
        ),
        "screen": (
            "the served screen is the route's own SYNTHETIC seeded "
            "demo (a parameterised mirror of the Studio ResponseAI "
            "page's 12x4 demo: a latent-factor sigmoid with graded "
            "drug potency and a NaN observation mask) — the stand-in "
            "for a CCLE/GDSC-sized table; no real CCLE, GDSC, CTRP or "
            "DepMap expression/mutation panel ships with the toolkit"
        ),
        "disclaimer": (
            "predicted response scores are model estimates over the "
            "synthetic screen — sensitivity rankings and biomarker "
            "weights are triage hypotheses, never clinical "
            "recommendations"
        ),
    }


def _finite(value: float) -> float | None:
    """JSON-safe float (NaN/inf metrics read as null, never as lies)."""
    number = float(value)
    return number if math.isfinite(number) else None


@router.get("/registry")
def registry(request: Request) -> dict:
    return {
        "model": {
            "family": "CaDRReS-style bilinear factorisation with a sigmoid link",
            "formulation": (
                "gamma = X.W (cell-line features + a bias row -> K "
                "latent gamma factors, linear); each drug owns a "
                "weight vector w_d over the latent layer (factors + a "
                "bias row); the predicted response score is the "
                "sigmoid of the bilinear form — training minimises "
                "squared error over the observed (cell, drug) pairs "
                "only, NaN entries carry no gradient"
            ),
            "direction": (
                "the score is the sigmoid-linked response — higher = "
                "stronger predicted response (the seeded screen's "
                "potency-driven convention); the sensitivity ranking "
                "is most-responsive first: highest predicted score = "
                "rank 1"
            ),
            "telemetry_fields": [
                "train_losses",
                "final_loss",
                "n_observed",
                "pearson_r",
                "n_epochs",
            ],
            "history_note": (
                "the trainer has no live per-epoch callback — the "
                "loss history lands after the fit; only the first and "
                "final loss ride this answer, never a fabricated epoch "
                "stream"
            ),
        },
        "hyperparameters": {
            "n_factors": {"default": 3, "max": _MAX_FACTORS},
            "lr": {"default": 0.1, "min": 0.001, "max": 0.5},
            "epochs": {"default": 200, "max": _MAX_EPOCHS},
            "l2": {"default": 1e-3, "max": 1.0},
            "seed": {"default": 42, "max": 2147483647},
            "note": (
                "the backend's own defaults are n_factors=10 / "
                "lr=0.1 / epochs=200 / l2=1e-3 (CaDRReS's f=10 runs); "
                "the route's defaults are the small-screen set and "
                "the caps are the sync-light budget"
            ),
        },
        "screen": {
            "defaults": {
                "n_cells": 12,
                "n_drugs": 4,
                "n_features": 8,
                "noise": 0.05,
                "observed": 0.85,
                "seed": 42,
            },
            "bounds": {
                "n_cells": f"4..{_MAX_CELLS}",
                "n_drugs": f"2..{_MAX_DRUGS}",
                "n_features": f"2..{_MAX_FEATURES}",
                "observed": "0.5..1.0",
            },
            "label": (
                "SYNTHETIC — a seeded response screen (a "
                "potency-weighted latent-factor sigmoid + noise + a "
                "NaN observation mask), the demo/test stand-in for a "
                "CCLE/GDSC-sized table, not real pharmacology"
            ),
            "scale": (
                "one response scale only — the unit sigmoid scale; "
                "incompatible response scales are never mixed (the "
                "part13 harmonisation rule: no raw IC50 or AUC "
                "entries ride alongside the sigmoid scores)"
            ),
        },
        "sensitivity_ranking": {
            "schema": ["cell_line", "drug", "predicted_score", "rank"],
            "ordering": (
                "most-responsive first — highest predicted response "
                "score = rank 1 (the screen's sigmoid convention: "
                "higher = stronger response)"
            ),
            "source": (
                "the model's whole drug axis scored per cell profile "
                "(the DrugResponseModel.predict matrix, top-k per "
                "profile)"
            ),
            "response_limits": {
                "profiles": f"1..{_MAX_PROFILES}",
                "top_k_per_profile": f"1..{_MAX_TOP_K}",
            },
        },
        "biomarkers": {
            "schema": ["feature", "weight"],
            "ordering": (
                "sorted by |weight| descending then feature name "
                "ascending (deterministic)"
            ),
            "source": (
                "DrugResponseModel.explain_drug — the composed "
                "bilinear readout W w_d: each weight is that "
                "feature's contribution to the drug's score logit"
            ),
            "response_limits": {
                "top_k_per_drug": f"1..{_MAX_TOP_K}",
            },
        },
        "cross_validation": {
            "n_folds": f"2..{_MAX_FOLDS}",
            "rule": (
                "folds split cell lines, never (cell line, drug) "
                "pairs — the leak the grouped design exists to "
                "prevent; n_folds must not exceed the cell count (the "
                "backend's own 422)"
            ),
            "summary": (
                "the across-fold means of the metric contract "
                "(pearson, spearman, rmse, mae, n)"
            ),
        },
        "metrics": {
            "keys": ["pearson", "spearman", "rmse", "mae", "n"],
            "note": (
                "the four numbers every response paper reports, plus "
                "the surviving pair count n — response_metrics on the "
                "observed entries; non-finite metrics read as null, "
                "never as zero"
            ),
        },
        "provenance": _provenance_block(),
        "analyze_url": "/api/v1/drugresponse/analyze",
        "note": (
            "sync-only light compute — dense bounded matrices; every "
            "predicted value is a model estimate over the synthetic "
            "screen, and the geometry bounds mirror the backend's own "
            "rules (4/2/2 minimums, 2 <= n_folds <= n_cells)"
        ),
    }


@router.post("/analyze")
def analyze(req: DrugResponseAnalyzeRequest, request: Request) -> dict:
    # the real backend round trip: seeded screen -> CaDRReS fit ->
    # predicted response matrix -> metric contract -> sensitivity
    # ranking -> biomarker weights -> cross-cell-line validation
    X, table = _seeded_screen(
        req.n_cells,
        req.n_drugs,
        req.n_features,
        req.noise,
        req.observed,
        req.seed,
    )
    model = DrugResponseModel(
        n_factors=req.n_factors,
        lr=req.lr,
        epochs=req.epochs,
        l2=req.l2,
        seed=req.seed,
    )
    history = model.fit(X, table)
    predicted = model.predict(X)

    # the metric contract on the observed entries only
    mask = table.notna()
    true_values = table.to_numpy()[mask.to_numpy()]
    pred_values = predicted.to_numpy()[mask.to_numpy()]
    metrics = response_metrics(true_values, pred_values)

    # the per-cell sensitivity ranking (most-responsive first)
    per_cell: dict[str, list[dict[str, Any]]] = {}
    ranked_rows = 0
    for cell in list(predicted.index)[: req.profiles]:
        scores = predicted.loc[cell]
        ordered = scores.sort_values(ascending=False)
        per_cell[str(cell)] = [
            {
                "drug": str(drug),
                "predicted_score": float(score),
                "rank": rank,
            }
            for rank, (drug, score) in enumerate(ordered.items(), start=1)
        ][: req.top_k]
        ranked_rows += len(scores)

    # the per-drug biomarker weights (the composed W w_d readout)
    per_drug: dict[str, list[dict[str, Any]]] = {}
    for drug in model.drug_names:
        report = model.explain_drug(drug)
        per_drug[str(drug)] = [
            {"feature": str(row.feature), "weight": float(row.weight)}
            for row in report.head(req.top_k).itertuples()
        ]

    # the cross-cell-line validation folds
    cv_report = cross_validate_response(
        lambda: DrugResponseModel(
            n_factors=req.n_factors,
            lr=req.lr,
            epochs=req.epochs,
            l2=req.l2,
            seed=req.seed,
        ),
        X,
        table,
        n_folds=req.n_folds,
        seed=req.seed,
    )
    folds = [
        {
            "fold": int(fold.fold),
            "n_test_cells": int(fold.n_test_cells),
            "n_observed": int(fold.n_observed),
            "metrics": {
                key: _finite(value) for key, value in fold.metrics.items()
            },
        }
        for fold in cv_report.folds
    ]
    summary = {key: _finite(value) for key, value in cv_report.summary.items()}

    observed_entries = int(mask.to_numpy().sum())
    total_entries = int(mask.to_numpy().size)
    first_loss = history.train_losses[0] if history.train_losses else None

    return {
        "request": {
            "n_cells": req.n_cells,
            "n_drugs": req.n_drugs,
            "n_features": req.n_features,
            "n_factors": req.n_factors,
            "lr": req.lr,
            "epochs": req.epochs,
            "l2": req.l2,
            "noise": req.noise,
            "observed": req.observed,
            "seed": req.seed,
            "top_k": req.top_k,
            "profiles": req.profiles,
            "n_folds": req.n_folds,
        },
        "screen": {
            "cells": list(table.index),
            "drugs": list(table.columns),
            "features": list(X.columns),
            "n_cells": req.n_cells,
            "n_drugs": req.n_drugs,
            "n_features": req.n_features,
            "n_observed": observed_entries,
            "n_entries": total_entries,
            "observed_fraction": (
                observed_entries / total_entries if total_entries else 0.0
            ),
            "label": "SYNTHETIC seeded screen — not real pharmacology",
        },
        "training": {
            "epochs_run": int(history.n_epochs),
            "max_epochs": req.epochs,
            "final_loss": _finite(history.final_loss),
            "first_loss": _finite(first_loss) if first_loss is not None else None,
            "n_observed": int(history.n_observed),
            "pearson_r": _finite(history.pearson_r),
            "n_factors": model.n_factors,
            "note": (
                "the trainer has no live per-epoch callback — the "
                "loss history lands after the fit; only the first and "
                "final loss ride this answer, never a fabricated "
                "epoch stream"
            ),
        },
        "metrics": {
            key: _finite(value) for key, value in metrics.items()
        },
        "ranking": {
            "n_profiles": len(per_cell),
            "n_ranked_rows_total": ranked_rows,
            "top_k": req.top_k,
            "per_cell": per_cell,
            "ordering": (
                "highest predicted response score = rank 1 "
                "(most responsive first)"
            ),
        },
        "biomarkers": {
            "n_drugs": len(per_drug),
            "top_k": req.top_k,
            "per_drug": per_drug,
            "ordering": "sorted by |weight| descending then feature name",
        },
        "cross_validation": {
            "n_folds": req.n_folds,
            "folds": folds,
            "summary": summary,
            "note": (
                "folds split cell lines, never (cell line, drug) "
                "pairs; the summary is the across-fold mean of the "
                "metric contract"
            ),
        },
        "provenance": _provenance_block(),
        "note": (
            "predicted response scores are model estimates over the "
            "synthetic screen — sensitivity rankings and biomarker "
            "weights are triage hypotheses, never clinical "
            "recommendations; real screens (CCLE/GDSC/CTRP/DepMap) "
            "are not shipped with the toolkit"
        ),
    }

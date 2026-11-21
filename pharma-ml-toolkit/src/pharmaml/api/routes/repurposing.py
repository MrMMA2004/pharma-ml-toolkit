"""``/api/v1/repurposing`` — the drug-repurposing lane over the REAL
in-tree backend (part13 Phase 5's next remainder item — the repurposing
API lane over :mod:`pharmaml.ml.response_model`).

* ``GET /api/v1/repurposing/registry`` — the module's own vocabulary
  verbatim: DeepResponse's split ladder (``random``,
  ``cell_stratified``, ``drug_stratified``, ``drug_cell_stratified`` —
  plus the two-table ``cross_domain`` entry point), the CaDRReS
  recommender's formulation (``S = mu + b_p + b_q + P.Q^T`` with the
  cell half driven by features), the fit hyperparameters with the
  route's sync-light caps, the ten-key metric contract
  (:func:`pharmaml.ml.response_model.response_metrics`), the
  repurposing table schema (``cell_line, drug, predicted_ic50,
  rank`` — most-sensitive first), and the honest provenance block: the
  SELFormer/CNN/FiLM PyTorch hybrid is NOT shipped; the screen the
  route serves is the backend's own SYNTHETIC seeded demo, never a
  GDSC/CCLE table.
* ``POST /api/v1/repurposing/screen`` — the full honest round trip over
  the real backend: the seeded synthetic screen, the CaDRReS fit
  (full-batch gradient descent with the stop-before-diverging guard),
  the predicted IC50 matrix, DeepResponse's metric contract on the
  observed entries, and the ranked repurposing candidates per
  profile.  Sync-only light compute (tiny dense matrices, bounded
  epochs); every predicted value is labelled a model estimate over the
  synthetic screen — the route never invents real pharmacology and
  never emits a clinical recommendation.

The honest notes everywhere: the candidates are ranked hypotheses for
triage on a synthetic stand-in screen; no real cell line, drug panel or
response table ships with this route.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pharmaml.ml.response_model import (
    METRIC_KEYS,
    SPLIT_STRATEGIES,
    fit_response_recommender,
    predict_responses,
    repurposing_candidates,
    response_metrics,
    synthetic_response_screen,
)

router = APIRouter(prefix="/repurposing", tags=["repurposing"])

_MAX_CELLS = 40
_MAX_DRUGS = 30
_MAX_FEATURES = 20
_MAX_FACTORS = 10
_MAX_EPOCHS = 1000
_MAX_TOP_K = 10
_MAX_PROFILES = 8


class RepurposingScreenRequest(BaseModel):
    """One seeded synthetic screen + fit + ranking request.

    The geometry bounds mirror the backend's own rules exactly
    (``>= 4`` cells, ``>= 3`` drugs, ``>= 2`` features, ``observed``
    in ``[0.3, 1.0]``); the remaining caps are this route's sync-light
    budget, stated in the registry.
    """

    n_cells: int = Field(default=12, ge=4, le=_MAX_CELLS)
    n_drugs: int = Field(default=8, ge=3, le=_MAX_DRUGS)
    n_features: int = Field(default=6, ge=2, le=_MAX_FEATURES)
    n_factors: int = Field(default=3, ge=1, le=_MAX_FACTORS)
    lr: float = Field(default=0.05, ge=0.001, le=0.5)
    max_epochs: int = Field(default=200, ge=1, le=_MAX_EPOCHS)
    noise: float = Field(default=0.3, ge=0.0, le=2.0)
    observed: float = Field(default=0.8, ge=0.3, le=1.0)
    seed: int = Field(default=42, ge=0, le=2147483647)
    top_k: int = Field(default=5, ge=1, le=_MAX_TOP_K)
    profiles: int = Field(default=3, ge=1, le=_MAX_PROFILES)


def _provenance_block() -> dict[str, Any]:
    """The provenance + limitations block (the §11.1-honest state)."""
    return {
        "backend": "pharmaml.ml.response_model",
        "source": (
            "CSB5/CaDRReS — the matrix-factorisation recommender "
            "(S = mu + b_p + b_q + P.Q^T, feature-driven cell half) "
            "absorbed with the paper's update order and "
            "stop-before-diverging guard; burakcan-izmirli/"
            "DeepResponse — the five-strategy split ladder, the "
            "thresholded metric contract and the "
            "RepurposingInferenceEngine concept this ranking "
            "reproduces; SmritiChawla/Precily — the grouped-split "
            "discipline"
        ),
        "n_repositories": 3,
        "shipped_checkpoints": 0,
        "shipped_checkpoints_note": (
            "DeepResponse's SELFormer drug encoder, CNN cell encoder "
            "and FiLM fusion head are NOT shipped (the PyTorch hybrid "
            "stays external) — this lane ships the recommender that "
            "trains in-tree, not a pretrained predictor"
        ),
        "screen": (
            "the served screen is the backend's own SYNTHETIC seeded "
            "demo (latent-factor structure + Gaussian noise + a "
            "masked observation pattern) — the stand-in for a "
            "GDSC-sized table; no real GDSC/CCLE/CTRP response table "
            "ships with the toolkit"
        ),
        "disclaimer": (
            "predicted IC50 values are model estimates over the "
            "synthetic screen — repurposing candidates are ranked "
            "hypotheses for triage, never clinical recommendations"
        ),
    }


def _finite(value: float) -> float | None:
    """JSON-safe float (NaN/inf metrics read as null, never as lies)."""
    number = float(value)
    return number if math.isfinite(number) else None


@router.get("/registry")
def registry(request: Request) -> dict:
    return {
        "split_strategies": {
            "within_table": list(SPLIT_STRATEGIES),
            "cross_domain": (
                "cross_domain_split — the fifth strategy takes two "
                "tables and has its own entry point "
                "(DeepResponse's honest-leakage ladder)"
            ),
            "source": "DeepResponse's five split strategies, verbatim",
        },
        "model": {
            "family": "caDRReS-style matrix factorisation",
            "formulation": (
                "S = mu + b_p + b_q + P.Q^T with the cell half "
                "P = X.WP driven by cell-line features and the drug "
                "half Q = WQ a free embedding (the Y = I identity "
                "trick)"
            ),
            "direction": (
                "the matrix reads IC50 (lower = more sensitive); the "
                "model trains in the flipped sensitivity space "
                "S = -IC50 and every prediction flips back before it "
                "leaves the backend"
            ),
            "telemetry_fields": [
                "history",
                "observed_rmse",
                "stopped_early",
            ],
            "history_note": (
                "history is CaDRReS's (epoch, observed_rmse, ndcg) "
                "triple per epoch — the trainer has no live per-epoch "
                "callback; the full list lands after the fit, never a "
                "fabricated epoch stream"
            ),
        },
        "hyperparameters": {
            "n_factors": {"default": 3, "max": _MAX_FACTORS},
            "lr": {"default": 0.05, "min": 0.001, "max": 0.5},
            "max_epochs": {"default": 200, "max": _MAX_EPOCHS},
            "seed": {"default": 42, "max": 2147483647},
            "note": (
                "the backend's own defaults are n_factors=10 / "
                "lr=0.01 / max_epochs=500 (the paper's f=10 runs); "
                "the route's defaults are the doctest-proven "
                "small-screen set and the caps are the sync-light "
                "budget"
            ),
        },
        "screen": {
            "defaults": {
                "n_cells": 12,
                "n_drugs": 8,
                "n_features": 6,
                "noise": 0.3,
                "observed": 0.8,
                "seed": 42,
            },
            "bounds": {
                "n_cells": f"4..{_MAX_CELLS}",
                "n_drugs": f"3..{_MAX_DRUGS}",
                "n_features": f"2..{_MAX_FEATURES}",
                "observed": "0.3..1.0",
            },
            "label": (
                "SYNTHETIC — a seeded IC50 screen (IC50 = 4 - latent), "
                "the demo/test stand-in for a GDSC-sized table, not "
                "real pharmacology"
            ),
        },
        "repurposing": {
            "schema": ["cell_line", "drug", "predicted_ic50", "rank"],
            "ordering": (
                "most-sensitive first — lowest predicted IC50 = rank 1 "
                "(the IC50 direction: lower = more sensitive)"
            ),
            "source": (
                "DeepResponse's RepurposingInferenceEngine concept — "
                "score the model's whole drug axis for a profile and "
                "export the ranked table"
            ),
            "response_limits": {
                "profiles": f"1..{_MAX_PROFILES}",
                "top_k_per_profile": f"1..{_MAX_TOP_K}",
            },
        },
        "metrics": {
            "keys": list(METRIC_KEYS),
            "threshold": 6.0,
            "note": (
                "DeepResponse's metric contract on the observed "
                "entries — regression (rmse/mae/r2/pcc/scc) plus "
                "thresholded classification (accuracy/precision/"
                "recall/f1/mcc) at BINARY_THRESHOLD=6.0; non-finite "
                "metrics read as null, never as zero"
            ),
        },
        "provenance": _provenance_block(),
        "screen_url": "/api/v1/repurposing/screen",
        "note": (
            "sync-only light compute — dense bounded matrices; every "
            "predicted value is a model estimate over the synthetic "
            "screen, and the geometry bounds mirror the backend's own "
            "rules (4/3/2 minimums, observed in [0.3, 1.0])"
        ),
    }


@router.post("/screen")
def screen(req: RepurposingScreenRequest, request: Request) -> dict:
    # the real backend round trip: seeded screen -> CaDRReS fit ->
    # predicted IC50 matrix -> ranked repurposing candidates
    response, features, _truth = synthetic_response_screen(
        req.n_cells,
        req.n_drugs,
        req.n_features,
        n_factors=req.n_factors,
        noise=req.noise,
        observed=req.observed,
        seed=req.seed,
    )
    model = fit_response_recommender(
        response,
        features,
        n_factors=req.n_factors,
        lr=req.lr,
        max_epochs=req.max_epochs,
        seed=req.seed,
    )
    predicted = predict_responses(model, features)

    # the metric contract on the observed entries only
    mask = response.notna()
    true_values = response.to_numpy()[mask.to_numpy()]
    pred_values = predicted.to_numpy()[mask.to_numpy()]
    metrics = response_metrics(true_values, pred_values)

    ranked = repurposing_candidates(model, features.iloc[: req.profiles])
    per_profile: dict[str, list[dict[str, Any]]] = {}
    for cell_line, group in ranked.groupby("cell_line"):
        per_profile[str(cell_line)] = [
            {
                "drug": str(row.drug),
                "predicted_ic50": float(row.predicted_ic50),
                "rank": int(row.rank),
            }
            for row in group.head(req.top_k).itertuples()
        ]

    observed_entries = int(mask.to_numpy().sum())
    total_entries = int(mask.to_numpy().size)
    first_epoch, first_rmse, first_ndcg = (
        model.history[0] if model.history else (0, None, None)
    )

    return {
        "request": {
            "n_cells": req.n_cells,
            "n_drugs": req.n_drugs,
            "n_features": req.n_features,
            "n_factors": req.n_factors,
            "lr": req.lr,
            "max_epochs": req.max_epochs,
            "noise": req.noise,
            "observed": req.observed,
            "seed": req.seed,
            "top_k": req.top_k,
            "profiles": req.profiles,
        },
        "screen": {
            "cells": list(model.cells),
            "drugs": list(model.drugs),
            "features": list(model.features),
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
            "epochs_run": len(model.history),
            "max_epochs": req.max_epochs,
            "stopped_early": bool(model.stopped_early),
            "observed_rmse": _finite(model.observed_rmse),
            "first_epoch_rmse": _finite(first_rmse) if first_rmse is not None else None,
            "first_epoch": int(first_epoch),
            "first_epoch_ndcg": (
                _finite(first_ndcg) if first_ndcg is not None else None
            ),
            "n_factors": model.n_factors,
            "note": (
                "the trainer has no live per-epoch callback — the "
                "(epoch, observed_rmse, ndcg) history lands after the "
                "fit; only the first epoch and the final observed RMSE "
                "ride this answer, never a fabricated epoch stream"
            ),
        },
        "metrics": {
            key: _finite(value) for key, value in metrics.items()
        },
        "candidates": {
            "n_profiles": len(per_profile),
            "n_ranked_rows_total": len(ranked),
            "top_k": req.top_k,
            "per_profile": per_profile,
            "ordering": "lowest predicted IC50 = rank 1 (most sensitive first)",
        },
        "provenance": _provenance_block(),
        "note": (
            "predicted IC50 values are model estimates over the "
            "synthetic screen — ranked hypotheses for triage, never "
            "clinical recommendations; real screens (GDSC/CCLE/CTRP) "
            "are not shipped with the toolkit"
        ),
    }

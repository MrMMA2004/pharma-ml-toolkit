"""``/api/v1/admet`` — the ADMET-AI registry + scorecard surface over the
REAL vendored backend (part13 Phase 5's next remainder item — the ADMET
API lane over :mod:`pharmaml.ml.admet`).

* ``GET /api/v1/admet/endpoints`` — the 52-endpoint registry verbatim
  from the vendored ``swansonk14/admet_ai`` ``admet.csv`` table (id,
  category, task type, units, display-clipping bounds, species, TDC
  dataset size), each row joined with its headline model quality
  (AUROC/AUPRC for classification, R^2 for regression), the
  low-confidence set (quality below ``LOW_PERFORMANCE_THRESHOLD`` —
  ADMET-AI's warning rule), and the five radial axes with their
  directionality.  The provenance block states exactly what the
  offline backend is: the registry, the computed physchem panel and
  the DrugBank percentile grid — the 41 TDC-trained chemprop
  ensembles are NOT shipped (the toolkit's own learned-model path is
  ``ml/mpnn``), so the route never invents endpoint predictions.
* ``POST /api/v1/admet/score`` — the per-molecule scorecard over a
  SMILES batch: :func:`pharmaml.ml.admet.summarize_molecule` — the
  11-property physicochemical panel (RDKit descriptors + the
  PAINS/BRENK/NIH FilterCatalog alert counters), the five-axis radial
  summary on the vendored DrugBank quantile grid (the honest offline
  stand-ins where a learned endpoint is not available), and the
  low-confidence endpoint list.  Sync-only (light compute — the
  chemistry-descriptors precedent); invalid SMILES rows are reported
  per-row, never dropped silently and never a 500.

The honest notes everywhere (part13 §11.1's own rules): predicted
ADMET values are model estimates, never experimentally established
facts; model provenance, quality and limitations ride every answer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.admet import (
    ADMET_ENDPOINTS,
    ADMET_MODEL_QUALITY,
    LOW_PERFORMANCE_THRESHOLD,
    RADIAL_AXES,
    low_confidence_endpoints,
    summarize_molecule,
)

from ..stores import json_safe

router = APIRouter(prefix="/admet", tags=["admet"])

_MAX_SMILES = 100


class AdmetScoreRequest(BaseModel):
    # unique OpenAPI component names (the family's anti-collision rule:
    # same-named models elsewhere make the exported contract
    # hash-seed-unstable)
    smiles: list[str] = Field(min_length=1, max_length=_MAX_SMILES)

    @field_validator("smiles")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if any(not isinstance(s, str) or not s.strip() for s in value):
            raise ValueError("Every SMILES must be a non-empty string.")
        return value


def _provenance_block() -> dict[str, Any]:
    """The provenance + limitations block (part13 §11.1's own rule)."""
    return {
        "backend": "pharmaml.ml.admet",
        "source": (
            "swansonk14/admet_ai v2.0.1 — resources/data/admet.csv "
            "(the 52-row endpoint table) vendored verbatim; the "
            "physchem panel and the DrugBank radial grid distilled "
            "from its own resource files"
        ),
        "n_endpoints": len(ADMET_ENDPOINTS),
        "n_learned_endpoints": len(ADMET_MODEL_QUALITY),
        "shipped_ensembles": 0,
        "shipped_ensembles_note": (
            "the 41 TDC-trained chemprop ensembles are NOT shipped "
            "(torch checkpoints); the toolkit's own learned-model path "
            "is ml/mpnn — this route serves the registry, the computed "
            "physchem panel and the percentile grid, never invented "
            "endpoint predictions"
        ),
        "low_performance_threshold": LOW_PERFORMANCE_THRESHOLD,
        "disclaimer": (
            "predicted ADMET properties are model estimates, not "
            "experimentally established facts"
        ),
    }


@router.get("/endpoints")
def endpoints(request: Request) -> dict:
    rows = []
    for row in ADMET_ENDPOINTS:
        quality = ADMET_MODEL_QUALITY.get(row[0])
        rows.append(
            {
                "id": row[0],
                "category": row[1],
                "task_type": row[2],
                "units": row[3],
                "minimum": row[4],
                "maximum": row[5],
                "species": row[6],
                "size": row[7],
                "quality": quality,
                "low_confidence": quality is not None
                and float(quality) < LOW_PERFORMANCE_THRESHOLD,
            }
        )
    return {
        "endpoints": rows,
        "categories": sorted({row[1] for row in ADMET_ENDPOINTS}),
        "radial_axes": [
            {"endpoint": endpoint, "label": label, "invert": invert}
            for endpoint, label, invert in RADIAL_AXES
        ],
        "low_confidence_endpoints": low_confidence_endpoints(),
        "provenance": _provenance_block(),
        "score_url": "/api/v1/admet/score",
        "note": (
            "sync-only light compute — the chemistry-descriptors "
            "precedent; up to 100 SMILES per call"
        ),
    }


@router.post("/score")
def score(req: AdmetScoreRequest, request: Request) -> dict:
    molecules: list[dict[str, Any]] = []
    for raw in req.smiles:
        smiles = raw.strip()
        try:
            card = summarize_molecule(smiles)
        except DataValidationError as exc:
            molecules.append(
                {
                    "smiles": smiles,
                    "valid": False,
                    "error": f"invalid SMILES: {smiles!r} ({exc})",
                }
            )
            continue
        molecules.append(
            {
                "smiles": smiles,
                "valid": True,
                "physchem": {
                    key: json_safe(value) for key, value in card["physchem"].items()
                },
                "radial": {
                    key: json_safe(value) for key, value in card["radial"].items()
                },
                "low_confidence_endpoints": list(card["low_confidence_endpoints"]),
            }
        )
    n_valid = sum(1 for row in molecules if row["valid"])
    return {
        "n_smiles": len(req.smiles),
        "n_valid": n_valid,
        "n_invalid": len(req.smiles) - n_valid,
        "molecules": molecules,
        "provenance": _provenance_block(),
        "note": (
            "the radial axes use the offline physchem stand-ins "
            "(logP/QED) where the learned endpoint is not shipped — "
            "honest substitution, stated per answer"
        ),
    }

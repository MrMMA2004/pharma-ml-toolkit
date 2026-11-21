"""``/api/v1/dta`` — the DeepDTA encoding/paired-feature boundary over
the REAL in-tree backend (part13 Phase 5's next remainder item — the
DTA API lane over :mod:`pharmaml.ml.dta`).

* ``GET /api/v1/dta/registry`` — the encoding registry verbatim: the
  three DeepDTA alphabets (the 25-symbol, 1-based protein alphabet with
  zero reserved for padding; the 64-symbol isomeric and 62-symbol
  canonical SMILES alphabets), the 20 standard amino acids, the AAC
  protein panel (3 summary + 20 composition columns, plus the optional
  400-dipeptide lane with its own standard-fraction summary), the
  2048-bit ECFP ligand pairing, the ``deepdta_rm2``
  external-validation metric, and the demo frame.  The provenance block
  states exactly what the offline backend is: the DeepDTA CNN,
  DeepPurpose's transformer/CNN_RNN stack, DGL-LifeSci's graph models
  and TorchDrug's protein language models are NOT shipped — this is
  the feature/metric boundary, and the resulting ordinary feature
  matrix trains on any regressor already in the toolkit.  The route
  never invents affinity predictions.
* ``POST /api/v1/dta/features`` — the paired featurization over a
  (smiles, protein, optional affinity) batch through
  :func:`pharmaml.ml.dta.paired_dta_features`: the per-row protein
  composition cards (length, standard/non-standard fractions, the 20
  AAC percentages), the kept/dropped row accounting (RDKit-invalid
  ligands are dropped together with their protein and affinity values
  and are reported per-row, never silently), and the batch-level
  matrix shape.  Sync-only light compute; protein sequence errors and
  non-finite affinities are hard 422s — the backend's own contract
  (dropping those silently would make a benchmark look cleaner than
  its source data).

The honest notes everywhere: the paired features and encodings are
deterministic transformations, not learned affinity predictions; no
Kd/Ki/pIC50 value is produced by this route, ever.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from pharmaml.ml.dta import (
    AMINO_ACIDS,
    DEEPDTA_CHARCANSMISET,
    DEEPDTA_CHARISOSMISET,
    DEEPDTA_CHARPROTSET,
    paired_dta_features,
    protein_feature_frame,
)

router = APIRouter(prefix="/dta", tags=["dta"])

_MAX_PAIRS = 100


class DtaPairRow(BaseModel):
    """One drug--target pair: a SMILES ligand, a protein sequence, and
    an optional affinity value (the column the backend validates as
    finite numeric at source rows when any row carries it)."""

    smiles: str
    protein: str
    affinity: float | None = None

    @field_validator("smiles", "protein")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every pair needs a non-empty SMILES and protein.")
        return value


class DtaFeaturesRequest(BaseModel):
    # unique OpenAPI component names (the family's anti-collision rule)
    pairs: list[DtaPairRow] = Field(min_length=1, max_length=_MAX_PAIRS)
    dipeptides: bool = False


def _provenance_block() -> dict[str, Any]:
    """The provenance + limitations block (the §11.1-honest state)."""
    return {
        "backend": "pharmaml.ml.dta",
        "source": (
            "hkmztrk/DeepDTA — source/datahelper.py's 25/64/62-symbol "
            "alphabets and source/emetrics.py's rm2 ported verbatim; the "
            "paired ligand features are the toolkit's own 2048-bit ECFP "
            "(chem.features.compute_feature_frame)"
        ),
        "n_alphabets": 3,
        "shipped_checkpoints": 0,
        "shipped_checkpoints_note": (
            "the DeepDTA CNN, DeepPurpose's transformer/CNN_RNN stack, "
            "DGL-LifeSci's graph models and TorchDrug's protein language "
            "models are NOT shipped (torch/DGL checkpoints, zero) — this "
            "is the feature/metric boundary; the resulting ordinary "
            "feature matrix trains on any regressor already in the "
            "toolkit, and this route never invents affinity predictions"
        ),
        "disclaimer": (
            "paired DTA features are deterministic encodings, not learned "
            "affinity predictions — no Kd/Ki/pIC50 value is produced here"
        ),
    }


def _protein_card(row: pd.Series) -> dict[str, Any]:
    """One row's protein composition — the backend's own columns."""
    aac = {
        symbol: float(row[f"prot_aac_{symbol}"]) for symbol in AMINO_ACIDS
    }
    return {
        "prot_length": int(row["prot_length"]),
        "prot_standard_fraction": float(row["prot_standard_fraction"]),
        "prot_nonstandard_fraction": float(row["prot_nonstandard_fraction"]),
        "aac": aac,
    }


@router.get("/registry")
def registry(request: Request) -> dict:
    n_base = 3 + len(AMINO_ACIDS)
    n_dipeptide_extras = 1 + len(AMINO_ACIDS) ** 2
    n_with_dipeptides = n_base + n_dipeptide_extras
    return {
        "alphabets": {
            "protein": {
                "n_symbols": len(DEEPDTA_CHARPROTSET),
                "padding_code": 0,
                "symbols": dict(DEEPDTA_CHARPROTSET),
                "encoding": (
                    "1-based labels with zero reserved for padding; "
                    "left-aligned truncation and zero right-padding "
                    "(DeepDTA's datahelper.py, reproduced exactly)"
                ),
                "source": "DeepDTA CHARPROTSET verbatim",
            },
            "smiles_isomeric": {
                "n_symbols": len(DEEPDTA_CHARISOSMISET),
                "padding_code": 0,
                "symbols": dict(DEEPDTA_CHARISOSMISET),
                "encoding": (
                    "the supplied string is encoded as-is — no silent "
                    "canonicalization first (DeepDTA's upstream default)"
                ),
                "source": "DeepDTA CHARISOSMISET verbatim",
            },
            "smiles_canonical": {
                "n_symbols": len(DEEPDTA_CHARCANSMISET),
                "padding_code": 0,
                "symbols": dict(DEEPDTA_CHARCANSMISET),
                "encoding": "same contract, the 62-symbol canonical set",
                "source": "DeepDTA CHARCANSMISET verbatim",
            },
        },
        "amino_acids": list(AMINO_ACIDS),
        "protein_features": {
            "n_base": n_base,
            "base_columns": (
                ["prot_length", "prot_standard_fraction",
                 "prot_nonstandard_fraction"]
                + [f"prot_aac_{symbol}" for symbol in AMINO_ACIDS]
            ),
            "n_dipeptide_extras": n_dipeptide_extras,
            "n_with_dipeptides": n_with_dipeptides,
            "note": (
                "non-standard residues (B/Z/U/O/X…) are counted "
                "explicitly in prot_nonstandard_fraction, never "
                "silently relabelled"
            ),
        },
        "ligand_features": {
            "family": "ecfp",
            "n_bits": 2048,
            "source": (
                "the toolkit's own chem.features.compute_feature_frame "
                "(families=('ecfp',))"
            ),
        },
        "paired_width": {
            "base": 2048 + n_base,
            "with_dipeptides": 2048 + n_with_dipeptides,
        },
        "metrics": [
            {
                "name": "deepdta_rm2",
                "description": (
                    "DeepDTA's get_rm2 external-prediction metric — "
                    "squared correlation penalised by deviation from the "
                    "zero-intercept fit (the same family as the "
                    "concordance/RMSE metrics in ml.benchmetrics)"
                ),
            }
        ],
        "demo_frame": {
            "rows": 12,
            "note": (
                "demo_dta_frame — twelve synthetic, clearly "
                "non-benchmark pairs for the offline GUI"
            ),
        },
        "provenance": _provenance_block(),
        "features_url": "/api/v1/dta/features",
        "note": (
            "sync-only light compute — the chemistry-descriptors "
            "precedent; up to 100 pairs per call; sequence errors and "
            "non-finite affinities are hard 422s (the backend's own "
            "contract)"
        ),
    }


@router.post("/features")
def features(req: DtaFeaturesRequest, request: Request) -> dict:
    frame = pd.DataFrame(
        {
            "smiles": [row.smiles.strip() for row in req.pairs],
            "protein": [row.protein for row in req.pairs],
        }
    )
    has_affinity = any(row.affinity is not None for row in req.pairs)
    if has_affinity:
        # the backend's own hard contract: the column must be finite
        # numeric at every source row — a missing value on any row is
        # its verbatim 422, never a silent drop
        frame["affinity"] = [row.affinity for row in req.pairs]

    # the per-row protein composition (raises the backend's own hard
    # DataValidationError for bad sequences — the verbatim 422)
    proteins = protein_feature_frame(frame["protein"].tolist())

    feature_set = paired_dta_features(
        frame,
        affinity_column="affinity" if has_affinity else None,
        dipeptides=req.dipeptides,
    )
    kept = set(feature_set.kept_indices)

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(req.pairs):
        card: dict[str, Any] = {
            "index": index,
            "smiles": pair.smiles.strip(),
            "protein": pair.protein,
            "valid": index in kept,
            "protein_row": _protein_card(proteins.iloc[index]),
        }
        if has_affinity:
            card["affinity"] = pair.affinity if index in kept else None
        rows.append(card)

    affinity_summary: dict[str, Any] | None = None
    if has_affinity and feature_set.y is not None and len(feature_set.y):
        values = [float(v) for v in feature_set.y.to_numpy()]
        affinity_summary = {
            "present": True,
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    dropped = [
        {
            "index": int(index),
            "smiles": smiles,
            "reason": (
                "invalid SMILES — dropped together with its protein and "
                "affinity value by the ECFP featurization (never silently)"
            ),
        }
        for index, smiles in feature_set.dropped
    ]

    return {
        "n_pairs": len(req.pairs),
        "n_kept": len(feature_set.kept_indices),
        "n_dropped": len(dropped),
        "kept_indices": [int(i) for i in feature_set.kept_indices],
        "dropped": dropped,
        "n_ligand_features": int(feature_set.n_ligand_features),
        "n_protein_features": int(feature_set.n_protein_features),
        "n_features": int(feature_set.X.shape[1]),
        "dipeptides": bool(req.dipeptides),
        "rows": rows,
        "affinity": affinity_summary,
        "provenance": _provenance_block(),
        "note": (
            "the feature matrix is the product — an all-invalid batch "
            "is the backend's own 422, not an empty answer; predicted "
            "affinities require a trained model the toolkit's "
            "regressors provide"
        ),
    }

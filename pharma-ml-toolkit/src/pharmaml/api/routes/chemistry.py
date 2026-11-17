"""``/api/v1/chemistry`` — descriptors & fingerprints over the REAL
backend (part10: "Preserve scientifically validated backend
functionality. Reuse existing algorithms… Avoid duplicating scientific
algorithms in C#").

Two request shapes:

* **small/synchronous** (``mode="sync"``): a handful of SMILES,
  computed inline — the endpoint is *thin*, the science is 100% the
  existing :func:`pharmaml.chem.descriptors.compute_descriptors`;
* **managed job** (default): any input size, runs through the Job
  Manager with cooperative cancellation and ``descriptor.progress``
  events — the GUI path for the Datasets workspace's sequential
  workflow.

Fingerprints honour the previous-requirement clauses: *exact* Morgan
``radius`` and ``n_bits`` (configurable, validated, never clamped),
the four ``kind`` values the backend supports, and honest validation
errors.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from pharmaml.chem import descriptors as chem_descriptors

from ..events import EventType
from ..jobs import CancelledJob, JobManager

router = APIRouter(prefix="/chemistry", tags=["chemistry"])


StrictInt = Annotated[int, Field(strict=True)]


class DescriptorsRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=500)
    descriptor_set: Literal["rdkit_all", "morgan", "mordred"] = "rdkit_all"
    mode: Literal["job", "sync"] = "job"

    @field_validator("smiles")
    @classmethod
    def _nonempty_strings(cls, v: list[str]) -> list[str]:
        if any(not isinstance(s, str) or not s.strip() for s in v):
            raise ValueError("Every SMILES must be a non-empty string.")
        return v


class FingerprintsRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=500)
    kind: Literal["morgan", "rdkit", "atom_pair", "avalon"] = "morgan"
    radius: StrictInt = Field(default=2, ge=0, le=6)
    n_bits: StrictInt = Field(default=2048, gt=32, le=16384)
    as_counts: bool = False
    mode: Literal["job", "sync"] = "job"

    @field_validator("radius", "n_bits")
    @classmethod
    def _not_bool(cls, v: int) -> int:
        # the backend rejects bools for these ints; so does the API
        if isinstance(v, bool):
            raise TypeError("Booleans are not valid radius/n_bits.")
        return v


def _frame_to_payload(df: pd.DataFrame) -> dict:
    """JSON-safe projection of the backend's DataFrame result."""
    return {
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns.tolist()],
        "rows": [
            [None if (isinstance(v, float) and np.isnan(v)) else v
             for v in row]
            for row in df.itertuples(index=False)
        ],
        "index": [str(i) for i in df.index.tolist()],
    }


def _fp_to_payload(arr: np.ndarray, invalid: list[int]) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "invalid_indices": invalid,
        "matrix": arr.astype(int).tolist() if arr.size else [],
    }


def _descriptors_body(
    smiles: list[str],
    descriptor_set: str,
    cancel_event=None,
    job_id: str | None = None,
    manager: JobManager | None = None,
) -> dict:
    """Job body: the real backend call, cooperative between stages."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before descriptor stage.")
    if (
        manager is not None
        and job_id is not None
        and len(smiles) > 25
    ):
        manager.report_progress(
            job_id, 50, "computing descriptors",
            event=EventType.DESCRIPTOR_PROGRESS,
        )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled between descriptor stages.")
    from typing import cast

    df = chem_descriptors.compute_descriptors(
        smiles,
        descriptor_set=cast(
            "Literal['rdkit_all', 'morgan', 'mordred']", descriptor_set
        ),
        progress=False,
    )
    return _frame_to_payload(df)


def _fingerprints_body(
    smiles: list[str],
    kind: str,
    radius: int,
    n_bits: int,
    as_counts: bool,
    cancel_event=None,
) -> dict:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before fingerprint stage.")
    from typing import cast

    arr, invalid = chem_descriptors.compute_fingerprints(
        smiles,
        kind=cast("Literal['morgan', 'rdkit', 'atom_pair', 'avalon']", kind),
        radius=radius,
        n_bits=n_bits,
        as_counts=as_counts,
    )
    return _fp_to_payload(arr, invalid)


@router.post("/descriptors")
def descriptors(req: DescriptorsRequest, request: Request) -> dict:
    manager: JobManager = request.app.state.job_manager
    if req.mode == "sync":
        return {
            "mode": "sync",
            "result": _descriptors_body(req.smiles, req.descriptor_set),
        }
    record = manager.submit(
        "chemistry.descriptors",
        "datasets",
        _descriptors_body,
        configuration={
            "smiles_count": len(req.smiles),
            "descriptor_set": req.descriptor_set,
        },
        smiles=req.smiles,
        descriptor_set=req.descriptor_set,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/fingerprints")
def fingerprints(req: FingerprintsRequest, request: Request) -> dict:
    manager: JobManager = request.app.state.job_manager
    if req.mode == "sync":
        return {
            "mode": "sync",
            "result": _fingerprints_body(
                req.smiles, req.kind, req.radius, req.n_bits,
                req.as_counts,
            ),
        }
    record = manager.submit(
        "chemistry.fingerprints",
        "datasets",
        _fingerprints_body,
        configuration={
            "smiles_count": len(req.smiles),
            "kind": req.kind,
            "radius": req.radius,
            "n_bits": req.n_bits,
            "as_counts": req.as_counts,
        },
        smiles=req.smiles,
        kind=req.kind,
        radius=req.radius,
        n_bits=req.n_bits,
        as_counts=req.as_counts,
    )
    return {"mode": "job", "job": record.public()}

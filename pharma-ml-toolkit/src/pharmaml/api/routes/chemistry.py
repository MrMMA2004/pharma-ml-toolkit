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

DS-3 adds the registration verb: ``POST /chemistry/frames/register``
computes the descriptor frame or fingerprint matrix over a dataset
SMILES column (re-read server-side, row-aligned) and registers the
featurized frame as a NEW derived dataset -- the source columns plus
the appended features -- through the same store path every upload
takes.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from pharmaml.chem import descriptors as chem_descriptors
from pharmaml.chem.csv_standardize import detect_smiles_column
from pharmaml.chem.standardize import (
    canonicalize_tautomer,
    reinvent_standardize_smiles,
    require_rdkit_standardize,
)
from pharmaml.data import load_dataset
from pharmaml.exceptions import DataValidationError

from ..events import EventType
from ..jobs import CancelledJob, JobManager
from ..stores import DatasetStore

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
    include_chirality: bool = False
    mode: Literal["job", "sync"] = "job"

    @field_validator("radius", "n_bits")
    @classmethod
    def _not_bool(cls, v: int) -> int:
        # the backend rejects bools for these ints; so does the API
        if isinstance(v, bool):
            raise TypeError("Booleans are not valid radius/n_bits.")
        return v


class StandardizeRequest(BaseModel):
    """DS-2 -- the Clean stage's missing verb, over the REAL engine.

    ``ops`` is the ordered list of standardization passes, each a real
    :mod:`pharmaml.chem.standardize` callable applied in sequence:

    * ``"canonical_tautomer"`` -- the REINVENT ligprep pipeline's
      TautomerEnumerator pass (keto/enol forms collapse to one);
    * ``"reinvent"`` -- the REINVENT4 ``default`` chain (largest
      fragment -> RemoveHs -> salt removal -> neutralisation -> cleanup
      -> unwanted-pattern/size/element gates).

    ``neutralize`` toggles the chain's own step-4 pass (the engine's
    documented knob; honoured only when ``"reinvent"`` rides ``ops``).
    The standardized frame registers as a NEW dataset with full
    provenance (``derived`` block: source id + column + ops); the
    source record is never mutated. Rows the engine rejects are NOT
    silently dropped: they leave the derived frame but ride the
    response's ``rejected`` list (row, input, reason).
    """

    dataset_id: str
    column: str | None = None  # None = the honest auto-detection
    ops: list[Literal["canonical_tautomer", "reinvent"]] = Field(
        default=["reinvent"], min_length=1
    )
    neutralize: bool = True
    mode: Literal["job", "sync"] = "job"

    @field_validator("ops")
    @classmethod
    def _unique_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("ops must list at least one pass.")
        if len(set(v)) != len(v):
            raise ValueError("ops must not repeat a pass.")
        return v


class FramesRegisterRequest(BaseModel):
    """DS-3 -- the Descriptor/Fingerprint stage's missing verb.

    Registers the computed descriptor frame or fingerprint matrix as a
    NEW derived dataset through the same load-and-register path every
    upload takes. The engine re-reads the source record's SMILES
    column server-side (row-aligned with the source frame -- the one
    source of truth), computes the REAL features through
    :func:`pharmaml.chem.descriptors.compute_descriptors` /
    :func:`compute_fingerprints`, and appends the feature columns to a
    staged copy of the source frame: every source column (the target
    included) rides the derived dataset, so the ML/DL workspaces can
    train over the featurized matrix directly -- the classical
    SMILES -> descriptors -> model loop, closed inside the GUI.

    ``target`` is optional: ``None`` inherits the source record's own
    target; an explicit override must name an existing source column
    (the task is then re-inferred at load time). The source record is
    never mutated; rows the engine rejects (unparseable, or a
    per-molecule descriptor failure) leave the derived frame but ride
    the response's ``rejected`` list -- never silently dropped.
    """

    dataset_id: str
    column: str | None = None  # None = the honest auto-detection
    featurization: Literal["descriptors", "fingerprints"] = "descriptors"
    # the descriptors branch (featurization="descriptors")
    descriptor_set: Literal["rdkit_all", "morgan", "mordred"] = "rdkit_all"
    # the fingerprints branch (featurization="fingerprints")
    kind: Literal["morgan", "rdkit", "atom_pair", "avalon"] = "morgan"
    radius: StrictInt = Field(default=2, ge=0, le=6)
    n_bits: StrictInt = Field(default=2048, gt=32, le=16384)
    as_counts: bool = False
    include_chirality: bool = False
    # the target column to carry into the derived dataset (None = the
    # source's own target when it has one)
    target: str | None = None
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


#: How many values the SMILES-column gate probes before declaring a
#: column non-molecular (the §15.4 detector's own 100-value window).
_SMILES_GATE_PROBE = 100


def _column_is_smiles_bearing(frame: pd.DataFrame, column: str) -> bool:
    """The honest non-SMILES-column gate (DS-2's 422 clause).

    Probes up to :data:`_SMILES_GATE_PROBE` non-null values with the
    engine's own parser; a column where NOTHING parses is not a
    SMILES column and gets a 422 naming it (never a silent
    all-rejected run that masquerades as a valid standardization).
    """
    Chem = require_rdkit_standardize()
    seen = 0
    parsed = 0
    for value in frame[column].tolist():
        if not isinstance(value, str) or not value.strip():
            continue
        seen += 1
        try:
            probe = Chem.MolFromSmiles(value)
        except Exception:  # noqa: BLE001 - a probe, never a crash
            probe = None
        if probe is not None:
            parsed += 1
        if seen >= _SMILES_GATE_PROBE:
            break
    return parsed > 0


def _resolve_std_column(frame: pd.DataFrame, column: str | None) -> str:
    """Resolve + validate the SMILES column (both 422s are honest)."""
    if column is None:
        # the same auto-pick the §15.4 quality card and the scaffold
        # split lane use; raises naming every column it tried
        return detect_smiles_column(frame)
    available = [str(c) for c in frame.columns]
    if column not in available:
        raise DataValidationError(
            f"column {column!r} not found; available: {available[:20]}"
        )
    if not _column_is_smiles_bearing(frame, column):
        detected = "none"
        try:
            detected = detect_smiles_column(frame)
        except DataValidationError:
            pass
        raise DataValidationError(
            f"column {column!r} is not a SMILES column (no probed "
            f"value parses as a molecule); the frame's detected SMILES "
            f"column is {detected!r}"
        )
    return column


def _standardize_one(
    smiles: str, ops: Sequence[str], neutralize: bool
) -> tuple[str | None, str]:
    """One row through the ordered passes -> (output, rejection reason)."""
    Chem = require_rdkit_standardize()
    current = smiles
    for op in ops:
        if op == "canonical_tautomer":
            try:
                current = canonicalize_tautomer(current)
            except DataValidationError:
                return None, "unparseable"
        else:  # "reinvent" -- the REINVENT4 default chain
            out = reinvent_standardize_smiles(current, neutralize=neutralize)
            if out is None:
                parses = (
                    isinstance(current, str)
                    and Chem.MolFromSmiles(current) is not None
                )
                return (
                    None,
                    "rejected-by-standardization-chain"
                    if parses
                    else "unparseable",
                )
            current = out
    return current, ""


def _standardize_body(
    frame: pd.DataFrame,
    dataset_id: str,
    name: str,
    target: str,
    task: str,
    column: str,
    ops: Sequence[str],
    neutralize: bool,
    store: DatasetStore,
    cancel_event=None,
) -> dict:
    """Job/sync body: the real engine per row + the derived registration.

    Reads ``frame`` (the source record's frame) -- never writes it: the
    standardized rows are staged into ``out_frame`` (a positional
    ``.copy()``), serialized through the same temp-CSV path every
    upload takes, and registered by the REAL ``load_dataset`` +
    ``DatasetStore.register`` with the DS-2 provenance block.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the standardization stage.")
    raw_values = frame[column].tolist()
    outputs: list[str | None] = []
    reasons: list[str] = []
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            outputs.append(None)
            reasons.append("blank")
            continue
        out, reason = _standardize_one(value, ops, neutralize)
        outputs.append(out)
        reasons.append(reason)
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled after the standardization stage.")
    keep = [i for i, out in enumerate(outputs) if out is not None]
    if not keep:
        raise DataValidationError(
            "every SMILES row was rejected by the standardization "
            "chain -- no derived dataset to register"
        )
    out_frame = frame.iloc[keep].copy()
    out_frame[column] = [outputs[i] for i in keep]
    with tempfile.TemporaryDirectory(prefix="pharmaml-std-") as tmp:
        path = Path(tmp) / "standardized.csv"
        out_frame.to_csv(path, index=False)
        data = path.read_bytes()
        dataset = load_dataset(path, target=target, task=task)
    provenance = {
        "kind": "standardized",
        "source_dataset_id": dataset_id,
        "column": column,
        "ops": list(ops),
        "neutralize": neutralize,
    }
    record = store.register(
        dataset,
        name=f"{name} (standardized)",
        source_bytes=len(data),
        source_filename=f"{name}-standardized.csv",
        raw_bytes=data,  # the durable write-through, same as uploads
        derived=provenance,
    )
    rejected = [
        {"row": i, "input": str(raw_values[i]), "reason": reasons[i]}
        for i in range(len(raw_values))
        if outputs[i] is None
    ]
    changed = sum(
        1 for i in keep if outputs[i] != raw_values[i]
    )
    return {
        "source_dataset_id": dataset_id,
        "column": column,
        "ops": list(ops),
        "counts": {
            "n_rows": len(raw_values),
            "changed": changed,
            "unchanged": len(keep) - changed,
            "rejected": len(rejected),
        },
        "rejected": rejected,
        "provenance": provenance,
        "dataset": record.view(preview_rows=10),
    }


def _frames_register_body(
    frame: pd.DataFrame,
    dataset_id: str,
    name: str,
    column: str,
    featurization: str,
    descriptor_set: str,
    kind: str,
    radius: int,
    n_bits: int,
    as_counts: bool,
    include_chirality: bool,
    target: str | None,
    task: str | None,
    store: DatasetStore,
    cancel_event=None,
) -> dict:
    """Job/sync body: the real featurization engines + the derived
    registration (DS-3, the mirror of DS-2's standardization body).

    Reads ``frame`` (the source record's frame) -- never writes it: the
    surviving source rows are staged into a positional ``.copy()`` (the
    engine's own kept-index accounting aligns them), the feature
    columns are appended through ``pd.concat`` axis=1, the derived
    frame is serialized through the same temp-CSV path every upload
    takes, and registered by the REAL ``load_dataset`` +
    ``DatasetStore.register`` with the DS-3 provenance block
    (``derived-descriptors`` / ``derived-fingerprints``).
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the featurization stage.")
    values = frame[column].tolist()
    n_rows = len(values)
    if featurization == "descriptors":
        # the engine's own honest refusals (e.g. "No valid SMILES
        # found" on an all-invalid column) propagate to the route's
        # 422 lane verbatim -- never swallowed
        df = chem_descriptors.compute_descriptors(
            values,
            descriptor_set=cast(
                "Literal['rdkit_all', 'morgan', 'mordred']", descriptor_set
            ),
            progress=False,
        )
        kept = [int(i) for i in df.index.tolist()]
        feature_names = [str(c) for c in df.columns if str(c) != "SMILES"]
        # the honest rejected accounting: a dropped row is unparseable
        # (the engine's own parser) or a per-molecule descriptor failure
        Chem = require_rdkit_standardize()
        rejects: list[dict] = []
        for i in range(n_rows):
            if i in kept:
                continue
            value = values[i]
            parses = (
                isinstance(value, str)
                and value.strip()
                and Chem.MolFromSmiles(value) is not None
            )
            rejects.append({
                "row": i,
                "input": str(value),
                "reason": "unparseable" if not parses
                else "descriptor-computation-failure",
            })
        features_frame = df[feature_names].reset_index(drop=True)
        provenance: dict = {
            "kind": "derived-descriptors",
            "source_dataset_id": dataset_id,
            "column": column,
            "descriptor_set": descriptor_set,
            "n_features": len(feature_names),
            "target": target,
        }
        suffix = "descriptors"
    else:
        arr, valid = chem_descriptors.compute_fingerprints(
            values,
            kind=cast("Literal['morgan', 'rdkit', 'atom_pair', 'avalon']", kind),
            radius=radius,
            n_bits=n_bits,
            as_counts=as_counts,
            include_chirality=include_chirality,
        )
        kept = [int(i) for i in valid]
        feature_names = [f"fp_{i}" for i in range(int(arr.shape[1]))]
        # the engine's own invalid list is the unparseable accounting
        rejects = [
            {"row": i, "input": str(values[i]), "reason": "unparseable"}
            for i in range(n_rows)
            if i not in kept
        ]
        features_frame = pd.DataFrame(
            arr.astype(int), columns=feature_names
        )
        provenance = {
            "kind": "derived-fingerprints",
            "source_dataset_id": dataset_id,
            "column": column,
            "fingerprint_kind": kind,
            "radius": radius,
            "n_bits": int(arr.shape[1]),
            "as_counts": as_counts,
            "include_chirality": include_chirality,
            "target": target,
        }
        suffix = "fingerprints"
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled after the featurization stage.")
    if not kept:
        raise DataValidationError(
            "every SMILES row failed the featurization -- no derived "
            "dataset to register"
        )
    collisions = [c for c in feature_names if c in frame.columns]
    if collisions:
        raise DataValidationError(
            f"feature column {collisions[0]!r} collides with an existing "
            f"source column -- rename the source column first"
        )
    out_frame = pd.concat(
        [frame.iloc[kept].copy().reset_index(drop=True), features_frame],
        axis=1,
    )
    with tempfile.TemporaryDirectory(prefix="pharmaml-frames-") as tmp:
        path = Path(tmp) / "featurized.csv"
        out_frame.to_csv(path, index=False)
        data = path.read_bytes()
        dataset = load_dataset(path, target=target, task=task)
    record = store.register(
        dataset,
        name=f"{name} ({suffix})",
        source_bytes=len(data),
        source_filename=f"{name}-{suffix}.csv",
        raw_bytes=data,  # the durable write-through, same as uploads
        derived=provenance,
    )
    return {
        "source_dataset_id": dataset_id,
        "column": column,
        "featurization": featurization,
        "counts": {
            "n_rows": n_rows,
            "kept": len(kept),
            "rejected": len(rejects),
            "features_appended": len(feature_names),
        },
        "rejected": rejects,
        "provenance": provenance,
        "dataset": record.view(preview_rows=10),
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
    include_chirality: bool = False,
    cancel_event=None,
) -> dict:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before fingerprint stage.")
    arr, invalid = chem_descriptors.compute_fingerprints(
        smiles,
        kind=cast("Literal['morgan', 'rdkit', 'atom_pair', 'avalon']", kind),
        radius=radius,
        n_bits=n_bits,
        as_counts=as_counts,
        include_chirality=include_chirality,
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
                req.as_counts, req.include_chirality,
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
            "include_chirality": req.include_chirality,
        },
        smiles=req.smiles,
        kind=req.kind,
        radius=req.radius,
        n_bits=req.n_bits,
        as_counts=req.as_counts,
        include_chirality=req.include_chirality,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/standardize")
def standardize(req: StandardizeRequest, request: Request) -> dict:
    """DS-2 -- ``POST /api/v1/chemistry/standardize``.

    Runs the REAL REINVENT-style standardization engine over one
    dataset SMILES column and registers the standardized frame as a
    NEW derived dataset (source untouched -- sha-256-pinned by the API
    tests, scanned by the frontend drift guard).
    """
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(req.dataset_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset_id {req.dataset_id!r}.",
        )
    try:
        column = _resolve_std_column(record.frame, req.column)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    manager: JobManager = request.app.state.job_manager
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_standardize_body(
                    record.frame,
                    req.dataset_id,
                    record.name,
                    record.target,
                    record.task,
                    column,
                    req.ops,
                    req.neutralize,
                    store,
                ),
            }
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    job = manager.submit(
        "chemistry.standardize",
        "datasets",
        _standardize_body,
        configuration={
            "dataset_id": req.dataset_id,
            "column": column,
            "ops": req.ops,
            "neutralize": req.neutralize,
        },
        frame=record.frame,
        dataset_id=req.dataset_id,
        name=record.name,
        target=record.target,
        task=record.task,
        column=column,
        ops=req.ops,
        neutralize=req.neutralize,
        store=store,
    )
    return {"mode": "job", "job": job.public()}


@router.post("/frames/register")
def frames_register(req: FramesRegisterRequest, request: Request) -> dict:
    """DS-3 -- ``POST /api/v1/chemistry/frames/register``.

    Runs the REAL descriptor/fingerprint engines over one dataset
    SMILES column (re-read server-side -- the row-aligned source of
    truth) and registers the featurized frame as a NEW derived
    dataset: every source column (the target included) plus the
    appended feature columns. The source record is untouched
    (sha-256-pinned by the API tests, scanned by the frontend drift
    guard); rejected rows are accounted, never silently dropped.
    """
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(req.dataset_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset_id {req.dataset_id!r}.",
        )
    try:
        column = _resolve_std_column(record.frame, req.column)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # the target resolution: an explicit override must name an existing
    # source column (the task is re-inferred at load time); None inherits
    # the source record's own target + task verbatim
    if req.target is not None:
        available = [str(c) for c in record.frame.columns]
        if req.target not in available:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"target column {req.target!r} not found; "
                    f"available: {available[:20]}"
                ),
            )
        target, task = req.target, None
    else:
        target, task = record.target, record.task
    manager: JobManager = request.app.state.job_manager
    configuration: dict = {
        "dataset_id": req.dataset_id,
        "column": column,
        "featurization": req.featurization,
        "target": target,
    }
    if req.featurization == "descriptors":
        configuration["descriptor_set"] = req.descriptor_set
    else:
        configuration.update({
            "kind": req.kind,
            "radius": req.radius,
            "n_bits": req.n_bits,
            "as_counts": req.as_counts,
            "include_chirality": req.include_chirality,
        })
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_frames_register_body(
                    record.frame,
                    req.dataset_id,
                    record.name,
                    column,
                    req.featurization,
                    req.descriptor_set,
                    req.kind,
                    req.radius,
                    req.n_bits,
                    req.as_counts,
                    req.include_chirality,
                    target,
                    task,
                    store,
                ),
            }
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    job = manager.submit(
        "chemistry.frames.register",
        "datasets",
        _frames_register_body,
        configuration=configuration,
        frame=record.frame,
        dataset_id=req.dataset_id,
        name=record.name,
        column=column,
        featurization=req.featurization,
        descriptor_set=req.descriptor_set,
        kind=req.kind,
        radius=req.radius,
        n_bits=req.n_bits,
        as_counts=req.as_counts,
        include_chirality=req.include_chirality,
        target=target,
        task=task,
        store=store,
    )
    return {"mode": "job", "job": job.public()}

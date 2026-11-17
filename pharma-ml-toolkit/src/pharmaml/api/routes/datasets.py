"""``/api/v1/datasets`` — the Datasets workspace's API surface
(part10's queued row: datasets + the file **upload** contract).

The science is 100% the existing backend:
:func:`pharmaml.data.load_dataset` (target auto-detection, task
inference, non-numeric column policy), ``profile_frame`` and
``split_table`` (stratified splits).  Nothing is reimplemented here.

Upload contract (the Phase-Eight preview delivered with this row):

* multipart ``POST /api/v1/datasets`` — one CSV/TSV/TXT/XLSX file per
  request, size-capped (:data:`MAX_UPLOAD_BYTES`), extension-checked,
  parsed by the REAL reader, registered in the session store, and
  answered with a JSON-safe view (shape, columns, dtypes, target,
  task, classes, preview rows).  ``mode="job"`` runs the parse through
  the JobManager exactly like the chemistry routes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from pharmaml.data import load_dataset, split_table
from pharmaml.data_profiling import profile_frame
from pharmaml.exceptions import DataValidationError
from pharmaml.utils.io import safe_filename

from ..jobs import CancelledJob
from ..stores import MAX_UPLOAD_BYTES, DatasetStore, frame_payload, json_safe

router = APIRouter(prefix="/datasets", tags=["datasets"])

#: The reader's documented file families (load_dataset's own contract).
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}

MAX_PREVIEW_ROWS = 500


class ProfileRequest(BaseModel):
    mode: Literal["job", "sync"] = "sync"


class SplitRequest(BaseModel):
    target: str | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    stratify: bool = True
    random_state: int = 42
    mode: Literal["job", "sync"] = "sync"


def _load_and_register(
    data: bytes,
    suffix: str,
    name: str,
    source_filename: str,
    target: str | None,
    task: str | None,
    store: DatasetStore,
    cancel_event=None,
) -> dict:
    """Job body/sync body: parse via the REAL loader, register, view."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the parse stage.")
    with tempfile.TemporaryDirectory(prefix="pharmaml-ds-") as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        dataset = load_dataset(path, target=target, task=task)
    record = store.register(
        dataset, name=name, source_bytes=len(data),
        source_filename=source_filename,
        raw_bytes=data,  # part11 Phase Eleven: durable write-through
    )
    return record.view(preview_rows=10)


def _profile_body(frame: pd.DataFrame, cancel_event=None) -> dict:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the profile stage.")
    return {"profile": json_safe(profile_frame(frame))}


def _side_view(frame: pd.DataFrame, target: str | None) -> dict:
    view: dict = {"rows": len(frame), "columns": [str(c) for c in frame.columns]}
    if target is not None and target in frame.columns:
        counts = frame[target].value_counts()
        if len(counts) <= 20:  # label-like target: show the balance
            view["target_balance"] = json_safe(
                {str(k): int(v) for k, v in counts.items()}
            )
    view["preview"] = frame_payload(frame, 3)
    return view


def _split_body(
    frame: pd.DataFrame,
    target: str | None,
    test_size: float,
    val_size: float | None,
    stratify: bool,
    random_state: int,
    cancel_event=None,
) -> dict:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the split stage.")
    result = split_table(
        frame=frame,
        target=target,
        test_size=test_size,
        val_size=val_size,
        stratify=stratify,
        random_state=random_state,
    )
    sides = {name: _side_view(fr, result.target)
             for name, fr in result.sides().items()}
    return {
        "target": result.target,
        "stratified": result.stratified,
        "random_state": result.random_state,
        "sides": sides,
        "n_rows": len(frame),
    }


@router.post("")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(..., description="CSV/TSV/TXT/XLSX table"),
    target: str | None = Form(None),
    task: str | None = Form(None),
    mode: str = Form("job"),
) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    filename = file.filename or "upload.csv"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported upload type {suffix or '<none>'!r}; allowed: "
                + ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
            ),
        )
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload of {len(data)} bytes exceeds the "
                f"{MAX_UPLOAD_BYTES}-byte session cap."
            ),
        )
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload.")
    name = safe_filename(Path(filename).stem) or "dataset"
    manager = request.app.state.job_manager
    if mode == "sync":
        try:
            return {
                "mode": "sync",
                "dataset": _load_and_register(
                    data, suffix, name, filename, target, task, store
                ),
            }
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = manager.submit(
        "datasets.load",
        "datasets",
        _load_and_register,
        configuration={
            "filename": filename,
            "size_bytes": len(data),
            "target": target,
            "task": task,
        },
        data=data,
        suffix=suffix,
        name=name,
        source_filename=filename,
        target=target,
        task=task,
        store=store,
    )
    return {"mode": "job", "job": record.public()}


@router.get("")
def list_datasets(request: Request) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    return {"datasets": store.list()}


@router.get("/{dataset_id}")
def dataset_view(dataset_id: str, request: Request) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    return record.view(preview_rows=10)


@router.get("/{dataset_id}/preview")
def dataset_preview(
    dataset_id: str,
    request: Request,
    limit: int = 10,
) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    if not 1 <= limit <= MAX_PREVIEW_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be in [1, {MAX_PREVIEW_ROWS}].",
        )
    return frame_payload(record.frame, limit)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    if not store.evict(dataset_id):
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    return {"evicted": True, "dataset_id": dataset_id}


@router.post("/{dataset_id}/profile")
def profile_dataset(
    dataset_id: str, req: ProfileRequest, request: Request
) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    if req.mode == "sync":
        return {"mode": "sync", **_profile_body(record.frame)}
    manager = request.app.state.job_manager
    job = manager.submit(
        "datasets.profile",
        "datasets",
        _profile_body,
        configuration={"dataset_id": dataset_id, "columns": len(record.frame.columns)},
        frame=record.frame,
    )
    return {"mode": "job", "job": job.public()}


@router.post("/{dataset_id}/split")
def split_dataset(
    dataset_id: str, req: SplitRequest, request: Request
) -> dict:
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_split_body(
                    record.frame, req.target, req.test_size, req.val_size,
                    req.stratify, req.random_state,
                ),
            }
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    manager = request.app.state.job_manager
    job = manager.submit(
        "datasets.split",
        "datasets",
        _split_body,
        configuration={
            "dataset_id": dataset_id,
            "test_size": req.test_size,
            "val_size": req.val_size,
            "stratify": req.stratify,
            "random_state": req.random_state,
        },
        frame=record.frame,
        target=req.target,
        test_size=req.test_size,
        val_size=req.val_size,
        stratify=req.stratify,
        random_state=req.random_state,
    )
    return {"mode": "job", "job": job.public()}

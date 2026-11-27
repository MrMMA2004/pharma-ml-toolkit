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
from math import ceil
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from pharmaml.chem.csv_standardize import detect_smiles_column
from pharmaml.data import load_dataset, split_table
from pharmaml.data_profiling import profile_frame
from pharmaml.data_quality import data_quality_report
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.splits import SPLIT_STRATEGIES, scaffold_groups, split_indices
from pharmaml.utils.io import safe_filename

from ..jobs import CancelledJob
from ..stores import MAX_UPLOAD_BYTES, DatasetStore, frame_payload, json_safe

router = APIRouter(prefix="/datasets", tags=["datasets"])

#: The reader's documented file families (load_dataset's own contract).
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}

MAX_PREVIEW_ROWS = 500


class ProfileRequest(BaseModel):
    mode: Literal["job", "sync"] = "sync"


class QualityRequest(BaseModel):
    """The §15.4 scan's knobs: column overrides + execution mode.

    ``smiles_column`` left null = the honest auto-detection; an
    explicit name overrides it (an empty string disables the
    chemistry pair). ``target`` left null = the record's own target.
    """

    smiles_column: str | None = None
    target: str | None = None
    mode: Literal["job", "sync"] = "sync"


class SplitRequest(BaseModel):
    """The split lane's knobs (DS-1: the engine's strategy vocabulary).

    ``strategy=None`` keeps the legacy ``split_table`` path exactly as
    it has always answered (the ``stratify`` boolean decides random vs
    stratified, with target auto-detection). Setting ``strategy``
    routes the split through :func:`pharmaml.ml.splits.split_indices`
    -- the same engine the CLI and the ML templates already use --
    adding the scaffold (Bemis-Murcko groups held out whole), group
    (explicit labels), kennard_stone and temporal strategies the
    legacy lane cannot express. ``smiles_column`` is the scaffold
    lane's input (auto-detected when omitted); ``group_column`` is the
    group lane's input (required there, never guessed).
    """

    target: str | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    stratify: bool = True
    random_state: int = 42
    strategy: Literal["random", "stratified", "scaffold", "group", "kennard_stone", "temporal"] | None = None
    smiles_column: str | None = None
    group_column: str | None = None
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


def _quality_body(
    frame: pd.DataFrame,
    target: str | None,
    smiles_column: str | None,
    cancel_event=None,
) -> dict:
    """Job body/sync body: the read-only §15.4 report.

    ``smiles_column`` passes through raw: ``None`` = auto-detection,
    ``""`` = the explicit disable, any other string = the override
    (the engine's own documented contract).
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the quality stage.")
    return {
        "quality": json_safe(
            data_quality_report(
                frame, target=target, smiles_column=smiles_column
            )
        )
    }


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
    strategy: str | None = None,
    smiles_column: str | None = None,
    group_column: str | None = None,
    cancel_event=None,
) -> dict:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the split stage.")
    if strategy is not None:
        return _strategy_split(
            frame,
            target=target,
            strategy=strategy,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            smiles_column=smiles_column,
            group_column=group_column,
        )
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
        "strategy": None,
        "sides": sides,
        "n_rows": len(frame),
        "group_census": None,
    }


def _strategy_split(
    frame: pd.DataFrame,
    *,
    target: str | None,
    strategy: str,
    test_size: float,
    val_size: float | None,
    random_state: int,
    smiles_column: str | None,
    group_column: str | None,
) -> dict:
    """The DS-1 strategy lane: one frame, one honest strategy split.

    Splits the frame through :func:`pharmaml.ml.splits.split_indices`
    -- the same engine the CLI and the in-backend workflow templates
    use -- instead of the legacy ``split_table`` boolean. The response
    keeps the legacy sides-view shape and adds ``strategy`` /
    ``smiles_column`` / ``group_column`` / ``group_census`` fields. The
    optional val side is carved by a SECOND pass of the same strategy
    over the train remainder: each pass only ever moves whole
    scaffold/group cohorts, so groups stay intact across all three
    sides. The census reports each side's group count, its eight
    largest cohorts, and the ``shared_groups`` list -- which must be
    empty for scaffold/group (a shared entry is a leakage verdict, not
    a warning).
    """
    if strategy not in SPLIT_STRATEGIES:
        raise DataValidationError(
            f"unknown split strategy {strategy!r} -- valid strategies: "
            + ", ".join(SPLIT_STRATEGIES)
        )
    n = len(frame)

    # -- prerequisites: honest 422s, never guesses --------------------
    if target is not None and target not in frame.columns:
        raise DataValidationError(
            f"target column {target!r} not found; available: "
            f"{[str(c) for c in frame.columns][:20]}"
        )
    if strategy == "stratified" and target is None:
        raise DataValidationError(
            "the 'stratified' strategy needs a target column "
            "(the legacy stratify lane auto-detects; pick the target "
            "explicitly here)"
        )
    smiles_col: str | None = None
    if strategy == "scaffold":
        smiles_col = smiles_column
        if smiles_col is None:
            # the same auto-pick the §15.4 quality card uses; raises
            # DataValidationError naming every column it tried
            smiles_col = detect_smiles_column(frame)
        elif smiles_col not in frame.columns:
            raise DataValidationError(
                f"smiles column {smiles_col!r} not found; available: "
                f"{[str(c) for c in frame.columns][:20]}"
            )
    group_col: str | None = None
    if strategy == "group":
        group_col = group_column
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
            str(c)
            for c in frame.columns
            if str(c) != target and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if not feature_cols:
            raise DataValidationError(
                "kennard_stone needs at least one numeric feature column "
                "(distances are computed on the numeric matrix)"
            )

    def _engine(sub: pd.DataFrame, frac: float):
        """One engine call over a sub-frame (positional, never index)."""
        if strategy == "kennard_stone":
            X: object = sub[feature_cols].to_numpy(dtype=float)
        else:
            X = sub
        y: object = sub[target].to_numpy() if strategy == "stratified" else None
        smiles = (
            [str(v) for v in sub[smiles_col].tolist()]
            if strategy == "scaffold"
            else None
        )
        groups = sub[group_col].tolist() if strategy == "group" else None
        return split_indices(
            X=X,
            y=y,
            strategy=strategy,
            test_size=frac,
            random_state=random_state,
            smiles=smiles,
            groups=groups,
        )

    train_idx, test_idx = _engine(frame, test_size)
    train_frame = frame.iloc[train_idx]
    test_frame = frame.iloc[test_idx]
    val_frame = None
    if val_size is not None:
        n_val_target = ceil(val_size * n)
        frac_val = n_val_target / max(len(train_frame), 1)
        if not 0.0 < frac_val < 1.0:
            raise DataValidationError(
                f"val_size={val_size} leaves no rows to train after the "
                f"test carve (train remainder is {len(train_frame)} rows)"
            )
        tr2_idx, val_idx = _engine(train_frame, frac_val)
        val_frame = train_frame.iloc[val_idx]
        train_frame = train_frame.iloc[tr2_idx]

    sides: dict[str, pd.DataFrame] = {"train": train_frame}
    if val_frame is not None:
        sides["val"] = val_frame
    sides["test"] = test_frame

    response: dict = {
        "target": target,
        "stratified": strategy == "stratified",
        "random_state": random_state,
        "strategy": strategy,
        "smiles_column": smiles_col,
        "group_column": group_col,
        "sides": {name: _side_view(fr, target) for name, fr in sides.items()},
        "n_rows": n,
        "group_census": None,
    }

    if strategy in ("scaffold", "group"):
        per_side: dict[str, dict[str, int]] = {}
        for name, fr in sides.items():
            if strategy == "scaffold":
                keys = scaffold_groups(
                    [str(v) for v in fr[smiles_col].tolist()]
                )
            else:
                keys = [str(v) for v in fr[group_col].tolist()]
            counts: dict[str, int] = {}
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
            per_side[name] = counts
        shared: set[str] = set()
        names = list(per_side)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared |= set(per_side[a]) & set(per_side[b])
        census: dict[str, object] = {
            name: {
                "n_groups": len(counts),
                "top_groups": [
                    [key, count]
                    for key, count in sorted(
                        counts.items(), key=lambda kv: (-kv[1], kv[0])
                    )[:8]
                ],
            }
            for name, counts in per_side.items()
        }
        census["shared_groups"] = sorted(shared)
        response["group_census"] = census
    return response


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


#: The chemistry engine's own request cap (``DescriptorsRequest.smiles``
#: / ``FingerprintsRequest.smiles`` ``max_length`` in routes/chemistry.py)
#: — the single limit this route's bound mirrors, so the GUI feeds the
#: chemistry step inside the engine's own documented ceiling (§11.1's
#: Descriptor/Fingerprint stage) instead of shipping an unbounded column.
CHEMISTRY_MAX_SMILES = 500


@router.get("/{dataset_id}/column/{column}")
def dataset_column(
    dataset_id: str,
    column: str,
    request: Request,
    limit: int = CHEMISTRY_MAX_SMILES,
) -> dict:
    """One column's values — the §11.1 chemistry-stage bridge.

    Bounded by the chemistry request models' own ``max_length`` cap and
    explicit about it: ``n_rows`` + ``truncated`` state the dataset's
    real size so a bounded view never masquerades as the whole column
    (§17.3's "a display limit must never silently truncate" rule).
    Values ride ``json_safe`` (NaN/Inf → ``null``).
    """
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    if column not in [str(c) for c in record.frame.columns]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown column {column!r}.",
        )
    if not 1 <= limit <= CHEMISTRY_MAX_SMILES:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be in [1, {CHEMISTRY_MAX_SMILES}].",
        )
    series = record.frame[column]
    values = [json_safe(v) for v in series.iloc[:limit].tolist()]
    return {
        "column": column,
        "dtype": str(series.dtype),
        "n_rows": len(series),
        "returned": len(values),
        "truncated": len(series) > limit,
        "values": values,
    }


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


@router.post("/{dataset_id}/quality")
def quality_dataset(
    dataset_id: str, req: QualityRequest, request: Request
) -> dict:
    """``POST /api/v1/datasets/{id}/quality`` -- the §15.4 scan.

    The part16 §15.4 report (SMILES/structure validation, duplicate
    detection, the leakage family, class imbalance, the pinned unit
    ranges) over one registered dataset -- read-only by
    construction: the report describes the frame, it never edits a
    user-provided cell (the ``read_only`` flag rides the payload).
    """
    store: DatasetStore = request.app.state.dataset_store
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    target = req.target if req.target is not None else record.target
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_quality_body(record.frame, target, req.smiles_column),
        }
    manager = request.app.state.job_manager
    job = manager.submit(
        "datasets.quality",
        "datasets",
        _quality_body,
        configuration={
            "dataset_id": dataset_id,
            "smiles_column": req.smiles_column,
            "target": target,
        },
        frame=record.frame,
        target=target,
        smiles_column=req.smiles_column,
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
    # DS-1: the strategy lane falls back to the registration-resolved
    # target when the request leaves it null (the legacy lane keeps its
    # own auto-detection inside split_table -- untouched).
    target = req.target
    if req.strategy is not None and target is None:
        target = record.target
    if req.mode == "sync":
        try:
            return {
                "mode": "sync",
                **_split_body(
                    record.frame, target, req.test_size, req.val_size,
                    req.stratify, req.random_state,
                    strategy=req.strategy,
                    smiles_column=req.smiles_column,
                    group_column=req.group_column,
                ),
            }
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            # sklearn's stratification boundary (e.g. the test fold is
            # smaller than the target's class count) is a user-input
            # problem, not a server fault — a clean 422, never a 500
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
            "strategy": req.strategy,
            "smiles_column": req.smiles_column,
            "group_column": req.group_column,
        },
        frame=record.frame,
        target=target,
        test_size=req.test_size,
        val_size=req.val_size,
        stratify=req.stratify,
        random_state=req.random_state,
        strategy=req.strategy,
        smiles_column=req.smiles_column,
        group_column=req.group_column,
    )
    return {"mode": "job", "job": job.public()}

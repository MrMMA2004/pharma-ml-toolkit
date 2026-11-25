"""``/api/v1/artifacts`` — the artifact surface + the file
**download** contract (part10's queued row; Phase-Eight preview).

Artifact kinds, each produced by the REAL backend:

* ``figure`` — a plot built through :mod:`pharmaml.viz.figures`
  (the toolkit's own trace builders) and exported through the §17
  four-format stack (:func:`pharmaml.viz.export.export_figure` —
  PNG/SVG/TIFF/JPEG, lossless TIFF codecs, validated writes);
* ``table`` — a registered dataset written by
  :func:`pharmaml.utils.io.write_table` (CSV/TSV/XLSX);
* ``model`` — a session-trained estimator persisted by
  :func:`pharmaml.ml.persistence.save_model` as the standard
  joblib bundle with full metadata sidecar.

Download contract (verified byte-for-byte in the tests):

* exact ``Content-Length`` on the body;
* ``Content-Type`` per artifact;
* ``Content-Disposition: attachment; filename=...``;
* strong ``ETag`` = the artifact's sha256, plus
  ``X-Artifact-SHA256`` / ``X-Artifact-Kind``;
* ``HEAD`` support with the same digest headers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.persistence import save_model
from pharmaml.utils.io import safe_filename, write_table
from pharmaml.viz.export import export_figure
from pharmaml.viz.figures import (
    Figure,
    bar_traces,
    histogram_traces,
    render_matplotlib,
    scatter_traces,
)

from ..stores import ArtifactStore, json_safe

router = APIRouter(prefix="/artifacts", tags=["artifacts"])

_CONTENT_TYPES = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".csv": "text/csv; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument"
        ".spreadsheetml.sheet"
    ),
    ".joblib": "application/octet-stream",
}

_FIGURE_EXTENSIONS = {
    "PNG": ".png",
    "SVG": ".svg",
    "TIFF": ".tiff",
    "JPEG": ".jpg",
}


class FigureRequest(BaseModel):
    kind: Literal["scatter", "bar", "histogram"] = "scatter"
    x: list[float | str] = Field(min_length=1, max_length=2000)
    y: list[float] | None = None
    color: list[str] | None = None
    title: str | None = None
    format: Literal["PNG", "SVG", "TIFF", "JPEG"] = "PNG"
    dpi: int = Field(default=150, ge=72, le=300)
    filename: str | None = None

    @field_validator("color")
    @classmethod
    def _color_len(cls, v, info):
        x = info.data.get("x")
        if v is not None and x is not None and len(v) != len(x):
            raise ValueError("color must be the same length as x")
        return v

    @field_validator("y")
    @classmethod
    def _y_len(cls, v, info):
        x = info.data.get("x")
        if v is not None and x is not None and len(v) != len(x):
            raise ValueError("y must be the same length as x")
        return v

    def frame(self) -> pd.DataFrame:
        data: dict = {"x": list(self.x)}
        if self.y is not None:
            data["y"] = list(self.y)
        if self.color is not None:
            data["color"] = list(self.color)
        return pd.DataFrame(data)


class ArtifactTableRequest(BaseModel):
    # renamed for a unique OpenAPI component name: a same-named
    # request model lives in routes/visualization.py (see the note
    # in routes/applicability_domain.py)
    dataset_id: str
    format: Literal["csv", "tsv", "xlsx"] = "csv"
    filename: str | None = None


class ModelArtifactRequest(BaseModel):
    model_id: str
    metrics: dict | None = None
    filename: str | None = None


def _safe_stem(base: str, fallback: str) -> str:
    """The user-supplied filename reduced to a portable stem.

    The penetration-test lane's finding F-356-01 (part17 §4.3
    "unsafe export names"): the raw ``Path(base).stem`` preserved
    CR/LF, quotes and other control characters, which flowed into
    the ``Content-Disposition`` header at download time — a header
    injection surface on every artifact kind. The stem is now passed
    through :func:`pharmaml.utils.io.safe_filename` (the datasets
    route's own lane), so the stored filename is a portable token
    and the download header stays a single well-formed line.
    """
    return safe_filename(Path(base).stem) or fallback


def _register_figure(req: FigureRequest, store: ArtifactStore) -> dict:
    """Build + export + register a figure artifact (real viz stack)."""
    if req.kind in ("scatter",) and req.y is None:
        raise DataValidationError("scatter needs y (same length as x)")
    frame = req.frame()
    if req.kind == "scatter":
        traces = scatter_traces(
            frame, x="x", y="y", color="color" if "color" in frame.columns else None
        )
    elif req.kind == "bar":
        traces = bar_traces(
            frame,
            x="x",
            y="y" if "y" in frame.columns else None,
            color="color" if "color" in frame.columns else None,
        )
    else:
        traces = histogram_traces(
            frame,
            x="x",
            color="color" if "color" in frame.columns else None,
        )
    figure = Figure()
    for trace in traces:
        figure.add_trace(trace)
    if req.title:
        figure.update_layout(title=req.title)
    mpl_figure = render_matplotlib(figure)
    extension = _FIGURE_EXTENSIONS[req.format]
    with tempfile.TemporaryDirectory(prefix="pharmaml-art-") as tmp:
        target = Path(tmp) / f"figure{extension}"
        result = export_figure(mpl_figure, target, dpi=float(req.dpi))
        data = target.read_bytes()
    base = req.filename or "figure"
    record = store.register(
        kind="figure",
        filename=f"{_safe_stem(base, 'figure')}{extension}",
        content_type=_CONTENT_TYPES[extension],
        data=data,
        meta={
            "figure_kind": req.kind,
            "format": result.format,
            "width": result.width,
            "height": result.height,
            "dpi": req.dpi,
            "trace_count": len(traces),
            "n_points": len(req.x),
            "export_warnings": list(result.warnings or []),
        },
    )
    return record.view()


def _register_table(
    req: ArtifactTableRequest, store, frame: pd.DataFrame
) -> dict:
    extension = f".{req.format}"
    with tempfile.TemporaryDirectory(prefix="pharmaml-art-") as tmp:
        target = Path(tmp) / f"table{extension}"
        write_table(frame, target)
        data = target.read_bytes()
    base = req.filename or "table"
    record = store.register(
        kind="table",
        filename=f"{_safe_stem(base, 'table')}{extension}",
        content_type=_CONTENT_TYPES[extension],
        data=data,
        meta={
            "rows": len(frame),
            "columns": int(frame.shape[1]),
            "column_names": [str(c) for c in frame.columns],
        },
    )
    return record.view()


def _register_model(req: ModelArtifactRequest, store, model_record) -> dict:
    metrics = dict(req.metrics or {})
    metrics.update(
        {
            "cv_mean": model_record.cv_mean,
            "cv_std": model_record.cv_std,
            "fit_seconds": model_record.fit_seconds,
        }
    )
    with tempfile.TemporaryDirectory(prefix="pharmaml-art-") as tmp:
        target = Path(tmp) / "model.joblib"
        save_model(
            model_record.model,
            target,
            name=model_record.name,
            task=model_record.task,
            features=model_record.features,
            best_params=model_record.best_params,
            metrics=metrics,
            target=model_record.target,
        )
        data = target.read_bytes()
    base = req.filename or f"{model_record.name}-model"
    record = store.register(
        kind="model",
        filename=f"{_safe_stem(base, 'model')}.joblib",
        content_type=_CONTENT_TYPES[".joblib"],
        data=data,
        meta={
            "model_name": model_record.name,
            "task": model_record.task,
            "features": model_record.features,
            "best_params": json_safe(model_record.best_params),
            "metrics": json_safe(metrics),
            "target": model_record.target,
            "n_train_rows": model_record.n_train_rows,
        },
    )
    return record.view()


@router.post("/figure")
def create_figure(req: FigureRequest, request: Request) -> dict:
    store: ArtifactStore = request.app.state.artifact_store
    try:
        return {"artifact": _register_figure(req, store)}
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/table")
def create_table(req: ArtifactTableRequest, request: Request) -> dict:
    store: ArtifactStore = request.app.state.artifact_store
    dataset = request.app.state.dataset_store.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    return {"artifact": _register_table(req, store, dataset.frame)}


@router.post("/model")
def create_model_artifact(
    req: ModelArtifactRequest, request: Request
) -> dict:
    store: ArtifactStore = request.app.state.artifact_store
    model_record = request.app.state.model_store.get(req.model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    return {"artifact": _register_model(req, store, model_record)}


@router.get("")
def list_artifacts(request: Request) -> dict:
    return {"artifacts": request.app.state.artifact_store.list()}


@router.get("/{artifact_id}")
def artifact_view(artifact_id: str, request: Request) -> dict:
    record = request.app.state.artifact_store.get(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown artifact_id.")
    return record.view()


def _download_headers(record) -> dict[str, str]:
    # defense-in-depth for the F-356-01 lane: the stored filename is
    # sanitised at registration, and the header builder additionally
    # strips CR/LF and quote characters, so no legacy record can ever
    # break the Content-Disposition into a second header line
    clean = "".join(
        ch for ch in record.filename if ch not in '\"\r\n\x00'
    ) or "artifact"
    return {
        "Content-Disposition": f'attachment; filename="{clean}"',
        "ETag": f'"{record.sha256}"',
        "X-Artifact-SHA256": record.sha256,
        "X-Artifact-Kind": record.kind,
    }


@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str, request: Request) -> Response:
    record = request.app.state.artifact_store.get(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown artifact_id.")
    return Response(
        content=record.data,
        media_type=record.content_type,
        headers=_download_headers(record),
    )


@router.head("/{artifact_id}/download")
def download_artifact_head(
    artifact_id: str, request: Request
) -> Response:
    record = request.app.state.artifact_store.get(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown artifact_id.")
    return Response(
        content=b"",
        media_type=record.content_type,
        headers={
            **_download_headers(record),
            "Content-Length": str(record.size),
        },
    )


@router.delete("/{artifact_id}")
def delete_artifact(artifact_id: str, request: Request) -> dict:
    if not request.app.state.artifact_store.evict(artifact_id):
        raise HTTPException(status_code=404, detail="Unknown artifact_id.")
    return {"evicted": True, "artifact_id": artifact_id}

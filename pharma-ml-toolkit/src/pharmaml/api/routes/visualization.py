"""``/api/v1/visualization`` — the Image Lab workspace's API surface
over the REAL backend (part10's row four).

Every figure is built by the toolkit's own viz layer and exported
through the §17 four-format stack
(:func:`pharmaml.viz.export.export_figure`), then registered in the
session artifact store with the same download contract as
``/api/v1/artifacts`` (exact bytes, sha256 ETag, HEAD support).

* ``GET /api/v1/visualization/catalog`` — the plot families this
  route serves (statplots, pharma figures, MD series, tables,
  reports) + ``PALETTE`` + the export formats.
* ``POST /api/v1/visualization/stat`` — the statplots family
  (:func:`plot_box`, :func:`plot_histogram`, :func:`plot_ecdf`,
  :func:`plot_regression`): the REAL stats dict plus the rendered
  figure artifact.
* ``POST /api/v1/visualization/concentration-profile`` —
  :func:`pharmaml.viz.pharma_plots.plot_concentration_profile`.
* ``POST /api/v1/visualization/dose-accumulation`` —
  :func:`pharmaml.viz.pharma_plots.plot_dose_accumulation`.
* ``POST /api/v1/visualization/md-series`` —
  :func:`pharmaml.viz.md_plots.plot_md_series` over parsed ``.xvg``
  text (one series or several).
* ``POST /api/v1/visualization/table`` —
  :func:`pharmaml.viz.report.export_table` (csv/tsv/xlsx/md/json/html).
* ``POST /api/v1/visualization/report`` —
  :class:`pharmaml.viz.report.HTMLReport` as a managed job: metrics,
  tables and figure sections assembled into one self-contained HTML
  artifact, with honest ``file.progress`` events per section.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.viz._compat import PALETTE, close_figure
from pharmaml.viz.export import export_figure
from pharmaml.viz.md_plots import plot_md_series
from pharmaml.viz.pharma_plots import (
    plot_concentration_profile,
    plot_dose_accumulation,
)
from pharmaml.viz.report import REPORT_THEMES, HTMLReport, export_table
from pharmaml.viz.statplots import (
    plot_box,
    plot_ecdf,
    plot_histogram,
    plot_regression,
)

from ..events import EventType
from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/visualization", tags=["visualization"])

_MAX_POINTS = 5000
_MAX_ROWS = 5000
_MAX_SERIES = 4

_FIGURE_EXTENSIONS = {
    "PNG": ".png",
    "SVG": ".svg",
    "TIFF": ".tiff",
    "JPEG": ".jpg",
}
_CONTENT_TYPES = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tiff": "image/tiff",
    ".jpg": "image/jpeg",
    ".csv": "text/csv; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
}
_TABLE_FORMATS = ("csv", "tsv", "xlsx", "md", "json", "html")


def _export_and_register(
    store,
    figure: Any,
    filename: str,
    format: str = "PNG",
    dpi: int = 150,
    meta: dict | None = None,
) -> dict:
    """The artifacts-route byte path: export -> bytes -> register."""
    extension = _FIGURE_EXTENSIONS[format]
    with tempfile.TemporaryDirectory(prefix="pharmaml-viz-") as tmp:
        target = Path(tmp) / f"figure{extension}"
        result = export_figure(figure, target, dpi=float(dpi))
        data = target.read_bytes()
    record = store.register(
        kind="figure",
        filename=f"{Path(filename).stem or 'figure'}{extension}",
        content_type=_CONTENT_TYPES[extension],
        data=data,
        meta={
            "format": result.format,
            "width": result.width,
            "height": result.height,
            "dpi": dpi,
            "export_warnings": list(result.warnings or []),
            **(meta or {}),
        },
    )
    return record.view()


class StatRequest(BaseModel):
    kind: Literal["box", "histogram", "ecdf", "regression"]
    values: list[float] | None = Field(
        default=None, min_length=4, max_length=_MAX_POINTS
    )
    x: list[float] | None = Field(
        default=None, min_length=4, max_length=_MAX_POINTS
    )
    y: list[float] | None = Field(
        default=None, min_length=4, max_length=_MAX_POINTS
    )
    bins: int | str = Field(default="auto")
    whis: float = Field(default=1.5, gt=0.0, le=10.0)
    n_boot: int = Field(default=200, ge=0, le=2000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    color: str | None = Field(default=None, max_length=32)
    with_artifact: bool = True
    # statplots writes through its own save_figure (png/svg only);
    # the four-format export stack serves the Figure-returning routes
    format: Literal["PNG", "SVG"] = "PNG"


class ConcentrationProfileRequest(BaseModel):
    times: list[float] = Field(min_length=2, max_length=_MAX_POINTS)
    conc: list[float] = Field(min_length=2, max_length=_MAX_POINTS)
    auc: float | None = None
    title: str = "Concentration-time profile"
    format: Literal["PNG", "SVG", "TIFF", "JPEG"] = "PNG"
    dpi: int = Field(default=150, ge=72, le=300)

    @staticmethod
    def _equal(values_a: list | None, values_b: list | None) -> None:
        if values_a is None or values_b is None:
            raise ValueError("times and conc are both required")
        if len(values_a) != len(values_b):
            raise ValueError(
                f"times has {len(values_a)} points; conc has {len(values_b)}"
            )


class DoseAccumulationRequest(BaseModel):
    dose: float = Field(gt=0.0, le=1e7)
    tau: float = Field(gt=0.0, le=10000.0)
    t_half: float | None = Field(default=None, gt=0.0, le=10000.0)
    ke: float | None = Field(default=None, gt=0.0, le=100.0)
    vd: float = Field(gt=0.0, le=1e7)
    f: float = Field(default=1.0, gt=0.0, le=1.0)
    n_doses: int = Field(default=6, ge=1, le=24)
    title: str = "Repeated dosing — approach to steady state"
    format: Literal["PNG", "SVG", "TIFF", "JPEG"] = "PNG"
    dpi: int = Field(default=150, ge=72, le=300)


class MdSeriesRequest(BaseModel):
    series: dict[str, str] = Field(min_length=1, max_length=_MAX_SERIES)
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    format: Literal["PNG", "SVG", "TIFF", "JPEG"] = "PNG"


class TableRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=_MAX_ROWS)
    format: Literal["csv", "tsv", "xlsx", "md", "json", "html"] = "csv"
    filename: str = "table"


class ReportRequest(BaseModel):
    title: str = Field(default="PharmaML results", max_length=200)
    subtitle: str | None = Field(default=None, max_length=200)
    theme: str = "auto"
    metrics: dict[str, float | str] | None = None
    tables: list[dict[str, Any]] | None = Field(
        default=None, min_length=1, max_length=5
    )
    concentration_profile: dict[str, Any] | None = None
    dose_accumulation: dict[str, Any] | None = None
    stat: dict[str, Any] | None = None
    mode: Literal["job", "sync"] = "job"


@router.get("/catalog")
def catalog(request: Request) -> dict:
    return {
        "families": [
            {
                "family": "stat",
                "kinds": ["box", "histogram", "ecdf", "regression"],
                "backend": "pharmaml.viz.statplots",
            },
            {
                "family": "pharma-figures",
                "kinds": ["concentration-profile", "dose-accumulation"],
                "backend": "pharmaml.viz.pharma_plots",
            },
            {
                "family": "md-series",
                "kinds": ["one or more .xvg series"],
                "backend": "pharmaml.viz.md_plots",
            },
            {
                "family": "table",
                "formats": list(_TABLE_FORMATS),
                "backend": "pharmaml.viz.report.export_table",
            },
            {
                "family": "report",
                "kinds": ["html (metrics + tables + figures)"],
                "backend": "pharmaml.viz.report.HTMLReport",
            },
        ],
        "palette": list(PALETTE),
        "export_formats": list(_FIGURE_EXTENSIONS),
        "report_themes": list(REPORT_THEMES),
    }


@router.post("/stat")
def stat(req: StatRequest, request: Request) -> dict:
    color = req.color or PALETTE[0]
    store = request.app.state.artifact_store
    stats: dict | None = None
    artifact: dict | None = None
    n_points = 0
    extension = _FIGURE_EXTENSIONS[req.format]
    try:
        with tempfile.TemporaryDirectory(prefix="pharmaml-viz-") as tmp:
            # statplots computes the stats AND (when a path is given)
            # writes the figure through its own save_figure, closing it
            path = (
                Path(tmp) / f"stat{extension}" if req.with_artifact else None
            )
            if req.kind == "regression":
                if req.x is None or req.y is None:
                    raise HTTPException(
                        status_code=422,
                        detail="regression needs both x=[...] and y=[...]",
                    )
                if len(req.x) != len(req.y):
                    raise HTTPException(
                        status_code=422,
                        detail=f"x has {len(req.x)} points; y has {len(req.y)}",
                    )
                stats = plot_regression(
                    req.x, req.y, path, n_boot=req.n_boot, seed=req.seed,
                    color=color,
                )
                n_points = len(req.x)
            else:
                if req.values is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"{req.kind} needs values=[...]",
                    )
                if req.kind == "box":
                    stats = plot_box(req.values, path, whis=req.whis, color=color)
                elif req.kind == "histogram":
                    stats = plot_histogram(
                        req.values, path, bins=req.bins, color=color
                    )
                else:
                    stats = plot_ecdf(req.values, path, color=color)
                n_points = len(req.values)
            if path is not None and path.exists():
                record = store.register(
                    kind="figure",
                    filename=f"stat-{req.kind}{extension}",
                    content_type=_CONTENT_TYPES[extension],
                    data=path.read_bytes(),
                    meta={"family": "statplots", "kind": req.kind},
                )
                artifact = record.view()
    except DependencyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "kind": req.kind,
        "n_points": int(n_points),
        "stats": json_safe(stats),
        "artifact": artifact,
    }


@router.post("/concentration-profile")
def concentration_profile(req: ConcentrationProfileRequest, request: Request) -> dict:
    try:
        req._equal(req.times, req.conc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store = request.app.state.artifact_store
    try:
        figure = plot_concentration_profile(
            req.times, req.conc, auc=req.auc, title=req.title
        )
    except (DataValidationError, DependencyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    view = _export_and_register(
        store, figure, "concentration-profile",
        format=req.format, dpi=req.dpi, meta={"n_points": len(req.times)},
    )
    close_figure(figure)
    return {"artifact": view, "n_points": len(req.times)}


@router.post("/dose-accumulation")
def dose_accumulation(req: DoseAccumulationRequest, request: Request) -> dict:
    if (req.t_half is None) == (req.ke is None):
        raise HTTPException(
            status_code=422,
            detail="give exactly one of t_half / ke",
        )
    store = request.app.state.artifact_store
    try:
        figure = plot_dose_accumulation(
            req.dose, req.tau, t_half=req.t_half, ke=req.ke, vd=req.vd,
            f=req.f, n_doses=req.n_doses, title=req.title,
        )
    except (DataValidationError, DependencyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    view = _export_and_register(
        store, figure, "dose-accumulation",
        format=req.format, dpi=req.dpi,
        meta={"tau": req.tau, "n_doses": req.n_doses},
    )
    close_figure(figure)
    return {"artifact": view, "tau": req.tau, "n_doses": req.n_doses}


@router.post("/md-series")
def md_series(req: MdSeriesRequest, request: Request) -> dict:
    from pharmaml.md.xvg import metric_stats, parse_xvg

    store = request.app.state.artifact_store
    series_map = {}
    for name, text in req.series.items():
        try:
            series_map[name] = parse_xvg(text)
        except Exception as exc:  # MDAnalysisError -> honest 422
            raise HTTPException(
                status_code=422,
                detail=f"series {name!r}: {exc}",
            ) from exc
    extension = _FIGURE_EXTENSIONS[req.format]
    try:
        with tempfile.TemporaryDirectory(prefix="pharmaml-viz-") as tmp:
            target = Path(tmp) / f"md-series{extension}"
            stats = plot_md_series(
                series_map, target, title=req.title,
                x_label=req.x_label, y_label=req.y_label,
            )
            data = target.read_bytes()
    except DependencyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = store.register(
        kind="figure",
        filename=f"md-series{extension}",
        content_type=_CONTENT_TYPES[extension],
        data=data,
        meta={
            "series": list(req.series.keys()),
            "n_series": len(series_map),
        },
    )
    return {
        "n_series": len(series_map),
        "series": {
            name: {
                "n_points": len(series),
                "label": series.label,
                "stats": json_safe(metric_stats(series.values)),
            }
            for name, series in series_map.items()
        },
        "plot": json_safe(stats) if isinstance(stats, dict) else None,
        "artifact": record.view(),
    }


@router.post("/table")
def table(req: TableRequest, request: Request) -> dict:
    store = request.app.state.artifact_store
    frame = pd.DataFrame(req.rows)
    extension = f".{req.format}"
    try:
        with tempfile.TemporaryDirectory(prefix="pharmaml-viz-") as tmp:
            target = Path(tmp) / f"table{extension}"
            export_table(frame, target)
            data = target.read_bytes()
    except (DataValidationError, DependencyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    base = Path(req.filename).stem or "table"
    record = store.register(
        kind="table",
        filename=f"{base}{extension}",
        content_type=_CONTENT_TYPES[extension],
        data=data,
        meta={
            "rows": len(frame),
            "columns": int(frame.shape[1]),
            "column_names": [str(c) for c in frame.columns],
        },
    )
    return {"artifact": record.view(), "rows": len(frame)}


def _report_body(
    title: str,
    subtitle: str | None,
    theme: str,
    metrics: dict | None,
    tables: list[dict[str, Any]] | None,
    concentration_profile: dict[str, Any] | None,
    dose_accumulation: dict[str, Any] | None,
    stat: dict[str, Any] | None,
    artifact_store=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: one REAL self-contained HTML report artifact."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the report stage.")

    def _report(pct: int, stage: str) -> None:
        if manager is not None and job_id is not None:
            manager.report_progress(
                job_id, pct, stage, event=EventType.FILE_PROGRESS
            )

    stages: list[str] = []
    report = HTMLReport(title, subtitle=subtitle, theme=theme)
    if metrics:
        report.add_metrics(metrics)
        stages.append(f"metrics ({len(metrics)} entries)")
    _report(20, "metrics section done")
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled after the metrics section.")
    if tables:
        for spec in tables:
            frame = pd.DataFrame(spec.get("rows", []))
            if frame.empty:
                continue
            report.add_table(frame, heading=str(spec.get("heading", "table")))
        stages.append(f"tables ({len(tables)})")
    _report(40, "table sections done")
    figures = 0
    if concentration_profile:
        figure = plot_concentration_profile(
            concentration_profile.get("times", []),
            concentration_profile.get("conc", []),
            title=str(concentration_profile.get("title", "Concentration profile")),
        )
        report.add_figure(
            figure, caption=str(concentration_profile.get("caption", "profile"))
        )
        close_figure(figure)
        figures += 1
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled after the table sections.")
    if dose_accumulation:
        figure = plot_dose_accumulation(
            float(dose_accumulation.get("dose", 1.0)),
            float(dose_accumulation.get("tau", 12.0)),
            t_half=dose_accumulation.get("t_half"),
            ke=dose_accumulation.get("ke"),
            vd=float(dose_accumulation.get("vd", 50.0)),
            f=float(dose_accumulation.get("f", 1.0)),
            n_doses=int(dose_accumulation.get("n_doses", 6)),
            title=str(dose_accumulation.get("title", "Dose accumulation")),
        )
        report.add_figure(
            figure, caption=str(dose_accumulation.get("caption", "accumulation"))
        )
        close_figure(figure)
        figures += 1
    if stat:
        values = stat.get("values", [])
        with tempfile.TemporaryDirectory(prefix="pharmaml-viz-") as tmp:
            path = Path(tmp) / "stat.png"
            if stat.get("kind") == "box":
                plot_box(values, path)
            elif stat.get("kind") == "histogram":
                plot_histogram(values, path)
            elif stat.get("kind") == "ecdf":
                plot_ecdf(values, path)
            elif stat.get("kind") == "regression":
                plot_regression(stat.get("x", []), stat.get("y", []), path)
            if path.exists():
                report.add_image(
                    path, caption=str(stat.get("caption", "stat panel"))
                )
                figures += 1
    stages.append(f"figures ({figures})")
    _report(80, "figure sections done")
    if artifact_store is None:
        return {"title": title, "stages": stages, "figures": figures}
    with tempfile.TemporaryDirectory(prefix="pharmaml-viz-") as tmp:
        target = Path(tmp) / "report.html"
        report.save(target)
        data = target.read_bytes()
    record = artifact_store.register(
        kind="report",
        filename=f"{Path(title).stem or 'report'}.html",
        content_type="text/html; charset=utf-8",
        data=data,
        meta={"title": title, "figures": figures, "stages": stages},
    )
    _report(95, "report artifact registered")
    return {
        "title": title,
        "stages": stages,
        "figures": figures,
        "artifact": record.view(),
    }


@router.post("/report")
def report(req: ReportRequest, request: Request) -> dict:
    if req.theme not in REPORT_THEMES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown theme {req.theme!r}; see GET /api/v1/visualization/catalog",
        )
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_report_body(
                req.title, req.subtitle, req.theme, req.metrics, req.tables,
                req.concentration_profile, req.dose_accumulation, req.stat,
                artifact_store=request.app.state.artifact_store,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "visualization.report",
        "imagelab",
        _report_body,
        configuration={
            "title": req.title,
            "theme": req.theme,
            "metrics": bool(req.metrics),
            "tables": len(req.tables or []),
            "concentration_profile": bool(req.concentration_profile),
            "dose_accumulation": bool(req.dose_accumulation),
            "stat": bool(req.stat),
        },
        title=req.title,
        subtitle=req.subtitle,
        theme=req.theme,
        metrics=req.metrics,
        tables=req.tables,
        concentration_profile=req.concentration_profile,
        dose_accumulation=req.dose_accumulation,
        stat=req.stat,
        artifact_store=request.app.state.artifact_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}

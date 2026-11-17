"""``/api/v1/md`` — the Molecular Dynamics workspace's API surface
over the REAL backend (part10's row four).

All analysis is the toolkit's own in-memory trajectory layer (numpy
only, no file dependencies for the demo path):

* ``GET /api/v1/md/catalog`` — the route inventory.
* ``POST /api/v1/md/synthetic`` —
  :func:`pharmaml.md.trajectory.synthetic_trajectory` (the seeded
  deterministic demo: alpha-helix + flexible loop) followed by the
  REAL metric suite — :func:`rmsd_trajectory`, :func:`rmsf`,
  :func:`radius_of_gyration_trajectory`,
  :func:`native_contact_fraction`, :func:`contact_map` occupancy
  summary and :func:`pharmaml.md.dimred.trajectory_pca`.
* ``POST /api/v1/md/analyze`` — the same metric bundle over inline
  coordinates (``frames x atoms x 3``).
* ``POST /api/v1/md/xvg`` — :func:`pharmaml.md.xvg.parse_xvg` over
  pasted GROMACS ``.xvg`` text with ``metric_stats`` +
  ``moving_average``.
* ``POST /api/v1/md/enm`` — :func:`pharmaml.md.enm.anm_fluctuations`.
* ``POST /api/v1/md/pca`` — standalone
  :func:`pharmaml.md.dimred.trajectory_pca`.
* ``POST /api/v1/md/study`` — the full bundle as a managed job:
  the synthetic trajectory, the metric suite, and two REAL figures
  (RMSD + Rg rendered by :func:`pharmaml.viz.md_plots.plot_md_series`)
  registered as downloadable artifacts; every stage reports honest
  ``job.progress`` events and cancelling between stages lands the
  job CANCELLED, never FAILED.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md.dimred import trajectory_pca
from pharmaml.md.enm import anm_fluctuations
from pharmaml.md.metrics import (
    contact_map,
    native_contact_fraction,
    radius_of_gyration_trajectory,
    rmsd_trajectory,
    rmsf,
)
from pharmaml.md.trajectory import Topology, Trajectory, synthetic_trajectory
from pharmaml.md.xvg import XvgSeries, metric_stats, moving_average, parse_xvg
from pharmaml.viz.md_plots import plot_md_series

from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/md", tags=["md"])

_MAX_FRAMES = 2000
_MAX_ATOMS = 300
_MAX_COORD_ELEMENTS = 240_000
_MAX_XVG_POINTS = 10000


class SyntheticRequest(BaseModel):
    n_frames: int = Field(default=80, ge=1, le=_MAX_FRAMES)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    n_residues: int = Field(default=10, ge=1, le=100)
    noise: float = Field(default=0.25, ge=0.0, le=10.0)
    drift_radians: float = Field(default=0.05, ge=0.0, le=3.0)
    flexible_residues: list[int] | None = Field(
        default=None, max_length=50
    )


class MDAnalyzeRequest(BaseModel):
    # renamed for a unique OpenAPI component name — see the note in
    # routes/applicability_domain.py
    coords: list[list[list[float]]] = Field(min_length=1, max_length=_MAX_FRAMES)
    masses: list[float] | None = Field(default=None, max_length=_MAX_ATOMS)


class XvgRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    moving_average_window: int = Field(default=3, ge=1, le=101)


class EnmRequest(BaseModel):
    coords: list[list[float]] = Field(min_length=3, max_length=2000)
    cutoff: float = Field(default=13.0, gt=0.0, le=100.0)
    gamma: float = Field(default=1.0, gt=0.0, le=1e6)


class PcaRequest(BaseModel):
    coords: list[list[list[float]]] = Field(min_length=2, max_length=_MAX_FRAMES)
    n_components: int = Field(default=2, ge=1, le=10)


class StudyRequest(BaseModel):
    n_frames: int = Field(default=60, ge=2, le=_MAX_FRAMES)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    n_residues: int = Field(default=10, ge=1, le=60)
    noise: float = Field(default=0.25, ge=0.0, le=10.0)
    drift_radians: float = Field(default=0.05, ge=0.0, le=3.0)
    title: str = "MD study"
    mode: Literal["job", "sync"] = "job"


@router.get("/catalog")
def catalog(request: Request) -> dict:
    return {
        "groups": [
            {"group": "trajectory", "routes": ["/synthetic", "/analyze"],
             "backend": "pharmaml.md.trajectory + pharmaml.md.metrics"},
            {"group": "xvg", "routes": ["/xvg"], "backend": "pharmaml.md.xvg"},
            {"group": "enm", "routes": ["/enm"], "backend": "pharmaml.md.enm"},
            {"group": "dimred", "routes": ["/pca"], "backend": "pharmaml.md.dimred"},
            {"group": "study", "routes": ["/study"],
             "backend": "metrics + viz.md_plots (figure artifacts)"},
        ]
    }


def _topology_for(n_atoms: int) -> Topology:
    return Topology(atom_names=[f"A{i}" for i in range(n_atoms)])


def _metric_bundle(traj: Trajectory) -> dict[str, Any]:
    """The REAL metric suite over one trajectory."""
    rmsd = rmsd_trajectory(traj)
    per_atom_rmsf = rmsf(traj)
    rg = radius_of_gyration_trajectory(traj)
    q = native_contact_fraction(traj)
    cmap = contact_map(traj)
    n_pairs = cmap.shape[0] * max(0, cmap.shape[1] - 1) // 2
    strong_pairs = int((cmap[np.triu_indices(cmap.shape[0], k=1)] >= 0.5).sum())
    pca = trajectory_pca(traj, n_components=2)
    projections = pca.transform_trajectory(traj)
    return {
        "n_frames": int(traj.n_frames),
        "n_atoms": int(traj.n_atoms),
        "rmsd": json_safe([float(v) for v in rmsd]),
        "rmsf": json_safe([float(v) for v in per_atom_rmsf]),
        "radius_of_gyration": json_safe([float(v) for v in rg]),
        "native_contact_fraction": json_safe([float(v) for v in q]),
        "contact_map": {
            "n_pairs": int(n_pairs),
            "strong_pairs": strong_pairs,
            "strong_pair_fraction": float(strong_pairs / n_pairs) if n_pairs else 0.0,
        },
        "pca": {
            "method": pca.method,
            "explained_variance_ratio": json_safe(
                [float(v) for v in pca.explained_variance_ratio]
            ),
            "projections": json_safe(
                [[float(v) for v in row] for row in np.asarray(projections)]
            ),
        },
    }


@router.post("/synthetic")
def synthetic(req: SyntheticRequest, request: Request) -> dict:
    kwargs: dict[str, Any] = {
        "n_frames": req.n_frames,
        "seed": req.seed,
        "n_residues": req.n_residues,
        "noise": req.noise,
        "drift_radians": req.drift_radians,
    }
    if req.flexible_residues is not None:
        if not req.flexible_residues:
            raise HTTPException(
                status_code=422,
                detail="flexible_residues must be non-empty when given",
            )
        kwargs["flexible_residues"] = tuple(req.flexible_residues)
    traj = synthetic_trajectory(**kwargs)
    return {"source": "synthetic", **_metric_bundle(traj)}


@router.post("/analyze")
def analyze(req: MDAnalyzeRequest, request: Request) -> dict:
    try:
        coords = np.asarray(req.coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise HTTPException(
            status_code=422,
            detail="coords must be a frames x atoms x 3 array",
        )
    n_frames, n_atoms = coords.shape[0], coords.shape[1]
    if n_atoms > _MAX_ATOMS or coords.size > _MAX_COORD_ELEMENTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"coords too large for the inline path ({n_frames} x {n_atoms}); "
                f"limits: atoms <= {_MAX_ATOMS}, elements <= {_MAX_COORD_ELEMENTS}"
            ),
        )
    masses = (
        np.asarray(req.masses, dtype=float)
        if req.masses is not None
        else None
    )
    if masses is not None and len(masses) != n_atoms:
        raise HTTPException(
            status_code=422,
            detail=f"masses has {len(masses)} entries; the trajectory has {n_atoms} atoms",
        )
    try:
        traj = Trajectory(
            coords=coords,
            topology=Topology(
                atom_names=[f"A{i}" for i in range(n_atoms)],
                masses=masses,
            ),
        )
        return {"source": "inline", **_metric_bundle(traj)}
    except MDAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/xvg")
def xvg(req: XvgRequest, request: Request) -> dict:
    try:
        series = parse_xvg(req.text)
    except MDAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    n = len(series)
    values = series.values
    smoothed = (
        moving_average(values, req.moving_average_window)
        if req.moving_average_window <= n
        else None
    )
    return {
        "source": series.source,
        "label": series.label,
        "n_points": int(n),
        "time": json_safe([float(v) for v in series.time]),
        "values": json_safe([float(v) for v in values]),
        "stats": json_safe(metric_stats(values)),
        "moving_average": (
            json_safe([float(v) for v in smoothed])
            if smoothed is not None
            else None
        ),
    }


@router.post("/enm")
def enm(req: EnmRequest, request: Request) -> dict:
    try:
        coords = np.asarray(req.coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise HTTPException(
            status_code=422, detail="coords must be an atoms x 3 array"
        )
    try:
        fluctuations = anm_fluctuations(coords, req.cutoff, req.gamma)
    except MDAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_atoms": int(coords.shape[0]),
        "cutoff": req.cutoff,
        "gamma": req.gamma,
        "fluctuations": json_safe([float(v) for v in fluctuations]),
    }


@router.post("/pca")
def pca(req: PcaRequest, request: Request) -> dict:
    try:
        coords = np.asarray(req.coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise HTTPException(
            status_code=422,
            detail="coords must be a frames x atoms x 3 array",
        )
    traj = Trajectory(coords=coords, topology=_topology_for(coords.shape[1]))
    try:
        result = trajectory_pca(traj, n_components=req.n_components)
    except MDAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    projections = np.asarray(result.transform_trajectory(traj))
    return {
        "method": result.method,
        "n_frames": int(traj.n_frames),
        "n_components": int(req.n_components),
        "explained_variance_ratio": json_safe(
            [float(v) for v in result.explained_variance_ratio]
        ),
        "projections": json_safe(
            [[float(v) for v in row] for row in projections]
        ),
    }


def _study_body(
    n_frames: int,
    seed: int,
    n_residues: int,
    noise: float,
    drift_radians: float,
    title: str,
    artifact_store=None,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the full REAL metric suite + two figure artifacts."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the trajectory stage.")

    def _report(pct: int, stage: str) -> None:
        if manager is not None and job_id is not None:
            manager.report_progress(job_id, pct, stage)

    _report(10, "building the synthetic trajectory")
    traj = synthetic_trajectory(
        n_frames=n_frames, seed=seed, n_residues=n_residues,
        noise=noise, drift_radians=drift_radians,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled after the trajectory stage.")
    _report(30, "computing RMSD / RMSF / Rg")
    rmsd = rmsd_trajectory(traj)
    per_atom_rmsf = rmsf(traj)
    rg = radius_of_gyration_trajectory(traj)
    _report(55, "computing contact map + PCA")
    q = native_contact_fraction(traj)
    pca = trajectory_pca(traj, n_components=2)
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled after the metric stage.")
    metrics = {
        "n_frames": int(traj.n_frames),
        "n_atoms": int(traj.n_atoms),
        "rmsd_first": float(rmsd[0]),
        "rmsd_last": float(rmsd[-1]),
        "rmsf_max": float(np.max(per_atom_rmsf)),
        "rg_first": float(rg[0]),
        "rg_last": float(rg[-1]),
        "native_contact_fraction_last": float(q[-1]),
        "pca_explained_variance_ratio": [
            float(v) for v in pca.explained_variance_ratio
        ],
    }
    _report(75, "rendering the figures")
    artifacts: list[dict] = []
    if artifact_store is not None:
        time_axis = np.arange(traj.n_frames, dtype=float)
        panels = {
            "rmsd": XvgSeries(time_axis, rmsd, "rmsd.xvg", "RMSD (nm->A)"),
            "rg": XvgSeries(time_axis, rg, "gyrate.xvg", "Radius of gyration"),
        }
        with tempfile.TemporaryDirectory(prefix="pharmaml-md-") as tmp:
            for name, series in panels.items():
                plot_md_series(
                    series, Path(tmp) / f"{name}.png", title=f"{title} — {name}"
                )
                data = (Path(tmp) / f"{name}.png").read_bytes()
                record = artifact_store.register(
                    kind="figure",
                    filename=f"md-{name}.png",
                    content_type="image/png",
                    data=data,
                    meta={"panel": name, "n_frames": traj.n_frames, "seed": seed},
                )
                artifacts.append(record.view())
    _report(95, "registering the artifacts")
    return {"title": title, "seed": seed, "metrics": metrics, "artifacts": artifacts}


@router.post("/study")
def study(req: StudyRequest, request: Request) -> dict:
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_study_body(
                req.n_frames, req.seed, req.n_residues, req.noise,
                req.drift_radians, req.title,
                artifact_store=request.app.state.artifact_store,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "md.study",
        "molecular_dynamics",
        _study_body,
        configuration={
            "n_frames": req.n_frames,
            "n_residues": req.n_residues,
            "seed": req.seed,
            "noise": req.noise,
            "drift_radians": req.drift_radians,
            "title": req.title,
        },
        n_frames=req.n_frames,
        seed=req.seed,
        n_residues=req.n_residues,
        noise=req.noise,
        drift_radians=req.drift_radians,
        title=req.title,
        artifact_store=request.app.state.artifact_store,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}

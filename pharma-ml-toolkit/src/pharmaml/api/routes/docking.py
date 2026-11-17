"""``/api/v1/docking`` — the Docking workspace's API surface over the
REAL backend (part10's row four).

* ``GET /api/v1/docking/engines`` — the engine catalog: what
  :func:`pharmaml.docking.workflow.locate_engine_binary` finds on this
  host, whether :func:`engine_argv` can actually launch it (the bundled
  ``vina``/``autogrid4``/``autodock4`` are Windows PE executables — on
  a non-Windows host without Wine the launch check fails honestly),
  plus the DOCKSTRING target/objective inventory.
* ``GET /api/v1/docking/dockstring/targets`` — the 58-target catalog
  (:func:`pharmaml.docking.dockstring.target_frame`); single-target
  view at ``/dockstring/targets/{name}``.
* ``POST /api/v1/docking/dockstring/objective`` —
  :func:`evaluate_objective` (F2 / promiscuous_PPAR / selective_JAK2)
  over precomputed docking scores.
* ``POST /api/v1/docking/receptor/prepare`` —
  :func:`pharmaml.docking.receptor.prepare_protein` on pasted PDBQT
  text; the cleaned receptor registers as a downloadable artifact.
* ``POST /api/v1/docking/ligand/pdbqt`` —
  :func:`pharmaml.docking.preparation.smiles_to_pdbqt` (RDKit ETKDG
  embedding + Meeko typing), the ligand PDBQT registered as artifact.
* ``POST /api/v1/docking/parse/vina`` /
  ``POST /api/v1/docking/parse/dlg`` — the REAL parsers
  (:func:`parse_vina_output`, :func:`parse_dlg_summary` +
  :func:`parse_dlg_cluster_table`) over pasted engine logs.
* ``POST /api/v1/docking/scoring/consensus`` /
  ``POST /api/v1/docking/scoring/ligand-efficiency`` —
  :func:`consensus_breakdown` / :func:`ligand_efficiency`.
* ``POST /api/v1/docking/contacts`` — :func:`compute_contacts` +
  :func:`plif_matrix` (protein-ligand interaction fingerprint).
* ``POST /api/v1/docking/pose-rmsd`` —
  :func:`pharmaml.docking.pose_rmsd.best_pose_rmsd` (symmetry-aware).
* ``POST /api/v1/docking/dock`` —
  :func:`pharmaml.docking.smiles_docking.dock_smiles_array` as a
  managed job: the backend's own ``progress(done, total, note)``
  callback drives ``job.progress`` events per finished ligand and
  returning ``False`` from the adapter when the job's cancel flag is
  set triggers the backend's native ``JobCancelled`` (partial results
  preserved) — the job lands CANCELLED, never FAILED.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from rdkit import Chem

from pharmaml.docking.contacts import (
    INTERACTION_CLASSES,
    ContactAtom,
    compute_contacts,
    plif_matrix,
)
from pharmaml.docking.dockstring import (
    DOCKSTRING_OBJECTIVES,
    DOCKSTRING_TARGETS,
    evaluate_objective,
    get_target,
    target_frame,
)
from pharmaml.docking.parsing import (
    parse_dlg_cluster_table,
    parse_dlg_summary,
    parse_vina_output,
)
from pharmaml.docking.pose_rmsd import best_pose_rmsd
from pharmaml.docking.preparation import smiles_to_pdbqt
from pharmaml.docking.receptor import prepare_protein
from pharmaml.docking.scoring import (
    consensus_breakdown,
    ligand_efficiency,
)
from pharmaml.docking.smiles_docking import dock_smiles_array
from pharmaml.docking.workflow import engine_argv, locate_engine_binary
from pharmaml.exceptions import (
    DataValidationError,
    DependencyError,
    DockingError,
    DockingParseError,
)
from pharmaml.exceptions import (
    JobCancelled as BackendJobCancelled,
)

from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/docking", tags=["docking"])

_MAX_RECEPTOR_TEXT = 4_000_000
_MAX_SMILES = 2000
_MAX_CONTACT_ATOMS = 400
_MAX_POSES = 200
_MAX_ROWS = 2000

_ENGINES = ("vina", "autogrid4", "autodock4")


def _engine_catalog() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    boundary: str | None = None
    for name in _ENGINES:
        path = locate_engine_binary(name, None)
        entry: dict[str, Any] = {
            "name": name,
            "bundled": path is not None,
            "binary": Path(path).name if path is not None else None,
            "launchable": False,
        }
        try:
            engine_argv(name, None)
            entry["launchable"] = True
        except DockingError as exc:
            if boundary is None:
                boundary = str(exc)
        entries.append(entry)
    for entry in entries:
        entry["boundary_note"] = (
            boundary
            if not entry["launchable"]
            else "engine resolves and launches on this host"
        )
    return entries


class ReceptorPrepareRequest(BaseModel):
    pdbqt_text: str = Field(min_length=1, max_length=_MAX_RECEPTOR_TEXT)
    filename: str = "receptor.pdbqt"
    remove_waters: bool = True
    keep_hetatm: bool = False
    chain: str | None = None
    strip_hydrogens: bool = True


class LigandPdbqtRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    name: str | None = Field(default=None, max_length=80)
    charge: int | None = Field(default=None, ge=-10, le=10)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ParseTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=_MAX_RECEPTOR_TEXT)


class ConsensusRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=_MAX_ROWS)
    columns: list[str] = Field(min_length=1, max_length=12)
    weights: list[float] | None = Field(default=None, min_length=1, max_length=12)
    higher_is_better: dict[str, bool] | None = None


class LigandEfficiencyRequest(BaseModel):
    affinity_kcal_mol: float
    num_heavy_atoms: int = Field(ge=1, le=500)


class ContactAtomSpec(BaseModel):
    serial: int = Field(ge=1, le=99999)
    name: str = Field(max_length=4)
    element: str = Field(max_length=2)
    resname: str = Field(max_length=3)
    chain: str = Field(max_length=1)
    resseq: int = Field(ge=0, le=99999)
    x: float
    y: float
    z: float
    autodock_type: str | None = Field(default=None, max_length=2)


class ContactsRequest(BaseModel):
    ligand: list[ContactAtomSpec] = Field(min_length=1, max_length=_MAX_CONTACT_ATOMS)
    receptor: list[ContactAtomSpec] = Field(min_length=1, max_length=_MAX_CONTACT_ATOMS)
    cutoff: float = Field(default=5.0, gt=0.0, le=20.0)


class PoseRmsdRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    coords_ref: list[list[float]] = Field(min_length=3, max_length=500)
    poses: list[list[list[float]]] = Field(min_length=1, max_length=_MAX_POSES)

    @staticmethod
    def _check_rows(coords: Any, what: str) -> None:
        if not coords:
            raise ValueError(f"{what} cannot be empty")
        width = len(coords[0])
        if width != 3:
            raise ValueError(f"{what} rows must be [x, y, z] triples")

    def _validate(self) -> None:
        self._check_rows(self.coords_ref, "coords_ref")
        n_atoms = len(self.coords_ref)
        for i, pose in enumerate(self.poses):
            self._check_rows(pose, f"poses[{i}]")
            if len(pose) != n_atoms:
                raise ValueError(
                    f"poses[{i}] has {len(pose)} atoms; the reference has {n_atoms}"
                )


class DockRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=_MAX_SMILES)
    receptor_pdbqt: str = Field(min_length=1, max_length=_MAX_RECEPTOR_TEXT)
    engine: Literal["vina", "autodock4"] = "vina"
    center: list[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    size: list[float] = Field(
        default=[22.5, 22.5, 22.5], min_length=3, max_length=3
    )
    exhaustiveness: int = Field(default=8, ge=1, le=64)
    num_modes: int = Field(default=9, ge=1, le=36)
    workers: int = Field(default=1, ge=1, le=16)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    timeout_s: float = Field(default=3600.0, gt=0.0, le=7200.0)
    mode: Literal["job", "sync"] = "job"


@router.get("/engines")
def engines(request: Request) -> dict:
    return {
        "engines": _engine_catalog(),
        "platform": sys.platform,
        "dockstring": {
            "n_targets": len(DOCKSTRING_TARGETS),
            "objectives": list(DOCKSTRING_OBJECTIVES),
        },
    }


@router.get("/dockstring/targets")
def dockstring_targets(request: Request) -> dict:
    frame = target_frame()
    return {
        "total": len(frame),
        "targets": json_safe(frame.to_dict(orient="records")),
    }


@router.get("/dockstring/targets/{name}")
def dockstring_target_view(name: str, request: Request) -> dict:
    target = get_target(name)  # DataValidationError -> honest 422
    return {
        "name": target.name,
        "center": [float(v) for v in target.center],
        "size": [float(v) for v in target.size],
        "config": target.config_text(),
    }


class ObjectiveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    scores: dict[str, float] = Field(default_factory=dict)
    qed: float = Field(ge=0.0, le=1.0)


@router.post("/dockstring/objective")
def dockstring_objective(req: ObjectiveRequest, request: Request) -> dict:
    value = evaluate_objective(req.name, req.scores, qed=req.qed)
    return {"objective": req.name, "scores": req.scores, "qed": req.qed, "value": float(value)}


@router.post("/receptor/prepare")
def receptor_prepare(req: ReceptorPrepareRequest, request: Request) -> dict:
    store = request.app.state.artifact_store
    try:
        with tempfile.TemporaryDirectory(prefix="pharmaml-dock-") as tmp:
            source = Path(tmp) / req.filename
            source.write_text(req.pdbqt_text)
            output = Path(tmp) / "receptor-clean.pdbqt"
            report = prepare_protein(
                source,
                output,
                remove_waters=req.remove_waters,
                keep_hetatm=req.keep_hetatm,
                chain=req.chain,
                strip_hydrogens=req.strip_hydrogens,
            )
            payload = report.as_dict()
            data = output.read_bytes()
    except (DockingError, DependencyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # the ephemeral tempdir paths are dead after the request: report
    # the caller's filename instead (no sandbox paths in payloads)
    payload["source"] = req.filename
    payload["output_pdbqt"] = "receptor-clean.pdbqt"
    record = store.register(
        kind="receptor-pdbqt",
        filename="receptor-clean.pdbqt",
        content_type="text/plain; charset=utf-8",
        data=data,
        meta={"input_atoms": report.input_atoms, "output_atoms": report.output_atoms},
    )
    return {"report": json_safe(payload), "artifact": record.view()}


@router.post("/ligand/pdbqt")
def ligand_pdbqt(req: LigandPdbqtRequest, request: Request) -> dict:
    store = request.app.state.artifact_store
    try:
        with tempfile.TemporaryDirectory(prefix="pharmaml-lig-") as tmp:
            target = smiles_to_pdbqt(
                req.smiles, tmp, name=req.name, charge=req.charge, seed=req.seed
            )
            data = target.read_bytes()
            filename = target.name
    except (DockingError, DependencyError, DataValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = store.register(
        kind="ligand-pdbqt",
        filename=filename,
        content_type="text/plain; charset=utf-8",
        data=data,
        meta={"smiles": req.smiles, "seed": req.seed},
    )
    return {
        "smiles": req.smiles,
        "name": req.name or filename,
        "size_bytes": len(data),
        "artifact": record.view(),
    }


@router.post("/parse/vina")
def vina_parse(req: ParseTextRequest, request: Request) -> dict:
    try:
        poses = parse_vina_output(req.text)
    except DockingParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_poses": len(poses),
        "poses": [
            {
                "mode": int(p.mode),
                "affinity": float(p.affinity),
                "rmsd_ub": float(p.rmsd_ub),
                "rmsd_lb": float(p.rmsd_lb),
            }
            for p in poses
        ],
    }


@router.post("/parse/dlg")
def dlg_parse(req: ParseTextRequest, request: Request) -> dict:
    try:
        summary = parse_dlg_summary(req.text)
        clusters = parse_dlg_cluster_table(req.text)
    except DockingParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "summary": json_safe(summary),
        "clusters": json_safe(
            [
                {
                    "rank": int(c.rank),
                    "lowest_energy": float(c.lowest_energy),
                    "mean_energy": float(c.mean_energy),
                    "num_in_cluster": int(c.num_in_cluster),
                }
                for c in clusters
            ]
        ),
    }


@router.post("/scoring/consensus")
def scoring_consensus(req: ConsensusRequest, request: Request) -> dict:
    if req.weights is not None and len(req.weights) != len(req.columns):
        raise HTTPException(
            status_code=422,
            detail="weights must align with the columns list",
        )
    frame = pd.DataFrame(req.rows)
    if req.weights is not None:
        req.weights = [float(w) for w in req.weights]
    try:
        breakdown = consensus_breakdown(
            frame,
            req.columns,
            weights=req.weights,
            higher_is_better=req.higher_is_better,
        )
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rows = breakdown.to_dict(orient="records")
    return {
        "columns": req.columns,
        "n_rows": len(breakdown),
        "rows": json_safe(rows),
    }


@router.post("/scoring/ligand-efficiency")
def scoring_ligand_efficiency(
    req: LigandEfficiencyRequest, request: Request
) -> dict:
    value = ligand_efficiency(req.affinity_kcal_mol, req.num_heavy_atoms)
    return {
        "affinity_kcal_mol": req.affinity_kcal_mol,
        "num_heavy_atoms": req.num_heavy_atoms,
        "ligand_efficiency": float(value),
    }


@router.post("/contacts")
def contacts(req: ContactsRequest, request: Request) -> dict:
    ligand = [
        ContactAtom(**spec.model_dump()) for spec in req.ligand
    ]
    receptor = [
        ContactAtom(**spec.model_dump()) for spec in req.receptor
    ]
    records = compute_contacts(ligand, receptor, cutoff=req.cutoff)
    plif = plif_matrix(records)
    plif_rows = []
    if len(records) > 0:
        for residue, row in zip(plif.index, plif.to_numpy()):
            entry: dict[str, Any] = {"residue": str(residue)}
            for cls, value in zip(INTERACTION_CLASSES, row[: len(INTERACTION_CLASSES)]):
                entry[cls] = int(value)
            entry["total"] = int(row[-1])
            plif_rows.append(entry)
    return {
        "cutoff": req.cutoff,
        "n_contacts": len(records),
        "interaction_classes": list(INTERACTION_CLASSES),
        "contacts": [
            {
                "ligand_serial": int(c.ligand_atom.serial),
                "ligand_atom": c.ligand_atom.name,
                "receptor_residue": c.receptor_atom.residue_key,
                "receptor_atom": c.receptor_atom.name,
                "distance": float(c.distance),
                "interaction": c.interaction,
            }
            for c in records[:_MAX_CONTACT_ATOMS]
        ],
        "plif": plif_rows,
    }


@router.post("/pose-rmsd")
def pose_rmsd(req: PoseRmsdRequest, request: Request) -> dict:
    try:
        req._validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(
            status_code=422, detail=f"invalid SMILES {req.smiles!r}"
        )
    import numpy as np

    ref = np.asarray(req.coords_ref, dtype=float)
    poses = [np.asarray(p, dtype=float) for p in req.poses]
    values = best_pose_rmsd(mol, ref, poses)
    return {
        "smiles": req.smiles,
        "n_atoms": int(mol.GetNumAtoms()),
        "n_poses": len(poses),
        "rmsd": [float(v) for v in values],
    }


def _dock_body(
    smiles: list[str],
    receptor_pdbqt: str,
    engine: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    workers: int,
    seed: int,
    timeout_s: float,
    manager=None,
    job_id: str | None = None,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL dock_smiles_array over a temp workspace."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the docking stage.")

    def progress_adapter(done: int, total: int, note: Any = None) -> bool:
        if manager is not None and job_id is not None:
            pct = min(99, int(100.0 * done / max(1, total)))
            stage = f"ligand {done}/{total}"
            if note:
                stage = f"{stage} ({note})"
            manager.report_progress(job_id, pct, stage)
        # returning False stops the batch at its own checkpoint: the
        # backend raises its native JobCancelled with partial results
        return not (cancel_event is not None and cancel_event.is_set())

    with tempfile.TemporaryDirectory(prefix="pharmaml-dock-run-") as tmp:
        receptor = Path(tmp) / "receptor.pdbqt"
        receptor.write_text(receptor_pdbqt)
        out_dir = Path(tmp) / "results"
        try:
            frame = dock_smiles_array(
                smiles,
                receptor,
                out_dir,
                engine=engine,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                workers=workers,
                seed=seed,
                timeout_s=timeout_s,
                keep_folders=False,
                progress=progress_adapter,
            )
        except BackendJobCancelled as exc:
            raise CancelledJob(str(exc)) from exc
        except DockingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    rows = frame.to_dict(orient="records")
    n_ok = sum(1 for r in rows if r.get("ok"))
    return {
        "n_ligands": len(rows),
        "n_ok": n_ok,
        "n_failed": len(rows) - n_ok,
        "rows": json_safe(rows),
    }


@router.post("/dock")
def dock(req: DockRequest, request: Request) -> dict:
    # pre-validate the engine launch up front (both modes): an
    # unlaunchable engine is an environmental boundary, not a job
    try:
        engine_argv("vina" if req.engine == "vina" else "autodock4", None)
    except DockingError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"docking engine unavailable on this host: {exc}",
        ) from exc
    center = (float(req.center[0]), float(req.center[1]), float(req.center[2]))
    size = (float(req.size[0]), float(req.size[1]), float(req.size[2]))
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_dock_body(
                req.smiles, req.receptor_pdbqt, req.engine, center, size,
                req.exhaustiveness, req.num_modes, req.workers, req.seed,
                req.timeout_s,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "docking.dock",
        "docking",
        _dock_body,
        configuration={
            "engine": req.engine,
            "n_smiles": len(req.smiles),
            "center": req.center,
            "size": req.size,
            "exhaustiveness": req.exhaustiveness,
            "num_modes": req.num_modes,
            "workers": req.workers,
            "seed": req.seed,
            "timeout_s": req.timeout_s,
        },
        smiles=req.smiles,
        receptor_pdbqt=req.receptor_pdbqt,
        engine=req.engine,
        center=center,
        size=size,
        exhaustiveness=req.exhaustiveness,
        num_modes=req.num_modes,
        workers=req.workers,
        seed=req.seed,
        timeout_s=req.timeout_s,
        manager=manager,
    )
    return {"mode": "job", "job": record.public()}

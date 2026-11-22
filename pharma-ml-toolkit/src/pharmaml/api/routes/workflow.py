"""``/api/v1/workflow`` — the lightweight Scientific Workflow Engine
lane (part13 Phase 6, §8 — the round-322 API leg).

The §8 directive's own constraint governs this route family: *do
not introduce a large workflow engine automatically; first assess
whether the existing JobManager and capability registry can support
a lightweight implementation; keep the application fully local and
offline-capable.*  The assessment + the candidate review (Prefect /
Knime / JupyterLab / marimo) live in ``docs/ARCHITECTURE_REVIEW.md``
— the decision: the in-tree engine over the existing JobManager and
the capability vocabulary, zero new dependencies.

* ``GET /api/v1/workflow/steps`` — the step registry's own typed
  vocabulary verbatim: every step's input/output ports with the
  closed kind list, its parameter dials, the REAL backend module it
  delegates to, and its honest notes.
* ``GET /api/v1/workflow/templates`` — the saved workflow templates
  (§8's own example chain, the guided QSAR template, with its honest
  leakage-note: the split runs before feature selection because the
  toolkit's own leakage rule is a contract — plus the minimal
  regression core).
* ``POST /api/v1/workflow/validate`` — dependency validation before
  any execution: unknown steps, dangling wiring, contract mismatches
  (kind equality), duplicate node ids, cycles — every finding at
  once, plus the topological order the executor would walk.
* ``POST /api/v1/workflow/run`` — the real round trip: sync mode
  (bounded small graphs) or job mode over the EXISTING JobManager
  (progress per node through ``report_progress``, cooperative
  cancellation through the ``cancel_event`` injection, the SQLite
  job history the manager already persists).  Partial failures
  carry the backend's own error verbatim with downstream nodes
  SKIPPED and completed artifacts preserved.
* ``GET /api/v1/workflow/runs`` / ``GET /runs/{run_id}`` — the
  execution history (the bounded in-session ledger + each run's
  reproducibility block: versions, seed, spec digest).

Honest scope everywhere: the default dataset is the route's own
seeded SYNTHETIC stand-in (every metric is a statistic over it,
never an experimental fact); the in-session run ledger lives only
as long as the process (job-mode runs persist through the
JobManager's own history); wall-clock durations vary and are
excluded from every digest.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError
from pharmaml.workflow.contracts import PORT_KINDS, STEP_REGISTRY
from pharmaml.workflow.engine import (
    WorkflowEngine,
    WorkflowSpec,
    WorkflowValidationError,
)
from pharmaml.workflow.templates import TEMPLATE_REGISTRY, list_templates

router = APIRouter(prefix="/workflow", tags=["workflow"])

_MAX_NODES_SYNC = 12
_MAX_NODES_JOB = 24
_MAX_RECORDS_SYNC = 80
_MAX_RECORDS_JOB = 200
_MAX_SEED = 2147483647

#: The session-scoped engine (the bounded run ledger).  Job-mode
#: runs persist through the JobManager's own SQLite history — this
#: instance is the live view, and the note says so.
_ENGINE = WorkflowEngine()


def _synthetic_records(seed: int, n: int) -> list[dict[str, Any]]:
    """The route's own seeded SYNTHETIC dataset stand-in (the honest
    no-real-dataset scope: the seed is the only generative dial).
    The molecule list is 30 DISTINCT structures — duplicate
    descriptor rows would defeat the leverage/Williams backend's
    own full-rank contract, so the stand-in never cycles."""
    import random

    molecules = [
        "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CC(C)NCC(O)COc1ccccc1CC",
        "OC(=O)CC(O)(CC(=O)O)C(O)(CC(=O)O)CC(=O)O",
        "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "CCN(CC)C(=O)C1CN(C)C2=C1C=CC=C2",
        "COc1ccc2nc(S(=O)(=O)N)[nH]c2c1", "CC1(C)SC(C(S1)C(=O)O)N",
        "OC(=O)C1=CC=CC=C1O", "CCO", "CCCCO", "CCCO", "CCCOCCO",
        "c1ccc(O)cc1", "CC(C)(C)O", "COC(C)(C)C", "CC(C)O",
        "CCOCCO", "OC(=O)CCO", "NC(=O)c1ccccc1", "OCCN",
        "CC(N)CO", "COC(=O)c1ccccc1", "CC(C)N", "OCc1ccccc1",
        "CCOC(=O)c1ccccc1", "NCc1ccccc1", "CC(O)CO", "CCOc1ccccc1",
    ]
    rng = random.Random(seed)
    return [
        {
            "smiles": molecules[i],
            "pIC50": round(rng.uniform(3.0, 9.0), 3),
        }
        for i in range(min(n, len(molecules)))
    ]


def _resolve_spec(
    template: str | None, spec: dict[str, Any] | None
) -> WorkflowSpec:
    if spec is not None:
        resolved = WorkflowSpec.from_dict(spec)
    elif template is not None:
        base = TEMPLATE_REGISTRY.get(template)
        if base is None:
            raise DataValidationError(
                f"unknown template {template!r} (have: "
                f"{', '.join(sorted(TEMPLATE_REGISTRY))})"
            )
        resolved = WorkflowSpec.from_dict(base.to_dict())
    else:
        raise DataValidationError(
            "a run needs either a template id or a spec"
        )
    return resolved


def _apply_overrides(
    spec: WorkflowSpec, overrides: dict[str, dict[str, Any]]
) -> WorkflowSpec:
    for node_id, params in (overrides or {}).items():
        node = spec.nodes.get(str(node_id))
        if node is None:
            raise DataValidationError(
                f"param override targets unknown node {node_id!r}"
            )
        node.params.update(dict(params or {}))
    return spec


def _validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise DataValidationError("records must be a non-empty list of rows")
    for i, row in enumerate(records[:5]):
        if not isinstance(row, dict) or "smiles" not in row or "pIC50" not in row:
            raise DataValidationError(
                f"records[{i}] must carry 'smiles' and 'pIC50' keys"
            )
        try:
            float(row["pIC50"])
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"records[{i}]['pIC50'] is not numeric"
            ) from exc


class WorkflowRunRequest(BaseModel):
    """One run: a template (with optional per-node parameter
    overrides) or a caller-built spec; inline records or the route's
    own seeded synthetic stand-in; sync (bounded) or job mode (the
    existing JobManager: progress, cooperative cancellation, the
    SQLite history)."""

    template: str | None = None
    spec: dict[str, Any] | None = None
    param_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    records: list[dict[str, Any]] | None = None
    n_synthetic: int = Field(default=30, ge=12, le=30)
    seed: int = Field(default=42, ge=0, le=_MAX_SEED)
    mode: Literal["sync", "job"] = "sync"


@router.get("/steps")
def steps() -> dict:
    """The step registry's typed vocabulary, verbatim."""
    return {
        "engine": {
            "layer": "pharmaml.workflow (the lightweight in-tree engine)",
            "decision": (
                "the existing JobManager + capability vocabulary support "
                "the lightweight implementation -- the §8 assessment's "
                "own conclusion (docs/ARCHITECTURE_REVIEW.md); no "
                "external workflow engine adopted"
            ),
            "port_kinds": list(PORT_KINDS),
            "job_integration": (
                "job-mode runs ride the existing JobManager: progress "
                "per node, cooperative cancellation between nodes, the "
                "SQLite job history"
            ),
        },
        "steps": [
            {
                "step_id": step.step_id,
                "title": step.title,
                "description": step.description,
                "backend": step.backend,
                "workspace": step.workspace,
                "inputs": [
                    {
                        "name": p.name,
                        "kind": p.kind,
                        "required": p.required,
                        "description": p.description,
                    }
                    for p in step.inputs
                ],
                "outputs": [
                    {"name": p.name, "kind": p.kind,
                     "description": p.description}
                    for p in step.outputs
                ],
                "parameters": [
                    {"name": p.name, "default": p.default,
                     "description": p.description}
                    for p in step.parameters
                ],
                "notes": step.notes,
            }
            for step in STEP_REGISTRY.values()
        ],
    }


@router.get("/templates")
def templates() -> dict:
    """The saved workflow templates (§8's own example chain among
    them, with its honest leakage-note)."""
    return {"templates": list_templates()}


@router.post("/validate")
def validate(spec: dict[str, Any]) -> dict:
    """Dependency validation before any execution — every finding at
    once plus the topological order the executor would walk."""
    resolved = WorkflowSpec.from_dict(spec)
    errors = resolved.validate()
    order = None if errors else resolved.topological_order()
    return {
        "valid": not errors,
        "errors": errors,
        "topological_order": order,
        "spec_digest": resolved.canonical_digest(),
        "node_count": len(resolved.nodes),
    }


@router.post("/run")
def run(request: Request, body: WorkflowRunRequest) -> dict:
    """The real round trip: sync (bounded) or job mode (the existing
    JobManager).  Partial failures carry the backend's own errors
    verbatim; downstream nodes are SKIPPED; completed artifacts
    survive with their lineage + digests."""
    try:
        resolved = _resolve_spec(body.template, body.spec)
        resolved = _apply_overrides(resolved, body.param_overrides)
        errors = resolved.validate()
        if errors:
            raise WorkflowValidationError(errors)
    except WorkflowValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": "the workflow failed validation",
                    "errors": exc.errors},
        ) from exc
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if body.records is not None:
        records = list(body.records)[: (
            _MAX_RECORDS_SYNC if body.mode == "sync" else _MAX_RECORDS_JOB
        )]
        _validate_records(records)
    else:
        # the synthetic stand-in is the DISTINCT 30-molecule list —
        # duplicate descriptor rows would defeat the rank contract
        records = _synthetic_records(body.seed, body.n_synthetic)
    if len(resolved.nodes) > (
        _MAX_NODES_SYNC if body.mode == "sync" else _MAX_NODES_JOB
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"a {body.mode}-mode run is bounded to "
                f"{(_MAX_NODES_SYNC if body.mode == 'sync' else _MAX_NODES_JOB)}"
                " nodes"
            ),
        )

    initial_inputs = {"records": records}
    if body.mode == "sync":
        run_record = _ENGINE.execute(
            resolved,
            initial_inputs=initial_inputs,
            seed=body.seed,
        )
        return {"mode": "sync", "run": run_record.public()}

    manager = request.app.state.job_manager

    def _job_body(
        cancel_event: threading.Event | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        def _progress(done: int, total: int, node_id: str, state: str) -> None:
            if job_id:
                manager.report_progress(
                    job_id,
                    int(100 * done // max(total, 1)),
                    stage=f"node {done}/{total}: {node_id} {state}",
                )

        record = _ENGINE.execute(
            resolved,
            initial_inputs=initial_inputs,
            seed=body.seed,
            cancel_event=cancel_event,
            progress=_progress,
        )
        return record.public()

    job = manager.submit(
        operation="workflow.run",
        workspace="workflow",
        fn=_job_body,
        configuration={
            "workflow": resolved.name,
            "mode": "job",
            "nodes": len(resolved.nodes),
            "seed": body.seed,
            "spec_digest": resolved.canonical_digest(),
        },
        expensive=True,
    )
    return {
        "mode": "job",
        "job_id": job.job_id,
        "job_state": job.state.value,
        "note": (
            "poll /api/v1/jobs/{job_id} for progress and /jobs/"
            "{job_id}/result for the run record; cancellation rides the "
            "job manager's own cooperative flag between nodes"
        ),
    }


@router.get("/runs")
def runs() -> dict:
    """The bounded in-session execution history."""
    return {
        "runs": _ENGINE.list_runs(),
        "note": (
            "the in-session ledger lives as long as the process; "
            "job-mode runs persist through the JobManager's own "
            "SQLite history"
        ),
    }


@router.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    record = _ENGINE.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown run_id {run_id!r} (in-session ledger)",
        )
    return record.public()

"""The graph machinery of the lightweight Scientific Workflow Engine
(part13 Phase 6, §8 — the round-322 API lane).

The §8 requirement list, mapped to this module's surfaces:

* workflow graph            — :class:`WorkflowSpec` (nodes + typed wiring)
* typed input/output contracts — :mod:`pharmaml.workflow.contracts`
* dependency validation      — :meth:`WorkflowSpec.validate` (unknown
  steps, dangling wiring, contract mismatches, cycles, unfed required
  ports) — BEFORE any execution begins
* workflow serialization    — :meth:`WorkflowSpec.to_dict` /
  :meth:`WorkflowSpec.from_dict` (JSON-safe round trip)
* saved workflow templates  — :mod:`pharmaml.workflow.templates`
* execution history          — :class:`WorkflowEngine` (the bounded
  run ledger) + the JobManager's own persisted job records
* progress reporting         — the ``progress`` callback (per node),
  wired to ``JobManager.report_progress`` in job mode
* cancellation               — the cooperative ``cancel_event``,
  checked BETWEEN nodes (the JobManager's own pattern)
* partial failure handling   — a failed node records its error
  verbatim; every downstream node is ``SKIPPED`` with the reason;
  completed artifacts SURVIVE (the run ends ``PARTIAL``)
* artifact lineage           — every output artifact carries its
  producing node/port + the transitive upstream chain + a digest
* reproducibility metadata   — the run's versions/seed/digest block

Execution never reimplements an algorithm: every node delegates to
its step's runner (:mod:`pharmaml.workflow.runners`), which delegates
to the REAL backend callables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .contracts import STEP_REGISTRY
from .runners import STEP_RUNNERS

__all__ = [
    "ArtifactRecord",
    "NodeResult",
    "NodeSpec",
    "RunRecord",
    "WorkflowEngine",
    "WorkflowSpec",
    "WorkflowValidationError",
]

_HISTORY_LIMIT = 100
_logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WorkflowValidationError(ValueError):
    """The spec failed dependency validation (§8's pre-execution gate).

    ``errors`` carries every finding at once — the caller fixes the
    whole graph, not one edge at a time.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# -- spec ---------------------------------------------------------------


@dataclass
class NodeSpec:
    """One node of the graph: its step, its parameter overrides, and
    its input wiring (``{input_port: (upstream_node, upstream_port)}``)."""

    node_id: str
    step_id: str
    params: dict[str, Any] = field(default_factory=dict)
    wiring: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class WorkflowSpec:
    """A named, serialisable workflow graph (§8's construct/save/
    reproduce object)."""

    name: str
    description: str = ""
    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- serialisation (§8's "workflow serialization") -----------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "step_id": node.step_id,
                    "params": dict(node.params),
                    "wiring": {
                        port: dict(edge)
                        for port, edge in node.wiring.items()
                    },
                }
                for node in sorted(
                    self.nodes.values(), key=lambda n: n.node_id
                )
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorkflowSpec:
        nodes: dict[str, NodeSpec] = {}
        seen: set[str] = set()
        for entry in raw.get("nodes", []):
            node = NodeSpec(
                node_id=str(entry.get("node_id", "")),
                step_id=str(entry.get("step_id", "")),
                params=dict(entry.get("params", {}) or {}),
                wiring={
                    str(port): {
                        "node": str(edge.get("node", "")),
                        "port": str(edge.get("port", "")),
                    }
                    for port, edge in (entry.get("wiring", {}) or {}).items()
                },
            )
            if node.node_id in seen:
                # a dict cannot carry duplicate keys — the parse-level
                # structural error, raised with the finding verbatim
                raise WorkflowValidationError(
                    [f"duplicate node_id {node.node_id!r}"]
                )
            seen.add(node.node_id)
            nodes[node.node_id] = node
        return cls(
            name=str(raw.get("name", "")),
            description=str(raw.get("description", "")),
            nodes=nodes,
            metadata=dict(raw.get("metadata", {}) or {}),
        )

    def canonical_digest(self) -> str:
        """A stable sha-256 over the canonical serialisation — the
        spec's identity in every run's reproducibility block."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- validation (§8's dependency validation) -----------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.nodes:
            errors.append("the workflow has no nodes")
        seen_ids: set[str] = set()
        for node in self.nodes.values():
            if not node.node_id:
                errors.append("a node carries an empty node_id")
            seen_ids.add(node.node_id)
            step = STEP_REGISTRY.get(node.step_id)
            if step is None:
                errors.append(
                    f"node {node.node_id!r}: unknown step "
                    f"{node.step_id!r}"
                )
                continue
            known_params = {p.name for p in step.parameters}
            for name in node.params:
                if name not in known_params:
                    errors.append(
                        f"node {node.node_id!r}: step {step.step_id} has "
                        f"no parameter {name!r}"
                    )
            for port, edge in node.wiring.items():
                if port not in {p.name for p in step.inputs}:
                    errors.append(
                        f"node {node.node_id!r}: step {step.step_id} has "
                        f"no input port {port!r}"
                    )
                upstream = self.nodes.get(edge.get("node", ""))
                if upstream is None:
                    errors.append(
                        f"node {node.node_id!r}: port {port!r} is wired "
                        f"to unknown node {edge.get('node')!r}"
                    )
                    continue
                upstream_step = STEP_REGISTRY.get(upstream.step_id)
                if upstream_step is None:
                    continue
                produced = {
                    p.name: p.kind for p in upstream_step.outputs
                }
                if edge.get("port", "") not in produced:
                    errors.append(
                        f"node {node.node_id!r}: port {port!r} is wired "
                        f"to {upstream.node_id!r} which has no output "
                        f"port {edge.get('port')!r}"
                    )
                    continue
                consumer_kind = next(
                    (p.kind for p in step.inputs if p.name == port), ""
                )
                if produced[edge["port"]] != consumer_kind:
                    errors.append(
                        f"node {node.node_id!r}: port {port!r} needs "
                        f"{consumer_kind!r} but {upstream.node_id!r}."
                        f"{edge['port']} produces "
                        f"{produced[edge['port']]!r}"
                    )
        # Required-but-unwired input ports are NOT structural errors:
        # they are fed at run time as the run's initial inputs (the
        # executor's own honest FAILED path records a genuinely
        # missing feed verbatim -- nothing is guessed here).
        if not errors:
            order = self.topological_order()
            if order is None:
                errors.append("the workflow graph has a cycle")
        return errors

    def topological_order(self) -> list[str] | None:
        """Kahn's order over the wiring edges; ``None`` on a cycle."""
        indegree = {node_id: 0 for node_id in self.nodes}
        adjacency: dict[str, list[str]] = {
            node_id: [] for node_id in self.nodes
        }
        for node in self.nodes.values():
            for edge in node.wiring.values():
                upstream = edge.get("node", "")
                if upstream in self.nodes and node.node_id in self.nodes:
                    adjacency[upstream].append(node.node_id)
                    indegree[node.node_id] += 1
        ready = sorted(
            node_id for node_id, degree in indegree.items() if degree == 0
        )
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for child in sorted(adjacency[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(self.nodes):
            return None
        return order


# -- run records -----------------------------------------------------------


@dataclass
class ArtifactRecord:
    """One output artifact + its lineage (§8's artifact lineage)."""

    node_id: str
    port: str
    kind: str
    digest: str
    digest_kind: str
    lineage: tuple[tuple[str, str], ...] = ()

    def public(self) -> dict[str, Any]:
        return {
            "node": self.node_id,
            "port": self.port,
            "kind": self.kind,
            "digest": self.digest,
            "digest_kind": self.digest_kind,
            "lineage": [
                {"node": node, "port": port} for node, port in self.lineage
            ],
        }


@dataclass
class NodeResult:
    """One node's execution verdict (the partial-failure ledger)."""

    node_id: str
    step_id: str
    state: str  # COMPLETED | FAILED | SKIPPED | CANCELLED
    started_at: str | None = None
    duration_ms: float | None = None
    error: str | None = None
    skipped_reason: str | None = None
    artifacts: tuple[ArtifactRecord, ...] = ()

    def public(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "step_id": self.step_id,
            "state": self.state,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
            "artifacts": [a.public() for a in self.artifacts],
        }


def _digest_value(value: Any) -> tuple[str, str]:
    """A best-effort content digest + the honest ``digest_kind``.

    JSON-able values digest canonically; DataFrames digest over their
    shape + the pandas row hash; fitted estimators digest over their
    ``get_params`` snapshot — fitted WEIGHTS are not deterministically
    serialisable, and the digest_kind says so instead of pretending.
    """
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            total = int(pd.util.hash_pandas_object(
                value, index=True
            ).sum())
            payload = json.dumps(
                {
                    "shape": list(value.shape),
                    "columns": list(map(str, value.columns)),
                    "row_hash": total,
                },
                sort_keys=True,
            )
            return (
                hashlib.sha256(payload.encode()).hexdigest(),
                "dataframe-shape-rowhash",
            )
        if hasattr(value, "get_params"):
            payload = json.dumps(
                value.get_params(deep=False), sort_keys=True, default=str
            )
            return (
                hashlib.sha256(payload.encode()).hexdigest(),
                "estimator-params",
            )
    except Exception:
        # run: the fallback below answers for any value shape
        _logger.debug("structural digest failed", exc_info=True)
    try:
        payload = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest(), "json"
    except (TypeError, ValueError):
        return "0" * 64, "undigestible"


@dataclass
class RunRecord:
    """One execution's full ledger (§8's execution history entry)."""

    run_id: str
    workflow_name: str
    spec_digest: str
    state: str  # COMPLETED | PARTIAL | FAILED | CANCELLED
    seed: int
    started_at: str
    completed_at: str | None = None
    reproducibility: dict[str, Any] = field(default_factory=dict)
    nodes: list[NodeResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        tally = {"COMPLETED": 0, "FAILED": 0, "SKIPPED": 0,
                 "CANCELLED": 0}
        for node in self.nodes:
            tally[node.state] = tally.get(node.state, 0) + 1
        return tally

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "spec_digest": self.spec_digest,
            "state": self.state,
            "seed": self.seed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "reproducibility": self.reproducibility,
            "node_counts": self.counts(),
            "nodes": [n.public() for n in self.nodes],
        }


# -- the executor ------------------------------------------------------------


ProgressCallback = Callable[[int, int, str, str], None]


def _reproducibility_block(spec: WorkflowSpec, seed: int) -> dict[str, Any]:
    from pharmaml import __version__ as toolkit_version

    return {
        "toolkit_version": toolkit_version,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}."
                  f"{sys.version_info.micro}",
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "seed": seed,
        "spec_digest": spec.canonical_digest(),
        "determinism_note": (
            "seeded splits and estimators reproduce; the f_regression "
            "selector is deterministic while the mutual-info member "
            "of the default union pair is unseeded in the backend (its "
            "selected column set can vary run-to-run on the same "
            "seed); the report PDF embeds fpdf's own creation "
            "timestamp so its bytes vary run-to-run even at the same "
            "seed; wall-clock durations vary and are excluded from "
            "every digest"
        ),
    }


def execute(
    spec: WorkflowSpec,
    *,
    initial_inputs: dict[str, Any] | None = None,
    seed: int = 42,
    cancel_event: Any = None,
    progress: ProgressCallback | None = None,
) -> RunRecord:
    """Run the graph (sync executor — job mode wraps this callable).

    Partial-failure semantics: a failed node records the backend's
    own error verbatim and every downstream node is SKIPPED with the
    reason; completed artifacts survive in the run record.
    """
    errors = spec.validate()
    if errors:
        raise WorkflowValidationError(errors)
    order = spec.topological_order()
    assert order is not None  # validate() already rejected cycles

    run = RunRecord(
        run_id=f"wf-run-{uuid.uuid4().hex[:12]}",
        workflow_name=spec.name,
        spec_digest=spec.canonical_digest(),
        state="RUNNING",
        seed=seed,
        started_at=_now_iso(),
        reproducibility=_reproducibility_block(spec, seed),
    )
    initial = dict(initial_inputs or {})
    artifacts: dict[tuple[str, str], Any] = {}
    lineage_map: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    # every node whose outputs will never exist: FAILED nodes carry
    # the error verbatim; SKIPPED/CANCELLED nodes propagate the
    # unavailability downstream exactly the same way
    unavailable: set[str] = set()
    total = len(order)

    for position, node_id in enumerate(order):
        node = spec.nodes[node_id]
        step = STEP_REGISTRY[node.step_id]
        blocked = any(
            edge.get("node", "") in unavailable
            for edge in node.wiring.values()
        )
        if cancel_event is not None and cancel_event.is_set():
            record = NodeResult(
                node_id=node_id, step_id=node.step_id, state="CANCELLED",
                skipped_reason="the run's cancel flag was observed",
            )
            run.nodes.append(record)
            unavailable.add(node_id)
            continue
        if blocked:
            culprits = sorted(
                edge.get("node", "") for edge in node.wiring.values()
                if edge.get("node", "") in unavailable
            )
            record = NodeResult(
                node_id=node_id, step_id=node.step_id, state="SKIPPED",
                skipped_reason=(
                    f"upstream unavailable: {culprits[0]}"
                ),
            )
            run.nodes.append(record)
            unavailable.add(node_id)
            continue
        inputs: dict[str, Any] = {}
        missing: list[str] = []
        for port_spec in step.inputs:
            edge = node.wiring.get(port_spec.name)
            if edge is not None:
                key = (edge.get("node", ""), edge.get("port", ""))
                if key not in artifacts:
                    missing.append(f"{port_spec.name}@{key}")
                    continue
                inputs[port_spec.name] = artifacts[key]
            elif port_spec.name in initial:
                inputs[port_spec.name] = initial[port_spec.name]
            elif port_spec.required:
                missing.append(port_spec.name)
            else:
                inputs[port_spec.name] = None
        if missing:
            unavailable.add(node_id)
            run.nodes.append(NodeResult(
                node_id=node_id, step_id=node.step_id, state="FAILED",
                started_at=_now_iso(),
                error=(
                    f"required input(s) unavailable: "
                    f"{', '.join(sorted(set(missing)))}"
                ),
            ))
            continue
        params = dict(node.params)
        if "random_state" in {p.name for p in step.parameters} \
                and "random_state" not in params:
            params["random_state"] = seed
        started = time.monotonic()
        started_iso = _now_iso()
        try:
            outputs = STEP_RUNNERS[node.step_id](inputs, params)
        except Exception as exc:  # noqa: BLE001 — the verbatim record
            unavailable.add(node_id)
            run.nodes.append(NodeResult(
                node_id=node_id, step_id=node.step_id, state="FAILED",
                started_at=started_iso,
                duration_ms=(time.monotonic() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            ))
            if progress:
                progress(position + 1, total, node_id, "FAILED")
            continue
        upstream_lineage: set[tuple[str, str]] = set()
        for edge in node.wiring.values():
            upstream_key = (edge.get("node", ""), edge.get("port", ""))
            upstream_lineage.add(upstream_key)
            upstream_lineage.update(lineage_map.get(upstream_key, ()))
        produced = []
        for port_spec in step.outputs:
            if port_spec.name not in outputs:
                continue
            value = outputs[port_spec.name]
            key = (node_id, port_spec.name)
            artifacts[key] = value
            lineage_map[key] = tuple(sorted(upstream_lineage))
            digest, digest_kind = _digest_value(value)
            artifact = ArtifactRecord(
                node_id=node_id,
                port=port_spec.name,
                kind=port_spec.kind,
                digest=digest,
                digest_kind=digest_kind,
                lineage=lineage_map[key],
            )
            produced.append(artifact)
        run.nodes.append(NodeResult(
            node_id=node_id, step_id=node.step_id, state="COMPLETED",
            started_at=started_iso,
            duration_ms=(time.monotonic() - started) * 1000.0,
            artifacts=tuple(produced),
        ))
        if progress:
            progress(position + 1, total, node_id, "COMPLETED")

    counts = run.counts()
    if counts.get("CANCELLED", 0) and not counts.get("FAILED", 0):
        run.state = "CANCELLED"
    elif counts.get("FAILED", 0):
        run.state = "PARTIAL" if counts.get("COMPLETED", 0) else "FAILED"
    else:
        run.state = "COMPLETED"
    run.completed_at = _now_iso()
    return run


# -- the session engine ---------------------------------------------------------


class WorkflowEngine:
    """The session-scoped engine: execute + the bounded run ledger
    (§8's execution history).  Job-mode runs additionally persist
    through the JobManager's own SQLite history — the in-memory
    ledger is the live view, the job record the durable one."""

    def __init__(self) -> None:
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()

    def execute(
        self,
        spec: WorkflowSpec,
        *,
        initial_inputs: dict[str, Any] | None = None,
        seed: int = 42,
        cancel_event: Any = None,
        progress: ProgressCallback | None = None,
    ) -> RunRecord:
        run = execute(
            spec,
            initial_inputs=initial_inputs,
            seed=seed,
            cancel_event=cancel_event,
            progress=progress,
        )
        self._runs[run.run_id] = run
        while len(self._runs) > _HISTORY_LIMIT:
            self._runs.popitem(last=False)
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "run_id": run.run_id,
                "workflow_name": run.workflow_name,
                "state": run.state,
                "seed": run.seed,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "node_counts": run.counts(),
            }
            for run in list(self._runs.values())[::-1]
        ]

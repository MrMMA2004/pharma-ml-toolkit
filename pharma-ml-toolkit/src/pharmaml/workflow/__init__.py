"""The lightweight Scientific Workflow Engine (part13 Phase 6, §8 —
the round-322 API lane).

§8's constraint is this package's constitution: *do not introduce a
large workflow engine automatically; first assess whether the
existing JobManager and capability registry can support a lightweight
implementation; keep the application fully local and
offline-capable.*  The assessment and the candidate review
(Prefect / Knime / JupyterLab / marimo) live in
``docs/ARCHITECTURE_REVIEW.md``; the decision — the in-tree layer —
is this package:

* :mod:`pharmaml.workflow.contracts` — the typed step contracts +
  the step registry (the capability-registry discipline applied to
  workflow steps);
* :mod:`pharmaml.workflow.runners` — the REAL backend delegations
  (zero algorithms reimplemented, lazy heavy imports);
* :mod:`pharmaml.workflow.engine` — the graph machinery: spec
  validation, serialisation, topological execution with partial
  failure + cooperative cancellation, artifact lineage, the
  reproducibility block, and the bounded run ledger;
* :mod:`pharmaml.workflow.templates` — §8's example chain (the
  leakage-safe split-then-select reorder, honestly noted) + the
  minimal core chain.

The REST lane lives at ``/api/v1/workflow`` (routes/workflow.py);
job-mode runs ride the existing JobManager (progress, cancellation,
and the SQLite-backed job history all come from it — the §8
assessment's own conclusion, executed).
"""

from __future__ import annotations

from .contracts import PORT_KINDS, STEP_REGISTRY, ParamSpec, PortSpec, StepSpec
from .engine import (
    ArtifactRecord,
    NodeResult,
    NodeSpec,
    RunRecord,
    WorkflowEngine,
    WorkflowSpec,
    WorkflowValidationError,
    execute,
)
from .runners import STEP_RUNNERS
from .templates import TEMPLATE_REGISTRY, list_templates

__all__ = [
    "PORT_KINDS",
    "STEP_REGISTRY",
    "STEP_RUNNERS",
    "TEMPLATE_REGISTRY",
    "ArtifactRecord",
    "NodeResult",
    "NodeSpec",
    "ParamSpec",
    "PortSpec",
    "RunRecord",
    "StepSpec",
    "WorkflowEngine",
    "WorkflowSpec",
    "WorkflowValidationError",
    "execute",
    "list_templates",
]

"""The pharmaml.studio package — Studio-side persistence for part5 §11.

The §11 nine-card previews must show *actual* loaded files, persisted runs,
or parsed campaign outputs — never fabricated metrics.  This package owns
both halves of that contract:

* :mod:`pharmaml.studio.runs` — the persisted-run record format and its
  atomic, best-effort writer (one small JSON file per finished Studio job).
* :mod:`pharmaml.studio.evidence` — the per-card evidence collectors that
  turn real sources (loaded datasets, run records, checkpoints, the live
  model registry) into :class:`~pharmaml.studio.evidence.CardEvidence`
  values, each carrying either real numbers or a precise empty state.

Nothing here imports Qt: the collectors are pure functions designed to run
inside a ``QThreadPool`` worker (the existing Studio architecture) and are
tested headlessly.
"""

from pharmaml.studio.capability_registry import (
    CAPABILITIES,
    CARD_ORDER,
    OTHER_CARD_PRESENT,
    SECTION_KEYS,
    Capability,
    RegistrySummary,
    capabilities_for_card,
    card_route_target,
    find,
    summary,
)
from pharmaml.studio.evidence import (
    EMPTY_STATES,
    CardEvidence,
    collect_evidence,
    dataset_evidence,
    dl_evidence,
    docking_evidence,
    empty_evidence,
    generative_evidence,
    md_evidence,
    ml_evidence,
    pkpd_evidence,
    viz_evidence,
)
from pharmaml.studio.glossary import (
    GLOSSARY,
    apply_help,
    explanation,
    missing_keys,
    tooltip_text,
)
from pharmaml.studio.runs import (
    RUN_SCHEMA,
    latest_runs,
    record_run,
    run_families,
    runs_root,
)
from pharmaml.studio.workspace import (
    WorkspaceEntry,
    current_project,
    entries,
    note,
    set_current_project,
    summary_counts,
    workspace_path,
)

__all__ = [
    "CAPABILITIES",
    "CARD_ORDER",
    "EMPTY_STATES",
    "GLOSSARY",
    "OTHER_CARD_PRESENT",
    "RUN_SCHEMA",
    "SECTION_KEYS",
    "Capability",
    "CardEvidence",
    "RegistrySummary",
    "WorkspaceEntry",
    "apply_help",
    "capabilities_for_card",
    "card_route_target",
    "collect_evidence",
    "current_project",
    "dataset_evidence",
    "dl_evidence",
    "docking_evidence",
    "empty_evidence",
    "entries",
    "explanation",
    "find",
    "generative_evidence",
    "latest_runs",
    "md_evidence",
    "missing_keys",
    "ml_evidence",
    "note",
    "pkpd_evidence",
    "record_run",
    "run_families",
    "runs_root",
    "set_current_project",
    "summary",
    "summary_counts",
    "tooltip_text",
    "viz_evidence",
    "workspace_path",
]

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
from pharmaml.studio.runs import (
    RUN_SCHEMA,
    latest_runs,
    record_run,
    run_families,
    runs_root,
)

__all__ = [
    "EMPTY_STATES",
    "RUN_SCHEMA",
    "CardEvidence",
    "collect_evidence",
    "dataset_evidence",
    "dl_evidence",
    "docking_evidence",
    "empty_evidence",
    "generative_evidence",
    "latest_runs",
    "md_evidence",
    "ml_evidence",
    "pkpd_evidence",
    "record_run",
    "run_families",
    "runs_root",
    "viz_evidence",
]

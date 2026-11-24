"""``/api/v1/system/resources`` — part13 §15's resource-monitoring lane.

A READ-ONLY snapshot over three surfaces: the host (CPU / memory / swap /
disk), this backend process (RSS / threads / CPU), and the JobManager's
own queue depth by state. §15's own constraint is preserved verbatim:
"Preserve the existing JobManager. Improve only where evidence identifies
a real gap" — the manager's bounded concurrency (the expensive-job
semaphore), progress reporting and cooperative cancellation are untouched;
this lane only makes the resource picture the audit named as the gap
visible to the GUI/API consumer.

psutil (§15's own reference) is used when importable and NEVER required:
without it the lane degrades to the standard library's honest subset
(cpu_count, POSIX loadavg, shutil disk usage, resource rusage) and the
answer's ``source`` flag states ``stdlib-fallback`` with the absent
numbers explicitly nulled and noted. GPU monitoring is deliberately
absent (§15: "GPU support must be optional" — CPU-only operation must
remain supported): no pynvml dependency enters the tree.

The scope block rides every answer so the not-implemented set is a
machine-readable contract, not prose: job priorities, checkpointing,
recovery-after-interruption and per-job memory estimates stay honestly
queued for a later small reversible lane.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import logging.handlers
import os
import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, Request

import pharmaml

router = APIRouter(prefix="/system", tags=["system"])

_JOB_STATES = (
    "QUEUED",
    "RUNNING",
    "CANCELLATION_REQUESTED",
    "CANCELLED",
    "COMPLETED",
    "FAILED",
)
_ACTIVE_STATES = ("QUEUED", "RUNNING", "CANCELLATION_REQUESTED")

_SCOPE = {
    "implemented": [
        ("host CPU/memory/swap/disk snapshot (psutil when importable, "
         "else the honest stdlib subset with nulled absent numbers)"),
        "this backend process's own RSS/threads/CPU snapshot",
        "the JobManager queue depth by state (read-only)",
    ],
    "not_implemented": [
        "job priorities",
        "checkpointing",
        "recovery after interruption",
        "per-job memory estimates",
        ("GPU monitoring (deliberately absent — §15's optional-GPU rule; "
         "no pynvml dependency)"),
    ],
    "note": (
        "Read-only snapshot lane: the JobManager itself is unchanged — "
        "its bounded concurrency, progress reporting and cooperative "
        "cancellation stay exactly as they are (§15's preserve rule). "
        "psutil is optional; without it the source flag says "
        "stdlib-fallback and the absent numbers are nulled, never "
        "fabricated."
    ),
}

_GPU_NOTE = (
    "GPU monitoring is deliberately absent: §15 keeps GPU support "
    "optional and CPU-only operation fully supported, so no pynvml "
    "dependency enters the tree. A future lane may add it behind the "
    "same optional-import pattern this route uses for psutil."
)


def _loadavg_or_none() -> list[float] | None:
    """POSIX 1/5/15-minute load; ``None`` where ``os.getloadavg`` is
    unavailable (Windows) — the honest null, never a fabricated zero."""
    try:
        return [round(float(x), 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _disk_block() -> dict:
    usage = shutil.disk_usage(Path.cwd())
    return {
        "path": str(Path.cwd()),
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
    }


def _psutil_system() -> dict | None:
    """The rich block when psutil is importable, else ``None`` (the
    caller degrades to the stdlib subset)."""
    try:
        import psutil
    except Exception:  # noqa: BLE001 — optionality is the designed state
        return None
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "cpu_count": os.cpu_count(),
        # non-blocking since-last-call semantics (psutil's own contract:
        # the first observation after import is 0.0 — stated, not hidden)
        "cpu_percent": psutil.cpu_percent(interval=None),
        "loadavg": _loadavg_or_none(),
        "memory": {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        },
        "swap": {
            "total": sw.total,
            "used": sw.used,
            "free": sw.free,
            "percent": sw.percent,
        },
        "disk": _disk_block(),
    }


def _stdlib_system() -> dict:
    """The honest fallback subset: every psutil-only number is explicitly
    nulled and noted rather than fabricated."""
    return {
        "cpu_count": os.cpu_count(),
        "cpu_percent": None,
        "loadavg": _loadavg_or_none(),
        "memory": None,
        "swap": None,
        "disk": _disk_block(),
        "notes": [
            ("memory, swap and cpu_percent are absent without psutil "
             "(the optional §15 reference) — nulled, never fabricated"),
        ],
    }


def _process_block() -> dict:
    """This backend process's own footprint (psutil when importable,
    POSIX ``resource`` rusage as the degraded RSS source)."""
    block: dict = {
        "pid": os.getpid(),
        "cpu_percent": None,
        "memory_rss_bytes": None,
        "num_threads": None,
    }
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        block["cpu_percent"] = proc.cpu_percent(interval=None)
        block["memory_rss_bytes"] = proc.memory_info().rss
        block["num_threads"] = proc.num_threads()
    except Exception:  # noqa: BLE001 — fall through to the rusage source
        with contextlib.suppress(Exception):
            import resource  # POSIX fallback

            ru = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is KiB on Linux — the platform this suite pins
            block["memory_rss_bytes"] = ru.ru_maxrss * 1024
    return block


@router.get("/resources")
def resources(request: Request) -> dict:
    """The §15 resource snapshot (see the module docstring for scope)."""
    state = request.app.state
    jobs = state.job_manager.list_jobs()
    by_state = {s: 0 for s in _JOB_STATES}
    for job in jobs:
        job_state = job.get("state")
        if job_state in by_state:
            by_state[job_state] += 1
    rich = _psutil_system()
    return {
        "source": "psutil" if rich is not None else "stdlib-fallback",
        "backend": {"name": "pharmaml", "version": pharmaml.__version__},
        "system": rich if rich is not None else _stdlib_system(),
        "process": _process_block(),
        "jobs": {
            "active": sum(by_state[s] for s in _ACTIVE_STATES),
            "by_state": by_state,
            "note": (
                "In-session ledger off the JobManager's own list (the "
                "SQLite job history outlives the process — the "
                "persistence contract is unchanged)."
            ),
        },
        "gpu": {"available": False, "note": _GPU_NOTE},
        "scope": _SCOPE,
    }


# ---------------------------------------------------------------------------
# part16 §15.5 — the diagnostics/recovery surface (the roadmap #6 remainder:
# the audit's "diagnostics/recovery: partial" row closed over this lane).
#
# The directive's own checklist, made read-only and honest:
#   * backend health diagnostics     -> the health probe exists (routes/health);
#     this lane adds the depth: the dependency + scientific-tool
#     availability checks, the storage health, the crash diagnostics,
#     the log configuration report, and the recoverable job state.
#   * "Do not delete user data during diagnostic or cleanup operations"
#     -> the cleanup lane's target list is ENUMERATED (the samples
#     stats cache — this backend's only derived cache) and the answer's
#     preserved list names every user-data surface; the tests pin the
#     guarantee by mutating user data, cleaning, and asserting nothing
#     moved.
# ---------------------------------------------------------------------------

#: (import name, role, tier) — the requirements.txt vocabulary, pinned.
#: ``tier`` is the file's own honest split: "required" rows are the
#: core the toolkit cannot boot without; "optional" rows are the
#: feature sets the file documents as opt-in (chemistry, docking,
#: boosting, SHAP, deep learning, the rich monitor source).
_DEPENDENCY_PROBES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "the numeric core (the 1.26.4 pin)", "required"),
    ("pandas", "the dataframe layer", "required"),
    ("scipy", "the scientific routines", "required"),
    ("sklearn", "the estimator families", "required"),
    ("joblib", "the parallel + model persistence layer", "required"),
    ("openpyxl", "the xlsx read/write lane", "required"),
    ("rdkit", "the chemistry core (SMILES, descriptors, fingerprints)",
     "optional"),
    ("torch", "the deep-learning estimators + generative", "optional"),
    ("matplotlib", "the figure rendering", "optional"),
    ("xgboost", "the gradient-boosting family", "optional"),
    ("lightgbm", "the gradient-boosting family", "optional"),
    ("catboost", "the gradient-boosting family", "optional"),
    ("shap", "the explanation values", "optional"),
    ("selfies", "the robust molecular string codec", "optional"),
    ("mordred", "the descriptor engine", "optional"),
    ("meeko", "the docking preparation (SMILES -> PDBQT)", "optional"),
    ("gemmi", "the structure-file handling", "optional"),
    ("fastapi", "this API", "required"),
    ("uvicorn", "this server", "required"),
    ("pydantic", "the request validation", "required"),
    ("psutil", "the resource monitor's rich source", "optional"),
)

_DIAG_SCOPE = {
    "implemented": [
        ("dependency availability checks (the requirements.txt "
         "vocabulary, import + version per probe)"),
        ("scientific-tool availability checks (rdkit / torch / "
         "selfies with functional probes — the §15.4 grammar start "
         "reused, a tiny tensor op, the codec round-trip)"),
        ("storage health (the persistence status + the data-dir disk "
         "block + the database/vault sizes, or the honest ephemeral "
         "note)"),
        ("crash diagnostics (the ten most recent FAILED jobs with "
         "their errors, from the SQLite history when persistent)"),
        "logs (the logging configuration report, read-only)",
        ("recoverable job state (the persisted counts by state + the "
         "in-session view; the SQLite history outlives the process)"),
        ("safe cache cleanup (POST /system/cache/cleanup — the "
         "enumerated derived cache only)"),
    ],
    "not_implemented": [
        ("log file streaming or download (the configuration is "
         "reported, the content lane is queued)"),
        ("automatic job resume after a crash (checkpointing stays the "
         "queued scope item the resource lane named)"),
        ("crash analytics beyond the recorded job errors (the "
         "history's own vocabulary is the surface)"),
    ],
    "note": (
        "§15.5's own rule verbatim: 'Do not delete user data during "
        "diagnostic or cleanup operations.' The cleanup lane's target "
        "list is enumerated — the samples stats cache is this "
        "backend's only derived cache — and every user-data surface "
        "(datasets, trained models, generative priors, artifacts, "
        "user preferences, the job history) is named preserved in the "
        "answer."
    ),
}


def _dependency_block() -> list[dict]:
    """Import + version per probe; a failure is a REPORTED state, not
    an exception (a broken dependency is exactly what this lane is
    for)."""
    rows: list[dict] = []
    for name, role, tier in _DEPENDENCY_PROBES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — the availability is the answer
            rows.append({
                "name": name,
                "role": role,
                "tier": tier,
                "importable": False,
                "version": None,
                "error": type(exc).__name__,
            })
            continue
        rows.append({
            "name": name,
            "role": role,
            "tier": tier,
            "importable": True,
            "version": str(getattr(module, "__version__", None) or ""),
            "error": None,
        })
    return rows


def _scientific_tools() -> list[dict]:
    """The §15.5 scientific-tool probes with functional checks.

    ``functional_probe`` is ``True``/``False`` when the tool ran its
    probe, or ``None`` when the tool is absent (the import failed) —
    absent is stated, never probed, and never fabricated.
    """
    tools: list[dict] = []

    # rdkit — the §15.4 two-stage start (grammar parse) reused.
    try:
        import rdkit
        from rdkit import Chem

        parsed = Chem.MolFromSmiles("CCO")
        tools.append({
            "name": "rdkit",
            "version": str(rdkit.__version__),
            "importable": True,
            "functional_probe": parsed is not None,
            "probe": "Chem.MolFromSmiles('CCO') is not None",
        })
    except Exception:  # noqa: BLE001
        tools.append({
            "name": "rdkit",
            "version": None,
            "importable": False,
            "functional_probe": None,
            "probe": "Chem.MolFromSmiles('CCO') is not None",
        })

    # torch — a tiny tensor op (2-element add, no network, no files).
    try:
        import torch

        tensor = torch.tensor([1.0, 2.0]) + torch.tensor([1.0, 1.0])
        tools.append({
            "name": "torch",
            "version": torch.__version__,
            "importable": True,
            "functional_probe": bool(
                abs(float(tensor.sum().item()) - 5.0) < 1e-6
            ),
            "probe": "torch.tensor([1,2]) + torch.tensor([1,1]) == [2,3]",
        })
    except Exception:  # noqa: BLE001
        tools.append({
            "name": "torch",
            "version": None,
            "importable": False,
            "functional_probe": None,
            "probe": "torch.tensor([1,2]) + torch.tensor([1,1]) == [2,3]",
        })

    # selfies — the codec round-trip (pure Python, stdlib-only).
    try:
        import selfies as sf

        encoded = sf.encoder("CCO")
        decoded = sf.decoder(encoded) if encoded is not None else None
        tools.append({
            "name": "selfies",
            "version": sf.__version__,
            "importable": True,
            "functional_probe": decoded is not None,
            "probe": "selfies.decoder(selfies.encoder('CCO'))",
        })
    except Exception:  # noqa: BLE001
        tools.append({
            "name": "selfies",
            "version": None,
            "importable": False,
            "functional_probe": None,
            "probe": "selfies.decoder(selfies.encoder('CCO'))",
        })

    return tools


def _storage_block(request: Request) -> dict:
    """The §15.5 storage health: persistence status + disk + sizes."""
    persistence = request.app.state.persistence
    if persistence is None:
        return {
            "persistent": False,
            "reason": (
                "ephemeral session — no data_dir was configured, so "
                "nothing is written to disk (the part11 Phase-Eleven "
                "contract); the disk block below is the process's "
                "working directory"
            ),
            "disk": _disk_block(),
            "database": None,
            "vault_bytes": None,
            "integrity": None,
        }
    status = persistence.status()
    data_dir = Path(str(persistence.data_dir))
    db_path = data_dir / "studio.db"
    vault_dir = data_dir / "vault"
    vault_bytes = 0
    if vault_dir.is_dir():
        for f in vault_dir.rglob("*"):
            if f.is_file():
                with contextlib.suppress(OSError):
                    vault_bytes += f.stat().st_size
    return {
        "persistent": True,
        "data_dir": str(data_dir),
        "disk": _disk_block(),
        "database": {
            "file": "studio.db",
            "size_bytes": (
                db_path.stat().st_size if db_path.is_file() else None
            ),
            "schema_version": status.get("schema_version"),
            "journal_mode": status.get("journal_mode"),
        },
        "tables": status.get("tables"),
        "vault_bytes": vault_bytes,
        "integrity": status.get("integrity"),
    }


def _crash_block(request: Request) -> dict:
    """The §15.5 crash diagnostics: the recent FAILED jobs, bounded."""
    persistence = request.app.state.persistence
    limit = 10
    if persistence is not None:
        rows = [
            r for r in persistence.list_job_history(limit=200)
            if r.get("state") == "FAILED"
        ][:limit]
        source = (
            "sqlite job history (the FAILED rows survive restarts)"
        )
    else:
        rows = [
            j for j in request.app.state.job_manager.list_jobs()
            if j.get("state") == "FAILED"
        ][:limit]
        source = "in-session JobManager ledger (ephemeral session)"
    return {
        "source": source,
        "recent_failed": [
            {
                "job_id": r.get("job_id"),
                "operation": r.get("operation"),
                "workspace": r.get("workspace"),
                "error": r.get("error"),
                "completed_at": r.get("completed_at"),
            }
            for r in rows
        ],
    }


def _logs_block() -> dict:
    """The §15.5 log surface: the configuration, read-only."""
    root = logging.getLogger()
    level = logging.getLevelName(root.getEffectiveLevel())
    handlers: list[dict] = []
    log_file: str | None = None
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            kind = "rotating-file" if isinstance(
                handler, logging.handlers.RotatingFileHandler
            ) else "file"
            log_file = handler.baseFilename
            handlers.append({"kind": kind, "path": handler.baseFilename})
        elif isinstance(handler, logging.StreamHandler):
            handlers.append({"kind": "console"})
        else:
            handlers.append({"kind": handler.__class__.__name__})
    return {
        "level": level,
        "handlers": handlers,
        "log_file": log_file,
        "note": (
            "read-only configuration report — the content streaming "
            "lane is queued scope (see the not-implemented list); "
            "the logger module's rotating file handler is available "
            "to hosts via configure_logging()"
        ),
    }


def _recovery_block(request: Request) -> dict:
    """The §15.5 recoverable job state, honestly split."""
    state = request.app.state
    in_session = {s: 0 for s in _JOB_STATES}
    for job in state.job_manager.list_jobs():
        job_state = job.get("state")
        if job_state in in_session:
            in_session[job_state] += 1
    persistence = state.persistence
    if persistence is not None:
        rows = persistence.list_job_history(limit=1000)
        persisted: dict[str, int] = {}
        for row in rows:
            persisted[row.get("state", "?")] = (
                persisted.get(row.get("state", "?"), 0) + 1
            )
        source = "sqlite job history (survives the process)"
    else:
        persisted = {}
        source = "none — ephemeral session keeps no history"
    return {
        "persisted_by_state": persisted,
        "in_session_by_state": in_session,
        "source": source,
        "note": (
            "the FAILED/COMPLETED rows are the recoverable record "
            "(inspectable + the job-history API); QUEUED/RUNNING rows "
            "from a dead process are honestly NOT resumed — "
            "checkpointing stays the queued scope item the resource "
            "lane named, and the two views (persisted history vs the "
            "in-session manager) never pretend to be one"
        ),
    }


@router.get("/diagnostics")
def diagnostics(request: Request) -> dict:
    """The §15.5 read-only diagnostics answer (see the section notes)."""
    return {
        "backend": {
            "name": "pharmaml",
            "version": pharmaml.__version__,
            "python": sys.version.split()[0],
        },
        "dependencies": _dependency_block(),
        "scientific_tools": _scientific_tools(),
        "storage": _storage_block(request),
        "crash": _crash_block(request),
        "logs": _logs_block(),
        "recovery": _recovery_block(request),
        "scope": _DIAG_SCOPE,
    }


@router.post("/cache/cleanup")
def cache_cleanup(request: Request) -> dict:
    """§15.5's safe cache cleanup — the enumerated derived cache only.

    The samples stats cache (this backend's only derived cache — see
    ``routes/samples.clear_stats_cache``) is dropped; it re-derives on
    the next catalog read. EVERY user-data surface is named preserved
    and is untouched: the route holds no reference to the stores, the
    persistence, or the vault, and the tests pin the guarantee.
    """
    from .samples import clear_stats_cache

    dropped = clear_stats_cache()
    return {
        "cleared": {
            "samples_stats_entries": dropped,
        },
        "preserved": [
            "datasets (the dataset store + the uploaded source bytes)",
            "trained models (the registry + the vault payloads)",
            "generative priors",
            "artifacts",
            "user preferences",
            "the job history (the SQLite audit trail)",
            "experiment runs (the §15.2 lineage)",
        ],
        "note": (
            "§15.5 verbatim: 'Do not delete user data during "
            "diagnostic or cleanup operations.' The cleared list is "
            "the complete inventory of this backend's derived caches; "
            "nothing outside it is touched."
        ),
    }

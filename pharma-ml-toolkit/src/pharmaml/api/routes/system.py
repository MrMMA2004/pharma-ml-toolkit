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
import os
import shutil
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

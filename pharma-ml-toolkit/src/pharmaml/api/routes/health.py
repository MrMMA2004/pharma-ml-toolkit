"""``/api/v1/health`` — liveness + identity (part10 §10.1 step 9).

The health probe validates the *whole* dependency chain the WinUI host
relies on before enabling the scientific workspaces: the session
security module, the event bus, and the job manager — plus the
scientific core (RDKit import check) so a broken runtime is visible
at startup, not at first job.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

import pharmaml

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health(request: Request) -> dict:
    state = request.app.state
    jobs = state.job_manager.list_jobs()
    active = [j for j in jobs
              if j["state"] in ("QUEUED", "RUNNING", "CANCELLATION_REQUESTED")]
    return {
        "status": "ok",
        "backend": "pharmaml",
        "version": pharmaml.__version__,
        "api": "v1",
        "loopback_only": True,
        "scientific_core": _rdkit_ready(),
        "event_bus_last_sequence": state.bus.last_sequence,
        "jobs_active": len(active),
        "token_sha256": state.security.token_sha256,
    }


def _rdkit_ready() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — a broken runtime is the
        # state being probed, not an error to propagate
        return False

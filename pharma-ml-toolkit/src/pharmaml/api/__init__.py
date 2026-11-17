"""The PharmaML Studio local backend service (part10 Phase Six).

``create_app()`` assembles the FastAPI application exactly as the
WinUI host will launch it:

* a fresh :class:`~pharmaml.api.security.SessionSecurity` (the token
  is generated here, stays in memory, and reaches the parent process
  only through the structured startup handshake);
* a fresh :class:`~pharmaml.api.events.EventBus` and
  :class:`~pharmaml.api.jobs.JobManager`;
* the security dependency applied to *every* route (loopback +
  token), with the OpenAPI schema served locally so the C# client
  can be generated from it (NSwag) — one DTO definition, never two;
* versioned routes under ``/api/v1`` (health, capabilities, jobs,
  events, chemistry — this opening round; the ML/DL/generative/
  docking/pkpd/md/viz/AD groups follow the same recipe in the
  campaign's next rounds).

Run it with :func:`pharmaml.api.server.serve` — loopback-bound,
port-0 (no hardcoded port, no check-then-bind race), structured
handshake line on stdout, graceful shutdown.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request

from .events import EventBus
from .jobs import JobManager
from .routes import api_router, ws_router
from .security import SessionSecurity, reset_session_security

__all__ = ["EventBus", "JobManager", "SessionSecurity", "create_app"]


def create_app(token: str | None = None) -> FastAPI:
    """Assemble the local backend app (fresh bus/jobs per app)."""
    secure: SessionSecurity = reset_session_security(token=token)
    bus = EventBus()
    job_manager = JobManager(bus)

    app = FastAPI(
        title="PharmaML Studio local backend",
        version="1",
        docs_url=None,  # §9.4: docs disabled unless explicitly required
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.state.security = secure
    app.state.bus = bus
    app.state.job_manager = job_manager

    def guard(request: Request) -> None:
        secure.guard_request(request)

    app.include_router(
        api_router,
        dependencies=[Depends(guard)],
    )
    app.include_router(ws_router)

    @app.on_event("shutdown")
    async def _shutdown_jobs() -> None:
        job_manager.shutdown()

    return app

"""The PharmaML Studio local backend service (part10 Phase Six →
part11 Phase Seven/Eleven).

``create_app()`` assembles the FastAPI application exactly as the
Tauri host will launch it:

* a fresh :class:`~pharmaml.api.security.SessionSecurity` (the token
  is generated here, stays in memory, and reaches the parent process
  only through the structured startup handshake);
* a narrowly scoped CORS preflight lane (part14 §3.2 — D-2): the
  packaged Tauri 2 WebView2 origin (``tauri.localhost``, scheme-strict
  and host-exact) and the loopback dev origins (any port — the Vite
  dev server) are the only origins whose cross-origin requests get
  answered with per-origin allow headers.  The lane NEVER bypasses
  the guard — an OPTIONS preflight is a policy probe, so it is
  answered before routing without the session token, but every
  actual request still needs the loopback Host + the valid session
  token exactly as before;
* a fresh :class:`~pharmaml.api.events.EventBus` and
  :class:`~pharmaml.api.jobs.JobManager`;
* the security dependency applied to *every* route (loopback +
  token), with the OpenAPI schema served locally so the TypeScript
  client can be generated from it (openapi-typescript/Orval — part11
  §12.2; one DTO definition, never two);
* versioned routes under ``/api/v1`` (health, capabilities, jobs,
  events, chemistry — the opening round; the second row added
  datasets, ml, artifacts + the file upload/download contracts; the
  third row added dl, semi-supervised, generative + the
  training-progress events; row four added the five domain surfaces:
  docking, pkpd, md, visualization and the applicability-domain
  analysis; **the part11 row adds the database surface** —
  ``/api/v1/database``: status, job history, preferences, backup);
* **persistence (part11 Phase Eleven)**: ``create_app(data_dir=...)``
  (or the ``PHARMAML_STUDIO_DATA`` environment variable) turns on
  SQLite + file-vault durability — stores write through, job history
  persists across restarts, and the prior registry is recovered on
  startup.  Without a data dir the app stays **ephemeral** — the
  documented library/test default the shipped host never takes
  (:func:`pharmaml.api.server.serve` always passes the platform
  user-data directory);
* the backend's typed client-input errors (``DataValidationError``/
  ``ModelError``) map to 422 responses so a bad target column or an
  unknown registry model is answered honestly instead of a 500.

Run it with :func:`pharmaml.api.server.serve` — loopback-bound,
port-0 (no hardcoded port, no check-then-bind race), structured
handshake line on stdout, graceful shutdown.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pharmaml.exceptions import DataValidationError, ModelError

_logger = logging.getLogger(__name__)

from .events import EventBus
from .jobs import JobManager
from .persistence import Persistence, default_data_dir, recover_registry
from .routes import api_router, ws_router
from .security import SessionSecurity, reset_session_security
from .stores import (
    ArtifactStore,
    DatasetStore,
    ModelStore,
    PriorStore,
)

__all__ = [
    "ArtifactStore",
    "DatasetStore",
    "EventBus",
    "JobManager",
    "ModelStore",
    "Persistence",
    "PriorStore",
    "SessionSecurity",
    "create_app",
    "default_data_dir",
]


def create_app(
    token: str | None = None,
    data_dir: str | Path | None = None,
) -> FastAPI:
    """Assemble the local backend app (fresh bus/jobs per app).

    ``data_dir``: pass a directory (or set ``PHARMAML_STUDIO_DATA``)
    to enable part11 Phase Eleven persistence — SQLite + file vault
    under that directory, recovered on startup.  ``None`` keeps the
    ephemeral session contract (nothing written to disk).
    """
    secure: SessionSecurity = reset_session_security(token=token)
    bus = EventBus()

    resolved = data_dir
    if resolved is None and os.environ.get("PHARMAML_STUDIO_DATA"):
        resolved = os.environ["PHARMAML_STUDIO_DATA"]
    persistence = (
        Persistence(Path(resolved).expanduser()) if resolved else None
    )

    job_manager = JobManager(bus, persistence=persistence)

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
    app.state.persistence = persistence
    app.state.dataset_store = DatasetStore(persistence=persistence)
    app.state.model_store = ModelStore(persistence=persistence)
    app.state.artifact_store = ArtifactStore(persistence=persistence)
    app.state.prior_store = PriorStore(persistence=persistence)

    recovery_report: dict | None = None
    if persistence is not None:
        recovery_report = recover_registry(
            persistence,
            app.state.dataset_store,
            app.state.model_store,
            app.state.prior_store,
            app.state.artifact_store,
        )
    app.state.recovery_report = recovery_report

    def guard(request: Request) -> None:
        secure.guard_request(request)

    # part14 §3.2 (D-2) — the scoped CORS lane, BEFORE the routers so
    # it wraps the guard: preflights from the trusted origins are
    # answered here (200 + the exact-origin allow headers) instead of
    # 405-ing into the browser's "Failed to fetch"; every other origin
    # gets NO allow headers (the disallowed preflight answer carries
    # no ACAO, so the browser keeps blocking the actual request).  The
    # regex is fully anchored (starlette 0.50 fullmatches it) and
    # admits exactly: the two Tauri virtual-host origins, and loopback
    # origins with any port — nothing else.  ``allow_credentials``
    # stays false: the session token is a custom header, never a
    # cookie lane.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^https?://tauri\.localhost$"
            r"|^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
        ),
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["content-type", "x-session-token", "authorization"],
        allow_credentials=False,
        max_age=86400,
    )

    app.include_router(
        api_router,
        dependencies=[Depends(guard)],
    )
    app.include_router(ws_router)

    @app.on_event("shutdown")
    async def _shutdown_jobs() -> None:
        job_manager.shutdown()
        if persistence is not None:
            try:  # §16.4 retention rule: prune after every graceful stop
                persistence.prune_jobs()
            except Exception:
                _logger.debug("job-history prune failed at shutdown",
                              exc_info=True)

    def _client_error(request: Request, exc: Exception) -> JSONResponse:
        # a typed client-input error from the backend → an honest 422
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    app.add_exception_handler(DataValidationError, _client_error)
    app.add_exception_handler(ModelError, _client_error)

    return app

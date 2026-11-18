"""Route aggregation — organised by domain (part10 §9.1).

The WebSocket channel is included WITHOUT the REST guard: FastAPI
applies router-level dependencies to websocket routes with the WS
scope, and the WS protocol carries the token in its own handshake
(``?token=`` / ``X-Session-Token`` — see routes/events.py), plus
the loopback bind keeps non-local peers off the socket entirely.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    applicability_domain,
    artifacts,
    capabilities,
    chemistry,
    database,
    datasets,
    dl,
    docking,
    events,
    generative,
    health,
    jobs,
    md,
    ml,
    pkpd,
    semi_supervised,
    visualization,
)

__all__ = ["api_router", "ws_router"]


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(capabilities.router)
api_router.include_router(jobs.router)
api_router.include_router(chemistry.router)
api_router.include_router(datasets.router)
api_router.include_router(ml.router)
api_router.include_router(artifacts.router)
api_router.include_router(dl.router)
api_router.include_router(semi_supervised.router)
api_router.include_router(generative.router)
api_router.include_router(docking.router)
api_router.include_router(pkpd.router)
api_router.include_router(md.router)
api_router.include_router(visualization.router)
api_router.include_router(applicability_domain.router)
api_router.include_router(database.router)

ws_router = APIRouter(prefix="/api/v1")
ws_router.include_router(events.router)

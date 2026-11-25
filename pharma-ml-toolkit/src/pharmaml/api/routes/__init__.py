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
    admet,
    applicability_domain,
    artifacts,
    capabilities,
    chemistry,
    database,
    datasets,
    dl,
    docking,
    drugresponse,
    dta,
    events,
    generative,
    health,
    jobs,
    md,
    ml,
    model_evaluation,
    multiobjective,
    multiresponse,
    pharmacogenomics,
    pkpd,
    repurposing,
    rsm,
    runs,
    samples,
    semi_supervised,
    system,
    visualization,
    workflow,
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
api_router.include_router(model_evaluation.router)
api_router.include_router(admet.router)
api_router.include_router(dta.router)
api_router.include_router(repurposing.router)
api_router.include_router(drugresponse.router)
api_router.include_router(pharmacogenomics.router)
api_router.include_router(multiobjective.router)
# part17 §5 (the campaign queue's M1 lane) — the generic
# multi-response ML over pharmaml.ml.multiresponse.
api_router.include_router(multiresponse.router)
# part17 §6 (the campaign queue's R1 lane) — the Response Surface
# Methodology engine over pharmaml.ml.rsm.
api_router.include_router(rsm.router)
api_router.include_router(workflow.router)
api_router.include_router(database.router)
api_router.include_router(system.router)
# part16 §8 (Phase 5) — the demo/sample-data lane over the REAL
# shipped examples/data files (curated metadata, no invention).
api_router.include_router(samples.router)
# part16 §15.2 (Phase 12) — the experiment-tracking / run-lineage lane:
# the query projection over the RunStore recorded at the training points.
api_router.include_router(runs.router)

ws_router = APIRouter(prefix="/api/v1")
ws_router.include_router(events.router)

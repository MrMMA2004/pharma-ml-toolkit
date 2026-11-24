"""Experiment-run lineage surface — part16 §15.2 (the Phase-12 lane).

The query projection over the RunStore: the lineage table (summaries,
filterable) and the full §15.2 record per run.  Read-only by design —
runs are recorded at the training points (routes/ml.py), never
hand-POSTed; the record is the system's, not the user's claim.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

__all__ = ["router"]

router = APIRouter(prefix="/runs", tags=["runs"])

_STATUSES = ("completed", "failed", "cancelled")


@router.get("")
def list_runs(
    request: Request,
    family: str | None = None,
    dataset_id: str | None = None,
    model: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict:
    """The lineage table: run summaries, newest first.

    ``status`` is validated (the three honest states) so a typo cannot
    silently filter everything away.
    """
    if status is not None and status not in _STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {', '.join(_STATUSES)}",
        )
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=422, detail="limit must be between 1 and 200"
        )
    store = request.app.state.run_store
    summaries = store.list(
        family=family,
        dataset_id=dataset_id,
        model_name=model,
        status=status,
        limit=limit,
    )
    return {"runs": summaries, "count": len(summaries)}


@router.get("/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    """The full §15.2 lineage record for one run."""
    store = request.app.state.run_store
    view = store.get(run_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    return view

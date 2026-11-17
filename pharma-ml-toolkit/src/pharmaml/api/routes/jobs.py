"""``/api/v1/jobs`` — submit / poll / cancel (part10 Phase Nine).

REST stays the authoritative status path: the WebSocket event stream
is a *convenience* view, and every recovery flow ends here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobView(BaseModel):
    job_id: str
    operation: str
    workspace: str
    state: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: int = 0
    stage: str
    configuration: dict
    cancellation: str
    error: str | None = None


@router.get("")
def list_jobs(request: Request, workspace: str | None = None) -> dict:
    return {"jobs": request.app.state.job_manager.list_jobs(workspace)}


@router.get("/{job_id}")
def job_status(job_id: str, request: Request) -> dict:
    record = request.app.state.job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    view = record.public()
    view["result_ready"] = record.state.value == "COMPLETED"
    return view


@router.get("/{job_id}/result")
def job_result(job_id: str, request: Request) -> dict:
    record = request.app.state.job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    if record.state.value == "CANCELLED":
        raise HTTPException(
            status_code=409, detail="Job was cancelled; no result."
        )
    if record.state.value != "COMPLETED":
        raise HTTPException(
            status_code=409,
            detail=f"Job state is {record.state.value}; not completed.",
        )
    return {"job_id": job_id, "result": record.result}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    record = request.app.state.job_manager.cancel(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    view = record.public()
    view["cancel_acknowledged"] = True
    return view

"""``/api/v1/database`` — the database surface (part11 Phase Eleven,
§12.1's "database-backed metadata retrieval" group).

The Python backend owns the SQLite database; this surface exposes
*metadata retrieval*, never raw file access:

* ``GET /api/v1/database`` — the honest status report: persistent
  mode returns the schema version, WAL mode, integrity check and
  per-table counts; the ephemeral library default returns
  ``persistent: false`` with the reason (the shipped host always
  takes the durable path);
* ``GET /api/v1/database/jobs`` — the persisted job history (the
  audit trail that survives restarts; the live ``/jobs`` routes stay
  the operational surface);
* ``GET/PUT/DELETE /api/v1/database/preferences`` — the durable
  user-preference key/value store;
* ``POST /api/v1/database/backup`` — the full backup bundle (the
  SQLite online-backup snapshot **plus every referenced vault
  payload**, zipped) registered as a standard ``database`` artifact
  — downloadable through the same verified download contract as
  every other artifact;
* ``POST /api/v1/database/restore`` — restores from such a backup
  artifact (integrity + schema-version validated first), then
  rebuilds the live registry from the restored database; the
  *complete* registry comes back, payloads included.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/database", tags=["database"])


class PreferenceValue(BaseModel):
    """One preference write — any strict-JSON-safe value."""

    value: Any = None


class PreferenceWrite(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    value: Any = None


class BackupRequest(BaseModel):
    filename: str | None = None


class RestoreRequest(BaseModel):
    artifact_id: str


def _persistence(request: Request):
    persistence = request.app.state.persistence
    if persistence is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "persistence is disabled for this app instance "
                "(no data_dir / PHARMAML_STUDIO_DATA); the shipped "
                "host always runs durable"
            ),
        )
    return persistence


@router.get("")
def database_status(request: Request) -> dict:
    persistence = request.app.state.persistence
    if persistence is None:
        return {
            "persistent": False,
            "reason": (
                "ephemeral app instance: create_app() without a data "
                "directory (the documented library/test default); "
                "the Tauri-hosted entry is always durable"
            ),
        }
    status = persistence.status()
    recovery: dict | None = request.app.state.recovery_report
    if recovery is not None:
        status["startup_recovery"] = recovery
    return status


@router.get("/jobs")
def job_history(
    request: Request,
    workspace: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    persistence = _persistence(request)
    return {
        "jobs": persistence.list_job_history(
            workspace=workspace, limit=limit
        )
    }


@router.get("/preferences")
def list_preferences(request: Request) -> dict:
    persistence = _persistence(request)
    return {"preferences": persistence.list_preferences()}


@router.get("/preferences/{key}")
def get_preference(key: str, request: Request) -> dict:
    persistence = _persistence(request)
    rows = persistence._execute(
        "SELECT key FROM user_preferences WHERE key = ?", (key,)
    )
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"Unknown preference: {key!r}."
        )
    return {"key": key, "value": persistence.get_preference(key)}


@router.put("/preferences/{key}")
def set_preference(
    key: str, body: PreferenceValue, request: Request
) -> dict:
    persistence = _persistence(request)
    if not key or len(key) > 128:
        raise HTTPException(
            status_code=422, detail="preference key must be 1-128 chars"
        )
    persistence.set_preference(key, body.value)
    return {
        "key": key,
        "value": persistence.get_preference(key),
        "saved": True,
    }


@router.delete("/preferences/{key}")
def delete_preference(key: str, request: Request) -> dict:
    persistence = _persistence(request)
    if not persistence.delete_preference(key):
        raise HTTPException(
            status_code=404, detail=f"Unknown preference: {key!r}."
        )
    return {"key": key, "deleted": True}


@router.post("/backup")
def create_backup(req: BackupRequest, request: Request) -> dict:
    persistence = _persistence(request)
    data = persistence.backup_bundle()
    store = request.app.state.artifact_store
    record = store.register(
        kind="database",
        filename=req.filename or "pharmaml-studio-backup.zip",
        content_type="application/zip",
        data=data,
        meta={
            "schema_version": persistence.schema_version(),
            "source": "sqlite3 online backup API + vault payloads",
        },
    )
    return {"artifact": record.view()}


@router.post("/restore")
def restore_database(req: RestoreRequest, request: Request) -> dict:
    persistence = _persistence(request)
    store = request.app.state.artifact_store
    record = store.get(req.artifact_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail="Unknown artifact_id."
        )
    if record.kind != "database":
        raise HTTPException(
            status_code=422,
            detail=(
                f"artifact {req.artifact_id} is kind={record.kind!r}; "
                "restore accepts database backups only"
            ),
        )
    try:
        result = persistence.restore_bundle(record.data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # rebuild the LIVE registry from the restored database: the live
    # stores are replaced by the backup's registry (the documented
    # rollback semantics — the pre-restore live state is discarded)
    from ..persistence import recover_registry

    request.app.state.dataset_store._items.clear()
    request.app.state.model_store._items.clear()
    request.app.state.prior_store._items.clear()
    request.app.state.artifact_store._items.clear()
    recovery = recover_registry(
        persistence,
        request.app.state.dataset_store,
        request.app.state.model_store,
        request.app.state.prior_store,
        request.app.state.artifact_store,
    )
    request.app.state.recovery_report = recovery
    return {"restore": result, "recovered": recovery}

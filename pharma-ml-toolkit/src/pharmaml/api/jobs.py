"""The centralized backend Job Manager (part10 Phase Nine).

Every substantial operation runs as a managed job, never inside a
FastAPI endpoint function.  A job is:

* submitted with a **callable + kwargs** (the real backend function —
  no algorithm is reimplemented here);
* given a unique id, operation name, workspace, creation timestamp,
  and its configuration snapshot;
* executed on a ``ThreadPoolExecutor`` sized to the CPU budget (the
  scientifically thread-hostile workloads stay GIL-bound inside one
  thread at a time — see ``max_concurrent_expensive_jobs``);
* driven through the honest state machine::

      QUEUED -> RUNNING -> COMPLETED | FAILED
                    -> CANCELLATION_REQUESTED -> CANCELLED

**Cancellation is real and cooperative**: each job carries a
``threading.Event`` cancel flag; the job's callable receives it
via the ``cancel_event`` kwarg (injected automatically when the
callable accepts it) and MUST check it between processing stages.
Callables that never accept ``cancel_event`` are recorded
``cancellation="unsupported"`` — the spec's requirement to *explain*
the limitation rather than pretend.

Progress flows through the bus as ``job.progress`` /
``descriptor.progress`` events; results are stored in the job record
and retrievable via REST (the authoritative recovery path).
"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future as _Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .events import EventBus, EventType

__all__ = ["JobManager", "JobRecord", "JobState"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class JobRecord:
    """One managed job (part10: every job carries the full ledger)."""

    job_id: str
    operation: str
    workspace: str
    created_at: str
    configuration: dict
    cancellation: str = "cooperative"
    started_at: str | None = None
    completed_at: str | None = None
    state: JobState = JobState.QUEUED
    progress: int = 0
    stage: str = "queued"
    result: Any = None
    error: str | None = None
    attempts: int = 0

    def public(self) -> dict:
        """The REST-facing view (result serialised by the route layer)."""
        return {
            "job_id": self.job_id,
            "operation": self.operation,
            "workspace": self.workspace,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "stage": self.stage,
            "configuration": self.configuration,
            "cancellation": self.cancellation,
            "error": self.error,
        }


class CancelledJob(Exception):
    """Raised inside a job body after it observes the cancel flag."""


@dataclass
class _JobHandle:
    record: JobRecord
    future: _Future | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    """Session-wide job manager: submit / poll / cancel / list."""

    def __init__(
        self,
        bus: EventBus,
        max_concurrent_expensive_jobs: int = 2,
        worker_threads: int = 4,
    ) -> None:
        self._bus = bus
        self._jobs: dict[str, _JobHandle] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix="pharmaml-job",
        )
        # unbounded concurrent *cheap* ops are allowed; expensive ones
        # gate through this semaphore (no CPU oversubscription).
        self._expensive_slots = threading.Semaphore(
            max_concurrent_expensive_jobs
        )
        self._shutdown = False

    # -- submission ---------------------------------------------------

    def submit(
        self,
        operation: str,
        workspace: str,
        fn: Callable[..., Any],
        *,
        configuration: dict | None = None,
        expensive: bool = True,
        **kwargs: Any,
    ) -> JobRecord:
        """Queue a callable as a managed job and return its record.

        ``cancel_event`` is injected automatically when ``fn`` accepts
        such a parameter; otherwise cancellation is honestly recorded
        as unsupported for this operation.
        """
        if self._shutdown:
            raise RuntimeError("JobManager is shut down; new jobs rejected.")
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        params = inspect.signature(fn).parameters
        cooperative = "cancel_event" in params
        self_aware = "job_id" in params
        record = JobRecord(
            job_id=job_id,
            operation=operation,
            workspace=workspace,
            created_at=_now_iso(),
            configuration=dict(configuration or {}),
            cancellation="cooperative" if cooperative else "unsupported",
        )
        handle = _JobHandle(record=record)
        with self._lock:
            self._jobs[job_id] = handle
        self._bus.publish(
            EventType.JOB_QUEUED,
            {"operation": operation, "workspace": workspace},
            job_id=job_id,
        )
        run_kwargs = dict(kwargs)
        if cooperative:
            run_kwargs["cancel_event"] = handle.cancel_event
        if self_aware:
            run_kwargs["job_id"] = job_id

        def _run() -> Any:
            if expensive:
                self._expensive_slots.acquire()
            try:
                if handle.cancel_event.is_set():
                    raise CancelledJob("Cancelled before start.")
                with self._lock:
                    record.state = JobState.RUNNING
                    record.started_at = _now_iso()
                    record.stage = "running"
                self._bus.publish(
                    EventType.JOB_STARTED, {"operation": operation}, job_id=job_id
                )
                try:
                    result = fn(**run_kwargs)
                except CancelledJob as exc:
                    with self._lock:
                        record.state = JobState.CANCELLED
                        record.completed_at = _now_iso()
                        record.stage = "cancelled"
                        record.error = str(exc) or "Cancelled."
                    self._bus.publish(
                        EventType.JOB_CANCELLED, {}, job_id=job_id
                    )
                    return None
                with self._lock:
                    record.state = JobState.COMPLETED
                    record.completed_at = _now_iso()
                    record.stage = "completed"
                    record.progress = 100
                    record.result = result
                self._bus.publish(
                    EventType.JOB_COMPLETED,
                    {"progress": 100},
                    job_id=job_id,
                )
                self._bus.publish(
                    EventType.RESULT_READY, {}, job_id=job_id
                )
                return result
            except BaseException as exc:  # real failure path
                with self._lock:
                    record.state = JobState.FAILED
                    record.completed_at = _now_iso()
                    record.stage = "failed"
                    record.error = f"{type(exc).__name__}: {exc}"
                self._bus.publish(
                    EventType.JOB_FAILED,
                    {"error": record.error},
                    job_id=job_id,
                )
                raise
            finally:
                if expensive:
                    self._expensive_slots.release()

        handle.future = self._executor.submit(_run)
        return record

    # -- queries ------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            handle = self._jobs.get(job_id)
            return handle.record if handle else None

    def list_jobs(self, workspace: str | None = None) -> list[dict]:
        with self._lock:
            records = [
                h.record for h in self._jobs.values()
                if workspace is None or h.record.workspace == workspace
            ]
        return [r.public() for r in sorted(records, key=lambda r: r.created_at)]

    # -- control ------------------------------------------------------

    def cancel(self, job_id: str) -> JobRecord | None:
        """Cooperative cancel: set the flag, publish the request."""
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return None
            record = handle.record
            if record.state in (JobState.RUNNING, JobState.QUEUED):
                handle.cancel_event.set()  # the real cooperative flag
            if record.state is JobState.RUNNING:
                record.state = JobState.CANCELLATION_REQUESTED
            elif record.state is JobState.QUEUED:
                record.state = JobState.CANCELLED
                record.completed_at = _now_iso()
                record.stage = "cancelled"
                record.error = "Cancelled before start."
        self._bus.publish(
            EventType.JOB_CANCEL_REQUESTED, {}, job_id=job_id
        )
        return record

    def report_progress(
        self,
        job_id: str,
        progress: int,
        stage: str | None = None,
        event: EventType = EventType.JOB_PROGRESS,
    ) -> None:
        """Job bodies call this between stages (cooperative pattern)."""
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return
            handle.record.progress = int(progress)
            if stage:
                handle.record.stage = stage
        self._bus.publish(event, {"progress": int(progress),
                                  "stage": stage or ""}, job_id=job_id)

    # -- shutdown -----------------------------------------------------

    def shutdown(self, drain_timeout: float = 5.0) -> dict:
        """Graceful stop (part10 §10.3): no new jobs, drain, cancel."""
        self._shutdown = True
        deadline = time.monotonic() + drain_timeout
        with self._lock:
            handles = list(self._jobs.values())
        for handle in handles:
            if handle.record.state in (
                JobState.QUEUED, JobState.RUNNING,
                JobState.CANCELLATION_REQUESTED,
            ):
                handle.cancel_event.set()
        self._bus.publish(EventType.SHUTDOWN, {})
        remaining: list[str] = []
        for handle in handles:
            if handle.future is not None:
                try:
                    handle.future.result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except Exception:  # noqa: BLE001 — the drain must
                    # not raise: an already-dead job is recorded as
                    # remaining, nothing more (§10.3's declared policy)
                    remaining.append(handle.record.job_id)
        self._executor.shutdown(wait=False, cancel_futures=True)
        return {"drained": len(handles) - len(remaining),
                "remaining": remaining}

"""Typed event envelopes and the per-session event bus (part10 §9.3).

Every WebSocket message the backend emits is a *typed event envelope*
(Pydantic model, so the OpenAPI schema and the generated C# client
share one definition — never two hand-edited DTOs):

.. code-block:: json

    {"event_id": "…", "sequence": 104, "job_id": "…",
     "event_type": "job.progress", "timestamp": "…Z",
     "payload": {"progress": 65, "stage": "Model training"}}

Guarantees implemented here:

* **real generated IDs** — ``secrets.token_urlsafe(12)``;
* **monotonic per-bus sequencing** — subscribers see a strictly
  increasing ``sequence``;
* **duplicate-event protection** — replay on reconnect uses
  ``since=<sequence>``, so a client that reconnects asks for exactly
  the gap, and the bus's ring buffer keeps the tail (bounded memory);
* **heartbeats** — the connection layer (routes/events.py) sends
  ``heartbeat`` envelopes on idle, so a dead TCP peer is detected;
* **backpressure** — per-subscriber bounded queues: a slow client
  drops nothing silently; the oldest buffered event is replaced by a
  ``bus.overflow`` notice and the REST status route remains the
  authoritative recovery path;
* **thread-to-loop safety** — job bodies publish from worker
  threads; each subscriber records its owning event loop and
  cross-thread publication rides ``call_soon_threadsafe`` (a plain
  ``asyncio.Queue.put_nowait`` from a foreign thread corrupts the
  queue's internal waiters — this bug was caught by the WS suite).
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

__all__ = [
    "EventBus",
    "EventEnvelope",
    "EventType",
]


class EventType(str, Enum):
    """The typed event vocabulary (part10 §9.3's list)."""

    JOB_QUEUED = "job.queued"
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_STAGE = "job.stage"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCEL_REQUESTED = "job.cancel_requested"
    JOB_CANCELLED = "job.cancelled"
    RESULT_READY = "result.ready"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_METRICS = "training.metrics"
    ITERATION_UPDATE = "iteration.update"
    DESCRIPTOR_PROGRESS = "descriptor.progress"
    FILE_PROGRESS = "file.progress"
    LOG_RECORD = "log.record"
    WARNING = "warning"
    ERROR = "error"
    HEALTHBEAT = "heartbeat"
    BUS_OVERFLOW = "bus.overflow"
    SHUTDOWN = "shutdown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EventEnvelope(BaseModel):
    """One WebSocket event — the single wire DTO."""

    event_id: str = Field(
        default_factory=lambda: secrets.token_urlsafe(12),
        description="Unique id for this event (real, generated).",
    )
    sequence: int = Field(
        description="Strictly increasing bus-wide sequence number.",
    )
    event_type: EventType
    timestamp: str = Field(default_factory=_now_iso)
    job_id: str | None = Field(default=None)
    payload: dict = Field(default_factory=dict)


class _Subscriber:
    """One live WebSocket connection's buffer + owning loop."""

    __slots__ = ("capacity", "loop", "queue")

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self.queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(capacity)
        self.loop: asyncio.AbstractEventLoop | None = None


class EventBus:
    """Session-wide typed event bus with replayable history.

    ``history`` is a bounded deque of the most recent envelopes; on
    reconnect a client passes ``since=<last sequence seen>`` and the
    handshake replays exactly the missed tail (duplicate protection:
    events with ``sequence <= since`` are never re-sent).
    """

    def __init__(self, history_size: int = 512) -> None:
        self._history: deque[EventEnvelope] = deque(maxlen=history_size)
        self._subs: set[_Subscriber] = set()
        self._seq = 0
        self._lock = threading.RLock()

    @property
    def last_sequence(self) -> int:
        return self._seq

    def _offer(self, sub: _Subscriber, envelope: EventEnvelope) -> None:
        """Thread-safe hand-off onto the subscriber's loop."""
        loop = sub.loop
        if loop is None or loop.is_closed():
            return  # dead subscriber; unsubscribe sweep will catch it
        try:
            loop.call_soon_threadsafe(self._put, sub, envelope)
        except RuntimeError:  # loop closed between the two lines
            pass

    def _put(self, sub: _Subscriber, envelope: EventEnvelope) -> None:
        """Runs ON the subscriber's loop: bounded-queue semantics."""
        if sub.queue.full():
            try:
                sub.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - raced
                pass
            overflow = EventEnvelope(
                sequence=envelope.sequence,
                event_type=EventType.BUS_OVERFLOW,
                job_id=envelope.job_id,
                payload={
                    "dropped": True,
                    "note": "REST status is authoritative",
                },
            )
            try:
                sub.queue.put_nowait(overflow)
            except asyncio.QueueFull:  # pragma: no cover
                pass
        try:
            sub.queue.put_nowait(envelope)
        except asyncio.QueueFull:  # pragma: no cover - just emptied
            pass

    def publish(
        self,
        event_type: EventType,
        payload: dict | None = None,
        job_id: str | None = None,
    ) -> EventEnvelope:
        """Assign the next sequence and fan out to subscribers.

        Backpressure: when a subscriber's bounded queue is full the
        oldest buffered envelope is dropped from that subscriber and
        a ``bus.overflow`` notice replaces it — the client is told,
        and REST status remains the authoritative recovery path.
        """
        with self._lock:
            self._seq += 1
            envelope = EventEnvelope(
                sequence=self._seq,
                event_type=event_type,
                job_id=job_id,
                payload=payload or {},
            )
            self._history.append(envelope)
            subs = list(self._subs)
        for sub in subs:
            self._offer(sub, envelope)
        return envelope

    def replay(self, since: int) -> list[EventEnvelope]:
        """Envelopes with ``sequence > since`` (duplicate protection)."""
        with self._lock:
            snapshot = list(self._history)
        return [e for e in snapshot if e.sequence > since]

    async def subscribe(self, since: int = -1) -> _Subscriber:
        """Register a subscriber (on its own loop) + replay the tail."""
        sub = _Subscriber()
        sub.loop = asyncio.get_running_loop()
        with self._lock:
            self._subs.add(sub)
        for envelope in self.replay(since):
            try:
                sub.queue.put_nowait(envelope)
            except asyncio.QueueFull:
                break
        return sub

    async def unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    async def next_event(
        self,
        sub: _Subscriber,
        timeout: float | None = None,
    ) -> EventEnvelope | None:
        """Await the next envelope for a subscriber (None on timeout)."""
        if timeout is None:
            return await sub.queue.get()
        try:
            return await asyncio.wait_for(sub.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

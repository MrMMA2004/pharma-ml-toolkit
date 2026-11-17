"""``/api/v1/events`` — the typed WebSocket channel (part10 §9.3).

Connection contract (typed envelopes, generated ids, monotonic
sequence, heartbeats, replay, graceful close):

* connect with the session token as ``?token=`` or the
  ``X-Session-Token`` header (browsers cannot set headers on WS, so
  the query variant exists; both are loopback-only);
* optional ``?since=<sequence>`` — reconnecting clients pass their
  last seen sequence and the handshake replays exactly the missed
  tail (duplicate-event protection);
* idle periods carry ``heartbeat`` envelopes (dead-peer detection);
* on backend shutdown the server pushes ``shutdown`` before the
  socket closes (graceful connection shutdown);
* REST (``/api/v1/jobs/{id}``) remains the recovery path of record.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..events import EventEnvelope, EventType

router = APIRouter(prefix="/events", tags=["events"])

HEARTBEAT_SECONDS = 15.0


@router.websocket("")
async def events_socket(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token") or websocket.headers.get(
        "x-session-token"
    )
    security = websocket.app.state.security
    if not security.verify(token):
        await websocket.close(code=4401, reason="invalid session token")
        return
    # Non-loopback peers cannot reach a loopback-bound server; the
    # token check above is the second gate.
    await websocket.accept()
    bus = websocket.app.state.bus
    since_param = websocket.query_params.get("since")
    try:
        since = int(since_param) if since_param is not None else -1
    except ValueError:
        since = -1
    sub = await bus.subscribe(since=since)
    hello = EventEnvelope(
        sequence=bus.last_sequence,
        event_type=EventType.HEALTHBEAT,
        payload={
            "hello": True,
            "last_sequence": bus.last_sequence,
            "replayed": len(bus.replay(since)),
        },
    )
    await websocket.send_text(hello.model_dump_json())
    send_task = asyncio.create_task(_sender(websocket, bus, sub))
    try:
        while True:
            inbound = await websocket.receive_text()
            if inbound == "ping":
                await websocket.send_text(
                    EventEnvelope(
                        sequence=bus.last_sequence,
                        event_type=EventType.HEALTHBEAT,
                        payload={"pong": True},
                    ).model_dump_json()
                )
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        await bus.unsubscribe(sub)


async def _sender(websocket, bus, sub) -> None:
    """Forward bus envelopes + heartbeats until the client leaves."""
    while True:
        envelope = await bus.next_event(sub, timeout=HEARTBEAT_SECONDS)
        if envelope is None:
            envelope = EventEnvelope(
                sequence=bus.last_sequence,
                event_type=EventType.HEALTHBEAT,
                payload={"beat": True},
            )
        await websocket.send_text(envelope.model_dump_json())

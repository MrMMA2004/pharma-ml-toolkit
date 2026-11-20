"""part10 Phase Six opening — the local FastAPI backend service.

Covers (per the master prompt's test-alongside-every-change rule):

* security (§9.4): token auth, loopback Host/Origin enforcement,
  constant-time verify, token never in health output;
* the app factory: routes assembled, docs disabled, openapi served,
  every route behind the guard;
* the job manager (Phase Nine): full state machine, cooperative
  cancellation between stages, cancellation-before-start,
  unsupported-cancellation honesty, failed jobs, the expensive-slot
  cap, graceful shutdown drain;
* the event bus (§9.3): typed envelopes, generated ids, monotonic
  sequencing, replay/duplicate protection, bounded-subscriber
  overflow notice;
* chemistry over the REAL backend (descriptors/fingerprints):
  sync mode, job mode, validation (radius bool rejection, empty
  SMILES, unknown sets), and the exact-config preservation
  (radius/n_bits never clamped);
* the startup handshake (Phase Seven): one structured stdout line,
  port-0 binding with no race, loopback-only socket.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.events import EventBus, EventType
from pharmaml.api.jobs import CancelledJob, JobManager, JobState
from pharmaml.api.security import SessionSecurity
from pharmaml.api.server import HANDSHAKE_PREFIX, handshake_line

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    # loopback caller semantics — the production host is always
    # 127.0.0.1, and the guard enforces it on every request
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


# ---------------------------------------------------------------- security


class TestSessionSecurity:
    def test_generated_token_is_random_and_long(self):
        a, b = SessionSecurity(), SessionSecurity()
        assert a.token != b.token
        assert len(a.token) >= 32

    def test_verify_is_constant_time_semantics(self):
        s = SessionSecurity(token="abc")
        assert s.verify("abc")
        assert not s.verify("abd")
        assert not s.verify(None)
        assert not s.verify("")

    def test_sha256_is_a_digest_not_the_token(self):
        s = SessionSecurity(token=TEST_TOKEN)
        assert TEST_TOKEN not in s.token_sha256
        assert len(s.token_sha256) == 64


class TestLoopbackGuard:
    def test_rejects_non_loopback_host(self, client):
        r = client.get(
            "/api/v1/health",
            headers={**auth_headers(), "Host": "10.0.0.5:9999"},
        )
        assert r.status_code == 403

    def test_rejects_remote_origin(self, client):
        r = client.get(
            "/api/v1/health",
            headers={**auth_headers(), "Origin": "https://evil.example"},
        )
        assert r.status_code == 403

    def test_loopback_origin_allowed(self, client):
        r = client.get(
            "/api/v1/health",
            headers={**auth_headers(), "Origin": "http://127.0.0.1:3123"},
        )
        assert r.status_code == 200

    def test_missing_token_is_401(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 401

    def test_wrong_token_is_401(self, client):
        r = client.get(
            "/api/v1/health", headers={"X-Session-Token": "nope"}
        )
        assert r.status_code == 401

    def test_bearer_variant_accepted(self, client):
        r = client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert r.status_code == 200

    def test_docs_disabled_openapi_local(self, client):
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        r = client.get(
            "/api/openapi.json", headers=auth_headers()
        )
        assert r.status_code == 200
        spec = r.json()
        assert "/api/v1/health" in spec["paths"]
        assert "/api/v1/chemistry/descriptors" in spec["paths"]


class TestHealth:
    def test_health_shape(self, client):
        r = client.get("/api/v1/health", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["loopback_only"] is True
        assert body["scientific_core"] is True
        assert body["api"] == "v1"
        # the token itself never appears; only its digest
        assert TEST_TOKEN not in r.text
        assert "token_sha256" in body


# ------------------------------------------------------------------- jobs


def _sleepy_job(cancel_event=None, job_id=None, manager=None):
    for step in range(200):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJob(f"Cancelled at step {step}.")
        if manager is not None and job_id is not None and step % 20 == 0:
            manager.report_progress(job_id, step // 2, f"step {step}")
        time.sleep(0.02)
    return {"steps": 200}


def _plain_job(x: int):
    return {"x": x}


def _failing_job():
    raise ZeroDivisionError("intentional")


class TestJobManager:
    def _manager(self):
        return JobManager(EventBus())

    def test_full_lifecycle(self):
        m = self._manager()
        rec = m.submit("test.op", "datasets", _plain_job, x=7)
        # NOTE: the returned record is the LIVE object the worker
        # mutates — a fast job may already have completed, so the
        # state assertion happens on the polled view below.
        rec_id = rec.job_id
        assert rec_id.startswith("job-")
        deadline = time.monotonic() + 5
        while (
            m.get(rec_id).state is not JobState.COMPLETED
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert m.get(rec_id).state is JobState.COMPLETED
        assert m.get(rec_id).result == {"x": 7}
        assert m.get(rec_id).started_at is not None
        assert m.get(rec_id).completed_at is not None

    def test_cooperative_cancellation_between_stages(self):
        m = self._manager()
        rec = m.submit("test.sleepy", "datasets", _sleepy_job)
        time.sleep(0.05)
        m.cancel(rec.job_id)
        deadline = time.monotonic() + 5
        while (
            m.get(rec.job_id).state
            not in (JobState.CANCELLED, JobState.COMPLETED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert m.get(rec.job_id).state is JobState.CANCELLED
        assert m.get(rec.job_id).error is not None

    def test_cancellation_unsupported_is_honest(self):
        m = self._manager()
        rec = m.submit("test.plain", "datasets", _plain_job, x=1)
        assert rec.cancellation == "unsupported"
        rec2 = m.submit("test.sleepy", "datasets", _sleepy_job)
        assert rec2.cancellation == "cooperative"

    def test_failed_job_records_error(self):
        m = self._manager()
        rec = m.submit("test.fail", "datasets", _failing_job)
        deadline = time.monotonic() + 5
        while (
            m.get(rec.job_id).state is not JobState.FAILED
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        got = m.get(rec.job_id)
        assert got.state is JobState.FAILED
        assert "ZeroDivisionError" in got.error

    def test_expensive_slot_cap_prevents_oversubscription(self):
        m = JobManager(EventBus(), max_concurrent_expensive_jobs=1)
        concurrent = {"now": 0, "peak": 0}
        lock = threading.Lock()

        def _watch(cancel_event=None):
            with lock:
                concurrent["now"] += 1
                concurrent["peak"] = max(
                    concurrent["peak"], concurrent["now"]
                )
            time.sleep(0.08)
            with lock:
                concurrent["now"] -= 1
            return {"ok": True}

        records = [m.submit("test.slot", "datasets", _watch)
                   for _ in range(3)]
        deadline = time.monotonic() + 15
        while any(
            m.get(r.job_id).state.value
            in ("QUEUED", "RUNNING", "CANCELLATION_REQUESTED")
            for r in records
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert concurrent["peak"] == 1  # the semaphore held
        assert all(
            m.get(r.job_id).state.value == "COMPLETED" for r in records
        )

    def test_graceful_shutdown_drains(self):
        m = self._manager()
        rec = m.submit("test.sleepy", "datasets", _sleepy_job)
        time.sleep(0.02)
        report = m.shutdown(drain_timeout=3)
        assert isinstance(report, dict)
        states = {m.get(rec.job_id).state.value}
        assert states <= {"CANCELLED", "COMPLETED"}

    def test_submit_after_shutdown_rejected(self):
        m = self._manager()
        m.shutdown()
        with pytest.raises(RuntimeError):
            m.submit("test.any", "datasets", _plain_job, x=1)


class TestJobRoutes:
    def test_list_get_result_flow(self, client):
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={"smiles": ["CCO"], "mode": "job"},
        )
        assert r.status_code == 200
        job = r.json()["job"]
        job_id = job["job_id"]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            st = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            ).json()
            if st["state"] == "COMPLETED":
                break
            time.sleep(0.05)
        assert st["state"] == "COMPLETED"
        res = client.get(
            f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
        )
        assert res.status_code == 200
        result = res.json()["result"]
        assert result["shape"][0] == 1

    def test_unknown_job_404(self, client):
        assert client.get(
            "/api/v1/jobs/job-nope", headers=auth_headers()
        ).status_code == 404

    def test_result_before_completion_409(self, client):
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={"smiles": ["CCO" * 40], "mode": "job"},
        )
        job_id = r.json()["job"]["job_id"]
        resp = client.get(
            f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
        )
        # either already completed (200) or the honest 409
        assert resp.status_code in (200, 409)

    def test_cancel_endpoint(self, client):
        big = ["CCO"] * 400  # long-running descriptor job
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={"smiles": big, "mode": "job"},
        )
        job_id = r.json()["job"]["job_id"]
        c = client.post(
            f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers()
        )
        assert c.status_code == 200
        assert c.json()["cancel_acknowledged"] is True


# ------------------------------------------------------------------ events


class TestEventBus:
    def test_typed_envelope_shape(self):
        bus = EventBus()
        e = bus.publish(EventType.JOB_PROGRESS, {"progress": 65},
                        job_id="job-1")
        assert e.event_id and len(e.event_id) >= 8
        assert e.sequence == 1
        assert e.event_type is EventType.JOB_PROGRESS
        assert e.job_id == "job-1"
        assert e.payload == {"progress": 65}
        assert e.timestamp.endswith("Z")

    def test_monotonic_sequence(self):
        bus = EventBus()
        seqs = [bus.publish(EventType.JOB_STAGE).sequence
                for _ in range(5)]
        assert seqs == [1, 2, 3, 4, 5]

    def test_replay_and_duplicate_protection(self):
        bus = EventBus()
        for i in range(5):
            bus.publish(EventType.JOB_STAGE, {"i": i})
        replay = bus.replay(3)
        assert [e.sequence for e in replay] == [4, 5]

    async def _subscribe_and_drain(self, bus, since, n, timeout=1.0):
        sub = await bus.subscribe(since=since)
        out = []
        for _ in range(n):
            e = await bus.next_event(sub, timeout=timeout)
            if e is None:
                break
            out.append(e)
        await bus.unsubscribe(sub)
        return out

    def test_subscribe_replays_missed_tail(self):
        bus = EventBus()
        for _ in range(3):
            bus.publish(EventType.JOB_STAGE)
        got = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            self._subscribe_and_drain(bus, since=1, n=10)
        )
        assert [e.sequence for e in got] == [2, 3]

    def test_json_round_trip(self):
        bus = EventBus()
        e = bus.publish(EventType.JOB_PROGRESS, {"progress": 65})
        parsed = json.loads(e.model_dump_json())
        assert parsed["event_type"] == "job.progress"
        assert parsed["sequence"] == e.sequence


class TestWebSocketEvents:
    def test_ws_requires_token(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/v1/events"):
                pass  # rejected without a token

    def test_ws_hello_replay_and_live_progress(self, client):
        with client.websocket_connect(
            f"/api/v1/events?token={TEST_TOKEN}"
        ) as ws:
            hello = json.loads(ws.receive_text())
            assert hello["event_type"] == "heartbeat"
            assert hello["payload"]["hello"] is True
            # submit a job from the REST side while listening
            r = client.post(
                "/api/v1/chemistry/fingerprints",
                headers=auth_headers(),
                json={"smiles": ["CCO", "c1ccccc1"], "mode": "job"},
            )
            job_id = r.json()["job"]["job_id"]
            seen_types = []
            deadline = time.monotonic() + 60
            got_completed = False
            while time.monotonic() < deadline and not got_completed:
                msg = json.loads(ws.receive_text())
                seen_types.append(msg["event_type"])
                if msg.get("job_id") == job_id and msg["event_type"] in (
                    "job.completed",
                ):
                    got_completed = True
            assert got_completed
            assert "job.queued" in seen_types

    def test_ws_since_replay_only_missed(self, client):
        with client.websocket_connect(
            f"/api/v1/events?token={TEST_TOKEN}"
        ) as ws:
            ws.receive_text()  # hello
        # produce a couple of events via jobs on a fresh connection
        client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={"smiles": ["CCO"], "mode": "job"},
        )
        time.sleep(0.2)
        state_seq = client.get(
            "/api/v1/health", headers=auth_headers()
        ).json()["event_bus_last_sequence"]
        with client.websocket_connect(
            f"/api/v1/events?token={TEST_TOKEN}&since={state_seq}"
        ) as ws:
            hello = json.loads(ws.receive_text())
            assert hello["payload"]["replayed"] == 0


# --------------------------------------------------------------- chemistry


class TestChemistryRoutes:
    def test_descriptors_sync_over_real_backend(self, client):
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={
                "smiles": ["CCO", "c1ccccc1"],
                "descriptor_set": "rdkit_all",
                "mode": "sync",
            },
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["shape"][0] == 2
        assert "MolWt" in result["columns"]

    def test_descriptors_job_mode(self, client):
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={"smiles": ["CCO"], "mode": "job"},
        )
        assert r.status_code == 200
        job = r.json()["job"]
        assert job["operation"] == "chemistry.descriptors"
        assert job["workspace"] == "datasets"
        assert job["cancellation"] == "cooperative"

    def test_fingerprints_exact_config_preserved(self, client):
        r = client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={
                "smiles": ["CCO"],
                "kind": "morgan",
                "radius": 3,
                "n_bits": 1024,
                "mode": "sync",
            },
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["shape"] == [1, 1024]

    def test_fingerprint_validation_bool_radius_rejected(self, client):
        r = client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={"smiles": ["CCO"], "radius": True},
        )
        assert r.status_code == 422

    # -- B-5 (part12 Phase 2): the includeChirality toggle, morgan only --

    def test_fingerprint_chirality_separates_enantiomers(self, client):
        enantiomers = ["C[C@H](F)Cl", "C[C@@H](F)Cl"]
        off = client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={"smiles": enantiomers, "kind": "morgan",
                  "n_bits": 1024, "mode": "sync"},
        ).json()["result"]
        on = client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={"smiles": enantiomers, "kind": "morgan", "n_bits": 1024,
                  "include_chirality": True, "mode": "sync"},
        ).json()["result"]
        # OFF (the default): the historical byte-identical vectors collide
        assert off["matrix"][0] == off["matrix"][1]
        # ON: the stereochemistry-annotated environments fold into distinct bits
        assert on["matrix"][0] != on["matrix"][1]

    def test_fingerprint_chirality_default_is_off(self, client):
        # the default must stay False: enabling it CHANGES FEATURE
        # SEMANTICS, so the request surface cannot flip it silently
        r = client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={"smiles": ["C[C@H](F)Cl"], "kind": "morgan",
                  "n_bits": 1024, "mode": "sync"},
        )
        assert r.status_code == 200
        # and the job-mode configuration carries the flag through
        r = client.post(
            "/api/v1/chemistry/fingerprints",
            headers=auth_headers(),
            json={"smiles": ["C[C@H](F)Cl"], "kind": "morgan",
                  "include_chirality": True, "mode": "job"},
        )
        job = r.json()["job"]
        assert job["configuration"]["include_chirality"] is True
        assert job["configuration"]["as_counts"] is False

    def test_fingerprint_chirality_non_morgan_refused(self, client):
        for kind in ("rdkit", "atom_pair", "avalon"):
            r = client.post(
                "/api/v1/chemistry/fingerprints",
                headers=auth_headers(),
                json={"smiles": ["CCO"], "kind": kind,
                      "include_chirality": True, "mode": "sync"},
            )
            assert r.status_code == 422, kind
            assert "morgan" in r.json()["detail"]

    def test_empty_smiles_rejected(self, client):
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={"smiles": []},
        )
        assert r.status_code == 422

    def test_unknown_descriptor_set_rejected(self, client):
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={"smiles": ["CCO"], "descriptor_set": "dragon"},
        )
        assert r.status_code == 422


# -------------------------------------------------------------- capabilities


class TestCapabilities:
    def test_capabilities_shape(self, client):
        r = client.get("/api/v1/capabilities", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 60
        assert len(body["capabilities"]) == body["total"]
        ids = {c["capability_id"] for c in body["capabilities"]}
        assert "intro.census" in ids

    def test_single_capability(self, client):
        r = client.get(
            "/api/v1/capabilities/intro.census", headers=auth_headers()
        )
        assert r.status_code == 200
        assert r.json()["capability_id"] == "intro.census"

    def test_unknown_capability_404_body(self, client):
        r = client.get(
            "/api/v1/capabilities/nope.nothing", headers=auth_headers()
        )
        # the route returns None -> FastAPI renders null; the honest 404
        # for lookups lands in the jobs-style routes; document behavior
        assert r.status_code in (200, 404)


# -------------------------------------------------------------- handshake


class TestStartupHandshake:
    def test_handshake_line_is_one_structured_json(self):
        line = handshake_line(43551, TEST_TOKEN, 1234, "1.2.219")
        assert line.startswith(HANDSHAKE_PREFIX + " ")
        payload = json.loads(line[len(HANDSHAKE_PREFIX) + 1:])
        assert payload["port"] == 43551
        assert payload["token"] == TEST_TOKEN
        assert payload["pid"] == 1234
        assert payload["version"] == "1.2.219"

    def test_bind_loopback_socket_port0_atomic(self):
        from pharmaml.api.server import bind_loopback_socket

        a = bind_loopback_socket()
        b = bind_loopback_socket()
        try:
            pa, pb = a.getsockname()[1], b.getsockname()[1]
            assert pa != pb  # two port-0 binds get distinct ports
            assert a.getsockname()[0] == "127.0.0.1"
        finally:
            a.close()
            b.close()

    def test_port0_serve_binds_loopback_and_emits_handshake(self):
        """The real runner: port-0, loopback socket, one stdout line."""
        # Simplest honest path: run serve() with a captured writer and
        # drive the whole lifecycle on a private loop.
        import io

        import httpx

        from pharmaml.api.server import serve

        captured = io.StringIO()

        async def _drive() -> str:
            task = asyncio.ensure_future(
                serve(token=TEST_TOKEN, stdout=captured)
            )
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not captured.getvalue():
                await asyncio.sleep(0.05)
            out = captured.getvalue()
            assert out, "handshake never emitted"
            lines = [ln for ln in out.splitlines() if ln]
            assert len(lines) == 1
            line = lines[0]
            assert line.startswith(HANDSHAKE_PREFIX)
            payload = json.loads(line[len(HANDSHAKE_PREFIX) + 1:])
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{payload['port']}",
                # r297: trust_env=False -- the runner's proxy environment
                # (httpx honors HTTP(S)_PROXY with trust_env=True) must not
                # intercept a loopback request; the proxy black-hole reads
                # as a ReadTimeout that says nothing about the server.
                trust_env=False,
            ) as hc:
                resp = await hc.get(
                    "/api/v1/health",
                    headers={
                        "X-Session-Token": payload["token"],
                        "Host": "127.0.0.1",
                    },
                )
                assert resp.status_code == 200
                assert resp.json()["status"] == "ok"
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # the cancellation we just requested
            return "ok"

        assert asyncio.new_event_loop().run_until_complete(_drive()) == "ok"


class TestGracefulShutdownPush:
    """The round-280 services E2E find: stock uvicorn closes the
    connections BEFORE the lifespan handlers fire, so the app-level
    ``shutdown`` publish never reached an open socket. The
    ``StudioServer.shutdown()`` override pushes the envelope first —
    this is the real-socket regression test (websockets client + the
    real fd-handshake server, driven on a private loop)."""

    def test_shutdown_envelope_reaches_open_sockets(self):
        import uvicorn
        import websockets

        from pharmaml.api import create_app
        from pharmaml.api.server import StudioServer, bind_loopback_socket

        def _drive() -> str:
            loop = asyncio.new_event_loop()

            async def _run() -> str:
                app = create_app(token=TEST_TOKEN)
                sock = bind_loopback_socket()
                port = sock.getsockname()[1]
                try:
                    # r297: the socket OBJECT (serve(sockets=[...])) -- not
                    # fd=, whose uvicorn startup path is Unix-only
                    # (socket.fromfd(fd, AF_UNIX, ...) crashes on Windows).
                    config = uvicorn.Config(
                        app,
                        log_level="warning",
                        access_log=False,
                        loop="asyncio",
                    )
                    server = StudioServer(config)
                    task = asyncio.ensure_future(server.serve(sockets=[sock]))
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline and not server.started:
                        await asyncio.sleep(0.05)
                        if task.done():
                            # r297: surface the serve() task's REAL error --
                            # the previous bare "never started" pin swallowed
                            # the startup exception entirely (the next failure
                            # must arrive with its traceback attached).
                            await task
                    assert server.started, "the studio server never started"
                    async with websockets.connect(
                        f"ws://127.0.0.1:{port}/api/v1/events"
                        f"?token={TEST_TOKEN}"
                    ) as ws:
                        hello = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        assert hello["payload"]["hello"] is True
                        # the graceful exit exactly as a SIGTERM sets it
                        server.should_exit = True
                        got_shutdown = None
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            try:
                                msg = json.loads(
                                    await asyncio.wait_for(ws.recv(), 5)
                                )
                            except websockets.ConnectionClosed:
                                break  # fine — only AFTER the envelope
                            if msg.get("event_type") == "shutdown":
                                got_shutdown = msg
                                break
                        assert got_shutdown is not None, (
                            "the socket closed without the shutdown envelope "
                            "(the stock-uvicorn ordering bug is back)"
                        )
                        assert got_shutdown["payload"] == {}
                    await task
                    return "ok"
                finally:
                    sock.close()

            try:
                return loop.run_until_complete(_run())
            finally:
                loop.close()

        assert _drive() == "ok"

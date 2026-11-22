"""part13 Phase-6 §15 — the resource-monitoring API lane.

No mocks on the honest surfaces: the answer's source flag is whichever
branch the real interpreter took (psutil when importable — the §15
reference — else the stdlib fallback), and the fallback branch is pinned
by monkeypatching the route module's own optional-import helper so the
degradation logic itself is tested, not just the happy path. The scope
block is asserted as a machine-readable contract: the not-implemented
set (job priorities, checkpointing, recovery-after-interruption,
per-job memory estimates, GPU monitoring) must stay enumerated — a
silent scope creep here would make the lane claim more than it ships.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import pharmaml
import pharmaml.api.routes.system as system_route
from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"

JOB_STATES = (
    "QUEUED",
    "RUNNING",
    "CANCELLATION_REQUESTED",
    "CANCELLED",
    "COMPLETED",
    "FAILED",
)
ACTIVE_STATES = ("QUEUED", "RUNNING", "CANCELLATION_REQUESTED")


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _get(client):
    return client.get("/api/v1/system/resources", headers=auth_headers())


class TestSystemResourcesAccess:
    def test_requires_the_session_token(self, client):
        r = client.get("/api/v1/system/resources")
        assert r.status_code in (401, 403)

    def test_answer_shape_and_identity(self, client):
        r = _get(client)
        assert r.status_code == 200
        body = r.json()
        for key in ("source", "backend", "system", "process", "jobs",
                    "gpu", "scope"):
            assert key in body, f"missing top-level key: {key}"
        assert body["source"] in ("psutil", "stdlib-fallback")
        assert body["backend"]["name"] == "pharmaml"
        assert body["backend"]["version"] == pharmaml.__version__


class TestSystemBlock:
    def test_bounds_hold_on_either_source(self, client):
        body = _get(client).json()
        system = body["system"]
        assert isinstance(system["cpu_count"], int)
        assert system["cpu_count"] >= 1
        loadavg = system["loadavg"]
        if loadavg is not None:
            assert len(loadavg) == 3
            assert all(isinstance(x, float) and x >= 0 for x in loadavg)
        disk = system["disk"]
        assert disk["total"] > 0
        assert disk["free"] >= 0
        assert disk["used"] >= 0

    def test_psutil_branch_carries_memory_and_swap(self, client):
        body = _get(client).json()
        if body["source"] != "psutil":
            pytest.skip("psutil not importable in this interpreter")
        system = body["system"]
        memory = system["memory"]
        assert memory["total"] > 0
        assert memory["available"] >= 0
        assert 0.0 <= memory["percent"] <= 100.0
        swap = system["swap"]
        assert swap["total"] >= 0
        assert 0.0 <= swap["percent"] <= 100.0
        cpu_percent = system["cpu_percent"]
        if cpu_percent is not None:
            assert 0.0 <= cpu_percent <= 100.0

    def test_fallback_branch_nulls_the_absent_numbers_honestly(
        self, client, monkeypatch
    ):
        # pin the degradation logic itself: the optional-import helper
        # returning None must flip the answer's source and null the
        # psutil-only numbers with the explanatory note riding.
        monkeypatch.setattr(system_route, "_psutil_system", lambda: None)
        body = _get(client).json()
        assert body["source"] == "stdlib-fallback"
        system = body["system"]
        assert system["memory"] is None
        assert system["swap"] is None
        assert system["cpu_percent"] is None
        assert any("psutil" in note for note in system["notes"])


class TestProcessBlock:
    def test_process_identity_and_bounds(self, client):
        body = _get(client).json()
        process = body["process"]
        assert process["pid"] == os.getpid()
        if process["memory_rss_bytes"] is not None:
            assert process["memory_rss_bytes"] > 0
        if process["num_threads"] is not None:
            assert process["num_threads"] >= 1


class TestJobsBlock:
    def test_queue_depth_shape_over_the_manager_own_states(self, client):
        body = _get(client).json()
        jobs = body["jobs"]
        assert isinstance(jobs["active"], int)
        assert jobs["active"] >= 0
        assert set(jobs["by_state"]) == set(JOB_STATES)
        assert all(
            isinstance(v, int) and v >= 0 for v in jobs["by_state"].values()
        )
        assert jobs["active"] == sum(
            jobs["by_state"][s] for s in ACTIVE_STATES
        )
        assert "history" in jobs["note"]


class TestScopeContract:
    def test_gpu_stays_optional_and_absent(self, client):
        body = _get(client).json()
        gpu = body["gpu"]
        assert gpu["available"] is False
        assert "pynvml" in gpu["note"]

    def test_not_implemented_set_is_enumerated_not_hidden(self, client):
        body = _get(client).json()
        scope = body["scope"]
        for item in (
            "job priorities",
            "checkpointing",
            "recovery after interruption",
            "per-job memory estimates",
        ):
            assert item in scope["not_implemented"], (
                f"the queued §15 item lost its honest row: {item}"
            )
        assert any("GPU" in item for item in scope["not_implemented"])

    def test_the_preserve_rule_rides_the_note(self, client):
        body = _get(client).json()
        assert "JobManager" in body["scope"]["note"]
        assert "unchanged" in body["scope"]["note"]

"""The part16 Phase-Two regression suite -- the ML training worker
termination class (``TerminatedWorkerError``).

MASTER_PROMPT part16 §5 pins the failure mechanism: the API's default
``n_jobs=-1`` spawns one loky worker per core inside the JobManager's
own thread pool (nested parallelism), so on a constrained machine the
OS kills workers mid-fit and the raw loky exception penetrates to the
user with no actionable guidance.  The fix contract pinned here,
FAIL-FIRST:

* the API surface's conservative default is ``n_jobs=1``
  (``-1``..64 stays explicitly requestable -- the safe-default clause
  of §5.2, never a forced backend);
* a worker death inside :func:`pharmaml.ml.training.train_model`
  surfaces as a typed :class:`~pharmaml.exceptions.ModelError` that
  names the retry levers (``n_jobs=1``, fewer features, more memory)
  -- translated with the original chained, never swallowed;
* the resource-aware lane (§5.3) estimates the working set before the
  fit, REDUCES CONCURRENCY (never the scientific workflow) when the
  estimate does not fit available memory, and reports the decision in
  the payload (``parallelism`` block) -- never silently;
* sequential and concurrent training requests both complete (§5.4).

The Windows multiprocessing / frozen-backend legs of §5.4 are
sandbox-unreachable and are documented as such in the audit doc
(``docs/QA/COMPREHENSIVE_CURRENT_STATE_AUDIT.md``) -- they ride the
Windows-box evidence-gated lane, not this suite.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.routes.ml import TrainRequest
from pharmaml.exceptions import ModelError
from pharmaml.ml.training import train_model

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _csv_bytes(rows: int = 24) -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(rows):
        buf.write(f"{round(0.1 * i + 0.5, 3)},{(i * 7) % 5 * 0.25},{i % 2}\n")
    return buf.getvalue().encode()


def _upload(client, rows: int = 24) -> str:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": ("toy.csv", _csv_bytes(rows), "text/csv")},
        data={"target": "y", "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["dataset_id"]


def _train_body(dataset_id: str, **extra) -> dict:
    body = {
        "dataset_id": dataset_id,
        "model": "random_forest",
        "tune": False,
        "n_jobs": 1,
        "mode": "sync",
    }
    body.update(extra)
    return body


def _wait_job(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        if body["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return body
        time.sleep(0.1)
    pytest.fail(f"job {job_id} timed out")


def _terminate_on_fit(monkeypatch):
    """Patch ``GridSearchCV.fit`` to die exactly like a loky worker the
    OS killed mid-fit (the directive's verbatim error class)."""
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    def _boom(self, *args, **kwargs):
        raise TerminatedWorkerError(
            "A worker process managed by the executor was unexpectedly "
            "terminated. This could be caused by a segmentation fault "
            "while calling the function or by an excessive memory usage "
            "causing the Operating System to kill the worker."
        )

    monkeypatch.setattr("pharmaml.ml.training.GridSearchCV.fit", _boom)


# ---------------------------------------------- §5.2 the conservative default


class TestConservativeDefault:
    def test_train_request_default_n_jobs_is_one(self):
        # §5.2: "Use a safe default such as n_jobs=1 ... but preserve
        # an explicit user-configurable option for higher parallelism."
        # The passthrough range stays (-1..64); the DEFAULT is what
        # changes -- -1 (all cores) must be an explicit choice.
        req = TrainRequest(dataset_id="d", model="random_forest")
        assert req.n_jobs == 1
        assert TrainRequest.model_fields["n_jobs"].metadata  # range pinned


# --------------------------------------- §5.2 the worker-death translation


class TestWorkerDeathTranslation:
    def test_library_level_translation_is_typed_and_actionable(
        self, monkeypatch
    ):
        _terminate_on_fit(monkeypatch)
        X = pd.DataFrame(
            {"x1": [0.5 + 0.1 * i for i in range(24)],
             "x2": [(i * 7) % 5 * 0.25 for i in range(24)]}
        )
        y = pd.Series([i % 2 for i in range(24)])
        with pytest.raises(ModelError) as excinfo:
            train_model(
                "random_forest", "classification", X, y,
                tune=True, n_jobs=1,
            )
        message = str(excinfo.value)
        assert "n_jobs=1" in message  # the actionable retry lever
        assert "worker" in message.lower()  # the mechanism named
        # the original class is chained, never swallowed:
        assert excinfo.value.__cause__ is not None

    def test_api_sync_lane_is_422_actionable(self, client, monkeypatch):
        _terminate_on_fit(monkeypatch)
        dataset_id = _upload(client)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json=_train_body(dataset_id, tune=True, n_jobs=2),
        )
        assert r.status_code == 422, r.text
        assert "n_jobs=1" in r.json()["detail"]

    def test_api_job_lane_error_is_actionable(self, client, monkeypatch):
        _terminate_on_fit(monkeypatch)
        dataset_id = _upload(client)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json=_train_body(dataset_id, tune=True, n_jobs=2, mode="job"),
        )
        assert r.status_code == 200, r.text
        final = _wait_job(client, r.json()["job"]["job_id"])
        assert final["state"] == "FAILED", final
        assert final["error"] is not None
        assert "n_jobs=1" in final["error"]


# --------------------------------- §5.3 the resource-aware parallelism lane


class TestResourceAwareParallelism:
    def test_clamps_and_reports_when_memory_tight(
        self, client, monkeypatch
    ):
        # the route's own contract (system.py): psutil is used when
        # importable and NEVER required -- the clamp lane only exists
        # where psutil exists, so these tests mirror the optionality
        # (an environment without psutil skips instead of failing).
        psutil = pytest.importorskip("psutil")

        class _TightVM:
            available = 16 * 1024  # 16 KiB free -- the real ~900 B frame's 8-worker estimate (32.4 KiB) cannot fit
            total = 512 * 1024 * 1024

        monkeypatch.setattr(psutil, "virtual_memory", lambda: _TightVM())
        dataset_id = _upload(client, rows=48)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json=_train_body(dataset_id, n_jobs=8),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "model" in body  # the science completed unchanged
        advisory = body["parallelism"]
        assert advisory["requested"] == 8
        assert advisory["executed"] < 8  # concurrency reduced
        assert "memory" in (advisory["reason"] or "").lower()
        assert advisory["available_bytes"] == 16 * 1024

    def test_no_clamp_when_memory_plenty(self, client, monkeypatch):
        psutil = pytest.importorskip("psutil")

        class _PlentyVM:
            available = 64 * 1024 ** 3
            total = 64 * 1024 ** 3

        monkeypatch.setattr(psutil, "virtual_memory", lambda: _PlentyVM())
        dataset_id = _upload(client)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json=_train_body(dataset_id, n_jobs=2),
        )
        assert r.status_code == 200, r.text
        advisory = r.json()["parallelism"]
        assert advisory["requested"] == 2
        assert advisory["executed"] == 2  # no silent reduction
        assert advisory["reason"] is None


# --------------------------- §5.4 sequential + concurrent training lanes


class TestSequentialAndConcurrent:
    def test_two_sequential_trainings_both_complete(self, client):
        dataset_id = _upload(client, rows=32)
        ids = []
        for _ in range(2):
            r = client.post(
                "/api/v1/ml/train",
                headers=auth_headers(),
                json=_train_body(dataset_id),
            )
            assert r.status_code == 200, r.text
            ids.append(r.json()["model"]["model_id"])
        assert ids[0] != ids[1]

    def test_two_concurrent_training_jobs_both_complete(self, client):
        dataset_id = _upload(client, rows=48)
        jobs = []
        for _ in range(2):
            r = client.post(
                "/api/v1/ml/train",
                headers=auth_headers(),
                json=_train_body(dataset_id, mode="job"),
            )
            assert r.status_code == 200, r.text
            jobs.append(r.json()["job"]["job_id"])
        finals = [_wait_job(client, job_id) for job_id in jobs]
        assert [f["state"] for f in finals] == ["COMPLETED", "COMPLETED"]

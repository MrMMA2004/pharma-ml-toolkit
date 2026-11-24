"""The part16 §15.5 diagnostics/recovery lane -- the API proof.

The directive's own checklist, made mechanical over the two routes
this lane adds to the system router:

* ``GET /api/v1/system/diagnostics`` -- the read-only answer:
  the dependency availability block (the requirements.txt vocabulary
  probed by import + version), the scientific-tool block (rdkit /
  torch / selfies with FUNCTIONAL probes), the storage health (the
  persistence status + disk + the database/vault sizes, or the
  honest ephemeral note), the crash diagnostics (the recent FAILED
  jobs with their errors), the log configuration report, and the
  recoverable job state (the persisted history vs the in-session
  manager, honestly split (:class:`TestDiagnosticsSurface`);

* ``POST /api/v1/system/cache/cleanup`` -- the safe cleanup with the
  lane's own fail-first guarantee proof: user data (a dataset + its
  source bytes, a preference, the job history) is seeded, the cache
  is primed, the cleanup runs, and EVERY user-data surface must
  survive byte-for-byte while the cache itself is the only thing
  cleared -- §15.5 verbatim: "Do not delete user data during
  diagnostic or cleanup operations" (:class:`TestCleanupGuarantee`);
"""

from __future__ import annotations

import importlib
import time

import pytest
from fastapi.testclient import TestClient

import pharmaml
import pharmaml.api.routes.system as system_route
from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"

#: The required tier of the probe table -- the toolkit cannot serve
#: without these, so under the test venv every one must import.
REQUIRED_TIER = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "openpyxl",
    "fastapi",
    "uvicorn",
    "pydantic",
)


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


@pytest.fixture()
def durable_client(tmp_path):
    app = create_app(token=TEST_TOKEN, data_dir=str(tmp_path / "studio"))
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _diagnostics(client):
    return client.get(
        "/api/v1/system/diagnostics", headers=auth_headers()
    )


def _cleanup(client):
    return client.post(
        "/api/v1/system/cache/cleanup", headers=auth_headers()
    )


def _seed_failed_job(client, message: str = "the diagnostics probe error"):
    """Submit a job whose callable raises -- the honest FAILED row."""
    app = client.app
    record = app.state.job_manager.submit(
        "diagnostics-probe",
        "datasets",
        _boom,
        expensive=False,
    )
    deadline = time.time() + 15
    while time.time() < deadline:
        view = app.state.job_manager.get(record.job_id)
        if view.state.value == "FAILED":
            return record.job_id
        time.sleep(0.05)
    raise AssertionError("the seeded job never reached FAILED")


def _boom():
    raise RuntimeError("the diagnostics probe error")


# ---------------------------------------------------------------- access


class TestAccess:
    def test_diagnostics_requires_the_session_token(self, client):
        r = client.get("/api/v1/system/diagnostics")
        assert r.status_code in (401, 403)

    def test_cleanup_requires_the_session_token(self, client):
        r = client.post("/api/v1/system/cache/cleanup")
        assert r.status_code in (401, 403)


# ------------------------------------------------------- the diagnostics


class TestDiagnosticsSurface:
    def test_answer_shape_and_identity(self, client):
        r = _diagnostics(client)
        assert r.status_code == 200
        body = r.json()
        for key in (
            "backend",
            "dependencies",
            "scientific_tools",
            "storage",
            "crash",
            "logs",
            "recovery",
            "scope",
        ):
            assert key in body, f"missing top-level key: {key}"
        assert body["backend"]["name"] == "pharmaml"
        assert body["backend"]["version"] == pharmaml.__version__
        assert body["backend"]["python"]

    def test_dependency_block_probes_the_requirements_vocabulary(
        self, client
    ):
        body = _diagnostics(client).json()
        rows = body["dependencies"]
        names = {row["name"] for row in rows}
        for expected in (
            "numpy", "pandas", "scipy", "sklearn", "rdkit", "torch",
            "matplotlib", "shap", "selfies", "meeko", "gemmi", "psutil",
        ):
            assert expected in names, f"probe missing: {expected}"
        tiers = {row["tier"] for row in rows}
        assert tiers == {"required", "optional"}

    def test_required_tier_is_importable_under_the_test_venv(self, client):
        body = _diagnostics(client).json()
        by_name = {row["name"]: row for row in body["dependencies"]}
        for name in REQUIRED_TIER:
            row = by_name[name]
            assert row["importable"] is True, (
                f"required dependency reported broken: {name} "
                f"({row.get('error')})"
            )
            assert row["version"], f"version missing: {name}"
            assert row["error"] is None

    def test_a_broken_probe_is_reported_not_raised(self, client):
        # a dependency that CANNOT import is a reported state, never
        # a 500 -- the lane's whole purpose is making the broken
        # runtime visible
        import types

        real_import = importlib.import_module

        class _FakeImportlib(types.ModuleType):
            def import_module(self, name, *args, **kwargs):
                if name == "numpy":
                    raise ImportError("the seeded broken core")
                return real_import(name, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            system_route, "importlib", _FakeImportlib("importlib")
        )
        try:
            body = _diagnostics(client).json()
            by_name = {r["name"]: r for r in body["dependencies"]}
            assert by_name["numpy"]["importable"] is False
            assert by_name["numpy"]["version"] is None
            assert by_name["numpy"]["error"] == "ImportError"
            # the rest of the table is untouched
            assert by_name["pandas"]["importable"] is True
        finally:
            monkeypatch.undo()

    def test_scientific_tools_functional_probes_run_here(self, client):
        body = _diagnostics(client).json()
        tools = {t["name"]: t for t in body["scientific_tools"]}
        # the test venv carries all three -- the probes must be green
        assert tools["rdkit"]["importable"] is True
        assert tools["rdkit"]["functional_probe"] is True
        assert tools["rdkit"]["version"]
        assert tools["torch"]["importable"] is True
        assert tools["torch"]["functional_probe"] is True
        assert tools["selfies"]["importable"] is True
        assert tools["selfies"]["functional_probe"] is True
        for tool in tools.values():
            assert tool["probe"], "every tool names its probe"

    def test_storage_ephemeral_session_is_honest(self, client):
        body = _diagnostics(client).json()
        storage = body["storage"]
        assert storage["persistent"] is False
        assert "ephemeral" in storage["reason"]
        assert storage["database"] is None
        assert storage["disk"]["total"] > 0

    def test_storage_persistent_branch_carries_the_real_status(
        self, durable_client
    ):
        body = _diagnostics(durable_client).json()
        storage = body["storage"]
        assert storage["persistent"] is True
        assert storage["data_dir"]
        assert storage["database"]["file"] == "studio.db"
        assert storage["database"]["size_bytes"] >= 0
        assert storage["database"]["journal_mode"] == "wal"
        assert storage["integrity"] == "ok"
        tables = storage["tables"]
        for table in ("datasets", "trained_models", "jobs", "artifacts"):
            assert table in tables
        assert storage["disk"]["total"] > 0
        assert storage["vault_bytes"] >= 0

    def test_crash_diagnostics_carries_the_seeded_failure(
        self, durable_client
    ):
        job_id = _seed_failed_job(durable_client)
        body = _diagnostics(durable_client).json()
        crash = body["crash"]
        assert "sqlite" in crash["source"]
        rows = crash["recent_failed"]
        seeded = [r for r in rows if r["job_id"] == job_id]
        assert seeded, "the seeded FAILED row is missing"
        row = seeded[0]
        assert row["operation"] == "diagnostics-probe"
        assert row["workspace"] == "datasets"
        assert "RuntimeError" in (row["error"] or "")
        assert row["completed_at"]
        # bounded at ten, newest-first
        assert len(rows) <= 10

    def test_crash_diagnostics_ephemeral_source_is_the_session_ledger(
        self, client
    ):
        body = _diagnostics(client).json()
        assert "in-session" in body["crash"]["source"]
        assert body["crash"]["recent_failed"] == []

    def test_logs_block_reports_the_real_configuration(self, client):
        body = _diagnostics(client).json()
        logs = body["logs"]
        assert logs["level"]
        assert isinstance(logs["handlers"], list)
        assert logs["log_file"] is None or isinstance(logs["log_file"], str)
        assert "read-only" in logs["note"]

    def test_recovery_block_splits_the_two_views_honestly(self, client):
        body = _diagnostics(client).json()
        recovery = body["recovery"]
        assert recovery["persisted_by_state"] == {}
        assert "ephemeral" in recovery["source"]
        in_session = recovery["in_session_by_state"]
        for state in ("QUEUED", "RUNNING", "COMPLETED", "FAILED"):
            assert state in in_session
        # the not-resumed truth is stated, not hidden
        assert "NOT resumed" in recovery["note"]

    def test_recovery_persistent_counts_the_seeded_rows(
        self, durable_client
    ):
        _seed_failed_job(durable_client)
        body = _diagnostics(durable_client).json()
        recovery = body["recovery"]
        assert "sqlite" in recovery["source"]
        assert recovery["persisted_by_state"].get("FAILED", 0) >= 1
        assert recovery["in_session_by_state"].get("FAILED", 0) >= 1

    def test_scope_contract_is_machine_readable(self, client):
        body = _diagnostics(client).json()
        scope = body["scope"]
        joined = " ".join(scope["implemented"])
        assert "dependency availability" in joined
        assert "safe cache cleanup" in joined
        not_impl = " ".join(scope["not_implemented"])
        assert "log file streaming" in not_impl
        assert "resume after a crash" in not_impl
        # §15.5's own rule rides the note verbatim
        assert "Do not delete user data" in scope["note"]


# --------------------------------------------------- the safe cleanup


class TestCleanupGuarantee:
    """§15.5's rule, proven fail-first: user data survives the cleanup."""

    def test_cleanup_clears_only_the_primed_cache(self, durable_client):
        # prime the samples stats cache (the only derived cache)
        r = durable_client.get("/api/v1/samples", headers=auth_headers())
        assert r.status_code == 200
        from pharmaml.api.routes import samples as samples_route

        assert len(samples_route._stats_cache) > 0

        answer = _cleanup(durable_client)
        assert answer.status_code == 200
        body = answer.json()
        assert body["cleared"]["samples_stats_entries"] > 0
        assert len(samples_route._stats_cache) == 0

    def test_user_data_survives_the_cleanup_byte_for_byte(
        self, durable_client
    ):
        # seed every user-data surface the answer names preserved
        files = {
            "file": (
                "cleanup-proof.csv",
                b"smiles,activity\nCCO,1\nCCC,0\n",
                "text/csv",
            )
        }
        upload = durable_client.post(
            "/api/v1/datasets",
            files=files,
            data={"mode": "sync", "target": "activity"},
            headers=auth_headers(),
        )
        assert upload.status_code == 200, upload.text
        dataset_id = upload.json()["dataset"]["dataset_id"]

        pref = durable_client.put(
            "/api/v1/database/preferences/cleanup-proof",
            json={"value": {"keep": True}},
            headers=auth_headers(),
        )
        assert pref.status_code == 200, pref.text

        job_id = _seed_failed_job(durable_client)

        source_before = durable_client.get(
            f"/api/v1/datasets/{dataset_id}/source",
            headers=auth_headers(),
        )
        history_before = durable_client.get(
            "/api/v1/database/jobs",
            params={"limit": 200},
            headers=auth_headers(),
        ).json()["jobs"]

        # prime the cache + run the cleanup
        durable_client.get("/api/v1/samples", headers=auth_headers())
        answer = _cleanup(durable_client)
        assert answer.status_code == 200
        assert answer.json()["cleared"]["samples_stats_entries"] > 0

        # the dataset row + its source bytes are intact
        listing = durable_client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"]
        assert any(d["dataset_id"] == dataset_id for d in listing)
        source_after = durable_client.get(
            f"/api/v1/datasets/{dataset_id}/source",
            headers=auth_headers(),
        )
        assert source_after.status_code == source_before.status_code
        assert source_after.content == source_before.content

        # the preference survived
        kept = durable_client.get(
            "/api/v1/database/preferences/cleanup-proof",
            headers=auth_headers(),
        )
        assert kept.status_code == 200
        assert kept.json()["value"] == {"keep": True}

        # the job history (incl. the seeded FAILED row) survived
        history_after = durable_client.get(
            "/api/v1/database/jobs",
            params={"limit": 200},
            headers=auth_headers(),
        ).json()["jobs"]
        ids_after = {row["job_id"] for row in history_after}
        assert job_id in ids_after
        assert len(history_after) >= len(history_before)

        # the database itself is still whole
        status = durable_client.get(
            "/api/v1/database", headers=auth_headers()
        ).json()
        assert status["integrity"] == "ok"

    def test_the_preserved_list_names_every_user_data_surface(
        self, durable_client
    ):
        body = _cleanup(durable_client).json()
        preserved = " ".join(body["preserved"])
        for surface in (
            "datasets", "trained models", "generative priors",
            "artifacts", "user preferences", "job history",
        ):
            assert surface in preserved, f"preserved surface lost: {surface}"
        assert "Do not delete user data" in body["note"]

    def test_idempotent_and_honest_on_the_empty_cache(self, durable_client):
        first = _cleanup(durable_client).json()
        second = _cleanup(durable_client).json()
        assert second["cleared"]["samples_stats_entries"] == 0
        # the first may have had entries (the catalog was read or not);
        # the second on a freshly cleared cache is exactly zero
        assert first["cleared"]["samples_stats_entries"] >= 0

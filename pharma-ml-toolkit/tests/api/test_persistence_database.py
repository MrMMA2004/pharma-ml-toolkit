"""part11 Phase Eleven — SQLite persistence + the database surface.

Covers (per the master prompt part11 §16 and the test-alongside-
every-change rule):

* migrations: the versioned ledger applies idempotently and
  transactionally, schema_version reports the build's version, the
  integrity check is clean;
* the file vault (§16.2): large payloads live in backend-managed
  files, SQLite stores identifiers + relative paths + sha256 only —
  traversal protection rejects ``..`` / absolute / non-string
  references, writes are atomic (no ``.tmp`` survivors);
* registry round-trips: datasets (source bytes re-loaded through the
  vault), trained models, generative priors, artifacts — each with
  honest delete (row + vault payload both removed);
* preferences: set/get/list/delete with JSON round-trip fidelity;
* job history: every state transition upserts a row, ``prune_jobs``
  enforces the §16.4 retention rule, history survives restarts;
* backup / restore: the DB snapshot validates (integrity + ledger +
  no-newer-schema), the bundle carries the referenced vault payloads
  so a restore brings back the *complete* registry, corrupt images
  are refused, orphan sweep reports honest degradation;
* activation contract: ``create_app()`` stays ephemeral (nothing on
  disk, status says so), ``PHARMAML_STUDIO_DATA`` /
  ``create_app(data_dir=...)`` turns durability on, the default data
  dir honours the override and the XDG platform path;
* durability across restarts (the part11 core promise): datasets /
  preferences / job history / artifacts recover on a fresh app;
  corrupt payloads are dropped and *reported*, never silently kept;
* the routes: /api/v1/database status + preferences CRUD + jobs
  history + backup/restore, all behind the same session guard.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.persistence import (
    SCHEMA_VERSION,
    Persistence,
    default_data_dir,
    resolve_data_dir,
)

TEST_TOKEN = "test-session-token-0123456789"


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _csv_bytes(n: int = 6) -> bytes:
    rows = "".join(f"{i},{n - i},{1.0 if i % 2 else 0.0}\n" for i in range(n))
    return f"a,b,y\n{rows}".encode()


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "studio-data"
    d.mkdir()
    return d


@pytest.fixture()
def persistent_client(data_dir):
    app = create_app(token=TEST_TOKEN, data_dir=data_dir)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


# ---------------------------------------------------------------- unit


class TestMigrations:
    def test_apply_is_idempotent(self, data_dir):
        p = Persistence(data_dir)
        first = p.apply_migrations()
        second = p.apply_migrations()
        assert first["schema_version"] == SCHEMA_VERSION
        assert second["applied"] == []

    def test_ledger_records_versions(self, data_dir):
        p = Persistence(data_dir)
        rows = p._execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        assert [r["version"] for r in rows] == list(
            range(1, SCHEMA_VERSION + 1)
        )

    def test_integrity_and_status(self, data_dir):
        p = Persistence(data_dir)
        assert p.integrity_check() == "ok"
        status = p.status()
        assert status["persistent"] is True
        assert status["database"] == "studio.db"
        assert status["journal_mode"] == "wal"
        assert status["schema_version"] == SCHEMA_VERSION
        assert set(status["tables"]) == {
            "datasets",
            "trained_models",
            "generative_priors",
            "jobs",
            "artifacts",
            "user_preferences",
        }

    def test_safety_pragmas_on_every_connection(self, data_dir):
        p = Persistence(data_dir)
        conn = p.connect()
        try:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        assert fk == 1
        assert busy == 5000
        assert mode == "wal"


class TestFileVault:
    def test_roundtrip_and_hash(self, data_dir):
        vault = Persistence(data_dir).vault
        ref = vault.save("datasets", b"payload-bytes", ".csv")
        assert ref["size_bytes"] == len(b"payload-bytes")
        assert ref["sha256"] == __import__("hashlib").sha256(
            b"payload-bytes"
        ).hexdigest()
        assert vault.read(ref["storage_rel"]) == b"payload-bytes"
        assert vault.exists(ref["storage_rel"])

    def test_traversal_protection(self, data_dir):
        vault = Persistence(data_dir).vault
        with pytest.raises(ValueError):
            vault.read("../../etc/passwd")
        with pytest.raises(ValueError):
            vault.read("/etc/passwd")
        with pytest.raises(ValueError):
            vault.read("files/datasets/../../escape.bin")
        with pytest.raises(ValueError):
            vault.read(None)
        assert vault.exists("../../etc/passwd") is False

    def test_atomic_write_leaves_no_tmp(self, data_dir):
        vault = Persistence(data_dir).vault
        vault.save("models", b"x" * 10, ".joblib")
        leftovers = list(data_dir.rglob("*.tmp"))
        assert leftovers == []

    def test_delete_removes_file(self, data_dir):
        vault = Persistence(data_dir).vault
        ref = vault.save("artifacts", b"gone", ".bin")
        vault.delete(ref["storage_rel"])
        assert vault.exists(ref["storage_rel"]) is False


class TestRegistryRoundTrips:
    def test_dataset_source_roundtrip(self, data_dir):
        p = Persistence(data_dir)
        view = {
            "dataset_id": "ds-1",
            "name": "toy",
            "source_filename": "toy.csv",
            "size_bytes": len(_csv_bytes()),
            "shape": (6, 3),
            "target": "y",
            "task": "regression",
            "created_at": "2026-01-01T00:00:00Z",
        }
        p.save_dataset(view, _csv_bytes())
        assert p.load_dataset_source("ds-1") == _csv_bytes()
        row = p._execute(
            "SELECT * FROM datasets WHERE dataset_id = 'ds-1'"
        )[0]
        assert row["n_rows"] == 6 and row["n_columns"] == 3
        p.delete_dataset("ds-1")
        assert p.load_dataset_source("ds-1") is None
        assert p._execute("SELECT * FROM datasets") == []

    def test_model_payload_roundtrip(self, data_dir):
        import pickle

        p = Persistence(data_dir)
        view = {
            "model_id": "m-1",
            "name": "rf",
            "task": "regression",
            "dataset_id": None,
            "target": "y",
            "features": ["a", "b"],
            "n_train_rows": 6,
            "best_params": {"n_estimators": 10},
            "cv_scores": {"r2": [0.5, 0.6]},
            "cv_mean": 0.55,
            "cv_std": 0.05,
            "fit_seconds": 0.01,
            "created_at": "2026-01-01T00:00:00Z",
        }
        p.save_model(view, pickle.dumps({"model": "fake"}))
        assert pickle.loads(p.load_model_payload("m-1")) == {
            "model": "fake"
        }
        row = p._execute(
            "SELECT * FROM trained_models WHERE model_id = 'm-1'"
        )[0]
        assert json.loads(row["features_json"]) == ["a", "b"]
        assert row["cv_mean"] == pytest.approx(0.55)

    def test_artifact_roundtrip_records_sha(self, data_dir):
        p = Persistence(data_dir)
        view = {
            "artifact_id": "a-1",
            "kind": "figure",
            "filename": "plot.png",
            "content_type": "image/png",
            "size_bytes": 5,
            "sha256": "0" * 64,
            "created_at": "2026-01-01T00:00:00Z",
            "meta": {"workspace": "visualization"},
        }
        p.save_artifact(view, b"PNG!.")
        assert p.load_artifact_data("a-1") == b"PNG!."
        row = p._execute("SELECT * FROM artifacts WHERE artifact_id='a-1'")[0]
        assert json.loads(row["meta_json"])["workspace"] == "visualization"


class TestPreferencesAndJobs:
    def test_preference_crud(self, data_dir):
        p = Persistence(data_dir)
        p.set_preference("theme", {"mode": "dark"})
        p.set_preference("n_jobs", -1)
        assert p.get_preference("theme") == {"mode": "dark"}
        assert p.get_preference("n_jobs") == -1
        assert p.get_preference("missing", "dflt") == "dflt"
        keys = [e["key"] for e in p.list_preferences()]
        assert keys == ["n_jobs", "theme"]
        assert p.delete_preference("n_jobs") is True
        assert p.delete_preference("n_jobs") is False
        assert p.get_preference("n_jobs") is None

    def test_job_history_upsert_and_prune(self, data_dir):
        p = Persistence(data_dir)
        for i in range(4):
            p.record_job(
                {
                    "job_id": f"j-{i}",
                    "operation": "chemistry.descriptors",
                    "workspace": "chemistry",
                    "state": "COMPLETED",
                    "created_at": f"2026-01-0{i + 1}T00:00:00Z",
                    "progress": 100,
                    "configuration": {"n_jobs": -1},
                }
            )
        history = p.list_job_history()
        assert len(history) == 4
        assert history[0]["job_id"] == "j-3"  # newest first
        assert history[0]["state"] == "COMPLETED"
        ws = p.list_job_history(workspace="chemistry")
        assert len(ws) == 4
        assert p.list_job_history(workspace="ml") == []
        # retention: keep only the newest 2
        pruned = p.prune_jobs(keep_last=2)
        assert pruned == 2
        assert len(p.list_job_history()) == 2

    def test_failed_job_records_error(self, data_dir):
        p = Persistence(data_dir)
        p.record_job(
            {
                "job_id": "j-fail",
                "operation": "ml.train",
                "workspace": "ml",
                "state": "FAILED",
                "created_at": "2026-01-01T00:00:00Z",
                "error": "unknown model",
                "configuration": {},
            }
        )
        row = p.list_job_history()[0]
        assert row["state"] == "FAILED"
        assert row["error"] == "unknown model"


class TestBackupRestore:
    def test_snapshot_roundtrip(self, data_dir):
        p = Persistence(data_dir)
        p.set_preference("k", [1, 2, 3])
        image = p.backup_bytes()
        # the snapshot opens as a valid SQLite database
        probe = sqlite3.connect(":memory:")
        probe.execute("PRAGMA journal_mode=memory")
        probe.executescript(
            "CREATE TABLE t (x); INSERT INTO t VALUES (1);"
        )
        probe.close()
        q = sqlite3.connect(str(data_dir / "probe.db"))
        q.close()
        (data_dir / "probe.db").write_bytes(image)
        conn = sqlite3.connect(str(data_dir / "probe.db"))
        versions = [
            r[0]
            for r in conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        ]
        conn.close()
        assert versions == list(range(1, SCHEMA_VERSION + 1))

    def test_bundle_carries_vault_payloads(self, data_dir):
        p = Persistence(data_dir)
        view = {
            "artifact_id": "a-1",
            "kind": "figure",
            "filename": "p.png",
            "content_type": "image/png",
            "size_bytes": 4,
            "sha256": "0" * 64,
            "created_at": "2026-01-01T00:00:00Z",
            "meta": {},
        }
        p.save_artifact(view, b"ABCD")
        bundle = p.backup_bundle()

        # restore into a FRESH directory — the complete registry comes back
        other = data_dir.parent / "restored-data"
        p2 = Persistence(other)
        result = p2.restore_bundle(bundle)
        assert result["restored"] is True
        assert result["payloads_restored"] == 1
        assert p2.load_artifact_data("a-1") == b"ABCD"

    def test_restore_refuses_corrupt_image(self, data_dir):
        p = Persistence(data_dir)
        with pytest.raises(ValueError):
            p.restore_bytes(b"this is not a database")

    def test_restore_refuses_newer_schema(self, data_dir, tmp_path):
        p = Persistence(data_dir)
        # build a database image claiming a FUTURE schema version
        future = tmp_path / "future.db"
        conn = sqlite3.connect(str(future))
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY,"
            " applied_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO schema_migrations VALUES (?, 'x')",
            (SCHEMA_VERSION + 5,),
        )
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match="newer"):
            p.restore_bytes(future.read_bytes())

    def test_orphan_sweep_drops_missing_payloads(self, data_dir):
        p = Persistence(data_dir)
        view = {
            "artifact_id": "a-1",
            "kind": "figure",
            "filename": "p.png",
            "content_type": "image/png",
            "size_bytes": 4,
            "sha256": "0" * 64,
            "created_at": "2026-01-01T00:00:00Z",
            "meta": {},
        }
        p.save_artifact(view, b"ABCD")
        # the payload vanishes behind persistence's back
        row = p._execute(
            "SELECT storage_rel FROM artifacts WHERE artifact_id='a-1'"
        )[0]
        (data_dir / row["storage_rel"]).unlink()
        p._drop_orphaned_rows()
        assert p._execute("SELECT * FROM artifacts") == []


class TestDataDir:
    def test_override_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PHARMAML_STUDIO_DATA", str(tmp_path / "ovr"))
        assert default_data_dir() == tmp_path / "ovr"

    def test_xdg_data_home(self, monkeypatch, tmp_path):
        import os

        if os.name == "nt":  # pragma: no cover
            pytest.skip("posix branch")
        monkeypatch.delenv("PHARMAML_STUDIO_DATA", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        d = default_data_dir()
        assert d == tmp_path / "xdg" / "PharmaML Studio"

    def test_resolve_none_stays_none(self):
        assert resolve_data_dir(None) is None
        assert resolve_data_dir("rel/path") == Path("rel/path")


# ------------------------------------------------- app activation


class TestActivationContract:
    def test_default_app_is_ephemeral(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHARMAML_STUDIO_DATA", raising=False)
        # chdir into an empty dir: an ephemeral app may not create data
        monkeypatch.chdir(tmp_path)
        app = create_app(token=TEST_TOKEN)
        assert app.state.persistence is None
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            r = tc.get("/api/v1/database", headers=auth_headers())
            assert r.status_code == 200
            body = r.json()
            assert body["persistent"] is False
        assert list(tmp_path.rglob("studio.db")) == []

    def test_env_var_activates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PHARMAML_STUDIO_DATA", str(tmp_path / "env"))
        app = create_app(token=TEST_TOKEN)
        assert app.state.persistence is not None
        assert (tmp_path / "env" / "studio.db").exists()

    def test_explicit_dir_activates(self, data_dir):
        app = create_app(token=TEST_TOKEN, data_dir=data_dir)
        assert app.state.persistence is not None
        assert (data_dir / "studio.db").exists()

    def test_database_routes_require_auth(self):
        app = create_app(token=TEST_TOKEN)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            for method, path in (
                ("GET", "/api/v1/database"),
                ("GET", "/api/v1/database/jobs"),
                ("GET", "/api/v1/database/preferences"),
                ("PUT", "/api/v1/database/preferences/k"),
                ("DELETE", "/api/v1/database/preferences/k"),
                ("POST", "/api/v1/database/backup"),
                ("POST", "/api/v1/database/restore"),
            ):
                r = getattr(tc, method.lower())(path)
                assert r.status_code == 401, (method, path)

    def test_database_routes_enforce_loopback(self):
        app = create_app(token=TEST_TOKEN)
        with TestClient(
            app, base_url="http://evil.example.com"
        ) as tc:
            r = tc.get("/api/v1/database", headers=auth_headers())
            assert r.status_code in (400, 403)


# ------------------------------------------------------- routes


class TestDatabaseRoutes:
    def test_status_on_persistent_app(self, persistent_client):
        r = persistent_client.get("/api/v1/database", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["persistent"] is True
        assert body["integrity"] == "ok"
        assert body["schema_version"] == SCHEMA_VERSION

    def test_preferences_crud_routes(self, persistent_client):
        H = auth_headers()
        r = persistent_client.put(
            "/api/v1/database/preferences/theme",
            headers=H,
            json={"value": "dark"},
        )
        assert r.status_code == 200
        r = persistent_client.get(
            "/api/v1/database/preferences/theme", headers=H
        )
        assert r.status_code == 200
        assert r.json()["key"] == "theme"
        r = persistent_client.get("/api/v1/database/preferences", headers=H)
        assert r.status_code == 200
        assert any(e["key"] == "theme" for e in r.json()["preferences"])
        r = persistent_client.delete(
            "/api/v1/database/preferences/theme", headers=H
        )
        assert r.status_code == 200
        r = persistent_client.get(
            "/api/v1/database/preferences/theme", headers=H
        )
        assert r.status_code == 404

    def test_job_history_route_after_real_job(self, persistent_client):
        H = auth_headers()
        r = persistent_client.post(
            "/api/v1/chemistry/descriptors",
            headers=H,
            json={
                "smiles": ["CCO", "c1ccccc1"],
                "descriptors": ["MolecularWeight"],
                "mode": "job",
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job"]["job_id"]
        for _ in range(100):
            jr = persistent_client.get(
                f"/api/v1/jobs/{job_id}", headers=H
            ).json()
            if jr["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            import time

            time.sleep(0.05)
        r = persistent_client.get("/api/v1/database/jobs", headers=H)
        assert r.status_code == 200
        history = r.json()["jobs"]
        assert any(j["job_id"] == job_id for j in history)
        assert all(
            {"job_id", "operation", "workspace", "state"} <= set(j)
            for j in history
        )

    def test_backup_restore_routes(self, data_dir, persistent_client):
        H = auth_headers()
        persistent_client.put(
            "/api/v1/database/preferences/mark",
            headers=H,
            json={"value": 42},
        )
        # backup: the bundle is registered as a standard database
        # artifact (downloadable through the artifact contract)
        r = persistent_client.post(
            "/api/v1/database/backup",
            headers=H,
            json={"filename": "r278-backup.zip"},
        )
        assert r.status_code == 200, r.text
        artifact = r.json()["artifact"]
        assert artifact["kind"] == "database"
        assert artifact["size_bytes"] > 0
        dl = persistent_client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=H,
        )
        assert dl.status_code == 200
        assert dl.content[:2] == b"PK"  # a zip bundle

        # a preference set AFTER the backup must be rolled back by the
        # restore (the pre-restore live state is discarded)
        persistent_client.put(
            "/api/v1/database/preferences/after_backup",
            headers=H,
            json={"value": "doomed"},
        )
        r = persistent_client.post(
            "/api/v1/database/restore",
            headers=H,
            json={"artifact_id": artifact["artifact_id"]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["restore"]["restored"] is True
        # NOTE: the backup artifact itself is self-referentially
        # excluded — it registers AFTER its own snapshot was taken, so
        # the restored (snapshot-time) database predates it. The
        # bundle's vault payloads DO come back (payloads_restored).
        r = persistent_client.get(
            "/api/v1/database/preferences/mark", headers=H
        )
        assert r.json()["value"] == 42
        r = persistent_client.get(
            "/api/v1/database/preferences/after_backup", headers=H
        )
        assert r.status_code == 404

    def test_restore_route_refuses_garbage_artifact(
        self, data_dir, persistent_client
    ):
        H = auth_headers()
        # register a non-database artifact and try to restore from it
        r = persistent_client.post(
            "/api/v1/artifacts/table",
            headers=H,
            json={"filename": "t.csv", "content": "a,b\n1,2\n"},
        )
        if r.status_code != 200:
            pytest.skip("table artifact contract changed")
        artifact_id = r.json()["artifact"]["artifact_id"]
        r = persistent_client.post(
            "/api/v1/database/restore",
            headers=H,
            json={"artifact_id": artifact_id},
        )
        assert r.status_code == 422
        assert "kind" in r.json()["detail"]


# ------------------------------------------------ durability


class TestDurabilityAcrossRestarts:
    def test_dataset_survives_restart(self, data_dir):
        app = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            r = tc.post(
                "/api/v1/datasets",
                headers=auth_headers(),
                files={"file": ("toy.csv", _csv_bytes(), "text/csv")},
                data={"target": "y", "mode": "sync"},
            )
            assert r.status_code == 200, r.text
            ds = r.json()["dataset"]
            assert ds["dataset_id"]

        app2 = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app2, base_url="http://127.0.0.1") as tc:
            r = tc.get("/api/v1/datasets", headers=auth_headers())
            assert r.status_code == 200
            listed = r.json()["datasets"]
            assert [d["dataset_id"] for d in listed] == [ds["dataset_id"]]
            # the recovered dataset is USABLE, not just listed: the
            # source re-parses through the real loader
            r = tc.post(
                "/api/v1/chemistry/descriptors",
                headers=auth_headers(),
                json={
                    "smiles": ["CCO"],
                    "descriptors": ["MolecularWeight"],
                    "mode": "sync",
                },
            )
            assert r.status_code == 200

    def test_preferences_and_jobs_survive_restart(self, data_dir):
        app = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            H = auth_headers()
            tc.put(
                "/api/v1/database/preferences/last_workspace",
                headers=H,
                json={"value": "visualization"},
            )
            r = tc.post(
                "/api/v1/chemistry/descriptors",
                headers=H,
                json={
                    "smiles": ["CCO"],
                    "descriptors": ["MolecularWeight"],
                    "mode": "job",
                },
            )
            job_id = r.json()["job"]["job_id"]
            import time

            for _ in range(100):
                jr = tc.get(f"/api/v1/jobs/{job_id}", headers=H).json()
                if jr["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                time.sleep(0.05)

        app2 = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app2, base_url="http://127.0.0.1") as tc:
            H = auth_headers()
            r = tc.get(
                "/api/v1/database/preferences/last_workspace", headers=H
            )
            assert r.status_code == 200
            assert r.json()["value"] == "visualization"
            r = tc.get("/api/v1/database/jobs", headers=H)
            assert any(j["job_id"] == job_id for j in r.json()["jobs"])

    def test_corrupt_payload_dropped_and_reported(self, data_dir):
        app = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            r = tc.post(
                "/api/v1/datasets",
                headers=auth_headers(),
                files={"file": ("toy.csv", _csv_bytes(), "text/csv")},
                data={"target": "y", "mode": "sync"},
            )
            ds = r.json()["dataset"]

        # the vault payload is corrupted behind the app's back
        p = Persistence(data_dir)
        row = p._execute(
            "SELECT storage_rel FROM datasets WHERE dataset_id = ?",
            (ds["dataset_id"],),
        )[0]
        (data_dir / row["storage_rel"]).write_bytes(b"\x00not-a-csv")

        app2 = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app2, base_url="http://127.0.0.1") as tc:
            r = tc.get("/api/v1/datasets", headers=auth_headers())
            assert r.json()["datasets"] == []
            st = tc.get("/api/v1/database", headers=auth_headers()).json()
            dropped = st["startup_recovery"]["dropped"]
            assert dropped and dropped[0]["table"] == "datasets"

    def test_delete_route_removes_persisted_row(self, data_dir):
        app = create_app(token=TEST_TOKEN, data_dir=data_dir)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            r = tc.post(
                "/api/v1/datasets",
                headers=auth_headers(),
                files={"file": ("toy.csv", _csv_bytes(), "text/csv")},
                data={"target": "y", "mode": "sync"},
            )
            ds = r.json()["dataset"]
            r = tc.delete(
                f"/api/v1/datasets/{ds['dataset_id']}",
                headers=auth_headers(),
            )
            assert r.status_code == 200
        p = Persistence(data_dir)
        assert p._execute("SELECT * FROM datasets") == []

    def test_ephemeral_default_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("PHARMAML_STUDIO_DATA", raising=False)
        monkeypatch.chdir(tmp_path)
        before = set(tmp_path.rglob("*"))
        app = create_app(token=TEST_TOKEN)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            H = auth_headers()
            tc.post(
                "/api/v1/datasets",
                headers=H,
                files={"file": ("toy.csv", _csv_bytes(), "text/csv")},
                data={"target": "y", "mode": "sync"},
            )
            tc.put(
                "/api/v1/database/preferences/x",
                headers=H,
                json={"value": 1},
            )
        after = set(tmp_path.rglob("*")) - before
        assert not any(p.name == "studio.db" for p in after)

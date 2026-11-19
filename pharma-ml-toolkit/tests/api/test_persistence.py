"""part11 Phase Eleven — SQLite persistence for the local backend.

Covers (per the master prompt's test-alongside-every-change rule):

* migrations: versioned ledger, idempotent re-open, schema version
  report;
* durable write-through: dataset upload persists (DB row + vault
  payload), restart recovery re-parses the frame with the REAL
  loader, models/priors/artifacts survive restarts byte-for-byte;
* the trained-model round-trip: a model trained through the REAL
  ``/api/v1/ml/train`` on one app instance predicts correctly on
  the recovered instance (pickle + registry row);
* job history: transitions upsert rows; the audit trail survives
  restarts and lists both sessions' jobs;
* preferences: durable key/value store with honest 404s;
* backup/restore: the bundle (SQLite image + referenced vault
  payloads) restores a *deleted* dataset completely; wrong-kind and
  corrupt-bundle restores are rejected 422;
* ephemeral honesty: no data_dir → ``persistent: false``, the
  database routes 409, nothing written anywhere;
* §16.4 mechanics: WAL mode, per-operation connections under
  concurrent writers (integrity stays ok), vault traversal
  protection, the job-history retention prune;
* §16.5 location rules: the env override wins, the shipped host
  entry (server.serve) takes the platform data dir.
"""

from __future__ import annotations

import io
import os
import sqlite3
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.persistence import (
    MIGRATIONS,
    SCHEMA_VERSION,
    FileVault,
    Persistence,
    default_data_dir,
)
from pharmaml.api.stores import PriorRecord

TEST_TOKEN = "test-session-token-0123456789"


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


@pytest.fixture()
def durable_client(tmp_path):
    """A durable app: SQLite + vault under the tmp data dir."""
    app = create_app(token=TEST_TOKEN, data_dir=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


@pytest.fixture()
def durable_dir(tmp_path):
    return tmp_path


def _reopen_client(data_dir):
    """Simulate a backend restart against the same data dir."""
    app = create_app(token=TEST_TOKEN, data_dir=data_dir)
    return TestClient(app, base_url="http://127.0.0.1")


def _csv_bytes(rows: int = 24) -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(rows):
        buf.write(f"{0.1 * i + 0.5:.3f},{(i * 7) % 5 * 0.25},{i % 2}\n")
    return buf.getvalue().encode()


def _upload(client, data: bytes | None = None, name: str = "toy.csv",
            target: str = "y", mode: str = "sync") -> dict:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, data or _csv_bytes(), "text/csv")},
        data={"target": target, "mode": mode},
    )
    assert r.status_code == 200, r.text
    if mode == "sync":
        return r.json()["dataset"]
    job = r.json()["job"]
    final = _wait_job(client, job["job_id"])
    assert final["state"] == "COMPLETED", final
    result = client.get(
        f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
    ).json()["result"]
    return result


def _train(client, dataset_id: str, model: str = "random_forest") -> dict:
    body = {
        "dataset_id": dataset_id,
        "model": model,
        "tune": False,
        "n_jobs": -1,
        "mode": "sync",
    }
    r = client.post("/api/v1/ml/train", headers=auth_headers(), json=body)
    assert r.status_code == 200, r.text
    return r.json()["model"]


def _wait_job(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers())
        if r.status_code != 200:
            break
        state = r.json()["state"]
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            return r.json()
        time.sleep(0.05)
    pytest.fail(f"job {job_id} never finished")


# ------------------------------------------------------------ migrations


class TestMigrations:
    def test_fresh_dir_gets_the_full_ledger(self, tmp_path):
        p = Persistence(tmp_path)
        assert p.schema_version() == SCHEMA_VERSION
        rows = p._execute("SELECT version FROM schema_migrations")
        assert sorted(r["version"] for r in rows) == [
            v for v, _ in MIGRATIONS
        ]

    def test_reopen_is_idempotent(self, tmp_path):
        Persistence(tmp_path)
        second = Persistence(tmp_path)
        assert second.schema_version() == SCHEMA_VERSION
        # exactly one ledger row per migration — no re-application
        rows = second._execute(
            "SELECT version, COUNT(*) AS n FROM schema_migrations"
            " GROUP BY version"
        )
        assert all(r["n"] == 1 for r in rows)

    def test_tables_and_indexes_exist(self, tmp_path):
        p = Persistence(tmp_path)
        names = {
            r["name"]
            for r in p._execute(
                "SELECT name FROM sqlite_master WHERE type IN"
                " ('table', 'index')"
            )
        }
        for table in (
            "datasets",
            "trained_models",
            "generative_priors",
            "jobs",
            "artifacts",
            "user_preferences",
            "schema_migrations",
            "idx_jobs_created",
        ):
            assert table in names, f"missing {table}"

    def test_foreign_keys_enforced(self, tmp_path):
        p = Persistence(tmp_path)
        p._execute_write(
            "INSERT INTO trained_models VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("mdl-x", "n", "regression", None, None, "[]", 0, "{}",
             None, None, None, 0.0, "2026-01-01T00:00:00Z", None),
        )
        # dataset_id references a missing dataset row -> rejected
        with pytest.raises(sqlite3.IntegrityError):
            p._execute_write(
                "INSERT INTO trained_models VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("mdl-y", "n", "regression", "ds-missing", None, "[]",
                 0, "{}", None, None, None, 0.0,
                 "2026-01-01T00:00:00Z", None),
            )


# ------------------------------------------------------------ durability


class TestDurableDatasets:
    def test_upload_persists_row_and_payload(self, durable_client,
                                             durable_dir):
        ds = _upload(durable_client)
        db = durable_dir / "studio.db"
        assert db.exists()
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT dataset_id, name, storage_rel FROM datasets"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == ds["dataset_id"]
        payload = durable_dir / rows[0][2]
        assert payload.exists()
        assert payload.read_bytes() == _csv_bytes()

    def test_restart_recovers_the_dataset(self, durable_client,
                                          durable_dir):
        ds = _upload(durable_client)
        with _reopen_client(durable_dir) as c2:
            listed = c2.get(
                "/api/v1/datasets", headers=auth_headers()
            ).json()["datasets"]
            assert [d["dataset_id"] for d in listed] == [ds["dataset_id"]]
            view = c2.get(
                f"/api/v1/datasets/{ds['dataset_id']}",
                headers=auth_headers(),
            ).json()
            # the REAL loader re-parsed the frame: shape + preview back
            assert view["shape"] == ds["shape"]
            assert view["n_features"] == ds["n_features"]
            assert view["preview"]["rows"][0] == ds["preview"]["rows"][0]

    def test_eviction_deletes_row_and_payload(self, durable_client,
                                              durable_dir):
        ds = _upload(durable_client)
        conn = sqlite3.connect(str(durable_dir / "studio.db"))
        rel = conn.execute(
            "SELECT storage_rel FROM datasets"
        ).fetchone()[0]
        conn.close()
        r = durable_client.delete(
            f"/api/v1/datasets/{ds['dataset_id']}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        assert not (durable_dir / rel).exists()
        conn = sqlite3.connect(str(durable_dir / "studio.db"))
        n = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        conn.close()
        assert n == 0

    def test_recovery_drops_rows_with_missing_payload(self, durable_dir):
        app = create_app(token=TEST_TOKEN, data_dir=durable_dir)
        with TestClient(app, base_url="http://127.0.0.1") as c:
            _upload(c)
        # sabotage: remove the vault payload but keep the row
        conn = sqlite3.connect(str(durable_dir / "studio.db"))
        rel = conn.execute(
            "SELECT storage_rel FROM datasets"
        ).fetchone()[0]
        conn.close()
        (durable_dir / rel).unlink()
        app2 = create_app(token=TEST_TOKEN, data_dir=durable_dir)
        report = app2.state.recovery_report
        assert report["datasets"] == 0
        assert report["dropped"][0]["table"] == "datasets"
        with TestClient(app2, base_url="http://127.0.0.1") as c2:
            listed = c2.get(
                "/api/v1/datasets", headers=auth_headers()
            ).json()["datasets"]
            assert listed == []


class TestDurableModels:
    def test_train_predict_across_restart(self, durable_client,
                                           durable_dir):
        ds = _upload(durable_client)
        model = _train(durable_client, ds["dataset_id"])
        predict_body = {
            "model_id": model["model_id"],
            "dataset_id": ds["dataset_id"],
        }
        first = durable_client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json=predict_body,
        ).json()
        with _reopen_client(durable_dir) as c2:
            # the recovered registry lists the model with its CV facts
            trained = c2.get(
                "/api/v1/ml/trained", headers=auth_headers()
            ).json()["models"]
            assert [m["model_id"] for m in trained] == [
                model["model_id"]
            ]
            assert trained[0]["cv_mean"] == pytest.approx(
                model["cv_mean"], nan_ok=True
            )
            # and the recovered estimator really predicts
            second = c2.post(
                "/api/v1/ml/predict", headers=auth_headers(),
                json=predict_body,
            ).json()
            assert second["n_rows"] == first["n_rows"]
            got = second.get("predictions") or second["result"]
            want = first.get("predictions") or first["result"]
            assert list(got) == pytest.approx(list(want))

    def test_prior_round_trip(self, durable_dir):
        app = create_app(token=TEST_TOKEN, data_dir=durable_dir)
        with TestClient(app, base_url="http://127.0.0.1") as c:
            record = PriorRecord(
                prior_id="pri-roundtrip",
                model={"kind": "fake-weights", "w": [0.1, 0.2]},
                name="fake-prior",
                params={"hidden": 16},
                n_train_smiles=7,
                losses=[2.0, 1.5, 1.2],
                corpus_source="unit",
            )
            c.app.state.prior_store.register(record)
        with _reopen_client(durable_dir) as c2:
            priors = c2.get(
                "/api/v1/generative/priors",
                headers=auth_headers(),
            ).json()
            listed = priors.get("priors") or priors.get("trained") or []
            ids = [p.get("prior_id") for p in listed]
            assert "pri-roundtrip" in ids


# ------------------------------------------------------------ job history


class TestJobHistory:
    def test_history_survives_restart(self, durable_client, durable_dir):
        r = durable_client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={"file": ("toy.csv", _csv_bytes(), "text/csv")},
            data={"target": "y", "mode": "job"},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job"]["job_id"]
        final = _wait_job(durable_client, job_id)
        assert final["state"] == "COMPLETED"
        with _reopen_client(durable_dir) as c2:
            r = c2.get(
                "/api/v1/database/jobs", headers=auth_headers()
            )
            assert r.status_code == 200
            jobs = r.json()["jobs"]
            assert any(j["job_id"] == job_id for j in jobs)
            row = next(j for j in jobs if j["job_id"] == job_id)
            assert row["state"] == "COMPLETED"
            assert row["operation"] == "datasets.load"

    def test_workspace_filter_and_limit(self, durable_client):
        _upload(durable_client, mode="job")
        r = durable_client.get(
            "/api/v1/database/jobs",
            headers=auth_headers(),
            params={"workspace": "datasets", "limit": 5},
        )
        assert r.status_code == 200
        assert all(j["workspace"] == "datasets" for j in r.json()["jobs"])

    def test_prune_keeps_newest(self, tmp_path):
        p = Persistence(tmp_path)
        for i in range(12):
            p.record_job({
                "job_id": f"job-{i:02d}",
                "operation": "op",
                "workspace": "w",
                "state": "COMPLETED",
                "created_at": f"2026-01-01T00:{i:02d}:00Z",
                "started_at": None,
                "completed_at": None,
                "progress": 100,
                "error": None,
                "configuration": {},
            })
        removed = p.prune_jobs(keep_last=5)
        assert removed == 7
        rows = p._execute("SELECT job_id FROM jobs ORDER BY job_id")
        assert [r["job_id"] for r in rows] == [
            f"job-{i:02d}" for i in range(7, 12)
        ]


# ------------------------------------------------------------ preferences


class TestPreferences:
    def test_round_trip_and_listing(self, durable_client):
        durable_client.put(
            "/api/v1/database/preferences/ui.theme",
            headers=auth_headers(),
            json={"value": {"mode": "dark", "density": 2}},
        )
        durable_client.put(
            "/api/v1/database/preferences/last.workspace",
            headers=auth_headers(),
            json={"value": "machine-learning"},
        )
        listed = durable_client.get(
            "/api/v1/database/preferences", headers=auth_headers()
        ).json()["preferences"]
        keys = [p["key"] for p in listed]
        assert keys == ["last.workspace", "ui.theme"]
        got = durable_client.get(
            "/api/v1/database/preferences/ui.theme",
            headers=auth_headers(),
        ).json()["value"]
        assert got == {"mode": "dark", "density": 2}

    def test_unknown_key_404_and_delete(self, durable_client):
        r = durable_client.get(
            "/api/v1/database/preferences/nope",
            headers=auth_headers(),
        )
        assert r.status_code == 404
        durable_client.put(
            "/api/v1/database/preferences/k",
            headers=auth_headers(),
            json={"value": 1},
        )
        r = durable_client.delete(
            "/api/v1/database/preferences/k",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        assert durable_client.get(
            "/api/v1/database/preferences/k",
            headers=auth_headers(),
        ).status_code == 404

    def test_preferences_survive_restart(self, durable_client,
                                         durable_dir):
        durable_client.put(
            "/api/v1/database/preferences/ui.theme",
            headers=auth_headers(),
            json={"value": "light"},
        )
        with _reopen_client(durable_dir) as c2:
            assert c2.get(
                "/api/v1/database/preferences/ui.theme",
                headers=auth_headers(),
            ).json()["value"] == "light"


# ------------------------------------------------------------ backup/restore


class TestBackupRestore:
    def test_bundle_restores_deleted_dataset(self, durable_client,
                                             durable_dir):
        ds = _upload(durable_client)
        art = durable_client.post(
            "/api/v1/database/backup", headers=auth_headers(), json={}
        ).json()["artifact"]
        # the artifact is a real zip container with db + payload
        raw = durable_client.get(
            f"/api/v1/artifacts/{art['artifact_id']}/download",
            headers=auth_headers(),
        ).content
        names = set(zipfile.ZipFile(io.BytesIO(raw)).namelist())
        assert "studio.db" in names
        assert any(n.startswith("files/") for n in names)
        # mutate: delete the dataset (row + payload gone)
        durable_client.delete(
            f"/api/v1/datasets/{ds['dataset_id']}",
            headers=auth_headers(),
        )
        assert durable_client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"] == []
        # restore: the COMPLETE registry comes back
        r = durable_client.post(
            "/api/v1/database/restore",
            headers=auth_headers(),
            json={"artifact_id": art["artifact_id"]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["recovered"]["datasets"] == 1
        assert r.json()["restore"]["payloads_restored"] >= 1
        view = durable_client.get(
            f"/api/v1/datasets/{ds['dataset_id']}",
            headers=auth_headers(),
        ).json()
        assert view["shape"] == ds["shape"]

    def test_restore_rejects_wrong_kind(self, durable_client):
        art = durable_client.post(
            "/api/v1/artifacts/figure",
            headers=auth_headers(),
            json={
                "kind": "scatter",
                "x": [1, 2, 3, 4],
                "y": [1.0, 2.0, 3.5, 4.5],
                "format": "SVG",
            },
        ).json()["artifact"]
        r = durable_client.post(
            "/api/v1/database/restore",
            headers=auth_headers(),
            json={"artifact_id": art["artifact_id"]},
        )
        assert r.status_code == 422

    def test_restore_rejects_corrupt_bundle(self, durable_client):
        from pharmaml.api.stores import ArtifactStore

        store = ArtifactStore(persistence=None)
        record = store.register(
            kind="database",
            filename="corrupt.zip",
            content_type="application/zip",
            data=b"PK\x03\x04 not really a zip",
        )
        durable_client.app.state.artifact_store._items[
            record.artifact_id
        ] = record
        r = durable_client.post(
            "/api/v1/database/restore",
            headers=auth_headers(),
            json={"artifact_id": record.artifact_id},
        )
        assert r.status_code == 422

    def test_restore_rejects_newer_schema(self, durable_client,
                                          durable_dir):
        from pharmaml.api.stores import ArtifactStore

        p: Persistence = durable_client.app.state.persistence
        honest = p.backup_bundle()
        # forge a bundle whose db claims a future schema version
        with zipfile.ZipFile(io.BytesIO(honest)) as zf:
            db_image = zf.read("studio.db")
            rels = [n for n in zf.namelist() if n != "studio.db"]
            payloads = {n: zf.read(n) for n in rels}
        probe_path = durable_dir / "forge.db"
        probe_path.write_bytes(db_image)
        conn = sqlite3.connect(str(probe_path))
        conn.execute(
            "INSERT INTO schema_migrations VALUES (?, ?)",
            (SCHEMA_VERSION + 1, "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        conn.close()
        forged_image = probe_path.read_bytes()
        probe_path.unlink()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("studio.db", forged_image)
            for n, payload in payloads.items():
                zf.writestr(n, payload)
        store: ArtifactStore = durable_client.app.state.artifact_store
        record = store.register(
            kind="database",
            filename="future.zip",
            content_type="application/zip",
            data=buf.getvalue(),
        )
        r = durable_client.post(
            "/api/v1/database/restore",
            headers=auth_headers(),
            json={"artifact_id": record.artifact_id},
        )
        assert r.status_code == 422
        assert "newer" in r.json()["detail"]


# ------------------------------------------------------------ ephemeral


class TestEphemeralHonesty:
    def test_no_data_dir_stays_ephemeral(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHARMAML_STUDIO_DATA", raising=False)
        monkeypatch.chdir(tmp_path)
        app = create_app(token=TEST_TOKEN)
        assert app.state.persistence is None
        with TestClient(app, base_url="http://127.0.0.1") as c:
            r = c.get("/api/v1/database", headers=auth_headers())
            assert r.status_code == 200
            body = r.json()
            assert body["persistent"] is False
            assert "reason" in body
            r = c.get("/api/v1/database/jobs", headers=auth_headers())
            assert r.status_code == 409

    def test_env_var_activates_persistence(self, tmp_path, monkeypatch):
        target = tmp_path / "from-env"
        monkeypatch.setenv("PHARMAML_STUDIO_DATA", str(target))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        app = create_app(token=TEST_TOKEN)
        assert app.state.persistence is not None
        assert app.state.persistence.data_dir == target

    def test_status_report_shape(self, durable_client):
        body = durable_client.get(
            "/api/v1/database", headers=auth_headers()
        ).json()
        assert body["persistent"] is True
        assert body["schema_version"] == SCHEMA_VERSION
        assert body["journal_mode"] == "wal"
        assert body["integrity"] == "ok"
        assert body["tables"]["user_preferences"] == 0


# ------------------------------------------------------------ §16.4 mechanics


class TestSafetyMechanics:
    def test_wal_mode_is_live(self, tmp_path):
        p = Persistence(tmp_path)
        conn = p.connect()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert str(mode).lower() == "wal"

    def test_concurrent_writers_keep_integrity(self, tmp_path):
        p = Persistence(tmp_path)

        def writer(n: int) -> None:
            for i in range(8):
                p.set_preference(f"k-{n}-{i}", i)

        threads = [
            threading.Thread(target=writer, args=(n,))
            for n in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert p.integrity_check() == "ok"
        assert len(p.list_preferences()) == 48

    def test_vault_traversal_rejected(self, tmp_path):
        vault = FileVault(tmp_path)
        with pytest.raises(ValueError):
            vault.read("../escape.bin")
        with pytest.raises(ValueError):
            vault.read("/etc/passwd")
        with pytest.raises(ValueError):
            vault.read(None)
        with pytest.raises(ValueError):
            vault.read("files/../../etc/passwd")

    def test_data_dir_default_shape(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PHARMAML_STUDIO_DATA", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        if os.name == "nt":
            # r297: the design's own platform split (§16.5) -- Windows
            # hosts use %LOCALAPPDATA%\PharmaML Studio and the XDG var
            # stays inert there; the posix branch keeps the XDG pin.
            from pathlib import Path as _Path

            assert default_data_dir() == _Path(
                os.environ["LOCALAPPDATA"]
            ) / "PharmaML Studio"
        else:
            assert default_data_dir() == (
                tmp_path / "xdg" / "PharmaML Studio"
            )

    def test_serve_uses_platform_data_dir(self, monkeypatch, tmp_path):
        from pharmaml.api import server

        monkeypatch.setenv("PHARMAML_STUDIO_DATA", str(tmp_path))
        # the shipped host entry resolves the platform data dir —
        # the durable-by-default contract (§16.5 + Phase Eleven)
        assert server.default_data_dir() == tmp_path


# ------------------------------------------------------------ startup wiring


class TestStartupWiring:
    def test_recovery_report_on_state(self, durable_client):
        # a fresh durable app recovered an empty registry — reported
        report = durable_client.app.state.recovery_report
        assert report is not None
        assert report["datasets"] == 0
        assert report["dropped"] == []

    def test_startup_recovery_reported_via_status(self, durable_client,
                                                  durable_dir):
        _upload(durable_client)
        with _reopen_client(durable_dir) as c2:
            body = c2.get(
                "/api/v1/database", headers=auth_headers()
            ).json()
            assert body["startup_recovery"]["datasets"] == 1

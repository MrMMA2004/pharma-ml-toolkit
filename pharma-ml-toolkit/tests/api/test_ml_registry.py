"""The part16 §15.3 (Phase 12) model-registry lane — the API proof.

The directive's §15.3 evaluation, made mechanical over the EXISTING
registry surface (the store and the export lane are not new; the
deletion lane is this round's addition):

* **the catalog** — the registry's own projection over every entry
  with the availability stated per row (the five catalog keys, the
  task filter's dual-task semantics, the ``only_available`` filter)
  (:class:`TestCatalogSurface`);
* **the deletion lane** — ``DELETE /api/v1/ml/trained/{model_id}``
  evicts the session row (an honest 404 on an unknown id, never a
  silent no-op), the answer carries the remaining count, and the
  eviction's scope rails hold: the §15.2 run lineage outlives the
  model (the record keeps its model-artifact reference verbatim --
  the fetch pointer degrades to the honest 404), and the exported
  joblib artifact is an independent copy that survives the row
  (:class:`TestEvictionLane`);
* **durability** — the eviction is persisted (the trained_models row
  and its vault payload are gone: the model does not resurrect on a
  restart) (:class:`TestDurability`);
* **the listing contract** — the trained registry rides the store's
  own oldest-first order (:class:`TestRegistryOrder`).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _load_demo(client, sample_id: str = "walkthrough-classification") -> dict:
    r = client.post(
        f"/api/v1/samples/{sample_id}/load", headers=auth_headers()
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _train_sync(client, dataset_id, **overrides) -> dict:
    payload = {
        "dataset_id": dataset_id,
        "model": "logistic_regression",
        "task": "classification",
        "tune": False,
        "cv": 2,
        "n_jobs": 1,
        "mode": "sync",
    }
    payload.update(overrides)
    r = client.post(
        "/api/v1/ml/train", headers=auth_headers(), json=payload
    )
    assert r.status_code == 200, r.text
    return r.json()


def _trained(client) -> list[dict]:
    r = client.get("/api/v1/ml/trained", headers=auth_headers())
    assert r.status_code == 200, r.text
    return r.json()["models"]


def _evict(client, model_id):
    return client.delete(
        f"/api/v1/ml/trained/{model_id}", headers=auth_headers()
    )


class TestCatalogSurface:
    """The registry catalog — the §15.3 evaluation view over every entry."""

    def test_the_full_catalog_lists_every_entry_with_the_five_keys(self, client):
        r = client.get(
            "/api/v1/ml/models",
            headers=auth_headers(),
            params={"only_available": "false"},
        )
        assert r.status_code == 200, r.text
        entries = r.json()["models"]
        assert entries, "the catalog must not be empty"
        for entry in entries:
            assert set(entry) == {
                "name",
                "task",
                "available",
                "requires",
                "notes",
            }, entry
            assert entry["task"] in {"classification", "regression", "both"}

    def test_the_task_filter_keeps_the_entry_and_the_dual_task_entries(self, client):
        for task, allowed in (
            ("classification", {"classification", "both"}),
            ("regression", {"regression", "both"}),
            ("both", {"both"}),
        ):
            r = client.get(
                "/api/v1/ml/models",
                headers=auth_headers(),
                params={"task": task, "only_available": "false"},
            )
            assert r.status_code == 200, r.text
            tasks = {entry["task"] for entry in r.json()["models"]}
            assert tasks <= allowed, (task, tasks)

    def test_only_available_filters_the_unavailable_entries(self, client):
        r_all = client.get(
            "/api/v1/ml/models",
            headers=auth_headers(),
            params={"only_available": "false"},
        )
        r_available = client.get(
            "/api/v1/ml/models", headers=auth_headers()
        )
        assert r_all.status_code == 200 and r_available.status_code == 200
        every = r_all.json()["models"]
        available = r_available.json()["models"]
        # the available listing is the available subset, never more
        assert len(available) <= len(every)
        assert all(entry["available"] for entry in available)
        assert {entry["name"] for entry in available} == {
            entry["name"] for entry in every if entry["available"]
        }


class TestEvictionLane:
    """`DELETE /api/v1/ml/trained/{model_id}` — the registry's deletion."""

    def test_evicting_a_trained_model_removes_it_from_the_registry(self, client):
        dataset = _load_demo(client)
        trained = _train_sync(client, dataset["dataset_id"])
        model_id = trained["model"]["model_id"]
        assert _trained(client), "the training registered the model"

        r = _evict(client, model_id)
        assert r.status_code == 200, r.text
        assert r.json()["evicted"] == model_id
        assert isinstance(r.json()["remaining"], int)

        ids = {row["model_id"] for row in _trained(client)}
        assert model_id not in ids, "the evicted model must leave the registry"
        view = client.get(
            f"/api/v1/ml/trained/{model_id}", headers=auth_headers()
        )
        assert view.status_code == 404, "the evicted view must degrade to 404"

    def test_evicting_an_unknown_model_id_is_an_honest_404(self, client):
        r = _evict(client, "mdl-doesnotexist")
        assert r.status_code == 404
        assert r.json()["detail"] == "Unknown model_id."

    def test_the_run_lineage_outlives_the_evicted_model(self, client):
        dataset = _load_demo(client)
        trained = _train_sync(client, dataset["dataset_id"])
        model_id = trained["model"]["model_id"]

        runs = client.get("/api/v1/runs", headers=auth_headers())
        assert runs.status_code == 200
        record = runs.json()["runs"][0]

        assert _evict(client, model_id).status_code == 200

        # the §15.2 record keeps its model-artifact reference verbatim
        # (never rewritten) — the fetch pointer degrades to the honest 404
        full = client.get(
            f"/api/v1/runs/{record['run_id']}", headers=auth_headers()
        )
        assert full.status_code == 200
        payload = full.json()
        assert payload["model_artifact"]["model_id"] == model_id
        gone = client.get(
            f"/api/v1/ml/trained/{model_id}", headers=auth_headers()
        )
        assert gone.status_code == 404

    def test_the_exported_artifact_outlives_the_evicted_model(self, client):
        dataset = _load_demo(client)
        trained = _train_sync(client, dataset["dataset_id"])
        model_id = trained["model"]["model_id"]

        export = client.post(
            "/api/v1/artifacts/model",
            headers=auth_headers(),
            json={"model_id": model_id},
        )
        assert export.status_code == 200, export.text
        artifact = export.json()["artifact"]
        assert artifact["kind"] == "model"
        assert artifact["filename"].endswith(".joblib")

        assert _evict(client, model_id).status_code == 200

        # the export is an independent copy — it survives the row
        view = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}",
            headers=auth_headers(),
        )
        assert view.status_code == 200, "the exported artifact survived the eviction"
        download = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert download.status_code == 200
        assert download.headers.get("x-artifact-sha256") == artifact["sha256"]


class TestDurability:
    """The eviction is persisted — the model does not resurrect."""

    def test_the_eviction_survives_the_restart(self, tmp_path):
        app = create_app(token=TEST_TOKEN, data_dir=tmp_path)
        with TestClient(app, base_url="http://127.0.0.1") as first:
            dataset = _load_demo(first)
            trained = _train_sync(first, dataset["dataset_id"])
            model_id = trained["model"]["model_id"]
            assert _evict(first, model_id).status_code == 200
        app2 = create_app(token=TEST_TOKEN, data_dir=tmp_path)
        with TestClient(app2, base_url="http://127.0.0.1") as second:
            ids = {row["model_id"] for row in _trained(second)}
            assert model_id not in ids, "the evicted model resurrected"
            view = second.get(
                f"/api/v1/ml/trained/{model_id}", headers=auth_headers()
            )
            assert view.status_code == 404


class TestRegistryOrder:
    """The listing rides the store's own oldest-first contract."""

    def test_the_trained_listing_is_oldest_first(self, client):
        dataset = _load_demo(client)
        first = _train_sync(client, dataset["dataset_id"])
        second = _train_sync(client, dataset["dataset_id"])
        rows = _trained(client)
        stamps = [row["created_at"] for row in rows]
        assert stamps == sorted(stamps), "the registry must be oldest-first"
        ids = [row["model_id"] for row in rows]
        assert ids.index(first["model"]["model_id"]) < ids.index(
            second["model"]["model_id"]
        )

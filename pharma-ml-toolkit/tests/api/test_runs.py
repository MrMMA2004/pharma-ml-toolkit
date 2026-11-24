"""The part16 §15.2 (Phase 12) experiment-tracking lane — the API proof.

The directive's §15.2 checklist, made mechanical.  Every run the
training lane produces must be traceable back to the exact input and
configuration:

* **the completed run** — one first-class record joining: dataset
  identifiers, feature configuration, preprocessing pipeline,
  algorithm + toolkit version, hyperparameters (requested AND tuned),
  random seed, cross-validation configuration, evaluation metrics,
  environment information, model artifact reference, execution
  timestamps, and the job reference (:class:`TestCompletedRun`);
* **the failed run** — a doomed configuration is recorded with the
  typed error, the fit-only fields honestly absent
  (:class:`TestFailedRun`);
* **traceability** — the record's dataset block == the dataset's own
  view identity, the model artifact == the trained store's view, the
  job reference == the JobManager history row
  (:class:`TestLineageJoins`);
* **the query surface** — the lineage table (filters, newest first,
  validated status, bounded limit) and the 404
  (:class:`TestQuerySurface`);
* **durable mode** — the full §15.2 record survives a restart
  (MIGRATIONS v3, JSON-only rehydration — no pickles)
  (:class:`TestDurability`);
* **job mode** — the async lane records with its job id at both exits
  (:class:`TestJobMode`).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from pharmaml import __version__ as TOOLKIT_VERSION
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


def _train_sync(client, dataset_id, **overrides):
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


def _runs(client, **params):
    r = client.get("/api/v1/runs", headers=auth_headers(), params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _run(client, run_id):
    return client.get(f"/api/v1/runs/{run_id}", headers=auth_headers())


class TestCompletedRun:
    """The §15.2 checklist — one record, every field, honestly sourced."""

    def test_training_records_a_completed_run(self, client):
        dataset = _load_demo(client)
        # the tuned lane: the evaluation block carries real CV metrics
        # (tune=False fits once and reports None — the honest absent)
        _train_sync(client, dataset["dataset_id"], tune=True)
        listing = _runs(client)
        assert listing["count"] >= 1, "no run recorded for a sync train"
        summary = listing["runs"][0]
        assert summary["family"] == "ml.train"
        assert summary["status"] == "completed"
        assert summary["mode"] == "sync"
        assert summary["model_name"] == "logistic_regression"
        assert summary["dataset_id"] == dataset["dataset_id"]
        assert summary["metric"] is not None

    def test_the_full_record_carries_the_1512_checklist(self, client):
        dataset = _load_demo(client)
        model = _train_sync(client, dataset["dataset_id"])["model"]
        record = _run(client, _runs(client)["runs"][0]["run_id"]).json()
        # dataset identifiers and versions
        assert record["dataset"]["dataset_id"] == dataset["dataset_id"]
        assert record["dataset"]["name"] == dataset["name"]
        assert record["dataset"]["source_filename"] == (
            dataset["source_filename"]
        )
        assert record["dataset"]["shape"] == dataset["shape"]
        assert record["dataset"]["target"] == dataset["target"]
        # feature configuration
        assert record["feature_configuration"]["feature_names"] == (
            dataset["feature_names"]
        )
        assert record["feature_configuration"]["n_features"] == (
            dataset["n_features"]
        )
        # preprocessing pipeline
        assert record["preprocessing"]["enabled"] is True
        assert "imputation" in record["preprocessing"]["pipeline"]
        # algorithm and version
        assert record["algorithm"]["name"] == "logistic_regression"
        assert record["algorithm"]["toolkit_version"] == TOOLKIT_VERSION
        # hyperparameters (requested + tuned + best)
        assert record["hyperparameters"]["tuned"] is False
        assert record["hyperparameters"]["best_params"] == (
            model["best_params"]
        )
        # random seed
        assert record["random_seed"] == 42
        # cross-validation configuration
        cv = record["cross_validation"]
        assert cv["cv_folds"] == 2
        assert cv["n_jobs_requested"] == 1
        assert cv["n_jobs_effective"] >= 1
        assert cv["parallelism_advisory"] is not None
        # evaluation metrics
        assert record["evaluation"]["cv_mean"] == model["cv_mean"]
        assert record["evaluation"]["cv_scores"] == model["cv_scores"]
        assert record["evaluation"]["primary_metric"] is not None
        # environment information
        env = record["environment"]
        assert env["toolkit"] == TOOLKIT_VERSION
        assert env["python"]
        assert "numpy" in env["packages"]
        # model artifact reference
        assert record["model_artifact"]["model_id"] == model["model_id"]
        assert record["model_artifact"]["fetch"] == (
            f"GET /api/v1/ml/trained/{model['model_id']}"
        )
        # execution timestamps
        assert record["started_at"] <= record["completed_at"]
        # the schema pin
        assert record["schema"] == "pharmaml-experiment-run/1"

    def test_requested_hyperparameters_and_seed_are_recorded(self, client):
        dataset = _load_demo(client)
        _train_sync(
            client,
            dataset["dataset_id"],
            random_state=7,
            model_params={"C": 0.5},
        )
        record = _run(client, _runs(client)["runs"][0]["run_id"]).json()
        assert record["random_seed"] == 7
        assert record["hyperparameters"]["requested"] == {"C": 0.5}


class TestFailedRun:
    """A doomed configuration is scientific information — record it."""

    def test_a_failed_fit_records_the_run_with_the_error(self, client):
        dataset = _load_demo(client)
        # sync mode: the raw estimator error propagates to the caller
        # (the route maps only typed client-input errors to 422); the
        # run record is the durable trace of what the caller saw.
        with pytest.raises(ValueError):
            client.post(
                "/api/v1/ml/train",
                headers=auth_headers(),
                json={
                    "dataset_id": dataset["dataset_id"],
                    "model": "random_forest",
                    "task": "classification",
                    "tune": False,
                    "cv": 2,
                    "n_jobs": 1,
                    "mode": "sync",
                    # sklearn refuses n_estimators <= 0 mid-fit
                    "model_params": {"n_estimators": 0},
                },
            )
        failed = _runs(client, status="failed")["runs"]
        assert failed, "the failed fit recorded no run"
        record = _run(client, failed[0]["run_id"]).json()
        assert record["status"] == "failed"
        assert record["error"]
        assert record["algorithm"]["name"] == "random_forest"
        assert record["model_artifact"] is None  # never invented
        assert record["evaluation"]["cv_mean"] is None
        assert record["hyperparameters"]["requested"] == {
            "n_estimators": 0
        }

    def test_the_pre_submit_client_errors_record_nothing(self, client):
        # an unknown model is refused BEFORE the run (a client error,
        # not a scientific computation)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={"dataset_id": "ds-none", "model": "not_a_model",
                  "mode": "sync"},
        )
        assert r.status_code == 404, r.text
        assert _runs(client)["count"] == 0


class TestLineageJoins:
    """Traceability: the record's blocks == the stores' own views."""

    def test_dataset_block_matches_the_dataset_view(self, client):
        dataset = _load_demo(client)
        _train_sync(client, dataset["dataset_id"])
        record = _run(client, _runs(client)["runs"][0]["run_id"]).json()
        r = client.get(
            f"/api/v1/datasets/{dataset['dataset_id']}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        live = r.json()
        assert record["dataset"]["dataset_id"] == live["dataset_id"]
        assert record["dataset"]["shape"] == live["shape"]
        assert record["dataset"]["target"] == live["target"]

    def test_model_artifact_reference_resolves(self, client):
        dataset = _load_demo(client)
        model = _train_sync(client, dataset["dataset_id"])["model"]
        record = _run(client, _runs(client)["runs"][0]["run_id"]).json()
        r = client.get(
            f"/api/v1/ml/trained/{record['model_artifact']['model_id']}",
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        assert r.json()["model_id"] == model["model_id"]


class TestQuerySurface:
    """The lineage table: filters, ordering, validation, 404."""

    def test_filters_by_dataset_model_and_status(self, client):
        first = _load_demo(client)  # walkthrough-classification
        _train_sync(client, first["dataset_id"])
        second = _load_demo(client, "qsar-regression")
        _train_sync(
            client, second["dataset_id"], model="ridge", task="regression"
        )
        by_dataset = _runs(client, dataset_id=second["dataset_id"])
        assert by_dataset["count"] == 1
        assert by_dataset["runs"][0]["model_name"] == "ridge"
        by_model = _runs(client, model="ridge")
        assert by_model["count"] == 1
        by_status = _runs(client, status="completed")
        assert by_status["count"] == 2
        by_family = _runs(client, family="ml.train")
        assert by_family["count"] == 2

    def test_the_listing_is_newest_first_and_bounded(self, client):
        dataset = _load_demo(client)
        _train_sync(client, dataset["dataset_id"])
        _train_sync(client, dataset["dataset_id"])
        listing = _runs(client)
        stamps = [r["created_at"] for r in listing["runs"]]
        assert stamps == sorted(stamps, reverse=True)
        bounded = _runs(client, limit=1)
        assert bounded["count"] == 1 and len(bounded["runs"]) == 1

    def test_the_status_filter_is_validated(self, client):
        r = client.get(
            "/api/v1/runs", headers=auth_headers(), params={"status": "bogus"}
        )
        assert r.status_code == 422, r.text

    def test_unknown_run_is_a_404(self, client):
        r = _run(client, "run-none")
        assert r.status_code == 404, r.text


class TestJobMode:
    """The async lane records with its job id at both exits."""

    def test_job_mode_records_the_job_reference(self, client):
        dataset = _load_demo(client)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "model": "logistic_regression",
                "task": "classification",
                "tune": False,
                "cv": 2,
                "n_jobs": 1,
                "mode": "job",
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job"]["job_id"]
        for _ in range(120):
            jr = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            )
            if jr.status_code == 200 and jr.json()["state"] == "COMPLETED":
                break
            time.sleep(0.25)
        else:
            pytest.fail("the training job never completed")
        record = _run(client, _runs(client)["runs"][0]["run_id"]).json()
        assert record["job_id"] == job_id
        assert record["logs"]["history"] == f"GET /api/v1/jobs/{job_id}"
        assert record["mode"] == "job"

    def test_job_mode_failure_records_the_failed_run(self, client):
        dataset = _load_demo(client)
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "model": "random_forest",
                "task": "classification",
                "tune": False,
                "cv": 2,
                "n_jobs": 1,
                "mode": "job",
                "model_params": {"n_estimators": 0},
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job"]["job_id"]
        for _ in range(120):
            jr = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            )
            if jr.status_code == 200 and jr.json()["state"] == "FAILED":
                break
            time.sleep(0.25)
        else:
            pytest.fail("the doomed job never failed")
        failed = _runs(client, status="failed")["runs"]
        assert failed, "the failed job mode run recorded no run"
        assert failed[0]["job_id"] == job_id
        assert failed[0]["error"]


class TestDurability:
    """The §15.2 record survives a restart (MIGRATIONS v3)."""

    def test_the_full_record_rehydrates_after_restart(self, tmp_path):
        app = create_app(token=TEST_TOKEN, data_dir=tmp_path)
        with TestClient(app, base_url="http://127.0.0.1") as first:
            dataset = _load_demo(first)
            _train_sync(first, dataset["dataset_id"])
            run_id = _runs(first)["runs"][0]["run_id"]
        app2 = create_app(token=TEST_TOKEN, data_dir=tmp_path)
        with TestClient(app2, base_url="http://127.0.0.1") as second:
            listing = _runs(second)
            assert listing["count"] == 1
            record = _run(second, run_id).json()
            assert record["dataset"]["dataset_id"] == dataset["dataset_id"]
            assert record["algorithm"]["toolkit_version"] == TOOLKIT_VERSION
            assert record["model_artifact"]["model_id"]
            assert record["schema"] == "pharmaml-experiment-run/1"

    def test_failed_runs_survive_the_restart_too(self, tmp_path):
        app = create_app(token=TEST_TOKEN, data_dir=tmp_path)
        with TestClient(app, base_url="http://127.0.0.1") as first:
            dataset = _load_demo(first)
            with pytest.raises(ValueError):
                first.post(
                    "/api/v1/ml/train",
                    headers=auth_headers(),
                    json={
                        "dataset_id": dataset["dataset_id"],
                        "model": "random_forest",
                        "task": "classification",
                        "tune": False,
                        "cv": 2,
                        "n_jobs": 1,
                        "mode": "sync",
                        "model_params": {"n_estimators": 0},
                    },
                )
        app2 = create_app(token=TEST_TOKEN, data_dir=tmp_path)
        with TestClient(app2, base_url="http://127.0.0.1") as second:
            failed = _runs(second, status="failed")["runs"]
            assert failed
            assert failed[0]["status"] == "failed"
            assert failed[0]["error"]

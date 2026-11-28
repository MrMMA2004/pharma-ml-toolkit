"""DL-1 (the roadmap's row 13) — ``POST /api/v1/dl/train`` with ``tune``.

The search lane over the SAME route, through the REAL in-house
TPE/Random engine (``pharmaml.ml.hpo.HPOStudy`` — the same engine the
classical ML lane's ``tune`` flag rides): a bounded per-key search
space over the network's own constructor vocabulary; each candidate
is built and fit on a seeded train side and scored on the held-out
val side; the winner refits on the full data and registers like any
trained model.

The roadmap's own acceptance drives the file:

* tune=true returns a trial table whose best config is among the
  candidates (the argmax row of the table is the reported best) and
  the refit metric >= the fixed-config baseline on a seeded fixture
  (the baseline = the SAME lane pinned to the fixed value, one
  scale, never a second evaluation authority);
* the categorical lane rides list-valued keys (``hidden_layers``
  choices);
* a doomed search is a 422 up front (no search block, an unknown
  key, inverted bounds, a non-list choices, the 12-row carve
  minimum) — never an asynchronous job failure;
* cancellation between trials lands CANCELLED (the population-route
  precedent).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
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


def _upload(client, csv: bytes, name: str, target: str = "y") -> dict:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, csv, "text/csv")},
        data={"target": target, "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _fixture_csv(n: int = 60) -> bytes:
    """A seeded, learnable binary classification fixture."""
    rng = np.random.default_rng(41)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] + 0.75 * X[:, 2] > 0).astype(int)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    frame["y"] = y
    return frame.to_csv(index=False).encode("utf-8")


BASE = {"epochs": 15, "batch_size": 16}


def _tune(client, dataset_id: str, **overrides) -> dict:
    body: dict = {
        "dataset_id": dataset_id,
        "network": "deep_mlp_classifier",
        "hyperparams": BASE,
        "tune": True,
        "search": {"learning_rate": {"low": 0.05, "high": 0.5}},
        "n_trials": 6,
        "sampler": "random",
        "seed": 0,
        "mode": "sync",
    }
    body.update(overrides)
    r = client.post("/api/v1/dl/train", headers=auth_headers(), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_search_returns_a_trial_table_with_the_best_among_the_candidates(client):
    dataset = _upload(client, _fixture_csv(), "tune-fixture.csv")
    payload = _tune(client, dataset["dataset_id"])

    model = payload["model"]
    assert model["engine"] == "torch"
    assert model["name"] == "deep_mlp_classifier"
    # the refit's best_params carries the tuned key over the base
    assert "learning_rate" in model["best_params"]
    assert model["best_params"]["epochs"] == 15

    tune = payload["tune"]
    assert tune["n_trials"] == 6
    trials = tune["trials"]
    assert len(trials) == 6
    # every trial carries the searched key and a value (negated
    # smaller-is-better, the optuna rule the engine documents)
    for trial in trials:
        assert "learning_rate" in trial["params"]
        assert isinstance(trial["value"], float)
    # the best config is among the candidates: the argmax trial row
    best_row = max(trials, key=lambda t: -t["value"])
    assert tune["best_params"] == best_row["params"]
    assert tune["best_score"] == pytest.approx(-best_row["value"])
    assert tune["best_score"] > 0.0
    # the refit's comparable score on the SAME carve rides the report
    assert isinstance(tune["refit_score"], float)
    # the searched block echoes the request's own bounds
    assert tune["searched"]["learning_rate"] == {"low": 0.05, "high": 0.5}


def test_the_refit_metric_meets_the_fixed_config_baseline(client):
    """the roadmap's acceptance clause, on one seeded fixture."""
    dataset = _upload(client, _fixture_csv(), "tune-baseline.csv")

    tuned = _tune(client, dataset["dataset_id"])

    # the fixed-config baseline: the SAME lane pinned to a near-zero
    # learning rate (a config that barely learns) — one scale, one
    # carve, never a second evaluation authority
    baseline = _tune(
        client,
        dataset["dataset_id"],
        search={"learning_rate": {"choices": [1e-5]}},
        n_trials=1,
    )
    assert baseline["tune"]["n_trials"] == 1

    assert tuned["tune"]["refit_score"] >= baseline["tune"]["refit_score"], (
        "a tuned search over [0.05, 0.5] must meet or beat a pinned "
        f"1e-5 learning rate on the same carve (got "
        f"{tuned['tune']['refit_score']} vs {baseline['tune']['refit_score']})"
    )


def test_the_categorical_lane_rides_list_valued_keys(client):
    dataset = _upload(client, _fixture_csv(), "tune-hidden.csv")
    payload = _tune(
        client,
        dataset["dataset_id"],
        search={
            "hidden_layers": {"choices": [[32], [64, 32], [128, 64, 32]]},
            "learning_rate": {"low": 0.05, "high": 0.5, "log": True},
        },
        n_trials=4,
    )
    tune = payload["tune"]
    assert len(tune["trials"]) == 4
    shapes = [tuple(t["params"]["hidden_layers"]) for t in tune["trials"]]
    assert set(shapes) <= {(32,), (64, 32), (128, 64, 32)}
    # the categorical values ride the refit's constructor kwargs verbatim
    assert isinstance(tune["best_params"]["hidden_layers"], list)


def test_a_doomed_search_is_a_422_up_front(client):
    dataset = _upload(client, _fixture_csv(), "tune-bad.csv")

    # no search block at all
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "network": "deep_mlp_classifier",
            "tune": True,
            "mode": "sync",
        },
    )
    assert r.status_code == 422
    assert "needs a search block" in r.json()["detail"]

    # an unknown key (not in the network's constructor vocabulary)
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "network": "deep_mlp_classifier",
            "tune": True,
            "search": {"nonsense_key": {"low": 1, "high": 5}},
            "mode": "sync",
        },
    )
    assert r.status_code == 422
    assert "unknown search key" in r.json()["detail"]

    # inverted bounds
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "network": "deep_mlp_classifier",
            "tune": True,
            "search": {"learning_rate": {"low": 0.5, "high": 0.05}},
            "mode": "sync",
        },
    )
    assert r.status_code == 422
    assert "low < high" in r.json()["detail"]

    # a non-list choices block
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "network": "deep_mlp_classifier",
            "tune": True,
            "search": {"activation": {"choices": "relu"}},
            "mode": "sync",
        },
    )
    assert r.status_code == 422
    assert "choices" in r.json()["detail"]

    # the objective carve's own row minimum (the route's own bound,
    # the same 422-before-submit honesty the fixed lane keeps)
    tiny = _upload(client, _fixture_csv(6), "tiny.csv")
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": tiny["dataset_id"],
            "network": "deep_mlp_classifier",
            "hyperparams": BASE,
            "tune": True,
            "search": {"learning_rate": {"low": 0.05, "high": 0.5}},
            "mode": "sync",
        },
    )
    assert r.status_code == 422
    assert "minimum is 12" in r.json()["detail"]


def test_job_mode_delivers_the_same_report(client):
    dataset = _upload(client, _fixture_csv(), "tune-job.csv")
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "network": "deep_mlp_classifier",
            "hyperparams": BASE,
            "tune": True,
            "search": {"learning_rate": {"low": 0.05, "high": 0.5}},
            "n_trials": 3,
            "sampler": "random",
            "seed": 0,
            "mode": "job",
        },
    )
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    job_id = job["job_id"]
    state = None
    for _ in range(240):
        state = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()["state"]
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.25)
    assert state == "COMPLETED"
    result = client.get(
        f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
    ).json()["result"]
    assert result["model"]["engine"] == "torch"
    assert result["tune"]["n_trials"] == 3
    assert len(result["tune"]["trials"]) == 3


def test_cancellation_between_trials_lands_cancelled(client):
    dataset = _upload(client, _fixture_csv(), "tune-cancel.csv")
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "network": "deep_mlp_classifier",
            "hyperparams": {"epochs": 40, "batch_size": 16},
            "tune": True,
            "search": {"learning_rate": {"low": 0.05, "high": 0.5}},
            "n_trials": 40,
            "sampler": "random",
            "seed": 0,
            "mode": "job",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job"]["job_id"]
    cancel = client.post(
        f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers()
    )
    assert cancel.status_code in (200, 202), cancel.text
    state = None
    for _ in range(240):
        state = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()["state"]
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.25)
    assert state == "CANCELLED", (
        "cancellation between trials must land CANCELLED (the "
        f"population-route precedent); got {state}"
    )

"""GEN-3 (the roadmap's row 17) — ``POST /api/v1/generative/rl-optimize``.

The REINVENT-style molecular RL optimizer over the REAL engine
(``pharmaml.generative.rl.run_rl_optimization`` — the policy-gradient
fine-tuning of a SMILES RNN prior against the scoring objective,
with the prior-drift anchor and the experience-replay buffer):
a managed job with per-step telemetry (the RNN train lane's event
pattern) and cancellation BETWEEN iterations (the population-route
precedent), or sync.

The roadmap's own acceptance drives the file:

* the score trajectory improves over the seeded steps on a QED-like
  fixture (the second half's mean beats the first half's on the
  seeded run — the engine's own introspection history);
* cancellation mid-run lands CANCELLED (between iterations);
* the honest refusals: an unknown prior (404), a feature-space prior
  (the molecular lane's own pointer), prior_id + corpus both, a
  too-small corpus, an unknown objective kind (the shared builder's
  own 422);
* a session prior rides the lane with its own view in the answer.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"

#: the seeded QED-like fixture: 10 iterations, batch 24, prior_epochs 6,
#: seed 3 — deterministic under the engine's own seeding; the second
#: half's mean beats the first half's on this exact carve
RL_BODY = {
    "objective": {"kind": "qed"},
    "iterations": 10,
    "batch_size": 24,
    "prior_epochs": 6,
    "seed": 3,
    "patience": 50,  # no early stop — the honest full trajectory
    "mode": "sync",
}


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _rl(client, **overrides) -> object:
    body = dict(RL_BODY)
    body.update(overrides)
    return client.post(
        "/api/v1/generative/rl-optimize", headers=auth_headers(), json=body
    )


def test_the_score_trajectory_improves_over_the_seeded_steps(client):
    r = _rl(client)
    assert r.status_code == 200, r.text
    body = r.json()
    optimization = body["optimization"]

    history = optimization["history"]
    assert len(history) == 10
    means = [row["mean_score"] for row in history]
    # the roadmap's own acceptance: the trajectory improves over the
    # seeded steps — the second half's mean beats the first half's
    first_half = sum(means[:5]) / 5
    second_half = sum(means[5:]) / 5
    assert second_half > first_half, (first_half, second_half)
    # every history row carries the engine's own introspection: the
    # drift diagnostics (the mode-collapse brake's story)
    for row in history:
        for key in (
            "iteration", "best_score", "mean_score", "valid_fraction",
            "drift", "entropy", "loss",
        ):
            assert key in row, f"the history lost its {key}"
    # the leaderboard is ranked best-first and carries the best row
    top = optimization["top"]
    assert len(top) > 0
    assert [row["score"] for row in top] == sorted(
        [row["score"] for row in top], reverse=True
    )
    assert optimization["best_score"] == pytest.approx(top[0]["score"])
    assert optimization["best_score"] > 0.6  # the seeded QED carve
    # the metadata rides the engine's own verdicts
    assert optimization["metadata"]["method"] == "reinforce_rl"
    assert optimization["metadata"]["seed"] == 3
    assert optimization["metadata"]["iterations"] == 10
    assert "REINVENT-style" in body["note"]


def test_a_session_prior_rides_the_lane_with_its_own_view(client):
    # a small molecular prior over the prior-train lane
    r = client.post(
        "/api/v1/generative/rnn/train",
        headers=auth_headers(),
        json={
            "smiles": [
                "CCO", "CCN", "CCC", "CCCO", "CCCN",
                "c1ccccc1", "Cc1ccccc1", "CC(=O)O", "CCN(CC)CC", "CCCC",
            ],
            "epochs": 4,
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    prior = r.json()["prior"]

    r = _rl(client, prior_id=prior["prior_id"], iterations=5, batch_size=16)
    assert r.status_code == 200, r.text
    body = r.json()
    # the session prior's own view rides the answer
    assert body["prior"]["prior_id"] == prior["prior_id"]
    assert body["prior"]["n_train_smiles"] == 10
    optimization = body["optimization"]
    assert len(optimization["history"]) == 5
    assert optimization["metadata"]["method"] == "reinforce_rl"


def test_the_honest_refusal_lanes(client):
    # an unknown prior is a 404
    r = _rl(client, prior_id="prior-does-not-exist")
    assert r.status_code == 404

    # prior_id AND an inline corpus is a 422 (never silently ignored)
    r = _rl(
        client, prior_id="prior-x", smiles=["CCO", "CCN", "CCC", "CCCC", "CCCO"],
    )
    assert r.status_code == 422
    assert "not both" in r.json()["detail"]

    # a too-small corpus is a 422 (the prior-train lane's own minimum)
    r = _rl(client, smiles=["CCO", "CCN"])
    assert r.status_code == 422
    assert "at least 5" in r.json()["detail"]

    # an unknown objective kind is a 422 up front (the shared
    # builder's own vocabulary — never an async job failure)
    r = _rl(client, objective={"kind": "not-a-real-objective"})
    assert r.status_code == 422
    assert "objective" in r.json()["detail"]

    # a feature-space prior is refused with the molecular pointer:
    # train a feature prior over the GAN lane first (a 2-feature
    # dataset uploaded through the standard datasets route)
    import io

    import pandas as pd

    feature_csv = pd.DataFrame(
        {"x1": [0.1 * i for i in range(40)],
         "x2": [0.05 * i for i in range(40)],
         "y": [i % 2 for i in range(40)]}
    ).to_csv(index=False).encode("utf-8")
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": ("gan-feat.csv", feature_csv, "text/csv")},
        data={"target": "y", "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    feature_dataset_id = r.json()["dataset"]["dataset_id"]
    r = client.post(
        "/api/v1/generative/gan/train",
        headers=auth_headers(),
        json={
            "dataset_id": feature_dataset_id,
            "epochs": 3,
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    feature_prior_id = r.json()["prior"]["prior_id"]
    r = _rl(client, prior_id=feature_prior_id)
    assert r.status_code == 422
    assert "MOLECULAR prior" in r.json()["detail"]


def test_cancellation_lands_between_iterations(client):
    r = _rl(client, mode="job", iterations=400, batch_size=32)
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
        "cancellation between RL iterations must land CANCELLED (the "
        f"population-route precedent); got {state}"
    )

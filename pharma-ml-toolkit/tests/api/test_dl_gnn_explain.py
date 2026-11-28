"""DL-2 (the roadmap's row 14) — ``POST /api/v1/dl/gnn-explain``.

The graph models' edge-attribution lane over the REAL
``pharmaml.ml.gnnxai`` engine: the chosen explainer's per-edge
importance for inline graph rows against a session-trained GNN, each
row carrying the fidelity_plus / fidelity_minus / sparsity verdicts
and the deterministic circular layout.

The roadmap's own acceptance drives the file:

* fidelity_plus/minus computed on a seeded two-graph fixture, the
  explained subgraph <= the full graph size (top_edges is a subset of
  the edge columns), the scores aligned with the graph's own
  edge_index;
* the honest 422 lanes: an unknown model (404), a non-graph model,
  labels with a non-pgexplainer method, pgexplainer without labels,
  a misaligned labels length, the iters knob on a sampling explainer,
  the attention explainer's own GAT requirement (the engine's
  message);
* the pgexplainer lane trains its scorer over the request's own
  labeled graphs and explains them;
* the subgraphx lane rides its own sampling scales.
"""

from __future__ import annotations

import numpy as np
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


def _train_gnn(client) -> dict:
    r = client.post(
        "/api/v1/dl/gnn-train",
        headers=auth_headers(),
        json={
            "synthetic_n": 12, "seed": 3, "epochs": 60, "lr": 0.08,
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _two_graph_fixture() -> list[dict]:
    """A seeded two-graph fixture (5 nodes each, different shapes).

    Node features stay 2-wide — the synthetic motif fixture's own
    `n_features` default (the trained model's node width).
    """
    rng = np.random.default_rng(14)
    graphs = []
    for n_edges in (6, 8):
        x = rng.normal(size=(5, 2)).round(4).tolist()
        edges: list[list[int]] = [[i % 5, (i + 1) % 5] for i in range(n_edges)]
        graphs.append({"x": x, "edge_index": edges})
    return graphs


def _explain(client, model_id: str, **overrides) -> dict:
    body: dict = {
        "model_id": model_id,
        "graphs": _two_graph_fixture(),
        "method": "gnnexplainer",
        "top_k": 4,
        "seed": 0,
        "iters": 20,
    }
    body.update(overrides)
    return client.post(
        "/api/v1/dl/gnn-explain", headers=auth_headers(), json=body,
    )


def _explain_ok(client, model_id: str, **overrides) -> dict:
    r = _explain(client, model_id, **overrides)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_mask_explainer_computes_fidelities_on_the_two_graph_fixture(client):
    payload = _train_gnn(client)
    model_id = payload["model"]["model_id"]

    answer = _explain_ok(client, model_id)

    assert answer["mode"] == "sync"
    assert answer["method"] == "gnnexplainer"
    assert answer["top_k"] == 4
    assert len(answer["graphs"]) == 2
    for index, row in enumerate(answer["graphs"]):
        n_edges = len(row["edge_index"])
        assert row["index"] == index
        assert row["n_nodes"] == 5
        assert row["n_edges"] == n_edges
        assert len(row["scores"]) == n_edges, (
            "the scores align with the graph's own edge_index columns"
        )
        for score in row["scores"]:
            assert 0.0 <= score <= 1.0
        # the explained subgraph never exceeds the full graph size:
        # top_edges is a sorted subset of the edge columns
        assert len(row["top_edges"]) <= n_edges
        assert row["top_edges"] == sorted(row["top_edges"])
        assert set(row["top_edges"]) <= set(range(n_edges))
        # the fidelity verdicts are computed (floats, never nulls)
        assert isinstance(row["fidelity_plus"], float)
        assert isinstance(row["fidelity_minus"], float)
        # sparsity is the removed fraction: 1 - |top|/n_edges
        assert row["sparsity"] == pytest.approx(
            1.0 - len(row["top_edges"]) / n_edges
        )
        # the layout: one deterministic node position per node
        assert row["layout"]["kind"] == "circular"
        assert len(row["layout"]["nodes"]) == 5
        # the target class is the model's own verdict (stamped or argmax)
        assert row["target_class"] in (0, 1, 2)


def test_the_target_stamp_picks_one_class_for_every_pass(client):
    payload = _train_gnn(client)
    model_id = payload["model"]["model_id"]

    answer = _explain_ok(client, model_id, target=1)
    for row in answer["graphs"]:
        assert row["target_class"] == 1, (
            "the stamped target rides every explainer and both "
            "fidelity passes"
        )


def test_subgraphx_rides_its_own_sampling_scales(client):
    payload = _train_gnn(client)
    model_id = payload["model"]["model_id"]

    answer = _explain_ok(
        client, model_id, method="subgraphx", iters=None,
    )
    assert answer["method"] == "subgraphx"
    for row in answer["graphs"]:
        assert row["method"] == "subgraphx"
        assert len(row["scores"]) == row["n_edges"]
        assert len(row["top_edges"]) <= row["n_edges"]
        assert 0.0 <= row["sparsity"] <= 1.0


def test_the_pgexplainer_lane_trains_over_the_requests_own_labels(client):
    payload = _train_gnn(client)
    model_id = payload["model"]["model_id"]

    answer = _explain_ok(
        client,
        model_id,
        method="pgexplainer",
        labels=[0, 1],
        iters=20,
    )
    assert answer["method"] == "pgexplainer"
    for row in answer["graphs"]:
        assert row["method"] == "pgexplainer"
        assert len(row["scores"]) == row["n_edges"]
        assert len(row["top_edges"]) <= row["n_edges"]
    # the PG targets rode the per-graph labels
    assert answer["graphs"][0]["target_class"] == 0
    assert answer["graphs"][1]["target_class"] == 1


def test_the_honest_404_and_422_lanes(client):
    # unknown model
    r = _explain(client, "no-such-model")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown model_id."

    # a non-graph model: train a tabular DL model, then explain
    import pandas as pd

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(30, 2)), columns=["a", "b"])
    frame["y"] = (frame["a"] > 0).astype(int)
    upload = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={
            "file": ("t.csv", frame.to_csv(index=False).encode(), "text/csv")
        },
        data={"target": "y", "mode": "sync"},
    )
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()["dataset"]["dataset_id"]
    trained = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset_id,
            "network": "deep_mlp_classifier",
            "hyperparams": {"epochs": 3, "batch_size": 8},
            "mode": "sync",
        },
    )
    assert trained.status_code == 200, trained.text
    tabular_id = trained.json()["model"]["model_id"]
    r = _explain(client, tabular_id)
    assert r.status_code == 422
    assert "is not a graph model" in r.json()["detail"]

    payload = _train_gnn(client)
    model_id = payload["model"]["model_id"]
    graphs = _two_graph_fixture()

    # labels with a non-pgexplainer method
    r = _explain(client, model_id, labels=[0, 1])
    assert r.status_code == 422
    assert "labels ride the pgexplainer lane only" in r.json()["detail"]

    # pgexplainer without labels
    r = _explain(client, model_id, method="pgexplainer")
    assert r.status_code == 422
    assert "the pgexplainer lane needs labels" in r.json()["detail"]

    # a misaligned labels length
    r = _explain(
        client, model_id, method="pgexplainer", labels=[0, 1, 2],
    )
    assert r.status_code == 422
    assert "one per graph" in r.json()["detail"]

    # the iters knob on a sampling explainer
    r = _explain(client, model_id, method="subgraphx", iters=50)
    assert r.status_code == 422
    assert "the iters knob rides the optimizer explainers only" in (
        r.json()["detail"]
    )

    # the attention explainer's own GAT requirement (the engine's
    # message rides verbatim — the trained model is a GCN stack)
    r = client.post(
        "/api/v1/dl/gnn-explain",
        headers=auth_headers(),
        json={
            "model_id": model_id,
            "graphs": graphs,
            "method": "attention",
        },
    )
    assert r.status_code == 422
    assert "no GAT layer" in r.json()["detail"]

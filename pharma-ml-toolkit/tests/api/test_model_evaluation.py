"""part13 Phase 4's queued row — the Model-Evaluation/Applicability-
Domain API connection: the conformal + coverage-diagnostics family
over the REAL backend.

Every route runs against the real conformal calibration (split
regression intervals, marginal + label-conditional classification
sets), the real coverage diagnostics (Wilson-bounded groups, the
scaffold-aware mode bridging the splits' own Murcko extractor), and
session-trained REAL models from /api/v1/ml/train. No mocks: the
finite-sample guarantee's honest failure modes (too-few calibration
rows for an alpha, classifier-vs-regressor confusion, unaligned
rows) surface the backend's own typed errors verbatim as 422s, and
the empirical coverage is asserted only as empirical.

The GNN/MPNN API lane (part13 Phase 5's opening leg) rides the same
file's second class: the real GraphClassifier trained over the
backend's own synthetic motif fixture and over inline graph specs,
its own prediction path, and the tabular surfaces' honest refusal
guards.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.ml.gnn import synthetic_motif_graphs

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _regression_csv_bytes(rows: int = 48) -> bytes:
    """A continuous-target CSV with a strong linear signal."""
    rng = np.random.default_rng(0)
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for _ in range(rows):
        a = round(float(rng.normal()), 4)
        b = round(float(rng.normal()), 4)
        y = round(2.0 * a - b + 0.1 * float(rng.normal()), 4)
        buf.write(f"{a},{b},{y}\n")
    return buf.getvalue().encode()


def _classification_csv_bytes(rows: int = 60) -> bytes:
    """A two-class CSV with a separating signal."""
    rng = np.random.default_rng(1)
    buf = io.StringIO()
    buf.write("x1,x2,label\n")
    for _ in range(rows):
        a = round(float(rng.normal()), 4)
        b = round(float(rng.normal()), 4)
        label = int(a + b > 0.25)
        buf.write(f"{a},{b},{label}\n")
    return buf.getvalue().encode()


def _upload(client, data: bytes, target: str) -> dict:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": ("toy.csv", data, "text/csv")},
        data={"target": target, "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _train(client, dataset_id: str, task: str, model: str = "random_forest") -> dict:
    r = client.post(
        "/api/v1/ml/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset_id,
            "model": model,
            "task": task,
            "tune": False,
            "n_jobs": -1,
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["model"]


def _wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {job}")


def _job_result(client, job_id: str) -> dict:
    r = client.get(
        f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
    )
    assert r.status_code == 200, r.text
    return r.json()["result"]


_SCAFFOLD_SMILES = [
    "CCO",             # acyclic (no ring)
    "c1ccccc1O",       # benzene scaffold
    "CCOCC",           # acyclic
    "c1ccccc1Cl",      # benzene scaffold
    "not a smiles",    # invalid sentinel
    "c1ccc(cc1)O",     # benzene scaffold (alternate spelling)
]


class TestModelEvaluationRoutes:
    """The /api/v1/model-evaluation family (part13 Phase 4)."""

    def _prepare_regression(self, client):
        ds = _upload(client, _regression_csv_bytes(), target="y")
        model = _train(client, ds["dataset_id"], task="regression")
        rows = _regression_csv_bytes().decode().splitlines()[1:]
        parsed = [
            [float(v) for v in line.split(",")] for line in rows
        ]
        return model, parsed

    def _prepare_classification(self, client):
        ds = _upload(client, _classification_csv_bytes(), target="label")
        model = _train(client, ds["dataset_id"], task="classification")
        rows = _classification_csv_bytes().decode().splitlines()[1:]
        parsed = [
            [float(v) if i < 2 else int(v) for i, v in enumerate(line.split(","))]
            for line in rows
        ]
        return model, parsed

    def test_catalog_lists_the_three_entry_points(self, client):
        r = client.get(
            "/api/v1/model-evaluation/catalog", headers=auth_headers()
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        routes = [e["route"] for e in payload["entry_points"]]
        assert routes == [
            "/conformal-regression",
            "/conformal-classification",
            "/coverage-diagnostics",
        ]
        assert payload["entry_points"][2]["backend"] == (
            "coverage_by_group | scaffold_aware_coverage"
        )
        assert "never the conformal guarantee" in " ".join(payload["notes"])

    def test_conformal_regression_full_lane_with_scaffold_diagnostics(self, client):
        model, parsed = self._prepare_regression(client)
        x_cal = [row[:2] for row in parsed[:20]]
        y_cal = [row[2] for row in parsed[:20]]
        x_test = [row[:2] for row in parsed[20:36]]
        y_test = [row[2] for row in parsed[20:36]]
        smiles = (_SCAFFOLD_SMILES * 3)[:16]
        r = client.post(
            "/api/v1/model-evaluation/conformal-regression",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": x_cal, "y_cal": y_cal,
                "x_test": x_test, "y_test": y_test,
                "smiles": smiles,
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["alpha"] == 0.1
        assert payload["n_calibration"] == 20
        # rank = ceil((20 + 1) * 0.9) = 19 (1-based)
        assert payload["rank"] == 19
        assert payload["radius"] > 0.0
        assert payload["mean_width"] == pytest.approx(2.0 * payload["radius"])
        assert payload["feature_names"] == ["x1", "x2"]
        assert len(payload["intervals"]) == 16
        first = payload["intervals"][0]
        assert set(first) == {"prediction", "lower", "upper"}
        assert first["lower"] < first["prediction"] < first["upper"]
        empirical = payload["empirical"]
        assert empirical["n_test"] == 16
        assert 0.0 <= empirical["coverage"] <= 1.0
        assert empirical["n_covered"] == pytest.approx(
            empirical["coverage"] * 16
        )
        diagnostics = payload["coverage_diagnostics"]
        assert diagnostics["mode"] == "scaffold"
        groups = [g["group"] for g in diagnostics["groups"]]
        # first-appearance order with the honest sentinels present
        assert groups[0] == "acyclic"
        assert "invalid" in groups
        assert "c1ccccc1" in groups
        overall = diagnostics["overall"]
        assert overall["n_test"] == 16
        assert 0.0 <= overall["lower"] <= overall["coverage"] <= overall["upper"] <= 1.0

    def test_conformal_regression_calibration_only(self, client):
        model, parsed = self._prepare_regression(client)
        r = client.post(
            "/api/v1/model-evaluation/conformal-regression",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": [row[:2] for row in parsed[:12]],
                "y_cal": [row[2] for row in parsed[:12]],
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_calibration"] == 12
        assert payload["intervals"] is None
        assert payload["empirical"] is None
        assert payload["coverage_diagnostics"] is None

    def test_conformal_regression_classifier_refused_up_front(self, client):
        model, _parsed = self._prepare_classification(client)
        r = client.post(
            "/api/v1/model-evaluation/conformal-regression",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": [[0.0, 0.0]] * 8,
                "y_cal": [0.5] * 8,
            },
        )
        assert r.status_code == 422
        assert "regression" in r.json()["detail"]

    def test_conformal_regression_too_few_rows_for_alpha(self, client):
        model, parsed = self._prepare_regression(client)
        r = client.post(
            "/api/v1/model-evaluation/conformal-regression",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": [row[:2] for row in parsed[:12]],
                "y_cal": [row[2] for row in parsed[:12]],
                "alpha": 0.05,
            },
        )
        assert r.status_code == 422
        # the backend's own finite-sample honesty, verbatim
        assert "cannot give finite intervals" in r.json()["detail"]

    def test_conformal_regression_unknown_model_404(self, client):
        r = client.post(
            "/api/v1/model-evaluation/conformal-regression",
            headers=auth_headers(),
            json={
                "model_id": "mdl-does-not-exist",
                "x_cal": [[0.0, 0.0]] * 8,
                "y_cal": [0.5] * 8,
            },
        )
        assert r.status_code == 404

    def test_conformal_regression_groups_without_y_test_refused(self, client):
        model, parsed = self._prepare_regression(client)
        r = client.post(
            "/api/v1/model-evaluation/conformal-regression",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": [row[:2] for row in parsed[:12]],
                "y_cal": [row[2] for row in parsed[:12]],
                "x_test": [row[:2] for row in parsed[12:16]],
                "groups": ["a", "b", "a", "b"],
            },
        )
        assert r.status_code == 422
        assert "need y_test" in r.json()["detail"]

    def test_conformal_classification_split_lane(self, client):
        model, parsed = self._prepare_classification(client)
        x_cal = [row[:2] for row in parsed[:30]]
        y_cal = [row[2] for row in parsed[:30]]
        x_test = [row[:2] for row in parsed[30:50]]
        y_test = [row[2] for row in parsed[30:50]]
        r = client.post(
            "/api/v1/model-evaluation/conformal-classification",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": x_cal, "y_cal": y_cal,
                "x_test": x_test, "y_test": y_test,
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["conditional"] == "split"
        assert payload["classes"] == [0, 1]
        assert payload["n_calibration"] == 30
        assert payload["threshold"] is not None
        assert payload["class_thresholds"] is None
        assert len(payload["sets"]) == 20
        row = payload["sets"][0]
        assert set(row) == {"prediction_set", "set_size", "covered"}
        assert row["set_size"] == len(row["prediction_set"])
        empirical = payload["empirical"]
        assert empirical["n_test"] == 20
        assert 0.0 <= empirical["coverage"] <= 1.0
        assert empirical["mean_set_size"] >= 1.0
        per_class = payload["per_class"]
        assert per_class["mode"] == "groups"
        assert {g["group"] for g in per_class["groups"]} == {0, 1}
        assert sum(g["n_test"] for g in per_class["groups"]) == 20

    def test_conformal_classification_label_conditional(self, client):
        model, parsed = self._prepare_classification(client)
        x_cal = [row[:2] for row in parsed[:30]]
        y_cal = [row[2] for row in parsed[:30]]
        r = client.post(
            "/api/v1/model-evaluation/conformal-classification",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": x_cal, "y_cal": y_cal,
                "conditional": "label",
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["conditional"] == "label"
        assert payload["threshold"] is None
        assert set(payload["class_thresholds"]) == {"0", "1"}  # JSON dict keys
        summary = payload["calibration_summary"]
        assert len(summary) == 2
        assert {row["class_label"] for row in summary} == {0, 1}
        assert payload["sets"] is None
        assert payload["empirical"] is None

    def test_conformal_classification_regression_refused(self, client):
        model, _parsed = self._prepare_regression(client)
        r = client.post(
            "/api/v1/model-evaluation/conformal-classification",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_cal": [[0.0, 0.0]] * 8,
                "y_cal": [0, 1, 0, 1, 0, 1, 0, 1],
            },
        )
        assert r.status_code == 422
        assert "classification" in r.json()["detail"]

    def test_coverage_diagnostics_groups_wilson_bounds(self, client):
        covered = [True, True, False, True, True, True, False, True]
        groups = ["train", "train", "train", "train", "test", "test", "test", "test"]
        sizes = [2.0] * 8
        r = client.post(
            "/api/v1/model-evaluation/coverage-diagnostics",
            headers=auth_headers(),
            json={
                "covered": covered, "groups": groups, "sizes": sizes,
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["mode"] == "groups"
        assert payload["confidence"] == 0.95
        by_group = {g["group"]: g for g in payload["groups"]}
        assert by_group["train"]["coverage"] == 0.75
        assert by_group["test"]["coverage"] == 0.75
        assert by_group["train"]["mean_size"] == 2.0
        overall = payload["overall"]
        assert overall["n_test"] == 8
        assert overall["n_covered"] == 6
        assert overall["coverage"] == 0.75
        # Wilson bounds bracket the point estimate
        assert overall["lower"] < 0.75 < overall["upper"]

    def test_coverage_diagnostics_scaffold_sentinels(self, client):
        covered = [True, True, False, True, False, True]
        smiles = ["CCO", "c1ccccc1O", "CCO", "c1ccccc1Cl", "not a smiles", "CCOCC"]
        r = client.post(
            "/api/v1/model-evaluation/coverage-diagnostics",
            headers=auth_headers(),
            json={"covered": covered, "smiles": smiles},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["mode"] == "scaffold"
        groups = {g["group"]: g for g in payload["groups"]}
        assert groups["acyclic"]["n_test"] == 3  # CCO, CCO, CCOCC
        assert groups["acyclic"]["coverage"] == 2.0 / 3.0
        assert groups["c1ccccc1"]["n_test"] == 2
        assert groups["c1ccccc1"]["coverage"] == 1.0
        assert groups["invalid"]["n_test"] == 1
        assert groups["invalid"]["coverage"] == 0.0

    def test_coverage_diagnostics_needs_a_grouping(self, client):
        r = client.post(
            "/api/v1/model-evaluation/coverage-diagnostics",
            headers=auth_headers(),
            json={"covered": [True, False]},
        )
        assert r.status_code == 422
        assert "either groups or smiles" in r.json()["detail"]

    def test_coverage_diagnostics_both_groupings_refused(self, client):
        r = client.post(
            "/api/v1/model-evaluation/coverage-diagnostics",
            headers=auth_headers(),
            json={
                "covered": [True, False],
                "groups": ["a", "b"],
                "smiles": ["CCO", "CCO"],
            },
        )
        assert r.status_code == 422
        assert "not both" in r.json()["detail"]

    def test_coverage_diagnostics_length_mismatch(self, client):
        r = client.post(
            "/api/v1/model-evaluation/coverage-diagnostics",
            headers=auth_headers(),
            json={"covered": [True, False], "groups": ["a", "b", "c"]},
        )
        assert r.status_code == 422
        assert "they must match" in r.json()["detail"]


class TestGnnApiLane:
    """part13 Phase 5's opening leg: the GNN/MPNN API lane."""

    def _train_synthetic(self, client, mode: str = "sync") -> dict:
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={
                "synthetic_n": 18, "seed": 11, "epochs": 120, "lr": 0.08,
                "mode": mode,
            },
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_networks_catalog_carries_the_gnn_block(self, client):
        r = client.get("/api/v1/dl/networks", headers=auth_headers())
        assert r.status_code == 200, r.text
        gnn = r.json()["gnn"]
        assert gnn["convs"] == ["gcn", "sage", "gin", "gat", "edge"]
        assert gnn["pools"] == ["mean", "sum", "max", "topk", "sag"]
        assert gnn["train_url"] == "/api/v1/dl/gnn-train"
        assert gnn["predict_url"] == "/api/v1/dl/gnn-predict"

    def test_gnn_train_sync_synthetic_motifs(self, client):
        payload = self._train_synthetic(client)
        assert payload["mode"] == "sync"
        model = payload["model"]
        assert model["graph_model"] is True
        assert model["engine"] == "torch"
        assert model["predict_url"] == "/api/v1/dl/gnn-predict"
        assert payload["n_graphs"] == 18
        assert payload["classes"] == [0, 1, 2]
        # the doctest's own bar: 120 epochs / lr 0.08 / seed 11 >= 0.9
        assert payload["train_accuracy"] >= 0.9
        assert len(payload["loss_history_tail"]) == 5
        assert payload["loss_history_tail"][-1] < 1.0

    def test_gnn_train_job_mode_completes(self, client):
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={
                "synthetic_n": 12, "seed": 3, "epochs": 60, "lr": 0.08,
                "mode": "job",
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        assert job["operation"] == "dl.gnn-train"
        assert job["workspace"] == "deep-learning"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = _job_result(client, job["job_id"])
        assert result["n_graphs"] == 12
        assert result["model"]["graph_model"] is True

    def test_gnn_train_inline_graphs(self, client):
        # two hand-built families: 3-node chains (label 0) vs
        # 4-node chains (label 1) — separable by size alone
        graphs = []
        labels = []
        for i in range(6):
            n = 3
            graphs.append({
                "x": [[1.0, float(i) / 6] for _ in range(n)],
                "edge_index": [[0, 1], [1, 2]],
            })
            labels.append(0)
        for i in range(6):
            n = 4
            graphs.append({
                "x": [[0.0, float(i) / 6] for _ in range(n)],
                "edge_index": [[0, 1, 2], [1, 2, 3]],
            })
            labels.append(1)
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={
                "graphs": graphs, "labels": labels,
                "epochs": 150, "lr": 0.08, "seed": 7, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_graphs"] == 12
        assert payload["classes"] == [0, 1]
        assert payload["train_accuracy"] >= 0.9

    def test_gnn_train_graphs_and_synthetic_both_refused(self, client):
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={
                "graphs": [
                    {"x": [[1.0]], "edge_index": []},
                    {"x": [[1.0]], "edge_index": []},
                ],
                "labels": [0, 1],
                "synthetic_n": 18,
            },
        )
        assert r.status_code == 422
        assert "not both" in r.json()["detail"]

    def test_gnn_train_needs_a_source(self, client):
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={"mode": "sync"},
        )
        assert r.status_code == 422
        assert "inline graphs+labels or synthetic_n" in r.json()["detail"]

    def test_gnn_train_labels_length_mismatch(self, client):
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={
                "graphs": [
                    {"x": [[1.0]], "edge_index": []},
                    {"x": [[1.0]], "edge_index": []},
                ],
                "labels": [0, 1, 0],
            },
        )
        assert r.status_code == 422
        assert "one label per graph" in r.json()["detail"]

    def test_gnn_train_mixed_node_feature_widths_refused(self, client):
        r = client.post(
            "/api/v1/dl/gnn-train",
            headers=auth_headers(),
            json={
                "graphs": [
                    {"x": [[1.0], [2.0]], "edge_index": [[0], [1]]},
                    {"x": [[1.0, 0.5]], "edge_index": []},
                ],
                "labels": [0, 1],
            },
        )
        assert r.status_code == 422
        assert "one node-feature width" in r.json()["detail"]

    def test_gnn_predict_roundtrip_with_probabilities(self, client):
        trained = self._train_synthetic(client)
        model_id = trained["model"]["model_id"]
        graphs, y = synthetic_motif_graphs(4, seed=11)
        specs = [
            {"x": g.x.tolist(), "edge_index": g.edge_index.tolist()}
            for g in graphs
        ]
        r = client.post(
            "/api/v1/dl/gnn-predict",
            headers=auth_headers(),
            json={"model_id": model_id, "graphs": specs, "probabilities": True},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_graphs"] == 4
        assert payload["classes"] == [0, 1, 2]
        # the trained model nails its own fixture families
        assert payload["predictions"] == [int(v) for v in y]
        assert len(payload["probabilities"]) == 4
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in payload["probabilities"])

    def test_gnn_predict_node_feature_width_guard(self, client):
        trained = self._train_synthetic(client)
        r = client.post(
            "/api/v1/dl/gnn-predict",
            headers=auth_headers(),
            json={
                "model_id": trained["model"]["model_id"],
                "graphs": [{"x": [[1.0]], "edge_index": []}],
            },
        )
        assert r.status_code == 422
        assert "node features" in r.json()["detail"]

    def test_gnn_predict_unknown_model_404(self, client):
        r = client.post(
            "/api/v1/dl/gnn-predict",
            headers=auth_headers(),
            json={
                "model_id": "mdl-nope",
                "graphs": [{"x": [[1.0]], "edge_index": []}],
            },
        )
        assert r.status_code == 404

    def test_gnn_predict_tabular_model_refused(self, client):
        ds = _upload(client, _classification_csv_bytes(24), target="label")
        model = _train(client, ds["dataset_id"], task="classification")
        r = client.post(
            "/api/v1/dl/gnn-predict",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "graphs": [{"x": [[1.0]], "edge_index": []}],
            },
        )
        assert r.status_code == 422
        assert "not a graph model" in r.json()["detail"]

    def test_tabular_predict_refuses_graph_models(self, client):
        trained = self._train_synthetic(client)
        r = client.post(
            "/api/v1/ml/predict",
            headers=auth_headers(),
            json={
                "model_id": trained["model"]["model_id"],
                "rows": [{"x1": 1.0, "x2": 2.0}],
            },
        )
        assert r.status_code == 422
        assert "gnn-predict" in r.json()["detail"]

    def test_tabular_explain_refuses_graph_models(self, client):
        trained = self._train_synthetic(client)
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={"model_id": trained["model"]["model_id"]},
        )
        assert r.status_code == 422
        assert "graph model" in r.json()["detail"]

    def test_gnn_predict_bad_edge_index_422(self, client):
        trained = self._train_synthetic(client)
        r = client.post(
            "/api/v1/dl/gnn-predict",
            headers=auth_headers(),
            json={
                "model_id": trained["model"]["model_id"],
                "graphs": [
                    {"x": [[1.0], [2.0]], "edge_index": [[0], [7]]},
                ],
            },
        )
        assert r.status_code == 422
        # the backend's own validation, verbatim
        assert "outside 0..n-1" in r.json()["detail"]

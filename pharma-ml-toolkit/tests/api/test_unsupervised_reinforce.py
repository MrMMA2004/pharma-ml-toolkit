"""part18 §4.3/§4.5/§6 — the unsupervised (clustering + anomaly) and
the experimental reinforcement-learning API tests over the REAL
in-tree backend (``pharmaml.ml.clustering`` / ``pharmaml.ml.anomaly``
/ ``pharmaml.ml.reinforce``).

No mocks: every answer is the real module's own arithmetic over the
seeded SYNTHETIC datasets (blobs with planted noise rows / planted
outliers -- the KNOWN structure the assertions verify).  The honesty
pins: the degenerate lanes (all-noise DBSCAN, single-cluster
mean-shift) surface their states instead of fabricating scores; the
anomaly response carries the contamination-not-ground-truth note;
the RL lane is labelled experimental everywhere and never touches
the supervised MODEL_REGISTRY; the spectral clustering route refuses
a missing n_clusters as a 422; the job mode's records public() view
round-trips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"

CLUSTER_URL = "/api/v1/unsupervised/cluster"
ANOMALY_URL = "/api/v1/unsupervised/anomaly"
PCA_URL = "/api/v1/unsupervised/pca"
UNSUPERVISED_METHODS_URL = "/api/v1/unsupervised/methods"
RL_TRAIN_URL = "/api/v1/reinforce/train"
RL_EVALUATE_URL = "/api/v1/reinforce/evaluate"
RL_METHODS_URL = "/api/v1/reinforce/methods"


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def _blobs_csv(planted_noise: bool = False) -> bytes:
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 0.3, 80), rng.normal(6, 0.3, 80)])
    y = np.concatenate([rng.normal(0, 0.3, 80), rng.normal(6, 0.3, 80)])
    frame = pd.DataFrame({"x": x, "y": y})
    if planted_noise:
        far = pd.DataFrame(
            {"x": rng.uniform(25, 30, 3), "y": rng.uniform(25, 30, 3)}
        )
        frame = pd.concat([frame, far], ignore_index=True)
    return frame.to_csv(index=False).encode("utf-8")


def _outliers_csv() -> bytes:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("wxyz"))
    frame.iloc[0] = frame.iloc[0] + 18.0
    frame.iloc[1] = frame.iloc[1] - 20.0
    frame.iloc[2] = frame.iloc[2] + 21.0
    return frame.to_csv(index=False).encode("utf-8")


def _upload(client, data: bytes, name: str) -> str:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, data, "text/csv")},
        data={"mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["dataset_id"]


# ---------------------------------------------------------------------------
# the unsupervised methods listing
# ---------------------------------------------------------------------------
class TestMethodsListing:
    def test_lists_the_six_families_plus_anomaly_and_pca(self, client):
        r = client.get(UNSUPERVISED_METHODS_URL, headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        assert {c["method"] for c in body["clustering"]} == {
            "dbscan", "meanshift", "spectral", "kmeans", "gmm",
            "hierarchical"
        }
        assert body["anomaly_detection"][0]["method"] == "isolation_forest"
        # ML-4's PCA sibling rides the same listing
        assert [d["method"] for d in body["dimensionality_reduction"]] == ["pca"]
        assert body["dimensionality_reduction"][0]["route"] == PCA_URL
        for entry in body["clustering"] + body["dimensionality_reduction"]:
            assert entry["route"] in (CLUSTER_URL, PCA_URL)
            assert entry["python"].startswith("pharmaml.ml.")


# ---------------------------------------------------------------------------
# the clustering routes
# ---------------------------------------------------------------------------
class TestClusterRoutes:
    def test_dbscan_sync(self, client):
        dataset_id = _upload(client, _blobs_csv(planted_noise=True), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "dbscan",
                "eps": 1.0, "min_samples": 4, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["method"] == "dbscan"
        assert body["eps_source"] == "manual"
        assert body["n_samples"] == 163
        diag = body["diagnostics"]
        assert diag["n_clusters"] == 2
        assert diag["noise_count"] >= 3
        assert diag["noise_ratio"] > 0.0
        assert diag["silhouette"] is not None and diag["silhouette"] > 0.5
        assert -1 in body["labels"]["unique"]
        assert "DBSCAN" in body["summary"]

    def test_dbscan_all_noise_degenerate_is_honest(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "dbscan",
                "eps": 0.0001, "mode": "sync",
            },
        )
        assert r.status_code == 200
        body = r.json()["unsupervised"]
        assert body["diagnostics"]["n_clusters"] == 0
        assert body["diagnostics"]["silhouette"] is None
        assert body["diagnostics"]["noise_ratio"] == 1.0

    def test_meanshift_sync(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "meanshift",
                "bandwidth": 1.5, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["method"] == "meanshift"
        assert body["bandwidth_source"] == "manual"
        assert body["n_modes"] >= 2
        assert body["diagnostics"]["n_clusters"] >= 2
        assert body["diagnostics"]["noise_ratio"] == 0.0

    def test_spectral_sync_rbf(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "spectral",
                "n_clusters": 2, "affinity": "rbf", "gamma": 0.5,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["n_clusters"] == 2
        assert body["affinity"] == "rbf"
        assert body["gamma"] is not None

    def test_spectral_missing_clusters_is_422(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "method": "spectral", "mode": "sync"},
        )
        assert r.status_code == 422

    def test_unknown_dataset_is_404(self, client):
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={"dataset_id": "nope", "method": "dbscan", "mode": "sync"},
        )
        assert r.status_code == 404

    def test_dbscan_job_mode(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "method": "dbscan", "eps": 1.0},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "job"
        record = body["job"]
        assert record["job_id"]
        assert record["state"] in (
            "QUEUED", "queued", "RUNNING", "running",
            "DONE", "done", "SUCCEEDED", "succeeded",
        )


# ---------------------------------------------------------------------------
# the ML-4 cluster families (kmeans / gmm / hierarchical — the
# WORKFLOW_ENHANCEMENT_ROADMAP's row 11)
# ---------------------------------------------------------------------------
class TestMl4ClusterRoutes:
    def test_kmeans_scan_picks_the_blobs(self, client):
        """k omitted: the engine's own silhouette scan (the known
        two-blob structure is found, k=2, and the scan table rides
        the response)."""
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "method": "kmeans", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["method"] == "kmeans"
        assert body["k"] == 2
        assert "silhouette scan" in body["k_source"]
        assert body["n_samples"] == 160
        assert body["diagnostics"]["n_clusters"] == 2
        assert body["diagnostics"]["silhouette"] > 0.5
        # the elbow + silhouette sweep the engine actually ran
        scan = body["scan"]
        assert len(scan) >= 2
        assert {"k", "inertia", "silhouette"}.issubset(set(scan[0]))
        best = max(scan, key=lambda row: row["silhouette"] or -1.0)
        assert best["k"] == body["k"]
        # cluster sizes carry the two 80-row blobs
        sizes = body["sizes"]
        assert sum(sizes.values()) == 160

    def test_kmeans_manual_k_is_manual(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "kmeans",
                "k": 3, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["k"] == 3
        assert body["k_source"] == "manual"
        assert body["scan"] == []

    def test_gmm_sweep_carries_the_bic_table(self, client):
        """n_components omitted: the AIC/BIC sweep rides the response,
        the criterion minimum is chosen, and the EM verdict is
        reported as its own boolean."""
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "gmm",
                "covariance_type": "full", "criterion": "bic",
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["method"] == "gmm"
        assert body["chosen_by"] == "bic"
        assert body["covariance_type"] == "full"
        assert body["converged"] is True
        assert isinstance(body["log_likelihood"], float)
        # the BIC table: every row carries the sweep's own keys
        table = body["bic_table"]
        assert len(table) >= 2
        assert {"n_components", "aic", "bic"}.issubset(set(table[0]))
        # the criterion minimum is the picked component count
        minimum = min(table, key=lambda row: row["bic"])
        assert minimum["n_components"] == body["n_components"]
        # the mixing weights sum to 1 (the responsibilities' parent)
        assert abs(sum(body["weights"].values()) - 1.0) < 1e-6

    def test_gmm_single_component_is_the_honest_degenerate(self, client):
        """n_components=1 over the two-blob fixture: a single cluster
        answer whose similarity metrics are None, never fabricated."""
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "gmm",
                "n_components": 1, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        diag = body["diagnostics"]
        assert diag["n_clusters"] == 1
        assert diag["silhouette"] is None
        assert diag["davies_bouldin"] is None
        assert diag["calinski_harabasz"] is None
        assert body["labels"]["unique"] == [0]

    def test_hierarchical_cut_carries_merge_heights(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "hierarchical",
                "n_clusters": 2, "linkage": "ward", "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["method"] == "hierarchical"
        assert body["k"] == 2
        assert body["k_source"] == "manual"
        assert body["linkage"] == "ward"
        assert body["diagnostics"]["n_clusters"] == 2
        assert body["diagnostics"]["silhouette"] > 0.5
        # the k-1 merge heights of the cut (scipy's Z[:,2] column)
        assert len(body["merge_heights"]) == body["k"] - 1
        assert all(height > 0.0 for height in body["merge_heights"])

    def test_hierarchical_default_k_is_the_engines_own(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            CLUSTER_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "method": "hierarchical",
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["unsupervised"]
        assert body["k"] == 3
        assert body["k_source"] == "the engine's own default (3)"


# ---------------------------------------------------------------------------
# the ML-4 PCA sibling route
# ---------------------------------------------------------------------------
class TestPcaRoute:
    def test_pca_sync_carries_the_variance_ladder_and_loadings(self, client):
        dataset_id = _upload(client, _blobs_csv(), "blobs.csv")
        r = client.post(
            PCA_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()["pca"]
        assert body["method"] == "pca"
        assert body["n_components"] >= 1
        assert body["scaled"] is True
        assert body["n_features"] == 1
        # the ladder: per-component shares + the cumulative run
        assert len(body["explained_variance_ratio"]) == body["n_components"]
        assert len(body["cumulative_variance"]) == body["n_components"]
        assert abs(sum(body["explained_variance_ratio"]) - 1.0) < 1e-9
        assert body["cumulative_variance"][-1] == body["total_variance_explained"]
        # the loadings preview (the what-does-PC1-mean table)
        assert len(body["loadings_preview"]) >= 1
        assert "PC1" in body["loadings_preview"][0]
        assert "honesty_note" in body

    def test_pca_explicit_components(self, client):
        dataset_id = _upload(client, _outliers_csv(), "outliers.csv")
        r = client.post(
            PCA_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "n_components": 2, "scale": False, "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()["pca"]
        assert body["n_components"] == 2
        assert body["scaled"] is False
        assert body["n_features"] == 3

    def test_pca_over_the_engine_cap_is_422(self, client):
        dataset_id = _upload(client, _outliers_csv(), "outliers.csv")
        r = client.post(
            PCA_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "n_components": 9, "mode": "sync"},
        )
        assert r.status_code == 422

    def test_pca_unknown_dataset_is_404(self, client):
        r = client.post(
            PCA_URL,
            headers=auth_headers(),
            json={"dataset_id": "nope", "mode": "sync"},
        )
        assert r.status_code == 404

    def test_pca_job_mode(self, client):
        dataset_id = _upload(client, _outliers_csv(), "outliers.csv")
        r = client.post(
            PCA_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "n_components": 2},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "job"
        assert body["job"]["job_id"]


# ---------------------------------------------------------------------------
# the anomaly route
# ---------------------------------------------------------------------------
class TestAnomalyRoute:
    def test_planted_outliers_flagged_sync(self, client):
        dataset_id = _upload(client, _outliers_csv(), "outliers.csv")
        r = client.post(
            ANOMALY_URL,
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id, "contamination": 0.03,
                "n_estimators": 100, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["anomaly"]
        assert body["method"] == "isolation_forest"
        assert set(body["outlier_positions"][:3]) == {0, 1, 2}
        assert body["n_outliers"] >= 3
        assert 0.02 <= body["anomaly_fraction"] <= 0.05
        assert body["score_threshold"] is not None
        assert body["feature_report"][0]["feature"] == "w"
        assert "ground truth" in body["honesty_note"]

    def test_bad_contamination_is_422(self, client):
        dataset_id = _upload(client, _outliers_csv(), "outliers.csv")
        r = client.post(
            ANOMALY_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "contamination": 0.9, "mode": "sync"},
        )
        assert r.status_code == 422

    def test_job_mode(self, client):
        dataset_id = _upload(client, _outliers_csv(), "outliers.csv")
        r = client.post(
            ANOMALY_URL,
            headers=auth_headers(),
            json={"dataset_id": dataset_id, "contamination": 0.03},
        )
        assert r.status_code == 200, r.text
        assert r.json()["mode"] == "job"


# ---------------------------------------------------------------------------
# the experimental RL lane
# ---------------------------------------------------------------------------
class TestReinforceRoutes:
    def test_methods_listing_labels_experimental(self, client):
        r = client.get(RL_METHODS_URL, headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "experimental"
        assert {a["method"] for a in body["agents"]} == {
            "q_learning", "sarsa", "dqn", "reinforce", "actor_critic"
        }
        assert "MODEL_REGISTRY" in body["separation_rule"]
        assert "no pharmaceutical RL environment" in body["scientific_honesty"]

    @pytest.mark.parametrize(
        "method,extra",
        [
            ("q_learning", {}),
            ("sarsa", {}),
            ("reinforce", {"lr": 0.1}),
            ("actor_critic", {}),
        ],
    )
    def test_trains_and_reports(self, client, method, extra):
        r = client.post(
            RL_TRAIN_URL,
            headers=auth_headers(),
            json={
                "method": method, "rows": 4, "cols": 5,
                "episodes": 150, "seed": 42, "mode": "sync", **extra,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["reinforcement"]
        assert body["method"] == method
        assert body["experimental"] is True
        assert body["success_rate"] >= 0.4
        curve = body["reward_curve"]
        assert curve["episodes"] == 150
        assert curve["last_20_mean"] > curve["first_20_mean"]
        assert body["policy"] and all(isinstance(v, int) for v in body["policy"].values())
        assert body["hyperparams"]["seed"] == 42

    def test_dqn_trains(self, client):
        r = client.post(
            RL_TRAIN_URL,
            headers=auth_headers(),
            json={
                "method": "dqn", "episodes": 60, "seed": 42,
                "batch": 16, "replay_size": 256, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["reinforcement"]
        assert body["method"] == "dqn"
        assert body["hyperparams"]["batch"] == 16

    def test_evaluate_route(self, client):
        train = client.post(
            RL_TRAIN_URL,
            headers=auth_headers(),
            json={
                "method": "q_learning", "episodes": 150, "seed": 42,
                "mode": "sync",
            },
        )
        assert train.status_code == 200
        policy = train.json()["reinforcement"]["policy"]
        r = client.post(
            RL_EVALUATE_URL,
            headers=auth_headers(),
            json={
                "method": "q_learning", "seed": 42, "episodes": 25,
                "policy": policy,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()["evaluation"]
        assert body["episodes"] == 25
        assert body["mean_reward"] > 0.5
        assert body["termination_rate"] == 1.0
        assert body["experimental"] is True

    def test_bad_policy_payload_is_422(self, client):
        r = client.post(
            RL_EVALUATE_URL,
            headers=auth_headers(),
            json={"policy": {"zero": 0}, "episodes": 5},
        )
        assert r.status_code == 422

    def test_job_mode(self, client):
        r = client.post(
            RL_TRAIN_URL,
            headers=auth_headers(),
            json={"method": "q_learning", "episodes": 50},
        )
        assert r.status_code == 200, r.text
        assert r.json()["mode"] == "job"

    def test_unknown_method_rejected_at_validation(self, client):
        r = client.post(
            RL_TRAIN_URL,
            headers=auth_headers(),
            json={"method": "ddpg", "mode": "sync"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# the drift guard: both routers are part of the OpenAPI contract
# ---------------------------------------------------------------------------
class TestContractDrift:
    def test_routes_in_the_openapi_contract(self, client):
        r = client.get("/api/openapi.json", headers=auth_headers())
        assert r.status_code == 200, r.text
        paths = r.json()["paths"]
        for url in (
            CLUSTER_URL, ANOMALY_URL, UNSUPERVISED_METHODS_URL,
            RL_TRAIN_URL, RL_EVALUATE_URL, RL_METHODS_URL,
        ):
            assert url in paths, f"{url} missing from the OpenAPI contract"

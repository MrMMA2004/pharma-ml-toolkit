"""part18 §5/§9 — the DL architecture expansion's API pins.

The new lanes' contract tests (the ``test_dl_semi_generative`` family's
conventions): every route here runs the REAL backend — the torch
tabular Transformer through the shared ``/dl/train`` lane, the
sequence CNN (synthetic fixture + inline sequences + its own predict
path + the single-path refusal on ``/ml/predict``), the tabular
autoencoder (train + encode + the honest unsupervised refusal), and
the feature-space GAN / DDPM lanes in the generative router (train +
sample + the distributional parity report + the not-molecules note).
"""

from __future__ import annotations

import io

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


def _csv_bytes(rows: int = 60) -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(rows):
        buf.write(f"{round(0.1 * i + 0.5, 3)},{(i * 7) % 5 * 0.25},{i % 2}\n")
    return buf.getvalue().encode()


@pytest.fixture()
def dataset_id(client) -> str:
    r = client.post(
        "/api/v1/datasets",
        files={"file": ("d.csv", _csv_bytes(), "text/csv")},
        data={"mode": "sync"},
        headers=auth_headers(),
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["dataset_id"]


class TestNetworksCatalog:
    def test_catalog_lists_the_transformer_families(self, client) -> None:
        r = client.get("/api/v1/dl/networks", headers=auth_headers())
        assert r.status_code == 200
        networks = {n["network"] for n in r.json()["networks"]}
        assert "tabular_transformer_classifier" in networks
        assert "tabular_transformer_regressor" in networks
        # the catalog's defaults are read off the class signatures — the
        # transformer entries carry their own (d_model/heads/n_blocks)
        entry = next(
            n for n in r.json()["networks"]
            if n["network"] == "tabular_transformer_classifier"
        )
        assert entry["engine"] == "torch"
        assert entry["defaults"]["d_model"] == 32
        assert entry["defaults"]["heads"] == 4

    def test_catalog_gnn_block_unchanged(self, client) -> None:
        r = client.get("/api/v1/dl/networks", headers=auth_headers())
        gnn = r.json()["gnn"]
        assert gnn["train_url"] == "/api/v1/dl/gnn-train"


class TestTabularTransformerLane:
    def test_sync_train_registers_a_model_that_predicts(
        self, client, dataset_id
    ) -> None:
        r = client.post(
            "/api/v1/dl/train",
            json={
                "dataset_id": dataset_id,
                "network": "tabular_transformer_classifier",
                "hyperparams": {
                    "d_model": 16, "heads": 4, "n_blocks": 1,
                    "dim_feedforward": 32, "epochs": 8,
                },
                "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        view = r.json()["model"]
        assert view["task"] == "classification"
        assert view["predict_url"] == "/api/v1/ml/predict"
        # the SINGLE session prediction path (never a second one)
        r = client.post(
            "/api/v1/ml/predict",
            json={"model_id": view["model_id"], "dataset_id": dataset_id,
                  "mode": "sync"},
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["predictions"]) == 60

    def test_unknown_hyperparameter_is_an_honest_422(
        self, client, dataset_id
    ) -> None:
        r = client.post(
            "/api/v1/dl/train",
            json={
                "dataset_id": dataset_id,
                "network": "tabular_transformer_classifier",
                "hyperparams": {"bogus_key": 1},
                "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 422
        assert "unknown hyperparameter" in r.json()["detail"]

    def test_mlp_rejects_transformer_only_keys(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/train",
            json={
                "dataset_id": dataset_id,
                "network": "deep_mlp_classifier",
                "hyperparams": {"d_model": 16},
                "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 422

    def test_task_mismatch_is_an_honest_422(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/train",
            json={
                "dataset_id": dataset_id,
                "network": "tabular_transformer_regressor",
                "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 422
        assert "regression model but" in r.json()["detail"]

    def test_job_mode_completes(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/train",
            json={
                "dataset_id": dataset_id,
                "network": "tabular_transformer_classifier",
                "hyperparams": {
                    "d_model": 16, "heads": 4, "n_blocks": 1,
                    "dim_feedforward": 32, "epochs": 5,
                },
                "mode": "job",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200
        job = r.json()["job"]
        assert job["state"] in {"PENDING", "RUNNING"}
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        # the job's result payload is the sync body's analog: a registered
        # model view with the single prediction path
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()
        view = result["result"]["model"]
        assert view["task"] == "classification"
        assert view["predict_url"] == "/api/v1/ml/predict"


def _wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    import time as _t

    deadline = _t.monotonic() + timeout
    job: dict = {}
    while _t.monotonic() < deadline:
        job = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        if job["state"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return job
        _t.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s: {job}")


class TestSequenceCnnLane:
    def test_synthetic_fixture_train_and_own_predict_path(self, client) -> None:
        r = client.post(
            "/api/v1/dl/cnn-train",
            json={
                "synthetic_n": 40, "epochs": 25, "channels": 8,
                "learning_rate": 3e-3, "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"]["sequence_model"] is True
        assert body["model"]["predict_url"] == "/api/v1/dl/cnn-predict"
        assert body["train_accuracy"] >= 0.9
        # the motif fixture is learnable by the kernel
        model_id = body["model"]["model_id"]
        motif_steps = [[0.0, 0.1, 0.2]] * 12 + [[3.0, 0.1, 0.2]] * 4
        plain_steps = [[0.0, 0.1, 0.2]] * 16
        r = client.post(
            "/api/v1/dl/cnn-predict",
            json={
                "model_id": model_id, "probabilities": True,
                "sequences": [
                    {"steps": motif_steps}, {"steps": plain_steps},
                ],
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert len(payload["predictions"]) == 2
        assert payload["probabilities"][0] and len(payload["probabilities"][0]) == 2

    def test_inline_sequences_train(self, client) -> None:
        import random

        rng = random.Random(0)
        sequences, labels = [], []
        for i in range(30):
            has_motif = i % 2 == 0
            steps = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(12)]
            if has_motif:
                for j in range(4, 8):
                    steps[j][0] += 3.0
            sequences.append({"steps": [[round(v, 4) for v in s] for s in steps]})
            labels.append(1 if has_motif else 0)
        r = client.post(
            "/api/v1/dl/cnn-train",
            json={
                "sequences": sequences, "labels": labels,
                "epochs": 30, "channels": 8, "learning_rate": 3e-3,
                "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        assert r.json()["model"]["task"] == "classification"

    def test_inline_regression_train(self, client) -> None:
        import random

        rng = random.Random(1)
        sequences, labels = [], []
        for i in range(24):
            steps = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(10)]
            sequences.append({"steps": [[round(v, 4) for v in s] for s in steps]})
            labels.append(round(sum(s[0] for s in steps) / 10.0, 4))
        r = client.post(
            "/api/v1/dl/cnn-train",
            json={
                "sequences": sequences, "labels": labels, "task": "regression",
                "epochs": 20, "channels": 8, "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        assert r.json()["model"]["task"] == "regression"

    def test_ml_predict_refuses_sequence_models(self, client) -> None:
        r = client.post(
            "/api/v1/dl/cnn-train",
            json={
                "synthetic_n": 20, "epochs": 3, "mode": "sync",
            },
            headers=auth_headers(),
        )
        model_id = r.json()["model"]["model_id"]
        r = client.post(
            "/api/v1/ml/predict",
            json={"model_id": model_id, "rows": [{"x1": 1.0, "x2": 2.0}]},
            headers=auth_headers(),
        )
        assert r.status_code == 422
        assert "sequence model" in r.json()["detail"]

    def test_both_sources_is_a_422(self, client) -> None:
        r = client.post(
            "/api/v1/dl/cnn-train",
            json={"synthetic_n": 20, "sequences": [{"steps": [[0.0]]}], "mode": "sync"},
            headers=auth_headers(),
        )
        assert r.status_code == 422

    def test_synthetic_regression_is_a_422(self, client) -> None:
        r = client.post(
            "/api/v1/dl/cnn-train",
            json={"synthetic_n": 20, "task": "regression", "mode": "sync"},
            headers=auth_headers(),
        )
        assert r.status_code == 422

    def test_predict_needs_a_sequence_model(self, client) -> None:
        r = client.post(
            "/api/v1/dl/cnn-predict",
            json={"model_id": "mdl-doesnotexist", "sequences": [{"steps": [[0.0]]}]},
            headers=auth_headers(),
        )
        assert r.status_code == 404


class TestAutoencoderLane:
    def test_train_and_encode(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/autoencoder-train",
            json={
                "dataset_id": dataset_id, "hidden_layers": [16],
                "latent_dim": 2, "epochs": 12, "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"]["task"] == "unsupervised"
        assert body["latent_dim"] == 2
        assert body["mean_reconstruction_error"] >= 0.0
        model_id = body["model"]["model_id"]
        r = client.post(
            "/api/v1/dl/autoencoder-encode",
            json={"model_id": model_id, "dataset_id": dataset_id, "top_k": 5},
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["latent_dim"] == 2
        assert len(payload["latent"]) == 60
        assert len(payload["reconstruction_errors"]) == 60
        assert len(payload["top_outliers"]) == 5
        assert "not ground truth" in payload["note"]

    def test_encode_inline_rows(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/autoencoder-train",
            json={
                "dataset_id": dataset_id, "hidden_layers": [8],
                "latent_dim": 2, "epochs": 3, "mode": "sync",
            },
            headers=auth_headers(),
        )
        model_id = r.json()["model"]["model_id"]
        r = client.post(
            "/api/v1/dl/autoencoder-encode",
            json={
                "model_id": model_id, "top_k": 2,
                "rows": [{"x1": 0.5, "x2": 0.25}, {"x1": 9.9, "x2": -9.9}],
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_rows"] == 2
        # the injected outlier tops the reconstruction-error ranking
        assert payload["top_outliers"][0]["index"] == 1

    def test_ml_predict_refuses_autoencoders(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/autoencoder-train",
            json={
                "dataset_id": dataset_id, "hidden_layers": [8],
                "latent_dim": 2, "epochs": 3, "mode": "sync",
            },
            headers=auth_headers(),
        )
        model_id = r.json()["model"]["model_id"]
        r = client.post(
            "/api/v1/ml/predict",
            json={"model_id": model_id, "dataset_id": dataset_id},
            headers=auth_headers(),
        )
        assert r.status_code == 422
        assert "unsupervised autoencoder" in r.json()["detail"]

    def test_explain_refuses_autoencoders(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/autoencoder-train",
            json={
                "dataset_id": dataset_id, "hidden_layers": [8],
                "latent_dim": 2, "epochs": 3, "mode": "sync",
            },
            headers=auth_headers(),
        )
        model_id = r.json()["model"]["model_id"]
        r = client.post(
            "/api/v1/ml/explain",
            json={"model_id": model_id, "dataset_id": dataset_id},
            headers=auth_headers(),
        )
        assert r.status_code == 422
        assert "unsupervised autoencoder" in r.json()["detail"]

    def test_both_sources_is_a_422(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/dl/autoencoder-encode",
            json={
                "model_id": "whatever", "dataset_id": dataset_id,
                "rows": [{"x1": 1.0}],
            },
            headers=auth_headers(),
        )
        assert r.status_code in {404, 422}

    def test_unknown_dataset_is_404(self, client) -> None:
        r = client.post(
            "/api/v1/dl/autoencoder-train",
            json={"dataset_id": "ds-missing", "mode": "sync"},
            headers=auth_headers(),
        )
        assert r.status_code == 404


class TestGanLane:
    def test_train_and_sample_with_parity(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/generative/gan/train",
            json={
                "dataset_id": dataset_id, "latent_dim": 8,
                "hidden_layers": [16], "epochs": 30, "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        prior = r.json()["prior"]
        assert prior["kind"] == "feature_gan"
        assert prior["n_train_rows"] == 60
        assert "NOT molecules" in prior["note"]
        prior_id = prior["prior_id"]
        r = client.post(
            "/api/v1/generative/gan/sample",
            json={"prior_id": prior_id, "n": 8, "seed": 3},
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert len(payload["rows"]) == 8 and len(payload["rows"][0]) == 2
        assert set(payload["quality"]) == {
            "n_ref", "mean_delta", "std_delta", "corr_l1"
        }
        assert "not molecules" in payload["note"]

    def test_sample_needs_a_feature_gan(self, client) -> None:
        r = client.post(
            "/api/v1/generative/gan/sample",
            json={"prior_id": "prior-missing"},
            headers=auth_headers(),
        )
        assert r.status_code == 404

    def test_unknown_dataset_is_404(self, client) -> None:
        r = client.post(
            "/api/v1/generative/gan/train",
            json={"dataset_id": "ds-missing", "mode": "sync"},
            headers=auth_headers(),
        )
        assert r.status_code == 404


class TestDiffusionLane:
    def test_train_and_sample_with_parity(self, client, dataset_id) -> None:
        r = client.post(
            "/api/v1/generative/diffusion/train",
            json={
                "dataset_id": dataset_id, "timesteps": 30,
                "hidden_layers": [16], "epochs": 30, "mode": "sync",
            },
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        prior = r.json()["prior"]
        assert prior["kind"] == "feature_diffusion"
        assert prior["n_epochs"] == 30
        prior_id = prior["prior_id"]
        r = client.post(
            "/api/v1/generative/diffusion/sample",
            json={"prior_id": prior_id, "n": 8, "seed": 3},
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert len(payload["rows"]) == 8 and len(payload["rows"][0]) == 2
        assert "corr_l1" in payload["quality"]

    def test_cross_kind_sampler_is_a_422(self, client, dataset_id) -> None:
        # a GAN prior cannot ride the diffusion sampler (one path per kind)
        r = client.post(
            "/api/v1/generative/gan/train",
            json={
                "dataset_id": dataset_id, "epochs": 2, "mode": "sync",
            },
            headers=auth_headers(),
        )
        prior_id = r.json()["prior"]["prior_id"]
        r = client.post(
            "/api/v1/generative/diffusion/sample",
            json={"prior_id": prior_id, "n": 4},
            headers=auth_headers(),
        )
        assert r.status_code == 422
        assert "not a feature diffusion model" in r.json()["detail"]

    def test_unknown_dataset_is_404(self, client) -> None:
        r = client.post(
            "/api/v1/generative/diffusion/train",
            json={"dataset_id": "ds-missing", "mode": "sync"},
            headers=auth_headers(),
        )
        assert r.status_code == 404


class TestContract:
    def test_openapi_carries_the_new_paths(self, client) -> None:
        spec = client.app.openapi()
        paths = spec["paths"]
        for path in (
            "/api/v1/dl/cnn-train",
            "/api/v1/dl/cnn-predict",
            "/api/v1/dl/autoencoder-train",
            "/api/v1/dl/autoencoder-encode",
            "/api/v1/generative/gan/train",
            "/api/v1/generative/gan/sample",
            "/api/v1/generative/diffusion/train",
            "/api/v1/generative/diffusion/sample",
        ):
            assert path in paths, path
        schemas = spec["components"]["schemas"]
        for name in (
            "CnnTrainRequest", "AutoencoderTrainRequest", "GanTrainRequest",
            "DiffusionTrainRequest",
        ):
            assert name in schemas, name

"""DL-3 (the roadmap's row 15) — ``POST /api/v1/dl/efficiency``.

The efficiency lab over the REAL weight engines
(``pharmaml.ml.quantization`` + ``pharmaml.ml.pruning``): the
persisted tabular estimator's own 2-D weight matrices (read off its
``_weights`` dict — the pickle-portable source of truth a stage-11
checkpoint carries) quantized through the engine's own
``quantization_report`` or magnitude-pruned through the engine's own
masks, with the metric delta scored on the model's own training
dataset through a pruned COPY.

The roadmap's own acceptance drives the file:

* the int8 report's dequantized error within the engine's own
  documented bound (rel < 0.02 — the ``1/2^8`` relative scale the
  engine's own test pins) on the seeded net's weight matrices;
* prune at 50% sparsity returns exactly 50% zeros (the local scope's
  own ceil arithmetic on the even-width layers);
* the persisted estimator is NEVER mutated (the pruned COPY is the
  only thing that scores after the mask);
* the honest 422/404 lanes: an unknown model (404), a non-tabular
  model (the autoencoder's own honest pointer), a prune without a
  level, a level on a quantize lane, a scope on a quantize lane;
* the report registers as a downloadable table artifact.
"""

from __future__ import annotations

import csv
import io

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


def _train_tabular(client, dataset_id: str) -> dict:
    r = client.post(
        "/api/v1/dl/train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset_id,
            "network": "deep_mlp_classifier",
            "hyperparams": {
                "hidden_layers": [8],
                "epochs": 20,
                "batch_size": 16,
                "random_state": 0,
            },
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _efficiency(client, model_id: str, **overrides) -> dict:
    body: dict = {"model_id": model_id, "mode": "quantize_int8"}
    body.update(overrides)
    return client.post(
        "/api/v1/dl/efficiency", headers=auth_headers(), json=body,
    )


def _efficiency_ok(client, model_id: str, **overrides) -> dict:
    r = _efficiency(client, model_id, **overrides)
    assert r.status_code == 200, r.text
    return r.json()


def test_int8_report_within_the_engines_own_documented_bound(client):
    dataset = _upload(client, _fixture_csv(), "eff-fixture.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    answer = _efficiency_ok(client, model_id, mode="quantize_int8")

    assert answer["mode"] == "sync"
    assert answer["efficiency_mode"] == "quantize_int8"
    assert answer["method"] == "int8"
    assert answer["network"] == "deep_mlp_classifier"
    assert answer["n_layers"] == len(answer["layers"]) == 2  # (8x6, 2x8)
    total_n = sum(row["n_weights"] for row in answer["layers"])
    assert answer["n_weights"] == total_n == 48 + 16
    for row in answer["layers"]:
        assert row["method"] == "int8"
        assert row["bits_per_weight"] == 8.0
        assert row["shape"] in ([8, 6], [2, 8])
        # the engine's own documented bound: rel < 0.02 (the 1/2^8
        # relative error scale its own test pins) on the seeded net
        assert row["relative_error"] < 0.02, (
            f"layer {row['layer']}: {row['relative_error']}"
        )
        assert row["rmse"] >= 0.0
    # the honest memory line over the float32 baseline
    assert answer["original_bits"] == 32 * total_n
    assert answer["quantized_bits"] == 8 * total_n
    assert answer["bits_per_weight"] == 8.0
    assert answer["memory_savings"] == pytest.approx(0.75)


def test_nf4_pack_carries_the_block_scale_bits(client):
    dataset = _upload(client, _fixture_csv(), "eff-nf4.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    answer = _efficiency_ok(client, model_id, mode="quantize_nf4")

    assert answer["method"] == "nf4"
    for row in answer["layers"]:
        # the block scale's own share rides the bits per weight
        # (4 + 8/block + the amortized offset) — on these tiny layers
        # one 64-block carries a whole 8-bit scale + the 32-bit offset,
        # so a 16-entry head reaches 4 + 40/16 = 6.5
        assert 4.0 < row["bits_per_weight"] < 7.0, (
            f"layer {row['layer']}: {row['bits_per_weight']}"
        )
        assert row["rmse"] < 0.12  # the engine's own N(0,1) bound
        assert 0.0 < row["relative_error"] < 1.0
    # the mean bits per weight is the count-weighted mean, not a lie
    mean = sum(
        r["bits_per_weight"] * r["n_weights"] for r in answer["layers"]
    ) / answer["n_weights"]
    assert answer["bits_per_weight"] == pytest.approx(mean, abs=1e-3)
    assert answer["memory_savings"] > 0.8  # ~5.25/32 on this tiny net


def test_prune_at_half_sparsity_returns_exactly_half_zeros(client):
    dataset = _upload(client, _fixture_csv(), "eff-prune.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    answer = _efficiency_ok(client, model_id, mode="prune", level=0.5)

    assert answer["efficiency_mode"] == "prune"
    assert answer["target_sparsity"] == 0.5
    assert answer["scope"] == "local"
    # the local scope's own ceil arithmetic on the even-width layers:
    # rows of width 6 keep ceil(0.5*6)=3 (3 zeroed), rows of width 8
    # keep ceil(0.5*8)=4 (4 zeroed) — exactly 50% overall
    for row in answer["layers"]:
        assert row["n_zeroed"] == row["n_weights"] // 2, (
            f"layer {row['layer']}: {row['n_zeroed']}/{row['n_weights']}"
        )
    assert answer["n_zeroed"] == answer["n_weights"] // 2
    assert answer["achieved_sparsity"] == pytest.approx(0.5)
    for row in answer["layers"]:
        assert row["n_kept"] == row["n_weights"] - row["n_zeroed"]
        assert row["sparsity"] == pytest.approx(0.5)


def test_the_metric_delta_scores_a_pruned_copy_never_the_model(client):
    """The prune lane's metric delta + the non-mutation pin."""
    dataset = _upload(client, _fixture_csv(), "eff-metric.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    answer = _efficiency_ok(client, model_id, mode="prune", level=0.5)

    metric = answer["metric"]
    assert metric["kind"] == "classification"
    assert isinstance(metric["before"], float)
    assert isinstance(metric["after"], float)
    assert metric["delta"] == pytest.approx(metric["after"] - metric["before"])
    assert 0.0 <= metric["before"] <= 1.0  # the accuracy domain
    assert 0.0 <= metric["after"] <= 1.0
    assert "never mutated" in metric["note"]

    # the non-mutation pin: re-run the int8 lane — the same RMSE
    # per layer proves the persisted weights stand untouched (the
    # pruned COPY was the only thing that scored)
    first = _efficiency_ok(client, model_id, mode="quantize_int8")
    _efficiency_ok(client, model_id, mode="prune", level=0.5)
    second = _efficiency_ok(client, model_id, mode="quantize_int8")
    assert first["layers"] == second["layers"], (
        "the persisted estimator's weights were mutated by the prune lane"
    )
    # and the model still scores identically on its own carve
    again = _efficiency_ok(client, model_id, mode="prune", level=0.5)
    assert again["metric"]["before"] == pytest.approx(metric["before"])


def test_the_global_scope_lane_uses_one_threshold_over_the_block(client):
    dataset = _upload(client, _fixture_csv(), "eff-global.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    answer = _efficiency_ok(
        client, model_id, mode="prune", level=0.5, scope="global",
    )
    assert answer["scope"] == "global"
    assert answer["achieved_sparsity"] == pytest.approx(0.5)
    # one threshold over the whole 64-entry block: 32 zeroed
    assert answer["n_zeroed"] == 32


def test_the_report_registers_as_a_downloadable_table_artifact(client):
    dataset = _upload(client, _fixture_csv(), "eff-artifact.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    answer = _efficiency_ok(
        client, model_id, mode="prune", level=0.5, artifact=True,
    )
    artifact = answer["artifact"]
    assert artifact is not None
    assert artifact["kind"] == "table"
    assert artifact["filename"] == "dl-efficiency-prune.csv"
    assert artifact["size_bytes"] > 0
    assert len(artifact["sha256"]) == 64
    assert artifact["meta"]["model_id"] == model_id
    assert artifact["meta"]["mode"] == "prune"

    # the download round-trip: the CSV parses with the layer rows
    r = client.get(artifact["download_url"], headers=auth_headers())
    assert r.status_code == 200, r.text
    reader = csv.DictReader(io.StringIO(r.content.decode("utf-8")))
    rows = list(reader)
    assert len(rows) == answer["n_layers"]
    assert {row["layer"] for row in rows} == {
        row["layer"] for row in answer["layers"]
    }


def test_the_honest_refusal_lanes(client):
    dataset = _upload(client, _fixture_csv(), "eff-refuse.csv")
    payload = _train_tabular(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    # an unknown model is a 404
    r = _efficiency(client, "mdl-does-not-exist")
    assert r.status_code == 404

    # a level on a quantize lane is a 422 (the knob rides prune only)
    r = _efficiency(client, model_id, mode="quantize_int8", level=0.5)
    assert r.status_code == 422
    assert "prune lane only" in r.json()["detail"]

    # a scope on a quantize lane is a 422
    r = _efficiency(
        client, model_id, mode="quantize_nf4", scope="global",
    )
    assert r.status_code == 422
    assert "scope knob rides the prune lane" in r.json()["detail"]

    # a prune without a level is a 422 (never a default guess)
    r = _efficiency(client, model_id, mode="prune", level=None)
    assert r.status_code == 422
    assert "needs a level" in r.json()["detail"]

    # an out-of-range level trips the pydantic bound
    r = _efficiency(client, model_id, mode="prune", level=1.5)
    assert r.status_code == 422

    # a non-tabular model is refused with the honest pointer
    ae_dataset = _upload(client, _fixture_csv(n=40), "eff-ae.csv")
    r = client.post(
        "/api/v1/dl/autoencoder-train",
        headers=auth_headers(),
        json={
            "dataset_id": ae_dataset["dataset_id"],
            "hidden_layers": [4],
            "latent_dim": 2,
            "epochs": 5,
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    ae_model_id = r.json()["model"]["model_id"]
    r = _efficiency(client, ae_model_id, mode="quantize_int8")
    assert r.status_code == 422
    assert "tabular DL families" in r.json()["detail"]

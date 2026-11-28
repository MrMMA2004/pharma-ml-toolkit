"""DL-4 (the roadmap's row 16) — ``POST /api/v1/dl/mpnn-train`` +
``POST /api/v1/dl/mpnn-predict``.

The D-MPNN molecular property predictor over the REAL numpy engine
(``pharmaml.ml.mpnn.DMPNN`` — chemprop's directed message passing,
ported file by file): a registered dataset's own SMILES column
re-read server-side, the honest rejected-rows accounting, the seeded
held-out carve the roadmap's own validation step names, and the
shipped model as the FULL-corpus refit.

The roadmap's own acceptance drives the file:

* a held-out RMSE on a seeded in-repo dataset within the
  PRE-REGISTERED tolerance: heavy-atom-count targets (std ~1.35)
  learned from structure alone — val RMSE < 0.75 (well under the
  naive mean-predictor's ~1.35) and val correlation > 0.8;
* the API/vitest lanes mirroring the GNN pair: the model registers
  with its own predict path over raw SMILES;
* the honest refusals: a non-regression dataset, an unknown SMILES
  column (the columns listed), a re-targeting attempt, a too-small
  parseable corpus, an unknown model (404), a non-molecular model.
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


def _upload(
    client, csv: bytes, name: str, target: str = "y", task: str | None = None
) -> dict:
    form = {"target": target, "mode": "sync"}
    if task is not None:
        # the honest user statement that the endpoint is continuous
        # (heavy-atom counts hold few distinct values — the inference
        # lane would call them classes; the mpnn lane is a regressor)
        form["task"] = task
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, csv, "text/csv")},
        data=form,
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


#: the seeded in-repo fixture: 40 small molecules whose target IS the
#: heavy-atom count — learnable from the molecular graph alone, the
#: honest MoleculeNet-scale validation's small-scale stand-in
_SMILES_ROWS = [
    "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC",
    "CCO", "CCCO", "CCN", "CC(=O)O", "c1ccccc1", "Cc1ccccc1", "CC(C)C",
    "CC(C)O", "OCCO", "CCCCN", "CCCCO", "CCOC", "CCSC", "CCCCl", "CCBr",
    "CCC", "OCC", "NCC", "CC=CC", "CCCC", "C1CCCCC1", "C1CCOC1",
    "C1CNCC1", "COCC", "CCOOC", "CNC", "CNC(C)C", "CC(C)Br", "CC(C)(C)O",
    "OCC(C)O", "CC(O)CO", "CC(C)N",
]


def _mpnn_fixture_csv() -> bytes:
    from pharmaml.chem.molgraph import parse_smiles

    targets = [float(len(parse_smiles(s).atoms)) for s in _SMILES_ROWS]
    frame = pd.DataFrame({"smiles": _SMILES_ROWS, "y": targets})
    return frame.to_csv(index=False).encode("utf-8")


def _train(client, dataset_id: str, **overrides) -> dict:
    body: dict = {
        "dataset_id": dataset_id,
        "smiles_column": "smiles",
        "hidden_size": 32,
        "depth": 3,
        "aggregation": "norm",
        "epochs": 200,
        "lr": 0.02,
        "val_split": 0.2,
        "seed": 0,
        "mode": "sync",
    }
    body.update(overrides)
    r = client.post("/api/v1/dl/mpnn-train", headers=auth_headers(), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_held_out_rmse_lands_within_the_pre_registered_tolerance(client):
    """The roadmap's own acceptance clause, pre-registered."""
    dataset = _upload(client, _mpnn_fixture_csv(), "mpnn-fixture.csv", task="regression")
    payload = _train(client, dataset["dataset_id"])

    model = payload["model"]
    assert model["engine"] == "numpy"
    assert model["molecular_model"] is True
    assert model["name"] == "dmpnn_molecular_regressor"
    assert model["task"] == "regression"
    assert model["predict_url"] == "/api/v1/dl/mpnn-predict"
    assert model["dataset_id"] == dataset["dataset_id"]
    assert model["n_train_rows"] == 40

    # the PRE-REGISTERED tolerance (the fixture's own std is ~1.35):
    # the held-out RMSE well under the naive mean-predictor's scale,
    # the held-out correlation high — learned from structure alone
    assert payload["val_rmse"] < 0.75, payload["val_rmse"]
    assert payload["val_correlation"] > 0.8, payload["val_correlation"]
    assert payload["val_n"] >= 2

    # the histories: both fits' tails are finite and the training loss
    # fell on the carve (the model actually learned, never a stub)
    assert len(payload["train_history_tail"]) == 5
    assert len(payload["refit_history_tail"]) == 5
    assert all(np.isfinite(payload["train_history_tail"]))
    assert payload["train_history_tail"][-1] <= payload["train_history_tail"][0]

    # the honest accounting: nothing rejected on the clean fixture
    assert payload["n_rows"] == 40
    assert payload["n_rejected"] == 0
    assert payload["n_parameters"] > 0
    assert "held out" in payload["held_out_note"]


def test_the_rejected_rows_ride_their_own_accounting(client):
    """A dirty column never silently drops a row."""
    bad_smiles = ["CCO", "not-a-smiles", "C(", None, "CCN",
                  "C1CC", "CCC"] + [
        f"{'C' * (i + 1)}O" for i in range(20)
    ]
    targets = [3.0] * len(bad_smiles)
    frame = pd.DataFrame(
        {
            "smiles": [
                ("" if s is None else s) for s in bad_smiles
            ],
            "y": targets,
        }
    )
    # the None cell rides as an empty string in the CSV round trip —
    # re-mark it honestly below through the route's own read of the
    # pandas NaN the upload lane produces
    dataset = _upload(
        client, frame.to_csv(index=False).encode("utf-8"), "mpnn-dirty.csv",
        task="regression",
    )
    payload = _train(client, dataset["dataset_id"])

    assert payload["n_rejected"] == 4  # 3 unparseable + the NaN cell
    assert payload["n_rows"] == len(bad_smiles) - 4
    rejected = payload["rejected"]
    assert len(rejected) == 4
    for row in rejected:
        assert "index" in row and "error" in row
        assert row["error"] != ""
    # the null-cell row is labeled as such (never Number(None) === 0)
    assert any("null" in row["error"] for row in rejected)
    # the parse failures carry the parser's own message
    assert any("parse" in row["error"] or "SMILES" in row["error"].upper()
               or "not" in row["error"] for row in rejected)
    # the lane still ships: the clean rows carry the model
    assert payload["model"]["n_train_rows"] == payload["n_rows"]


def test_the_predict_lane_over_raw_smiles(client):
    dataset = _upload(client, _mpnn_fixture_csv(), "mpnn-predict.csv", task="regression")
    payload = _train(client, dataset["dataset_id"])
    model_id = payload["model"]["model_id"]

    r = client.post(
        "/api/v1/dl/mpnn-predict",
        headers=auth_headers(),
        json={
            "model_id": model_id,
            "smiles": ["CCO", "not-a-smiles", "c1ccccc1", "CCCC"],
        },
    )
    assert r.status_code == 200, r.text
    answer = r.json()
    assert answer["model_id"] == model_id
    assert answer["n_valid"] == 3
    assert answer["n_rejected"] == 1
    # the predictions are in the ORIGINAL target units, aligned with
    # the served index (the rejected row is not silently skipped)
    rows = answer["predictions"]
    assert [row["index"] for row in rows] == [0, 2, 3]
    assert rows[0]["smiles"] == "CCO"
    # heavy-atom counts: CCO=3, benzene=6, CCCC=4 — the model's own
    # scale (the tolerance is the pre-registered one, generous for a
    # 4-row ask)
    assert abs(rows[0]["prediction"] - 3.0) < 1.5
    assert abs(rows[2]["prediction"] - 4.0) < 1.5
    assert answer["rejected"][0]["index"] == 1
    assert answer["rejected"][0]["error"] != ""


def test_the_job_mode_completes_with_the_model(client):
    dataset = _upload(client, _mpnn_fixture_csv(), "mpnn-job.csv", task="regression")
    r = client.post(
        "/api/v1/dl/mpnn-train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "smiles_column": "smiles",
            "hidden_size": 16,
            "epochs": 40,
            "lr": 0.02,
            "mode": "job",
        },
    )
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        state = client.get(
            f"/api/v1/jobs/{job['job_id']}", headers=auth_headers()
        ).json()
        if state["state"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        time.sleep(0.05)
    assert state["state"] == "COMPLETED", state
    result = client.get(
        f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
    ).json()["result"]
    assert result["model"]["molecular_model"] is True
    assert result["model"]["predict_url"] == "/api/v1/dl/mpnn-predict"
    assert isinstance(result["val_rmse"], float)


def test_the_honest_refusal_lanes(client):
    dataset = _upload(client, _mpnn_fixture_csv(), "mpnn-refuse.csv", task="regression")

    # an unknown dataset is a 404
    r = client.post(
        "/api/v1/dl/mpnn-train",
        headers=auth_headers(),
        json={"dataset_id": "ds-does-not-exist", "smiles_column": "smiles"},
    )
    assert r.status_code == 404

    # an unknown SMILES column lists the dataset's own columns
    r = client.post(
        "/api/v1/dl/mpnn-train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "smiles_column": "nope",
        },
    )
    assert r.status_code == 422
    assert "smiles" in r.json()["detail"] and "columns" in r.json()["detail"]

    # a re-targeting attempt is refused (the lane never re-targets)
    r = client.post(
        "/api/v1/dl/mpnn-train",
        headers=auth_headers(),
        json={
            "dataset_id": dataset["dataset_id"],
            "smiles_column": "smiles",
            "target_column": "smiles",
        },
    )
    assert r.status_code == 422
    assert "re-target" in r.json()["detail"]

    # a classification dataset is refused with the honest pointer
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 2))
    cls_frame = pd.DataFrame(X, columns=["a", "b"])
    cls_frame["y"] = (X[:, 0] > 0).astype(int)
    cls_dataset = _upload(
        client,
        cls_frame.to_csv(index=False).encode("utf-8"),
        "mpnn-cls.csv",
    )
    r = client.post(
        "/api/v1/dl/mpnn-train",
        headers=auth_headers(),
        json={
            "dataset_id": cls_dataset["dataset_id"],
            "smiles_column": "a",
        },
    )
    assert r.status_code == 422
    assert "regressor" in r.json()["detail"]

    # a corpus with too few parseable rows is a 422 (the carve needs
    # both sides)
    tiny = pd.DataFrame(
        {"smiles": ["CCO", "CCN", "C1CC"], "y": [3.0, 3.0, 4.0]}
    )
    tiny_dataset = _upload(
        client, tiny.to_csv(index=False).encode("utf-8"), "mpnn-tiny.csv",
        task="regression",
    )
    r = client.post(
        "/api/v1/dl/mpnn-train",
        headers=auth_headers(),
        json={
            "dataset_id": tiny_dataset["dataset_id"],
            "smiles_column": "smiles",
        },
    )
    assert r.status_code == 422
    assert "12" in r.json()["detail"]

    # the predict lane's own refusals
    r = client.post(
        "/api/v1/dl/mpnn-predict",
        headers=auth_headers(),
        json={"model_id": "mdl-does-not-exist", "smiles": ["CCO"]},
    )
    assert r.status_code == 404

    payload = _train(client, dataset["dataset_id"])
    # a non-molecular model is refused (the autoencoder's own pointer)
    ae_dataset = _upload(
        client,
        pd.DataFrame(
            {"x1": [0.1, 0.2, 0.3, 0.4, 0.5] * 12,
             "y": [0, 1, 0, 1, 0] * 12},
        ).to_csv(index=False).encode("utf-8"),
        "mpnn-ae.csv",
    )
    r = client.post(
        "/api/v1/dl/autoencoder-train",
        headers=auth_headers(),
        json={
            "dataset_id": ae_dataset["dataset_id"],
            "hidden_layers": [4],
            "latent_dim": 2,
            "epochs": 3,
            "mode": "sync",
        },
    )
    assert r.status_code == 200, r.text
    ae_model_id = r.json()["model"]["model_id"]
    r = client.post(
        "/api/v1/dl/mpnn-predict",
        headers=auth_headers(),
        json={"model_id": ae_model_id, "smiles": ["CCO"]},
    )
    assert r.status_code == 422
    assert "molecular D-MPNN" in r.json()["detail"]

    # an all-invalid SMILES list is a 422, never a fake prediction
    r = client.post(
        "/api/v1/dl/mpnn-predict",
        headers=auth_headers(),
        json={
            "model_id": payload["model"]["model_id"],
            "smiles": ["not-a-smiles", "also not one"],
        },
    )
    assert r.status_code == 422
    assert "at least one parseable" in r.json()["detail"]

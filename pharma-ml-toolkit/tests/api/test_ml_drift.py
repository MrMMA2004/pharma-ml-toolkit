"""ML-5 (the roadmap's row 12) — ``POST /api/v1/ml/drift``.

The route over the REAL engine (``pharmaml.ml.drift.detect_drift`` —
the same statistics the batch-training CLI treats as mandatory): a
reference frame (a session-trained model's own training dataset, or a
registered dataset) compared against a registered current dataset.

The roadmap's own fail-first acceptance drives the file:

* a shifted column flags (PSI >= 0.2, verdict ``shifted``, named in
  ``drifted`` worst-first) while an identical column does not
  (verdict ``stable``, absent from ``drifted``);
* the model reference lane resolves the model's own training dataset
  and restricts the comparison to its feature vocabulary;
* the reference is exactly one of ``model_id`` / ``dataset_id`` — the
  both/neither case is the route's own 422;
* unknown ids are 404s, never silent no-ops;
* a current frame sharing no numeric reference features is the honest
  ``n_features: 0`` answer, never a fabricated verdict;
* the engine's own exported threshold constants ride the payload (the
  GUI never re-encodes a threshold);
* the credit-risk middle band is reachable (a moderate column reads
  ``moderate`` — never forced into the shifted/stable binary).
"""

from __future__ import annotations

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


def _frame_csv(a: np.ndarray, b: np.ndarray) -> bytes:
    frame = pd.DataFrame({"a": a, "b": b})
    frame["y"] = 1.0 + a * 0.5 - b * 0.25
    return frame.to_csv(index=False).encode("utf-8")


def _reference_pair():
    """Seeded reference batch: a ~ N(0,1), b ~ N(5,2)."""
    rng = np.random.default_rng(21)
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(5.0, 2.0, 400)
    return a, b


def _drift(client, **body) -> dict:
    r = client.post(
        "/api/v1/ml/drift", headers=auth_headers(), json=body,
    )
    return r


def _drift_ok(client, **body) -> dict:
    r = _drift(client, **body)
    assert r.status_code == 200, r.text
    return r.json()


def test_shifted_column_flags_and_identical_does_not(client):
    """The roadmap's fail-first clause, verbatim."""
    a, b = _reference_pair()
    ref = _upload(client, _frame_csv(a, b), "ref.csv")
    rng = np.random.default_rng(22)
    # a is moved by three sigma; b is drawn from the SAME distribution
    # (seeded fresh — a same-law batch, not a copied one)
    moved_a = rng.normal(3.0, 1.0, 400)
    same_law_b = rng.normal(5.0, 2.0, 400)
    current = _upload(client, _frame_csv(moved_a, same_law_b), "current.csv")

    payload = _drift_ok(
        client, dataset_id=ref["dataset_id"], new_dataset_id=current["dataset_id"],
    )

    by_name = {row["feature"]: row for row in payload["features"]}
    assert by_name["a"]["verdict"] == "shifted", (
        "a three-sigma mean shift must flag as shifted"
    )
    assert by_name["a"]["psi"] >= payload["shift_psi"]
    assert by_name["b"]["verdict"] == "stable", (
        "a same-law column must not flag"
    )
    assert by_name["b"]["psi"] < payload["warn_psi"]
    drifted_names = [row["feature"] for row in payload["drifted"]]
    assert "a" in drifted_names and "b" not in drifted_names
    assert payload["n_drifted"] == 1
    assert payload["drift_share"] == pytest.approx(0.5)
    # the direction is stated: a climbed
    assert by_name["a"]["mean_direction"] == "up"
    # the drifted list is worst-first (its head carries the max PSI)
    assert payload["drifted"][0]["psi"] == max(
        row["psi"] for row in payload["features"]
    )
    # the engine's own exported constants ride the payload
    assert payload["warn_psi"] == pytest.approx(0.1)
    assert payload["shift_psi"] == pytest.approx(0.2)
    # KS is present on the frame lane (raw reference values available)
    assert by_name["a"]["ks_statistic"] is not None
    assert by_name["a"]["ks_pvalue"] is not None


def test_model_reference_lane_uses_the_training_dataset(client):
    """``model_id`` resolves the model's own training dataset."""
    a, b = _reference_pair()
    ref = _upload(client, _frame_csv(a, b), "ref.csv")
    rng = np.random.default_rng(23)
    moved_a = rng.normal(3.0, 1.0, 400)
    same_law_b = rng.normal(5.0, 2.0, 400)
    current = _upload(client, _frame_csv(moved_a, same_law_b), "current.csv")

    r = client.post(
        "/api/v1/ml/train",
        headers=auth_headers(),
        json={"dataset_id": ref["dataset_id"], "model": "random_forest",
               "task": "regression", "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    model = r.json()["model"]

    payload = _drift_ok(
        client, model_id=model["model_id"],
        new_dataset_id=current["dataset_id"],
    )

    assert payload["reference"]["kind"] == "model"
    assert payload["reference"]["model_id"] == model["model_id"]
    assert payload["reference"]["dataset_id"] == ref["dataset_id"]
    by_name = {row["feature"] for row in payload["features"]}
    assert by_name == {"a", "b"}, (
        "the comparison rides the model's own feature vocabulary"
    )
    moved = [row for row in payload["features"] if row["feature"] == "a"]
    assert moved and moved[0]["verdict"] == "shifted"
    stable = [row for row in payload["features"] if row["feature"] == "b"]
    assert stable and stable[0]["verdict"] == "stable"


def test_reference_is_exactly_one_source(client):
    """both/neither is the route's own 422, message verbatim."""
    r = _drift(client, model_id="m-1", dataset_id="d-1",
               new_dataset_id="d-2")
    assert r.status_code == 422
    assert "either model_id" in r.json()["detail"]
    r = _drift(client, new_dataset_id="d-2")
    assert r.status_code == 422
    assert "either model_id" in r.json()["detail"]


def test_unknown_ids_are_404s(client):
    r = _drift(client, model_id="no-such-model", new_dataset_id="d-2")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown model_id."

    a, b = _reference_pair()
    ref = _upload(client, _frame_csv(a, b), "ref.csv")
    r = _drift(client, dataset_id=ref["dataset_id"],
               new_dataset_id="no-such-current")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown new_dataset_id."

    r = _drift(client, dataset_id="no-such-ref", new_dataset_id="d-2")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown dataset_id."


def test_disjoint_columns_are_the_honest_zero(client):
    """no shared numeric features -> n_features 0, never an error."""
    a, b = _reference_pair()
    ref = _upload(client, _frame_csv(a, b), "ref.csv")
    rng = np.random.default_rng(24)
    other = pd.DataFrame(
        {"z1": rng.normal(size=200), "z2": rng.normal(size=200)}
    )
    other["y"] = other["z1"] * 0.5
    current = _upload(
        client, other.to_csv(index=False).encode("utf-8"), "other.csv",
    )

    payload = _drift_ok(
        client, dataset_id=ref["dataset_id"],
        new_dataset_id=current["dataset_id"],
    )
    assert payload["n_features"] == 0
    assert payload["n_drifted"] == 0
    assert payload["drift_share"] == 0.0
    assert payload["features"] == []
    assert payload["drifted"] == []
    assert "no numeric features" in payload["summary"]


def test_moderate_band_is_reachable(client):
    """the middle verdict is honest, not a binary collapse."""
    a, b = _reference_pair()
    ref = _upload(client, _frame_csv(a, b), "ref.csv")
    rng = np.random.default_rng(25)
    # a 0.35-sigma shift lands PSI ≈ 0.17 (measured against the
    # engine over this exact seed) — inside the 0.1..0.2 moderate
    # band, clear of both edges
    mild_a = rng.normal(0.35, 1.0, 800)
    same_law_b = rng.normal(5.0, 2.0, 800)
    mild = pd.DataFrame({"a": mild_a, "b": same_law_b})
    mild["y"] = 1.0 + mild_a * 0.5 - same_law_b * 0.25
    current = _upload(
        client, mild.to_csv(index=False).encode("utf-8"), "mild.csv",
    )

    payload = _drift_ok(
        client, dataset_id=ref["dataset_id"],
        new_dataset_id=current["dataset_id"],
    )
    by_name = {row["feature"]: row for row in payload["features"]}
    assert by_name["a"]["verdict"] == "moderate", (
        "a 0.35-sigma shift is the moderate band, never forced binary"
    )
    assert payload["warn_psi"] <= by_name["a"]["psi"] < payload["shift_psi"]
    assert by_name["b"]["verdict"] == "stable"
    assert payload["n_drifted"] == 0, (
        "moderate is not shifted — the drifted list stays empty"
    )

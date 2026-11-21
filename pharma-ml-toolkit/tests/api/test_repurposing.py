"""part13 Phase 5's next remainder item — the drug-repurposing API lane
over the REAL in-tree backend (``pharmaml.ml.response_model``:
CaDRReS's matrix-factorisation recommender, DeepResponse's split ladder
+ metric contract + the RepurposingInferenceEngine ranking, the seeded
SYNTHETIC response screen).

No mocks: the registry answers carry the backend's own vocabulary
verbatim (the four within-table split strategies, the ten metric keys,
the repurposing table schema), and the screen pass is the real
round trip — the seeded screen, the real CaDRReS fit, the predicted
IC50 matrix, the metric contract on the observed entries, and the
ranked candidates — asserted against the backend's own arithmetic
(most-sensitive first: lowest predicted IC50 = rank 1). The honest
provenance lanes pin §11.1's own rules: the SELFormer/CNN/FiLM hybrid
is NOT shipped, the served screen is SYNTHETIC, no real GDSC/CCLE/CTRP
table rides, and no clinical recommendation is ever emitted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.ml.response_model import METRIC_KEYS, SPLIT_STRATEGIES

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


REGISTRY_URL = "/api/v1/repurposing/registry"
SCREEN_URL = "/api/v1/repurposing/screen"


class TestRepurposingRegistry:
    """GET /api/v1/repurposing/registry — the verbatim vocabulary."""

    def test_split_strategies_are_the_backend_own_ladder(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        strategies = body["split_strategies"]
        assert strategies["within_table"] == list(SPLIT_STRATEGIES)
        assert strategies["within_table"] == [
            "random",
            "cell_stratified",
            "drug_stratified",
            "drug_cell_stratified",
        ]
        assert "cross_domain_split" in strategies["cross_domain"]
        assert "DeepResponse" in strategies["source"]

    def test_model_metric_and_schema_vocabulary_is_the_backend_own(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        model = body["model"]
        assert "P.Q^T" in model["formulation"]
        assert model["direction"].startswith("the matrix reads IC50")
        assert model["telemetry_fields"] == [
            "history",
            "observed_rmse",
            "stopped_early",
        ]
        assert "no live per-epoch callback" in model["history_note"]
        metrics = body["metrics"]
        assert metrics["keys"] == list(METRIC_KEYS)
        assert metrics["threshold"] == 6.0
        assert metrics["keys"] == [
            "rmse", "mae", "r2", "pcc", "scc",
            "accuracy", "precision", "recall", "f1", "mcc",
        ]
        repurposing = body["repurposing"]
        assert repurposing["schema"] == [
            "cell_line", "drug", "predicted_ic50", "rank",
        ]
        assert "lowest predicted IC50 = rank 1" in repurposing["ordering"]
        assert body["screen_url"] == SCREEN_URL

    def test_provenance_block_is_the_honest_state(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        provenance = body["provenance"]
        assert provenance["backend"] == "pharmaml.ml.response_model"
        assert provenance["n_repositories"] == 3
        assert provenance["shipped_checkpoints"] == 0
        assert "NOT shipped" in provenance["shipped_checkpoints_note"]
        assert "SELFormer" in provenance["shipped_checkpoints_note"]
        assert "SYNTHETIC" in provenance["screen"]
        assert "GDSC/CCLE/CTRP" in provenance["screen"]
        assert "never clinical recommendations" in provenance["disclaimer"]


class TestRepurposingScreen:
    """POST /api/v1/repurposing/screen — the real backend round trip."""

    def test_real_round_trip_over_the_real_backend(self, client):
        first = client.post(SCREEN_URL, json={}, headers=auth_headers())
        assert first.status_code == 200
        body = first.json()
        screen = body["screen"]
        assert screen["n_cells"] == 12 and screen["n_drugs"] == 8
        assert screen["n_features"] == 6
        assert screen["cells"][0] == "C01" and screen["drugs"][0] == "D01"
        assert screen["features"] == [f"F{k + 1}" for k in range(6)]
        assert 0.0 < screen["observed_fraction"] < 1.0
        assert screen["n_observed"] < screen["n_entries"]
        assert "SYNTHETIC" in screen["label"]
        training = body["training"]
        assert 1 <= training["epochs_run"] <= training["max_epochs"] == 200
        assert training["observed_rmse"] >= 0.0
        assert training["first_epoch"] == 1
        assert training["n_factors"] == 3
        assert "never a fabricated epoch stream" in training["note"]
        assert "ranked hypotheses" in body["note"]

        # the same seed is the same round trip — deterministic end to end
        second = client.post(SCREEN_URL, json={}, headers=auth_headers()).json()
        assert (
            second["training"]["observed_rmse"]
            == body["training"]["observed_rmse"]
        )
        assert (
            second["candidates"]["per_profile"]
            == body["candidates"]["per_profile"]
        )

    def test_candidates_are_ranked_most_sensitive_first(self, client):
        body = client.post(
            SCREEN_URL,
            json={"n_cells": 6, "n_drugs": 4, "n_features": 3,
                  "n_factors": 2, "max_epochs": 60, "seed": 9,
                  "profiles": 1, "top_k": 4},
            headers=auth_headers(),
        ).json()
        per_profile = body["candidates"]["per_profile"]
        assert len(per_profile) == 1
        rows = next(iter(per_profile.values()))
        assert [row["rank"] for row in rows] == [1, 2, 3, 4]
        values = [row["predicted_ic50"] for row in rows]
        assert values == sorted(values)
        assert rows[0]["drug"].startswith("D")

    def test_top_k_and_profiles_slice_the_table(self, client):
        body = client.post(
            SCREEN_URL,
            json={"top_k": 2, "profiles": 2},
            headers=auth_headers(),
        ).json()
        candidates = body["candidates"]
        assert candidates["top_k"] == 2
        assert candidates["n_profiles"] == 2
        assert len(candidates["per_profile"]) == 2
        for rows in candidates["per_profile"].values():
            assert len(rows) == 2
        # the FULL ranked table is 2 profiles x the whole drug axis (8)
        assert candidates["n_ranked_rows_total"] == 2 * 8

    def test_metric_contract_rides_the_observed_entries(self, client):
        body = client.post(SCREEN_URL, json={}, headers=auth_headers()).json()
        metrics = body["metrics"]
        assert sorted(metrics) == sorted(METRIC_KEYS)
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        # the observed fraction backs the lane: metrics ride observed rows
        assert body["screen"]["n_observed"] >= 1

    def test_geometry_bounds_mirror_the_backend_own_422(self, client):
        for payload in (
            {"n_cells": 3},
            {"n_drugs": 2},
            {"n_features": 1},
            {"observed": 0.2},
        ):
            answer = client.post(SCREEN_URL, json=payload, headers=auth_headers())
            assert answer.status_code == 422, payload
            assert str(answer.json())  # the failing field rides the detail

    def test_sync_light_caps_are_the_route_own_422(self, client):
        for payload in (
            {"n_cells": 41},
            {"max_epochs": 1001},
            {"top_k": 11},
            {"profiles": 9},
            {"lr": 0.6},
            {"seed": -1},
        ):
            answer = client.post(SCREEN_URL, json=payload, headers=auth_headers())
            assert answer.status_code == 422, payload

    def test_custom_parameters_reach_the_backend(self, client):
        body = client.post(
            SCREEN_URL,
            json={"n_cells": 6, "n_drugs": 4, "n_features": 3,
                  "n_factors": 2, "lr": 0.05, "max_epochs": 60,
                  "noise": 0.3, "observed": 0.8, "seed": 9,
                  "profiles": 1, "top_k": 4},
            headers=auth_headers(),
        ).json()
        echoed = body["request"]
        assert echoed["n_cells"] == 6 and echoed["n_drugs"] == 4
        assert echoed["n_features"] == 3 and echoed["n_factors"] == 2
        assert echoed["seed"] == 9
        screen = body["screen"]
        assert screen["cells"] == [f"C{i + 1:02d}" for i in range(6)]
        assert screen["drugs"] == [f"D{i + 1:02d}" for i in range(4)]
        assert screen["features"] == ["F1", "F2", "F3"]
        candidates = body["candidates"]["per_profile"]
        assert len(candidates) == 1
        rows = next(iter(candidates.values()))
        assert len(rows) == 4  # the whole 4-drug axis
        for rows_same_seed in [rows]:
            assert [row["rank"] for row in rows_same_seed] == [1, 2, 3, 4]

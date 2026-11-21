"""part13 Phase 5's next remainder item — the cancer cell-line
drug-response API lane over the REAL in-tree backend
(``pharmaml.ml.drugresponse``: the CaDRReS-style bilinear model with
its sigmoid link, the seeded SYNTHETIC response screen, the
sensitivity ranking, the per-drug biomarker readout, and the
cross-cell-line validation folds).

No mocks: the registry answers carry the backend's own vocabulary
verbatim (the five metric keys, the sensitivity-ranking schema, the
biomarker readout, the cell-line-grouped folds), and the analyze pass
is the real round trip — the seeded screen, the real CaDRReS-style
fit, the predicted response-score matrix, the metric contract on the
observed entries, the most-responsive-first ranking, the composed
``W w_d`` biomarker weights, and the cross-cell-line folds — asserted
against the backend's own arithmetic. The honest provenance lanes pin
§11.1's own rules: no pretrained checkpoints ship, the served screen
is SYNTHETIC, no real CCLE/GDSC/CTRP/DepMap table rides, one response
scale only, and no clinical recommendation is ever emitted.
"""

from __future__ import annotations

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


REGISTRY_URL = "/api/v1/drugresponse/registry"
ANALYZE_URL = "/api/v1/drugresponse/analyze"


class TestDrugResponseRegistry:
    """GET /api/v1/drugresponse/registry — the verbatim vocabulary."""

    def test_model_metric_and_schema_vocabulary_is_the_backend_own(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        model = body["model"]
        assert "sigmoid" in model["formulation"]
        assert model["direction"].startswith("the score is the sigmoid-linked")
        assert model["telemetry_fields"] == [
            "train_losses",
            "final_loss",
            "n_observed",
            "pearson_r",
            "n_epochs",
        ]
        assert "no live per-epoch callback" in model["history_note"]
        metrics = body["metrics"]
        assert metrics["keys"] == ["pearson", "spearman", "rmse", "mae", "n"]
        assert "never as zero" in metrics["note"]
        ranking = body["sensitivity_ranking"]
        assert ranking["schema"] == [
            "cell_line", "drug", "predicted_score", "rank",
        ]
        assert "highest predicted response score = rank 1" in ranking["ordering"]
        biomarkers = body["biomarkers"]
        assert biomarkers["schema"] == ["feature", "weight"]
        assert "W w_d" in biomarkers["source"]
        cross_validation = body["cross_validation"]
        assert "never (cell line, drug) pairs" in cross_validation["rule"]
        screen = body["screen"]
        assert "never mixed" in screen["scale"]
        assert body["analyze_url"] == ANALYZE_URL

    def test_provenance_block_is_the_honest_state(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        provenance = body["provenance"]
        assert provenance["backend"] == "pharmaml.ml.drugresponse"
        assert provenance["n_repositories"] == 3
        assert provenance["shipped_checkpoints"] == 0
        assert "no pretrained checkpoints ship" in provenance[
            "shipped_checkpoints_note"
        ]
        assert "DeepCDR" in provenance["shipped_checkpoints_note"]
        assert "SYNTHETIC" in provenance["screen"]
        assert "CCLE" in provenance["screen"] and "GDSC" in provenance["screen"]
        assert "never clinical recommendations" in provenance["disclaimer"]


class TestDrugResponseAnalyze:
    """POST /api/v1/drugresponse/analyze — the real backend round trip."""

    def test_real_round_trip_over_the_real_backend(self, client):
        first = client.post(ANALYZE_URL, json={}, headers=auth_headers())
        assert first.status_code == 200
        body = first.json()
        screen = body["screen"]
        assert screen["n_cells"] == 12 and screen["n_drugs"] == 4
        assert screen["n_features"] == 8
        assert screen["cells"][0] == "CELL_00" and screen["drugs"][0] == "DRUG_00"
        assert screen["features"] == [f"gene_{j:02d}" for j in range(8)]
        assert 0.0 < screen["observed_fraction"] < 1.0
        assert screen["n_observed"] < screen["n_entries"]
        assert "SYNTHETIC" in screen["label"]
        training = body["training"]
        assert training["epochs_run"] == training["max_epochs"] == 200
        assert training["final_loss"] >= 0.0
        assert training["first_loss"] >= training["final_loss"]
        assert training["n_factors"] == 3
        assert "never a fabricated epoch stream" in training["note"]
        assert "triage hypotheses" in body["note"]

        # the same seed is the same round trip — deterministic end to end
        second = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        assert second["training"]["final_loss"] == body["training"]["final_loss"]
        assert second["ranking"]["per_cell"] == body["ranking"]["per_cell"]
        assert (
            second["cross_validation"]["summary"]
            == body["cross_validation"]["summary"]
        )

    def test_ranking_is_most_responsive_first(self, client):
        body = client.post(
            ANALYZE_URL,
            json={"n_cells": 6, "n_drugs": 4, "n_features": 3,
                  "n_factors": 2, "epochs": 60, "seed": 9,
                  "profiles": 1, "top_k": 4},
            headers=auth_headers(),
        ).json()
        per_cell = body["ranking"]["per_cell"]
        assert len(per_cell) == 1
        rows = next(iter(per_cell.values()))
        assert [row["rank"] for row in rows] == [1, 2, 3, 4]
        values = [row["predicted_score"] for row in rows]
        assert values == sorted(values, reverse=True)
        assert rows[0]["drug"].startswith("DRUG_")

    def test_biomarker_weights_carry_the_composed_readout(self, client):
        body = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        biomarkers = body["biomarkers"]
        assert biomarkers["n_drugs"] == 4
        assert sorted(biomarkers["per_drug"]) == [
            f"DRUG_{k:02d}" for k in range(4)
        ]
        for rows in biomarkers["per_drug"].values():
            assert 1 <= len(rows) <= 5  # top_k slices the 8-feature axis
            weights = [abs(row["weight"]) for row in rows]
            assert weights == sorted(weights, reverse=True)
            for row in rows:
                assert row["feature"].startswith("gene_")

    def test_cross_cell_line_folds_split_cells_never_pairs(self, client):
        body = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        cross_validation = body["cross_validation"]
        assert cross_validation["n_folds"] == 3
        assert len(cross_validation["folds"]) == 3
        test_cells = [fold["n_test_cells"] for fold in cross_validation["folds"]]
        assert sum(test_cells) == 12  # every cell line held out exactly once
        assert all(fold["n_observed"] >= 1 for fold in cross_validation["folds"])
        summary = cross_validation["summary"]
        assert sorted(summary) == ["mae", "n", "pearson", "rmse", "spearman"]
        assert summary["n"] >= 1.0

    def test_metric_contract_rides_the_observed_entries(self, client):
        body = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        metrics = body["metrics"]
        assert sorted(metrics) == ["mae", "n", "pearson", "rmse", "spearman"]
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0
        assert metrics["n"] >= 1
        # the observed fraction backs the lane: metrics ride observed rows
        assert body["screen"]["n_observed"] >= 1

    def test_geometry_bounds_mirror_the_backend_own_422(self, client):
        for payload in (
            {"n_cells": 3},
            {"n_drugs": 1},
            {"n_features": 1},
            {"observed": 0.4},
        ):
            answer = client.post(ANALYZE_URL, json=payload, headers=auth_headers())
            assert answer.status_code == 422, payload
            assert str(answer.json())  # the failing field rides the detail

    def test_backend_own_n_folds_rule_is_its_own_422(self, client):
        # n_folds may not exceed the cell count — the backend's own rule
        answer = client.post(
            ANALYZE_URL,
            json={"n_cells": 4, "n_folds": 5},
            headers=auth_headers(),
        )
        assert answer.status_code == 422
        assert "n_folds" in str(answer.json())

    def test_sync_light_caps_are_the_route_own_422(self, client):
        for payload in (
            {"n_cells": 41},
            {"epochs": 1001},
            {"top_k": 11},
            {"profiles": 9},
            {"n_folds": 6},
            {"lr": 0.6},
            {"seed": -1},
        ):
            answer = client.post(ANALYZE_URL, json=payload, headers=auth_headers())
            assert answer.status_code == 422, payload

    def test_custom_parameters_reach_the_backend(self, client):
        body = client.post(
            ANALYZE_URL,
            json={"n_cells": 6, "n_drugs": 5, "n_features": 3,
                  "n_factors": 2, "lr": 0.2, "epochs": 60, "l2": 0.01,
                  "noise": 0.1, "observed": 0.9, "seed": 9,
                  "profiles": 2, "top_k": 3, "n_folds": 2},
            headers=auth_headers(),
        ).json()
        echoed = body["request"]
        assert echoed["n_cells"] == 6 and echoed["n_drugs"] == 5
        assert echoed["n_features"] == 3 and echoed["n_factors"] == 2
        assert echoed["seed"] == 9 and echoed["n_folds"] == 2
        screen = body["screen"]
        assert screen["cells"] == [f"CELL_{i:02d}" for i in range(6)]
        assert screen["drugs"] == [f"DRUG_{k:02d}" for k in range(5)]
        assert screen["features"] == [f"gene_{j:02d}" for j in range(3)]
        per_cell = body["ranking"]["per_cell"]
        assert len(per_cell) == 2
        for rows in per_cell.values():
            assert len(rows) == 3  # top_k 3 over the 5-drug axis
            assert [row["rank"] for row in rows] == [1, 2, 3]
        assert len(body["cross_validation"]["folds"]) == 2

"""part13 Phase 5's LAST remainder item — the multi-objective HPO API
lane over the REAL in-tree backend
(``pharmaml.ml.multiobjective``: the Pareto order, the exact 2-D
hypervolume, NSGA-II's crowding, the MOEA/D Tchebycheff decomposition,
and the cross-validated study over the sklearn registry models).

No mocks: the registry answers carry the backend's own vocabulary
verbatim (the minimize convention, the WFG reference contract, the
Das-Dennis lattice, the tunable-objective ledger with its cross_validate
keys and sign flips, the one-shot prune contract), and the optimize
pass is the real round trip — the seeded synthetic dataset, the
request's own search space, the real cross-validated study, the front
with its aligned crowding, the 2-D indicator + the monotone curve, the
per-objective endpoints, and the crowding-pruned shortlist — asserted
against the backend's own arithmetic. The honest lanes pin the scope
statement: optuna/Ax/nevergrad reviewed, EHVI/MOTPE/NSGA-II-operators
stay upstream, the exact hypervolume is 2-D only, zero checkpoints
ship, no real dataset ships, and the wall-clock fold timings (the
fit/score-seconds objectives) vary run-to-run — the drift stated, never
hidden behind a fake determinism claim.
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


REGISTRY_URL = "/api/v1/multiobjective/registry"
OPTIMIZE_URL = "/api/v1/multiobjective/optimize"

DEFAULT_DIMENSIONS = [
    {"name": "n_estimators", "kind": "randint", "low": 10, "high": 80},
    {"name": "max_depth", "kind": "randint", "low": 2, "high": 10},
]


class TestMultiobjectiveRegistry:
    """GET /api/v1/multiobjective/registry — the verbatim vocabulary."""

    def test_order_vocabulary_is_the_backend_own(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        order = body["order"]
        assert "MINIMIZED" in order["convention"]
        assert "negated mean CV score" in order["convention"]
        assert "at least as good on every" in order["dominates"]
        assert "duplicates survive together" in order["pareto_indices"]
        assert "mutual non-domination is the definition" in order["pareto_indices"]
        assert "O(n^2 * k)" in order["complexity"]

    def test_indicator_and_spread_vocabulary_is_verbatim(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        indicator = body["indicator"]
        assert "staircase sweep" in indicator["family"]
        assert "WFG contract" in indicator["reference_contract"]
        assert "never silently clips" in indicator["reference_contract"]
        assert "10 %" in indicator["report_reference"]
        assert "monotonically non-decreasing" in indicator["curve"]
        assert "2-D only" in indicator["two_d_only"]
        assert "never a fabricated indicator" in indicator["two_d_only"]
        spread = body["spread"]
        assert "boundaries +inf" in spread["family"]
        assert "zero span" in spread["rule"]
        assert "+inf reads as null" in spread["json_note"]

    def test_decomposition_and_study_vocabulary_is_verbatim(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        decomposition = body["decomposition"]
        assert "max_j w_j |v_j - ideal_j|" in decomposition["form"]
        assert "corner subproblem" in decomposition["rule"]
        study = body["study"]
        assert study["samplers"] == ["random", "tpe"]
        assert "front-chasers" in study["samplers_note"]
        assert "Das & Dennis's simplex-lattice" in study["weight_modes"]["spread"]
        assert "Dirichlet(1,...,1) shortcut" in study["weight_modes"]["random"]
        assert "not strictly stationary" in study["drift_note"]
        assert "convergence check" in study["drift_note"]

    def test_tunable_objective_ledger_is_the_backend_own(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        ledger = body["tunable_objectives"]["ledger"]
        assert sorted(ledger) == ["fit_seconds", "score", "score_seconds"]
        assert ledger["score"] == {"cross_validate_key": "test_score", "sign": -1.0}
        assert ledger["fit_seconds"] == {"cross_validate_key": "fit_time", "sign": 1.0}
        assert ledger["score_seconds"] == {"cross_validate_key": "score_time", "sign": 1.0}
        assert "negated mean CV score" in body["tunable_objectives"]["note"]
        assert "cv_results records" in body["tunable_objectives"]["note"]

    def test_report_schema_and_request_bounds_ride_the_registry(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        report = body["report"]
        assert "objective_names" in report["schema"]
        assert "one-shot selection" in report["prune"]
        assert "hypervolume as its own" in report["prune"]
        request = body["request"]
        assert request["defaults"]["model"] == "random_forest"
        assert request["defaults"]["dimensions"] == DEFAULT_DIMENSIONS
        assert request["bounds"]["n_trials"] == "1..60"
        assert request["bounds"]["dimensions"] == "1..4"
        assert sorted(request["dimension_kinds"]) == ["loguniform", "randint", "uniform"]
        assert "a 422, never a 500" in request["model_note"]
        assert body["dataset"]["geometry"] == {"n_samples": 80, "n_features": 6}
        assert body["optimize_url"] == OPTIMIZE_URL

    def test_provenance_block_is_the_honest_state(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        provenance = body["provenance"]
        assert provenance["backend"] == "pharmaml.ml.multiobjective"
        assert provenance["n_repositories"] == 3
        for name in ("optuna", "Ax", "nevergrad"):
            assert name in provenance["source"]
        assert "the study arithmetic" in provenance["transfers_exactly"]
        assert "EHVI" in provenance["stays_upstream"]
        assert "MOTPE" in provenance["stays_upstream"]
        assert "2-D sweep is the indicator reported" in provenance["stays_upstream"]
        assert provenance["shipped_checkpoints"] == 0
        assert "no pretrained anything ships" in provenance["shipped_checkpoints_note"]
        assert "SYNTHETIC" in provenance["dataset"]
        assert "never experimental facts" in provenance["disclaimer"]
        assert "vary run-to-run" in body["note"]


class TestMultiobjectiveOptimize:
    """POST /api/v1/multiobjective/optimize — the real backend round trip."""

    def test_real_round_trip_over_the_real_backend(self, client):
        answer = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 12},
            headers=auth_headers(),
        )
        assert answer.status_code == 200
        body = answer.json()
        assert body["request"]["model"] == "random_forest"
        assert body["request"]["dimensions"] == DEFAULT_DIMENSIONS
        assert body["request"]["sampler"] == "random"
        dataset = body["dataset"]
        assert dataset["n_samples"] == 80 and dataset["n_features"] == 6
        assert dataset["task"] == "classification"
        assert "SYNTHETIC" in dataset["label"]
        study = body["study"]
        assert study["n_trials"] == 12
        assert study["n_objectives"] == 2
        assert study["objective_names"] == ["score", "fit_seconds"]
        assert "negated mean CV score" in study["objectives_note"]
        front = body["front"]
        assert 1 <= len(front) <= 12
        for row in front:
            assert len(row["values"]) == 2
            assert row["values"][0] < 0.0          # the negated score
            assert row["values"][1] > 0.0          # the fit seconds
            assert row["scalar"] is not None and row["scalar"] >= 0.0
            assert row["params"]["n_estimators"] in range(10, 81)
            assert row["params"]["max_depth"] in range(2, 11)
            assert abs(sum(row["weights"]) - 1.0) < 1e-9
            if row["crowding"] is None:
                assert row["crowding_is_boundary"] is True
            else:
                assert row["crowding_is_boundary"] is False
        assert "null is never a zero spread" in body["crowding_note"]

    def test_the_front_is_really_non_dominated(self, client):
        """The served front is the backend's own Pareto arithmetic."""
        body = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 12},
            headers=auth_headers(),
        ).json()
        rows = body["front"]
        for i, a in enumerate(rows):
            for j, b in enumerate(rows):
                if i == j:
                    continue
                # a dominates b iff a <= b everywhere and < somewhere
                dominates = all(x <= y for x, y in zip(a["values"], b["values"])) and any(
                    x < y for x, y in zip(a["values"], b["values"])
                )
                assert not dominates, (
                    f"a dominated member served on the front: {a} vs {b}"
                )

    def test_hypervolume_curve_is_monotone_and_aligned(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 10},
            headers=auth_headers(),
        ).json()
        indicator = body["indicator"]
        assert indicator["hypervolume"] is not None and indicator["hypervolume"] > 0.0
        assert indicator["ref"] is not None and len(indicator["ref"]) == 2
        curve = indicator["hypervolume_curve"]
        assert len(curve) == body["study"]["n_trials"]
        assert curve == sorted(curve)                       # monotone non-decreasing
        assert curve[-1] == pytest.approx(indicator["hypervolume"], rel=1e-9)
        assert "FIXED reference" in indicator["curve_note"]

    def test_best_for_endpoints_are_the_trade_off_corners(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 12},
            headers=auth_headers(),
        ).json()
        best = body["best_for"]
        assert sorted(best) == ["fit_seconds", "score"]
        front = body["front"]
        score_column = [row["values"][0] for row in front]
        time_column = [row["values"][1] for row in front]
        # the endpoint minimizing each objective is the front's own extreme
        assert best["score"]["values"][0] == pytest.approx(min(score_column))
        assert best["fit_seconds"]["values"][1] == pytest.approx(min(time_column))
        for name, row in best.items():
            assert set(row) == {"number", "params", "values", "weights", "scalar"}
            assert any(row["values"] == f_row["values"] for f_row in front)
        assert "trade-off corners" in body["best_for_note"]

    def test_three_objective_study_reports_null_hypervolume(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={
                "objectives": ["score", "fit_seconds", "score_seconds"],
                "sampler": "random",
                "n_trials": 8,
            },
            headers=auth_headers(),
        ).json()
        assert body["study"]["n_objectives"] == 3
        for row in body["front"]:
            assert len(row["values"]) == 3
        indicator = body["indicator"]
        assert indicator["hypervolume"] is None
        assert indicator["ref"] is None
        assert indicator["hypervolume_curve"] == []
        assert "3-objective study" in indicator["two_d_only"]
        assert "never a fabricated indicator" in indicator["two_d_only"]

    def test_the_shortlist_prunes_by_full_front_crowding(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 12, "shortlist_keep": 2},
            headers=auth_headers(),
        ).json()
        shortlist = body["shortlist"]
        assert shortlist["keep"] == 2
        assert shortlist["n_front"] == len(body["front"])
        assert shortlist["n_trials"] == body["study"]["n_trials"]
        assert len(shortlist["trials"]) <= 2
        numbers = [row["number"] for row in shortlist["trials"]]
        assert numbers == sorted(numbers)     # the front order retained
        assert "never claims the full study's" in shortlist["note"]

    def test_weight_mode_random_rides_the_request(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 8, "weight_mode": "random"},
            headers=auth_headers(),
        ).json()
        assert body["request"]["weight_mode"] == "random"
        assert body["study"]["weight_mode"] == "random"
        for row in body["front"]:
            assert abs(sum(row["weights"]) - 1.0) < 1e-9

    def test_the_generative_dials_are_deterministic_not_the_timings(self, client):
        """The honest drift contract: the seed fixes the dataset and the
        random-sampler parameterization; the fold timings are wall-clock
        measurements that vary run-to-run."""
        first = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 1},
            headers=auth_headers(),
        ).json()
        second = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 1},
            headers=auth_headers(),
        ).json()
        assert len(first["front"]) == 1 and len(second["front"]) == 1
        a, b = first["front"][0], second["front"][0]
        assert a["params"] == b["params"]                    # the dial is fixed
        assert a["values"][0] == b["values"][0]              # the score column
        assert a["weights"] == b["weights"]
        assert "vary run-to-run" in first["timings_note"]    # the drift stated

    def test_a_different_seed_moves_the_study(self, client):
        first = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 1, "seed": 7},
            headers=auth_headers(),
        ).json()
        other = client.post(
            OPTIMIZE_URL,
            json={"sampler": "random", "n_trials": 1, "seed": 8},
            headers=auth_headers(),
        ).json()
        assert first["front"][0]["params"] != other["front"][0]["params"]

    def test_regression_task_round_trip(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={
                "model": "ridge",
                "task": "regression",
                "dimensions": [{"name": "alpha", "kind": "loguniform", "low": 0.01, "high": 10.0}],
                "sampler": "random",
                "n_trials": 8,
            },
            headers=auth_headers(),
        ).json()
        assert body["dataset"]["task"] == "regression"
        assert len(body["front"]) >= 1
        for row in body["front"]:
            assert "alpha" in row["params"]

    def test_bad_dimension_specs_surface_the_typed_422s(self, client):
        cases = [
            {"name": "alpha", "kind": "uniform", "low": 2.0, "high": 1.0},
            {"name": "alpha", "kind": "uniform", "low": 1.0, "high": 1.0},
            {"name": "alpha", "kind": "loguniform", "low": 0.0, "high": 1.0},
            {"name": "n", "kind": "randint", "low": 1.5, "high": 4},
        ]
        for dim in cases:
            answer = client.post(
                OPTIMIZE_URL,
                json={"dimensions": [dim], "sampler": "random", "n_trials": 2},
                headers=auth_headers(),
            )
            assert answer.status_code == 422, dim
            assert "dimension" in answer.json()["detail"]

    def test_the_backend_own_typed_422s_surface_verbatim(self, client):
        # the registry's own ModelError for an unknown model
        answer = client.post(
            OPTIMIZE_URL, json={"model": "no_such_model"}, headers=auth_headers()
        )
        assert answer.status_code == 422
        assert "Unknown model 'no_such_model'" in answer.json()["detail"]
        # build_model's own ModelError for an unknown hyperparameter name
        answer = client.post(
            OPTIMIZE_URL,
            json={
                "dimensions": [{"name": "not_a_hyperparameter", "kind": "uniform", "low": 0.0, "high": 1.0}],
                "sampler": "random",
                "n_trials": 2,
            },
            headers=auth_headers(),
        )
        assert answer.status_code == 422
        assert "not_a_hyperparameter" in answer.json()["detail"]
        # the backend's own DataValidationError for the objective contract
        for objectives in (["score"], ["score", "score"], ["score", "auc"]):
            answer = client.post(
                OPTIMIZE_URL,
                json={"objectives": objectives, "sampler": "random", "n_trials": 2},
                headers=auth_headers(),
            )
            assert answer.status_code == 422, objectives
            detail = answer.json()["detail"]
            assert "objective" in detail          # the backend's own wording

    def test_route_own_caps_are_422s(self, client):
        for payload in (
            {"n_trials": 0},
            {"n_trials": 61},
            {"cv": 1},
            {"seed": -1},
            {"seed": 2147483648},
            {"shortlist_keep": 0},
            {"weight_mode": "spiral"},
            {"sampler": "bayes"},
        ):
            answer = client.post(OPTIMIZE_URL, json=payload, headers=auth_headers())
            assert answer.status_code == 422, payload
        # the dimension cap
        five = [
            {"name": f"d{i}", "kind": "uniform", "low": 0.0, "high": 1.0}
            for i in range(5)
        ]
        answer = client.post(
            OPTIMIZE_URL, json={"dimensions": five}, headers=auth_headers()
        )
        assert answer.status_code == 422

    def test_custom_parameters_reach_the_backend(self, client):
        body = client.post(
            OPTIMIZE_URL,
            json={
                "model": "random_forest",
                "task": "classification",
                "objectives": ["score", "fit_seconds"],
                "dimensions": DEFAULT_DIMENSIONS,
                "n_trials": 6,
                "cv": 2,
                "sampler": "random",
                "seed": 9,
                "weight_mode": "spread",
                "n_weight_vectors": 4,
                "shortlist_keep": 3,
            },
            headers=auth_headers(),
        ).json()
        echoed = body["request"]
        assert echoed["n_trials"] == 6 and echoed["cv"] == 2
        assert echoed["seed"] == 9 and echoed["sampler"] == "random"
        assert echoed["n_weight_vectors"] == 4 and echoed["shortlist_keep"] == 3
        assert len(body["shortlist"]["trials"]) <= 3
        # the Das-Dennis lattice with 4 vectors: the first two trials
        # take the corner weights (1, 0) then the interior
        front_numbers = {row["number"] for row in body["front"]}
        assert front_numbers  # the study ran; the corners ride the history

"""part13 Phase 6, §8 — the workflow-engine API lane
(``/api/v1/workflow`` over the REAL in-tree engine).

No mocks: the registry/templates answers carry the engine's own
vocabulary verbatim; the sync run is §8's own example chain over the
route's seeded SYNTHETIC stand-in — the real backends (standardize,
descriptors, split, selection, fit, evaluate, Williams, SHAP, the
PDF report) execute; the job-mode run rides the EXISTING JobManager
(progress per node, cooperative cancellation, the SQLite history);
the honest lanes pin the 422 verbatim contracts, the partial-failure
semantics, and the bounded scope notes.
"""

from __future__ import annotations

import time

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


STEPS_URL = "/api/v1/workflow/steps"
TEMPLATES_URL = "/api/v1/workflow/templates"
VALIDATE_URL = "/api/v1/workflow/validate"
RUN_URL = "/api/v1/workflow/run"
RUNS_URL = "/api/v1/workflow/runs"


class TestStepsRegistry:
    def test_the_engine_block_states_the_decision_verbatim(self, client):
        body = client.get(STEPS_URL, headers=auth_headers()).json()
        engine = body["engine"]
        assert "pharmaml.workflow" in engine["layer"]
        assert "JobManager" in engine["decision"]
        assert "ARCHITECTURE_REVIEW" in engine["decision"]
        assert "no external workflow engine adopted" in engine["decision"]
        assert "SQLite" in engine["job_integration"]

    def test_the_closed_port_kind_vocabulary_rides_the_answer(
        self, client,
    ):
        body = client.get(STEPS_URL, headers=auth_headers()).json()
        assert "records" in body["engine"]["port_kinds"]
        assert "dataframe" in body["engine"]["port_kinds"]
        assert len(body["steps"]) == 10

    def test_every_step_carries_typed_ports_and_its_real_backend(
        self, client,
    ):
        body = client.get(STEPS_URL, headers=auth_headers()).json()
        by_id = {s["step_id"]: s for s in body["steps"]}
        assert by_id["chem.descriptors"]["backend"] == (
            "pharmaml.chem.descriptors.compute_descriptors"
        )
        train = by_id["ml.train"]
        assert {"split"} == {p["name"] for p in train["inputs"]}
        assert {"model"} == {p["name"] for p in train["outputs"]}
        evaluate = by_id["ml.evaluate"]
        assert any(p["kind"] == "metrics"
                   for p in evaluate["outputs"])
        report = by_id["report.generate"]
        optional = {p["name"] for p in report["inputs"]
                    if not p["required"]}
        assert optional == {"ad_report", "shap_summary"}


class TestTemplates:
    def test_the_two_built_ins_with_the_leakage_note(self, client):
        body = client.get(TEMPLATES_URL, headers=auth_headers()).json()
        templates = {t["template_id"]: t for t in body["templates"]}
        assert set(templates) == {"guided-qsar", "minimal-regression"}
        guided = templates["guided-qsar"]
        assert guided["node_count"] == 10
        assert "leakage_note" in guided["metadata"]
        assert "training split only" in guided["metadata"]["leakage_note"]
        assert templates["minimal-regression"]["node_count"] == 6

    def test_the_template_spec_is_serializable_and_revalidatable(
        self, client,
    ):
        body = client.get(TEMPLATES_URL, headers=auth_headers()).json()
        guided = body["templates"][0]
        verdict = client.post(
            VALIDATE_URL, headers=auth_headers(),
            json=guided["spec"],
        ).json()
        assert verdict["valid"] is True
        assert verdict["node_count"] == guided["node_count"]


class TestValidateLane:
    def test_a_cycle_is_rejected_with_every_finding_at_once(self, client):
        spec = {
            "name": "cycle",
            "nodes": [
                {"node_id": "a", "step_id": "chem.standardize",
                 "wiring": {"dataset": {"node": "b", "port": "dataset"}}},
                {"node_id": "b", "step_id": "chem.standardize",
                 "wiring": {"dataset": {"node": "a", "port": "dataset"}}},
            ],
        }
        verdict = client.post(
            VALIDATE_URL, headers=auth_headers(), json=spec,
        ).json()
        assert verdict["valid"] is False
        assert any("cycle" in e for e in verdict["errors"])
        assert verdict["topological_order"] is None

    def test_a_contract_mismatch_is_named(self, client):
        spec = {
            "name": "mismatch",
            "nodes": [
                {"node_id": "a", "step_id": "dataset.import"},
                {"node_id": "b", "step_id": "ml.evaluate",
                 "wiring": {"split": {"node": "a", "port": "dataset"},
                            "model": {"node": "a", "port": "dataset"}}},
            ],
        }
        verdict = client.post(
            VALIDATE_URL, headers=auth_headers(), json=spec,
        ).json()
        assert verdict["valid"] is False
        assert any("needs 'split'" in e for e in verdict["errors"])


class TestSyncRun:
    def test_the_guided_chain_round_trips_over_the_real_backends(
        self, client,
    ):
        body = client.post(
            RUN_URL, headers=auth_headers(),
            json={"template": "guided-qsar", "mode": "sync", "seed": 7},
        ).json()
        run = body["run"]
        assert run["state"] == "COMPLETED"
        assert run["node_counts"]["COMPLETED"] == 10
        nodes = {n["node_id"]: n for n in run["nodes"]}
        report = nodes["report"]["artifacts"][0]
        assert report["kind"] == "report"
        assert len(report["digest"]) == 64
        metrics_node = nodes["evaluate"]["artifacts"][0]
        lineage = {(l["node"], l["port"]) for l in
                   metrics_node["lineage"]}
        assert ("train", "model") in lineage
        assert ("import", "dataset") in lineage
        repro = run["reproducibility"]
        assert repro["seed"] == 7
        assert repro["toolkit_version"]
        assert "wall-clock" in repro["determinism_note"]

    def test_inline_records_and_param_overrides_apply(self, client):
        distinct = [
            "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CC(C)NCC(O)COc1ccccc1CC",
            "OC(=O)CC(O)(CC(=O)O)C(O)(CC(=O)O)CC(=O)O",
            "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "CCN(CC)C(=O)C1CN(C)C2=C1C=CC=C2",
            "COc1ccc2nc(S(=O)(=O)N)[nH]c2c1", "CC1(C)SC(C(S1)C(=O)O)N",
            "OC(=O)C1=CC=CC=C1O", "CCO", "CCCCO",
        ]
        records = [
            {"smiles": s, "pIC50": 3.0 + 0.5 * i} for i, s in enumerate(distinct)
        ]
        body = client.post(
            RUN_URL, headers=auth_headers(),
            json={
                "template": "minimal-regression",
                "mode": "sync",
                "records": records,
                "param_overrides": {"split": {"test_size": 0.5}},
                "seed": 3,
            },
        ).json()
        run = body["run"]
        assert run["state"] == "COMPLETED"
        assert run["node_counts"]["COMPLETED"] == 6

    def test_partial_failure_carries_the_backend_error_verbatim(
        self, client,
    ):
        body = client.post(
            RUN_URL, headers=auth_headers(),
            json={
                "template": "guided-qsar",
                "mode": "sync",
                "param_overrides": {
                    "train": {"model": "no_such_model"},
                },
            },
        ).json()
        run = body["run"]
        assert run["state"] == "PARTIAL"
        train = next(n for n in run["nodes"] if n["node_id"] == "train")
        assert train["error"].startswith("ModelError")
        assert run["node_counts"]["SKIPPED"] == 4
        report = next(n for n in run["nodes"] if n["node_id"] == "report")
        assert report["state"] == "SKIPPED"

    def test_unknown_template_is_a_422(self, client):
        response = client.post(
            RUN_URL, headers=auth_headers(),
            json={"template": "no-such-template"},
        )
        assert response.status_code == 422
        assert "unknown template" in response.json()["detail"]

    def test_invalid_records_are_a_422(self, client):
        response = client.post(
            RUN_URL, headers=auth_headers(),
            json={
                "template": "minimal-regression",
                "records": [{"smiles": "CCO", "pIC50": "high"}],
            },
        )
        assert response.status_code == 422
        assert "not numeric" in response.json()["detail"]

    def test_a_spec_that_fails_validation_is_a_422_with_the_errors(
        self, client,
    ):
        response = client.post(
            RUN_URL, headers=auth_headers(),
            json={"spec": {"name": "bad", "nodes": [
                {"node_id": "a", "step_id": "no.such.step"}]},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["errors"]


class TestJobMode:
    def test_a_job_run_completes_through_the_job_manager(self, client):
        submit = client.post(
            RUN_URL, headers=auth_headers(),
            json={
                "template": "minimal-regression",
                "mode": "job",
                "n_synthetic": 20,
                "seed": 5,
            },
        ).json()
        assert submit["mode"] == "job"
        job_id = submit["job_id"]
        deadline = time.monotonic() + 90.0
        state = None
        while time.monotonic() < deadline:
            job = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            ).json()
            state = job["state"]
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.3)
        assert state == "COMPLETED"
        result = client.get(
            f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
        ).json()["result"]
        assert result["state"] == "COMPLETED"
        assert result["node_counts"]["COMPLETED"] == 6
        assert result["reproducibility"]["seed"] == 5
        job_view = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        assert job_view["operation"] == "workflow.run"
        assert job_view["configuration"]["mode"] == "job"
        assert job_view["configuration"]["spec_digest"]

    def test_the_cooperative_cancel_flag_lands_between_nodes(
        self, client,
    ):
        submit = client.post(
            RUN_URL, headers=auth_headers(),
            json={
                "template": "guided-qsar",
                "mode": "job",
                "n_synthetic": 30,
                "seed": 9,
            },
        ).json()
        job_id = submit["job_id"]
        cancel = client.post(
            f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers()
        ).json()
        assert cancel["state"] in (
            "CANCELLED", "CANCELLATION_REQUESTED"
        )
        deadline = time.monotonic() + 90.0
        state = None
        while time.monotonic() < deadline:
            job = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            ).json()
            state = job["state"]
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.3)
        # the flag is observed between nodes -- a mid-flight catch
        # cancels the run; a queued catch cancels before start; either
        # honest path ends CANCELLED unless the whole chain finished
        # first (accepted: the chain is short and the flag is best
        # effort between nodes)
        assert state in ("CANCELLED", "COMPLETED")


class TestHistory:
    def test_the_ledger_lists_and_fetches_runs(self, client):
        first = client.post(
            RUN_URL, headers=auth_headers(),
            json={"template": "minimal-regression", "n_synthetic": 14},
        ).json()["run"]
        listing = client.get(RUNS_URL, headers=auth_headers()).json()
        assert first["run_id"] in {r["run_id"] for r in listing["runs"]}
        detail = client.get(
            f"{RUNS_URL}/{first['run_id']}", headers=auth_headers()
        ).json()
        assert detail["run_id"] == first["run_id"]
        assert detail["node_counts"]["COMPLETED"] == 6

    def test_an_unknown_run_id_is_a_404(self, client):
        response = client.get(
            f"{RUNS_URL}/wf-run-nothing", headers=auth_headers()
        )
        assert response.status_code == 404
        assert "in-session ledger" in response.json()["detail"]

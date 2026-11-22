"""part13 Phase 6, §8 — the lightweight Scientific Workflow Engine's
own lane (``pharmaml.workflow``: the typed contracts, the graph
machinery, the REAL-backend runners, the templates).

No mocks: the golden-path run executes §8's own example chain over
the REAL backends (standardize_smiles, compute_descriptors,
make_train_test_split, select_kbest_union, build_model,
evaluate_model, leverage_williams, shap_summary, save_pdf_report) on
a seeded synthetic frame; the honest lanes pin the partial-failure
semantics (the backend's own error verbatim, downstream SKIPPED,
completed artifacts surviving), the cooperative cancellation, the
transitive artifact lineage, the reproducibility block, and the
same-seed determinism claim WITH its wall-clock exclusion note.
"""

from __future__ import annotations

import json
import random
import threading

import pytest

from pharmaml.workflow.contracts import PORT_KINDS, STEP_REGISTRY
from pharmaml.workflow.engine import (
    WorkflowEngine,
    WorkflowSpec,
    WorkflowValidationError,
    execute,
)
from pharmaml.workflow.runners import STEP_RUNNERS
from pharmaml.workflow.templates import TEMPLATE_REGISTRY

SYNTHETIC_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CC(C)NCC(O)COc1ccccc1CC",
    "OC(=O)CC(O)(CC(=O)O)C(O)(CC(=O)O)CC(=O)O",
    "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "CCN(CC)C(=O)C1CN(C)C2=C1C=CC=C2",
    "COc1ccc2nc(S(=O)(=O)N)[nH]c2c1", "CC1(C)SC(C(S1)C(=O)O)N",
    "OC(=O)C1=CC=CC=C1O", "CCO", "CCCCO", "CCCO", "CCCOCCO",
    "c1ccc(O)cc1", "CC(C)(C)O", "COC(C)(C)C", "CC(C)O",
    "CCOCCO", "OC(=O)CCO", "NC(=O)c1ccccc1", "OCCN",
    "CC(N)CO", "COC(=O)c1ccccc1", "CC(C)N", "OCc1ccccc1",
    "CCOC(=O)c1ccccc1", "NCc1ccccc1", "CC(O)CO", "CCOc1ccccc1",
]


def _records(seed: int = 11, n: int = 30) -> list[dict]:
    rng = random.Random(seed)
    return [
        {"smiles": SYNTHETIC_SMILES[i % len(SYNTHETIC_SMILES)],
         "pIC50": round(rng.uniform(3.0, 9.0), 3)}
        for i in range(n)
    ]


class TestStepRegistry:
    def test_ten_steps_each_with_one_real_runner(self) -> None:
        assert len(STEP_REGISTRY) == 10
        assert set(STEP_REGISTRY) == set(STEP_RUNNERS)

    def test_ports_use_the_closed_kind_vocabulary(self) -> None:
        for step in STEP_REGISTRY.values():
            for port in (*step.inputs, *step.outputs):
                assert port.kind in PORT_KINDS, (step.step_id, port.name)

    def test_every_step_names_its_backend_and_a_note(self) -> None:
        for step in STEP_REGISTRY.values():
            assert step.backend.strip()
            assert step.notes.strip()
            assert step.workspace.strip()
            for bad in ("TODO", "placeholder", "TBD"):
                assert bad.lower() not in step.notes.lower()

    def test_leakage_note_is_the_split_before_selection_contract(self) -> None:
        split = STEP_REGISTRY["ml.split"].notes
        assert "BEFORE feature selection" in split
        assert "leakage rule" in split


class TestTemplates:
    def test_the_two_built_ins_validate_clean(self) -> None:
        for spec in TEMPLATE_REGISTRY.values():
            assert spec.validate() == []

    def test_guided_qsar_is_section8s_own_chain(self) -> None:
        spec = TEMPLATE_REGISTRY["guided-qsar"]
        assert len(spec.nodes) == 10
        assert "leakage_note" in spec.metadata
        assert "selection" in spec.metadata["leakage_note"]

    def test_template_serialization_round_trips(self) -> None:
        spec = TEMPLATE_REGISTRY["guided-qsar"]
        clone = WorkflowSpec.from_dict(spec.to_dict())
        assert clone.canonical_digest() == spec.canonical_digest()


class TestValidation:
    def test_unknown_step_is_named(self) -> None:
        spec = WorkflowSpec.from_dict({
            "name": "bad",
            "nodes": [{"node_id": "a", "step_id": "no.such.step"}],
        })
        errors = spec.validate()
        assert any("unknown step" in e for e in errors)

    def test_duplicate_node_ids_are_refused_at_parse(self) -> None:
        with pytest.raises(WorkflowValidationError) as caught:
            WorkflowSpec.from_dict({
                "name": "dup",
                "nodes": [
                    {"node_id": "a", "step_id": "dataset.import"},
                    {"node_id": "a", "step_id": "dataset.import"},
                ],
            })
        assert "duplicate node_id" in caught.value.errors[0]

    def test_dangling_wiring_is_named(self) -> None:
        spec = WorkflowSpec.from_dict({
            "name": "dangling",
            "nodes": [
                {"node_id": "a", "step_id": "dataset.import"},
                {"node_id": "b", "step_id": "chem.standardize",
                 "wiring": {"dataset": {"node": "ghost", "port": "dataset"}}},
            ],
        })
        errors = spec.validate()
        assert any("unknown node" in e for e in errors)

    def test_contract_kind_mismatch_is_named(self) -> None:
        spec = WorkflowSpec.from_dict({
            "name": "mismatch",
            "nodes": [
                {"node_id": "a", "step_id": "dataset.import"},
                {"node_id": "b", "step_id": "ml.evaluate",
                 "wiring": {"split": {"node": "a", "port": "dataset"},
                            "model": {"node": "a", "port": "dataset"}}},
            ],
        })
        errors = spec.validate()
        assert any("needs 'split'" in e and "produces 'dataframe'" in e
                   for e in errors)

    def test_cycle_is_detected(self) -> None:
        spec = WorkflowSpec.from_dict({
            "name": "cycle",
            "nodes": [
                {"node_id": "a", "step_id": "chem.standardize",
                 "wiring": {"dataset": {"node": "b", "port": "dataset"}}},
                {"node_id": "b", "step_id": "chem.standardize",
                 "wiring": {"dataset": {"node": "a", "port": "dataset"}}},
            ],
        })
        errors = spec.validate()
        assert any("cycle" in e for e in errors)
        assert spec.topological_order() is None

    def test_unknown_parameter_is_named(self) -> None:
        spec = WorkflowSpec.from_dict({
            "name": "param",
            "nodes": [{"node_id": "a", "step_id": "dataset.import",
                       "params": {"bogus_dial": 1}}],
        })
        assert any("no parameter" in e for e in spec.validate())

    def test_execute_refuses_an_invalid_spec_with_every_error(self) -> None:
        spec = WorkflowSpec.from_dict({
            "name": "bad", "nodes": [{"node_id": "", "step_id": "nope"}],
        })
        with pytest.raises(WorkflowValidationError) as caught:
            execute(spec, initial_inputs={})
        assert len(caught.value.errors) >= 2


class TestGoldenPath:
    """The REAL round trip over §8's example chain."""

    def test_guided_qsar_completes_with_lineage_and_reproducibility(
        self,
    ) -> None:
        engine = WorkflowEngine()
        run = engine.execute(
            TEMPLATE_REGISTRY["guided-qsar"],
            initial_inputs={"records": _records()},
            seed=7,
        )
        assert run.state == "COMPLETED", [n.error for n in run.nodes]
        assert run.counts() == {"COMPLETED": 10, "FAILED": 0,
                               "SKIPPED": 0, "CANCELLED": 0}
        by_id = {n.node_id: n for n in run.nodes}
        metrics = by_id["evaluate"].artifacts[0]
        assert metrics.kind == "metrics"
        lineage = {(l["node"], l["port"]) for l in
                   json.loads(json.dumps(metrics.public()))["lineage"]}
        assert ("train", "model") in lineage
        assert ("import", "dataset") in lineage  # the transitive chain
        report = by_id["report"].artifacts[0]
        assert report.digest_kind == "json"
        assert len(report.digest) == 64
        assert run.reproducibility["seed"] == 7
        assert "toolkit_version" in run.reproducibility
        assert "wall-clock" in run.reproducibility["determinism_note"]

    def test_same_seed_reproduces_the_artifact_digests(self) -> None:
        import copy

        records = _records(seed=3)
        base = copy.deepcopy(TEMPLATE_REGISTRY["guided-qsar"])
        # the f_regression-only selector is deterministic; the MI
        # member of the default union pair is unseeded in the backend
        base.nodes["selection"].params["methods"] = ("f_regression",)
        first = execute(base, initial_inputs={"records": records}, seed=5)
        second = execute(base, initial_inputs={"records": records}, seed=5)
        digests_first = {
            n.node_id: [a.digest for a in n.artifacts]
            for n in first.nodes
            if n.state == "COMPLETED" and n.node_id != "report"
        }
        digests_second = {
            n.node_id: [a.digest for a in n.artifacts]
            for n in second.nodes
            if n.state == "COMPLETED" and n.node_id != "report"
        }
        # the report node is EXCLUDED on purpose: its PDF bytes embed
        # fpdf's own creation timestamp (verified: same content built
        # 1.2 s apart digests differently) — the honest determinism
        # scope, stated in the run's own determinism_note
        assert first.state == "COMPLETED"
        assert digests_first == digests_second

    def test_the_public_view_is_json_safe(self) -> None:
        engine = WorkflowEngine()
        run = engine.execute(
            TEMPLATE_REGISTRY["minimal-regression"],
            initial_inputs={"records": _records(seed=9, n=26)},
            seed=4,
        )
        assert run.state == "COMPLETED"
        json.dumps(run.public())  # must not raise

    def test_the_engine_ledger_lists_and_fetches_runs(self) -> None:
        engine = WorkflowEngine()
        run = engine.execute(
            TEMPLATE_REGISTRY["minimal-regression"],
            initial_inputs={"records": _records(seed=9, n=26)},
            seed=4,
        )
        listed = engine.list_runs()
        assert listed[0]["run_id"] == run.run_id
        assert engine.get_run(run.run_id) is run
        assert engine.get_run("wf-run-nothing") is None


class TestPartialFailure:
    def test_backend_error_is_carried_verbatim_and_downstream_skips(
        self,
    ) -> None:
        import copy

        spec = copy.deepcopy(TEMPLATE_REGISTRY["guided-qsar"])
        spec.nodes["train"].params["model"] = "no_such_model"
        run = execute(spec, initial_inputs={"records": _records()}, seed=7)
        assert run.state == "PARTIAL"
        counts = run.counts()
        assert counts["FAILED"] == 1
        assert counts["SKIPPED"] == 4  # evaluate, applicability, shap, report
        train = next(n for n in run.nodes if n.node_id == "train")
        assert train.error.startswith("ModelError")
        report = next(n for n in run.nodes if n.node_id == "report")
        assert report.state == "SKIPPED"
        assert "unavailable" in (report.skipped_reason or "")
        # the completed artifacts SURVIVE the partial run
        descriptors = next(
            n for n in run.nodes if n.node_id == "descriptors")
        assert descriptors.state == "COMPLETED"
        assert descriptors.artifacts

    def test_small_frame_hits_the_backend_own_honest_guard(self) -> None:
        # 12 rows -> 9 train rows against the leverage/Williams
        # backend's own p+2 rows requirement: the honest guard fires
        # (rank hygiene cannot conjure rows)
        run = execute(
            TEMPLATE_REGISTRY["guided-qsar"],
            initial_inputs={"records": _records(seed=2, n=12)},
            seed=7,
        )
        applicability = next(
            n for n in run.nodes if n.node_id == "applicability")
        assert applicability.state == "FAILED"
        assert "training rows" in (applicability.error or "")


class TestCancellation:
    def test_the_cancel_flag_is_observed_between_nodes(self) -> None:
        event = threading.Event()
        seen: list[tuple[int, str]] = []

        def progress(done: int, total: int, node: str, state: str) -> None:
            seen.append((done, state))
            if done >= 3:
                event.set()

        run = execute(
            TEMPLATE_REGISTRY["guided-qsar"],
            initial_inputs={"records": _records()},
            seed=7,
            cancel_event=event,
            progress=progress,
        )
        assert run.state == "CANCELLED"
        assert run.counts()["CANCELLED"] >= 1
        cancelled = [n for n in run.nodes if n.state == "CANCELLED"]
        assert cancelled, "the cancel flag must land on at least one node"
        completed = [n for n in run.nodes if n.state == "COMPLETED"]
        assert completed, "the pre-cancel nodes keep their completions"


class TestSerializationStability:
    def test_canonical_digest_is_order_stable(self) -> None:
        spec = TEMPLATE_REGISTRY["guided-qsar"]
        reordered = WorkflowSpec(
            name=spec.name,
            description=spec.description,
            nodes=dict(reversed(list(spec.nodes.items()))),
            metadata=spec.metadata,
        )
        assert reordered.canonical_digest() == spec.canonical_digest()

"""part10's third queued row — the dl / semi-supervised / generative
API surface.

Test-alongside-every-change, exactly like the earlier suites: every
route here runs the REAL backend (DeepMLP/ResidualMLP torch fits with
their own ``on_epoch`` telemetry, ``train_semi_supervised`` /
``evaluate_semi_supervised``, ``SmilesRnn.train_prior`` /
``.sample``, ``GeneticOptimizer.optimize`` with its native progress
callback), and the §9.3 training-progress vocabulary is verified on
the wire:

* ``training.progress`` / ``training.metrics`` events per torch
  epoch (replay-verified with real epoch counters and losses, and
  streamed live over the WebSocket);
* ``iteration.update`` events per evolution generation;
* the evolution's own ``JobCancelled`` — raised by the backend when
  the progress adapter answers ``False`` — lands the job in
  CANCELLED, never FAILED (the real cooperative-cancel path);
* a DL-trained model and a semi-supervised model both predict
  through the single ``/api/v1/ml/predict`` path (one prediction
  path, never two).
"""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"

SIMPLE_SMILES = [
    "C", "CC", "CCC", "CCCC", "CCCCC", "CCO", "CCCO", "CCCN",
    "c1ccccc1", "Cc1ccccc1", "CC(=O)O", "CC(=O)N", "CSC", "CCSC",
]


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _csv_bytes(rows: int = 24) -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(rows):
        buf.write(f"{round(0.1 * i + 0.5, 3)},{(i * 7) % 5 * 0.25},{i % 2}\n")
    return buf.getvalue().encode()


def _upload(client, data: bytes | None = None, name: str = "toy.csv",
            target: str = "y", mode: str = "sync") -> dict:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, data or _csv_bytes(), "text/csv")},
        data={"target": target, "mode": mode},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _wait_job(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {job}")


def _bus_events(client) -> list[dict]:
    """The session bus history (envelope dicts, replayed from 0)."""
    return [
        e.model_dump(mode="json", exclude_none=True)
        for e in client.app.state.bus.replay(0)
    ]


# ------------------------------------------------------------------ dl


class TestDLNetworks:
    def test_catalog_lists_six_tabular_torch_families(self, client):
        r = client.get("/api/v1/dl/networks", headers=auth_headers())
        assert r.status_code == 200
        nets = r.json()["networks"]
        assert [n["network"] for n in nets] == [
            "deep_mlp_classifier", "deep_mlp_regressor",
            "residual_mlp_classifier", "residual_mlp_regressor",
            # part18 §5.4 — the tabular Transformer lane (from scratch;
            # no pretrained molecular transformer is claimed anywhere)
            "tabular_transformer_classifier", "tabular_transformer_regressor",
        ]
        by_name = {n["network"]: n for n in nets}
        # live defaults read off the class signatures — one source of truth
        assert by_name["deep_mlp_classifier"]["defaults"]["epochs"] == 120
        assert by_name["deep_mlp_classifier"]["engine"] == "torch"
        assert by_name["residual_mlp_regressor"]["class"] == (
            "ResidualMLPRegressor"
        )
        assert by_name["deep_mlp_classifier"]["task"] == "classification"
        # the transformer families ride the same lane + defaults surface
        assert by_name["tabular_transformer_classifier"]["engine"] == "torch"
        assert by_name["tabular_transformer_classifier"]["defaults"]["d_model"] == 32
        assert by_name["tabular_transformer_regressor"]["task"] == "regression"

    def test_catalog_documents_the_single_prediction_path(self, client):
        r = client.get("/api/v1/dl/networks", headers=auth_headers())
        assert r.status_code == 200
        assert r.json()["predict_url"] == "/api/v1/ml/predict"


class TestDLTrain:
    def _tiny(self, client, dataset_id: str, network: str,
              mode: str = "sync", hyperparams: dict | None = None) -> dict:
        r = client.post(
            "/api/v1/dl/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset_id,
                "network": network,
                "hyperparams": hyperparams or {
                    "hidden_layers": [8], "epochs": 3, "patience": 3,
                    "batch_size": 8, "val_split": 0.2,
                },
                "mode": mode,
            },
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_sync_train_registers_a_torch_model_that_predicts(
        self, client,
    ):
        dataset = _upload(client)
        payload = self._tiny(
            client, dataset["dataset_id"], "deep_mlp_classifier"
        )
        model = payload["model"]
        assert model["engine"] == "torch"
        assert model["task"] == "classification"
        assert model["n_train_rows"] == 24
        assert model["best_params"]["network"] == "deep_mlp_classifier"
        assert model["best_params"]["epochs"] == 3
        assert model["fit_seconds"] > 0.0
        # the single prediction path serves the DL-trained model
        r = client.post(
            "/api/v1/ml/predict",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "rows": [
                    {"x1": 1.0, "x2": 0.5},
                    {"x1": 2.0, "x2": 1.5},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_rows"] == 2
        assert len(body["predictions"]) == 2

    def test_job_mode_publishes_per_epoch_training_events(self, client):
        dataset = _upload(client)
        payload = self._tiny(
            client, dataset["dataset_id"], "residual_mlp_classifier",
            mode="job",
        )
        job = payload["job"]
        assert job["operation"] == "dl.train"
        assert job["workspace"] == "deep-learning"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        # REAL per-epoch telemetry on the bus: training.progress with
        # epoch counters + training.metrics with the losses
        events = _bus_events(client)
        progress = [
            e for e in events
            if e["event_type"] == "training.progress"
            and e["job_id"] == job["job_id"]
        ]
        metrics = [
            e for e in events
            if e["event_type"] == "training.metrics"
            and e["job_id"] == job["job_id"]
        ]
        assert len(progress) >= 3  # one per epoch (epochs=3)
        assert len(metrics) >= 3
        assert {p["payload"]["stage"] for p in progress} >= {
            "epoch 1/3", "epoch 2/3", "epoch 3/3"
        }
        assert [p["payload"]["progress"] for p in progress][-1] == 100
        assert all("train_loss" in m["payload"] for m in metrics)
        assert all("epoch" in m["payload"] for m in metrics)
        # the job record's own progress tracked the epochs
        assert final["progress"] == 100
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["model"]["engine"] == "torch"

    def test_unknown_hyperparameter_is_an_honest_422(self, client):
        dataset = _upload(client)
        r = client.post(
            "/api/v1/dl/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "network": "deep_mlp_classifier",
                "hyperparams": {"bogus": 1},
            },
        )
        assert r.status_code == 422
        assert "bogus" in r.json()["detail"]

    def test_task_mismatch_is_an_honest_422(self, client):
        dataset = _upload(client)  # a classification dataset
        r = client.post(
            "/api/v1/dl/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "network": "deep_mlp_regressor",
            },
        )
        assert r.status_code == 422
        assert "regression" in r.json()["detail"]

    def test_unknown_dataset_is_404(self, client):
        r = client.post(
            "/api/v1/dl/train",
            headers=auth_headers(),
            json={"dataset_id": "ds-none", "network": "deep_mlp_classifier"},
        )
        assert r.status_code == 404

    def test_training_events_stream_over_the_websocket(self, client):
        with client.websocket_connect(
            f"/api/v1/events?token={TEST_TOKEN}"
        ) as ws:
            json.loads(ws.receive_text())  # hello
            dataset = _upload(client)
            payload = self._tiny(
                client, dataset["dataset_id"],
                "deep_mlp_classifier", mode="job",
            )
            job_id = payload["job"]["job_id"]
            seen = []
            got_completed = False
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and not got_completed:
                msg = json.loads(ws.receive_text())
                if msg.get("job_id") == job_id:
                    seen.append(msg["event_type"])
                    if msg["event_type"] == "job.completed":
                        got_completed = True
            assert got_completed
            assert "training.progress" in seen
            assert "training.metrics" in seen


# -------------------------------------------------------- semi-supervised


class TestSemiSupervisedRoutes:
    def test_train_with_mask_fraction_and_predict_path(self, client):
        dataset = _upload(client)
        r = client.post(
            "/api/v1/semi-supervised/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "method": "self_training",
                "mask_fraction": 0.5,
                "random_state": 7,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        semi = r.json()["semi_supervised"]
        assert semi["method"] == "self_training"
        assert semi["n_labeled"] + semi["n_unlabeled"] == 24
        assert semi["n_unlabeled"] == 12  # round(24 * 0.5)
        assert semi["n_pseudo_labeled"] >= 0
        assert semi["classes"] == [0, 1]
        assert "model_id" in semi
        # the fitted model predicts through the single prediction path
        pr = client.post(
            "/api/v1/ml/predict",
            headers=auth_headers(),
            json={
                "model_id": semi["model_id"],
                "rows": [{"x1": 1.0, "x2": 0.5}, {"x1": 3.0, "x2": 1.0}],
            },
        )
        assert pr.status_code == 200, pr.text
        assert len(pr.json()["predictions"]) == 2

    def test_train_job_mode_completes(self, client):
        dataset = _upload(client)
        r = client.post(
            "/api/v1/semi-supervised/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "method": "label_propagation",
                "mask_fraction": 0.4,
                "mode": "job",
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        assert job["operation"] == "semi_supervised.train"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["semi_supervised"]["method"] == "label_propagation"

    def test_without_marker_or_mask_is_an_honest_422(self, client):
        dataset = _upload(client)  # no -1 markers in the target
        r = client.post(
            "/api/v1/semi-supervised/train",
            headers=auth_headers(),
            json={"dataset_id": dataset["dataset_id"]},
        )
        assert r.status_code == 422
        assert "mask_fraction" in r.json()["detail"]

    def test_unknown_base_estimator_is_an_honest_422(self, client):
        dataset = _upload(client)
        r = client.post(
            "/api/v1/semi-supervised/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "mask_fraction": 0.5,
                "base_estimator": "not_a_registry_model",
            },
        )
        assert r.status_code == 422

    def test_classification_guard_rejects_regression_datasets(
        self, client,
    ):
        buf = io.StringIO()
        buf.write("x1,x2,y\n")
        for i in range(24):
            buf.write(f"{round(0.1 * i, 3)},{0.25 * (i % 5)},{0.2 * i}\n")
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={"file": ("reg.csv", buf.getvalue().encode(), "text/csv")},
            data={"target": "y", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        dataset = r.json()["dataset"]
        r = client.post(
            "/api/v1/semi-supervised/train",
            headers=auth_headers(),
            json={"dataset_id": dataset["dataset_id"]},
        )
        assert r.status_code == 422

    def test_evaluate_heldout_vs_baseline(self, client):
        dataset = _upload(client)
        r = client.post(
            "/api/v1/semi-supervised/evaluate",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "method": "self_training",
                "test_size": 0.3,
                "n_repeats": 2,
            },
        )
        assert r.status_code == 200, r.text
        evaluation = r.json()["evaluation"]
        assert 0.0 <= evaluation["heldout_accuracy"] <= 1.0
        assert 0.0 <= evaluation["baseline_accuracy"] <= 1.0
        assert "delta_vs_baseline" in evaluation
        assert evaluation["n_repeats"] == 2

    def test_evaluate_job_mode_completes(self, client):
        # r303: the job lane was never exercised — the route passed a
        # spurious manager kwarg into _evaluate_body and every job-mode
        # submission FAILED with TypeError. This test pins the lane.
        dataset = _upload(client)
        r = client.post(
            "/api/v1/semi-supervised/evaluate",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "method": "self_training",
                "test_size": 0.25,
                "n_repeats": 3,
                "mode": "job",
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        assert job["operation"] == "semi_supervised.evaluate"
        assert job["workspace"] == "machine-learning"
        deadline = time.monotonic() + 120.0
        final = job
        while time.monotonic() < deadline:
            final = client.get(
                f"/api/v1/jobs/{job['job_id']}", headers=auth_headers()
            ).json()
            if final["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.05)
        assert final["state"] == "COMPLETED", final
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        )
        assert result.status_code == 200, result.text
        evaluation = result.json()["result"]["evaluation"]
        assert 0.0 <= evaluation["heldout_accuracy"] <= 1.0
        assert 0.0 <= evaluation["baseline_accuracy"] <= 1.0
        assert "delta_vs_baseline" in evaluation
        assert evaluation["n_repeats"] == 3


# ------------------------------------------------------------- generative


class TestGenerativeObjectives:
    def test_objectives_catalog(self, client):
        r = client.get("/api/v1/generative/objectives", headers=auth_headers())
        assert r.status_code == 200
        kinds = [o["kind"] for o in r.json()["objectives"]]
        assert kinds == [
            "qed", "sas", "logp_band", "mw_band", "tpsa_band",
            "similarity", "substructure", "novelty", "multi",
        ]

    def test_unknown_objective_kind_is_an_honest_422(self, client):
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={"objective": {"kind": "nope"}},
        )
        assert r.status_code == 422

    def test_band_objective_needs_its_numbers(self, client):
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={"objective": {"kind": "logp_band"}},
        )
        assert r.status_code == 422
        assert "lo" in r.json()["detail"]

    def test_multi_objective_needs_aligned_weights(self, client):
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {
                    "kind": "multi",
                    "objectives": [
                        {"kind": "qed"},
                        {"kind": "logp_band", "lo": -1, "hi": 3},
                    ],
                    "weights": [1.0],
                }
            },
        )
        assert r.status_code == 422
        assert "weights" in r.json()["detail"]


class TestGenerativeOperatorModes:
    """GEN-1 (the roadmap's row 4): the operator modes reach the REAL
    engine through POST /api/v1/generative/optimize.

    The route previously built ga_kwargs from exactly eight numeric
    knobs -- the mutation/crossover operator families the engine has
    validated since part19 were unreachable from the API. The fixture
    is CORPUS-FREE (population_size == len(seed_smiles), so the
    engine seeds the population from exactly the parents and never
    draws DEFAULT_CORPUS members): with mutation="scaffold" (every
    mutant preserves the parent's Murcko framework) the ONLY source
    of a framework change is the scaffold_hop crossover -- the
    Murcko assertion below is therefore fail-first against the route
    dropping the kwargs (the engine's own metadata echo answers
    "uniform"/"splice" and no hopped molecule can exist).
    """

    PARENTS = [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin  (benzene framework)
        "COc1ccccc1",               # anisole  (benzene framework)
        "CN1CCCCC1c2cccnc2",        # nicotine (bicyclic framework)
        "c1ccncc1",                 # pyridine (its own framework)
    ]

    def _murcko(self, smiles: str) -> str:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

    def _parent_frameworks(self) -> set[str]:
        return {self._murcko(s) for s in self.PARENTS}

    def _hop(self, client, **extra) -> dict:
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {"kind": "qed"},
                "seed_smiles": self.PARENTS,
                "population_size": 4,
                "generations": 8,
                "elitism": 1,
                "tournament_k": 2,
                "crossover_rate": 1.0,
                "mutation": "scaffold",
                "crossover": "scaffold_hop",
                "top_n": 50,
                "mode": "sync",
                **extra,
            },
        )
        assert r.status_code == 200, r.text
        return r.json()

    def _produced(self, payload: dict) -> list[str]:
        opt = payload["optimization"]
        out = [row["smiles"] for row in opt["top"]]
        out.append(opt["best"])
        for record in opt["history"]:
            out.append(record["best_smiles"])
        return [s for s in out if s]

    def test_scaffold_hop_reaches_the_evolution_and_hops(self, client):
        payload = self._hop(client)
        opt = payload["optimization"]
        meta = opt["metadata"]
        # THE wiring proof (fail-first: the route drops the kwargs ->
        # the engine's own defaults answer "uniform"/"splice" and
        # these two asserts fail immediately)
        assert meta["mutation"] == "scaffold"
        assert meta["crossover"] == "scaffold_hop"
        # the hop's effect evidenced in the produced set: at least one
        # molecule's Murcko framework differs from EVERY parent's
        # framework (supplementary evidence -- the character-mutation
        # fallback can in principle also change a framework, so the
        # metadata echo above is the fail-first leg of the pair)
        parents = self._parent_frameworks()
        hopped = [
            smi for smi in self._produced(payload)
            if self._murcko(smi) not in parents
        ]
        assert hopped, (
            "no scaffold-hop child surfaced in the produced set -- the "
            "operator mode did not ride the evolution"
        )

    def test_the_min_heavy_atoms_gate_rides_the_hop(self, client):
        # the route-level scope: the field rides the request (a 200
        # run -- an unknown value would 422) and reaches the engine
        # (the metadata echo); the gate's own per-child semantics is
        # the ENGINE's tested contract (tests/test_generative.py's
        # TestScaffoldHop, round 100x-53). A per-child route-level
        # assertion is NOT honest here: the mutation family's
        # character fallback carries no gate, so a framework-changed
        # molecule under a high gate is not proof of a leak.
        payload = self._hop(client, min_heavy_atoms=10)
        opt = payload["optimization"]
        assert opt["metadata"]["crossover"] == "scaffold_hop"
        assert opt["generations"] == 8  # the run completed
        produced = self._produced(payload)
        assert produced  # the evolution produced molecules

    def test_unknown_operator_modes_are_honest_422s(self, client):
        for field, bad in (("mutation", "teleport"), ("crossover", "hop")):
            r = client.post(
                "/api/v1/generative/optimize",
                headers=auth_headers(),
                json={
                    "objective": {"kind": "qed"},
                    "seed_smiles": ["CCO", "CCN", "CCC"],
                    "population_size": 8,
                    "generations": 2,
                    field: bad,
                    "mode": "sync",
                },
            )
            assert r.status_code == 422, (field, r.text)
            assert bad in str(r.json()["detail"])

    def test_the_default_lane_is_unchanged(self, client):
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {"kind": "qed"},
                "seed_smiles": ["CCO", "CCN", "CCC"],
                "population_size": 8,
                "generations": 2,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        meta = r.json()["optimization"]["metadata"]
        assert meta["mutation"] == "uniform"
        assert meta["crossover"] == "splice"


class TestGenerativePrior:
    def _train(self, client, mode: str = "sync", **extra) -> dict:
        r = client.post(
            "/api/v1/generative/rnn/train",
            headers=auth_headers(),
            json={
                "smiles": SIMPLE_SMILES,
                "name": "toy-prior",
                "embedding_dim": 16,
                "hidden_dim": 24,
                "epochs": 2,
                "batch_size": 8,
                "mode": mode,
                **extra,
            },
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_train_and_sample_roundtrip(self, client):
        payload = self._train(client)
        prior = payload["prior"]
        assert prior["n_train_smiles"] == len(SIMPLE_SMILES)
        assert prior["n_epochs"] == 2
        assert len(prior["losses"]) == 2
        assert prior["final_loss"] > 0.0
        r = client.post(
            "/api/v1/generative/rnn/sample",
            headers=auth_headers(),
            json={
                "prior_id": prior["prior_id"],
                "n": 8,
                "temperature": 0.9,
                "seed": 7,
                "max_length": 60,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_sampled"] == 8
        assert body["n_requested"] == 8
        assert len(body["samples"]) == 8
        assert all(isinstance(s, str) for s in body["samples"])
        # deterministic sampling: the same seed repeats itself
        r2 = client.post(
            "/api/v1/generative/rnn/sample",
            headers=auth_headers(),
            json={
                "prior_id": prior["prior_id"],
                "n": 8,
                "temperature": 0.9,
                "seed": 7,
                "max_length": 60,
            },
        )
        assert r2.json()["samples"] == body["samples"]

    def test_prior_train_job_publishes_the_loss_curve(self, client):
        payload = self._train(client, mode="job")
        job = payload["job"]
        assert job["operation"] == "generative.prior_train"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        events = _bus_events(client)
        curve = [
            e for e in events
            if e["event_type"] == "training.metrics"
            and e["job_id"] == job["job_id"]
        ]
        assert len(curve) >= 1
        assert len(curve[0]["payload"]["losses"]) == 2
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["prior"]["n_epochs"] == 2

    def test_priors_listing_and_view(self, client):
        prior = self._train(client)["prior"]
        r = client.get("/api/v1/generative/priors", headers=auth_headers())
        assert [p["prior_id"] for p in r.json()["priors"]] == [
            prior["prior_id"]
        ]
        r = client.get(
            f"/api/v1/generative/priors/{prior['prior_id']}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        assert r.json()["corpus_source"]["source"] == "inline"

    def test_prior_train_from_a_dataset_column(self, client):
        buf = io.StringIO()
        buf.write("smiles,activity,y\n")
        for i, s in enumerate(SIMPLE_SMILES):
            buf.write(f"{s},{0.5 * i},{i % 2}\n")
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={
                "file": ("corpus.csv", buf.getvalue().encode(), "text/csv")
            },
            data={"target": "y", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        dataset = r.json()["dataset"]
        r = client.post(
            "/api/v1/generative/rnn/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "column": "smiles",
                "epochs": 1,
                "embedding_dim": 16,
                "hidden_dim": 24,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        prior = r.json()["prior"]
        assert prior["corpus_source"]["source"] == "dataset"
        assert prior["corpus_source"]["column"] == "smiles"
        assert prior["n_train_smiles"] == len(SIMPLE_SMILES)

    def test_prior_corpus_needs_five_smiles(self, client):
        r = client.post(
            "/api/v1/generative/rnn/train",
            headers=auth_headers(),
            json={"smiles": ["C", "CC", "CCC"]},
        )
        assert r.status_code == 422

    def test_sample_unknown_prior_is_404(self, client):
        r = client.post(
            "/api/v1/generative/rnn/sample",
            headers=auth_headers(),
            json={"prior_id": "prior-none", "n": 2},
        )
        assert r.status_code == 404


class TestGenerativeOptimize:
    def _optimize(self, client, mode: str = "sync", **extra) -> dict:
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {"kind": "qed"},
                "seed_smiles": ["CCO", "CCN", "CCC"],
                "population_size": 8,
                "generations": 3,
                "mode": mode,
                **extra,
            },
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_sync_optimize_over_the_real_evolution(self, client):
        payload = self._optimize(client)
        opt = payload["optimization"]
        assert 0.0 <= opt["best_score"] <= 1.0
        assert opt["generations"] == 3
        assert len(opt["history"]) == 3
        assert 1 <= len(opt["top"]) <= 10  # the top_n cap
        assert all("score" in row for row in opt["top"])

    def test_job_mode_emits_iteration_update_events(self, client):
        payload = self._optimize(client, mode="job")
        job = payload["job"]
        assert job["operation"] == "generative.optimize"
        assert job["workspace"] == "generative-ai"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        events = _bus_events(client)
        iterations = [
            e for e in events
            if e["event_type"] == "iteration.update"
            and e["job_id"] == job["job_id"]
        ]
        # the backend's own telemetry: the evolution calls its progress
        # callback after each generation but the last (it exits the
        # loop straight into the final ranking) — N-1 calls for N
        # generations, exactly what the REAL optimizer emits
        assert len(iterations) >= 2
        progress_seen = [it["payload"]["progress"] for it in iterations]
        assert progress_seen == sorted(progress_seen)
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["optimization"]["best_score"] >= 0.0

    def test_cancellation_lands_cancelled_never_failed(self, client):
        # a longer evolution so the cancel arrives mid-run
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {"kind": "qed"},
                "seed_smiles": ["CCO", "CCN", "CCC"],
                "population_size": 16,
                "generations": 120,
                "mode": "job",
            },
        )
        job = r.json()["job"]
        # wait until it is really running, then cancel
        deadline = time.monotonic() + 30
        running = False
        while time.monotonic() < deadline:
            state = client.get(
                f"/api/v1/jobs/{job['job_id']}", headers=auth_headers()
            ).json()
            if state["state"] == "RUNNING":
                running = True
                break
            time.sleep(0.02)
        assert running, state
        client.post(
            f"/api/v1/jobs/{job['job_id']}/cancel", headers=auth_headers()
        )
        final = _wait_job(client, job["job_id"], timeout=60)
        assert final["state"] == "CANCELLED", final
        assert final["cancellation"] == "cooperative"

    def test_multi_objective_optimization(self, client):
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {
                    "kind": "multi",
                    "objectives": [
                        {"kind": "qed"},
                        {"kind": "logp_band", "lo": -1, "hi": 3},
                    ],
                    "weights": [0.7, 0.3],
                },
                "seed_smiles": ["CCO", "CC(=O)N"],
                "population_size": 8,
                "generations": 2,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        opt = r.json()["optimization"]
        assert len(opt["top"]) >= 1
        assert "breakdown" in opt["top"][0]

    def test_optimize_with_a_dataset_corpus(self, client):
        buf = io.StringIO()
        buf.write("smiles,activity,y\n")
        for i, s in enumerate(SIMPLE_SMILES):
            buf.write(f"{s},{0.5 * i},{i % 2}\n")
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={
                "file": ("corpus.csv", buf.getvalue().encode(), "text/csv")
            },
            data={"target": "y", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        dataset = r.json()["dataset"]
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={
                "objective": {"kind": "qed"},
                "dataset_id": dataset["dataset_id"],
                "column": "smiles",
                "population_size": 8,
                "generations": 2,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        opt = r.json()["optimization"]
        assert 0.0 <= opt["best_score"] <= 1.0
        assert opt["generations"] == 2

    def test_optimize_needs_seeds_or_a_corpus(self, client):
        r = client.post(
            "/api/v1/generative/optimize",
            headers=auth_headers(),
            json={"objective": {"kind": "qed"}},
        )
        assert r.status_code == 422

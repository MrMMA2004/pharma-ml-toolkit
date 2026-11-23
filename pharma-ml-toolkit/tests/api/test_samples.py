"""The part16 §8 (Phase 5) demo/sample-data lane — the API proof.

The directive's asks, made mechanical (test-alongside-every-change):

* **"Do not invent sample datasets"** — every catalog entry is
  re-verified against the REAL shipped file (existence, size,
  sha256) by :class:`TestCatalogReality`; the guard fails red the
  day a curated row drifts from ``examples/data`` reality;
* **preview-before-load** — the raw file head, never the imported
  view, and a preview registers nothing
  (:class:`TestPreview`);
* **one-click load + no-overwrite** — the tabular import rides the
  REAL ``load_dataset`` + ``DatasetStore.register`` path (a fresh
  ``ds-<uuid>`` every time — two loads never replace each other)
  (:class:`TestLoadTabular`);
* **"each sample entry actually loads and can be used by its target
  workflow"** — :class:`TestWorkflowUsability` runs the real target
  lanes: the ML workspace trains on the loaded demo datasets (sync,
  tune off, the conservative n_jobs=1), the PK/PD workspace computes
  NCA over the delivered series, the docking workspace parses the
  delivered DLG, the chemistry/datasets lanes profile the loaded
  library, and the column bridge serves the SMILES seeds the
  generative stage consumes;
* **the NCA zero-row fix rides this lane** (fail-first: the shipped
  ``examples/data/pk_profile.csv`` starts with the pre-dose t=0 row
  and the route's blanket positive-only validation rejected the very
  file the demo lane ships — the library's own NCA core deliberately
  accepts intentional zeros; ``NcaRequest`` now mirrors the
  steady-state precedent) — :class:`TestSeriesLoad` pins it green and
  pins negatives still rejected.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.routes import samples as samples_module

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _catalog(client) -> list[dict]:
    r = client.get("/api/v1/samples", headers=auth_headers())
    assert r.status_code == 200, r.text
    return r.json()["samples"]


def _load(client, sample_id: str) -> dict:
    r = client.post(f"/api/v1/samples/{sample_id}/load", headers=auth_headers())
    assert r.status_code == 200, r.text
    return r.json()


class TestCatalogReality:
    """The no-invention rule, enforced: catalog <-> shipped files."""

    def test_every_entry_names_a_real_shipped_file(self):
        root = samples_module.SAMPLES_DIR
        assert root.is_dir(), f"sample root missing: {root}"
        for entry in samples_module._CATALOG:
            path = root / str(entry["filename"])
            assert path.is_file(), (
                f"curated sample {entry['id']!r} names a file that does "
                f"not exist on disk: {entry['filename']!r}"
            )

    def test_catalog_reports_every_entry_available_with_digest(self, client):
        entries = _catalog(client)
        assert len(entries) == len(samples_module._CATALOG)
        for view in entries:
            assert view["available"] is True, view
            assert view["size_bytes"] and view["size_bytes"] > 0, view
            assert isinstance(view["sha256"], str) and len(view["sha256"]) == 64
            path = samples_module.SAMPLES_DIR / view["filename"]
            real = hashlib.sha256(path.read_bytes()).hexdigest()
            assert view["sha256"] == real, view["sample_id"]
            for key in (
                "sample_id", "title", "filename", "kind", "file_type",
                "description", "scientific_context", "workflows",
                "cleanup",
            ):
                assert view[key], f"{view['sample_id']}: missing {key}"

    def test_every_directive_workspace_has_samples(self, client):
        required = {
            "datasets", "machine-learning", "deep-learning",
            "generative-ai", "molecular-docking", "pkpd",
            "molecular-dynamics", "visualization",
        }
        have = set()
        for view in _catalog(client):
            have.update(view["workflows"])
        missing = required - have
        assert not missing, f"workspaces with no demo data: {missing}"

    def test_workflow_filter_and_unknown_tag(self, client):
        r = client.get(
            "/api/v1/samples?workflow=machine-learning",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        ids = [s["sample_id"] for s in r.json()["samples"]]
        assert "qsar-classification" in ids
        assert "pk-profile" not in ids
        r = client.get("/api/v1/samples?workflow=nonsense", headers=auth_headers())
        assert r.status_code == 422

    def test_unknown_sample_404(self, client):
        r = client.get(
            "/api/v1/samples/does-not-exist/preview", headers=auth_headers()
        )
        assert r.status_code == 404
        r = client.post(
            "/api/v1/samples/does-not-exist/load", headers=auth_headers()
        )
        assert r.status_code == 404


class TestPreview:
    """Preview-before-load: the RAW file head, registering nothing."""

    def test_tabular_preview_is_the_raw_head(self, client):
        r = client.get(
            "/api/v1/samples/qsar-classification/preview?limit=3",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "tabular"
        assert body["columns"][:3] == ["descr_00", "descr_01", "descr_02"]
        assert body["returned"] == 3
        assert body["n_rows"] >= body["returned"]
        assert body["dtypes"]

    def test_preview_registers_nothing(self, client):
        before = len(client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"])
        for sample_id in ("qsar-classification", "lysozyme-receptor"):
            r = client.get(
                f"/api/v1/samples/{sample_id}/preview",
                headers=auth_headers(),
            )
            assert r.status_code == 200, r.text
        after = len(client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"])
        assert after == before

    def test_text_preview_lines_and_truncation(self, client):
        r = client.get(
            "/api/v1/samples/lysozyme-receptor/preview?limit=4",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "text_file"
        assert body["returned"] == 4
        assert body["truncated"] is True
        assert body["n_rows"] > 4
        assert body["lines"][0].startswith(("HEADER", "REMARK", "ATOM", "TITLE"))

    def test_preview_limit_bounds_422(self, client):
        for bad in (0, 21):
            r = client.get(
                f"/api/v1/samples/qsar-classification/preview?limit={bad}",
                headers=auth_headers(),
            )
            assert r.status_code == 422, bad


class TestLoadTabular:
    """One-click import over the REAL upload path, structurally no-overwrite."""

    def test_load_registers_a_fresh_usable_dataset(self, client):
        body = _load(client, "qsar-classification")
        assert body["kind"] == "tabular"
        dataset = body["dataset"]
        assert dataset["dataset_id"].startswith("ds-")
        # the curated hint rides the REAL loader: target/task inferred
        assert dataset["target"] == "activity"
        assert dataset["task"] == "classification"
        assert dataset["name"] == "Demo — QSAR classification set"
        assert dataset["source_filename"] == "qsar_classification.csv"
        # the record is live in the registry the workspaces consume
        listed = client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"]
        assert dataset["dataset_id"] in {d["dataset_id"] for d in listed}
        r = client.get(
            f"/api/v1/datasets/{dataset['dataset_id']}/preview?limit=5",
            headers=auth_headers(),
        )
        assert r.status_code == 200

    def test_loading_twice_never_replaces(self, client):
        first = _load(client, "walkthrough-classification")["dataset"]
        second = _load(client, "walkthrough-classification")["dataset"]
        assert first["dataset_id"] != second["dataset_id"]
        # both records stay alive — a demo load never evicts user data
        for dataset_id in (first["dataset_id"], second["dataset_id"]):
            r = client.get(
                f"/api/v1/datasets/{dataset_id}", headers=auth_headers()
            )
            assert r.status_code == 200, dataset_id

    def test_target_override_rides_the_curated_hint(self, client):
        r = client.post(
            "/api/v1/samples/qsar-multiclass/load",
            headers=auth_headers(),
            json={"target": "Target", "task": "classification"},
        )
        assert r.status_code == 200
        assert r.json()["dataset"]["target"] == "Target"


class TestSeriesLoad:
    """The PK lane's series kind + the NcaRequest zero-row fix."""

    def test_pk_profile_delivers_the_nca_input_shape(self, client):
        body = _load(client, "pk-profile")
        assert body["kind"] == "series"
        assert body["columns"] == {"time": "time_h", "value": "conc_ng_ml"}
        assert len(body["times"]) == body["n_points"] == len(body["conc"])
        assert body["n_points"] >= 2
        # the real shipped file starts with the pre-dose zero row
        assert body["conc"][0] == 0.0

    def test_series_feeds_nca_end_to_end(self, client):
        """The shipped demo profile completes the PK workflow (fail-first).

        Red before the NcaRequest fix: the route's blanket
        positive-only validation rejected the very file the lane ships
        (t=0 pre-dose row), while the library's own NCA core and the
        shipped examples/run_pk_analysis.py consume it fine.
        """
        body = _load(client, "pk-profile")
        r = client.post(
            "/api/v1/pkpd/nca",
            headers=auth_headers(),
            json={"times": body["times"], "conc": body["conc"]},
        )
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["n_points"] == body["n_points"]
        assert result["auc_trapezoid_linear"] > 0.0
        # the zero row correctly disables the log-trapezoid AUC lane
        assert result["auc_trapezoid_log"] is None
        assert result["terminal"]["half_life"] > 0.0

    def test_nca_still_rejects_negative_concentrations(self, client):
        r = client.post(
            "/api/v1/pkpd/nca",
            headers=auth_headers(),
            json={"times": [0.0, 1.0, 2.0], "conc": [0.0, -1.0, 2.0]},
        )
        assert r.status_code == 422


class TestTextLoad:
    """The docking/MD/visualization inputs take the content verbatim."""

    def test_text_load_carries_the_file_reality(self, client):
        body = _load(client, "lysozyme-receptor")
        assert body["kind"] == "text_file"
        assert body["file_type"] == "pdb"
        path = (
            samples_module.SAMPLES_DIR / "lysozyme_1aki.pdb"
        )
        assert body["size_bytes"] == path.stat().st_size
        real = hashlib.sha256(path.read_bytes()).hexdigest()
        assert body["sha256"] == real
        assert body["truncated"] is False
        assert body["content"]
        assert "ATOM" in body["content"]

    def test_docking_log_loads_and_parses(self, client):
        """The docking workflow's DLG lane consumes the delivered log."""
        body = _load(client, "aspirin-docking-log")
        assert body["kind"] == "text_file"
        assert body["file_type"] == "dlg"
        r = client.post(
            "/api/v1/docking/parse/dlg",
            headers=auth_headers(),
            json={"text": body["content"]},
        )
        assert r.status_code == 200, r.text
        parsed = r.json()
        assert parsed["clusters"], "the aspirin DLG parsed to no clusters"
        assert parsed["clusters"][0]["rank"] == 1
        assert parsed["clusters"][0]["lowest_energy"] < 0.0

    def test_pdbqt_ligand_loads(self, client):
        body = _load(client, "demo-ligand-pdbqt")
        assert body["file_type"] == "pdbqt"
        assert "ATOM" in body["content"]


class TestWorkflowUsability:
    """Each sample loads AND its target workflow consumes it (§8's ask)."""

    def test_ml_classification_workflow_trains_on_the_demo_set(self, client):
        dataset = _load(client, "walkthrough-classification")["dataset"]
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "model": "logistic_regression",
                "task": "classification",
                "tune": False,
                "cv": 2,
                "n_jobs": 1,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        model = r.json()["model"]
        assert model["model_id"].startswith("mdl-")
        assert model["task"] == "classification"

    def test_ml_regression_workflow_trains_on_the_demo_set(self, client):
        dataset = _load(client, "qsar-regression")["dataset"]
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": dataset["dataset_id"],
                "model": "ridge",
                "task": "regression",
                "tune": False,
                "cv": 2,
                "n_jobs": 1,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["model"]["task"] == "regression"

    def test_datasets_workflow_profiles_the_demo_library(self, client):
        dataset = _load(client, "descriptor-library")["dataset"]
        r = client.post(
            f"/api/v1/datasets/{dataset['dataset_id']}/profile",
            headers=auth_headers(),
            json={"mode": "sync"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["profile"]

    def test_dl_lane_dataset_view_serves_the_response_table(self, client):
        dataset = _load(client, "cellline-screen")["dataset"]
        assert dataset["task"] == "regression"
        assert dataset["target"] == "LN_IC50"
        r = client.get(
            f"/api/v1/datasets/{dataset['dataset_id']}/preview?limit=5",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        assert "LN_IC50" in r.json()["columns"]

    def test_generative_lane_delivers_the_seed_smiles(self, client):
        """The generative seed stage consumes the delivered SMILES."""
        body = _load(client, "seed-molecules")
        assert body["kind"] == "library"
        assert body["n_molecules"] == len(body["smiles"]) == len(body["names"])
        assert body["n_molecules"] >= 2
        assert all(isinstance(s, str) and s for s in body["smiles"])

    def test_chemistry_lane_descriptors_over_the_demo_library(self, client):
        """The chemistry workflow's descriptor stage over the demo library."""
        body = _load(client, "screening-library")
        assert body["kind"] == "library"
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={
                "smiles": body["smiles"][:2],
                "descriptor_set": "rdkit_all",
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        result = r.json()["result"]
        assert result["rows"] and result["columns"]

    def test_docking_ligand_preparation_over_the_demo_library(self, client):
        """The docking workflow's ligand-prep stage over the demo list."""
        body = _load(client, "campaign-ligands")
        assert body["n_molecules"] == 14
        r = client.post(
            "/api/v1/docking/ligand/pdbqt",
            headers=auth_headers(),
            json={"smiles": body["smiles"][0]},
        )
        assert r.status_code == 200, r.text


class TestSecurity:
    """The demo lane rides the same loopback + token guard as the rest."""

    def test_samples_require_the_session_token(self, client):
        for method, path in (
            ("get", "/api/v1/samples"),
            ("get", "/api/v1/samples/qsar-classification/preview"),
            ("post", "/api/v1/samples/qsar-classification/load"),
        ):
            r = getattr(client, method)(path)
            assert r.status_code == 401, (method, path, r.status_code)

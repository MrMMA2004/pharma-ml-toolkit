"""part13 Phase 5's next remainder item — the DTA API lane over the
REAL in-tree backend (``pharmaml.ml.dta``: DeepDTA's verbatim
25/64/62-symbol alphabets, the AAC protein panel, the 2048-bit ECFP
ligand pairing, the ``deepdta_rm2`` external-validation metric).

No mocks: the registry answers carry the backend's own alphabets
verbatim (CHARPROTSET's 1-based codes, both SMILES alphabets' sizes),
the feature pass is the real ``paired_dta_features`` over real pairs —
the matrix width is asserted against the backend's own arithmetic
(2048 ECFP + 23 AAC = 2071; +401 dipeptide lane = 2472), the demo
frame's honest note rides, and the hard-error lanes carry the backend's
own messages verbatim: protein sequence errors and non-finite
affinities are 422s (dropping those silently would make a benchmark
look cleaner than its source data); RDKit-invalid ligands are dropped
together with their protein and affinity and are reported per-row,
never silently. The honest provenance lanes pin §11.1's own rules: the
DeepDTA CNN / DeepPurpose / DGL-LifeSci / TorchDrug checkpoints are
NOT shipped and no affinity prediction is ever invented here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"

PROTEIN = "MKTAYIAKQRQISFVKSHFSRQ"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


class TestDtaRegistry:
    """GET /api/v1/dta/registry — the verbatim encoding registry."""

    def test_alphabets_are_the_backend_own_cells(self, client):
        r = client.get("/api/v1/dta/registry", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        alphabets = body["alphabets"]
        # the three DeepDTA alphabets, sizes + padding verbatim
        assert alphabets["protein"]["n_symbols"] == 25
        assert alphabets["smiles_isomeric"]["n_symbols"] == 64
        assert alphabets["smiles_canonical"]["n_symbols"] == 62
        for block in alphabets.values():
            assert block["padding_code"] == 0
            assert len(block["symbols"]) == block["n_symbols"]
        # CHARPROTSET's own 1-based codes (A=1, C=2, X=24, Z=25)
        protein_symbols = alphabets["protein"]["symbols"]
        assert protein_symbols["A"] == 1
        assert protein_symbols["C"] == 2
        assert protein_symbols["X"] == 24
        assert protein_symbols["Z"] == 25
        # the isomeric alphabet's own two-out-of-order cells
        iso = alphabets["smiles_isomeric"]["symbols"]
        assert iso["C"] == 42 and iso["@"] == 8 and iso["y"] == 64

    def test_pairing_numbers_are_the_backend_own_arithmetic(self, client):
        r = client.get("/api/v1/dta/registry", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        # the 20 standard amino acids
        assert len(body["amino_acids"]) == 20
        assert body["amino_acids"][0] == "A" and body["amino_acids"][-1] == "Y"
        protein_features = body["protein_features"]
        assert protein_features["n_base"] == 23  # 3 summaries + 20 AAC
        assert protein_features["n_with_dipeptides"] == 424  # +1 +400
        assert protein_features["n_dipeptide_extras"] == 401
        assert body["ligand_features"] == {
            "family": "ecfp",
            "n_bits": 2048,
            "source": (
                "the toolkit's own chem.features.compute_feature_frame "
                "(families=('ecfp',))"
            ),
        }
        assert body["paired_width"] == {"base": 2071, "with_dipeptides": 2472}
        # the metric + the demo frame ride their honest descriptions
        assert [m["name"] for m in body["metrics"]] == ["deepdta_rm2"]
        assert body["demo_frame"]["rows"] == 12

    def test_provenance_block_is_the_honest_state(self, client):
        r = client.get("/api/v1/dta/registry", headers=auth_headers())
        assert r.status_code == 200
        prov = r.json()["provenance"]
        assert prov["backend"] == "pharmaml.ml.dta"
        assert prov["n_alphabets"] == 3
        # the honest absence, stated — never invented predictions
        assert prov["shipped_checkpoints"] == 0
        assert "NOT shipped" in prov["shipped_checkpoints_note"]
        for repository in ("DeepDTA CNN", "DeepPurpose", "DGL-LifeSci", "TorchDrug"):
            assert repository in prov["shipped_checkpoints_note"]
        assert "not learned affinity predictions" in prov["disclaimer"]


class TestDtaFeatures:
    """POST /api/v1/dta/features — the real paired featurization."""

    def test_real_pair_card_over_the_real_backend(self, client):
        r = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={
                "pairs": [
                    {"smiles": "CCO", "protein": PROTEIN, "affinity": 5.0},
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_pairs"] == 1 and body["n_kept"] == 1
        assert body["n_dropped"] == 0
        # the backend's own arithmetic, asserted against the doctest
        assert body["n_ligand_features"] == 2048
        assert body["n_protein_features"] == 23
        assert body["n_features"] == 2071
        row = body["rows"][0]
        assert row["valid"] is True
        protein_row = row["protein_row"]
        assert protein_row["prot_length"] == len(PROTEIN)
        assert protein_row["prot_standard_fraction"] == 1.0
        assert protein_row["prot_nonstandard_fraction"] == 0.0
        # the 20 AAC percentages sum to 1 over a standard-only sequence
        assert len(protein_row["aac"]) == 20
        assert abs(sum(protein_row["aac"].values()) - 1.0) < 1e-9
        assert row["affinity"] == 5.0
        assert body["affinity"] == {
            "present": True,
            "n": 1,
            "min": 5.0,
            "max": 5.0,
            "mean": 5.0,
        }
        assert "estimates" not in body["provenance"]["disclaimer"]

    def test_mixed_validity_batch_reports_dropped_rows(self, client):
        r = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={
                "pairs": [
                    {"smiles": "CCO", "protein": PROTEIN, "affinity": 5.0},
                    {"smiles": "CCN", "protein": "GAVLILKKGGSSHHTRPFSL",
                     "affinity": 6.0},
                    {"smiles": "not-a-molecule", "protein": PROTEIN,
                     "affinity": 7.0},
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_pairs"] == 3 and body["n_kept"] == 2
        assert body["n_dropped"] == 1
        # the dropped row reports itself with its source index
        dropped = body["dropped"]
        assert dropped[0]["index"] == 2
        assert dropped[0]["smiles"] == "not-a-molecule"
        assert "invalid SMILES" in dropped[0]["reason"]
        # the invalid row still reports its protein composition card
        invalid_row = body["rows"][2]
        assert invalid_row["valid"] is False
        assert invalid_row["protein_row"]["prot_length"] == len(PROTEIN)
        # its affinity dropped with it — the summary carries only the kept
        assert body["affinity"]["n"] == 2
        assert body["affinity"]["min"] == 5.0 and body["affinity"]["max"] == 6.0
        # kept_indices preserve the surviving source-row positions
        assert body["kept_indices"] == [0, 1]

    def test_dipeptides_flag_widens_the_matrix(self, client):
        r = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={
                "pairs": [
                    {"smiles": "CCO", "protein": "ACDEFGHIKLMNPQRSTVWY"},
                ],
                "dipeptides": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dipeptides"] is True
        assert body["n_protein_features"] == 424
        assert body["n_features"] == 2472

    def test_bad_protein_symbol_is_the_backend_own_422(self, client):
        r = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={"pairs": [{"smiles": "CCO", "protein": "MKAJY"}]},
        )
        assert r.status_code == 422
        # the backend's own message, verbatim — never softened
        assert "symbols outside DeepDTA's alphabet" in r.json()["detail"]
        assert "J" in r.json()["detail"]

    def test_non_finite_affinity_is_the_backend_own_422(self, client):
        r = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={
                "pairs": [
                    {"smiles": "CCO", "protein": "MK", "affinity": 5.0},
                    {"smiles": "CCN", "protein": "MK", "affinity": None},
                ]
            },
        )
        assert r.status_code == 422
        # the backend's own contract: a missing value is a hard error
        assert "affinity must be finite numeric at source rows" in r.json()["detail"]

    def test_all_invalid_smiles_is_the_backend_own_422(self, client):
        r = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={
                "pairs": [
                    {"smiles": "%%%", "protein": "MK"},
                    {"smiles": "!!!", "protein": "MK"},
                ]
            },
        )
        assert r.status_code == 422
        # the feature matrix is the product — never an empty 200
        assert "no valid ligand/protein pairs remain" in r.json()["detail"]

    def test_empty_blank_and_cap_lanes(self, client):
        empty = client.post(
            "/api/v1/dta/features", headers=auth_headers(), json={"pairs": []}
        )
        assert empty.status_code == 422
        blank = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={"pairs": [{"smiles": "   ", "protein": "MK"}]},
        )
        assert blank.status_code == 422
        over_cap = client.post(
            "/api/v1/dta/features",
            headers=auth_headers(),
            json={
                "pairs": [
                    {"smiles": "CCO", "protein": "MK"} for _ in range(101)
                ]
            },
        )
        assert over_cap.status_code == 422

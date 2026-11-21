"""part13 Phase 5's next remainder item — the ADMET API lane over the
REAL vendored backend (``pharmaml.ml.admet``: the swansonk14/admet_ai
52-endpoint registry, the RDKit physchem panel with the three
FilterCatalog alert counters, the DrugBank radial quantile grid).

No mocks: the registry answers carry the vendored table verbatim
(52 rows, the quality joins, the low-confidence warning set), and the
scorecard is the real ``summarize_molecule`` pass over real SMILES —
aspirin's panel is asserted against RDKit's own numbers, the radial
scores are 0-100 by construction, and invalid SMILES rows report
themselves per-row (never dropped silently, never a 500). The honest
provenance lanes pin part13 §11.1's own rules: predicted ADMET values
are model estimates, not experimental facts; the 41 TDC-trained
chemprop ensembles are NOT shipped and never invented here.
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


class TestAdmetEndpoints:
    """GET /api/v1/admet/endpoints — the vendored registry."""

    def test_registry_has_52_rows_with_quality_joins(self, client):
        r = client.get("/api/v1/admet/endpoints", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        rows = body["endpoints"]
        assert len(rows) == 52
        ids = {row["id"] for row in rows}
        # one row from every category of the vendored table
        for expected in (
            "molecular_weight",
            "Caco2_Wang",
            "BBB_Martins",
            "Half_Life_Obach",
            "CYP3A4_Veith",
            "hERG",
            "SR-p53",
        ):
            assert expected in ids
        by_id = {row["id"]: row for row in rows}
        # the verbatim table cells (admet.csv's own values)
        assert by_id["LD50_Zhu"]["units"] == "log(1/(mol/kg))"
        assert by_id["LD50_Zhu"]["species"] == "rat"
        assert by_id["LD50_Zhu"]["size"] == 7342
        assert by_id["LD50_Zhu"]["task_type"] == "regression"
        assert by_id["PPBR_AZ"]["maximum"] == 100.0
        assert by_id["Lipinski"]["maximum"] == 4.0
        # the quality joins
        assert by_id["hERG"]["quality"] == 0.839
        assert by_id["AMES"]["quality"] == 0.882
        # the categories + the radial axes block
        assert set(body["categories"]) == {
            "Physicochemical",
            "Absorption",
            "Distribution",
            "Excretion",
            "Metabolism",
            "Toxicity",
        }
        axes = body["radial_axes"]
        assert len(axes) == 5
        assert [axis["endpoint"] for axis in axes] == [
            "BBB_Martins",
            "ClinTox",
            "Solubility_AqSolDB",
            "Bioavailability_Ma",
            "hERG",
        ]
        inverted = {axis["endpoint"] for axis in axes if axis["invert"]}
        assert inverted == {"BBB_Martins", "ClinTox", "hERG"}

    def test_low_confidence_set_is_the_vendor_warning_rule(self, client):
        r = client.get("/api/v1/admet/endpoints", headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        low = set(body["low_confidence_endpoints"])
        # the below-0.6 quality rows, verbatim from the vendor's own rule
        assert "VDss_Lombardo" in low and "Half_Life_Obach" in low
        assert "Clearance_Hepatocyte_AZ" in low
        assert "AMES" not in low and "hERG" not in low
        # the per-row flag agrees with the set
        flagged = {
            row["id"]
            for row in body["endpoints"]
            if row["low_confidence"]
        }
        assert flagged == low

    def test_provenance_block_is_the_honest_vendor_state(self, client):
        r = client.get("/api/v1/admet/endpoints", headers=auth_headers())
        assert r.status_code == 200
        prov = r.json()["provenance"]
        assert prov["backend"] == "pharmaml.ml.admet"
        assert prov["n_endpoints"] == 52
        assert prov["n_learned_endpoints"] == 41
        # the honest absence, stated — never invented predictions
        assert prov["shipped_ensembles"] == 0
        assert "NOT shipped" in prov["shipped_ensembles_note"]
        assert prov["low_performance_threshold"] == 0.6
        assert "not" in prov["disclaimer"]


class TestAdmetScore:
    """POST /api/v1/admet/score — the real per-molecule scorecard."""

    def test_aspirin_card_over_the_real_rdkit_panel(self, client):
        r = client.post(
            "/api/v1/admet/score",
            headers=auth_headers(),
            json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_smiles"] == 1 and body["n_valid"] == 1
        mol = body["molecules"][0]
        assert mol["valid"] is True
        physchem = mol["physchem"]
        # the real RDKit numbers the backend's own doctest pins
        assert round(physchem["molecular_weight"], 1) == 180.2
        assert physchem["hydrogen_bond_donors"] == 1.0
        assert 0.0 <= physchem["QED"] <= 1.0
        assert physchem["PAINS_alert"] == 0.0
        assert 0.0 <= physchem["logP"] <= 3.0
        assert physchem["tpsa"] > 60.0
        # the five-axis radial summary, 0-100 by construction
        radial = mol["radial"]
        assert len(radial) == 5
        assert all(0.0 <= value <= 100.0 for value in radial.values())
        # the honest notes ride every answer
        assert "low_confidence_endpoints" in mol
        assert "estimates" in body["provenance"]["disclaimer"]

    def test_batch_score_with_mixed_validity_reports_rows(self, client):
        r = client.post(
            "/api/v1/admet/score",
            headers=auth_headers(),
            json={
                "smiles": [
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                    "not-a-molecule",
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_smiles"] == 3 and body["n_valid"] == 2
        assert body["n_invalid"] == 1
        rows = body["molecules"]
        assert rows[0]["valid"] is True
        assert rows[1]["valid"] is True
        caffeine = rows[1]["physchem"]
        assert round(caffeine["molecular_weight"], 1) == 194.2
        # the invalid row reports itself — never dropped, never a 500
        assert rows[2]["valid"] is False
        assert "invalid SMILES" in rows[2]["error"]

    def test_empty_and_blank_smiles_are_422(self, client):
        r = client.post(
            "/api/v1/admet/score", headers=auth_headers(), json={"smiles": []}
        )
        assert r.status_code == 422
        r = client.post(
            "/api/v1/admet/score", headers=auth_headers(), json={"smiles": ["   "]}
        )
        assert r.status_code == 422

    def test_batch_cap_is_the_chemistry_precedent(self, client):
        r = client.post(
            "/api/v1/admet/score",
            headers=auth_headers(),
            json={"smiles": ["CCO"] * 101},
        )
        assert r.status_code == 422

    def test_all_invalid_rows_still_answer_200_with_provenance(self, client):
        r = client.post(
            "/api/v1/admet/score",
            headers=auth_headers(),
            json={"smiles": ["%%%", "!!!"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_valid"] == 0 and body["n_invalid"] == 2
        assert all(row["valid"] is False for row in body["molecules"])
        assert body["provenance"]["shipped_ensembles"] == 0

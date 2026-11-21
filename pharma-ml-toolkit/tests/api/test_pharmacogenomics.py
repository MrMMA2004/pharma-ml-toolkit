"""part13 Phase 5's next remainder item — the pharmacogenomic
association API lane over the REAL in-tree backend
(``pharmaml.ml.pharmacogenomics``: the classic GDSC ANOVA with its
design matrix and feasibility gates, the Storey-Tibshirani q-values
and BH FDR, the volcano transforms, the guarded metric family and
the additive mean-effects ruler with the normalized metric).

No mocks: the registry answers carry the backend's own vocabulary
verbatim (the gates 6/3/3 and 2-2, the row schema, the percent
convention, the volcano hit rule, the degenerate-input guards, the
CNA vocabulary, the enrichment branches), and the analyze pass is
the real round trip — the seeded screen, the real ANOVA over every
feasible drug x feature pair, the q-value/FDR columns, the volcano
verdicts, the top associations table, and the mean-effects ruler
evaluated through the guarded family with the normalized score
beyond the naive drug-mean ruler — asserted against the backend's
own arithmetic. The honest provenance lanes pin §11.1's own rules:
five repositories reviewed file by file, zero real
GDSC/CCLE/CTRP/DepMap tables ship, the served screen is the
backend's own SYNTHETIC seeded demo, one response scale only (the
natural-log µM scale), and no clinical recommendation is ever
emitted.
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


REGISTRY_URL = "/api/v1/pharmacogenomics/registry"
ANALYZE_URL = "/api/v1/pharmacogenomics/analyze"


class TestPharmacogenomicsRegistry:
    """GET /api/v1/pharmacogenomics/registry — the verbatim vocabulary."""

    def test_anova_vocabulary_is_the_backend_own(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        associations = body["associations"]
        assert "QR F-test" in associations["design_matrix"]
        assert "first sorted level" in associations["design_matrix"]
        gates = associations["feasibility_gates"]
        assert ">= 6 non-NA" in gates["drug"]
        assert ">= 3 positives AND >= 3 negatives" in gates["feature"]
        assert ">= 2 lines" in gates["msi"]
        assert "single tissue" in gates["tissue"]
        assert associations["row_schema"][:4] == [
            "drug", "feature", "n_pos", "n_neg",
        ]
        assert "qvalue" in associations["row_schema"]
        assert "fdr_percent" in associations["row_schema"]
        assert "sorted by pvalue ascending" in associations["ordering"]

    def test_multiple_testing_layer_is_verbatim(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        multiple = body["multiple_testing"]
        assert multiple["methods"] == ["bh_fdr", "qvalue"]
        assert "monotone back-substitution" in multiple["bh_fdr"]
        assert "arange(0, 0.9 + 1e-8, 0.05) (19 values)" in multiple["qvalue"]
        assert "cubic spline (df = 3)" in multiple["qvalue"]
        assert "25 means 25 %" in multiple["percent_convention"]
        volcano = body["volcano"]
        assert "signed" in volcano["transforms"]
        assert "pvalue <= 0.001" in volcano["hit_rule"]
        assert "fdr_percent <= 25" in volcano["hit_rule"]
        assert "sensitive" in volcano["verdicts"]
        assert "resistant" in volcano["verdicts"]

    def test_evaluation_vocabulary_is_drevalpy_own(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        evaluation = body["evaluation"]
        assert "overall + tissue + cell-residual + drug" in evaluation["ruler"]
        assert "no double counting" in evaluation["ruler"]
        assert evaluation["guarded_metrics"] == [
            "pearson", "spearman", "kendall",
        ]
        assert "scores 0.0" in evaluation["guards"]
        assert "NaN" in evaluation["guards"]
        assert "BEYOND the naive ruler" in evaluation["normalized_metric"]

    def test_cbioportal_vocabulary_rides_the_registry(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        vocabulary = body["vocabulary"]
        assert vocabulary["cna_values"]["-2"].startswith("deep loss")
        assert vocabulary["cna_values"]["2"].startswith("high-level")
        assert vocabulary["cna_values"]["0"] == "diploid"
        assert "diploid samples" in vocabulary["diploid_zscores"]
        assert "0.3 default" in vocabulary["coexpression"]
        assert "Fisher exact" in vocabulary["enrichment_dispatchers"]
        assert "NaN" in vocabulary["enrichment_dispatchers"]

    def test_screen_block_states_the_demo_contract(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        screen = body["screen"]
        assert screen["geometry"] == {
            "n_cells": 40, "n_drugs": 12, "n_features": 10,
        }
        assert "TP53_mut" in screen["planted_associations"]
        assert "SYNTHETIC" in screen["label"]
        assert "natural-log" in screen["scale"]
        assert "IC50 = exp(LN_IC50)" in screen["scale"]
        assert "~15 %" in screen["missing"]
        assert body["analyze_url"] == ANALYZE_URL

    def test_provenance_block_is_the_honest_state(self, client):
        body = client.get(REGISTRY_URL, headers=auth_headers()).json()
        provenance = body["provenance"]
        assert provenance["backend"] == "pharmaml.ml.pharmacogenomics"
        assert provenance["n_repositories"] == 5
        assert "gdsctools" in provenance["source"]
        assert "drevalpy" in provenance["source"]
        assert "cbioportal" in provenance["source"]
        assert provenance["shipped_tables"] == 0
        assert "no real GDSC, CCLE, CTRP or DepMap" in provenance[
            "shipped_tables_note"
        ]
        assert provenance["shipped_checkpoints"] == 0
        assert "SYNTHETIC" in provenance["screen"]
        assert "never clinical recommendations" in provenance["disclaimer"]


class TestPharmacogenomicsAnalyze:
    """POST /api/v1/pharmacogenomics/analyze — the real backend round trip."""

    def test_real_round_trip_over_the_real_backend(self, client):
        first = client.post(ANALYZE_URL, json={}, headers=auth_headers())
        assert first.status_code == 200
        body = first.json()
        screen = body["screen"]
        assert screen["n_cells"] == 40 and screen["n_drugs"] == 12
        assert screen["n_features"] == 10
        assert screen["drugs"][0] == "D01" and screen["drugs"][-1] == "D12"
        assert "TP53_mut" in screen["features"]
        assert 0.0 < screen["observed_fraction"] < 1.0
        assert screen["n_observed"] < screen["n_entries"]
        assert "SYNTHETIC" in screen["label"]
        assert sorted(screen["tissues"]) == ["breast", "colon", "lung"]
        assert screen["msi"]["msi_positive"] >= 1
        associations = body["associations"]
        assert associations["n_rows"] > 40  # the gates still leave plenty
        assert associations["n_drugs_tested"] == 12
        assert associations["n_features_tested"] >= 8
        assert "triage hypotheses" in body["note"]

        # the same seed is the same round trip — deterministic end to end
        second = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        assert second["associations"]["top"] == associations["top"]
        assert second["evaluation"] == body["evaluation"]
        assert second["volcano"] == body["volcano"]

    def test_the_planted_association_wins_the_top_table(self, client):
        body = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        top = body["associations"]["top"]
        assert top[0]["feature"] in ("TP53_mut", "KRAS_mut")
        for row in top:
            assert row["drug"].startswith("D")
            assert row["n_pos"] >= 3 and row["n_neg"] >= 3
            assert 0.0 <= row["pvalue"] <= 1.0
            assert row["qvalue"] is not None
            assert row["fdr_percent"] is not None
            assert row["verdict"] in (
                "sensitive", "resistant", "not significant",
            )
        # the sensitive plants are negative shifts on the ln-IC50 scale
        sensitive_rows = [r for r in top if r["verdict"] == "sensitive"]
        for row in sensitive_rows:
            assert row["delta_mean"] < 0.0

    def test_top_k_slices_the_full_table(self, client):
        body = client.post(
            ANALYZE_URL, json={"top_k": 3}, headers=auth_headers()
        ).json()
        associations = body["associations"]
        assert associations["top_k"] == 3
        assert len(associations["top"]) == 3
        pvalues = [row["pvalue"] for row in associations["top"]]
        assert pvalues == sorted(pvalues)

    def test_volcano_verdict_census_sums_to_the_table(self, client):
        body = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        volcano = body["volcano"]
        census = (
            volcano["n_sensitive"]
            + volcano["n_resistant"]
            + volcano["n_not_significant"]
        )
        assert census == body["associations"]["n_rows"]
        assert volcano["n_sensitive"] >= 1   # the planted shifts fire
        assert volcano["n_resistant"] >= 1  # the +0.8 plant fires

    def test_the_mean_effects_ruler_rides_the_guarded_family(self, client):
        body = client.post(ANALYZE_URL, json={}, headers=auth_headers()).json()
        evaluation = body["evaluation"]
        assert evaluation["n_observed"] == body["screen"]["n_observed"]
        for key in ("pearson", "spearman", "kendall"):
            assert key in evaluation["guarded"]
            value = evaluation["guarded"][key]
            assert value is not None
            assert -1.0 <= value <= 1.0
        for family in ("naive_drug_mean", "naive_cell_mean"):
            assert family in evaluation
            assert evaluation[family]["pearson"] is not None
        normalized = evaluation["normalized_beyond_drug_mean"]
        assert sorted(normalized) == ["pearson", "spearman"]
        assert "constant prediction scores 0" in evaluation["guards_note"]
        assert "beyond the naive drug-mean baseline" in evaluation[
            "normalized_note"
        ]

    def test_include_msi_false_drops_the_factor(self, client):
        body = client.post(
            ANALYZE_URL,
            json={"include_msi": False},
            headers=auth_headers(),
        ).json()
        top = body["associations"]["top"]
        for row in top:
            assert row["msi_pvalue"] is None  # the factor left the design
        assert body["request"]["include_msi"] is False

    def test_seed_is_the_only_generative_dial(self, client):
        first = client.post(
            ANALYZE_URL, json={"seed": 7}, headers=auth_headers()
        ).json()
        second = client.post(
            ANALYZE_URL, json={"seed": 7}, headers=auth_headers()
        ).json()
        other = client.post(
            ANALYZE_URL, json={"seed": 8}, headers=auth_headers()
        ).json()
        assert first["associations"]["top"] == second["associations"]["top"]
        assert first["associations"]["top"] != other["associations"]["top"]
        assert first["screen"]["n_observed"] != other["screen"][
            "n_observed"
        ] or first["evaluation"]["guarded"] != other["evaluation"]["guarded"]

    def test_geometry_and_caps_are_the_route_own_422(self, client):
        for payload in ({"seed": -1}, {"seed": 2147483648}, {"top_k": 0}, {"top_k": 21}):
            answer = client.post(
                ANALYZE_URL, json=payload, headers=auth_headers()
            )
            assert answer.status_code == 422, payload
            assert str(answer.json())  # the failing field rides the detail

    def test_custom_parameters_reach_the_backend(self, client):
        body = client.post(
            ANALYZE_URL,
            json={"seed": 9, "top_k": 8, "include_msi": True},
            headers=auth_headers(),
        ).json()
        echoed = body["request"]
        assert echoed["seed"] == 9 and echoed["top_k"] == 8
        assert echoed["include_msi"] is True
        assert len(body["associations"]["top"]) == 8

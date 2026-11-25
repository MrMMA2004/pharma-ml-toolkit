"""part17 §6 (the R1 lane) -- the Response Surface Methodology API
tests over the REAL in-tree backend (``pharmaml.ml.rsm``: the design
generators, the second-order fit with its ANOVA / diagnostics /
hierarchy / aliasing disclosure, the zone-vocabulary prediction lane,
the contour / desirability grids, and the Derringer-Suich optimiser).

No mocks: every answer is the real module's own arithmetic over the
seeded SYNTHETIC demo experiment (the KNOWN noise-free surfaces -- so
the exact-coefficient-recovery and the optimum verifications are
EXACT, not approximate) or over uploaded CSV experiments built from
the same known surfaces.  The honest lanes pin the scope: blocking is
the two-block CCD split only (Box-Behnken / full-factorial refusals
asserted as 422s), lack-of-fit needs genuinely replicated design
points (nulls asserted where they cannot exist), the noise-free
pure-error F is undefined and reported null (never invented), the
optimiser is confined to the declared ranges (extrapolation flagged
only on the prediction lane), and the desirability bounds fall back to
the observed design min/max with the source stated, never silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.ml.rsm import (
    _DEMO_COEFS,
    DEMO_FACTORS,
    DEMO_RESPONSES,
    build_design,
    build_terms,
    demo_rsm_dataset,
    fit_response_surface,
    model_matrix,
)

TEST_TOKEN = "test-session-token-0123456789"

REGISTRY_URL = "/api/v1/rsm/registry"
DESIGN_URL = "/api/v1/rsm/design"
FIT_URL = "/api/v1/rsm/fit"
PREDICT_URL = "/api/v1/rsm/predict"
CONTOUR_URL = "/api/v1/rsm/contour"
OPTIMIZE_URL = "/api/v1/rsm/optimize"
MODELS_URL = "/api/v1/rsm/models"

_DEMO_FACTORS_BODY = [
    {"name": "temperature_c", "low": 50.0, "high": 90.0, "unit": "degC"},
    {"name": "ph", "low": 5.0, "high": 9.0, "unit": "pH"},
    {"name": "substrate_mM", "low": 10.0, "high": 50.0, "unit": "mM"},
]


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def _fit_demo(client) -> str:
    r = client.post(FIT_URL, headers=auth_headers(), json={})
    assert r.status_code == 200, r.text
    return r.json()["model"]["model_id"]


def _upload(client, data: bytes, name: str = "rsm.csv") -> str:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, data, "text/csv")},
        data={"target": "yield_pct", "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["dataset_id"]


def _demo_csv() -> bytes:
    return demo_rsm_dataset().to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_carries_the_backend_vocabulary(self, client):
        r = client.get(REGISTRY_URL, headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        keys = {d["key"] for d in body["designs"]}
        assert keys == {"ccd", "box_behnken", "full_factorial"}
        by_key = {d["key"]: d for d in body["designs"]}
        # CCD: the three alpha modes + the two-block split supported
        assert by_key["ccd"]["alpha_types"] == ["rotatable", "faced", "custom"]
        assert by_key["ccd"]["blocking"] is True
        assert by_key["ccd"]["runs_formula"] == "2^k + 2k + centers (x replicates)"
        # Box-Behnken: blocking honestly NOT supported
        assert by_key["box_behnken"]["blocking"] is False
        assert by_key["box_behnken"]["min_factors"] == 3
        # the full factorial cap
        assert by_key["full_factorial"]["max_factors"] == 8
        for d in body["designs"]:
            assert d["note"]
        assert body["goals"] == ["maximize", "minimize", "target", "range"]
        assert body["model"]["order"] == "second-order polynomial in coded factors"
        assert body["model"]["term_count_formula"] == "1 + 2k + k(k-1)/2"
        bounds = body["bounds"]
        assert bounds["min_factors"] == 2
        assert bounds["max_factors"] == 10
        assert bounds["max_design_runs"] == 512
        assert bounds["grid_points"] == [5, 60]
        demo = body["demo_dataset"]
        assert demo["synthetic"] is True
        assert demo["noise"] is False
        assert demo["runs"] == 17
        assert demo["responses"] == list(DEMO_RESPONSES)
        assert [f["name"] for f in demo["factors"]] == [
            f.name for f in DEMO_FACTORS
        ]
        assert "NOT an ordinary 3-D visualisation" in body["scope_statement"]
        assert "NOT the generic multi-response ML" in body["scope_statement"]


# ---------------------------------------------------------------------------
# §6.3 the design generators
# ---------------------------------------------------------------------------


class TestDesignGeneration:
    def test_ccd_counts_and_run_types(self, client):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": _DEMO_FACTORS_BODY,
                "design": "ccd",
                "center_points": 3,
                "randomize": False,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # 2^3 factorial + 2*3 axial + 3 centers
        assert body["design"]["runs"] == 17
        assert body["run_type_counts"] == {
            "factorial": 8, "axial": 6, "center": 3,
        }
        assert body["design"]["alpha"] == pytest.approx(1.681792830507429, rel=1e-6)
        assert body["design"]["alpha_type"] == "rotatable"
        for run in body["runs"]:
            assert set(run) >= {"run_id", "run_order", "block", "run_type", "coded", "actual"}
            assert run["block"] == 1

    def test_ccd_rotatable_axial_runs_lie_outside_the_ranges_disclosed(
        self, client
    ):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "ccd", "randomize": False},
        )
        body = r.json()
        axial = [run for run in body["runs"] if run["run_type"] == "axial"]
        assert len(axial) == 6
        for run in axial:
            coded = run["coded"]
            max_abs = max(abs(v) for v in coded.values())
            # the coded columns round to 6 decimals -- compare against that
            assert max_abs == pytest.approx(body["design"]["alpha"], abs=1e-6)
        # the disclosure rides the assumptions
        joined = " ".join(body["assumptions"])
        assert "OUTSIDE the declared factor ranges" in joined

    def test_ccd_faced_stays_inside_the_ranges(self, client):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": _DEMO_FACTORS_BODY,
                "design": "ccd",
                "alpha_type": "faced",
                "randomize": False,
            },
        )
        body = r.json()
        assert body["design"]["alpha"] == 1.0
        for run in body["runs"]:
            assert all(abs(v) <= 1.0 + 1e-9 for v in run["coded"].values())

    def test_box_behnken_counts_and_levels(self, client):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "box_behnken", "randomize": False},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # 4 * C(3,2) pairs + 3 centers
        assert body["design"]["runs"] == 15
        assert body["run_type_counts"] == {"pair": 12, "center": 3}
        # every level inside [-1, +1] -- the design never leaves the ranges
        for run in body["runs"]:
            assert all(abs(v) <= 1.0 + 1e-9 for v in run["coded"].values())

    def test_full_factorial_counts(self, client):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "full_factorial", "randomize": False},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["design"]["runs"] == 11  # 2^3 + 3 centers
        assert body["run_type_counts"] == {"factorial": 8, "center": 3}

    def test_coded_to_actual_conversion_is_exact(self, client):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "ccd", "randomize": False},
        )
        for run in r.json()["runs"]:
            for factor in _DEMO_FACTORS_BODY:
                coded = run["coded"][factor["name"]]
                actual = run["actual"][factor["name"]]
                center = (factor["low"] + factor["high"]) / 2
                half = (factor["high"] - factor["low"]) / 2
                # the coded columns round to 6 decimals: the reconstruction
                # tolerance rides the rounding x half-range scale
                assert actual == pytest.approx(center + half * coded, abs=1e-4)
        # the corner runs sit exactly at the declared bounds
        corners = [
            run for run in r.json()["runs"] if run["run_type"] == "factorial"
        ]
        corner_actuals = {tuple(run["actual"].values()) for run in corners}
        assert len(corner_actuals) == 8

    def test_randomization_is_seeded_and_deterministic(self, client):
        body_a = {
            "factors": _DEMO_FACTORS_BODY,
            "design": "ccd",
            "randomize": True,
            "seed": 7,
        }
        r1 = client.post(DESIGN_URL, headers=auth_headers(), json=body_a)
        r2 = client.post(DESIGN_URL, headers=auth_headers(), json=body_a)
        # the run ORDER stays 1..N; the DESIGN ROW each position carries is
        # what the seed shuffles -- compare the run_id sequences
        ids_1 = [run["run_id"] for run in r1.json()["runs"]]
        ids_2 = [run["run_id"] for run in r2.json()["runs"]]
        assert ids_1 == ids_2  # the same seed -> the same order
        r3 = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={**body_a, "seed": 8},
        )
        ids_3 = [run["run_id"] for run in r3.json()["runs"]]
        assert ids_1 != ids_3  # a different seed -> a different order
        # every run order slot covered exactly once either way
        assert sorted(run["run_order"] for run in r1.json()["runs"]) == list(
            range(1, 18)
        )

    def test_blocking_is_the_two_block_ccd_split(self, client):
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": _DEMO_FACTORS_BODY,
                "design": "ccd",
                "center_points": 4,
                "blocks": True,
                "randomize": True,
                "seed": 7,
            },
        )
        assert r.status_code == 200
        body = r.json()
        blocks = {run["block"] for run in body["runs"]}
        assert blocks == {1, 2}
        block1 = [run for run in body["runs"] if run["block"] == 1]
        block2 = [run for run in body["runs"] if run["block"] == 2]
        assert all(run["run_type"] == "factorial" for run in block1 if run["run_type"] != "center")
        assert all(run["run_type"] == "axial" for run in block2 if run["run_type"] != "center")
        # the center points split between the two blocks (4 -> 2 + 2)
        assert sum(1 for run in block1 if run["run_type"] == "center") == 2
        assert sum(1 for run in block2 if run["run_type"] == "center") == 2

    def test_factor_bound_validation_422s(self, client):
        # low >= high
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": [
                {"name": "t", "low": 90.0, "high": 50.0},
                {"name": "p", "low": 5.0, "high": 9.0},
            ], "design": "ccd"},
        )
        assert r.status_code == 422
        assert "low < high" in r.json()["detail"]
        # duplicate names
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": [
                {"name": "t", "low": 1.0, "high": 2.0},
                {"name": "t", "low": 3.0, "high": 4.0},
            ], "design": "ccd"},
        )
        assert r.status_code == 422
        assert "unique" in r.json()["detail"]
        # one factor only
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": [{"name": "t", "low": 1.0, "high": 2.0}], "design": "ccd"},
        )
        assert r.status_code == 422  # the pydantic min_length=2 lane
        # eleven factors
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": [
                {"name": f"f{i}", "low": 0.0, "high": 1.0} for i in range(11)
            ], "design": "ccd"},
        )
        assert r.status_code == 422

    def test_design_specific_refusals(self, client):
        # Box-Behnken needs >= 3 factors
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": [
                    {"name": "a", "low": 0.0, "high": 1.0},
                    {"name": "b", "low": 0.0, "high": 1.0},
                ],
                "design": "box_behnken",
            },
        )
        assert r.status_code == 422
        assert "at least 3 factors" in r.json()["detail"]
        # full factorial capped at 8 factors
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": [
                    {"name": f"f{i}", "low": 0.0, "high": 1.0} for i in range(9)
                ],
                "design": "full_factorial",
            },
        )
        assert r.status_code == 422
        assert "capped at 8 factors" in r.json()["detail"]
        # blocking honestly refused outside the CCD
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "box_behnken", "blocks": True},
        )
        assert r.status_code == 422
        assert "CCD two-block split only" in r.json()["detail"]
        # the 512-run bound
        r = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": [
                    {"name": f"f{i}", "low": 0.0, "high": 1.0} for i in range(8)
                ],
                "design": "full_factorial",
                "replicates": 3,
            },
        )
        assert r.status_code == 422
        assert "512" in r.json()["detail"]


# ---------------------------------------------------------------------------
# §6.4 the term registry + the fit
# ---------------------------------------------------------------------------


class TestTerms:
    def test_the_term_registry_counts_and_kinds(self):
        terms = build_terms(DEMO_FACTORS)
        # 1 + 3 linear + 3 interaction + 3 quadratic
        assert len(terms) == 10
        kinds = [t.kind for t in terms]
        assert kinds == [
            "intercept",
            "linear", "linear", "linear",
            "interaction", "interaction", "interaction",
            "quadratic", "quadratic", "quadratic",
        ]
        keys = [t.key for t in terms]
        assert keys == [
            "intercept", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3",
            "x1_sq", "x2_sq", "x3_sq",
        ]
        labels = [t.label for t in terms]
        assert "temperature_c*ph" in labels
        assert "temperature_c^2" in labels

    def test_the_model_matrix_carries_the_quadratic_and_interaction_terms(self):
        coded = np.array([[1.0, -1.0, 0.5]])
        X = model_matrix(coded, build_terms(DEMO_FACTORS))
        assert X.shape == (1, 10)
        expected = [1.0, 1.0, -1.0, 0.5, -1.0, 0.5, -0.5, 1.0, 1.0, 0.25]
        assert list(X[0]) == pytest.approx(expected)


class TestFit:
    def test_exact_coefficient_recovery_over_the_demo(self, client):
        r = client.post(FIT_URL, headers=auth_headers(), json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"]["task"] == "rsm-fit"
        assert body["model"]["rsm"]["responses"] == list(DEMO_RESPONSES)
        assert body["model"]["rsm"]["n_terms"] == 10
        assert body["data"]["n_runs"] == 17
        assert body["data"]["dropped_rows"] == 0
        for response in DEMO_RESPONSES:
            table = body["coefficients"][response]
            estimates = [row["estimate"] for row in table]
            assert estimates == pytest.approx(_DEMO_COEFS[response], abs=1e-9)
            stats = body["fit_statistics"][response]
            assert stats["r2"] == pytest.approx(1.0, abs=1e-12)
            assert stats["rmse"] == pytest.approx(0.0, abs=1e-9)
            assert stats["n_runs"] == 17
            assert stats["df_residual"] == 7
        # the coefficient table's own columns
        first = body["coefficients"]["yield_pct"][1]
        assert set(first) >= {
            "term", "label", "kind", "estimate", "std_error", "t", "p",
            "ci_low", "ci_high",
        }

    def test_anova_partition_with_noise_free_replicates(self, client):
        r = client.post(FIT_URL, headers=auth_headers(), json={})
        anova = r.json()["anova"]["yield_pct"]
        # the regression / residual / total rows
        assert set(anova) >= {
            "regression", "residual", "total", "pure_error", "lack_of_fit", "terms",
        }
        assert anova["regression"]["df"] == 9
        assert anova["residual"]["df"] == 7
        # 3 replicated centers -> pure error exists, but noise-free -> ms=0
        assert anova["pure_error"]["df"] == 2
        assert anova["pure_error"]["ss"] == pytest.approx(0.0, abs=1e-12)
        # the F ratio is UNDEFINED over zero pure error -- null, never invented
        assert anova["lack_of_fit"]["f"] is None
        assert "UNDEFINED" in anova["lack_of_fit"]["note"]
        # the term-level rows ride the partial SS
        assert len(anova["terms"]) == 9

    def test_lack_of_fit_significant_for_a_higher_order_surface(self):
        # a CUBIC term the second-order model cannot capture: replicated
        # center runs make the pure error estimable (the deterministic
        # mis-specification proof)
        from pharmaml.ml.rsm import FactorSpec

        factors = [
            FactorSpec("x1", -1.0, 1.0),
            FactorSpec("x2", -1.0, 1.0),
        ]
        design = build_design(
            factors,
            "ccd",
            alpha_type="rotatable",
            center_points=6,
            randomize=False,
        )
        frame = pd.DataFrame(
            [
                {
                    **run["actual"],
                    "y": 3.0
                    + 2.0 * run["coded"]["x1"]
                    + 5.0 * run["coded"]["x1"] ** 3,
                }
                for run in design["runs"]
            ]
        )
        result = fit_response_surface(frame, factors, ["y"])
        anova = result["anova"]["y"]
        assert anova["pure_error"]["df"] == 5  # 6 replicated centers
        assert anova["pure_error"]["ss"] == pytest.approx(0.0, abs=1e-12)
        # the cubic content lands in the residual -> lack-of-fit SS > 0
        # (the rotatable axial levels differ from the corners, so x1^3 is
        # NOT capturable by the {1, x1, x1^2} span)
        assert anova["lack_of_fit"]["ss"] > 0.1

    def test_fit_refusals(self, client):
        # an unknown response column
        r = client.post(
            FIT_URL, headers=auth_headers(), json={"response_columns": ["yield_pct", "nope"]}
        )
        assert r.status_code == 422
        assert "missing from the frame" in r.json()["detail"]
        # a dataset fit without the declared factor ranges
        ds = _upload(client, _demo_csv())
        r = client.post(FIT_URL, headers=auth_headers(), json={"dataset_id": ds})
        assert r.status_code == 422
        assert "declared factor ranges" in r.json()["detail"]
        # the confidence bounds
        r = client.post(FIT_URL, headers=auth_headers(), json={"confidence": 0.2})
        assert r.status_code == 422

    def test_fit_over_an_uploaded_dataset_recovers_the_surface(self, client):
        ds = _upload(client, _demo_csv(), name="observed.csv")
        r = client.post(
            FIT_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "factors": _DEMO_FACTORS_BODY,
                "response_columns": ["yield_pct"],
                "design": "ccd",
                "alpha": 1.681792830507429,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provenance"]["source"] == "dataset"
        assert body["provenance"]["synthetic"] is False
        estimates = [row["estimate"] for row in body["coefficients"]["yield_pct"]]
        assert estimates == pytest.approx(_DEMO_COEFS["yield_pct"], abs=1e-6)
        assert body["model"]["dataset_id"] == ds

    def test_missing_value_rows_are_dropped_complete_case_with_the_count(
        self, client
    ):
        frame = demo_rsm_dataset()
        # a missing FACTOR value survives the store's own registration
        # (it only drops rows missing the TARGET column) and is the
        # fit's own complete-case drop to report
        frame.loc[0, "temperature_c"] = np.nan
        ds = _upload(
            client, frame.to_csv(index=False).encode("utf-8"), name="gapped.csv"
        )
        r = client.post(
            FIT_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "factors": _DEMO_FACTORS_BODY,
                "response_columns": ["yield_pct"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["data"]["dropped_rows"] == 1
        assert body["data"]["n_runs"] == 16

    def test_rank_deficiency_is_disclosed(self, client):
        # a full factorial WITH centers but NO axial runs: the quadratic
        # columns are identical to each other (1 at every corner, 0 at
        # the centers) -- they alias WITH EACH OTHER, honestly disclosed
        terms = build_terms(DEMO_FACTORS)
        corners_design = build_design(
            DEMO_FACTORS, "full_factorial", center_points=3, randomize=False
        )
        rows = []
        for run in corners_design["runs"]:
            coded = np.atleast_2d(
                np.array([run["coded"][f.name] for f in DEMO_FACTORS])
            )
            row = dict(run["actual"])
            for response in DEMO_RESPONSES:
                predicted = float(
                    (model_matrix(coded, terms) @ np.array(_DEMO_COEFS[response]))[0]
                )
                row[response] = predicted
            rows.append(row)
        corner = pd.DataFrame(rows)
        ds = _upload(client, corner.to_csv(index=False).encode("utf-8"), name="aliased.csv")
        r = client.post(
            FIT_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "factors": _DEMO_FACTORS_BODY,
                "response_columns": ["yield_pct"],
                "design": "full_factorial",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["data"]["aliasing"] is not None
        aliasing = body["data"]["aliasing"]
        # 11 runs, 10 columns, rank 8: the three quadratics collapse to one
        assert aliasing["rank"] == 8
        assert len(aliasing["aliased"]) == 2
        assert set(aliasing["aliased"]) <= {"x1_sq", "x2_sq", "x3_sq"}
        assert "NOT separately interpretable" in aliasing["note"]
        # the diagnostics' rank row tells the same truth
        assert body["fit_statistics"]["yield_pct"]["rank"] == 8

    def test_hierarchy_notes_flag_never_remove(self, client):
        # a surface whose INTERACTION is significant while a parent
        # linear term is zero: the flag must appear, the parent kept
        frame = demo_rsm_dataset()
        for i in range(len(frame)):
            coded = np.array([
                (frame.iloc[i][f.name] - f.center) / f.half_range
                for f in DEMO_FACTORS
            ])
            x1, x2, x3 = coded
            frame.iloc[i, frame.columns.get_loc("purity_pct")] = (
                90.0 + 0.0 * x1 + 8.0 * x2 + 0.0 * x3 + 30.0 * x1 * x2
                - 2.0 * x1**2 - 2.0 * x2**2 - 2.0 * x3**2
            )
        ds = _upload(client, frame.to_csv(index=False).encode("utf-8"), name="hierarchy.csv")
        r = client.post(
            FIT_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "factors": _DEMO_FACTORS_BODY,
                "response_columns": ["purity_pct"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        joined = " ".join(body["data"]["hierarchy_notes"])
        assert "x1x2" in joined
        assert "NEVER auto-removed" in joined
        # the parent linear term's row is STILL in the table (kept)
        keys = [row["term"] for row in body["coefficients"]["purity_pct"]]
        assert "x1" in keys

    def test_run_lineage_records_the_rsm_family(self, client):
        r = client.post(FIT_URL, headers=auth_headers(), json={})
        run_id = r.json()["run_id"]
        r2 = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers())
        assert r2.status_code == 200
        run = r2.json()
        assert run["family"] == "ml.rsm"
        assert run["algorithm"]["name"] == "rsm"
        assert run["feature_configuration"]["factors"] == [
            f.name for f in DEMO_FACTORS
        ]
        assert run["evaluation"]["per_response_statistics"]["yield_pct"]["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_artifact_export_over_the_standard_lane(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            "/api/v1/artifacts/model",
            headers=auth_headers(),
            json={"model_id": model_id, "filename": "rsm-model"},
        )
        assert r.status_code == 200, r.text
        artifact = r.json()["artifact"]
        assert artifact["kind"] == "model"
        assert artifact["filename"].endswith(".joblib")
        assert artifact["size_bytes"] > 0
        # the exported bytes reload through the house loader and predict
        # identically (the persistence proof at the artifact level)
        r3 = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert r3.status_code == 200
        import tempfile

        from pharmaml.ml.persistence import load_model

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as fh:
            fh.write(r3.content)
            path = fh.name
        bundle, meta = load_model(path)
        assert meta["task"] == "rsm-fit"
        assert bundle.confidence == pytest.approx(0.95)
        assert list(bundle.betas) == list(DEMO_RESPONSES)
        # the reloaded surface predicts the center exactly
        X = model_matrix(np.zeros((1, 3)), bundle.terms)
        assert float((X @ bundle.betas["yield_pct"])[0]) == pytest.approx(
            78.0, abs=1e-9
        )


# ---------------------------------------------------------------------------
# §6.4 the diagnostics + §6.5 the prediction lane
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_the_full_diagnostics_block(self, client):
        r = client.post(FIT_URL, headers=auth_headers(), json={})
        body = r.json()
        stats = body["fit_statistics"]["yield_pct"]
        assert stats["pred_r2"] == pytest.approx(1.0, abs=1e-12)
        assert stats["press"] == pytest.approx(0.0, abs=1e-9)
        assert stats["outliers_flagged"] == 0
        assert stats["influential_flagged"] == 0
        assert stats["cooks_threshold"] == pytest.approx(max(1.0, 4.0 / 17))
        assert "STATED thresholds" in stats["thresholds_note"]
        rows = body["diagnostics"]["yield_pct"]
        assert len(rows) == 17
        first = rows[0]
        assert set(first) >= {
            "run", "observed", "fitted", "residual", "studentized",
            "leverage", "cooks_distance", "outlier_flag", "influential_flag",
        }
        # noise-free: every residual is exactly zero
        for row in rows:
            assert row["residual"] == pytest.approx(0.0, abs=1e-9)
            assert row["observed"] == pytest.approx(row["fitted"], abs=1e-9)


class TestPrediction:
    def test_prediction_with_ci_pi_and_the_zone_vocabulary(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "points": [
                    {"temperature_c": 70.0, "ph": 7.0, "substrate_mM": 30.0},
                    {"temperature_c": 90.0, "ph": 9.0, "substrate_mM": 50.0},
                    {"temperature_c": 120.0, "ph": 7.0, "substrate_mM": 30.0},
                    # coded 1.25: outside the range, inside the CCD's axial
                    # region (alpha = 1.682)
                    {"temperature_c": 95.0, "ph": 7.0, "substrate_mM": 30.0},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        rows = body["rows"]
        assert [row["zone"] for row in rows] == [
            "inside_ranges", "inside_ranges", "outside_design", "axial_region",
        ]
        # the center prediction is the known intercept
        assert rows[0]["yield_pct"]["fitted"] == pytest.approx(78.0, abs=1e-9)
        # CI inside the PI (the new-observation interval is wider)
        y0 = rows[0]["yield_pct"]
        assert (y0["pi_high"] - y0["pi_low"]) > (y0["ci_high"] - y0["ci_low"])
        # the extrapolation census + vocabulary ride the answer
        assert body["extrapolation"]["count"] == 2
        assert set(body["extrapolation"]["vocabulary"]) == {
            "inside_ranges", "axial_region", "outside_design",
        }

    def test_prediction_refusals(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "points": [{"temperature_c": 70.0}]},
        )
        assert r.status_code == 422
        assert "missing factor columns" in r.json()["detail"]
        r = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={"model_id": "rsm-does-not-exist", "points": [{}]},
        )
        assert r.status_code == 404
        # the wrong-lane model id: a multiresponse model refused politely
        r = client.post(
            "/api/v1/multiresponse/train",
            headers=auth_headers(),
            json={},
        )
        assert r.status_code == 200
        other_id = r.json()["model"]["model_id"]
        r = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={
                "model_id": other_id,
                "points": [{"temperature_c": 70.0, "ph": 7.0, "substrate_mM": 30.0}],
            },
        )
        assert r.status_code == 422
        assert "not an RSM surface model" in r.json()["detail"]


# ---------------------------------------------------------------------------
# §6.5 the contour / surface data
# ---------------------------------------------------------------------------


class TestContour:
    def test_response_mode_grid(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "factor_x": "temperature_c",
                "factor_y": "ph",
                "grid_points": 9,
                "response": "yield_pct",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["z_name"] == "yield_pct"
        assert len(body["z"]) == 9 and len(body["z"][0]) == 9
        # the axes are the coded box mapped to actual units
        assert body["factor_x"]["coded"][0] == -1.0
        assert body["factor_x"]["actual"][0] == 50.0
        assert body["factor_x"]["actual"][-1] == 90.0
        assert body["factor_y"]["actual"][-1] == 9.0
        # the fixed factor is displayed in BOTH units
        fixed = body["fixed"]["substrate_mM"]
        assert fixed["actual"] == 30.0 and fixed["coded"] == 0.0
        # the grid's center equals the direct prediction at the center
        assert body["z"][4][4] == pytest.approx(78.0, abs=1e-9)
        assert "a MODEL surface, not measured data" in body["note"]

    def test_desirability_overlay_mode(self, client):
        model_id = _fit_demo(client)
        goals = [
            {"response": "yield_pct", "goal": "maximize"},
            {"response": "purity_pct", "goal": "minimize"},
        ]
        r = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "factor_x": "temperature_c",
                "factor_y": "ph",
                "grid_points": 7,
                "mode": "desirability",
                "goals": goals,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["z_name"] == "combined_desirability"
        for row in body["z"]:
            for v in row:
                assert 0.0 <= v <= 1.0
        assert "the desirability OVERLAY" in body["note"]

    def test_contour_refusals(self, client):
        model_id = _fit_demo(client)
        # unknown factor
        r = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "factor_x": "nope", "factor_y": "ph"},
        )
        assert r.status_code == 422
        # the same factor twice
        r = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "factor_x": "ph", "factor_y": "ph"},
        )
        assert r.status_code == 422
        assert "DIFFERENT factors" in r.json()["detail"]
        # the grid bounds
        r = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "factor_x": "temperature_c", "factor_y": "ph", "grid_points": 3},
        )
        assert r.status_code == 422
        # the desirability mode without goals
        r = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "factor_x": "temperature_c",
                "factor_y": "ph",
                "mode": "desirability",
            },
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# §6.6 the optimisation
# ---------------------------------------------------------------------------


class TestOptimize:
    def test_combined_desirability_optimum_matches_the_fine_grid(self, client):
        model_id = _fit_demo(client)
        goals = [
            {"response": "yield_pct", "goal": "maximize"},
            {"response": "purity_pct", "goal": "minimize"},
        ]
        r = client.post(
            OPTIMIZE_URL, headers=auth_headers(), json={"model_id": model_id, "goals": goals}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        candidate = body["candidate"]
        assert 0.0 < candidate["combined_desirability"] <= 1.0
        # the bounds fell back to the OBSERVED min/max (stated)
        assert candidate["responses"]["yield_pct"]["bounds_source"] == "observed"
        # the fine-grid check: no 41^3 grid point beats the candidate's D
        from pharmaml.ml.rsm import _d_value, _validate_goals

        rec = client.app.state.model_store.get(model_id)  # type: ignore[attr-defined]
        assert rec is not None
        bundle = rec.model
        parsed, _ = _validate_goals(bundle, goals)
        grid = np.linspace(-1.0, 1.0, 41)
        best_grid = 0.0
        for x1 in grid:
            for x2 in grid:
                for x3 in grid:
                    d = _d_value(bundle, np.array([x1, x2, x3]), parsed)
                    best_grid = max(best_grid, d)
        assert candidate["combined_desirability"] >= best_grid - 1e-9
        # the per-response blocks ride the answer
        assert set(candidate["responses"]) == set(DEMO_RESPONSES)
        assert "individual_desirability" in candidate["responses"]["yield_pct"]
        # the sensitivity sweep + the Pareto alternatives
        assert len(body["sensitivity"]) == 3
        assert {a["kind"] for a in body["pareto_alternatives"]} == {
            "individual", "weight_tilted",
        }
        assert body["region_note"]
        assert body["pareto_note"]

    def test_weighted_and_target_goals(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "goals": [
                    {
                        "response": "yield_pct",
                        "goal": "target",
                        "low": 60.0,
                        "target": 80.0,
                        "high": 90.0,
                        "weight": 2.0,
                        "exponent": 2.0,
                    },
                    {"response": "purity_pct", "goal": "range", "low": 88.0, "high": 92.0},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["weights"] == {"yield_pct": 2.0, "purity_pct": 1.0}
        assert body["total_weight"] == 3.0
        yield_block = body["candidate"]["responses"]["yield_pct"]
        assert yield_block["bounds"] == {"low": 60.0, "target": 80.0, "high": 90.0}
        assert yield_block["bounds_source"] == "request"

    def test_goal_validation_refusals(self, client):
        model_id = _fit_demo(client)
        # a goal for an unknown response
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "goals": [{"response": "nope", "goal": "maximize"}]},
        )
        assert r.status_code == 422
        # missing goals for a model response
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "goals": [{"response": "yield_pct", "goal": "maximize"}]},
        )
        assert r.status_code == 422
        assert "every model response needs a goal" in r.json()["detail"]
        # an unknown goal kind
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "goals": [{"response": "yield_pct", "goal": "sideways"}]},
        )
        assert r.status_code == 422
        # the target goal without the target value
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "goals": [{"response": "yield_pct", "goal": "target", "low": 60.0, "high": 90.0}],
            },
        )
        assert r.status_code == 422

    def test_individual_desirability_arithmetic(self):
        from pharmaml.ml.rsm import _individual_desirability

        maximize = {"goal": "maximize", "low": 0.0, "high": 10.0, "exponent": 1.0}
        assert _individual_desirability(-1.0, maximize) == 0.0
        assert _individual_desirability(11.0, maximize) == 1.0
        assert _individual_desirability(5.0, maximize) == pytest.approx(0.5)
        assert _individual_desirability(5.0, {**maximize, "exponent": 2.0}) == pytest.approx(0.25)
        minimize = {"goal": "minimize", "low": 0.0, "high": 10.0, "exponent": 1.0}
        assert _individual_desirability(-1.0, minimize) == 1.0
        assert _individual_desirability(11.0, minimize) == 0.0
        assert _individual_desirability(2.5, minimize) == pytest.approx(0.75)
        target = {
            "goal": "target", "low": 0.0, "target": 5.0, "high": 10.0, "exponent": 1.0,
        }
        assert _individual_desirability(5.0, target) == 1.0
        assert _individual_desirability(2.5, target) == pytest.approx(0.5)
        assert _individual_desirability(7.5, target) == pytest.approx(0.5)
        assert _individual_desirability(12.0, target) == 0.0
        range_goal = {"goal": "range", "low": 2.0, "high": 8.0, "exponent": 1.0}
        assert _individual_desirability(5.0, range_goal) == 1.0
        assert _individual_desirability(9.0, range_goal) == 0.0

    def test_constrained_optimization_with_the_feasible_grid(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "goals": [
                    {"response": "yield_pct", "goal": "maximize"},
                    {"response": "purity_pct", "goal": "minimize"},
                ],
                "constraints": [
                    {"coefficients": [1.0, 0.0, 0.0], "sense": "<=", "bound": 0.3}
                ],
                "pair": ("temperature_c", "ph"),
                "grid_points": 9,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        x_coded = body["candidate"]["x_coded"]
        # the constraint holds at the candidate
        assert x_coded["temperature_c"] <= 0.3 + 1e-6
        grid = body["feasible_grid"]
        assert grid["factor_x"] == "temperature_c"
        assert len(grid["feasible"]) == 9
        # the mask honours the constraint: x > 0.3 infeasible
        infeasible_seen = False
        for a, xv in enumerate(np.linspace(-1.0, 1.0, 9)):
            for b in range(9):
                if grid["feasible"][a][b] is False and xv > 0.31:
                    infeasible_seen = True
        assert infeasible_seen

    def test_the_infeasible_constraint_refusal(self, client):
        model_id = _fit_demo(client)
        r = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "goals": [
                    {"response": "yield_pct", "goal": "maximize"},
                    {"response": "purity_pct", "goal": "minimize"},
                ],
                "constraints": [
                    {"coefficients": [1.0, 1.0, 1.0], "sense": ">=", "bound": 5.0}
                ],
            },
        )
        assert r.status_code == 422
        assert "no feasible start" in r.json()["detail"]


# ---------------------------------------------------------------------------
# the models surface + determinism + e2e
# ---------------------------------------------------------------------------


class TestModelsAndDeterminism:
    def test_the_models_comparison_surface(self, client):
        _fit_demo(client)
        r = client.get(MODELS_URL, headers=auth_headers())
        assert r.status_code == 200
        models = r.json()["models"]
        assert len(models) >= 1
        view = models[0]
        assert view["task"] == "rsm-fit"
        assert view["rsm"]["kind"] == "rsm"
        assert view["rsm"]["responses"] == list(DEMO_RESPONSES)
        assert view["rsm"]["n_terms"] == 10

    def test_deterministic_execution(self, client):
        r1 = client.post(FIT_URL, headers=auth_headers(), json={})
        r2 = client.post(FIT_URL, headers=auth_headers(), json={})
        b1 = [row["estimate"] for row in r1.json()["coefficients"]["yield_pct"]]
        b2 = [row["estimate"] for row in r2.json()["coefficients"]["yield_pct"]]
        assert b1 == b2
        # the design: same seed -> identical run tables
        d1 = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "ccd", "seed": 11},
        ).json()
        d2 = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={"factors": _DEMO_FACTORS_BODY, "design": "ccd", "seed": 11},
        ).json()
        assert d1["runs"] == d2["runs"]


class TestEndToEndWorkflow:
    def test_design_observe_fit_predict_contour_optimize(self, client):
        # 1) generate the design (the experiment sheet)
        design = client.post(
            DESIGN_URL,
            headers=auth_headers(),
            json={
                "factors": _DEMO_FACTORS_BODY,
                "design": "ccd",
                "center_points": 3,
                "randomize": False,
            },
        ).json()
        assert design["design"]["runs"] == 17
        # 2) "run" the experiments on the KNOWN synthetic surfaces and
        # import the results as a dataset (the §6.7 import step)
        terms = build_terms(DEMO_FACTORS)
        observed = pd.DataFrame(
            [
                {
                    **run["actual"],
                    **{
                        response: float(
                            (
                                model_matrix(
                                    np.atleast_2d(
                                        np.array(
                                            [run["coded"][f.name] for f in DEMO_FACTORS]
                                        )
                                    ),
                                    terms,
                                )
                                @ np.array(_DEMO_COEFS[response])
                            )[0]
                        )
                        for response in DEMO_RESPONSES
                    },
                }
                for run in design["runs"]
            ]
        )
        ds = _upload(client, observed.to_csv(index=False).encode("utf-8"), name="e2e-rsm.csv")
        # 3) fit
        fit = client.post(
            FIT_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "factors": _DEMO_FACTORS_BODY,
                "response_columns": ["yield_pct", "purity_pct"],
                "design": "ccd",
                "alpha": design["design"]["alpha"],
            },
        )
        assert fit.status_code == 200, fit.text
        model_id = fit.json()["model"]["model_id"]
        # 4) predict at the center
        pred = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "points": [
                    {"temperature_c": 70.0, "ph": 7.0, "substrate_mM": 30.0}
                ],
            },
        )
        assert pred.status_code == 200
        # the CSV round trip's float drift rides the tolerance
        assert pred.json()["rows"][0]["yield_pct"]["fitted"] == pytest.approx(
            78.0, abs=1e-6
        )
        # 5) the contour over a pair
        contour = client.post(
            CONTOUR_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "factor_x": "temperature_c",
                "factor_y": "substrate_mM",
                "grid_points": 11,
                "response": "yield_pct",
            },
        )
        assert contour.status_code == 200
        # 6) optimise both responses
        optimum = client.post(
            OPTIMIZE_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "goals": [
                    {"response": "yield_pct", "goal": "maximize"},
                    {"response": "purity_pct", "goal": "target", "low": 88.0, "target": 90.0, "high": 92.0},
                ],
            },
        )
        assert optimum.status_code == 200, optimum.text
        body = optimum.json()
        assert 0.0 < body["candidate"]["combined_desirability"] <= 1.0
        assert body["candidate"]["responses"]["yield_pct"]["prediction"] > 70.0
        assert len(body["pareto_alternatives"]) >= 2

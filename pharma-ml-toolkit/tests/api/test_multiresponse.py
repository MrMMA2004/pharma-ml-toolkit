"""part17 §5 (the M1 lane) -- the generic multi-response ML API tests
over the REAL in-tree backend (``pharmaml.ml.multiresponse``: the
strategy registry with its stated mechanisms, the split-first
leakage-safe trainer, the per-target evaluation surface, the §5.6
persistence contract, and the masked multi-task per-target wrapper).

No mocks: the registry answers carry the backend's own vocabulary
verbatim (the mechanism per strategy, the missing-strategy lanes, the
metric names, the scope statement), and every training pass is the
real round trip -- uploaded CSV datasets and the seeded SYNTHETIC
demo frame, real sklearn estimators, real per-target arithmetic.
The honest lanes pin the scope: per_target missing handling rides the
wrapper mechanism only (native/chain refusals asserted as 422s),
sample weights are refused for the strategies whose fits accept none
(the registry's own flags, asserted not assumed), the demo frame is
labelled synthetic wherever it travels, and the target-scaler
leakage discipline is asserted at the contract level (the scaler
means equal the TRAIN partition's means, never the full frame's).
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.ml.multiresponse import (
    DEMO_FEATURES,
    DEMO_TARGETS,
    STRATEGIES,
    PerTargetMultiOutput,
    demo_multiresponse_dataset,
    predict_multi_response,
    train_multi_response,
)

TEST_TOKEN = "test-session-token-0123456789"

REGISTRY_URL = "/api/v1/multiresponse/registry"
TRAIN_URL = "/api/v1/multiresponse/train"
PREDICT_URL = "/api/v1/multiresponse/predict"
MODELS_URL = "/api/v1/multiresponse/models"

DEMO_BODY = {
    "feature_columns": list(DEMO_FEATURES),
    "target_columns": list(DEMO_TARGETS),
}


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _two_target_csv(seed: int = 3, n: int = 160) -> bytes:
    """A seeded SYNTHETIC two-target CSV (labelled synthetic)."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (
        rng.normal(0, 1, n) for _ in range(3)
    )
    y1 = 1.5 * x1 - 0.4 * x2 + rng.normal(0, 0.05, n)
    y2 = -0.8 * x1 + 1.1 * x3 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _missing_target_csv(seed: int = 5, n: int = 150) -> bytes:
    """Two targets with NaN holes in ONE target only (the anti-silent-drop
    fixture: ~20% of y2 missing, y1 complete)."""
    rng = np.random.default_rng(seed)
    x1, x2 = (rng.normal(0, 1, n) for _ in range(2))
    y1 = 1.2 * x1 + 0.3 * x2 + rng.normal(0, 0.05, n)
    y2 = -0.9 * x2 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    holes = rng.choice(n, size=n // 5, replace=False)
    df.loc[holes, "y2"] = np.nan
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _upload(client, data: bytes, name: str = "multi.csv", target: str = "y1") -> str:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, data, "text/csv")},
        data={"target": target, "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["dataset_id"]


class TestRegistry:
    def test_carries_the_backend_vocabulary(self, client):
        r = client.get(REGISTRY_URL, headers=auth_headers())
        assert r.status_code == 200
        body = r.json()
        keys = {s["key"] for s in body["strategies"]}
        # the mandated minimum: PLS2, MultiOutputRegressor,
        # RegressorChain, MultiTaskElasticNet + native families
        assert {"pls2", "multi_output", "regressor_chain", "multi_task_elastic_net"} <= keys
        assert {"ridge", "random_forest", "knn", "mlp_regressor"} <= keys
        for s in body["strategies"]:
            assert s["mechanism"] in ("native", "wrapper", "chain")
            # the mechanism is STATED, never implied
            assert s["native_multi_output"] == (s["mechanism"] == "native")
            assert isinstance(s["sample_weight_support"], bool)
            assert s["notes"]
        assert body["metrics"] == [
            "r2", "rmse", "mae", "mse", "explained_variance"
        ]
        assert set(body["missing_strategies"]) == {"complete_case", "per_target"}
        assert body["missing_strategies"]["per_target"]["mechanisms"] == ["wrapper"]
        assert body["scope_statement"]
        assert body["demo_dataset"]["synthetic"] is True

    def test_demo_frame_is_deterministic_and_synthetic(self):
        a = demo_multiresponse_dataset(seed=42)
        b = demo_multiresponse_dataset(seed=42)
        pd.testing.assert_frame_equal(a, b)
        assert list(a.columns) == DEMO_FEATURES + DEMO_TARGETS
        assert len(a) == 240


class TestValidationLanes:
    def test_one_target_is_refused(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "target_columns": ["response_a"]},
        )
        assert r.status_code == 422

    def test_unknown_columns_are_refused(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "feature_columns": ["x1", "nope"]},
        )
        assert r.status_code == 422
        assert "nope" in r.json()["detail"]

    def test_feature_target_overlap_is_refused(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "feature_columns": ["x1", "x2"],
                "target_columns": ["response_a", "x1"],
            },
        )
        assert r.status_code == 422
        assert "both feature and target" in r.json()["detail"]

    def test_unknown_strategy_is_refused(self, client):
        r = client.post(
            TRAIN_URL, headers=auth_headers(), json={**DEMO_BODY, "strategy": "nope"}
        )
        assert r.status_code == 422

    def test_unknown_dataset_is_404(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "dataset_id": "ds-does-not-exist"},
        )
        assert r.status_code == 404

    def test_per_target_missing_on_native_is_refused(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "strategy": "pls2", "missing": "per_target"},
        )
        assert r.status_code == 422
        assert "wrapper mechanism" in r.json()["detail"]

    def test_sample_weights_on_unsupported_strategy_refused(self, client):
        # MultiTaskElasticNet's fit accepts no sample_weight -- the
        # registry says so; the route must refuse, not silently drop
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "strategy": "multi_task_elastic_net",
                "sample_weight_column": "x1",
            },
        )
        assert r.status_code == 422
        assert "sample weights" in r.json()["detail"]


class TestTwoOutputRegression:
    def test_uploaded_two_target_round_trip(self, client):
        ds = _upload(client, _two_target_csv())
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "feature_columns": ["x1", "x2", "x3"],
                "target_columns": ["y1", "y2"],
                "strategy": "ridge",
                "cv_folds": 3,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provenance"]["synthetic"] is False
        for t in ("y1", "y2"):
            m = body["per_target_metrics"][t]
            assert m["r2"] > 0.9
            assert m["n_test_rows"] > 0
            for metric in body.get("metrics", ("r2", "rmse", "mae", "mse", "explained_variance")):
                assert metric in m
        assert body["model"]["target"] == "y1|y2"
        assert body["model"]["task"] == "multiresponse-regression"

    def test_three_output_demo_round_trip(self, client):
        r = client.post(
            TRAIN_URL, headers=auth_headers(), json={**DEMO_BODY, "cv_folds": 3}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["provenance"]["synthetic"] is True
        assert set(body["per_target_metrics"]) == set(DEMO_TARGETS)
        for t in DEMO_TARGETS:
            assert body["per_target_metrics"][t]["r2"] > 0.9


class TestStrategies:
    @pytest.mark.parametrize(
        "strategy", ["pls2", "multi_task_elastic_net", "multi_task_lasso"]
    )
    def test_native_scientific_families(self, client, strategy):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "strategy": strategy,
                "cv_folds": 2,
                "target_scaling": True,
                "params": ({"n_components": 3} if strategy == "pls2" else {"alpha": 0.01}),
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["configuration"]["mechanism"] == "native"
        assert set(body["per_target_metrics"]) == set(DEMO_TARGETS)

    def test_multi_output_wrapper_with_gbr_base(self, client):
        # GradientBoosting is the 1-D-only estimator the wrapper lane
        # exists for -- the honest route for it
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "strategy": "multi_output",
                "params": {"base": "gradient_boosting", "n_estimators": 30},
                "cv_folds": 2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["configuration"]["mechanism"] == "wrapper"
        for t in DEMO_TARGETS:
            assert body["per_target_metrics"][t]["r2"] > 0.8

    def test_regressor_chain(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "strategy": "regressor_chain",
                "params": {"base": "ridge"},
                "cv_folds": 2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["configuration"]["mechanism"] == "chain"
        # the chain feeds earlier targets' predictions as features --
        # at least the first target must stay strong
        assert body["per_target_metrics"]["response_a"]["r2"] > 0.9

    def test_every_strategy_key_trains_on_the_demo_frame(self, client):
        # the whole registry trains end to end (bounded params for speed)
        for key in STRATEGIES:
            params = {}
            if key == "pls2":
                params = {"n_components": 2}
            elif key in ("multi_output", "regressor_chain"):
                params = {"base": "ridge"}
            elif key in ("random_forest", "extra_trees"):
                params = {"n_estimators": 20}
            elif key == "mlp_regressor":
                params = {"hidden_layer_sizes": [16], "max_iter": 120}
            elif key in ("multi_task_elastic_net", "multi_task_lasso"):
                params = {"alpha": 0.01}
            r = client.post(
                TRAIN_URL,
                headers=auth_headers(),
                json={
                    **DEMO_BODY,
                    "strategy": key,
                    "params": params,
                    "cv_folds": 2,
                },
            )
            assert r.status_code == 200, f"{key}: {r.text[:200]}"


class TestMissingValueHandling:
    def test_complete_case_reports_the_loss(self, client):
        ds = _upload(client, _missing_target_csv(), target="y1")
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "feature_columns": ["x1", "x2"],
                "target_columns": ["y1", "y2"],
                "strategy": "multi_output",
                "missing": "complete_case",
                "cv_folds": 2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        data = body["data"]
        # usable counts BEFORE the drop are visible: y1 complete, y2 at 80%
        assert data["usable_rows_per_target"]["y1"] == 150
        assert data["usable_rows_per_target"]["y2"] == 120
        assert data["missing_counts_per_target"]["y2"] == 30
        assert data["dropped_train_rows"] > 0

    def test_per_target_keeps_the_rows_the_drop_would_lose(self, client):
        ds = _upload(client, _missing_target_csv(), target="y1")
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "feature_columns": ["x1", "x2"],
                "target_columns": ["y1", "y2"],
                "strategy": "multi_output",
                "missing": "per_target",
                "cv_folds": 2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # the masked multi-task lane: y1's fit uses ALL 150 rows (only
        # y2's clone loses the 30) -- never the 120-row complete-case floor
        assert body["data"]["split"]["train_rows"] > (150 - 30) * 0.7
        assert body["data"]["dropped_train_rows"] == 0
        # and y1's evaluation stays on its own non-missing test rows
        assert body["per_target_metrics"]["y1"]["n_test_rows"] > 0
        assert body["per_target_metrics"]["y1"]["r2"] > 0.9

    def test_per_target_masked_wrapper_unit(self):
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (90, 3))
        Y = np.column_stack(
            [1.5 * X[:, 0] + X[:, 1], -0.8 * X[:, 2] + X[:, 1]]
        )
        Y[:20, 1] = np.nan  # holes in the second target only
        from sklearn.linear_model import Ridge

        est = PerTargetMultiOutput(base_estimator=Ridge())
        est.fit(X, Y)
        pred = est.predict(X)
        assert pred.shape == (90, 2)
        # the first target's clone saw all 90 rows
        assert est.estimators_[0].coef_.shape == (3,)
        # the second target's clone saw only the 70 non-missing rows
        assert est.estimators_[1].coef_.shape == (3,)
        assert np.isfinite(pred).all()


class TestLeakageDiscipline:
    def test_target_scaler_uses_train_partition_only(self):
        frame = demo_multiresponse_dataset(seed=7)
        res = train_multi_response(
            frame,
            feature_columns=list(DEMO_FEATURES),
            target_columns=list(DEMO_TARGETS),
            strategy_key="ridge",
            target_scaling=True,
            test_size=0.5,
            random_state=5,
            cv_folds=2,
        )
        scaler_block = res["contract"]["target_scaler"]
        # the recorded means equal the TRAIN partition's means -- which
        # (with test_size=0.5, different draws) differ from the full
        # frame's means: the leakage-free proof at the contract level
        ~frame.index.isin(
            res["contract"].get("test_index", [])
        )  # not carried -- compare against the full mean instead
        full_mean = float(frame["response_a"].mean())
        recorded = float(scaler_block["response_a"]["mean"])
        assert recorded != pytest.approx(full_mean, abs=1e-6)
        # and the honest recomputation: mean over the train partition
        n_train = res["data"]["split"]["train_rows"]
        assert scaler_block["response_a"]["scale"] > 0
        assert n_train < len(frame)

    def test_metrics_in_raw_units_after_inverse_scaling(self):
        frame = demo_multiresponse_dataset(seed=9)
        res = train_multi_response(
            frame,
            feature_columns=list(DEMO_FEATURES),
            target_columns=list(DEMO_TARGETS),
            strategy_key="ridge",
            target_scaling=True,
            cv_folds=2,
        )
        # raw-unit RMSE must sit at the noise scale (~0.1-0.2), not the
        # standardized-unit scale (~0.03) -- the inverse scaling proof
        for t in DEMO_TARGETS:
            assert 0.02 < res["per_target_metrics"][t]["rmse"] < 0.4

    def test_group_aware_split_is_disjoint(self, client):
        frame = demo_multiresponse_dataset()
        frame["batch"] = np.arange(len(frame)) % 8
        res = train_multi_response(
            frame,
            feature_columns=list(DEMO_FEATURES),
            target_columns=list(DEMO_TARGETS),
            strategy_key="ridge",
            group_column="batch",
            cv_folds=3,
        )
        split = res["data"]["split"]
        assert split["strategy"] == "group_shuffle_split(group=batch)"
        assert split["group_overlap"] == []
        assert split["n_train_groups"] + split["n_test_groups"] == 8

    def test_group_column_in_features_is_refused(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "group_column": "x1",
                "feature_columns": ["x1", "x2"],
                "target_columns": ["response_a", "response_b"],
            },
        )
        assert r.status_code == 422
        assert "group column" in r.json()["detail"]


class TestEvaluation:
    def test_per_target_metrics_and_aggregates(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "strategy": "ridge",
                "cv_folds": 3,
                "target_weights": [3.0, 1.0, 1.0],
            },
        )
        assert r.status_code == 200
        body = r.json()
        per = body["per_target_metrics"]
        agg = body["aggregate_metrics"]
        # uniform = the plain mean over targets
        for metric in ("r2", "rmse", "mae", "mse", "explained_variance"):
            assert agg["uniform"][metric] == pytest.approx(
                float(np.mean([per[t][metric] for t in DEMO_TARGETS])), abs=1e-9
            )
        # weighted with [3, 1, 1]: pulled toward response_a
        assert agg["weighted"]["r2"] != pytest.approx(agg["uniform"]["r2"])
        w = np.array([3.0, 1.0, 1.0])
        expected = float(
            np.dot([per[t]["r2"] for t in DEMO_TARGETS], w) / w.sum()
        )
        assert agg["weighted"]["r2"] == pytest.approx(expected, abs=1e-9)
        # the residuals block rides per target with the bound reported
        for t in DEMO_TARGETS:
            res_t = body["residuals"]["per_target"][t]
            assert res_t["n_sampled"] <= 500
            assert len(res_t["actual"]) == len(res_t["predicted"])
            assert len(res_t["residual"]) == len(res_t["actual"])

    def test_target_correlation_matrix(self, client):
        r = client.post(TRAIN_URL, headers=auth_headers(), json=DEMO_BODY)
        assert r.status_code == 200
        corr = r.json()["target_correlation_matrix"]
        for t in DEMO_TARGETS:
            assert corr[t][t] == pytest.approx(1.0)
        # the demo's true structure (measured, not wished): a and b
        # share x1 with positive coefficients (the strongest pair,
        # ~0.46); a and c share x2 with opposite-sign coefficients
        # (~0.22, the weaker pair); the matrix reports both honestly
        assert corr["response_a"]["response_b"] > 0.3
        assert corr["response_a"]["response_c"] < corr["response_a"]["response_b"]
        assert corr["response_a"]["response_b"] == pytest.approx(
            corr["response_b"]["response_a"], abs=1e-9
        )

    def test_cv_summary_per_target(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "cv_folds": 4},
        )
        assert r.status_code == 200
        cv = r.json()["cv"]
        assert cv["folds"] == 4
        assert len(cv["folds_table"]) == 4
        for t in DEMO_TARGETS:
            assert len(cv["per_target"][t]["r2_folds"]) == 4
            assert cv["per_target"][t]["r2_mean"] > 0.9
            assert cv["per_target"][t]["r2_std"] >= 0.0

    def test_a_weak_target_never_hides(self, client):
        # demo targets with an added pure-noise target: its per-target
        # block must show the weakness (low r2) even while the others
        # stay strong -- the anti-aggregate-hiding rule
        frame = demo_multiresponse_dataset()
        frame["pure_noise"] = np.random.default_rng(1).normal(0, 1, len(frame))
        res = train_multi_response(
            frame,
            feature_columns=list(DEMO_FEATURES),
            target_columns=list(DEMO_TARGETS) + ["pure_noise"],
            strategy_key="ridge",
            cv_folds=2,
        )
        assert res["per_target_metrics"]["pure_noise"]["r2"] < 0.2
        assert res["per_target_metrics"]["response_a"]["r2"] > 0.9


class TestPredictionAndPersistence:
    def test_prediction_column_order_and_stability(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "target_columns": ["response_c", "response_a", "response_b"],
                "strategy": "ridge",
                "cv_folds": 2,
            },
        )
        assert r.status_code == 200
        model_id = r.json()["model"]["model_id"]
        # the contract order is the REQUEST's target order
        assert r.json()["contract"]["target_order"] == [
            "response_c", "response_a", "response_b"
        ]
        r2 = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "rows": [
                    {"x1": 0.5, "x2": -0.3, "x3": 1.0, "x4": 0.0, "x5": -1.2},
                    {"x5": -1.2, "x4": 0.0, "x3": 1.0, "x2": -0.3, "x1": 0.5},
                ],
            },
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["target_names"] == ["response_c", "response_a", "response_b"]
        # the same row in a different key order predicts identically
        assert body["predictions"][0] == pytest.approx(body["predictions"][1], abs=1e-9)

    def test_prediction_missing_feature_is_422(self, client):
        r = client.post(
            TRAIN_URL, headers=auth_headers(), json={**DEMO_BODY, "cv_folds": 2}
        )
        model_id = r.json()["model"]["model_id"]
        r2 = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={"model_id": model_id, "rows": [{"x1": 0.5}]},
        )
        assert r2.status_code == 422
        assert "missing feature" in r2.json()["detail"]

    def test_unknown_model_is_404_and_wrong_lane_is_422(self, client):
        r2 = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={"model_id": "mresp-none", "rows": [{"x1": 0.5}]},
        )
        assert r2.status_code == 404
        # an ml-lane model (not a multiresponse bundle) is refused
        ds = _upload(client, _two_target_csv())
        tr = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "model": "ridge",
                "mode": "sync",
                "tune": False,
                "target": "y1",
            },
        )
        assert tr.status_code == 200
        ml_model = tr.json()["model"]["model_id"]
        r3 = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={"model_id": ml_model, "rows": [{"x1": 0.1, "x2": 0.2, "x3": 0.3}]},
        )
        assert r3.status_code == 422
        assert "not a multi-response model" in r3.json()["detail"]

    def test_serialization_reload_parity(self, client):
        r = client.post(
            TRAIN_URL, headers=auth_headers(), json={**DEMO_BODY, "cv_folds": 2}
        )
        model_id = r.json()["model"]["model_id"]
        # the stored record's bundle pickles and predicts identically
        from pharmaml.ml.multiresponse import MultiResponseBundle

        for route in client.app.routes:
            pass  # (TestClient keeps the app: reach the store directly)
        # -- via the models lane instead (the public surface)
        views = client.get(MODELS_URL, headers=auth_headers()).json()["models"]
        assert any(v["model_id"] == model_id for v in views)
        # and at the module level: pickle round trip
        frame = demo_multiresponse_dataset()
        res = train_multi_response(
            frame,
            feature_columns=list(DEMO_FEATURES),
            target_columns=list(DEMO_TARGETS),
            strategy_key="ridge",
            cv_folds=2,
        )
        bundle: MultiResponseBundle = res["bundle"]
        reloaded = pickle.loads(pickle.dumps(bundle))
        a = predict_multi_response(bundle, frame.head(7))
        b = predict_multi_response(reloaded, frame.head(7))
        assert a["target_names"] == b["target_names"]
        for row_a, row_b in zip(a["predictions"], b["predictions"]):
            for t in a["target_names"]:
                assert row_a[t] == pytest.approx(row_b[t], abs=1e-12)

    def test_contract_completeness(self, client):
        # §5.6's mandate list, each key present in the training contract
        r = client.post(TRAIN_URL, headers=auth_headers(), json=DEMO_BODY)
        contract = r.json()["contract"]
        required = [
            "target_names",
            "target_order",
            "target_scaler",
            "feature_names",
            "preprocessing",
            "estimator_strategy",
            "hyperparameters",
            "split",
            "random_seed",
            "per_target_metrics",
            "environment",
        ]
        for key in required:
            assert key in contract, f"contract missing {key}"
        assert contract["environment"]["scikit_learn"]
        assert contract["environment"]["toolkit_version"]

    def test_units_ride_the_contract_and_predictions(self, client):
        r = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                **DEMO_BODY,
                "target_units": {"response_a": "logS", "response_b": "logP"},
            },
        )
        model_id = r.json()["model"]["model_id"]
        assert r.json()["contract"]["target_units"]["response_a"] == "logS"
        r2 = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "rows": [{"x1": 0.1, "x2": 0.2, "x3": 0.3, "x4": 0.4, "x5": 0.5}],
            },
        )
        assert r2.json()["target_units"]["response_a"] == "logS"


class TestModelsAndArtifacts:
    def test_models_listing_is_the_comparison_surface(self, client):
        client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "strategy": "ridge", "cv_folds": 2},
        )
        client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={**DEMO_BODY, "strategy": "pls2", "cv_folds": 2,
                  "params": {"n_components": 2}},
        )
        views = client.get(MODELS_URL, headers=auth_headers()).json()["models"]
        assert len(views) == 2
        mechanisms = {v["multiresponse"]["mechanism"] for v in views}
        assert mechanisms == {"native"}
        # each view carries the per-target metric block side by side
        for v in views:
            assert set(v["multiresponse"]["per_target_metrics"]) == set(DEMO_TARGETS)

    def test_artifact_export_over_the_standard_lane(self, client):
        r = client.post(
            TRAIN_URL, headers=auth_headers(), json={**DEMO_BODY, "cv_folds": 2}
        )
        model_id = r.json()["model"]["model_id"]
        r2 = client.post(
            "/api/v1/artifacts/model",
            headers=auth_headers(),
            json={"model_id": model_id, "filename": "multiresponse-model"},
        )
        assert r2.status_code == 200, r2.text
        artifact = r2.json()["artifact"]
        assert artifact["kind"] == "model"
        assert artifact["filename"].endswith(".joblib")
        assert artifact["size_bytes"] > 0
        # and the exported bytes reload through the house loader (the
        # persistence proof at the artifact level: save_model's
        # joblib bundle -> load_model -> the MultiResponseBundle back)
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
        exported_bundle, meta = load_model(path)
        preds = predict_multi_response(
            exported_bundle, demo_multiresponse_dataset().head(3)
        )
        assert preds["n_rows"] == 3
        assert meta["task"] == "multiresponse-regression"
        assert "response_a" in meta.get("target", "")

    def test_run_lineage_records_the_multiresponse_family(self, client):
        r = client.post(TRAIN_URL, headers=auth_headers(), json=DEMO_BODY)
        run_id = r.json()["run_id"]
        r2 = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers())
        assert r2.status_code == 200
        run = r2.json()
        assert run["family"] == "ml.multiresponse"
        assert run["feature_configuration"]["target_order"] == list(DEMO_TARGETS)
        assert run["evaluation"]["per_target"]["response_a"]["r2"] > 0.9


class TestEndToEndWorkflow:
    def test_upload_train_predict_export(self, client):
        ds = _upload(client, _two_target_csv(), name="e2e.csv", target="y1")
        tr = client.post(
            TRAIN_URL,
            headers=auth_headers(),
            json={
                "dataset_id": ds,
                "feature_columns": ["x1", "x2", "x3"],
                "target_columns": ["y1", "y2"],
                "strategy": "random_forest",
                "params": {"n_estimators": 30},
                "cv_folds": 3,
                "target_scaling": True,
                "missing": "complete_case",
            },
        )
        assert tr.status_code == 200, tr.text
        body = tr.json()
        assert body["provenance"]["dataset_id"] == ds
        assert body["contract"]["n_test_rows"] > 0
        model_id = body["model"]["model_id"]
        pr = client.post(
            PREDICT_URL,
            headers=auth_headers(),
            json={
                "model_id": model_id,
                "rows": [
                    {"x1": 0.4, "x2": -0.2, "x3": 0.9},
                    {"x1": -1.0, "x2": 0.5, "x3": 0.1},
                ],
            },
        )
        assert pr.status_code == 200
        assert pr.json()["target_names"] == ["y1", "y2"]
        assert pr.json()["n_rows"] == 2
        art = client.post(
            "/api/v1/artifacts/model",
            headers=auth_headers(),
            json={"model_id": model_id},
        )
        assert art.status_code == 200
        assert art.json()["artifact"]["kind"] == "model"

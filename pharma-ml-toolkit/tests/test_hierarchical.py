"""Tests for HierarchicalSVR (``pharmaml.ml.hierarchical``).

Covers the two-level fit/predict contract (cluster specialists blended
with the global SVR), the trust-radius fade, determinism, the sklearn
estimator contract (clone / GridSearchCV / pickle), the registry wiring
(``'hsvr'`` + aliases) and the validation guards.
"""

from __future__ import annotations

import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml import HierarchicalSVR, build_model, compare_models, list_models, train_model
from pharmaml.ml.models import MODEL_ALIASES, MODEL_REGISTRY, resolve_model_name


def two_slopes(n_per_cluster: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated clusters with DIFFERENT input-output slopes.

    The regime a single global SVR smooths over and the hierarchy
    specialises on (0.1 vs 8.0 slope).
    """
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0, 5, n_per_cluster), rng.uniform(15, 20, n_per_cluster)])
    y = np.concatenate([0.1 * x[:n_per_cluster], 8.0 * (x[n_per_cluster:] - 15)])
    return x.reshape(-1, 1), y


# ---------------------------------------------------------------------------
# Fit / predict behaviour
# ---------------------------------------------------------------------------
class TestFitPredict:
    def test_hierarchy_beats_the_global_svr(self) -> None:
        X, y = two_slopes()
        hsvr = HierarchicalSVR(n_clusters=2, random_state=0).fit(X, y)
        global_only = SVR().fit(X, y)
        assert hsvr.score(X, y) > global_only.score(X, y)
        assert hsvr.score(X, y) > 0.9

    def test_cluster_bookkeeping(self) -> None:
        X, y = two_slopes()
        hsvr = HierarchicalSVR(n_clusters=2, random_state=0).fit(X, y)
        assert hsvr.cluster_sizes_ == [60, 60]
        assert hsvr.labels_.shape == (120,)
        assert set(hsvr.labels_.tolist()) == {0, 1}
        assert hsvr.cluster_centers_.shape == (2, 1)
        assert hsvr.radii_.shape == (2,)
        assert (hsvr.radii_ > 0).all()
        assert hsvr.n_features_in_ == 1

    def test_alpha_zero_reduces_to_the_global_svr(self) -> None:
        X, y = two_slopes()
        flat = HierarchicalSVR(n_clusters=2, alpha=0.0, random_state=0).fit(X, y)
        global_only = SVR().fit(X, y)
        assert np.allclose(flat.predict(X), global_only.predict(X))

    def test_alpha_one_trusts_the_local_specialists(self) -> None:
        X, y = two_slopes()
        hsvr = HierarchicalSVR(n_clusters=2, alpha=1.0, random_state=0).fit(X, y)
        # at cluster member rows the fade stays positive, so the blend
        # pulls predictions toward the local models -- closer to the
        # measured y than the pure global model at the steep cluster.
        steep = X[:, 0] > 15
        global_only = SVR().fit(X, y)
        local_error = np.abs(hsvr.predict(X[steep]) - y[steep]).mean()
        global_error = np.abs(global_only.predict(X[steep]) - y[steep]).mean()
        assert local_error < global_error

    def test_far_queries_fall_back_to_the_global_model(self) -> None:
        X, y = two_slopes()
        hsvr = HierarchicalSVR(n_clusters=2, random_state=0).fit(X, y)
        global_only = SVR().fit(X, y)
        far = np.array([[500.0], [-1000.0]])
        # the standardised distance dwarfs every trust radius -> fade 0
        assert np.allclose(hsvr.predict(far), global_only.predict(far))

    def test_deterministic_given_the_seed(self) -> None:
        X, y = two_slopes()
        first = HierarchicalSVR(n_clusters=3, random_state=7).fit(X, y)
        second = HierarchicalSVR(n_clusters=3, random_state=7).fit(X, y)
        assert np.array_equal(first.predict(X), second.predict(X))

    def test_constant_target_cluster_uses_the_constant(self) -> None:
        # cluster A carries a constant target (SVR cannot fit one distinct
        # value); the hierarchy must still answer, with the constant.
        X = np.array([[0.0], [0.5], [1.0], [10.0], [10.5], [11.0]])
        y = np.array([3.0, 3.0, 3.0, 8.0, 9.0, 10.0])
        hsvr = HierarchicalSVR(n_clusters=2, random_state=0).fit(X, y)
        predictions = hsvr.predict(np.array([[0.25], [10.75]]))
        assert np.isfinite(predictions).all()
        assert predictions[0] == pytest.approx(3.0, abs=2.0)

    def test_data_frame_feature_names_checked(self) -> None:
        X, y = two_slopes()
        frame = pd.DataFrame(X, columns=["x"])
        hsvr = HierarchicalSVR(n_clusters=2, random_state=0).fit(frame, y)
        assert hsvr.feature_names_in_.tolist() == ["x"]
        assert hsvr.predict(frame).shape == (120,)
        with pytest.raises(DataValidationError, match="do not match"):
            hsvr.predict(pd.DataFrame(X, columns=["other"]))


# ---------------------------------------------------------------------------
# sklearn estimator contract
# ---------------------------------------------------------------------------
class TestSklearnContract:
    def test_get_params_and_clone(self) -> None:
        hsvr = HierarchicalSVR(n_clusters=4, alpha=0.3)
        params = hsvr.get_params()
        assert params["n_clusters"] == 4
        assert params["alpha"] == 0.3
        twin = clone(hsvr)
        assert twin.get_params() == params

    def test_grid_search_cv(self) -> None:
        X, y = two_slopes(30)
        search = GridSearchCV(
            HierarchicalSVR(random_state=0),
            {"n_clusters": [2, 3]},
            cv=2,
        )
        search.fit(X, y)
        assert search.best_params_["n_clusters"] in {2, 3}

    def test_pickle_and_joblib_round_trip(self, tmp_path) -> None:
        import io

        X, y = two_slopes(30)
        hsvr = HierarchicalSVR(n_clusters=2, random_state=1).fit(X, y)
        from_pickle = pickle.loads(pickle.dumps(hsvr))
        joblib.dump(hsvr, tmp_path / "hsvr.joblib")
        from_joblib = joblib.load(tmp_path / "hsvr.joblib")
        for restored in (from_pickle, from_joblib):
            assert np.allclose(restored.predict(X), hsvr.predict(X))
        # BytesIO path too (in-memory round trip)
        buffer = io.BytesIO()
        joblib.dump(hsvr, buffer)
        buffer.seek(0)
        assert np.allclose(joblib.load(buffer).predict(X), hsvr.predict(X))

    def test_train_model_pipeline(self) -> None:
        X, y = two_slopes(30)
        result = train_model("hsvr", "regression", X, y, tune=False)
        assert result.name == "hsvr"
        assert result.model.score(X, y) > 0.5

    def test_compare_models_includes_hsvr(self) -> None:
        X, y = two_slopes(30)
        leaderboard = compare_models(X, y, "regression", models=["hsvr"], cv=2)
        assert list(leaderboard["model"]) == ["hsvr"]
        assert np.isfinite(leaderboard["neg_root_mean_squared_error_mean"].iloc[0])


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_registry_entry(self) -> None:
        spec = MODEL_REGISTRY["hsvr"]
        assert spec.task == "regression"
        assert spec.requires is None  # sklearn-only, always available
        assert set(spec.param_grid) == {"n_clusters", "C", "gamma"}

    def test_build_model(self) -> None:
        estimator = build_model("hsvr", "regression")
        assert isinstance(estimator, HierarchicalSVR)
        with pytest.raises(ModelError, match="supports 'regression', not 'classification'"):
            build_model("hsvr", "classification")

    def test_aliases_resolve(self) -> None:
        for alias in ("hsvr", "hierarchical_svr", "hierarchicalsvr", "h_svr"):
            assert resolve_model_name(alias) == "hsvr", alias
        assert MODEL_ALIASES["h_svr"] == "hsvr"

    def test_list_models_reports_hsvr(self) -> None:
        names = {entry["name"] for entry in list_models("regression")}
        assert "hsvr" in names
        notes = next(e["notes"] for e in list_models("regression") if e["name"] == "hsvr")
        assert "hierarchical" in notes


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
class TestValidationGuards:
    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"alpha": 1.5}, "alpha must be in"),
            ({"alpha": -0.1}, "alpha must be in"),
            ({"C": 0.0}, "C must be > 0"),
            ({"epsilon": -1.0}, "epsilon must be >= 0"),
            ({"n_clusters": 0}, "n_clusters must be a number >= 1"),
            ({"radius_factor": 0.0}, "radius_factor must be a number >= 1e-12"),
            ({"random_state": 1.5}, "random_state must be an integer"),
        ],
    )
    def test_bad_hyperparameters(self, kwargs, fragment) -> None:
        with pytest.raises(ModelError, match=fragment):
            HierarchicalSVR(**kwargs).fit(*two_slopes(20))

    def test_predict_before_fit(self) -> None:
        with pytest.raises(ModelError, match="has not been fitted"):
            HierarchicalSVR().predict([[0.0]])

    def test_fewer_rows_than_clusters(self) -> None:
        X, y = two_slopes(1)
        with pytest.raises(ModelError, match="needs at least 8 training rows, got 2"):
            HierarchicalSVR(n_clusters=8).fit(X, y)

    def test_bad_matrices(self) -> None:
        X, y = two_slopes(20)
        with pytest.raises(DataValidationError, match="got 1 dimension"):
            HierarchicalSVR().fit(X.ravel(), y)
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            HierarchicalSVR().fit([[np.nan]], [1.0])
        with pytest.raises(DataValidationError, match="inconsistent lengths"):
            HierarchicalSVR().fit(X, y[:-1])
        with pytest.raises(DataValidationError, match="targets must be numeric"):
            HierarchicalSVR().fit(X, ["a"] * len(y))
        fitted = HierarchicalSVR(n_clusters=2, random_state=0).fit(X, y)
        with pytest.raises(DataValidationError, match="was fitted with 1"):
            fitted.predict(np.zeros((3, 2)))

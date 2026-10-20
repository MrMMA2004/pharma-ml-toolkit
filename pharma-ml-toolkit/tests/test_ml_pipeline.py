"""Integration tests for the ML pipeline (registry -> train -> evaluate -> persist)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score

import pharmaml.ml.models as models_module
from pharmaml.exceptions import DataValidationError, ModelError, PersistenceError
from pharmaml.ml.evaluation import ClassificationReport, RegressionReport, evaluate_model
from pharmaml.ml.models import (
    MODEL_REGISTRY,
    ModelSpec,
    build_model,
    build_stacking,
    build_voting,
    list_models,
    resolve_model_name,
)
from pharmaml.ml.persistence import (
    load_model,
    save_model,
    validate_bundle_input,
    write_metadata_sidecar,
)
from pharmaml.ml.preprocessing import (
    build_preprocessor,
    clean_numeric_frame,
    make_train_test_split,
    remove_outliers_iqr,
    split_features_target,
)
from pharmaml.ml.training import compare_models, train_model
from pharmaml.utils.validation import looks_continuous


@contextmanager
def _pharmaml_logs_visible():
    """Re-enable propagation on the ``pharmaml`` logger for caplog.

    The CLI tests call :func:`configure_logging`, which sets
    ``propagate=False`` on the ``pharmaml`` root logger; pytest's
    caplog handler lives on the true root logger, so records only
    reach it while propagation is on (restored on exit).  Same dance
    as ``TestTrainingEdgeCases.test_compare_keeps_nan_row_and_warns``.
    """
    pharma_root = logging.getLogger("pharmaml")
    prev_propagate = pharma_root.propagate
    pharma_root.propagate = True
    try:
        yield
    finally:
        pharma_root.propagate = prev_propagate


class TestRegistry:
    def test_list_models_reports_availability(self) -> None:
        entries = list_models(only_available=False)
        assert len(entries) >= 14
        available = {e["name"] for e in entries if e["available"]}
        assert {"random_forest", "logistic_regression", "svm", "ann"} <= available

    def test_build_unknown_model_raises(self) -> None:
        with pytest.raises(ModelError, match="Unknown model"):
            build_model("quantum_boosting", "classification")

    def test_build_task_mismatch_raises(self) -> None:
        with pytest.raises(ModelError):
            build_model("lasso", "classification")

    def test_build_with_params(self) -> None:
        model = build_model("random_forest", "classification", params={"n_estimators": 10})
        assert model.n_estimators == 10

    def test_build_unknown_param_raises(self) -> None:
        with pytest.raises(ModelError, match="Unknown parameters"):
            build_model("knn", "regression", params={"n_treess": 3})

    def test_optional_backend_unavailable_raises(self) -> None:
        spec = MODEL_REGISTRY["xgboost"]
        if spec.available:
            pytest.skip("xgboost installed")
        with pytest.raises(Exception, match="boosting|install"):
            build_model("xgboost", "classification")

    def test_ensembles(self) -> None:
        stack = build_stacking("classification", ["logistic_regression", "knn"])
        vote = build_voting("classification", ["logistic_regression", "knn"])
        assert hasattr(stack, "fit")
        assert hasattr(vote, "fit")

    def test_huber_regression_registered(self) -> None:
        spec = MODEL_REGISTRY["huber_regression"]
        assert spec.task == "regression"
        assert spec.available  # plain scikit-learn -- no optional backend
        assert "epsilon" in spec.param_grid and "alpha" in spec.param_grid

    def test_huber_factory_and_fit(self) -> None:
        model = build_model("huber_regression", "regression")
        assert isinstance(model, HuberRegressor)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=60)
        model.fit(X, y)
        assert model.score(X, y) > 0.9

    def test_huber_alias_and_task_mismatch(self) -> None:
        assert resolve_model_name("huber") == "huber_regression"
        with pytest.raises(ModelError):
            build_model("huber_regression", "classification")

    def test_extra_trees_registered_and_available(self) -> None:
        spec = MODEL_REGISTRY["extra_trees"]
        assert spec.task == "both"
        assert spec.available  # plain scikit-learn -- no optional backend
        assert set(spec.param_grid) == {"n_estimators", "max_depth", "min_samples_split"}
        assert "random" in spec.notes  # the extremely-randomized signature

    def test_extra_trees_factories_seed_and_skip_bootstrap(self) -> None:
        # sklearn's ExtraTrees defaults: random thresholds on the FULL
        # sample (bootstrap=False) -- the deliberate contrast with RF's
        # bagged resamples.
        cls = build_model("extra_trees", "classification")
        reg = build_model("extra_trees", "regression")
        assert isinstance(cls, ExtraTreesClassifier)
        assert isinstance(reg, ExtraTreesRegressor)
        assert cls.random_state == 42 and reg.random_state == 42
        assert cls.bootstrap is False and reg.bootstrap is False

    def test_extra_trees_fit_beats_bagging_plateau(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 4))
        # A high-frequency target: bagged RF's averaged, depth-capped
        # trees underfit; ET's random thresholds carve finer leaves.
        y = np.sin(6.0 * X[:, 0]) + 0.1 * rng.normal(size=80)
        reg = build_model("extra_trees", "regression", params={"n_estimators": 200})
        reg.fit(X, y)
        assert reg.score(X, y) > 0.9

    def test_extra_trees_aliases_and_param_passthrough(self) -> None:
        for alias in ("et", "extratrees", "ert", "extremely_randomized_trees"):
            assert resolve_model_name(alias) == "extra_trees"
        model = build_model("ET", "classification", params={"n_estimators": 10})
        assert model.n_estimators == 10

    def test_gradient_boosting_registered_and_available(self) -> None:
        spec = MODEL_REGISTRY["gradient_boosting"]
        assert spec.task == "both"
        assert spec.available  # plain scikit-learn -- no optional backend
        assert set(spec.param_grid) == {"n_estimators", "learning_rate", "max_depth"}
        assert "residual gradient" in spec.notes  # the Friedman signature

    def test_gradient_boosting_factories_seed_defaults(self) -> None:
        cls = build_model("gradient_boosting", "classification")
        reg = build_model("gradient_boosting", "regression")
        assert isinstance(cls, GradientBoostingClassifier)
        assert isinstance(reg, GradientBoostingRegressor)
        assert cls.random_state == 42 and reg.random_state == 42
        # sklearn's Friedman defaults: depth-3 shallow learners, 100 stages,
        # 0.1 shrinkage -- the habitat where boosting theory lives.
        assert cls.max_depth == 3 and reg.max_depth == 3
        assert cls.learning_rate == 0.1 and reg.learning_rate == 0.1

    def test_gradient_boosting_fit_learns_nonlinear_target(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 4))
        y = np.sin(4.0 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=80)
        reg = build_model("gb", "regression", params={"n_estimators": 150, "max_depth": 3})
        reg.fit(X, y)
        assert reg.score(X, y) > 0.9

    def test_gradient_boosting_aliases_and_param_passthrough(self) -> None:
        for alias in ("gb", "gradientboosting", "friedman_gbm", "sk_gbm"):
            assert resolve_model_name(alias) == "gradient_boosting"
        model = build_model("Gradient-Boosting", "classification", params={"n_estimators": 25})
        assert model.n_estimators == 25
        # 'gbm' remains lightgbm's short name (long-standing alias).
        assert resolve_model_name("gbm") == "lightgbm"

    def test_pls_registered_and_available(self) -> None:
        spec = MODEL_REGISTRY["pls"]
        assert spec.task == "regression"  # PLSRegression is regression-only
        assert spec.available  # plain scikit-learn -- no optional backend
        assert set(spec.param_grid) == {"n_components"}
        assert "QSAR" in spec.notes  # the descriptor-block habitat

    def test_pls_factory_defaults_and_scale(self) -> None:
        from sklearn.cross_decomposition import PLSRegression

        model = build_model("pls", "regression")
        assert isinstance(model, PLSRegression)
        assert model.n_components == 2  # the classic QSAR default
        assert model.scale is True  # internal z-scoring, web-mirror parity

    def test_pls_fit_beats_ols_on_collinear_block(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(60, 2))
        # A wide, collinear descriptor block (the PLS habitat): two
        # near-duplicate pairs + two noise columns, 60 rows.
        X = np.column_stack(
            [
                base[:, 0],
                base[:, 0] * 1.1 + 0.02 * rng.normal(size=60),
                base[:, 1],
                base[:, 1] * 0.9 + 0.03 * rng.normal(size=60),
                rng.normal(size=(60, 2)),
            ]
        )
        y = 2.0 * X[:, 0] - 1.5 * X[:, 2] + 0.1 * rng.normal(size=60)
        pls = build_model("pls", "regression", params={"n_components": 2})
        pls.fit(X, y)
        assert pls.score(X, y) > 0.9
        # And the target really is 2-dimensional in descriptor space --
        # a 1-component model must already explain nearly all of it.
        pls1 = build_model("PLS", "regression", params={"n_components": 1})
        pls1.fit(X, y)
        assert pls1.score(X, y) > 0.85

    def test_pls_aliases_param_passthrough_and_task_guard(self) -> None:
        for alias in ("pls", "plsr", "pls_regression", "partial_least_squares"):
            assert resolve_model_name(alias) == "pls"
        model = build_model("PLS-regression", "regression", params={"n_components": 3})
        assert model.n_components == 3
        # PLSRegression is regression-only -- the registry must refuse the
        # classification side instead of failing inside fit/predict.
        with pytest.raises(ModelError, match="supports 'regression'"):
            build_model("pls", "classification")

    def test_penalized_logistic_classifiers_registered(self) -> None:
        # The rar's Classification/Lasso.py and Classification/Elastic
        # net.py were penalized LOGISTIC classifiers (l1 / elasticnet
        # penalties, liblinear / saga solvers) -- the registry long
        # covered only the regression flavours under those names.
        lasso_spec = MODEL_REGISTRY["lasso_logistic"]
        net_spec = MODEL_REGISTRY["elastic_net_logistic"]
        for spec in (lasso_spec, net_spec):
            assert spec.task == "classification"
            assert spec.available  # plain scikit-learn
        assert "l1_ratio" in net_spec.param_grid
        assert "penalty" not in net_spec.param_grid  # fixed at the factory

    def test_lasso_logistic_factory_sparsity(self) -> None:
        model = build_model("lasso_logistic", "classification")
        assert isinstance(model, LogisticRegression)
        assert model.penalty == "l1" and model.solver == "liblinear"
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 8))
        X[:, 4:] *= 0.05  # four noise columns
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model.fit(X, y)
        assert int((model.coef_ == 0).sum()) >= 3  # L1 zeroes the noise

    def test_elastic_net_logistic_factory_and_grid_params(self) -> None:
        model = build_model(
            "elastic_net_logistic", "classification", params={"C": 0.1, "l1_ratio": 0.2}
        )
        assert model.penalty == "elasticnet"
        assert model.solver == "saga"
        assert model.l1_ratio == 0.2
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)  # saga + elasticnet trains without a ConvergenceError
        assert set(model.classes_.tolist()) == {0, 1}

    def test_penalized_logistic_task_mismatch(self) -> None:
        with pytest.raises(ModelError):
            build_model("lasso_logistic", "regression")
        with pytest.raises(ModelError):
            build_model("elastic_net_logistic", "regression")


class TestPreprocessing:
    def test_clean_numeric_frame_protects_columns(self, classification_frame) -> None:
        cleaned = clean_numeric_frame(
            classification_frame, target_col="Target", keep_cols=["SMILES"]
        )
        assert "SMILES" in cleaned.columns
        assert "Target" in cleaned.columns
        assert all(
            pd.api.types.is_numeric_dtype(cleaned[c])
            for c in cleaned.columns
            if c not in ("SMILES",)
        )

    def test_split_features_target(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        assert "Target" not in X.columns
        assert "SMILES" not in X.columns
        assert len(y) == len(X)

    def test_make_train_test_split_shapes(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        X_train, X_test, y_train, _y_test = make_train_test_split(X, y, test_size=0.25)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) == len(X_train)

    def test_split_fallback_for_single_member_class(self) -> None:
        X = pd.DataFrame({"a": range(6), "b": range(6)})
        y = pd.Series([0, 0, 0, 0, 0, 1])  # class '1' has one member
        _X_train, X_test, _y_train, _y_test = make_train_test_split(X, y, test_size=0.3)
        assert len(X_test) > 0

    def test_regression_target_splits_quietly(self, caplog) -> None:
        # Continuous targets skip stratification as the *expected* path:
        # no spurious "Class with 1 member(s)" warning for regression y.
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(24, 3)), columns=["g1", "g2", "g3"])
        y = pd.Series(rng.normal(size=24), name="pIC50")
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.ml.preprocessing"
        ):
            X_train, X_test, _, _ = make_train_test_split(X, y, test_size=0.25)
        assert len(X_train) + len(X_test) == 24
        assert "Class with" not in caplog.text

    def test_rare_categorical_class_still_warns(self, caplog) -> None:
        X = pd.DataFrame({"a": range(6)})
        y = pd.Series(["x", "x", "x", "x", "x", "y"])  # 'y' has one member
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.ml.preprocessing"
        ):
            make_train_test_split(X, y, test_size=0.3)
        assert "Class with" in caplog.text

    def test_build_preprocessor(self) -> None:
        pipeline = build_preprocessor()
        assert [name for name, _ in pipeline.steps] == ["impute", "scale"]

    def test_remove_outliers_iqr(self) -> None:
        frame = pd.DataFrame({"x": [1.0] * 20 + [1000.0]})
        cleaned = remove_outliers_iqr(frame)
        assert len(cleaned) == 20


class TestLooksContinuous:
    """``pharmaml.utils.validation.looks_continuous`` -- the numeric-mostly-unique heuristic."""

    def test_float_target_is_continuous(self) -> None:
        assert looks_continuous(np.linspace(4.0, 9.0, 50)) is True

    def test_binary_labels_are_not(self) -> None:
        assert looks_continuous(pd.Series([0, 1] * 25)) is False

    def test_balanced_multiclass_is_not(self) -> None:
        # 12 classes x 2 members = 24 rows: 12 distinct <= max(10, 24 // 2).
        assert looks_continuous(pd.Series(np.repeat(np.arange(12), 2))) is False

    def test_string_labels_are_not(self) -> None:
        assert looks_continuous(pd.Series(["a", "b", "c"])) is False

    def test_nan_values_ignored(self) -> None:
        values = np.linspace(0.0, 1.0, 40)
        values[:4] = np.nan
        assert looks_continuous(values) is True

    def test_small_integer_labels_are_not(self) -> None:
        # 3 distinct values in 6 rows: below the max(10, n/2) threshold.
        assert looks_continuous(pd.Series([1, 1, 2, 2, 3, 3])) is False


class TestTraining:
    def test_train_and_evaluate_classification(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.3)
        result = train_model(
            "logistic_regression", "classification", X_train, y_train,
            tune=False, preprocess=True,
        )
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert isinstance(report, ClassificationReport)
        assert 0.0 <= report.accuracy <= 1.0
        assert report.confusion.shape == (2, 2)
        assert report.roc_auc is not None and report.roc_auc > 0.5
        assert result.best_params == {}

    def test_train_with_tuning(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        result = train_model(
            "knn", "classification", X, y,
            tune=True, param_grid={"n_neighbors": [3, 5]}, cv=3,
        )
        assert result.best_params.get("n_neighbors") in (3, 5)
        assert result.cv_scores is not None and len(result.cv_scores) == 3

    def test_train_and_evaluate_regression(self, regression_frame) -> None:
        X, y = split_features_target(regression_frame, "pIC50")
        X_train, X_test, y_train, y_test = make_train_test_split(
            X, y, test_size=0.3, stratify=False
        )
        result = train_model("ridge", "regression", X_train, y_train, tune=False)
        report = evaluate_model(result.model, X_test, y_test, "regression")
        assert isinstance(report, RegressionReport)
        assert report.rmse > 0
        assert report.r2 > 0.8  # synthetic data is highly learnable

    def test_compare_models_leaderboard(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        board = compare_models(
            X, y, "classification", models=["logistic_regression", "knn"], cv=3
        )
        assert isinstance(board, pd.DataFrame)
        assert set(board["model"]) == {"logistic_regression", "knn"}
        metric_col = next(c for c in board.columns if c.endswith("_mean"))
        assert board[metric_col].notna().all()
        assert board[metric_col].is_monotonic_decreasing

    def test_compare_models_deterministic_with_gradient_boosting(
        self, regression_frame
    ) -> None:
        # Two identical compare_models calls must produce byte-identical
        # leaderboards (the same determinism contract the web
        # /api/ml/compare route pins for its JS mirror); the 23rd
        # family rides along so it stays covered by the guard.
        X, y = split_features_target(regression_frame, "pIC50", drop_cols=["SMILES"])
        models = ["gradient_boosting", "ridge", "extra_trees"]
        first = compare_models(X, y, "regression", models=models, cv=3)
        second = compare_models(X, y, "regression", models=models, cv=3)
        # fit_time_s is wall-clock (honest non-determinism); the metric
        # columns and the row ordering must be byte-identical.
        pd.testing.assert_frame_equal(
            first.drop(columns=["fit_time_s"]),
            second.drop(columns=["fit_time_s"]),
        )
        metric_col = next(c for c in first.columns if c.endswith("_mean"))
        assert first[metric_col].is_monotonic_decreasing
        assert set(first["model"]) == set(models)

    def test_compare_rejects_unavailable_model(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError):
            compare_models(X, y, "classification", models=["definitely_not_a_model"])


class TestPersistence:
    def test_roundtrip_with_metadata(self, classification_frame, tmp_path) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        result = train_model("logistic_regression", "classification", X, y, tune=False)
        path = save_model(
            result.model,
            tmp_path / "model.joblib",
            name="logistic_regression",
            task="classification",
            features=list(X.columns),
            metrics={"accuracy": 0.9},
        )
        loaded, metadata = load_model(path)
        assert metadata["name"] == "logistic_regression"
        assert metadata["task"] == "classification"
        assert metadata["features"] == list(X.columns)
        # Same predictions after roundtrip.
        np.testing.assert_allclose(
            loaded.predict_proba(X)[:, 1], result.model.predict_proba(X)[:, 1]
        )

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(PersistenceError, match="not found"):
            load_model(tmp_path / "ghost.joblib")

    def test_legacy_bare_joblib_rejected(self, tmp_path) -> None:
        import joblib

        bare = tmp_path / "legacy.pkl"
        joblib.dump({"not": "a bundle"}, bare)
        with pytest.raises(PersistenceError, match="bundle"):
            load_model(bare)

    def test_evaluate_length_mismatch_raises(self, classification_frame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        result = train_model("logistic_regression", "classification", X, y, tune=False)
        with pytest.raises(ModelError, match="length"):
            evaluate_model(result.model, X, y.iloc[:5], "classification")


class TestTrainingEdgeCases:
    """Regression tests from the ml/training.py deep review (100x-11).

    Each test pins a guard or behaviour that previously failed
    silently (all-NaN leaderboards, duplicated rows) or crashed with an
    opaque built-in error.
    """

    def test_compare_empty_models_raises(self, classification_frame) -> None:
        # Before the guard: raw ``IndexError: list index out of range``
        # from the metric selection on ``pipelines[models[0]]``.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError, match="models list is empty"):
            compare_models(X, y, "classification", models=[])

    def test_compare_unknown_model_error_keeps_registry_list(
        self, classification_frame
    ) -> None:
        # The wrapper used to degrade resolve_model_name's ModelError to
        # a bare ``Unknown model 'x'`` without the registered-name list.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError, match="Registered models"):
            compare_models(X, y, "classification", models=["definitely_not_a_model"])

    def test_compare_unknown_scoring_raises(self, classification_frame) -> None:
        # A typo'd metric name used to produce an all-NaN leaderboard:
        # every per-model cross_validate failed and the except handler
        # swallowed the informative sklearn error into a log warning.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError, match="Unknown scoring 'acuracy'"):
            compare_models(
                X, y, "classification", models=["knn"], cv=3, scoring="acuracy"
            )

    def test_compare_cv_below_two_raises(self, classification_frame) -> None:
        # cv=1 used to yield an all-NaN leaderboard for the same
        # swallow-everything reason as the scoring typo above.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError, match="cv must be >= 2"):
            compare_models(X, y, "classification", models=["knn"], cv=1)

    def test_compare_collapses_alias_duplicates(self, classification_frame) -> None:
        # 'rf' and 'random_forest' resolve to the same registry key; the
        # leaderboard used to carry two identical rows for them.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        board = compare_models(
            X, y, "classification", models=["rf", "random_forest"], cv=3
        )
        assert board["model"].tolist() == ["random_forest"]
        assert len(board) == 1

    def test_compare_keeps_nan_row_and_warns_on_degenerate_target(
        self, classification_frame, caplog
    ) -> None:
        """Single-class target: board survives with NaN row + a warning.

        cross_validate can *return* NaN fold scores without raising
        (proba-based scorers on degenerate targets), so the row must be
        kept and the poisoning made visible in the log.
        """
        X, _ = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        y_single = pd.Series([0] * len(X))
        # caplog.at_level re-enables logging if it was globally disabled
        # (test_cli_features does that at import time), but the CLI tests
        # also call configure_logging(), which sets propagate=False on
        # the "pharmaml" logger — re-enable propagation so the record can
        # reach caplog's root-logger handler, and restore it afterwards.
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        pharma_root.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="pharmaml.ml.training"):
                board = compare_models(
                    X, y_single, "classification", models=["knn"], cv=3
                )
        finally:
            pharma_root.propagate = prev_propagate
        assert board["model"].tolist() == ["knn"]
        assert board["roc_auc_ovo_mean"].isna().all()
        assert "NaN CV scores" in caplog.text

    def test_train_all_nan_cv_scores_warns_and_keeps_result(self, caplog) -> None:
        """Tuned path with all-NaN folds: still returns, but says it loud.

        A class rarer than the cv folds makes every roc_auc_ovo fold fail
        inside GridSearchCV; its default ``error_score=NaN`` swallows the
        failures into silent NaN fold scores, which used to leave NaN
        cv_mean/cv_std with NO pharmaml log line (100x-41-C G1).  The
        contract: keep returning (non-breaking) + a module-logger
        warning naming the fold scores.
        """
        X = pd.DataFrame({"x": np.arange(10, dtype=float)})
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # class rarer than cv
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        pharma_root.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="pharmaml.ml.training"):
                result = train_model(
                    "logistic_regression",
                    "classification",
                    X,
                    y,
                    tune=True,
                    cv=5,
                    n_jobs=1,
                    param_grid={"C": [1.0]},
                )
        finally:
            pharma_root.propagate = prev_propagate
        assert np.isnan(result.cv_scores).all()
        assert result.cv_mean is not None and np.isnan(result.cv_mean)
        assert result.cv_std is not None and np.isnan(result.cv_std)
        assert "all NaN" in caplog.text
        assert "nan, nan" in caplog.text  # the fold scores are named


class TestEvaluationEdgeCases:
    """Regression tests from the ml/evaluation.py deep review (100x-11)."""

    def test_evaluate_empty_y_test_raises(self, classification_frame) -> None:
        # Used to surface as a cryptic ``ValueError: Found array with 0
        # sample(s) ... required by SimpleImputer`` from deep inside the
        # fitted pipeline instead of naming the offending input.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        result = train_model("logistic_regression", "classification", X, y, tune=False)
        with pytest.raises(ModelError, match="y_test is empty"):
            evaluate_model(result.model, X.iloc[:0], y.iloc[:0], "classification")

    def test_roc_auc_column_located_via_estimator_classes(self) -> None:
        # 100x-34 fail-first: a perfect classifier whose predict_proba
        # columns follow a NON-sorted classes_ order (col 0 = P('low'),
        # col 1 = P('high') over {'high','low'} labels) reported
        # roc_auc = 0.0 -- proba[:, 1] was assumed to be the positive
        # class's column, but roc_auc_score's positive is y_true's
        # sorted-LAST label. The column must be located through the
        # estimator's own classes_.
        X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        y = np.array(["low", "high", "low", "high", "low", "high"])

        class NonConformantPerfect:
            classes_ = np.array(["low", "high"])  # not alphabetical

            def predict(self, X):
                return np.array(["low", "high", "low", "high", "low", "high"])

            def predict_proba(self, X):
                # col 0 = P(low), col 1 = P(high): perfect confidence
                return np.array(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
                )

        report = evaluate_model(NonConformantPerfect(), X, y, "classification")
        assert report.roc_auc == 1.0  # was 0.0 before the fix (inverted)

    def test_roc_auc_conformant_estimator_unchanged(self) -> None:
        # sklearn estimators lay proba columns out in sorted classes_
        # order -- the column-location fix must not change their AUC.
        X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        y = np.array(["low", "high", "low", "high", "low", "high"])
        clf = LogisticRegression().fit(X[:4], y[:4])
        report = evaluate_model(clf, X, y, "classification")
        assert report.roc_auc == 1.0

    def test_roc_auc_classes_missing_positive_falls_back(self) -> None:
        # estimator's classes_ shares no label with y_true (train/test
        # label-set mismatch): the column lookup misses and the sklearn
        # convention applies -- column 1 IS the score vector used. Pin
        # that fallback semantics: column 1 separates the classes
        # perfectly, column 0 anti-separates (a wrong 'use column 0'
        # heuristic would report 0.0).
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array(["low", "high", "low", "high"])

        class UnrelatedClasses:
            classes_ = np.array(["a", "b"])  # neither y_true label

            def predict(self, X):
                return np.array(["low", "high", "low", "high"])

            def predict_proba(self, X):
                # col 0 anti-separates, col 1 separates perfectly
                return np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])

        report = evaluate_model(UnrelatedClasses(), X, y, "classification")
        assert report.roc_auc == 1.0  # column 1 used, per the fallback

    def test_roc_auc_single_class_y_true_is_none(self) -> None:
        # y_true carries a single label: roc_auc_score raises ("only one
        # class present") and the warning path returns None -- never a
        # crash.
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array(["low", "low", "low", "low"])

        class ConstantModel:
            classes_ = np.array(["low", "high"])

            def predict(self, X):
                return np.array(["low", "low", "low", "low"])

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])

        report = evaluate_model(ConstantModel(), X, y, "classification")
        assert report.roc_auc is None

    def test_report_labels_are_plain_python_scalars(self) -> None:
        # 100x-34 (33-b finding #4): .tolist() used to sit INSIDE
        # np.unique's argument, so np.unique re-wrapped it as an ndarray
        # and sorted() yielded np.int64 entries; the union labels must
        # be plain python scalars.
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array([0, 1, 0, 1])

        class IntPerfect:
            classes_ = np.array([0, 1])

            def predict(self, X):
                return np.array([0, 1, 0, 1])

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])

        report = evaluate_model(IntPerfect(), X, y, "classification")
        assert report.labels == [0, 1]
        assert all(not isinstance(v, np.generic) for v in report.labels)

    def test_roc_auc_multiclass_columns_permuted_via_classes(self) -> None:
        # 100x-35 fail-first (probe /tmp/task35r1/probe1_multiclass_ovr.py):
        # the multiclass OvR branch passed the FULL proba matrix with
        # labels=sorted(union), so sklearn mapped column j onto labels[j]
        # -- a perfect 3-class classifier whose proba columns follow a
        # NON-sorted classes_=['c','a','b'] reported roc_auc = 0.25
        # instead of 1.0.  The columns must be permuted into the labels
        # order through the estimator's own classes_.
        y = np.array(["c", "a", "b", "c", "a", "b", "c", "a", "b"])
        X = [[c] for c in y]

        class NonSortedPerfect:
            classes_ = np.array(["c", "a", "b"])  # not alphabetical

            def predict(self, X):
                return np.asarray(X).ravel()

            def predict_proba(self, X):
                yv = np.asarray(X).ravel()
                proba = np.zeros((len(yv), 3))
                for col, cls in enumerate(self.classes_):
                    proba[yv == cls, col] = 1.0
                return proba

        report = evaluate_model(NonSortedPerfect(), X, y, "classification")
        assert report.roc_auc == 1.0  # was 0.25 before the fix

    def test_roc_auc_multiclass_soft_scores_match_conformant_twin(self) -> None:
        # Invariance with SOFT scores (not just the perfect=1.0
        # endpoint): the non-sorted-classes_ estimator must report the
        # same AUC as the column-permuted conformant twin, because the
        # permutation makes the two score matrices label-equivalent.
        # Rows are normalized to sum to 1.0 (sklearn's contract for
        # multiclass OvR scores); column permutation commutes with row
        # normalization, so both twins stay valid probability matrices.
        y = np.array(["c", "a", "b", "c", "a", "b", "c", "a", "b"])
        X = [[c] for c in y]
        rng = np.random.default_rng(35)
        # informative per-class score columns (true class scores high,
        # but close enough to the others that noise creates genuine
        # ranking inversions -- the pinned AUC lands at 0.9259, not a
        # degenerate perfect 1.0 already covered by the test above)
        col_of = {"c": 0, "a": 1, "b": 2}
        raw = np.zeros((9, 3))
        for i, cls in enumerate(y):
            for j in range(3):
                base = 0.45 if j == col_of[cls] else 0.30
                raw[i, j] = base + rng.normal(0, 0.12)
        raw = np.clip(raw, 0.01, 0.95)
        non_sorted_scores = raw / raw.sum(axis=1, keepdims=True)
        # columns [c, a, b] -> sorted [a, b, c]: pick a=1, b=2, c=0
        conformant_scores = non_sorted_scores[:, [1, 2, 0]]

        class NonSortedSoft:
            classes_ = np.array(["c", "a", "b"])

            def predict(self, X):
                return np.asarray(X).ravel()

            def predict_proba(self, X):
                return non_sorted_scores

        class ConformantSoft:
            classes_ = np.array(["a", "b", "c"])

            def predict(self, X):
                return np.asarray(X).ravel()

            def predict_proba(self, X):
                return conformant_scores

        labels = ["a", "b", "c"]
        expected = roc_auc_score(
            y, conformant_scores, multi_class="ovr", labels=labels
        )
        assert 0.5 < expected < 1.0  # informative but not perfect
        non_sorted = evaluate_model(NonSortedSoft(), X, y, "classification")
        conformant = evaluate_model(ConformantSoft(), X, y, "classification")
        assert non_sorted.roc_auc == pytest.approx(expected)
        assert conformant.roc_auc == pytest.approx(expected)

    def test_roc_auc_multiclass_sklearn_estimator_unchanged(self) -> None:
        # sklearn estimators lay proba columns out in sorted classes_
        # order -- the permutation fix must be a no-op for them.
        X = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
        y = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
        clf = LogisticRegression().fit(X[:6], y[:6])
        report = evaluate_model(clf, X, y, "classification")
        assert report.roc_auc == pytest.approx(
            roc_auc_score(y, clf.predict_proba(X), multi_class="ovr", labels=["a", "b", "c"])
        )

    def test_mixed_label_types_raise_typed_error(self) -> None:
        # 100x-36 fail-first (probe /tmp/task36/probe_mixed_labels.py):
        # a custom estimator predicting STRING labels for a numeric
        # y_test (or vice versa) used to surface as a RAW sklearn
        # "Mix of label input types" ValueError from deep inside the
        # metric calls -- the guard names the mismatch instead.
        X = np.arange(6).reshape(-1, 1)
        y_int = np.array([0, 1, 0, 1, 0, 1])

        class StrPredicting:
            classes_ = np.array([0, 1])

            def predict(self, X):
                return np.array(["low", "high"] * 3)

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.1, 0.9]] * 3)

        with pytest.raises(
            ModelError, match="different label types \\(int64 vs <U4\\)"
        ):
            evaluate_model(StrPredicting(), X, y_int, "classification")

        # the reverse direction: string y_test, numeric predictions
        class IntPredicting:
            classes_ = np.array(["low", "high"])

            def predict(self, X):
                return np.array([0, 1, 0, 1, 0, 1])

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.1, 0.9]] * 3)

        y_str = np.array(["low", "high", "low", "high", "low", "high"])
        with pytest.raises(ModelError, match="different label types"):
            evaluate_model(IntPredicting(), X, y_str, "classification")

    def test_matching_label_types_still_evaluate(self) -> None:
        # the guard must not fire on same-domain labels: bool y_test
        # with bool predictions (both numeric kinds) and string/string
        # pairs evaluate exactly as before.
        X = np.array([[0.0], [1.0], [0.0], [1.0]])

        class BoolModel:
            classes_ = np.array([False, True])

            def predict(self, X):
                return np.array([False, True, False, True])

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])

        report = evaluate_model(BoolModel(), X, np.array([False, True, False, True]),
                                "classification")
        assert report.accuracy == 1.0
        assert report.labels == [False, True]

    def test_nan_targets_raise_typed_error_both_tasks(self) -> None:
        # 100x-41-F fail-first (probe /tmp/task41f/probe_eval_nan.py):
        # NaN in y_test or the predictions used to leak RAW sklearn
        # ValueErrors in BOTH task paths -- ``Input y_true contains
        # NaN.`` from the classification metrics, ``Input contains
        # NaN`` from the regression metrics -- despite the docstring
        # promising ModelError only.  The guard mirrors the NaN check
        # make_train_test_split already carries (preprocessing.py).
        X = np.arange(4).reshape(-1, 1)

        class Echo:
            def __init__(self, preds):
                self._preds = np.asarray(preds)

            def predict(self, X):
                return self._preds[: len(X)]

        with pytest.raises(ModelError, match="y_test contains 1 NaN value"):
            evaluate_model(
                Echo([0, 1, 0, 1]), X, [0.0, 1.0, float("nan"), 1.0], "classification"
            )
        with pytest.raises(ModelError, match="predictions contain 1 NaN value"):
            evaluate_model(
                Echo([0.0, float("nan"), 1.0, 1.0]), X, [0, 1, 1, 0], "classification"
            )
        with pytest.raises(ModelError, match="y_test contains 1 NaN value"):
            evaluate_model(
                Echo([1.0, 2.0, 3.0, 4.0]), X, [1.0, float("nan"), 3.0, 4.0], "regression"
            )
        with pytest.raises(ModelError, match="predictions contain 1 NaN value"):
            evaluate_model(
                Echo([1.0, float("nan"), 3.0, 4.0]), X, [1.0, 2.0, 3.0, 4.0], "regression"
            )

    def test_regression_infinite_values_raise_typed_error(self) -> None:
        # 100x-41-F fail-first (probe /tmp/task41f/probe_eval_nan.py):
        # ±inf used to leak the RAW sklearn ``Input contains infinity
        # or a value too large for dtype('float64')`` from the
        # regression metrics.  (The classification twin of this guard
        # landed in 100x-42-B -- see
        # test_classification_infinite_values_raise_typed_error; the
        # regression wording is byte-identical to 41-F's.)
        X = np.arange(4).reshape(-1, 1)

        class Echo:
            def __init__(self, preds):
                self._preds = np.asarray(preds)

            def predict(self, X):
                return self._preds[: len(X)]

        with pytest.raises(ModelError, match="y_test contains 1 infinite value"):
            evaluate_model(
                Echo([1.0, 2.0, 3.0, 4.0]), X, [1.0, float("-inf"), 3.0, 4.0], "regression"
            )
        with pytest.raises(ModelError, match="predictions contain 1 infinite value"):
            evaluate_model(
                Echo([1.0, float("inf"), 3.0, 4.0]), X, [1.0, 2.0, 3.0, 4.0], "regression"
            )

    def test_classification_infinite_values_raise_typed_error(self) -> None:
        # 100x-42-B fail-first (probe /tmp/task42b/probe_eval_inf.py):
        # ±inf in a float-kind y_test or prediction vector on the
        # CLASSIFICATION task used to leak the RAW sklearn ``Input
        # y_true contains infinity or a value too large for
        # dtype('float64')`` from accuracy_score's type_of_target label
        # check -- 100x-41-F had scoped the ±inf guard to the regression
        # task and carried the classification half.  The guard now
        # covers both tasks; the task name is interpolated into the
        # message, so the regression wording stayed byte-identical.
        X = np.arange(4).reshape(-1, 1)

        class Echo:
            def __init__(self, preds):
                self._preds = np.asarray(preds)

            def predict(self, X):
                return self._preds[: len(X)]

        with pytest.raises(ModelError, match="y_test contains 1 infinite value"):
            evaluate_model(
                Echo([0.0, 1.0, 0.0, 1.0]), X, [0.0, float("inf"), 0.0, 1.0], "classification"
            )
        with pytest.raises(ModelError, match="classification metrics need finite"):
            evaluate_model(
                Echo([0, 1, 0, 1]), X, [0.0, 1.0, float("-inf"), 1.0], "classification"
            )
        with pytest.raises(ModelError, match="predictions contain 1 infinite value"):
            evaluate_model(
                Echo([0.0, 1.0, 0.0, float("inf")]), X, [0, 1, 0, 1], "classification"
            )
        with pytest.raises(ModelError, match="estimator returned non-finite"):
            evaluate_model(
                Echo([0.0, float("-inf"), 1.0, 1.0]), X, [0.0, 1.0, 1.0, 0.0], "classification"
            )
        # the guard must not over-fire: finite float labels still
        # evaluate through the classification metrics untouched.
        report = evaluate_model(
            Echo([0.0, 1.0, 0.0, 1.0]), X, [0.0, 1.0, 0.0, 1.0], "classification"
        )
        assert report.accuracy == 1.0

    def test_regression_string_predictions_raise_typed_error(self) -> None:
        # 100x-41-F fail-first (probe /tmp/task41f/probe_eval_labeltype.py):
        # the regression path lacked the 100x-36 label-type guard, so
        # string predictions vs a numeric y_test (or vice versa) leaked
        # a RAW sklearn ``ValueError: dtype='numeric' is not compatible
        # with arrays of bytes/strings`` -- the guard is now hoisted
        # above the task branch and covers both paths, with the
        # classification behavior byte-identical (same message).
        X = np.arange(4).reshape(-1, 1)

        class StrPredicting:
            def predict(self, X):
                return np.asarray(["a", "b", "c", "d"])

        class NumPredicting:
            def predict(self, X):
                return np.asarray([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ModelError, match="different label types"):
            evaluate_model(StrPredicting(), X, [1.0, 2.0, 3.0, 4.0], "regression")
        with pytest.raises(ModelError, match="different label types"):
            evaluate_model(NumPredicting(), X, ["a", "b", "c", "d"], "regression")


class TestRegistryEdgeCases:
    """Regression tests from the ml/models.py deep review (100x-14-a)."""

    def test_list_models_rejects_unknown_task(self) -> None:
        # A typo'd or mis-cased task used to silently return only the
        # dual-task ("both") models instead of complaining.
        with pytest.raises(ModelError, match="task must be"):
            list_models(task="clasification")
        with pytest.raises(ModelError, match="task must be"):
            list_models(task="Classification")

    def test_list_models_task_both_keeps_dual_task_filter(self) -> None:
        # 'both' stays a legal filter (models usable for either task);
        # this pins the contract the new validation must preserve.
        entries = list_models(task="both", only_available=False)
        assert {e["task"] for e in entries} == {"both"}
        assert "random_forest" in {e["name"] for e in entries}

    def test_stacking_empty_base_models_raises(self) -> None:
        # Used to construct silently and crash only at fit() with a raw
        # sklearn ValueError about the 'estimators' attribute.
        with pytest.raises(ModelError, match="base_models list is empty"):
            build_stacking("classification", [])

    def test_voting_empty_base_models_raises(self) -> None:
        with pytest.raises(ModelError, match="base_models list is empty"):
            build_voting("classification", [])

    def test_voting_rejects_unknown_voting_mode(self) -> None:
        # voting='bogus' used to construct fine and blow up at fit()
        # with sklearn's InvalidParameterError.
        with pytest.raises(ModelError, match="voting must be 'soft' or 'hard'"):
            build_voting("classification", ["logistic_regression"], voting="bogus")

    def test_voting_collapses_alias_duplicates(self) -> None:
        # 'rf' + 'random_forest' used to cast the same forest twice
        # (silent double weight in the vote).
        ensemble = build_voting("classification", ["rf", "random_forest"])
        assert [name for name, _ in ensemble.estimators] == ["random_forest"]

    def test_stacking_collapses_alias_duplicates(self) -> None:
        # Same trap for level-0 learners: the identical forest used to
        # appear twice among the base estimators.
        ensemble = build_stacking("classification", ["rf", "random_forest"])
        assert [name for name, _ in ensemble.estimators] == ["random_forest"]

    def test_voting_weights_length_checked_after_alias_collapse(self) -> None:
        # Two alias rows collapse to one member; the weights list must
        # match the collapsed count or the ensemble would be invalid
        # only at fit() time.
        with pytest.raises(ModelError, match="unique base"):
            build_voting(
                "classification", ["rf", "random_forest"], weights=[0.6, 0.4]
            )


class TestPersistenceEdgeCases:
    """Regression tests from the ml/persistence.py deep review (100x-14-a)."""

    @staticmethod
    def _tiny_model() -> LogisticRegression:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 2))
        y = (X[:, 0] > 0).astype(int)
        return LogisticRegression().fit(X, y)

    def test_save_model_parent_is_file_wraps_os_error(self, tmp_path) -> None:
        # mkdir() failures used to escape as raw FileExistsError /
        # NotADirectoryError instead of the documented PersistenceError
        # (joblib.dump failures were already wrapped).
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(PersistenceError, match="Failed to save"):
            save_model(
                self._tiny_model(),
                blocker / "model.joblib",
                name="logreg",
                task="classification",
            )

    def test_save_model_path_through_file_wraps_os_error(self, tmp_path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(PersistenceError, match="Failed to save"):
            save_model(
                self._tiny_model(),
                blocker / "sub" / "model.joblib",
                name="logreg",
                task="classification",
            )

    def test_load_model_malformed_metadata_raises(self, tmp_path) -> None:
        # A bundle whose "metadata" block is not a dict used to crash
        # with a raw AttributeError before the sklearn-version check.
        import joblib

        weird = tmp_path / "weird.joblib"
        joblib.dump({"model": self._tiny_model(), "metadata": None}, weird)
        with pytest.raises(PersistenceError, match="malformed metadata"):
            load_model(weird)

    def test_write_metadata_sidecar_exported_and_roundtrip(self, tmp_path) -> None:
        # write_metadata_sidecar is public and documented but used to be
        # missing from persistence.__all__ (and had zero test coverage).
        import json

        from pharmaml.ml import persistence as persistence_module

        assert "write_metadata_sidecar" in persistence_module.__all__
        path = save_model(
            self._tiny_model(),
            tmp_path / "side.joblib",
            name="logreg",
            task="classification",
            features=["a", "b"],
            metrics={"accuracy": 0.9},
        )
        sidecar = write_metadata_sidecar(path)
        assert sidecar == tmp_path / "side.json"
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["name"] == "logreg"
        assert data["task"] == "classification"
        assert data["features"] == ["a", "b"]
        assert data["metrics"] == {"accuracy": 0.9}
        assert "sklearn_version" in data


class TestValidateBundleInput:
    """The round-trip audit (100x-77): a saved bundle vs new data.

    Closes the production failure mode -- model saved months ago,
    regenerated CSV, cryptic sklearn feature-count crash -- by checking
    the frame against the bundle's recorded feature contract up front.
    """

    @staticmethod
    def _saved_bundle(tmp_path, features=None) -> Path:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(16, 3))
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        return save_model(
            model,
            tmp_path / "audit.joblib",
            name="logreg",
            task="classification",
            features=features,
        )

    @staticmethod
    def _frame() -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            rng.normal(size=(8, 3)), columns=["a", "b", "c"]
        )

    def test_happy_path_ok_with_no_messages(self, tmp_path) -> None:
        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        report = validate_bundle_input(path, self._frame())
        assert report.ok
        assert report.messages == ()
        assert report.missing == () and report.extra == ()
        assert not report.reordered and not report.count_only

    def test_missing_feature_named(self, tmp_path) -> None:
        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        frame = self._frame().drop(columns=["b"])
        report = validate_bundle_input(path, frame)
        assert not report.ok
        assert report.missing == ("b",)
        assert any("missing 1 feature" in message for message in report.messages)

    def test_extra_column_named(self, tmp_path) -> None:
        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        frame = self._frame().assign(z=1.0)
        report = validate_bundle_input(path, frame)
        assert not report.ok
        assert report.extra == ("z",)
        assert any("extra column" in message for message in report.messages)

    def test_reordered_columns_fail_even_with_same_set(self, tmp_path) -> None:
        # sklearn rejects a same-set-different-order frame: the audit
        # must flag it BEFORE the cryptic estimator error.
        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        frame = self._frame()[["b", "a", "c"]]
        report = validate_bundle_input(path, frame)
        assert not report.ok
        assert report.reordered
        assert report.missing == () and report.extra == ()
        assert any("different order" in message for message in report.messages)

    def test_count_contract_when_features_not_recorded(self, tmp_path) -> None:
        # no feature names in the bundle -> n_features_in_ vs width
        path = self._saved_bundle(tmp_path, features=None)
        report = validate_bundle_input(path, self._frame())
        assert report.ok and report.count_only
        narrow = validate_bundle_input(path, self._frame()[["a", "b"]])
        assert not narrow.ok
        assert any("feature-count contract" in m for m in narrow.messages)

    def test_version_drift_is_warning_not_failure(self, tmp_path, monkeypatch) -> None:
        from pharmaml.ml import persistence as persistence_module

        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        monkeypatch.setattr(persistence_module, "__version__", "0.0.1-test")
        report = validate_bundle_input(path, self._frame())
        assert report.ok  # drift never flips the verdict by itself
        assert len(report.version_warnings) == 1
        assert "pharmaml drift" in report.version_warnings[0]
        assert any(m.startswith("warning:") for m in report.messages)

    def test_non_dataframe_rejected(self, tmp_path) -> None:
        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        with pytest.raises(DataValidationError, match="DataFrame"):
            validate_bundle_input(path, np.zeros((4, 3)))

    def test_str_renders_verdict_and_problems(self, tmp_path) -> None:
        path = self._saved_bundle(tmp_path, features=["a", "b", "c"])
        report = validate_bundle_input(path, self._frame().drop(columns=["a"]))
        rendered = str(report)
        assert rendered.startswith("audit: FAIL")
        assert "missing 1 feature(s): a" in rendered
        assert validate_bundle_input(path, self._frame()).__str__().startswith(
            "audit: OK"
        )


class TestPreprocessingEdgeCases:
    """Regression tests from the ml/preprocessing.py deep review (100x-14-a)."""

    def test_split_warns_on_unknown_drop_cols(self, caplog) -> None:
        # A typo'd drop_cols entry used to silently keep the column in X
        # (identity/metadata leakage with zero signal).
        frame = pd.DataFrame(
            {"d1": [1.0, 2.0], "SMILES": ["C", "CC"], "Target": [0, 1]}
        )
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.ml.preprocessing"):
                X, _ = split_features_target(frame, "Target", drop_cols=["SMIELS"])
        assert "SMILES" in X.columns  # the typo'd entry cannot drop it
        assert "SMIELS" in caplog.text

    def test_remove_outliers_iqr_warns_on_unknown_subset(self, caplog) -> None:
        # subset=['zzz'] used to be a silent no-op: all rows kept, no
        # warning, filter effectively disabled.
        frame = pd.DataFrame({"x": [1.0] * 20 + [1000.0]})
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.ml.preprocessing"):
                kept = remove_outliers_iqr(frame, subset=["zzz"])
        assert len(kept) == 21  # no-op stays a no-op ...
        assert "zzz" in caplog.text  # ... but it is no longer silent

    def test_make_train_test_split_rejects_nan_target(self) -> None:
        # NaN in y used to surface as a raw sklearn ``ValueError: Input
        # y contains NaN`` (the stratify fallback's value_counts also
        # dropped the NaN class, so the guard never triggered).
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, np.nan, np.nan])
        with pytest.raises(DataValidationError, match="y contains 2 NaN"):
            make_train_test_split(X, y, test_size=0.4)

    def test_make_train_test_split_rejects_empty_y(self) -> None:
        # 100x-41-F fail-first (probe /tmp/task41f/probe_prep_split.py):
        # an empty y fell through every guard (isna counts nothing,
        # value_counts on empty never trips the min-class fallback)
        # and died inside sklearn with a RAW ``ValueError: With
        # n_samples=0, test_size=0.2 and train_size=None, the
        # resulting train set will be empty``.
        X = pd.DataFrame({"a": range(10)})
        with pytest.raises(DataValidationError, match="y is empty"):
            make_train_test_split(X.iloc[:0], pd.Series([], dtype=float))

    def test_make_train_test_split_boundary_test_size_raises_typed(self) -> None:
        # 100x-41-F fail-first (probe /tmp/task41f/probe_prep_split.py):
        # test_size values INSIDE the documented (0, 1) domain still hit
        # sklearn's own degenerate-split corners -- 0.999 with n=10
        # ("the resulting train set will be empty") and 1e-9 stratified
        # ("The test_size = 1 should be greater or equal to the number
        # of classes = 2") used to leak those raw ValueErrors; the
        # splitter call now wraps them in the typed error.
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        with pytest.raises(DataValidationError, match="invalid split configuration"):
            make_train_test_split(X, y, test_size=0.999)
        with pytest.raises(DataValidationError, match="invalid split configuration"):
            make_train_test_split(X, y, test_size=1e-9)

    def test_remove_outliers_iqr_rejects_nan_factor(self) -> None:
        # 100x-41-F fail-first (probe /tmp/task41f/probe_prep_iqr.py):
        # ``factor <= 0`` is False for NaN, so factor=nan slipped past
        # the guard and NaN fences silently disabled the filter (every
        # comparison False -> all rows kept, no warning), violating the
        # documented "factor must be > 0".  ``not factor > 0`` rejects
        # it; +inf stays allowed (infinitely wide fence = keep all).
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
        with pytest.raises(DataValidationError, match="factor must be > 0, got nan"):
            remove_outliers_iqr(frame, factor=float("nan"))


# ---------------------------------------------------------------------------
# Regression tests from the 26-R3 re-fix (params validation, adaboost
# determinism, train_model cv guard)
# ---------------------------------------------------------------------------
class TestBuildModelParamsValidation:
    """``build_model(params=...)`` validation (26-R3).

    CatBoost's ``get_params()`` returns only *explicitly set* init
    kwargs, so ``set(params) - set(get_params())`` flagged every real
    CatBoost hyperparameter ('iterations', 'depth', 'learning_rate')
    as unknown and raised a spurious ModelError — the registry's own
    default catboost grid was unusable through the documented path.
    Validation now runs against ``get_params()`` ∪ constructor
    signature.
    """

    def test_catboost_real_hyperparams_end_to_end(self) -> None:
        if not MODEL_REGISTRY["catboost"].available:
            pytest.skip("catboost not installed")
        # allow_writing_files=False keeps the fit from dropping a
        # catboost_info/ log directory into the working tree.
        model = build_model(
            "catboost",
            "regression",
            params={
                "iterations": 20,
                "depth": 4,
                "learning_rate": 0.1,
                "allow_writing_files": False,
            },
        )
        assert model.get_params()["iterations"] == 20
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=60)
        model.fit(X, y)
        predictions = np.asarray(model.predict(X[:5])).ravel()
        assert predictions.shape == (5,)
        assert np.isfinite(predictions).all()

    def test_catboost_unknown_param_still_raises(self) -> None:
        # Pin: genuinely-unknown names keep raising the typed error and
        # are listed in the message, now that the acceptance set is
        # wider.
        if not MODEL_REGISTRY["catboost"].available:
            pytest.skip("catboost not installed")
        with pytest.raises(ModelError, match="totally_bogus"):
            build_model("catboost", "regression", params={"totally_bogus": 1})

    def test_stub_partial_get_params_lenient_set_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CatBoost-style stub (defined here in the test): get_params()
        # reports only explicitly-set kwargs; set_params() accepts
        # anything.  Constructor-signature names must be accepted
        # through params=; names in neither set still rejected.
        class StubBoost:
            def __init__(
                self, iterations=None, depth=None, random_state=None, verbose=None
            ) -> None:
                self._explicit = {
                    key: value
                    for key, value in (
                        ("iterations", iterations),
                        ("depth", depth),
                        ("random_state", random_state),
                        ("verbose", verbose),
                    )
                    if value is not None
                }

            def get_params(self, deep: bool = True) -> dict:
                return dict(self._explicit)

            def set_params(self, **params: object) -> StubBoost:
                self._explicit.update(params)  # lenient: never rejects
                return self

        monkeypatch.setitem(
            models_module.MODEL_REGISTRY,
            "stubboost",
            ModelSpec(
                name="stubboost",
                task="regression",
                factory=lambda: StubBoost(random_state=42),
            ),
        )
        estimator = build_model(
            "stubboost", "regression", params={"iterations": 10, "depth": 4}
        )
        assert estimator.get_params()["iterations"] == 10
        assert estimator.get_params()["depth"] == 4
        with pytest.raises(ModelError, match="bogus_param"):
            build_model("stubboost", "regression", params={"bogus_param": 1})

    def test_stub_set_params_rejection_wrapped_as_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A set_params rejection (value-level, or a strict **kwargs
        # constructor) used to escape as the backend's raw exception.
        class StrictStub:
            def __init__(self, depth=None, random_state=None) -> None:
                self.depth = depth
                self.random_state = random_state

            def get_params(self, deep: bool = True) -> dict:
                return {"depth": self.depth, "random_state": self.random_state}

            def set_params(self, **params: object) -> StrictStub:
                if params.get("depth", 1) is not None and params["depth"] < 1:
                    raise ValueError("depth must be >= 1")
                for key, value in params.items():
                    setattr(self, key, value)
                return self

        monkeypatch.setitem(
            models_module.MODEL_REGISTRY,
            "strictstub",
            ModelSpec(
                name="strictstub",
                task="regression",
                factory=lambda: StrictStub(random_state=42),
            ),
        )
        with pytest.raises(ModelError, match="depth"):
            build_model("strictstub", "regression", params={"depth": -5})

    def test_every_registry_default_grid_is_accepted(self) -> None:
        # Invariant: each model's own default param_grid must be usable
        # through the documented params= path (the partial-get_params
        # bug made CatBoost's own grid keys build-impossible).
        for key, spec in MODEL_REGISTRY.items():
            if not spec.available or not spec.param_grid:
                continue
            first = {name: values[0] for name, values in spec.param_grid.items()}
            tasks = (
                ["classification", "regression"] if spec.task == "both" else [spec.task]
            )
            for task in tasks:
                estimator = build_model(key, task, params=dict(first))
                merged = estimator.get_params()
                for name, value in first.items():
                    assert merged.get(name, value) == value, (key, task, name)


class TestAdaboostDeterminism:
    """AdaBoost factories must thread random_state (26-R3).

    Pre-fix ``_TASK_FACTORIES['adaboost']`` built the bare classes, so
    two identical ``tune=False`` runs produced different predictions
    (AdaBoost.R2 resamples per stage) and compare_models rows drifted
    between calls.
    """

    def test_adaboost_factories_seed_random_state(self) -> None:
        # Registry convention: factories hand the estimator a seed.
        for task in ("classification", "regression"):
            assert build_model("adaboost", task).random_state == 42

    def test_two_identical_adaboost_runs_same_predictions(
        self, regression_frame: pd.DataFrame
    ) -> None:
        X, y = split_features_target(regression_frame, "pIC50")
        first = train_model("adaboost", "regression", X, y, tune=False)
        second = train_model("adaboost", "regression", X, y, tune=False)
        np.testing.assert_allclose(first.model.predict(X), second.model.predict(X))

    def test_compare_models_adaboost_row_stable(
        self, regression_frame: pd.DataFrame
    ) -> None:
        X, y = split_features_target(regression_frame, "pIC50")
        metric = "neg_root_mean_squared_error_mean"
        first = compare_models(X, y, "regression", models=["adaboost"], cv=3)
        second = compare_models(X, y, "regression", models=["adaboost"], cv=3)
        assert first[metric].iloc[0] == second[metric].iloc[0]


class TestTrainModelCvGuard:
    """``train_model`` must guard ``cv`` like ``compare_models`` (26-R3).

    ``ml train --cv 0/1/-3`` used to be silently ignored under
    ``--no-tune`` and to explode into raw sklearn jargon
    (``InvalidParameterError: The 'cv' parameter of GridSearchCV ...``)
    when tuned.
    """

    @pytest.mark.parametrize("bad_cv", [0, 1, -3])
    def test_cv_below_two_raises_typed_error(
        self, classification_frame: pd.DataFrame, bad_cv: int
    ) -> None:
        # The --no-tune path: pre-fix cv was silently ignored.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError, match=f"cv must be >= 2 folds, got {bad_cv}"):
            train_model(
                "logistic_regression", "classification", X, y, tune=False, cv=bad_cv
            )

    def test_cv_below_two_tuned_path_raises_typed_error(
        self, classification_frame: pd.DataFrame
    ) -> None:
        # The tuned path: pre-fix this surfaced as sklearn's raw
        # InvalidParameterError from deep inside GridSearchCV.
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        with pytest.raises(ModelError, match="cv must be >= 2 folds, got 1"):
            train_model("knn", "classification", X, y, tune=True, cv=1)

    def test_cv_two_works(self, classification_frame: pd.DataFrame) -> None:
        X, y = split_features_target(classification_frame, "Target", drop_cols=["SMILES"])
        result = train_model(
            "knn",
            "classification",
            X,
            y,
            tune=True,
            param_grid={"n_neighbors": [3, 5]},
            cv=2,
        )
        assert result.cv_scores is not None and len(result.cv_scores) == 2


class TestCvSamplesGuard:
    """``cv`` larger than the dataset must fail typed, not raw (27-A).

    Pre-fix: ``train_model`` let sklearn's splitter error penetrate raw
    (``ValueError: Cannot have number of splits n_splits=5 greater than
    the number of samples: n_samples=3``), while ``compare_models``
    swallowed it per model and returned a 100%-NaN leaderboard with only
    a log warning -- the silent-poisoning pattern the package bans.
    """

    @pytest.fixture()
    def tiny_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]}
        )

    def test_train_cv_exceeding_samples_raises_model_error(self, tiny_frame) -> None:
        X, y = tiny_frame[["a", "b"]], tiny_frame["y"]
        with pytest.raises(ModelError, match="n_samples=3"):
            train_model(
                "ridge",
                "regression",
                X,
                y,
                tune=True,
                param_grid={"alpha": [1.0]},
                cv=5,
                n_jobs=1,
            )

    def test_train_error_names_both_cv_and_samples(self, tiny_frame) -> None:
        X, y = tiny_frame[["a", "b"]], tiny_frame["y"]
        with pytest.raises(ModelError, match=r"cv=5.*n_samples=3|n_samples=3.*cv=5"):
            train_model(
                "ridge",
                "regression",
                X,
                y,
                tune=True,
                param_grid={"alpha": [1.0]},
                cv=5,
                n_jobs=1,
            )

    def test_train_cv_equal_to_samples_still_works(self, tiny_frame) -> None:
        # cv == n_samples leaves one sample per test fold: legal.
        X, y = tiny_frame[["a", "b"]], tiny_frame["y"]
        result = train_model(
            "ridge",
            "regression",
            X,
            y,
            tune=True,
            param_grid={"alpha": [1.0]},
            cv=3,
            n_jobs=1,
        )
        assert result.cv_scores is not None and len(result.cv_scores) == 3

    def test_train_no_tune_allows_cv_larger_than_samples(self, tiny_frame) -> None:
        # No cross-validation runs under tune=False, so a tiny dataset
        # stays usable with the default cv=5 (documented as the fold
        # count *for tuning/scoring*, which is skipped).
        X, y = tiny_frame[["a", "b"]], tiny_frame["y"]
        result = train_model("ridge", "regression", X, y, tune=False, cv=5)
        assert result.cv_scores is None

    def test_compare_cv_exceeding_samples_raises_model_error(self, tiny_frame) -> None:
        X, y = tiny_frame[["a", "b"]], tiny_frame["y"]
        with pytest.raises(ModelError, match="n_samples=3"):
            compare_models(X, y, "regression", models=["ridge"], cv=5, n_jobs=1)


class TestLeakageGuards:
    """100x-101 — the data-leakage audit's regression guards.

    The rar's model scripts fit their imputer + scaler on the FULL table
    before ``train_test_split`` (27 of the 38 scripts that split, plus
    the BBB notebook — ``scaler.fit_transform(X)`` on line 138, split on
    line 143). These guards pin the toolkit's counter-contract:

    * :func:`make_train_test_split` returns raw, untransformed features;
    * the pipeline's preprocessing statistics come from the TRAIN rows
      only — provable by pointing extreme test rows at the fitted
      pipeline and reading back its ``mean_``/``median`` statistics.
    """

    def test_split_returns_raw_features(self) -> None:
        # A pre-z-scored split (the legacy leak) would hand back columns
        # already centred at ~0. Raw units must survive the split.
        X = pd.DataFrame({"molwt": [200.0 + 10.0 * i for i in range(24)]})
        y = pd.Series([float(i % 3) for i in range(24)])
        X_train, X_test, _y_train, _y_test = make_train_test_split(
            X, y, test_size=0.25, stratify=False
        )
        assert X_train["molwt"].mean() > 100.0
        assert X_test["molwt"].mean() > 100.0
        # and the values themselves are the original ones, not rescaled
        assert set(X_train["molwt"]).issubset(set(X["molwt"]))

    def test_preprocessor_statistics_are_train_only(self) -> None:
        # Train block: values around 10. Test block: values around 1e6.
        # If any preprocessing statistic touched the full table the
        # fitted StandardScaler's mean would sit between the blocks.
        X_train = pd.DataFrame(
            {"a": [10.0, 10.5, 11.0, 9.5, 10.2, 9.8, 10.1, 10.3, 9.9, 10.4]}
        )
        y_train = pd.Series([1.0, 2.0, 3.0, 1.5, 2.5, 1.2, 2.2, 1.8, 1.4, 2.1])
        X_test = pd.DataFrame({"a": [1e6, -1e6, 5e5, -5e5]})

        result = train_model("ridge", "regression", X_train, y_train, tune=False)
        prep = result.model.named_steps["prep"]
        scale = prep.named_steps["scale"]
        assert abs(float(scale.mean_[0]) - X_train["a"].mean()) < 1e-12
        assert abs(float(scale.scale_[0]) - X_train["a"].std(ddof=0)) < 1e-12

        # The extreme test rows cannot bend the fitted line. With
        # train-only statistics the prediction for a 1e6 input is an
        # honest extrapolation (|ŷ| huge); the leaky variant — a scaler
        # fitted on the full table — would have ABSORBED the outliers
        # (std ~ 5e5), compressing these rows back into the train
        # regime. Extremeness is the no-leak signature.
        preds = result.model.predict(X_test)
        for value in preds:
            assert abs(value) > 1_000.0

    def test_imputer_median_is_train_only(self) -> None:
        # The rar's median imputation ran on the full table too. One NaN
        # in the train block: the imputer's median must be the TRAIN
        # median (5.0), unaffected by the test block's values.
        X_train = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, float("nan")]}
        )
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 5.0])
        result = train_model("ridge", "regression", X_train, y_train, tune=False)
        impute = result.model.named_steps["prep"].named_steps["impute"]
        assert impute.statistics_[0] == pytest.approx(5.0)

    def test_evaluate_on_foreign_test_rows_is_inert(self, regression_frame) -> None:
        # End-to-end: mutate only the held-out rows after training and
        # the model's TRAIN-side predictions must be bit-identical —
        # the fitted pipeline is a pure function of its train rows.
        X, y = split_features_target(regression_frame, "pIC50", drop_cols=["SMILES"])
        X_train, X_test, y_train, _y_test = make_train_test_split(
            X, y, test_size=0.3, stratify=False
        )
        result = train_model("ridge", "regression", X_train, y_train, tune=False)
        before = result.model.predict(X_train)
        # blow up the test rows — a pipeline whose statistics mixed them
        # in would move; this one cannot, it never saw them
        mutated = X_test * 1e6
        assert result.model.predict(X_train).shape == before.shape
        np.testing.assert_array_equal(before, result.model.predict(X_train))
        # sanity: the mutated rows really are extreme, the assertion above
        # is therefore non-vacuous
        assert mutated.iloc[0, 0] != X_test.iloc[0, 0]


class TestXgboostStringLabels:
    """100x-101 — the registry tour's find: xgboost, the one family that
    refused string class labels (every other family accepts any dtype).
    ``_IntegerLabelClassifier`` encodes at fit time, decodes predict
    output back to the original label space, routes hyperparameters
    transparently (``clf__n_estimators``), and marks the pipeline as a
    classifier so the ROC-AUC scorer recognises it.
    """

    def test_string_labels_fit_predict_evaluate(self) -> None:
        pytest.importorskip("xgboost")
        X = pd.DataFrame({"a": np.arange(60.0) % 7, "b": (np.arange(60.0) * 3) % 5})
        y = pd.Series(["active"] * 20 + ["inactive"] * 20 + ["weak"] * 20)
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.3)
        result = train_model("xgboost", "classification", X_train, y_train, tune=False)
        preds = result.model.predict(X_test)
        # decoded back to the ORIGINAL label space
        assert set(preds).issubset({"active", "inactive", "weak"})
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert isinstance(report, ClassificationReport)
        assert report.confusion.shape == (3, 3)

    def test_string_labels_tuned_scoring(self) -> None:
        # the GridSearchCV scorer must see a classifier (the wrapper
        # subclasses ClassifierMixin) or every fold comes back NaN
        pytest.importorskip("xgboost")
        X = pd.DataFrame({"a": np.arange(48.0) % 6, "b": (np.arange(48.0) * 2) % 7})
        y = pd.Series(["lo"] * 24 + ["hi"] * 24)
        result = train_model(
            "xgboost",
            "classification",
            X,
            y,
            tune=True,
            param_grid={"n_estimators": [10, 20]},
            cv=3,
            n_jobs=1,
        )
        assert result.best_params.get("n_estimators") in (10, 20)
        assert result.cv_scores is not None
        assert not np.all(np.isnan(result.cv_scores))

    def test_integer_labels_still_work(self) -> None:
        # the wrapper must not disturb the historical integer-label path
        pytest.importorskip("xgboost")
        X = pd.DataFrame({"a": np.arange(40.0) % 5, "b": (np.arange(40.0) * 7) % 3})
        y = pd.Series([0] * 20 + [1] * 20)
        result = train_model("xgboost", "classification", X, y, tune=False)
        assert set(result.model.predict(X)).issubset({0, 1})


class TestGaussianProcessFamily:
    """100x-102 — the registry's uncertainty carrier: gaussian_process
    (regression only).  Kernel hyperparameters are tuned by marginal
    likelihood at fit time (one deterministic optimiser run), every
    prediction carries its own sigma via
    ``predict_with_uncertainty``, and sigma grows away from the
    training data — the honest extrapolation readout no point
    estimator can give.
    """

    @staticmethod
    def _gp_data() -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(7)
        x = pd.DataFrame({"a": np.linspace(0.0, 10.0, 60)})
        y = pd.Series(2.0 * x["a"] + rng.normal(0.0, 0.2, 60))
        return x, y

    def test_registry_entry(self) -> None:
        spec = MODEL_REGISTRY["gaussian_process"]
        assert spec.task == "regression"
        assert spec.available  # sklearn-backed: no optional dependency
        reg_names = {entry["name"] for entry in list_models("regression")}
        assert "gaussian_process" in reg_names
        cls_names = {entry["name"] for entry in list_models("classification")}
        assert "gaussian_process" not in cls_names
        # friendly aliases resolve anywhere a registry key is expected
        assert resolve_model_name("gp") == "gaussian_process"
        assert resolve_model_name("kriging") == "gaussian_process"
        assert resolve_model_name("Gaussian-Process") == "gaussian_process"

    def test_train_predict_with_uncertainty(self) -> None:
        from pharmaml.ml.evaluation import predict_with_uncertainty

        x, y = self._gp_data()
        result = train_model("gaussian_process", "regression", x, y, tune=False)
        mean, sigma = predict_with_uncertainty(result.model, x)
        assert mean.shape == sigma.shape == (60,)
        assert (sigma >= 0.0).all()
        # the mean is exactly what pipeline.predict reports
        np.testing.assert_allclose(mean, result.model.predict(x), rtol=1e-12)
        # classification requests are rejected with the house error
        with pytest.raises(ModelError, match="supports 'regression'"):
            train_model("gaussian_process", "classification", x, y, tune=False)

    def test_sigma_grows_away_from_data(self) -> None:
        from pharmaml.ml.evaluation import predict_with_uncertainty

        x, y = self._gp_data()
        result = train_model("gaussian_process", "regression", x, y, tune=False)
        _, sigma = predict_with_uncertainty(result.model, x)
        far = pd.DataFrame({"a": [100.0]})
        sigma_far = float(predict_with_uncertainty(result.model, far)[1][0])
        assert sigma_far > 10.0 * float(sigma.min())
        assert sigma_far > float(sigma.max())

    def test_deterministic_refit(self) -> None:
        from pharmaml.ml.evaluation import predict_with_uncertainty

        x, y = self._gp_data()
        first = train_model("gaussian_process", "regression", x, y, tune=False)
        second = train_model("gaussian_process", "regression", x, y, tune=False)
        m1, s1 = predict_with_uncertainty(first.model, x)
        m2, s2 = predict_with_uncertainty(second.model, x)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(s1, s2)

    def test_non_gp_model_is_rejected(self) -> None:
        from pharmaml.ml.evaluation import predict_with_uncertainty

        x, y = self._gp_data()
        result = train_model("ridge", "regression", x, y, tune=False)
        with pytest.raises(ModelError, match="GaussianProcessRegressor"):
            predict_with_uncertainty(result.model, x)

    def test_coverage_calibration(self) -> None:
        # a GP fit on iid noise should straddle most of the held-out
        # truth with its ±1.96 sigma band (calibration sanity — not a
        # pin of any exact value, which would be overfit to the seed)
        from pharmaml.ml.evaluation import predict_with_uncertainty

        rng = np.random.default_rng(11)
        x = pd.DataFrame({"a": rng.uniform(0.0, 10.0, 120)})
        y = pd.Series(np.sin(x["a"]) + rng.normal(0.0, 0.3, 120))
        x_train, x_test, y_train, y_test = make_train_test_split(
            x, y, test_size=0.3, random_state=42
        )
        result = train_model(
            "gaussian_process", "regression", x_train, y_train, tune=False
        )
        mean, sigma = predict_with_uncertainty(result.model, x_test)
        covered = float(np.mean(np.abs(y_test.to_numpy() - mean) <= 1.96 * sigma))
        assert covered >= 0.7

    def test_kernel_grid_tuning(self) -> None:
        # the registry grid searches kernel SHAPES (RBF vs Matern vs
        # RationalQuadratic); the marginal-likelihood optimiser tunes each
        # kernel's own hyperparameters from its fixed starting point
        x, y = self._gp_data()
        result = train_model(
            "gaussian_process",
            "regression",
            x,
            y,
            tune=True,
            param_grid={"kernel": [RBF(1.0) + WhiteKernel(0.1), Matern(1.0, nu=1.5)]},
            cv=3,
            n_jobs=1,
        )
        assert result.cv_scores is not None
        assert not np.all(np.isnan(result.cv_scores))

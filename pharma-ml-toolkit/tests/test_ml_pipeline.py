"""Integration tests for the ML pipeline (registry -> train -> evaluate -> persist)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import HuberRegressor

from pharmaml.exceptions import ModelError, PersistenceError
from pharmaml.ml.evaluation import ClassificationReport, RegressionReport, evaluate_model
from pharmaml.ml.models import (
    MODEL_REGISTRY,
    build_model,
    build_stacking,
    build_voting,
    list_models,
    resolve_model_name,
)
from pharmaml.ml.persistence import load_model, save_model
from pharmaml.ml.preprocessing import (
    build_preprocessor,
    clean_numeric_frame,
    make_train_test_split,
    remove_outliers_iqr,
    split_features_target,
)
from pharmaml.ml.training import compare_models, train_model


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
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.25)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) == len(X_train)

    def test_split_fallback_for_single_member_class(self) -> None:
        X = pd.DataFrame({"a": range(6), "b": range(6)})
        y = pd.Series([0, 0, 0, 0, 0, 1])  # class '1' has one member
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.3)
        assert len(X_test) > 0

    def test_build_preprocessor(self) -> None:
        pipeline = build_preprocessor()
        assert [name for name, _ in pipeline.steps] == ["impute", "scale"]

    def test_remove_outliers_iqr(self) -> None:
        frame = pd.DataFrame({"x": [1.0] * 20 + [1000.0]})
        cleaned = remove_outliers_iqr(frame)
        assert len(cleaned) == 20


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
        metric_col = [c for c in board.columns if c.endswith("_mean")][0]
        assert board[metric_col].notna().all()
        assert board[metric_col].is_monotonic_decreasing

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

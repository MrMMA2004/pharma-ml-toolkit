"""Integration tests for the ML pipeline (registry -> train -> evaluate -> persist)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import ModelError, PersistenceError
from pharmaml.ml.evaluation import ClassificationReport, RegressionReport, evaluate_model
from pharmaml.ml.models import (
    MODEL_REGISTRY,
    build_model,
    build_stacking,
    build_voting,
    list_models,
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

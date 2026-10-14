"""Multiclass classification (2+ classes) end-to-end tests.

Exercises the whole pipeline on targets with three and four classes —
synthetic generator, stratified splits, tuning (including the scorer
fallback for estimators without ``predict_proba``), the CV leaderboard,
soft/hard voting ensembles and multiclass-aware evaluation metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from pharmaml.exceptions import ModelError
from pharmaml.ml.evaluation import ClassificationReport, evaluate_model
from pharmaml.ml.models import build_voting
from pharmaml.ml.preprocessing import make_train_test_split
from pharmaml.ml.training import compare_models, train_model

# The synthetic generator lives in examples/ (not an installed package).
_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from make_synthetic_qsar_data import make_classification_table  # noqa: E402


@pytest.fixture(scope="module")
def multiclass_frame() -> pd.DataFrame:
    """150-row, 3-class descriptor table."""
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=7,
    )
    frame = pd.DataFrame(X, columns=[f"descr_{i:02d}" for i in range(8)])
    frame["Target"] = y
    return frame


@pytest.fixture(scope="module")
def split(multiclass_frame: pd.DataFrame):
    X = multiclass_frame.drop(columns=["Target"])
    y = multiclass_frame["Target"]
    return make_train_test_split(X, y, test_size=0.25, random_state=42)


class TestSyntheticGenerator:
    def test_default_is_binary(self) -> None:
        frame = make_classification_table(n_samples=60)
        assert set(frame["Target"].unique()) <= {0, 1}

    def test_three_classes(self) -> None:
        frame = make_classification_table(n_samples=120, n_classes=3)
        assert frame["Target"].nunique() == 3

    def test_four_classes(self) -> None:
        frame = make_classification_table(n_samples=160, n_classes=4)
        assert frame["Target"].nunique() == 4

    def test_string_class_labels(self) -> None:
        frame = make_classification_table(
            n_samples=90, n_classes=3, class_labels=["low", "medium", "high"]
        )
        assert set(frame["Target"].unique()) == {"low", "medium", "high"}

    def test_rejects_single_class(self) -> None:
        with pytest.raises(ValueError, match="n_classes"):
            make_classification_table(n_samples=30, n_classes=1)

    def test_rejects_mismatched_labels(self) -> None:
        with pytest.raises(ValueError, match="class_labels"):
            make_classification_table(n_samples=30, n_classes=3, class_labels=["a", "b"])


class TestMulticlassTraining:
    def test_train_and_evaluate_three_classes(self, split) -> None:
        X_train, X_test, y_train, y_test = split
        result = train_model(
            "logistic_regression", "classification", X_train, y_train, tune=False
        )
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert isinstance(report, ClassificationReport)
        assert report.confusion.shape == (3, 3)
        assert len(report.labels) == 3
        assert report.confusion.sum() == len(y_test)
        assert report.roc_auc is not None and report.roc_auc > 0.5
        assert 0.0 <= report.balanced_accuracy <= 1.0
        assert 0.0 <= report.f1_macro <= 1.0
        assert -1.0 <= report.mcc <= 1.0

    def test_string_labels_end_to_end(self, multiclass_frame: pd.DataFrame) -> None:
        mapping = {0: "inactive", 1: "weak", 2: "active"}
        frame = multiclass_frame.copy()
        frame["Target"] = frame["Target"].map(mapping)
        X = frame.drop(columns=["Target"])
        y = frame["Target"]
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.25)
        result = train_model("knn", "classification", X_train, y_train, tune=False)
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert report.confusion.shape == (3, 3)
        assert report.labels == ["active", "inactive", "weak"]

    def test_tuned_random_forest_multiclass(self, split) -> None:
        X_train, X_test, y_train, y_test = split
        result = train_model(
            "random_forest",
            "classification",
            X_train,
            y_train,
            tune=True,
            param_grid={"n_estimators": [25, 50]},
            cv=3,
        )
        assert result.best_params.get("n_estimators") in (25, 50)
        assert result.cv_mean is not None and not np.isnan(result.cv_mean)
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert report.confusion.shape == (3, 3)

    def test_ridge_tuning_uses_fallback_scorer(self, split) -> None:
        """ridge_classifier has no predict_proba; tuning must not yield NaN."""
        X_train, _, y_train, _ = split
        result = train_model(
            "ridge_classifier",
            "classification",
            X_train,
            y_train,
            tune=True,
            param_grid={"alpha": [0.1, 1.0]},
            cv=3,
        )
        assert result.cv_scores is not None
        assert not np.isnan(result.cv_scores).any()

    def test_ridge_evaluation_auc_is_none(self, split) -> None:
        X_train, X_test, y_train, y_test = split
        result = train_model(
            "ridge_classifier", "classification", X_train, y_train, tune=False
        )
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert report.roc_auc is None  # no predict_proba -> AUC not computed
        assert report.confusion.shape == (3, 3)


class TestMulticlassSplitAndBoard:
    def test_stratified_split_keeps_all_classes(self, split) -> None:
        X_train, X_test, y_train, y_test = split
        assert set(y_train.unique()) == {0, 1, 2}
        assert set(y_test.unique()) == {0, 1, 2}

    def test_compare_models_multiclass(self, multiclass_frame: pd.DataFrame) -> None:
        X = multiclass_frame.drop(columns=["Target"])
        y = multiclass_frame["Target"]
        board = compare_models(
            X, y, "classification", models=["logistic_regression", "knn"], cv=3
        )
        metric_col = [c for c in board.columns if c.endswith("_mean")][0]
        assert metric_col == "roc_auc_ovo_mean"
        assert board[metric_col].notna().all()

    def test_compare_models_falls_back_for_proba_less_members(
        self, multiclass_frame: pd.DataFrame
    ) -> None:
        X = multiclass_frame.drop(columns=["Target"])
        y = multiclass_frame["Target"]
        board = compare_models(
            X,
            y,
            "classification",
            models=["logistic_regression", "ridge_classifier"],
            cv=3,
        )
        # The whole board is scored by the label-only fallback metric.
        metric_col = [c for c in board.columns if c.endswith("_mean")][0]
        assert metric_col == "balanced_accuracy_mean"
        assert board[metric_col].notna().all()

    def test_compare_models_regression_untouched(
        self, regression_frame: pd.DataFrame
    ) -> None:
        X = regression_frame.drop(columns=["pIC50"])
        y = regression_frame["pIC50"]
        board = compare_models(X, y, "regression", models=["ridge"], cv=3)
        assert "neg_root_mean_squared_error_mean" in board.columns


class TestMulticlassEnsembles:
    def test_hard_voting_multiclass(self, split) -> None:
        X_train, X_test, y_train, y_test = split
        ensemble = build_voting(
            "classification",
            ["logistic_regression", "knn", "ridge_classifier"],
            voting="hard",
        )
        ensemble.fit(X_train, y_train)
        accuracy = float((ensemble.predict(X_test) == y_test).mean())
        assert 0.0 < accuracy <= 1.0

    def test_soft_voting_multiclass(self, split) -> None:
        X_train, X_test, y_train, y_test = split
        ensemble = build_voting(
            "classification", ["logistic_regression", "knn"], voting="soft"
        )
        ensemble.fit(X_train, y_train)
        accuracy = float((ensemble.predict(X_test) == y_test).mean())
        assert 0.0 < accuracy <= 1.0

    def test_soft_voting_rejects_proba_less_member(self) -> None:
        with pytest.raises(ModelError, match="predict_proba"):
            build_voting(
                "classification", ["ridge_classifier", "knn"], voting="soft"
            )

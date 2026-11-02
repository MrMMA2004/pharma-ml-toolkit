"""Tests for the PyTorch deep-learning estimators (``pharmaml.ml.neural``).

Covers import-safety without torch (lazy-import contract), the
``'dnn'``/``'dnn_residual'`` registry wiring, fit/predict determinism,
softmax ``predict_proba`` (binary + multiclass), sklearn compatibility
(get_params/clone/GridSearchCV/joblib pickle), pipeline integration
(train_model/evaluate_model/compare_models), validation guards, and
the residual architecture (identity skips vs learned projections).

Every test keeps the data tiny and ``epochs`` small — the sandbox runs
on CPU only.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.validation import check_is_fitted

import pharmaml.ml.models as models_module
from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.ml import (
    DeepMLPClassifier,
    DeepMLPRegressor,
    ResidualMLPClassifier,
    ResidualMLPRegressor,
    build_model,
    compare_models,
    evaluate_model,
    list_models,
    resolve_model_name,
    train_model,
)
from pharmaml.ml.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers: tiny data + fast estimator configs
# ---------------------------------------------------------------------------
def fast_classifier(**overrides: Any) -> DeepMLPClassifier:
    """DeepMLPClassifier tuned for test speed (small net, few epochs)."""
    params: dict[str, Any] = {
        "hidden_layers": (16, 8),
        "epochs": 30,
        "patience": 6,
        "random_state": 7,
    }
    params.update(overrides)
    return DeepMLPClassifier(**params)


def fast_regressor(**overrides: Any) -> DeepMLPRegressor:
    """DeepMLPRegressor tuned for test speed."""
    params: dict[str, Any] = {
        "hidden_layers": (16, 8),
        "epochs": 30,
        "patience": 6,
        "random_state": 7,
    }
    params.update(overrides)
    return DeepMLPRegressor(**params)


@pytest.fixture(scope="module")
def binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small two-class problem (train + test split)."""
    X, y = make_classification(
        n_samples=160, n_features=12, n_informative=8, random_state=0
    )
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


@pytest.fixture(scope="module")
def multiclass_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small three-class problem (train + test split)."""
    X, y = make_classification(
        n_samples=180,
        n_features=12,
        n_informative=9,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=3,
    )
    return train_test_split(X, y, test_size=0.25, random_state=3, stratify=y)


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearly linear regression problem (train + test split)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5 * X[:, 2] + 0.2 * rng.normal(size=200)
    return train_test_split(X, y, test_size=0.25, random_state=1)


def _hide_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a torch-less machine (import of torch raises ImportError)."""
    monkeypatch.setitem(sys.modules, "torch", None)


def _hide_torch_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate torch being undetectable by the registry availability probe."""
    real = models_module._module_available

    def fake(name: str) -> bool:
        return False if name == "torch" else real(name)

    monkeypatch.setattr(models_module, "_module_available", fake)


# ---------------------------------------------------------------------------
# Import safety (lazy torch contract)
# ---------------------------------------------------------------------------
class TestImportSafety:
    def test_module_imports_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Executing the module source with torch "missing" must succeed:
        # all torch imports live inside functions (or TYPE_CHECKING).
        _hide_torch(monkeypatch)
        spec = importlib.util.find_spec("pharmaml.ml.neural")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in (
            "DeepMLPClassifier",
            "DeepMLPRegressor",
            "ResidualMLPClassifier",
            "ResidualMLPRegressor",
        ):
            assert hasattr(module, name)

    def test_fit_raises_dependency_error_without_torch(
        self, monkeypatch: pytest.MonkeyPatch, binary_data: tuple
    ) -> None:
        # The lazy import fires at fit time with an actionable message.
        X_train, _, y_train, _ = binary_data
        _hide_torch(monkeypatch)
        with pytest.raises(DependencyError, match="torch"):
            fast_classifier().fit(X_train, y_train)

    def test_dependency_error_names_the_deep_extra(
        self, monkeypatch: pytest.MonkeyPatch, binary_data: tuple
    ) -> None:
        X_train, _, y_train, _ = binary_data
        _hide_torch(monkeypatch)
        with pytest.raises(DependencyError, match=r"pharmaml\[deep\]"):
            fast_classifier().fit(X_train, y_train)

    def test_build_model_raises_dependency_error_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hide_torch_spec(monkeypatch)
        with pytest.raises(DependencyError, match="torch"):
            build_model("dnn", "classification")
        with pytest.raises(DependencyError, match=r"pharmaml\[deep\]"):
            build_model("dnn_residual", "regression")


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_dnn_entries_registered(self) -> None:
        for key in ("dnn", "dnn_residual"):
            assert key in MODEL_REGISTRY
            spec = MODEL_REGISTRY[key]
            assert spec.task == "both"
            assert spec.requires == "torch"
            assert "PyTorch" in spec.notes

    def test_registry_has_twenty_eight_entries(self) -> None:
        entries = list_models(only_available=False)
        # 26 + voting + stacking (1.2.49) + sgd_classifier (1.2.86,
        # xxx.ipynb cell 26) + the three naive-Bayes variants (1.2.93,
        # PDSH 05.05) + dnn_numpy (1.2.103, the autograd engine): every
        # dual-task family
        assert len(entries) == 33
        dnn_entries = [e for e in entries if e["name"] in {"dnn", "dnn_residual"}]
        assert len(dnn_entries) == 2
        assert {e["name"] for e in entries} >= {
            "hsvr",
            "dnn",
            "dnn_residual",
            "dnn_numpy",
            "extra_trees",
            "gradient_boosting",
            "pls",
            "gaussian_process",
            "gaussian_process_classifier",
            "sgd_classifier",
        }
        if not all(e["available"] for e in dnn_entries):
            pytest.skip("torch not installed in this environment")
        assert len(list_models()) == 33

    def test_aliases_resolve(self) -> None:
        assert resolve_model_name("deep_mlp") == "dnn"
        assert resolve_model_name("resnet") == "dnn_residual"
        assert resolve_model_name("residual_dnn") == "dnn_residual"
        assert resolve_model_name("dnn") == "dnn"
        assert resolve_model_name("DNN-Residual") == "dnn_residual"

    def test_build_model_dispatches_per_task(self) -> None:
        assert isinstance(build_model("dnn", "classification"), DeepMLPClassifier)
        assert isinstance(build_model("dnn", "regression"), DeepMLPRegressor)
        assert isinstance(
            build_model("dnn_residual", "classification"), ResidualMLPClassifier
        )
        assert isinstance(
            build_model("dnn_residual", "regression"), ResidualMLPRegressor
        )
        # random_state is threaded through the factories (registry convention)
        assert build_model("dnn", "classification").random_state == 42
        assert build_model("dnn_residual", "regression").random_state == 42

    def test_param_grid_keys_match_constructor_params(self) -> None:
        constructor_keys = set(DeepMLPClassifier().get_params())
        for key in ("dnn", "dnn_residual"):
            assert set(MODEL_REGISTRY[key].param_grid) <= constructor_keys
        # tuples are legal grid values for hidden_layers
        assert MODEL_REGISTRY["dnn"].param_grid["hidden_layers"] == [
            (64, 32),
            (128, 64, 32),
        ]

    def test_unavailable_simulation_flags_and_excludes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hide_torch_spec(monkeypatch)
        entries = list_models(only_available=False)
        flagged = {e["name"]: e["available"] for e in entries}
        assert flagged["dnn"] is False and flagged["dnn_residual"] is False
        # only_available=True drops them (26 core/boosting/hsvr/pls/gp/gpc
        # + voting/stacking + sgd_classifier + 3 naive-Bayes entries remain,
        # and dnn_numpy -- the autograd engine needs no torch)
        assert len(list_models()) == 31
        assert flagged["dnn_numpy"] is True  # the numpy engine never flags
        with pytest.raises(DependencyError, match="torch"):
            build_model("dnn", "classification")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_classifier_same_seed_identical_predictions(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _ = binary_data
        first = fast_classifier(random_state=7).fit(X_train, y_train)
        second = fast_classifier(random_state=7).fit(X_train, y_train)
        assert np.array_equal(first.predict(X_test), second.predict(X_test))
        assert np.array_equal(first.predict_proba(X_test), second.predict_proba(X_test))

    def test_regressor_same_seed_identical_predictions(
        self, regression_data: tuple
    ) -> None:
        X_train, X_test, y_train, _ = regression_data
        first = fast_regressor(random_state=11).fit(X_train, y_train)
        second = fast_regressor(random_state=11).fit(X_train, y_train)
        assert np.array_equal(first.predict(X_test), second.predict(X_test))

    def test_different_seed_changes_predictions(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _ = binary_data
        first = fast_classifier(random_state=1).fit(X_train, y_train)
        second = fast_classifier(random_state=2).fit(X_train, y_train)
        assert not np.array_equal(first.predict(X_test), second.predict(X_test))


# ---------------------------------------------------------------------------
# predict_proba semantics
# ---------------------------------------------------------------------------
class TestPredictProba:
    def test_binary_proba_rows_sum_to_one(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _y_test = binary_data
        model = fast_classifier().fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()

    def test_proba_columns_match_classes_order(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _ = binary_data
        model = fast_classifier().fit(X_train, y_train)
        assert list(model.classes_) == [0, 1]
        # argmax of proba == predict
        assert np.array_equal(
            model.classes_[np.argmax(model.predict_proba(X_test), axis=1)],
            model.predict(X_test),
        )

    def test_multiclass_proba(self, multiclass_data: tuple) -> None:
        X_train, X_test, y_train, _ = multiclass_data
        model = fast_classifier(epochs=40, patience=8).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(model.classes_) == {0, 1, 2}
        assert np.array_equal(
            model.classes_[np.argmax(proba, axis=1)], model.predict(X_test)
        )

    def test_string_labels_are_label_encoded(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _ = binary_data
        y_strings = np.where(y_train == 0, "inactive", "active")
        model = fast_classifier().fit(X_train, y_strings)
        assert list(model.classes_) == ["active", "inactive"]
        predicted = model.predict(X_test)
        assert set(predicted) <= {"active", "inactive"}
        assert model.predict_proba(X_test).shape == (len(X_test), 2)


# ---------------------------------------------------------------------------
# Learning quality (small data, few epochs)
# ---------------------------------------------------------------------------
class TestQuality:
    def test_multiclass_blobs_accuracy(self) -> None:
        # Explicit well-separated centres (make_blobs' own random centres
        # can nearly coincide — with random_state=5 logistic regression
        # tops out at 0.844, which would cap any model).
        X, y = make_blobs(
            n_samples=180,
            centers=[(-4.0, -4.0), (0.0, 4.0), (4.0, -4.0)],
            cluster_std=0.8,
            random_state=5,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=5, stratify=y
        )
        model = fast_classifier(epochs=40, patience=8).fit(X_train, y_train)
        assert accuracy_score(y_test, model.predict(X_test)) > 0.85

    def test_regression_r2_on_linear_function(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, y_test = regression_data
        model = fast_regressor(epochs=60, patience=10).fit(X_train, y_train)
        assert r2_score(y_test, model.predict(X_test)) > 0.7

    def test_mixin_scores(self, binary_data: tuple, regression_data: tuple) -> None:
        Xc_train, Xc_test, yc_train, yc_test = binary_data
        clf = fast_classifier().fit(Xc_train, yc_train)
        assert np.isclose(
            clf.score(Xc_test, yc_test),
            accuracy_score(yc_test, clf.predict(Xc_test)),
        )
        Xr_train, Xr_test, yr_train, yr_test = regression_data
        reg = fast_regressor().fit(Xr_train, yr_train)
        assert np.isclose(
            reg.score(Xr_test, yr_test),
            r2_score(yr_test, reg.predict(Xr_test)),
        )

    def test_history_and_n_iter_exposed(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        model = fast_classifier(epochs=25, val_split=0.15).fit(X_train, y_train)
        assert isinstance(model.history_, list) and len(model.history_) >= 1
        assert {"epoch", "train_loss", "val_loss"} == set(model.history_[0])
        assert model.n_iter_ == len(model.history_) <= 25
        assert model.n_features_in_ == X_train.shape[1]
        assert model.history_[-1]["val_loss"] is not None  # val split active

    def test_on_epoch_progress_stream(self, binary_data: tuple) -> None:
        # 100x-106: the GUI progress hook. One payload per epoch, in
        # order, ending exactly when training ends (early stopping
        # included); the fraction epoch/epochs is an honest ETA input.
        X_train, _, y_train, _ = binary_data
        events: list[dict[str, Any]] = []
        model = fast_classifier(epochs=6, val_split=0.2)
        model.fit(X_train, y_train, on_epoch=events.append)
        assert [event["epoch"] for event in events] == list(range(1, 7))
        assert all(event["epochs"] == 6 for event in events)
        assert len(events) == len(model.history_) == model.n_iter_
        assert all(set(event) == {
            "epoch", "epochs", "train_loss", "val_loss", "best_val", "stale",
        } for event in events)
        assert all(
            event["val_loss"] is not None and event["best_val"] is not None
            for event in events
        )  # the split is active
        assert events[0]["stale"] == 0  # the first epoch always improves

    def test_on_epoch_absent_keeps_clone_and_get_params_clean(
        self, binary_data: tuple
    ) -> None:
        # the callback is a transient fit argument, NOT a hyperparameter:
        # clone/get_params must never see it, and a fit without it stays
        # bit-identical to the pre-100x-106 behaviour.
        from sklearn.base import clone

        X_train, _, y_train, _ = binary_data
        model = fast_classifier(epochs=3, val_split=0.2)
        assert "on_epoch" not in model.get_params()
        clone(model).fit(X_train, y_train)  # positional fit stays valid
        quiet = fast_classifier(epochs=3, val_split=0.2).fit(X_train, y_train)
        loud = fast_classifier(epochs=3, val_split=0.2)
        loud.fit(X_train, y_train, on_epoch=lambda info: None)
        assert len(quiet.history_) == len(loud.history_)

    def test_val_split_zero_disables_early_stopping(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _ = binary_data
        model = fast_classifier(epochs=5, val_split=0.0).fit(X_train, y_train)
        assert model.n_iter_ == 5
        assert all(entry["val_loss"] is None for entry in model.history_)
        assert model.predict(X_test).shape == (len(X_test),)


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------
class TestSklearnCompatibility:
    def test_get_params_has_all_constructor_keys(self) -> None:
        expected = {
            "hidden_layers",
            "dropout",
            "activation",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "val_split",
            "random_state",
            "device",
            "batchnorm",
        }
        for estimator in (
            DeepMLPClassifier(),
            DeepMLPRegressor(),
            ResidualMLPClassifier(),
            ResidualMLPRegressor(),
        ):
            assert set(estimator.get_params()) == expected

    def test_clone_round_trip(self) -> None:
        original = DeepMLPClassifier(hidden_layers=(9,), epochs=3, dropout=0.25)
        duplicated = clone(original)
        assert isinstance(duplicated, DeepMLPClassifier)
        assert duplicated is not original
        assert duplicated.get_params() == original.get_params()

    def test_clone_works_for_all_four_estimators(self) -> None:
        for original in (
            DeepMLPRegressor(epochs=2),
            ResidualMLPClassifier(epochs=2),
            ResidualMLPRegressor(epochs=2),
        ):
            assert clone(original).get_params() == original.get_params()

    def test_grid_search_cv(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        grid = {
            "hidden_layers": [(16,), (32,)],
            "learning_rate": [1e-3],
        }
        search = GridSearchCV(
            fast_classifier(epochs=15, patience=4),
            grid,
            cv=3,
            scoring="accuracy",
        )
        search.fit(X_train, y_train)
        assert search.best_params_["hidden_layers"] in {(16,), (32,)}
        # refit=True: the search predicts with the winning configuration
        assert accuracy_score(y_test, search.predict(X_test)) > 0.5
        assert np.isfinite(search.best_score_)

    def test_grid_search_cv_with_pipeline_predict_proba(
        self, binary_data: tuple
    ) -> None:
        # roc_auc scoring (the train_model default) needs predict_proba.
        X_train, _, y_train, _ = binary_data
        search = GridSearchCV(
            fast_classifier(epochs=10, patience=4),
            {"hidden_layers": [(16,)]},
            cv=3,
            scoring="roc_auc",
        )
        search.fit(X_train, y_train)
        assert np.isfinite(search.best_score_)

    def test_joblib_round_trip(
        self, tmp_path: Any, binary_data: tuple, regression_data: tuple
    ) -> None:
        Xc_train, Xc_test, yc_train, _ = binary_data
        clf = fast_classifier().fit(Xc_train, yc_train)
        path = tmp_path / "deep_clf.joblib"
        joblib.dump(clf, path)
        loaded = joblib.load(path)
        assert np.array_equal(loaded.predict(Xc_test), clf.predict(Xc_test))
        assert np.array_equal(
            loaded.predict_proba(Xc_test), clf.predict_proba(Xc_test)
        )

        Xr_train, Xr_test, yr_train, _ = regression_data
        reg = fast_regressor().fit(Xr_train, yr_train)
        path = tmp_path / "deep_reg.joblib"
        joblib.dump(reg, path)
        assert np.array_equal(joblib.load(path).predict(Xr_test), reg.predict(Xr_test))

    def test_pickle_contains_no_torch_objects(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        clf = fast_classifier().fit(X_train, y_train)
        state = clf.__getstate__()
        # weights survive as plain numpy arrays -> portable pickles
        assert state["_network"] is None
        assert all(isinstance(v, np.ndarray) for v in clf._weights.values())

    def test_set_params_after_fit_keeps_predictions_working(
        self, binary_data: tuple
    ) -> None:
        # The live network is rebuilt from the *fitted* architecture
        # snapshot, so a later set_params cannot desynchronise it.
        X_train, X_test, y_train, _ = binary_data
        model = fast_classifier().fit(X_train, y_train)
        before = model.predict(X_test)
        model.set_params(hidden_layers=(99,))
        assert np.array_equal(model.predict(X_test), before)


# ---------------------------------------------------------------------------
# Pipeline integration (registry -> train -> evaluate -> compare)
# ---------------------------------------------------------------------------
class TestPipelineIntegration:
    def test_train_and_evaluate_dnn_classification(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        result = train_model("dnn", "classification", X_train, y_train, tune=False)
        assert result.name == "dnn"
        assert result.task == "classification"
        report = evaluate_model(result.model, X_test, y_test, "classification")
        assert 0.0 <= report.accuracy <= 1.0
        assert report.roc_auc is not None and 0.0 <= report.roc_auc <= 1.0

    def test_train_and_evaluate_dnn_residual_regression(
        self, regression_data: tuple
    ) -> None:
        X_train, X_test, y_train, y_test = regression_data
        result = train_model(
            "dnn_residual", "regression", X_train, y_train, tune=False
        )
        assert result.name == "dnn_residual"
        report = evaluate_model(result.model, X_test, y_test, "regression")
        assert np.isfinite(report.r2)
        assert report.r2 > 0.5  # nearly linear synthetic target

    def test_train_model_with_dataframes(self, binary_data: tuple) -> None:
        # train_model feeds DataFrames through a Pipeline (impute+scale).
        X_train, _, y_train, _ = binary_data
        frame = pd.DataFrame(X_train, columns=[f"descr_{i:02d}" for i in range(X_train.shape[1])])
        target = pd.Series(y_train, name="Target")
        result = train_model("dnn", "classification", frame, target, tune=False)
        report = evaluate_model(result.model, frame.iloc[:10], target.iloc[:10], "classification")
        assert report.n_samples == 10

    def test_compare_models_with_dnn(self, regression_data: tuple) -> None:
        X, _, y, _ = regression_data
        board = compare_models(X, y, "regression", models=["elastic_net", "dnn"], cv=3)
        assert len(board) == 2
        assert set(board["model"]) == {"elastic_net", "dnn"}
        metric_col = next(c for c in board.columns if c.endswith("_mean"))
        dnn_row = board[board["model"] == "dnn"]
        assert dnn_row[metric_col].notna().all()
        assert np.isfinite(float(dnn_row[metric_col].iloc[0]))

    def test_compare_models_classification_with_dnn(self, binary_data: tuple) -> None:
        X, _, y, _ = binary_data
        board = compare_models(
            X, y, "classification", models=["logistic_regression", "dnn"], cv=3
        )
        assert len(board) == 2
        metric_col = next(c for c in board.columns if c.endswith("_mean"))
        assert metric_col == "roc_auc_ovo_mean"
        assert board[metric_col].notna().all()


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
class TestEdgeGuards:
    def test_empty_hidden_layers_builds_linear_model(self, binary_data: tuple) -> None:
        # Documented behaviour: () == a single linear layer (regularised
        # logistic/linear regression), not an error.
        X_train, X_test, y_train, _ = binary_data
        model = fast_classifier(hidden_layers=()).fit(X_train, y_train)
        assert model.predict_proba(X_test).shape == (len(X_test), 2)
        assert np.allclose(model.predict_proba(X_test).sum(axis=1), 1.0)
        assert model.n_iter_ >= 1

    def test_epochs_zero_raises(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(ModelError, match="epochs"):
            fast_classifier(epochs=0).fit(X_train, y_train)

    def test_unknown_activation_raises(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(ModelError, match="activation"):
            fast_classifier(activation="leaky_relu").fit(X_train, y_train)

    @pytest.mark.parametrize("hidden", [(0,), (8, -1), (4, 2.5), ("wide",)])
    def test_bad_hidden_layers_raises(
        self, binary_data: tuple, hidden: Any
    ) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(ModelError, match="hidden_layers"):
            fast_classifier(hidden_layers=hidden).fit(X_train, y_train)

    def test_batch_size_and_patience_guards(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(ModelError, match="batch_size"):
            fast_classifier(batch_size=0).fit(X_train, y_train)
        with pytest.raises(ModelError, match="patience"):
            fast_classifier(patience=0).fit(X_train, y_train)

    def test_bad_device_token_raises(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(ModelError, match="device"):
            fast_classifier(device="tpu").fit(X_train, y_train)

    def test_empty_X_raises(self) -> None:
        model = fast_classifier()
        with pytest.raises(DataValidationError, match="0 rows"):
            model.fit(np.empty((0, 4)), np.array([], dtype=int))
        with pytest.raises(DataValidationError, match="0 feature columns"):
            model.fit(np.empty((5, 0)), np.zeros(5))

    def test_length_mismatch_raises(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(DataValidationError, match="inconsistent lengths"):
            fast_classifier().fit(X_train, y_train[:-3])

    def test_nan_features_raise(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        X_nan = X_train.astype(float).copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            fast_classifier().fit(X_nan, y_train)

    def test_nan_regression_target_raises(self, regression_data: tuple) -> None:
        X_train, _, y_train, _ = regression_data
        y_nan = y_train.astype(float).copy()
        y_nan[0] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            fast_regressor().fit(X_train, y_nan)

    def test_single_class_target_raises(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        with pytest.raises(DataValidationError, match="2 distinct classes"):
            fast_classifier().fit(X_train, np.zeros(len(y_train), dtype=int))

    def test_predict_before_fit_raises(self, binary_data: tuple) -> None:
        _, X_test, _, _ = binary_data
        with pytest.raises(ModelError, match="not been fitted"):
            DeepMLPClassifier().predict(X_test)
        with pytest.raises(ModelError, match="not been fitted"):
            DeepMLPRegressor().predict(X_test)
        with pytest.raises(ModelError, match="not been fitted"):
            DeepMLPClassifier().predict_proba(X_test)

    def test_predict_proba_missing_on_regressor(self) -> None:
        # Regressors have no predict_proba (voting='soft' guard relies on it)
        assert not hasattr(DeepMLPRegressor(), "predict_proba")


# ---------------------------------------------------------------------------
# Residual architecture
# ---------------------------------------------------------------------------
class TestResidualMLP:
    def test_classifier_fit_predict_smoke(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        model = ResidualMLPClassifier(
            hidden_layers=(16, 16), epochs=30, patience=6, random_state=5
        ).fit(X_train, y_train)
        assert accuracy_score(y_test, model.predict(X_test)) > 0.6
        proba = model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_regressor_fit_predict_smoke(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, y_test = regression_data
        model = ResidualMLPRegressor(
            hidden_layers=(16, 16), epochs=40, patience=8, random_state=5
        ).fit(X_train, y_train)
        assert r2_score(y_test, model.predict(X_test)) > 0.7

    def test_determinism(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, _ = regression_data
        first = ResidualMLPRegressor(
            hidden_layers=(16, 16), epochs=20, patience=5, random_state=9
        ).fit(X_train, y_train)
        second = ResidualMLPRegressor(
            hidden_layers=(16, 16), epochs=20, patience=5, random_state=9
        ).fit(X_train, y_train)
        assert np.array_equal(first.predict(X_test), second.predict(X_test))

    def test_changing_widths_exercise_projection_path(self, binary_data: tuple) -> None:
        # (32, 64, 16) changes width at every block -> learned linear
        # skip projections instead of identity shortcuts.
        X_train, X_test, y_train, _ = binary_data
        model = ResidualMLPClassifier(
            hidden_layers=(32, 64, 16), epochs=25, patience=5, random_state=4
        ).fit(X_train, y_train)
        assert model._net_spec_["hidden"] == (32, 64, 16)
        assert any("skip.weight" in key for key in model._weights)
        assert np.isfinite(model.predict(X_test)).all()
        assert np.allclose(model.predict_proba(X_test).sum(axis=1), 1.0)

    def test_matching_widths_use_identity_skips(self, binary_data: tuple) -> None:
        # 12 features -> (12, 12): every block skips by identity, so no
        # learnable projection parameters exist in the state dict.
        X_train, _, y_train, _ = binary_data
        model = ResidualMLPClassifier(
            hidden_layers=(12, 12), epochs=5, patience=2, random_state=4
        ).fit(X_train, y_train)
        assert not any("skip.weight" in key for key in model._weights)

    def test_default_hidden_layers_are_uniform(self) -> None:
        # The shipped default (64, 64, 64) is a uniform-width residual
        # stack (identity skips inside; projection only at the stem).
        assert ResidualMLPClassifier().hidden_layers == (64, 64, 64)
        assert ResidualMLPRegressor().hidden_layers == (64, 64, 64)
        assert DeepMLPClassifier().hidden_layers == (64, 32)


# ---------------------------------------------------------------------------
# Regression tests from the ml/neural.py deep review (100x-18-b)
# ---------------------------------------------------------------------------
class TestEarlyStoppingRegression:
    """Early stopping / best-weight restore / refit state (100x-18-b)."""

    def test_best_epoch_weights_restored_not_last(self, binary_data: tuple) -> None:
        # A deliberately hot learning rate makes the validation loss dip
        # early and then WORSEN; the stored weights must be the ones from
        # the best epoch, not the final one.  A second estimator trained
        # for exactly best_epoch epochs (same seed -> identical epoch
        # sequence) ends up holding those best-epoch weights.
        X_train, _, y_train, _ = binary_data
        common: dict[str, Any] = {
            "hidden_layers": (16, 8),
            "epochs": 60,
            "patience": 60,
            "val_split": 0.2,
            "random_state": 7,
            "learning_rate": 0.05,
        }
        full = DeepMLPClassifier(**common).fit(X_train, y_train)
        val_losses = [entry["val_loss"] for entry in full.history_]
        best_epoch = int(np.argmin(val_losses)) + 1
        # meaningfulness precondition: the curve really does worsen
        assert val_losses[-1] > val_losses[best_epoch - 1]
        truncated = DeepMLPClassifier(
            **{**common, "epochs": best_epoch, "patience": best_epoch}
        ).fit(X_train, y_train)
        assert full._weights.keys() == truncated._weights.keys()
        assert all(
            np.allclose(full._weights[key], truncated._weights[key])
            for key in full._weights
        )

    def test_small_patience_stops_before_max_epochs(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        model = DeepMLPClassifier(
            hidden_layers=(16, 8),
            epochs=40,
            patience=1,
            val_split=0.2,
            random_state=7,
            learning_rate=0.05,
        ).fit(X_train, y_train)
        # n_iter_ documents the epochs RUN (not the best epoch)
        assert model.n_iter_ == len(model.history_) <= 40
        assert model.n_iter_ < 40  # early stopping actually fired

    def test_diverged_training_raises_model_error(self) -> None:
        # Extreme learning rate -> NaN weights: fit must refuse loudly
        # instead of leaving a silently-NaN estimator (predict used to
        # return all-NaN with no error).
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 6))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.normal(size=120)
        model = DeepMLPRegressor(
            hidden_layers=(16, 8),
            epochs=12,
            patience=12,
            val_split=0.15,
            random_state=3,
            learning_rate=1e9,
        )
        with pytest.raises(ModelError, match="diverged"):
            model.fit(X, y)
        # a refused fit must not leave a usable-but-NaN estimator behind
        with pytest.raises(ModelError, match="not been fitted"):
            model.predict(X[:3])

    def test_refit_resets_history_and_n_iter(self, binary_data: tuple) -> None:
        # fit() twice on the same instance: history_ must be REPLACED,
        # never appended to.
        X_train, X_test, y_train, _ = binary_data
        model = DeepMLPClassifier(
            hidden_layers=(8,), epochs=6, patience=6, val_split=0.2, random_state=5
        ).fit(X_train, y_train)
        assert len(model.history_) == 6
        model.fit(X_train, y_train)  # second fit, same instance
        assert len(model.history_) == 6  # NOT 12
        assert model.n_iter_ == len(model.history_)
        assert model.n_features_in_ == X_train.shape[1]
        assert np.isfinite(model.predict(X_test[:5])).all()


class TestValidationSplitRegression:
    """Stratification edges + train-only standardisation (100x-18-b)."""

    def test_single_member_class_stays_in_training_fold(self) -> None:
        # class 2 has exactly one row; a large val_split used to move that
        # row to the validation fold, leaving the class untrainable.
        X = np.random.default_rng(0).normal(size=(13, 4))
        y = np.array([0] * 6 + [1] * 6 + [2])
        est = DeepMLPClassifier(
            hidden_layers=(4,), epochs=1, val_split=0.6, random_state=7
        )
        train_idx, _val_idx = est._train_val_split(y, len(y), np.random.default_rng(7))
        assert set(y[train_idx].tolist()) == {0, 1, 2}  # every class trainable
        # end-to-end fit still works with the tiny stratified split
        fitted = DeepMLPClassifier(
            hidden_layers=(4,), epochs=3, val_split=0.6, random_state=7
        ).fit(X, y)
        assert fitted.history_[-1]["val_loss"] is not None
        assert np.isfinite(fitted.predict_proba(X)).all()

    def test_standardisation_fitted_on_training_rows_only(self) -> None:
        # The scaler must be fitted on the TRAINING rows only — the
        # early-stopping validation rows (here: a x500 outlier) must not
        # leak into mean/std.
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] > 0).astype(int)
        X[0, 0] = 500.0  # giant outlier
        model = DeepMLPClassifier(
            hidden_layers=(4,), epochs=2, val_split=0.15, random_state=11
        ).fit(X, y)
        # replicate the split to know which rows trained
        train_idx, val_idx = model._train_val_split(y, len(y), np.random.default_rng(11))
        assert 0 in set(val_idx.tolist())  # outlier really is in the val fold
        assert np.isclose(model._mean_[0], X[train_idx][:, 0].mean())
        assert not np.isclose(model._mean_[0], X[:, 0].mean())  # no full-data leak


class TestInputContractRegression:
    """Error contracts for X/y shapes and dtypes (100x-18-b)."""

    def test_predict_rejects_wrong_feature_count(self, binary_data: tuple) -> None:
        # Too FEW columns used to *broadcast* silently against the
        # per-feature mean (garbage predictions); too MANY raised a raw
        # numpy broadcast error.  Both must be typed DataValidationErrors.
        X_train, X_test, y_train, _ = binary_data
        model = DeepMLPClassifier(
            hidden_layers=(8,), epochs=2, random_state=1
        ).fit(X_train, y_train)
        n_features = X_train.shape[1]
        with pytest.raises(DataValidationError, match=f"{n_features - 1} feature"):
            model.predict(X_test[:, : n_features - 1])
        with pytest.raises(DataValidationError, match=f"{n_features + 1} feature"):
            model.predict(np.hstack([X_test, np.zeros((len(X_test), 1))]))
        with pytest.raises(DataValidationError, match="fitted with"):
            model.predict_proba(X_test[:, :1])

    def test_classifier_rejects_none_labels(self, binary_data: tuple) -> None:
        # None mixed with string labels used to escape as a raw
        # TypeError from np.unique's sort.
        X_train, _, _, _ = binary_data
        y = ["a"] * 10 + ["b"] * 9 + [None]
        with pytest.raises(DataValidationError, match="label-encoded"):
            DeepMLPClassifier(
                hidden_layers=(4,), epochs=1, random_state=1
            ).fit(X_train[:20], y)

    def test_hidden_layers_generator_does_not_degenerate(self) -> None:
        # A generator used to be consumed by validation, so the training
        # loop saw an EMPTY architecture and silently built a single
        # linear layer.
        X, y = make_blobs(
            n_samples=60, centers=[(-3.0, -3.0), (3.0, 3.0)], random_state=1
        )
        model = DeepMLPClassifier(
            hidden_layers=(w for w in (16, 8)), epochs=2, random_state=1
        ).fit(X, y)
        assert model._net_spec_["hidden"] == (16, 8)
        # three weight matrices (two hidden layers + head), not the one a
        # degenerate single-linear-layer model would have
        assert sum(key.endswith(".weight") for key in model._weights) == 3

    def test_failed_fit_leaves_estimator_unfitted(
        self, monkeypatch: pytest.MonkeyPatch, binary_data: tuple
    ) -> None:
        # A DependencyError from a missing torch used to leave
        # classes_/n_features_in_/_mean_ behind, so sklearn's
        # check_is_fitted considered the estimator fitted.
        X_train, _, y_train, _ = binary_data
        _hide_torch(monkeypatch)
        model = DeepMLPClassifier(
            hidden_layers=(4,), epochs=1, random_state=1
        )
        with pytest.raises(DependencyError, match="torch"):
            model.fit(X_train, y_train)
        for attr in ("classes_", "n_features_in_", "feature_names_in_", "_mean_"):
            assert not hasattr(model, attr)
        with pytest.raises(NotFittedError):
            check_is_fitted(model)

    def test_proba_columns_follow_sorted_classes_multiclass(self) -> None:
        # y values arrive UNsorted ('b','a','c'): classes_ must be sorted
        # and predict_proba column j must be P(classes_[j]).
        X, y_int = make_blobs(
            n_samples=90,
            centers=[(-4.0, -4.0), (0.0, 4.0), (4.0, -4.0)],
            cluster_std=0.6,
            random_state=2,
        )
        mapping = {0: "b", 1: "a", 2: "c"}  # 'a' lives at centre (0, 4)
        y = np.array([mapping[v] for v in y_int])
        model = DeepMLPClassifier(
            hidden_layers=(16,), epochs=60, patience=10, random_state=2
        ).fit(X, y)
        assert [str(c) for c in model.classes_] == ["a", "b", "c"]
        centers = np.array([(-4.0, -4.0), (0.0, 4.0), (4.0, -4.0)])
        proba = model.predict_proba(centers)
        # column 0 == P('a') must win at the 'a' centre (0, 4), etc.
        assert int(np.argmax(proba[1])) == 0
        assert int(np.argmax(proba[0])) == 1
        assert int(np.argmax(proba[2])) == 2
        assert model.predict(centers[[1, 0, 2]]).tolist() == ["a", "b", "c"]

    def test_dropout_disabled_at_predict_time(self) -> None:
        # Repeated predict_proba calls must be identical: the live
        # network runs in eval mode (dropout off).
        X, y = make_blobs(
            n_samples=60, centers=[(-3.0, -3.0), (3.0, 3.0)], random_state=4
        )
        model = DeepMLPClassifier(
            hidden_layers=(32, 32), dropout=0.5, epochs=3, random_state=4
        ).fit(X, y)
        first = model.predict_proba(X)
        second = model.predict_proba(X)
        assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Target standardisation (pharmaceutical endpoints are far from zero)
# ---------------------------------------------------------------------------
class TestTargetStandardisation:
    """Regression targets are z-scaled internally and mapped back at predict.

    Real pharma endpoints (pIC50 ~ 5, LN_IC50 ~ 2-4, AUC in [0, 1]) are
    never zero-centred: on a raw-scale MSE the output bias crawls toward
    the target mean at ~learning_rate per step and early stopping fires
    long before the intercept is learned.  Pre-fix repro (this exact
    test's data, 60 epochs): rmse 4.39 -> post-fix 0.36.
    """

    def test_pic50_scale_target_is_learnable(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.normal(size=(120, 6))
        # pIC50-style target: informative signal + noise, centred ~5.3.
        y = 5.3 + 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.05 * rng.normal(size=120)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=11
        )
        model = fast_regressor(epochs=60, patience=10).fit(X_train, y_train)
        rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
        # A linear baseline reaches ~0.05; the pre-fix network sat at
        # ~4.4 (never learned the intercept).  1.0 is generous but
        # still fails the pre-fix code by 4x.
        assert rmse < 1.0, f"off-centre target not learned: rmse={rmse:.3f}"

    def test_predictions_come_back_on_target_scale(self) -> None:
        rng = np.random.default_rng(12)
        X = rng.normal(size=(80, 4))
        y = 6.0 + 0.5 * X[:, 0] + 0.02 * rng.normal(size=80)
        model = fast_regressor(epochs=20).fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (80,)
        # Predictions live on the y scale, not the z scale.
        assert predictions.min() > 3.0 and predictions.max() < 9.0

    def test_y_scaling_attributes_fitted_on_training_rows(self) -> None:
        rng = np.random.default_rng(13)
        X = rng.normal(size=(100, 3))
        y = 4.0 + X[:, 0]
        model = fast_regressor(epochs=3, val_split=0.25).fit(X, y)
        assert isinstance(model._y_mean_, float)
        assert isinstance(model._y_std_, float) and model._y_std_ > 0
        # ~75 training rows of y centred at 4.0.
        assert 3.9 < model._y_mean_ < 4.1

    def test_constant_target_predicts_the_constant(self) -> None:
        rng = np.random.default_rng(14)
        X = rng.normal(size=(40, 3))
        y = np.full(40, 5.5)
        # Early stopping disabled (constant target has no signal to
        # validate on) and enough epochs for the output to reach 0 on
        # the z scale == 5.5 on the target scale.
        model = fast_regressor(epochs=40, val_split=0.0).fit(X, y)
        assert np.allclose(model.predict(X), 5.5, atol=0.05)

    def test_classifier_has_no_y_scaling(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        model = fast_classifier(epochs=2).fit(X_train, y_train)
        assert model._y_mean_ is None and model._y_std_ is None

    def test_residual_regressor_maps_back_to_target_scale(self) -> None:
        rng = np.random.default_rng(15)
        X = rng.normal(size=(90, 5))
        y = 2.7 + 0.6 * X[:, 2] + 0.05 * rng.normal(size=90)
        model = ResidualMLPRegressor(
            hidden_layers=(16, 16), epochs=40, patience=10, random_state=7
        ).fit(X, y)
        rmse = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
        assert rmse < 0.5, f"residual regressor off-centre target: rmse={rmse:.3f}"


# ---------------------------------------------------------------------------
# Global-RNG isolation + feature-name contract (26-R3 re-fix)
# ---------------------------------------------------------------------------
class TestGlobalRNGIsolation:
    """``fit()`` must leave the caller's *global* torch RNG untouched (26-R3).

    The training loop seeds torch's global generator with the
    estimator's ``random_state`` (the same-seed determinism contract);
    pre-fix it never restored the previous state, silently clobbering
    whatever stream the caller had seeded.
    """

    def test_fit_restores_global_torch_rng_state(self, binary_data: tuple) -> None:
        torch = pytest.importorskip("torch")
        X_train, _, y_train, _ = binary_data
        # No-fit control: what a 123-seeded stream yields.
        torch.manual_seed(123)
        control = torch.rand(8).clone()
        # Rewind to the identical stream position, then train an
        # estimator seeded differently (random_state=7): the draws after
        # fit must still come from the 123 stream, not the 7 one.
        torch.manual_seed(123)
        fast_classifier(epochs=2, random_state=7).fit(X_train, y_train)
        after = torch.rand(8)
        assert torch.equal(after, control)

    def test_two_fits_same_random_state_remain_identical(self, binary_data: tuple) -> None:
        # Pin: the restore must not weaken the same-seed determinism
        # contract (manual_seed still runs inside the protected window).
        X_train, X_test, y_train, _ = binary_data
        first = fast_classifier(epochs=4, random_state=7).fit(X_train, y_train)
        second = fast_classifier(epochs=4, random_state=7).fit(X_train, y_train)
        assert np.array_equal(first.predict(X_test), second.predict(X_test))


class TestFeatureNameContract:
    """``feature_names_in_`` is enforced at predict time (26-R3).

    Pre-fix the attribute was stored but never checked: predicting on a
    column-PERMUTED (or renamed) DataFrame silently mis-assigned
    features (values are consumed positionally), and it went stale
    after an ndarray refit.  Arrays stay positional, sklearn-style.
    """

    @staticmethod
    def _frame(X: np.ndarray, prefix: str) -> pd.DataFrame:
        return pd.DataFrame(X, columns=[f"{prefix}{i}" for i in range(X.shape[1])])

    def test_predict_permuted_columns_raises(self, regression_data: tuple) -> None:
        X_train, _, y_train, _ = regression_data
        frame = self._frame(X_train, "f")
        model = fast_regressor(epochs=2).fit(frame, y_train)
        permuted = frame[frame.columns[::-1]]
        with pytest.raises(DataValidationError, match="feature names"):
            model.predict(permuted)

    def test_predict_renamed_columns_raise(self, regression_data: tuple) -> None:
        # Same order, different names: still a contract violation.
        X_train, _, y_train, _ = regression_data
        model = fast_regressor(epochs=2).fit(self._frame(X_train, "a"), y_train)
        with pytest.raises(DataValidationError, match="feature names"):
            model.predict(self._frame(X_train, "b"))

    def test_predict_same_order_dataframe_works(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, _ = regression_data
        model = fast_regressor(epochs=2).fit(self._frame(X_train, "f"), y_train)
        from_frame = model.predict(self._frame(X_test, "f"))
        from_array = model.predict(X_test)
        assert np.allclose(from_frame, from_array)

    def test_ndarray_refit_deletes_stale_feature_names(
        self, regression_data: tuple
    ) -> None:
        # sklearn reset semantics: a positional (array) refit drops the
        # remembered names instead of enforcing the stale column order.
        X_train, _, y_train, _ = regression_data
        model = fast_regressor(epochs=2).fit(self._frame(X_train, "f"), y_train)
        assert hasattr(model, "feature_names_in_")
        model.fit(X_train, y_train)
        assert not hasattr(model, "feature_names_in_")
        assert np.isfinite(model.predict(X_train[:3])).all()

    def test_dataframe_refit_refreshes_feature_names(
        self, regression_data: tuple
    ) -> None:
        X_train, _, y_train, _ = regression_data
        model = fast_regressor(epochs=2).fit(self._frame(X_train, "a"), y_train)
        model.fit(self._frame(X_train, "b"), y_train)
        assert list(model.feature_names_in_) == [
            f"b{i}" for i in range(X_train.shape[1])
        ]


# ---------------------------------------------------------------------------
# Batch normalisation (batchnorm=True)
# ---------------------------------------------------------------------------
class TestBatchNormalization:
    """The optional BatchNorm1d layer between linear maps and activations."""

    def test_regressor_learns_with_batchnorm(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, y_test = regression_data
        model = fast_regressor(batchnorm=True, epochs=60).fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9
        assert np.isfinite(model.predict(X_test)).all()

    def test_running_statistics_persist_in_the_weight_dict(self) -> None:
        X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 1.0
        model = fast_regressor(batchnorm=True, epochs=10).fit(X, y)
        bn_keys = [key for key in model._weights if "running_mean" in key]
        assert bn_keys, "BatchNorm buffers missing from the portable weights"
        assert all(np.isfinite(model._weights[key]).all() for key in bn_keys)

    def test_determinism_with_batchnorm(self, regression_data: tuple) -> None:
        X_train, _, y_train, _ = regression_data
        first = fast_regressor(batchnorm=True, epochs=5).fit(X_train, y_train)
        second = fast_regressor(batchnorm=True, epochs=5).fit(X_train, y_train)
        assert np.array_equal(first.predict(X_train), second.predict(X_train))

    def test_classifier_with_batchnorm(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        model = fast_classifier(batchnorm=True, epochs=40).fit(X_train, y_train)
        assert accuracy_score(y_test, model.predict(X_test)) > 0.85
        probabilities = model.predict_proba(X_test)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_residual_estimators_with_batchnorm(self, regression_data: tuple) -> None:
        X_train, _, y_train, _ = regression_data
        for estimator in (
            ResidualMLPRegressor(hidden_layers=(16, 16), epochs=30, random_state=0),
            ResidualMLPClassifier(hidden_layers=(16, 16), epochs=30, random_state=0),
        ):
            fitted = estimator.set_params(batchnorm=True).fit(X_train, y_train)
            assert np.isfinite(fitted.predict(X_train)).all()

    def test_batchnorm_requires_a_real_batch_size(self) -> None:
        X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
        y = X[:, 0]
        with pytest.raises(ModelError, match="batch_size must be >= 2"):
            DeepMLPRegressor(batchnorm=True, batch_size=1).fit(X, y)

    def test_batchnorm_rejects_non_bool(self) -> None:
        with pytest.raises(ModelError, match="batchnorm must be a bool"):
            DeepMLPRegressor(batchnorm=1).fit(
                np.zeros((4, 2)), np.zeros(4)
            )

    def test_singleton_tail_batches_are_skipped(self) -> None:
        # 8 training rows with batch_size 7 leaves a 1-sample tail batch;
        # BatchNorm1d cannot normalise it, so the epoch skips it instead
        # of crashing.
        X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = 2.0 * X[:, 0]
        model = fast_regressor(batchnorm=True, batch_size=7, epochs=5).fit(X, y)
        assert np.isfinite(model.predict(X)).all()

    def test_batchnorm_needs_two_training_rows(self) -> None:
        # val_split steals every row but one from training
        X = np.zeros((2, 1)) + np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with pytest.raises(ModelError, match="at least 2 training rows"):
            DeepMLPRegressor(
                batchnorm=True, val_split=0.5, epochs=2, random_state=0
            ).fit(X, y)

    def test_pickle_round_trip_rebuilds_the_bn_layers(self, regression_data: tuple) -> None:
        import io

        X_train, _, y_train, _ = regression_data
        model = fast_regressor(batchnorm=True, epochs=3).fit(X_train, y_train)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        assert np.allclose(restored.predict(X_train), model.predict(X_train))

    def test_get_params_and_clone_carry_batchnorm(self) -> None:
        params = DeepMLPRegressor(batchnorm=True).get_params()
        assert params["batchnorm"] is True
        twin = clone(DeepMLPRegressor(batchnorm=True))
        assert twin.get_params()["batchnorm"] is True

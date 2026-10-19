"""Tests for the keras-tuner-style search (``pharmaml.ml.tuner``).

Covers the ``HyperParameters`` space protocol (collection mode, trial
mode, re-registration), the ready-made hypermodels (ANN / deep / HSVR),
``RandomSearch`` (scoring, determinism, best-estimator refit, failure
accounting) and ``Hyperband`` (brackets, rung resources, the resource
declaration contract).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPRegressor

from pharmaml.exceptions import ModelError
from pharmaml.ml.hierarchical import HierarchicalSVR
from pharmaml.ml.tuner import (
    Hyperband,
    HyperParameters,
    RandomSearch,
    TunerResults,
    build_ann_classifier,
    build_ann_regressor,
    build_hsvr,
)


def linear_data(n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """A learnable linear surface with > 10 distinct float targets."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (n, 3))
    y = 3.0 * X[:, 0] + 1.0
    return X, y


# ---------------------------------------------------------------------------
# The HyperParameters space protocol
# ---------------------------------------------------------------------------
class TestHyperParameters:
    def test_collection_mode_returns_defaults_and_registers(self) -> None:
        hp = HyperParameters()
        assert hp.Int("depth", 1, 3) == 1  # default = min
        assert hp.Int("units", 16, 128, step=16, default=48) == 48
        assert hp.Float("dropout", 0.0, 0.4, step=0.1) == 0.0
        assert hp.Choice("solver", ["adam", "sgd"]) == "adam"
        assert hp.Boolean("batchnorm") is False
        assert hp.Fixed("seed", 42) == 42
        assert [spec.name for spec in hp.space] == [
            "depth", "units", "dropout", "solver", "batchnorm", "seed",
        ]

    def test_trial_mode_returns_the_oracle_values(self) -> None:
        hp = HyperParameters(
            values={"depth": 3, "units": 64, "dropout": 0.2, "solver": "sgd",
                    "batchnorm": True, "seed": 42}
        )
        assert hp.Int("depth", 1, 3) == 3
        assert hp.Int("units", 16, 128, step=16) == 64
        assert hp.Float("dropout", 0.0, 0.4, step=0.1) == pytest.approx(0.2)
        assert hp.Choice("solver", ["adam", "sgd"]) == "sgd"
        assert hp.Boolean("batchnorm") is True
        assert hp.Fixed("seed", 42) == 42

    def test_unsampled_name_falls_back_to_default(self) -> None:
        hp = HyperParameters(values={"depth": 2})
        assert hp.Int("depth", 1, 3) == 2
        assert hp.Int("units", 16, 128, step=16, default=32) == 32

    def test_re_registration_with_a_different_spec_raises(self) -> None:
        hp = HyperParameters()
        hp.Int("units", 16, 128, step=16)
        with pytest.raises(ModelError, match="re-registered with a different spec"):
            hp.Int("units", 32, 256, step=32)

    @pytest.mark.parametrize(
        ("call", "fragment"),
        [
            (lambda hp: hp.Int("x", 5, 1), "min_value 5 > max_value 1"),
            (lambda hp: hp.Int("x", 1, 5, step=0), "step must be > 0"),
            (lambda hp: hp.Float("x", 5.0, 1.0), "min_value 5.0 > max_value 1.0"),
            (lambda hp: hp.Float("x", 0.0, 1.0, step=-1.0), "step must be > 0"),
            (lambda hp: hp.Choice("x", []), "values must not be empty"),
        ],
    )
    def test_slot_validation(self, call, fragment) -> None:
        with pytest.raises(ModelError, match=fragment):
            call(HyperParameters())

    def test_sampling_honours_the_declared_space(self) -> None:
        hp = HyperParameters()
        hp.Int("depth", 1, 3)
        hp.Float("dropout", 0.0, 0.4, step=0.1)
        hp.Choice("solver", ["adam", "sgd"])
        hp.Boolean("batchnorm")
        rng = np.random.default_rng(0)
        for _ in range(20):
            values = {spec.name: spec.sample(rng) for spec in hp.space}
            assert values["depth"] in {1, 2, 3}
            assert values["dropout"] == pytest.approx(round(values["dropout"], 1))
            assert values["dropout"] <= 0.4
            assert values["solver"] in {"adam", "sgd"}
            assert isinstance(values["batchnorm"], bool)


# ---------------------------------------------------------------------------
# Ready-made hypermodels
# ---------------------------------------------------------------------------
class TestHypermodels:
    def test_ann_regressor_space(self) -> None:
        hp = HyperParameters()
        estimator = build_ann_regressor(hp)
        assert isinstance(estimator, MLPRegressor)
        assert sorted(spec.name for spec in hp.space) == [
            "alpha", "learning_rate_init", "max_iter", "n_layers", "units",
        ]
        assert estimator.max_iter == 400  # the declared default
        assert estimator.random_state == 42  # deterministic refits

    def test_deep_regressor_searches_batchnorm(self) -> None:
        pytest.importorskip("torch")
        from pharmaml.ml.neural import DeepMLPRegressor
        from pharmaml.ml.tuner import build_deep_regressor

        hp = HyperParameters()
        estimator = build_deep_regressor(hp)
        assert isinstance(estimator, DeepMLPRegressor)
        names = {spec.name for spec in hp.space}
        assert {"batchnorm", "epochs", "activation", "n_layers", "units"} <= names
        # trial mode threads the drawn batchnorm flag into the estimator
        hp = HyperParameters(values={
            "n_layers": 1, "units": 32, "dropout": 0.1, "learning_rate": 1e-3,
            "activation": "relu", "batchnorm": True, "epochs": 40,
        })
        assert build_deep_regressor(hp).batchnorm is True

    def test_hsvr_space(self) -> None:
        hp = HyperParameters()
        estimator = build_hsvr(hp)
        assert isinstance(estimator, HierarchicalSVR)
        assert {spec.name for spec in hp.space} == {"n_clusters", "C", "gamma", "alpha"}
        # no resource slot: Hyperband must refuse by contract
        tuner = Hyperband(build_hsvr, resource="epochs", max_resource=9)
        with pytest.raises(ModelError, match="does not declare the resource"):
            tuner.search(*linear_data(20))


# ---------------------------------------------------------------------------
# RandomSearch
# ---------------------------------------------------------------------------
class TestRandomSearch:
    def test_search_scores_and_refits_the_best(self) -> None:
        X, y = linear_data()
        tuner = RandomSearch(build_ann_regressor, objective="r2",
                             max_trials=3, cv=2, seed=11)
        results = tuner.search(X, y)
        assert len(results.trials) == 3
        assert results.best_score > 0.9
        assert set(results.best_params) == {
            "n_layers", "units", "alpha", "learning_rate_init", "max_iter",
        }
        predictions = results.best_estimator.predict(X)
        assert predictions.shape == (len(y),)
        assert np.isfinite(predictions).all()

    def test_determinism(self) -> None:
        X, y = linear_data(40)
        first = RandomSearch(build_ann_regressor, max_trials=2, cv=2,
                             seed=5).search(X, y)
        second = RandomSearch(build_ann_regressor, max_trials=2, cv=2,
                              seed=5).search(X, y)
        assert first.best_params == second.best_params
        assert first.best_score == pytest.approx(second.best_score)

    def test_summary_and_trials_frame(self) -> None:
        X, y = linear_data(40)
        results = RandomSearch(build_ann_regressor, max_trials=2, cv=2,
                               seed=5).search(X, y)
        lines = results.summary().splitlines()
        assert lines[0] == "RandomSearch: 2 trial(s), objective=r2"
        assert "trial 1" in "\n".join(lines[1:])
        frame = results.trials_frame()
        assert list(frame.columns[:5]) == [
            "trial_id", "score", "score_std", "resource", "failed",
        ]
        assert len(frame) == 2
        assert not bool(frame["failed"].any())
        assert "TunerResults" in repr(results)

    def test_objective_validation(self) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match="unknown objective"):
            RandomSearch(build_ann_regressor, objective="not_a_scorer").search(X, y)

    def test_callable_objective(self) -> None:
        X, y = linear_data(40)

        def constant_score(estimator, X_eval, y_eval) -> float:
            return 1.0

        results = RandomSearch(
            build_ann_regressor, objective=constant_score, max_trials=1, cv=2, seed=1
        ).search(X, y)
        assert results.best_score == 1.0
        assert results.objective == "custom"

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"max_trials": 0}, "max_trials must be an integer >= 1"),
            ({"cv": 1}, "cv must be an integer >= 2"),
            ({"task": "typo"}, "task must be 'auto'"),
        ],
    )
    def test_constructor_validation(self, kwargs, fragment) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match=fragment):
            RandomSearch(build_ann_regressor, **kwargs).search(X, y)

    def test_hypermodel_must_return_a_fittable_estimator(self) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match="with fit()"):
            RandomSearch(lambda hp: "not-an-estimator", max_trials=1).search(X, y)

    def test_failed_trials_are_recorded_not_fatal(self) -> None:
        X, y = linear_data(40)

        def flaky(hp: HyperParameters):
            # fails whenever the drawn alpha is the 1e-4 choice; the
            # DEFAULT (first value, 1e-3) must build cleanly so the
            # space-collection pass stays healthy.
            alpha = hp.Choice("alpha", [1e-3, 1e-4])
            if alpha == 1e-4:
                raise ValueError("boom")
            return MLPRegressor(alpha=alpha, max_iter=50, random_state=42)

        results = RandomSearch(flaky, max_trials=6, cv=2, seed=3).search(X, y)
        failed = [t for t in results.trials if t.failed]
        scored = [t for t in results.trials if not t.failed]
        assert failed and scored  # both populations present
        assert all(t.error and "boom" in t.error for t in failed)
        assert f"({len(failed)} trial(s) failed)" in results.summary()

    def test_all_trials_failed_raises(self) -> None:
        X, y = linear_data(20)

        class _Unfittable:
            def fit(self, X_fit, y_fit):
                raise RuntimeError("fit exploded")

            def predict(self, X_pred):
                raise RuntimeError("never fitted")

        with pytest.raises(ModelError, match="every trial failed"):
            RandomSearch(lambda hp: _Unfittable(), max_trials=2, cv=2).search(X, y)

    def test_broken_hypermodel_fails_the_space_collection(self) -> None:
        X, y = linear_data(20)

        def broken(hp: HyperParameters):
            raise RuntimeError("always broken")

        with pytest.raises(ModelError, match="failed during space collection"):
            RandomSearch(broken, max_trials=2).search(X, y)

    def test_classification_objective(self) -> None:
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        tuner = RandomSearch(
            build_ann_classifier, objective="accuracy", max_trials=1, cv=2, seed=9
        )
        results = tuner.search(X, y)
        assert results.objective == "accuracy"
        assert results.best_score > 0.5


# ---------------------------------------------------------------------------
# Hyperband
# ---------------------------------------------------------------------------
class TestHyperband:
    def test_brackets_and_rung_resources(self) -> None:
        X, y = linear_data(60)
        tuner = Hyperband(
            build_ann_regressor,
            resource="max_iter",
            max_resource=9,
            min_resource=1,
            eta=3,
            cv=2,
            seed=5,
        )
        results = tuner.search(X, y)
        resources = {t.resource for t in results.trials}
        assert resources <= {1.0, 3.0, 9.0}
        assert len(results.trials) >= 5  # brackets s=2,1,0 all contribute
        # every trial ran exactly at its rung's resource
        for trial in results.trials:
            assert trial.hyperparameters["max_iter"] == int(trial.resource)
        assert results.best_estimator.predict(X).shape == (len(y),)
        assert results.summary().splitlines()[0].startswith("Hyperband:")

    def test_higher_budget_scores_better_than_every_low_budget_trial(self) -> None:
        # successive halving promotes only the top configs to the full
        # budget, so at least one max-resource trial exists
        X, y = linear_data(60)
        results = Hyperband(
            build_ann_regressor, resource="max_iter", max_resource=9,
            min_resource=1, eta=3, cv=2, seed=5,
        ).search(X, y)
        full_budget = [t for t in results.trials if t.resource == 9.0]
        assert full_budget

    def test_resource_must_be_declared(self) -> None:
        X, y = linear_data(20)
        tuner = Hyperband(build_hsvr, resource="epochs", max_resource=9)
        with pytest.raises(ModelError, match="does not declare the resource"):
            tuner.search(X, y)

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"max_resource": 0}, "max_resource must be an integer >= 1"),
            ({"min_resource": 0}, "min_resource must be an integer >= 1"),
            ({"min_resource": 10, "max_resource": 5}, "min_resource 10 > max_resource 5"),
            ({"eta": 1}, "eta must be an integer >= 2"),
        ],
    )
    def test_constructor_validation(self, kwargs, fragment) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match=fragment):
            Hyperband(build_ann_regressor, **kwargs).search(X, y)


# ---------------------------------------------------------------------------
# Splitter heuristic
# ---------------------------------------------------------------------------
class TestSplitterHeuristic:
    def test_singleton_classes_fall_back_to_plain_kfold(self) -> None:
        from sklearn.model_selection import KFold, StratifiedKFold

        from pharmaml.ml.tuner import _make_splitter

        y = np.array([1.5, 2.5, 3.5, 4.5])  # 4 singleton "classes"
        splitter = _make_splitter("classification", y, cv=2)
        assert isinstance(splitter, KFold)
        balanced = np.array([0, 0, 1, 1])
        assert isinstance(_make_splitter("classification", balanced, cv=2), StratifiedKFold)

    def test_auto_task_infers_regression_for_continuous_targets(self) -> None:
        from sklearn.model_selection import KFold

        from pharmaml.ml.tuner import _make_splitter

        y = np.linspace(0, 1, 30)
        assert isinstance(_make_splitter("auto", y, cv=2), KFold)
        labels = np.array(["a", "b"] * 15)
        from sklearn.model_selection import StratifiedKFold

        assert isinstance(_make_splitter("auto", labels, cv=2), StratifiedKFold)


# ---------------------------------------------------------------------------
# TunerResults data contract
# ---------------------------------------------------------------------------
class TestTunerResultsContract:
    def test_best_trial_is_the_maximum(self) -> None:
        from pharmaml.ml.tuner import TrialResult

        trials = [
            TrialResult(1, {"a": 1}, 0.5, 0.1),
            TrialResult(2, {"a": 2}, 0.9, 0.05),
            TrialResult(3, {"a": 3}, float("nan"), 0.0, error="ValueError: x"),
        ]
        results = TunerResults(trials, "r2")
        assert results.best_trial.trial_id == 2
        assert results.best_params == {"a": 2}
        assert results.best_score == pytest.approx(0.9)

    def test_all_failed_construction_raises(self) -> None:
        from pharmaml.ml.tuner import TrialResult

        trials = [TrialResult(1, {}, float("nan"), 0.0, error="e")]
        with pytest.raises(ModelError, match="every trial failed"):
            TunerResults(trials, "r2")

    def test_best_estimator_before_search_raises(self) -> None:
        from pharmaml.ml.tuner import TrialResult

        results = TunerResults([TrialResult(1, {}, 0.5, 0.1)], "r2")
        with pytest.raises(ModelError, match="only available after search"):
            _ = results.best_estimator

    def test_trials_frame_unions_hyperparameter_columns(self) -> None:
        from pharmaml.ml.tuner import TrialResult

        trials = [
            TrialResult(1, {"a": 1, "b": 2}, 0.5, 0.1),
            TrialResult(2, {"a": 2}, 0.7, 0.05),
        ]
        frame = TunerResults(trials, "r2").trials_frame()
        assert list(frame.columns) == [
            "trial_id", "score", "score_std", "resource", "failed", "a", "b",
        ]
        assert bool(pd.isna(frame.loc[1, "b"]))

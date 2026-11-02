"""Tests for the keras-tuner-style search (``pharmaml.ml.tuner``).

Covers the ``HyperParameters`` space protocol (collection mode, trial
mode, re-registration, log-domain floats), the ready-made hypermodels
(ANN / deep / HSVR), ``RandomSearch`` (scoring, determinism,
best-estimator refit, failure accounting), ``TPESearch`` (the Optuna
Parzen below/above machinery, startup behaviour, gamma), the
``PercentilePruner`` / ``MedianPruner`` fold loop, the Optuna trial
states (complete/pruned/fail), ``param_importances`` (the MDI
evaluator), ``Hyperband`` (brackets, rung resources, the resource
declaration contract) and the public ``make_cv_splitter`` fold
protocol + ``resource_bounds`` ladder helper.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPRegressor

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.hierarchical import HierarchicalSVR
from pharmaml.ml.models import MODEL_REGISTRY
from pharmaml.ml.tuner import (
    Hyperband,
    HyperParameters,
    MedianPruner,
    PercentilePruner,
    RandomSearch,
    TPESearch,
    TrialResult,
    TunerResults,
    _below_above_split,
    _gamma,
    _hyperband_total_trials,
    build_ann_classifier,
    build_ann_regressor,
    build_hsvr,
    hypermodel_by_name,
    make_cv_splitter,
    param_importances,
    registry_hypermodel,
    registry_hypermodels,
    resource_bounds,
)


def plt_close(figure: Any) -> None:
    """Close a matplotlib figure (the Agg-suite discipline)."""
    import matplotlib.pyplot as plt

    plt.close(figure)


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
# (100x-169) the registry-grid hypermodels
# ---------------------------------------------------------------------------
class TestRegistryHypermodels:
    """Every grid-bearing sklearn registry family is searchable.

    The Tune surfaces (CLI + card) used to offer ann/hsvr/dnn only;
    the factory here derives a hypermodel from the registry's OWN
    param_grid, so the keras-tuner-style searchers cover the whole
    sklearn zoo.
    """

    def test_factory_space_mirrors_the_registry_grid(self) -> None:
        builder = registry_hypermodel("random_forest", "regression")
        assert builder.__name__ == "build_random_forest_regressor"
        hp = HyperParameters()
        estimator = builder(hp)
        from sklearn.ensemble import RandomForestRegressor

        assert isinstance(estimator, RandomForestRegressor)
        grid = MODEL_REGISTRY["random_forest"].param_grid
        assert sorted(spec.name for spec in hp.space) == sorted(grid)
        assert estimator.random_state == 42  # deterministic refits
        # collection mode returns each slot's first grid value
        assert estimator.n_estimators == grid["n_estimators"][0]

    def test_trial_mode_threads_drawn_values(self) -> None:
        builder = registry_hypermodel("knn", "classification")
        hp = HyperParameters(values={"n_neighbors": 11, "weights": "distance"})
        estimator = builder(hp)
        from sklearn.neighbors import KNeighborsClassifier

        assert isinstance(estimator, KNeighborsClassifier)
        assert estimator.n_neighbors == 11
        assert estimator.weights == "distance"

    def test_unknown_family_and_task_refused(self) -> None:
        with pytest.raises(ModelError, match="Unknown model"):
            registry_hypermodel("banana", "regression")
        with pytest.raises(ModelError, match="supports"):
            registry_hypermodel("ridge", "classification")

    def test_random_search_over_a_registry_family(self) -> None:
        X, y = linear_data(60)
        tuner = RandomSearch(
            registry_hypermodel("ridge", "regression"), objective="r2",
            max_trials=3, cv=2, seed=42,
        )
        results = tuner.search(X, y)
        assert len(results.trials) == 3
        alphas = MODEL_REGISTRY["ridge"].param_grid["alpha"]
        assert results.best_params["alpha"] in alphas
        assert results.best_score > 0.9  # a linear surface is learnable

    def test_by_name_resolution(self) -> None:
        builder = hypermodel_by_name("build_knn_classifier")
        assert builder is registry_hypermodel("knn", "classification")
        assert hypermodel_by_name("build_hsvr") is build_hsvr
        with pytest.raises(ModelError, match="no hypermodel named"):
            hypermodel_by_name("build_nope_regressor")

    def test_coverage_pin(self) -> None:
        mapping = registry_hypermodels()
        # every token resolves in the model registry
        tokens = {key[0] for key in mapping}
        assert tokens <= set(MODEL_REGISTRY)
        # representative members of each exclusion class stay out
        for token in (
            "voting", "stacking",               # meta-ensembles
            "gaussian_process",                  # kernel-OBJECT grids
            "gaussian_process_classifier",
            "linear_regression",                 # empty grid
            "ann", "dnn", "hsvr",               # bespoke hypermodels
            "dnn_residual", "dnn_numpy",         # the dnn family twins
        ):
            assert token not in tokens
        # ...while every OTHER grid-bearing family is in, with the
        # task pairing the registry itself declares
        for key, spec in MODEL_REGISTRY.items():
            if key in {
                "voting", "stacking", "gaussian_process",
                "gaussian_process_classifier", "linear_regression",
                "ann", "dnn", "hsvr", "dnn_residual", "dnn_numpy",
            }:
                continue
            if not spec.param_grid:
                continue
            tasks = ("regression", "classification") if spec.task == "both" else (spec.task,)
            for task in tasks:
                assert (key, task) in mapping, (key, task)

    def test_names_follow_the_regressor_classifier_convention(self) -> None:
        mapping = registry_hypermodels()
        assert mapping[("ridge", "regression")].__name__ == "build_ridge_regressor"
        assert (
            mapping[("logistic_regression", "classification")].__name__
            == "build_logistic_regression_classifier"
        )

    def test_booster_family_searchable_when_installed(self) -> None:
        pytest.importorskip("xgboost")
        X, y = linear_data(48)
        tuner = RandomSearch(
            registry_hypermodel("xgboost", "regression"), objective="r2",
            max_trials=2, cv=2, seed=7,
        )
        results = tuner.search(X, y)
        assert len(results.trials) == 2
        assert "n_estimators" in results.best_params


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

    def test_on_trial_progress_payloads(self) -> None:
        # 100x-106: one callback per trial, in order, carrying the
        # progress-bar contract (trial/total) plus the running best; the
        # Optuna round (100x-160) added 'state' + 'fold_scores' on top.
        X, y = linear_data(60)
        events: list[dict[str, Any]] = []
        tuner = RandomSearch(
            build_ann_regressor, objective="r2", max_trials=3, cv=2, seed=11,
            on_trial=events.append,
        )
        results = tuner.search(X, y)
        assert len(events) == 3 == len(results.trials)
        assert [event["trial"] for event in events] == [1, 2, 3]
        assert all(event["total"] == 3 for event in events)
        assert all(set(event) == {
            "trial", "total", "score", "best_score", "error",
            "hyperparameters", "state", "fold_scores",
        } for event in events)
        assert all(event["state"] == "complete" for event in events)
        assert all(event["fold_scores"] is None for event in events)  # no pruner
        assert events[0]["best_score"] is not None  # trial 1 set the best
        assert events[-1]["best_score"] == pytest.approx(results.best_score)


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

    def test_on_trial_total_matches_the_real_trial_count(self) -> None:
        # 100x-106: the schedule mirror must equal the run's own
        # evaluation count exactly -- the ETA fraction depends on it.
        X, y = linear_data(60)
        events: list[dict[str, Any]] = []
        tuner = Hyperband(
            build_ann_regressor, resource="max_iter", max_resource=9,
            min_resource=1, eta=3, cv=2, seed=5, on_trial=events.append,
        )
        results = tuner.search(X, y)
        assert len(events) == len(results.trials)
        total = events[0]["total"]
        assert total == _hyperband_total_trials(3, 9, 1)
        assert [event["trial"] for event in events] == list(range(1, total + 1))
        assert events[-1]["trial"] == total
        assert all(set(event) == {
            "trial", "total", "score", "best_score", "error", "resource",
        } for event in events)

    def test_hyperband_total_trials_schedule_mirror(self) -> None:
        # the classic textbook grid: eta=3, R=81, r_min=3 -> 70 trials
        assert _hyperband_total_trials(3, 81, 3) == 70
        assert _hyperband_total_trials(3, 27, 3) == 23
        assert _hyperband_total_trials(2, 8, 8) == 1  # degenerate: one rung
        assert _hyperband_total_trials(4, 4, 1) == 7  # s_max=1: (2) + (4+1)


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
# make_cv_splitter (the public fold protocol, 100x-90)
# ---------------------------------------------------------------------------
class TestMakeCvSplitter:
    def test_delegates_to_the_tuners_own_protocol(self) -> None:
        """The public helper and the tuners' internal rule agree everywhere."""
        from sklearn.model_selection import KFold, StratifiedKFold

        from pharmaml.ml.tuner import _make_splitter, make_cv_splitter

        cases = [
            ("regression", np.linspace(0.0, 1.0, 30), 3),
            ("classification", np.array(["a", "b"] * 15), 2),
            ("classification", np.array([1.5, 2.5, 3.5, 4.5]), 2),  # singletons
            ("auto", np.linspace(0.0, 1.0, 30), 5),
            ("auto", np.array(["a", "b"] * 15), 4),
        ]
        for task, y, cv in cases:
            public = make_cv_splitter(task, y, cv)
            private = _make_splitter(task, np.asarray(y), cv)
            assert type(public) is type(private), (task, cv)
            assert public.n_splits == private.n_splits
            assert public.random_state == private.random_state == 0
            assert public.shuffle is True
        assert isinstance(make_cv_splitter("regression", np.arange(30.0), 3), KFold)
        assert isinstance(
            make_cv_splitter("classification", ["a"] * 10 + ["b"] * 10, 2),
            StratifiedKFold,
        )

    def test_accepts_list_targets_and_validates_arguments(self) -> None:
        from pharmaml.ml.tuner import make_cv_splitter as _mcs

        # lists (the GUI passes a pandas Series / list) are asarray'd fine
        splitter = _mcs("regression", [float(i) for i in range(20)], 4)
        assert splitter.n_splits == 4
        with pytest.raises(ModelError, match="cv must be an integer >= 2"):
            _mcs("regression", np.arange(10.0), 1)
        with pytest.raises(ModelError, match="cv must be an integer >= 2"):
            _mcs("regression", np.arange(10.0), cv=True)
        with pytest.raises(ModelError, match="task must be"):
            _mcs("regress", np.arange(10.0), 3)

    def test_exported_from_the_ml_package(self) -> None:
        import pharmaml.ml

        assert pharmaml.ml.make_cv_splitter is make_cv_splitter
        assert "make_cv_splitter" in pharmaml.ml.__all__


# ---------------------------------------------------------------------------
# resource_bounds — the GUI's Hyperband ladder (100x-92)
# ---------------------------------------------------------------------------
class TestResourceBounds:
    def test_shipped_families_pin_their_declared_grids(self) -> None:
        from pharmaml.ml.tuner import (
            build_deep_classifier,
            build_deep_regressor,
        )

        # the CLI's resource rule (max_iter for ann, epochs for dnn)
        # backed by each family's own declared grid
        assert resource_bounds(build_ann_regressor, "max_iter") == (200, 1000)
        assert resource_bounds(build_ann_classifier, "max_iter") == (200, 1000)
        assert resource_bounds(build_deep_regressor, "epochs") == (30, 150)
        assert resource_bounds(build_deep_classifier, "epochs") == (30, 150)

    def test_missing_slot_raises_the_actionable_error(self) -> None:
        # hsvr declares no resource slot -- the library and the GUI's
        # pre-flight guard agree there is nothing to halve.
        with pytest.raises(
            ModelError, match="does not declare the resource hyperparameter 'epochs'"
        ):
            resource_bounds(build_hsvr, "epochs")

    def test_non_int_slots_are_rejected(self) -> None:
        from sklearn.linear_model import Ridge

        def float_resource(hp: HyperParameters) -> object:
            hp.Float("budget", 0.5, 2.5)
            return Ridge()

        with pytest.raises(ModelError, match="must be an hp.Int slot"):
            resource_bounds(float_resource, "budget")

    def test_the_declared_grid_sizes_a_real_hyperband(self) -> None:
        X, y = linear_data(60)
        r_min, r_max = resource_bounds(build_ann_regressor, "max_iter")
        results = Hyperband(
            build_ann_regressor, resource="max_iter",
            max_resource=r_max, min_resource=r_min, eta=3, cv=2, seed=5,
        ).search(X, y)
        # the hand-computed schedule: s_max = floor(log(1000/200, 3)) = 1
        # -> bracket 1 runs 3 trials @333 + 1 promoted @999, bracket 0
        # runs 2 @1000 -- six trials, three rung resources, every rung
        # inside the declared space.
        assert len(results.trials) == 6
        resources = sorted({int(t.resource) for t in results.trials})
        assert resources == [333, 999, 1000]
        for trial in results.trials:
            assert 200 <= int(trial.resource) <= 1000
            assert trial.hyperparameters["max_iter"] == int(trial.resource)

    def test_exported_from_the_ml_package(self) -> None:
        import pharmaml.ml

        assert pharmaml.ml.resource_bounds is resource_bounds
        assert "resource_bounds" in pharmaml.ml.__all__


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
            "state", "duration", "n_folds",
        ]
        assert bool(pd.isna(frame.loc[1, "b"]))
        assert list(frame["state"]) == ["complete", "complete"]
        assert list(frame["n_folds"]) == [0, 0]


# ---------------------------------------------------------------------------
# Log-domain floats (100x-160)
# ---------------------------------------------------------------------------
class TestLogFloat:
    def test_log_draws_stay_inside_the_positive_bounds(self) -> None:
        hp = HyperParameters()
        hp.Float("lr", 1e-5, 1e-1, log=True)
        spec = hp.space[-1]
        rng = np.random.default_rng(0)
        draws = [spec.sample(rng) for _ in range(200)]
        assert all(1e-5 <= v <= 1e-1 for v in draws)
        assert all(v > 0.0 for v in draws)  # none at 0, none negative

    def test_log_draws_mass_concentrates_low(self) -> None:
        # exp(uniform(log low, log high)) puts the mean of the LOGS at
        # the geometric midpoint, well below the arithmetic one.
        hp = HyperParameters()
        hp.Float("lr", 1e-5, 1e-1, log=True)
        rng = np.random.default_rng(1)
        draws = [hp.space[-1].sample(rng) for _ in range(400)]
        log_mid = 0.5 * (np.log(1e-5) + np.log(1e-1))
        assert float(np.mean(np.log(draws))) == pytest.approx(log_mid, abs=0.2)

    def test_log_sampling_is_deterministic(self) -> None:
        hp = HyperParameters()
        hp.Float("lr", 1e-5, 1e-1, log=True)
        first = [hp.space[-1].sample(np.random.default_rng(3)) for _ in range(10)]
        second = [hp.space[-1].sample(np.random.default_rng(3)) for _ in range(10)]
        assert first == second

    def test_step_with_log_is_refused(self) -> None:
        with pytest.raises(ModelError, match="log=True requires step=None"):
            HyperParameters().Float("lr", 1e-5, 1e-1, step=0.1, log=True)

    def test_non_positive_bounds_are_refused(self) -> None:
        with pytest.raises(ModelError, match="log=True requires min_value > 0"):
            HyperParameters().Float("lr", 0.0, 1.0, log=True)
        with pytest.raises(ModelError, match="log=True requires min_value > 0"):
            HyperParameters().Float("lr", -1.0, 1.0, log=True)

    def test_log_spec_is_distinguishable_on_re_registration(self) -> None:
        hp = HyperParameters()
        hp.Float("lr", 1e-5, 1e-1, log=True)
        with pytest.raises(ModelError, match="re-registered with a different spec"):
            hp.Float("lr", 1e-5, 1e-1, log=False)


# ---------------------------------------------------------------------------
# TPESearch (100x-160)
# ---------------------------------------------------------------------------
class TestTPESearch:
    def test_search_scores_a_learnable_surface(self) -> None:
        X, y = linear_data()
        tuner = TPESearch(
            build_ann_regressor, objective="r2", max_trials=5, cv=2,
            seed=11, n_startup_trials=2,
        )
        results = tuner.search(X, y)
        assert len(results.trials) == 5
        assert results.best_score > 0.9
        assert set(results.best_params) == {
            "n_layers", "units", "alpha", "learning_rate_init", "max_iter",
        }
        assert results.best_estimator.predict(X).shape == (len(y),)

    def test_determinism_twin(self) -> None:
        X, y = linear_data(40)
        first = TPESearch(
            build_ann_regressor, max_trials=4, cv=2, seed=5, n_startup_trials=1
        ).search(X, y)
        second = TPESearch(
            build_ann_regressor, max_trials=4, cv=2, seed=5, n_startup_trials=1
        ).search(X, y)
        assert first.best_params == second.best_params
        assert [t.hyperparameters for t in first.trials] == [
            t.hyperparameters for t in second.trials
        ]
        assert [t.score for t in first.trials] == pytest.approx(
            [t.score for t in second.trials]
        )

    def test_startup_trials_reproduce_random_search(self) -> None:
        # n < n_startup_trials: every draw is a uniform spec.sample(rng),
        # byte-identical to RandomSearch's own sequence for the seed.
        X, y = linear_data(40)
        rand = RandomSearch(
            build_ann_regressor, max_trials=3, cv=2, seed=5
        ).search(X, y)
        tpe = TPESearch(
            build_ann_regressor, max_trials=3, cv=2, seed=5, n_startup_trials=10
        ).search(X, y)
        assert [t.hyperparameters for t in tpe.trials] == [
            t.hyperparameters for t in rand.trials
        ]

    def test_zero_startup_exercises_the_parzen_machinery(self) -> None:
        X, y = linear_data(40)
        tpe = TPESearch(
            build_ann_regressor, max_trials=2, cv=2, seed=5, n_startup_trials=0
        ).search(X, y)
        assert len(tpe.trials) == 2
        assert all(t.resolved_state == "complete" for t in tpe.trials)
        # the Parzen machinery consumes the rng differently from the
        # uniform draw, so the sequence must NOT be RandomSearch's
        rand = RandomSearch(
            build_ann_regressor, max_trials=2, cv=2, seed=5
        ).search(X, y)
        assert [t.hyperparameters for t in tpe.trials] != [
            t.hyperparameters for t in rand.trials
        ]

    def test_gamma_helper_pins_optunas_rule(self) -> None:
        assert _gamma(0) == 0
        assert _gamma(1) == 1
        assert _gamma(9) == 1
        assert _gamma(10) == 1
        assert _gamma(11) == 2
        assert _gamma(59) == 6
        assert _gamma(60) == 6
        assert _gamma(249) == 25
        assert _gamma(250) == 25  # the default cap clamps
        assert _gamma(1000) == 25
        assert _gamma(60, 5) == 5  # a tighter cap clamps earlier

    def test_below_above_split_over_sixty_fake_trials(self) -> None:
        trials = [TrialResult(i, {"x": i}, float(i), 0.0) for i in range(1, 61)]
        below, above = _below_above_split(trials, _gamma(60))
        assert len(below) == 6
        assert len(above) == 54
        # both sets come back in trial order (Optuna sorts its split)
        assert [t.trial_id for t in below] == [55, 56, 57, 58, 59, 60]
        assert [t.trial_id for t in above] == list(range(1, 55))
        # ties keep the earlier trial (stable sort over trial order)
        tied = [TrialResult(i, {}, 1.0, 0.0) for i in range(1, 4)]
        below, above = _below_above_split(tied, 1)
        assert [t.trial_id for t in below] == [1]
        assert [t.trial_id for t in above] == [2, 3]

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"max_trials": 0}, "max_trials must be an integer >= 1"),
            ({"n_ei_candidates": 0}, "n_ei_candidates must be an integer >= 1"),
            ({"n_startup_trials": -1}, "n_startup_trials must be an integer >= 0"),
            ({"gamma_cap": 0}, "gamma_cap must be an integer >= 1"),
            ({"cv": 1}, "cv must be an integer >= 2"),
            ({"task": "typo"}, "task must be 'auto'"),
        ],
    )
    def test_constructor_validation(self, kwargs, fragment) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match=fragment):
            TPESearch(build_ann_regressor, **kwargs).search(X, y)

    def test_unknown_objective_validation(self) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match="unknown objective"):
            TPESearch(build_ann_regressor, objective="not_a_scorer").search(X, y)

    def test_failed_trials_never_inform_the_sampler(self) -> None:
        # a flaky hypermodel: the failing draws never enter below/above
        # (Optuna's COMPLETE+PRUNED count), the search still completes
        X, y = linear_data(40)

        def flaky(hp: HyperParameters):
            quality = hp.Float("quality", 0.0, 1.0, default=0.5)
            if quality < 0.2:
                raise ValueError("boom")
            return _QualityRegressor(quality)

        results = TPESearch(
            flaky, objective=_quality_score, max_trials=6, cv=2, seed=2,
            n_startup_trials=2,
        ).search(X, y)
        failed = [t for t in results.trials if t.failed]
        assert all("boom" in (t.error or "") for t in failed)
        assert results.best_score == pytest.approx(
            max(t.score for t in results.trials if not t.failed)
        )


# ---------------------------------------------------------------------------
# The deterministic quality hypermodel (pruning / TPE test surface)
# ---------------------------------------------------------------------------
class _QualityRegressor:
    """Estimator whose cross-validated score is exactly its knob.

    Implements the sklearn clone protocol (``get_params`` + a
    parameterless-ish constructor) so the ``cross_val_score`` fast path
    can duplicate it per fold.
    """

    def __init__(self, quality: float = 0.5) -> None:
        self.quality = quality

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"quality": self.quality}

    def set_params(self, **params: Any) -> _QualityRegressor:
        if "quality" in params:
            self.quality = float(params["quality"])
        return self

    def fit(self, X, y):  # dummy estimator
        return self

    def predict(self, X):  # dummy estimator
        return np.full(len(X), self.quality)


def _quality_hypermodel(hp: HyperParameters) -> Any:
    """A hypermodel whose cv score is deterministic per config."""
    quality = hp.Float("quality", 0.0, 1.0)
    return _QualityRegressor(quality)


def _quality_score(estimator: Any, X: Any, y: Any) -> float:
    """The custom objective: the estimator's own quality knob."""
    return float(estimator.quality)


# ---------------------------------------------------------------------------
# Pruners (100x-160)
# ---------------------------------------------------------------------------
class TestPruners:
    def test_below_startup_never_prunes(self) -> None:
        pruner = PercentilePruner(percentile=50.0, n_startup_trials=5)
        assert pruner.should_prune([[0.9], [0.8]], [0.1]) is False

    def test_same_step_percentile_beats_prunes(self) -> None:
        pruner = PercentilePruner(percentile=50.0, n_startup_trials=1)
        completed = [[0.9], [0.7]]  # median at step 1 = 0.8
        assert pruner.should_prune(completed, [0.5]) is True
        assert pruner.should_prune(completed, [0.85]) is False

    def test_only_curves_long_enough_count(self) -> None:
        pruner = PercentilePruner(percentile=50.0, n_startup_trials=1)
        completed = [[0.9, 0.92], [0.7]]  # only the first reaches step 2
        assert pruner.should_prune(completed, [0.8, 0.5]) is True  # p=0.92
        # step 3: no completed curve reaches it -> nothing to compare
        assert pruner.should_prune(completed, [0.8, 0.5, 0.1]) is False

    def test_n_min_trials_guard(self) -> None:
        pruner = PercentilePruner(percentile=50.0, n_startup_trials=1, n_min_trials=2)
        completed = [[0.9, 0.92], [0.7]]  # one value at step 2 only
        assert pruner.should_prune(completed, [0.8, 0.1]) is False
        completed = [[0.9, 0.92], [0.7, 0.72]]  # now two values at step 2
        assert pruner.should_prune(completed, [0.1, 0.1]) is True

    def test_running_best_not_last_fold_compares(self) -> None:
        # the rule uses max(current), so a recovering trial survives
        pruner = PercentilePruner(percentile=50.0, n_startup_trials=1)
        completed = [[0.9], [0.7]]
        assert pruner.should_prune(completed, [0.1, 0.9]) is False

    def test_all_nan_same_step_does_not_prune(self) -> None:
        pruner = PercentilePruner(n_startup_trials=1)
        completed = [[float("nan")], [float("nan")]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert pruner.should_prune(completed, [0.1]) is False

    def test_median_pruner_is_percentile_50(self) -> None:
        median = MedianPruner(n_startup_trials=1)
        assert median.percentile == 50.0
        p50 = PercentilePruner(percentile=50.0, n_startup_trials=1)
        completed = [[0.9], [0.7], [0.5]]
        for current in ([0.1], [0.6], [0.8], [0.95]):
            assert median.should_prune(completed, current) is (
                p50.should_prune(completed, current)
            )

    def test_guards(self) -> None:
        with pytest.raises(ModelError, match=r"percentile must be in \(0, 100\]"):
            PercentilePruner(percentile=0.0)
        with pytest.raises(ModelError, match=r"percentile must be in \(0, 100\]"):
            PercentilePruner(percentile=101.0)
        with pytest.raises(ModelError, match="n_startup_trials must be an integer >= 1"):
            PercentilePruner(n_startup_trials=0)
        with pytest.raises(ModelError, match="n_min_trials must be an integer >= 1"):
            PercentilePruner(n_min_trials=0)
        PercentilePruner(percentile=100.0)  # the closed endpoint is allowed

    def test_bad_pruner_type_is_refused_by_the_tuners(self) -> None:
        X, y = linear_data(20)
        with pytest.raises(ModelError, match="pruner must be a PercentilePruner"):
            RandomSearch(_quality_hypermodel, pruner="median").search(X, y)
        with pytest.raises(ModelError, match="pruner must be a PercentilePruner"):
            TPESearch(_quality_hypermodel, pruner=object()).search(X, y)


# ---------------------------------------------------------------------------
# A search with a pruner wired in (100x-160)
# ---------------------------------------------------------------------------
class TestPrunedSearch:
    def test_median_pruner_cuts_losing_trials_early(self) -> None:
        X, y = linear_data(40)
        events: list[dict[str, Any]] = []
        tuner = RandomSearch(
            _quality_hypermodel, objective=_quality_score, max_trials=8, cv=3,
            seed=3, pruner=MedianPruner(n_startup_trials=2), on_trial=events.append,
        )
        results = tuner.search(X, y)

        pruned = [t for t in results.trials if t.pruned]
        assert pruned, "the deterministic surface must fire the pruner"
        for trial in pruned:
            assert math.isnan(trial.score)
            assert math.isnan(trial.score_std)
            assert trial.error is None
            assert not trial.failed
            assert trial.fold_scores is not None
            assert 1 <= len(trial.fold_scores) < 3  # partial, not full cv

        # pruned trials are out of the best-trial race but in the frame
        assert all(results.best_trial.trial_id != t.trial_id for t in pruned)
        frame = results.trials_frame()
        assert set(frame["state"]) >= {"complete", "pruned"}
        assert int(frame.loc[frame["state"] == "pruned", "n_folds"].iloc[0]) < 3

        # the payload carries state + fold scores
        assert len(events) == len(results.trials)
        assert all({"state", "fold_scores"} <= set(event) for event in events)
        pruned_events = [e for e in events if e["state"] == "pruned"]
        assert pruned_events
        assert all(
            e["fold_scores"] is not None and len(e["fold_scores"]) < 3
            for e in pruned_events
        )

        # durations recorded
        assert all(
            t.duration is not None and t.duration >= 0.0 for t in results.trials
        )

        # summary carries the pruned count line (exact format)
        assert f"({len(pruned)} trial(s) pruned by the pruner)" in results.summary()

    def test_pruned_trials_do_not_inform_the_tpe_sampler(self) -> None:
        # TPE with a pruner: pruned trials count towards n (the
        # COMPLETE+PRUNED startup rule) but never enter the mixtures.
        X, y = linear_data(40)
        tuner = TPESearch(
            _quality_hypermodel, objective=_quality_score, max_trials=8, cv=3,
            seed=3, pruner=MedianPruner(n_startup_trials=2), n_startup_trials=3,
        )
        results = tuner.search(X, y)
        pruned = [t for t in results.trials if t.pruned]
        assert pruned
        assert any(t.resolved_state == "complete" for t in results.trials)
        assert results.best_score == pytest.approx(
            max(t.score for t in results.trials if t.resolved_state == "complete")
        )

    def test_no_pruner_keeps_the_fast_path(self) -> None:
        # pruner=None: the cross_val_score fast path — states derive,
        # fold_scores stay None, exactly the legacy behaviour.
        X, y = linear_data(40)
        results = RandomSearch(
            _quality_hypermodel, objective=_quality_score, max_trials=3, cv=3, seed=3
        ).search(X, y)
        assert all(t.state is None for t in results.trials)
        assert all(t.fold_scores is None for t in results.trials)
        assert all(t.resolved_state == "complete" for t in results.trials)

    def test_failing_fold_fails_the_trial_under_a_pruner(self) -> None:
        X, y = linear_data(20)

        def flaky(hp: HyperParameters):
            quality = hp.Float("quality", 0.0, 1.0, default=0.7)
            if quality < 0.5:
                raise ValueError("boom")
            return _QualityRegressor(quality)

        results = RandomSearch(
            flaky, objective=_quality_score, max_trials=6, cv=3, seed=7,
            pruner=MedianPruner(n_startup_trials=1),
        ).search(X, y)
        failed = [t for t in results.trials if t.failed]
        assert failed
        assert all("boom" in (t.error or "") for t in failed)
        assert all(t.resolved_state == "fail" for t in failed)


# ---------------------------------------------------------------------------
# Optuna trial states (100x-160)
# ---------------------------------------------------------------------------
class TestTrialStates:
    def test_legacy_records_derive_their_state(self) -> None:
        assert TrialResult(1, {}, 0.5, 0.1).resolved_state == "complete"
        assert (
            TrialResult(2, {}, float("nan"), 0.0, error="ValueError: x").resolved_state
            == "fail"
        )
        assert TrialResult(3, {}, float("nan"), 0.0).resolved_state == "fail"
        assert TrialResult(4, {}, float("inf"), 0.0).resolved_state == "fail"

    def test_pruned_is_not_failed(self) -> None:
        trial = TrialResult(
            1, {"quality": 0.3}, float("nan"), float("nan"), state="pruned"
        )
        assert trial.pruned is True
        assert trial.failed is False
        assert trial.resolved_state == "pruned"

    def test_explicit_states_win(self) -> None:
        fail = TrialResult(1, {}, float("nan"), 0.0, error="e", state="fail")
        assert (fail.failed, fail.pruned, fail.resolved_state) == (True, False, "fail")
        complete = TrialResult(2, {}, 0.5, 0.0, state="complete")
        assert (complete.failed, complete.pruned) == (False, False)

    def test_pruned_trials_are_excluded_from_the_best_race(self) -> None:
        trials = [
            TrialResult(1, {"a": 1}, 0.5, 0.1),
            TrialResult(2, {"a": 2}, float("nan"), 0.05, state="pruned"),
            TrialResult(3, {"a": 3}, 0.7, 0.02),
        ]
        results = TunerResults(trials, "r2")
        assert results.best_trial.trial_id == 3
        ranked_rows = [
            line for line in results.summary().splitlines()
            if line.startswith("  trial")
        ]
        assert all("trial 2:" not in line for line in ranked_rows)
        assert "(1 trial(s) pruned by the pruner)" in results.summary()

    def test_trials_frame_column_order_pin(self) -> None:
        trials = [
            TrialResult(
                1, {"a": 1}, 0.5, 0.1, state="complete",
                fold_scores=(0.4, 0.6), duration=0.25,
            ),
            TrialResult(2, {"a": 2}, float("nan"), float("nan"), state="pruned"),
        ]
        frame = TunerResults(trials, "r2").trials_frame()
        # the old five columns keep their positions; the Optuna tail is
        # appended after everything (params included)
        assert list(frame.columns[:5]) == [
            "trial_id", "score", "score_std", "resource", "failed",
        ]
        assert list(frame.columns[-3:]) == ["state", "duration", "n_folds"]
        assert list(frame.columns[5:7]) == ["a", "state"]
        assert frame.loc[0, "state"] == "complete"
        assert frame.loc[1, "state"] == "pruned"
        assert int(frame.loc[1, "n_folds"]) == 0
        assert float(frame.loc[0, "duration"]) == 0.25
        assert bool(pd.isna(frame.loc[1, "duration"]))


# ---------------------------------------------------------------------------
# param_importances (100x-160)
# ---------------------------------------------------------------------------
class TestParamImportances:
    @staticmethod
    def _driven(n: int = 12) -> TunerResults:
        """score == lr exactly; depth is an uncorrelated knob."""
        rng = np.random.default_rng(0)
        trials = []
        for i in range(n):
            lr = float(rng.uniform(0.05, 0.95))
            depth = int(rng.integers(1, 4))
            trials.append(TrialResult(i + 1, {"lr": lr, "depth": depth}, lr, 0.0))
        return TunerResults(trials, "r2")

    def test_driver_ranks_first_with_half_the_mass(self) -> None:
        importance = param_importances(self._driven())
        assert importance.ranking()[0][0] == "lr"
        assert importance.importances[importance.names.index("lr")] > 0.5
        assert sum(importance.importances) == pytest.approx(1.0)
        assert importance.n_trials == 12
        assert importance.method == "mdi"
        assert importance.objective == "r2"

    def test_determinism(self) -> None:
        first = param_importances(self._driven())
        second = param_importances(self._driven())
        assert first.importances == second.importances
        assert first.names == second.names

    def test_two_completed_trials_required(self) -> None:
        trials = [TrialResult(1, {"lr": 0.1}, 0.1, 0.0)]
        with pytest.raises(ModelError, match="at least 2 completed trials, got 1"):
            param_importances(TunerResults(trials, "r2"))
        # a pruned trial does not count as completed
        trials = [
            TrialResult(1, {"lr": 0.1}, 0.1, 0.0),
            TrialResult(2, {"lr": 0.2}, float("nan"), float("nan"), state="pruned"),
        ]
        with pytest.raises(ModelError, match="at least 2 completed trials, got 1"):
            param_importances(TunerResults(trials, "r2"))

    def test_constant_param_reported_at_zero(self) -> None:
        trials = [
            TrialResult(1, {"lr": 0.1, "solver": "adam"}, 0.1, 0.0),
            TrialResult(2, {"lr": 0.4, "solver": "adam"}, 0.4, 0.0),
            TrialResult(3, {"lr": 0.7, "solver": "adam"}, 0.7, 0.0),
            TrialResult(4, {"lr": 0.9, "solver": "adam"}, 0.9, 0.0),
        ]
        importance = param_importances(TunerResults(trials, "r2"))
        assert importance.names == ("lr", "solver")
        assert importance.importances[importance.names.index("solver")] == 0.0
        assert importance.importances[importance.names.index("lr")] > 0.5

    def test_one_hot_choice_aggregation(self) -> None:
        # a Choice param drives the score through its one-hot columns
        rng = np.random.default_rng(1)
        base = {"adam": 0.9, "sgd": 0.5, "lbfgs": 0.1}
        trials = []
        for i in range(12):
            solver = ["adam", "sgd", "lbfgs"][i % 3]
            score = base[solver] + 0.01 * float(rng.uniform())
            noise = float(rng.uniform())
            trials.append(
                TrialResult(i + 1, {"solver": solver, "noise": noise}, score, 0.0)
            )
        importance = param_importances(TunerResults(trials, "r2"))
        assert importance.ranking()[0][0] == "solver"
        assert importance.importances[importance.names.index("solver")] > 0.5
        assert sum(importance.importances) == pytest.approx(1.0)

    def test_record_surfaces(self) -> None:
        importance = param_importances(self._driven())
        frame = importance.as_frame()
        assert list(frame.columns) == ["parameter", "importance"]
        assert len(frame) == 2
        assert importance.ranking(top_n=1) == importance.ranking()[:1]
        assert str(importance).startswith(
            "hyperparameter importances (mdi, objective=r2, n=12)"
        )


class TestOptunaStudyPlots:
    """The seven optuna.visualization figures over the tuner's records (100x-160).

    The plotting layer is pure matplotlib and duck-typed: every figure
    accepts the TunerResults record or a bare list of TrialResults, and
    the guards are the typed DataValidationError naming the exact
    misuse (no trials, no completed trials, no fold_scores, ...).
    """

    def _results(self) -> TunerResults:
        """A crafted 6-trial study: 4 complete, 1 pruned, 1 fail."""
        trials = [
            TrialResult(1, {"units": 8, "alpha": 0.10}, 0.50, 0.02,
                        fold_scores=(0.4, 0.5, 0.6), duration=1.5),
            TrialResult(2, {"units": 16, "alpha": 0.05}, 0.80, 0.02,
                        fold_scores=(0.7, 0.8, 0.9), duration=2.0),
            TrialResult(3, {"units": 32, "alpha": 0.20}, 0.65, 0.02,
                        fold_scores=(0.6, 0.6, 0.75), duration=1.8),
            TrialResult(4, {"units": 64, "alpha": 0.05}, 0.72, 0.02,
                        fold_scores=(0.7, 0.7, 0.76), duration=1.6),
            TrialResult(5, {"units": 128, "alpha": 0.30}, float("nan"),
                        float("nan"), state="pruned",
                        fold_scores=(0.1, 0.2), duration=0.5),
            TrialResult(6, {"units": 4, "alpha": 0.40}, float("nan"),
                        float("nan"), error="ValueError: bad", duration=0.2),
        ]
        return TunerResults(trials, "r2")

    def test_optimization_history(self) -> None:
        from pharmaml.viz import plot_optimization_history

        figure = plot_optimization_history(self._results())
        ax = figure.axes[0]
        assert ax.get_xlabel() == "trial"
        assert ax.get_ylabel() == "r2"
        # the completed dots + the best-so-far step line + the pruned
        # open markers at their last intermediate (0.2).
        assert len(ax.collections) >= 2
        assert len(ax.lines) >= 1
        pruned = [collection for collection in ax.collections
                  if collection.get_offsets().shape[0] == 1]
        assert any(
            pytest.approx(float(collection.get_offsets()[0][1]), abs=1e-9) == 0.2
            for collection in pruned
        )
        plt_close(figure)

    def test_optimization_history_guards(self) -> None:
        from pharmaml.viz import plot_optimization_history

        with pytest.raises(DataValidationError, match="no trials"):
            plot_optimization_history([])
        with pytest.raises(DataValidationError, match="TunerResults record or a list"):
            plot_optimization_history({"not": "a study"})
        all_pruned = [
            TrialResult(1, {"units": 8}, float("nan"), float("nan"),
                         state="pruned", fold_scores=(0.1,))
        ]
        with pytest.raises(DataValidationError, match="at least one completed"):
            plot_optimization_history(all_pruned)

    def test_slice(self) -> None:
        from pharmaml.viz import plot_slice

        figure = plot_slice(self._results())
        labels = [ax.get_xlabel() for ax in figure.axes if ax.get_visible()]
        assert sorted(labels) == ["alpha", "units"]
        single = plot_slice(self._results(), params=["units"])
        assert [ax.get_xlabel() for ax in single.axes] == ["units"]
        plt_close(figure)
        plt_close(single)

    def test_slice_categorical_param(self) -> None:
        from pharmaml.viz import plot_slice

        trials = [
            TrialResult(1, {"solver": "adam"}, 0.9, 0.0),
            TrialResult(2, {"solver": "sgd"}, 0.5, 0.0),
        ]
        figure = plot_slice(trials)
        ax = figure.axes[0]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["adam", "sgd"]
        plt_close(figure)

    def test_slice_guards(self) -> None:
        from pharmaml.viz import plot_slice

        with pytest.raises(DataValidationError, match="no trials"):
            plot_slice([])
        all_pruned = [
            TrialResult(1, {"units": 8}, float("nan"), float("nan"), state="pruned")
        ]
        with pytest.raises(DataValidationError, match="at least one completed trial"):
            plot_slice(all_pruned)
        with pytest.raises(DataValidationError, match="no completed trial declares"):
            plot_slice(self._results(), params=["not_a_param"])

    def test_parallel_coordinate(self) -> None:
        from pharmaml.viz import plot_parallel_coordinate

        figure = plot_parallel_coordinate(self._results())
        ax = figure.axes[0]
        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert tick_labels == ["units", "alpha", "r2"]
        # four completed trials -> four polylines, coloured by score.
        assert len(ax.lines) == 4
        plt_close(figure)

    def test_parallel_coordinate_guards(self) -> None:
        from pharmaml.viz import plot_parallel_coordinate

        with pytest.raises(DataValidationError, match="at least two completed"):
            plot_parallel_coordinate(
                [TrialResult(1, {"units": 8}, 0.5, 0.0)]
            )
        with pytest.raises(DataValidationError, match="no completed trial declares"):
            plot_parallel_coordinate(self._results(), params=["nope"])

    def test_param_importances_plot(self) -> None:
        from pharmaml.viz import plot_param_importances

        record = param_importances(self._results())
        figure = plot_param_importances(record)
        ax = figure.axes[0]
        ticks = [tick.get_text() for tick in ax.get_yticklabels()]
        assert set(ticks) == {"units", "alpha"}
        assert ax.get_xlabel() == "importance (mdi)"
        # the DataFrame twin carries the same contract
        frame_form = plot_param_importances(record.as_frame())
        assert frame_form.axes[0].get_xlabel() == "importance (importance)"
        plt_close(figure)
        plt_close(frame_form)

    def test_param_importances_plot_guard(self) -> None:
        from pharmaml.viz import plot_param_importances

        with pytest.raises(DataValidationError, match="HyperparameterImportance record"):
            plot_param_importances({"not": "importances"})

    def test_edf(self) -> None:
        from pharmaml.viz import plot_edf

        figure = plot_edf(self._results())
        ax = figure.axes[0]
        assert ax.get_xlabel() == "r2"
        assert ax.get_ylabel() == "fraction of trials ≤ value"
        # the step curve reaches 1.0 at the last completed score.
        line = ax.lines[0]
        assert float(line.get_ydata()[-1]) == pytest.approx(1.0)
        plt_close(figure)

    def test_edf_guard(self) -> None:
        from pharmaml.viz import plot_edf

        with pytest.raises(DataValidationError, match="no trials"):
            plot_edf([])
        all_pruned = [
            TrialResult(1, {"units": 8}, float("nan"), float("nan"), state="pruned")
        ]
        with pytest.raises(DataValidationError, match="at least one completed"):
            plot_edf(all_pruned)

    def test_timeline(self) -> None:
        from pharmaml.viz import plot_timeline

        figure = plot_timeline(self._results())
        ax = figure.axes[0]
        assert ax.get_xlabel() == "cumulative seconds"
        y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert set(y_labels) == {f"trial {i}" for i in range(1, 7)}
        # three states present -> three legend patches.
        legend_labels = {text.get_text() for text in ax.get_legend().get_texts()}
        assert legend_labels == {"complete", "pruned", "fail"}
        plt_close(figure)

    def test_timeline_guards(self) -> None:
        from pharmaml.viz import plot_timeline

        with pytest.raises(DataValidationError, match="recorded duration"):
            plot_timeline(
                [TrialResult(1, {"units": 8}, 0.5, 0.0)]
            )

    def test_intermediate_values(self) -> None:
        from pharmaml.viz import plot_intermediate_values

        figure = plot_intermediate_values(self._results())
        ax = figure.axes[0]
        assert ax.get_xlabel() == "fold (running mean)"
        assert ax.get_ylabel() == "r2"
        legend_labels = {text.get_text() for text in ax.get_legend().get_texts()}
        assert "trial 5 (pruned)" in legend_labels
        # the pruned trial's curve stops after its 2 recorded folds.
        pruned_line = next(
            line for line in ax.lines
            if line.get_label() == "trial 5 (pruned)"
        )
        assert len(pruned_line.get_xdata()) == 2
        plt_close(figure)

    def test_intermediate_values_guard(self) -> None:
        from pharmaml.viz import plot_intermediate_values

        with pytest.raises(DataValidationError, match="fold_scores"):
            plot_intermediate_values(
                [TrialResult(1, {"units": 8}, 0.5, 0.0)]
            )

    def test_the_viz_exports(self) -> None:
        # all seven names ride the viz package's public surface.
        from pharmaml import viz

        for name in (
            "plot_optimization_history",
            "plot_slice",
            "plot_parallel_coordinate",
            "plot_param_importances",
            "plot_edf",
            "plot_timeline",
            "plot_intermediate_values",
        ):
            assert name in viz.__all__
            assert hasattr(viz, name)

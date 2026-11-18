"""Tests for ``pharmaml.ml.hpo_presets`` -- the Part-3 §18.2/§18.3 layer.

Pins the §18.3 tables for every hand-authored family, the deterministic
preset narrowing (conservative ⊂ balanced ⊂ exploratory), the dataset and
resource adaptation (each leaving its note), the compatibility validation
(solver-penalty matrix, kernel- and task-conditioned parameters), the JSON
round-trip, the budget estimates, and the bridge that compiles a
recommendation into a tuner-searchable hypermodel (with a real TPE search
running end to end over an advised space).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.hpo_presets import (
    PRESET_STYLES,
    ParamAdvice,
    advice_families,
    estimate_evaluations,
    hypermodel_from_advice,
    preset_styles,
    recommend_search,
    recommendation_from_json,
    recommendation_to_json,
    validate_space,
    validate_values,
)
from pharmaml.ml.tuner import HyperParameters, RandomSearch, TPESearch

# -- the §18.3 tables, family by family ----------------------------------- #


@pytest.mark.parametrize(
    ("model", "task", "expected_essential"),
    [
        ("logistic_regression", "classification", {"C", "penalty", "solver"}),
        ("random_forest", "classification", {"n_estimators", "max_depth"}),
        ("random_forest", "regression", {"n_estimators", "max_depth"}),
        ("svm", "classification", {"C", "kernel"}),
        ("svm", "regression", {"C", "kernel"}),
        ("xgboost", "regression", {"n_estimators", "max_depth", "learning_rate"}),
        ("lightgbm", "classification", {"n_estimators", "num_leaves", "learning_rate"}),
        ("catboost", "regression", {"iterations", "depth", "learning_rate"}),
        ("knn", "regression", {"n_neighbors"}),
        ("ann", "regression", {"n_layers", "units"}),
        ("dnn", "classification", {"n_layers", "units", "learning_rate"}),
        ("hsvr", "regression", {"n_clusters", "C", "gamma"}),
    ],
)
def test_every_family_has_its_essential_parameters(
    model: str, task: str, expected_essential: set[str]
) -> None:
    rec = recommend_search(model, task, preset="exploratory")
    essential = {p.name for p in rec.params if p.priority == "essential"}
    assert essential == expected_essential


def test_every_table_parameter_carries_a_purpose() -> None:
    """§18.2: 'Explain the purpose of important parameters'."""
    for model in advice_families():
        for task in ("regression", "classification"):
            try:
                rec = recommend_search(model, task, preset="exploratory")
            except ModelError:
                continue  # hsvr: regression-only
            for p in rec.params:
                assert p.purpose, f"{model}/{task}:{p.name} has no purpose"
                assert p.priority in {"essential", "key", "extended", "fixed"}


def test_logistic_table_matches_the_directive_ranges() -> None:
    rec = recommend_search("logistic_regression", "classification", preset="exploratory")
    by = {p.name: p for p in rec.params}
    assert by["C"].log and by["C"].low == pytest.approx(1e-4) and by["C"].high == pytest.approx(1e2)
    assert set(by["penalty"].choices) == {"l2", "l1"}
    # compatibility order: saga (all penalties) first, so preset trims
    # never admit an invalid pair; lbfgs (l2-only) rides last
    assert by["solver"].choices == ("saga", "liblinear", "lbfgs")
    assert by["class_weight"].choices == (None, "balanced")
    assert by["max_iter"].low == 500 and by["max_iter"].high == 3000


def test_svr_table_is_task_aware() -> None:
    """epsilon is SVR-only; class_weight is SVC-only; degree rides SVR."""
    svr = {p.name for p in recommend_search("svm", "regression", preset="exploratory").params}
    svc = {p.name for p in recommend_search("svm", "classification", preset="exploratory").params}
    assert {"epsilon", "degree"} <= svr
    assert "epsilon" not in svc
    assert "class_weight" in svc and "class_weight" not in svr
    gamma = next(
        p
        for p in recommend_search("svm", "regression", preset="exploratory").params
        if p.name == "gamma"
    )
    assert "scale" in gamma.choices and 0.001 in gamma.choices


def test_dnn_table_covers_the_directive_dl_rows() -> None:
    by = {
        p.name: p
        for p in recommend_search("dnn", "classification", preset="exploratory").params
    }
    assert by["learning_rate"].log
    assert by["learning_rate"].low == pytest.approx(1e-5)
    assert by["learning_rate"].high == pytest.approx(1e-2)
    assert by["batch_size"].low == 16 and by["batch_size"].high == 128
    assert by["epochs"].low == 50 and by["epochs"].high == 300
    assert by["patience"].low == 5 and by["patience"].high == 30
    assert by["dropout"].high == pytest.approx(0.5)
    assert by["weight_decay"].log
    assert {str(c) for c in by["activation"].choices} == {"relu", "gelu", "tanh"}
    # the honest-scope row: the optimizer is fixed, documented not searched
    fixed = by["optimizer"]
    assert not fixed.searchable and fixed.value == "adamw"


def test_registry_families_get_grid_derived_advice() -> None:
    """§18.3 'Additional Models': fallback from the registry's own grid."""
    rec = recommend_search("ridge", "regression")
    assert rec.params and all(p.kind == "choice" for p in rec.params)
    assert "alpha" in {p.name for p in rec.params}
    rec = recommend_search("decision_tree", "regression")
    assert "max_depth" in {p.name for p in rec.params}


def test_aliases_and_refusals() -> None:
    assert recommend_search("rf", "regression").model == "random_forest"
    assert recommend_search("deep", "regression").model == "dnn"
    with pytest.raises(ModelError, match="task must be"):
        recommend_search("knn", "banana")
    with pytest.raises(ModelError, match="not.*classification"):
        recommend_search("hsvr", "classification")
    with pytest.raises(ModelError, match="Unknown model"):
        recommend_search("banana", "regression")
    with pytest.raises(ModelError, match="no param_grid"):
        recommend_search("linear_regression", "regression")
    with pytest.raises(ModelError, match="preset"):
        recommend_search("knn", "regression", preset="turbo")
    with pytest.raises(ModelError, match="resources"):
        recommend_search("knn", "regression", resources="infinite")


# -- the presets (§18.2) --------------------------------------------------- #


def test_preset_styles_exported() -> None:
    assert preset_styles() == ("conservative", "balanced", "exploratory")
    assert PRESET_STYLES == preset_styles()


def test_preset_parameter_counts_nest() -> None:
    """conservative searches fewer parameters than exploratory."""
    for model in ("random_forest", "xgboost", "catboost", "dnn"):
        counts = [
            len(recommend_search(model, "regression", preset=style).searchable)
            for style in PRESET_STYLES
        ]
        assert counts[0] < counts[2], (model, counts)
        assert counts[1] <= counts[2], (model, counts)


def test_conservative_narrows_ranges_deterministically() -> None:
    rec = recommend_search("random_forest", "regression", preset="conservative")
    n_estimators = next(p for p in rec.params if p.name == "n_estimators")
    # [100, 800] step 100 -> middle half snapped: [300, 600]
    assert (n_estimators.low, n_estimators.high) == (300.0, 600.0)
    full = next(
        p
        for p in recommend_search("random_forest", "regression", preset="exploratory").params
        if p.name == "n_estimators"
    )
    assert (full.low, full.high) == (100.0, 800.0)
    # log ranges narrow by the middle half in the log domain
    lr = next(
        p
        for p in recommend_search("xgboost", "regression", preset="conservative").params
        if p.name == "learning_rate"
    )
    assert lr.low == pytest.approx(0.01 * (30.0) ** 0.25)
    assert lr.high == pytest.approx(0.01 * (30.0) ** 0.75)


def test_preset_choice_lists_trim_in_order() -> None:
    """The recommended-first ordering survives the trim."""
    cons = recommend_search("random_forest", "regression", preset="conservative")
    # key parameters join from balanced up, not conservative
    assert "max_features" not in {p.name for p in cons.params}
    rec = recommend_search("random_forest", "regression", preset="balanced")
    mf = next(p for p in rec.params if p.name == "max_features")
    assert mf.choices == ("sqrt", "log2", 0.3, 0.5)
    full = next(
        p
        for p in recommend_search("random_forest", "regression", preset="exploratory").params
        if p.name == "max_features"
    )
    assert len(full.choices) == 6


def test_fixed_parameters_ride_every_preset() -> None:
    for style in PRESET_STYLES:
        rec = recommend_search("dnn", "regression", preset=style)
        fixed = [p for p in rec.params if not p.searchable]
        assert [p.name for p in fixed] == ["optimizer"]


# -- the adaptation (§18.2) ------------------------------------------------ #


def test_small_dataset_caps_budgets_and_notes_it() -> None:
    rec = recommend_search(
        "random_forest", "regression", preset="exploratory", n_samples=180
    )
    n_estimators = next(p for p in rec.params if p.name == "n_estimators")
    assert n_estimators.high == 300.0
    assert any("small dataset" in n and "300" in n for n in rec.notes)


def test_small_dataset_caps_knn_neighbors() -> None:
    rec = recommend_search("knn", "regression", n_samples=40)
    k = next(p for p in rec.params if p.name == "n_neighbors")
    assert k.high == 20.0  # n/2, never below the narrowed low
    rec_big = recommend_search("knn", "regression", n_samples=5000)
    k_big = next(p for p in rec_big.params if p.name == "n_neighbors")
    assert k_big.high == 50.0


def test_large_dataset_prefers_model_based_search() -> None:
    rec = recommend_search(
        "random_forest", "regression", preset="exploratory", n_samples=50_000
    )
    assert any("RandomizedSearch/TPESearch" in n for n in rec.notes)


def test_low_resources_cap_budgets_and_batches() -> None:
    rec = recommend_search("dnn", "classification", preset="exploratory", resources="low")
    epochs = next(p for p in rec.params if p.name == "epochs")
    assert epochs.high == 100.0
    batch = next(p for p in rec.params if p.name == "batch_size")
    assert batch.low == 32  # 16 dropped
    assert any("low resources" in n for n in rec.notes)
    rf = recommend_search("xgboost", "regression", resources="low")
    rounds = next(p for p in rf.params if p.name == "n_estimators")
    assert rounds.high == 300.0


def test_cap_below_declared_low_leaves_parameter_alone() -> None:
    """max_iter's floor is 500 -- a 300 cap must not bend the range."""
    rec = recommend_search(
        "logistic_regression", "classification", preset="exploratory", n_samples=100
    )
    max_iter = next(p for p in rec.params if p.name == "max_iter")
    assert (max_iter.low, max_iter.high) == (500.0, 3000.0)


def test_high_dimensionality_note_is_family_aware() -> None:
    rf = recommend_search("random_forest", "regression", n_features=200)
    assert any("max_features" in n for n in rf.notes)
    knn = recommend_search("knn", "regression", n_features=200)
    assert any("PCA" in n for n in knn.notes)
    lr = recommend_search("logistic_regression", "classification", n_features=5)
    assert not lr.notes


def test_class_imbalance_keeps_balanced_searchable() -> None:
    rec = recommend_search(
        "random_forest",
        "classification",
        preset="conservative",
        class_counts=[500, 25],
    )
    cw = next(p for p in rec.params if p.name == "class_weight")
    assert "balanced" in cw.choices  # re-added even after the preset trim
    assert any("imbalance 20.0:1" in n for n in rec.notes)


def test_imbalance_notes_the_honest_alternatives() -> None:
    xgb = recommend_search("xgboost", "classification", class_counts=[400, 25])
    assert any("scale_pos_weight" in n for n in xgb.notes)
    cat = recommend_search("catboost", "classification", class_counts=[400, 25])
    assert any("auto_class_weights" in n for n in cat.notes)
    knn = recommend_search("knn", "classification", class_counts=[400, 25])
    assert any("no class_weight parameter" in n for n in knn.notes)


def test_balanced_imbalance_ratio_is_not_flagged() -> None:
    rec = recommend_search(
        "random_forest", "classification", class_counts=[100, 40]
    )
    assert not any("imbalance" in n for n in rec.notes)


def test_summary_renders_the_budget_review() -> None:
    rec = recommend_search("knn", "regression", preset="conservative", n_samples=40)
    text = rec.summary()
    assert "knn (regression) -- conservative preset" in text
    # full range [3, 50] capped at n/2 = 20, then narrowed to its middle half
    assert "n_neighbors (essential): [8, 15]" in text
    assert "note: small dataset" in text


# -- the validation (§18.2) ------------------------------------------------ #


def test_validate_space_default_tables_are_clean() -> None:
    for model in advice_families():
        for task in ("regression", "classification"):
            try:
                issues = validate_space(model, task)
            except ModelError:
                continue
            errors = [i for i in issues if i.is_error]
            assert not errors, (model, task, errors)


def test_validate_space_flags_the_solver_penalty_trap() -> None:
    issues = validate_space("logistic_regression", "classification", preset="exploratory")
    warnings = [i for i in issues if i.parameter == "penalty"]
    assert warnings and "lbfgs" in warnings[0].message
    # the balanced preset trims penalty to l2-only: no trap left
    balanced = validate_space("logistic_regression", "classification", preset="balanced")
    assert not any(i.parameter == "penalty" for i in balanced)


def test_validate_space_rejects_unknown_and_broken_parameters() -> None:
    bogus = ParamAdvice(
        name="n_estimators", kind="int", priority="essential", purpose="", low=10.0, high=1.0
    )
    issues = validate_space("knn", "regression", [bogus])
    assert any(i.is_error and "not a supported parameter" in i.message for i in issues)
    bad_range = ParamAdvice(
        name="n_neighbors", kind="int", priority="essential", purpose="", low=50.0, high=3.0
    )
    issues = validate_space("knn", "regression", [bad_range])
    assert any(i.is_error and "invalid range" in i.message for i in issues)
    bad_log = ParamAdvice(
        name="C", kind="float", priority="essential", purpose="", low=-1.0, high=1.0, log=True
    )
    issues = validate_space("svm", "regression", [bad_log])
    assert any("positive bounds" in i.message for i in issues)
    with pytest.raises(ModelError, match="not both"):
        validate_space("knn", "regression", [], preset="balanced")


def test_validate_space_kernel_conditioning() -> None:
    linear_only = [
        ParamAdvice(
            name="kernel", kind="choice", priority="essential", purpose="", choices=("linear",)
        ),
        ParamAdvice(
            name="gamma", kind="choice", priority="key", purpose="", choices=("scale", "auto")
        ),
        ParamAdvice(name="degree", kind="int", priority="extended", purpose="", low=2.0, high=5.0),
    ]
    issues = validate_space("svm", "regression", linear_only)
    assert any(i.parameter == "degree" and "poly" in i.message for i in issues)
    assert any(i.parameter == "gamma" and "linear kernel" in i.message for i in issues)


def test_validate_space_task_only_parameters() -> None:
    epsilon_in_cls = ParamAdvice(
        name="epsilon", kind="float", priority="key", purpose="", low=0.01, high=0.5
    )
    issues = validate_space("svm", "classification", [epsilon_in_cls])
    assert any(i.is_error and "SVR-only" in i.message for i in issues)
    cw_in_reg = ParamAdvice(
        name="class_weight", kind="choice", priority="key", purpose="", choices=(None, "balanced")
    )
    issues = validate_space("random_forest", "regression", [cw_in_reg])
    assert any(i.is_error and "classification-only" in i.message for i in issues)


def test_validate_space_catboost_loss_matches_task() -> None:
    wrong = ParamAdvice(
        name="loss_function",
        kind="choice",
        priority="key",
        purpose="",
        choices=("Logloss", "RMSE"),
    )
    for task in ("classification", "regression"):
        issues = validate_space("catboost", task, [wrong])
        assert any(i.is_error and "loss_function" in i.parameter for i in issues), task


def test_validate_values_solver_penalty_matrix() -> None:
    assert not validate_values(
        "logistic_regression", "classification",
        {"C": 1.0, "penalty": "l1", "solver": "saga"},
    )
    bad = validate_values(
        "logistic_regression", "classification",
        {"C": 1.0, "penalty": "l1", "solver": "lbfgs"},
    )
    assert bad and bad[0].is_error and "lbfgs" in bad[0].message
    assert not validate_values(
        "logistic_regression", "classification",
        {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
    )


def test_validate_values_unknown_and_conditioned() -> None:
    issues = validate_values("knn", "regression", {"n_neighbors": 5, "turbo": 1})
    assert any(i.is_error and "turbo" in i.parameter for i in issues)
    noop = validate_values(
        "svm", "regression", {"C": 1.0, "kernel": "rbf", "degree": 3}
    )
    assert any(i.parameter == "degree" and "no effect" in i.message for i in noop)
    ignored = validate_values(
        "svm", "regression", {"C": 1.0, "kernel": "linear", "gamma": "scale"}
    )
    assert any(i.parameter == "gamma" and "ignored" in i.message for i in ignored)


# -- the budget estimate (§18.5) ------------------------------------------- #


def test_grid_estimate_multiplies_point_counts() -> None:
    rec = recommend_search("random_forest", "regression", preset="conservative")
    est = estimate_evaluations(rec, strategy="grid")
    # n_estimators [300, 600] step 100 -> 4 points; max_depth 2 choices
    assert est.count == 8 and est.strategy == "grid" and not est.assumptions
    expl = estimate_evaluations(
        recommend_search("random_forest", "regression", preset="exploratory"),
        strategy="grid",
    )
    assert expl.count > est.count  # full budget dominance


def test_continuous_floats_carry_the_assumption() -> None:
    rec = recommend_search("svm", "regression", preset="conservative")
    est = estimate_evaluations(rec, strategy="grid")
    assert est.assumptions and "20-bin" in est.assumptions[0]


def test_model_based_estimates_are_the_trial_counts() -> None:
    rec = recommend_search("knn", "regression")
    assert estimate_evaluations(rec, strategy="tpe", n_trials=40).count == 40
    assert estimate_evaluations(rec, strategy="random", n_trials=7).count == 7
    with pytest.raises(ModelError, match="strategy"):
        estimate_evaluations(rec, strategy="hyperband")


# -- the JSON interchange (§18.5) ----------------------------------------- #


def test_json_round_trip_is_lossless() -> None:
    rec = recommend_search(
        "dnn",
        "classification",
        preset="balanced",
        n_samples=180,
        n_features=120,
        class_counts=[150, 30],
        resources="low",
    )
    loaded = recommendation_from_json(recommendation_to_json(rec))
    assert loaded == rec


def test_json_round_trip_preserves_none_choices_and_fixed_rows() -> None:
    rec = recommend_search("random_forest", "classification", preset="exploratory")
    loaded = recommendation_from_json(recommendation_to_json(rec))
    cw = next(p for p in loaded.params if p.name == "class_weight")
    assert cw.choices == (None, "balanced")
    dnn = recommendation_from_json(
        recommendation_to_json(recommend_search("dnn", "regression"))
    )
    fixed = next(p for p in dnn.params if not p.searchable)
    assert fixed.value == "adamw"


@pytest.mark.parametrize(
    "mangle",
    [
        "not json",
        "[]",
        (
            '{"schema_version": 99, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": []}'
        ),
        (
            '{"schema_version": 1, "task": "regression", "preset": "balanced",'
            ' "params": []}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "banana", "params": []}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": []}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": [{"name": "k", "kind": "turbo"}]}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": [{"name": "k", "kind": "int",'
            ' "priority": "meh", "low": 1, "high": 2}]}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": [{"name": "k", "kind": "int",'
            ' "low": 5, "high": 1}]}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": [{"name": "k", "kind": "float",'
            ' "low": -1, "high": 2, "log": true}]}'
        ),
        (
            '{"schema_version": 1, "model": "knn", "task": "regression",'
            ' "preset": "balanced", "params": [{"name": "k", "kind": "choice",'
            ' "choices": []}]}'
        ),
    ],
)
def test_json_schema_violations_are_typed(mangle: str) -> None:
    with pytest.raises(DataValidationError):
        recommendation_from_json(mangle)


def test_json_export_is_stable_and_inspectable() -> None:
    text = recommendation_to_json(recommend_search("knn", "regression"))
    payload = json.loads(text)
    assert payload["schema_version"] == 1
    assert payload["model"] == "knn"
    assert payload["preset"] == "balanced"
    assert all("purpose" in row for row in payload["params"])


# -- the bridge (recommendation -> searchable hypermodel) ------------------ #


def test_bridge_collects_the_advised_space() -> None:
    builder = hypermodel_from_advice(
        "random_forest", "classification", preset="conservative"
    )
    assert builder.__name__ == "build_random_forest_classifier_advised"
    hp = HyperParameters()
    estimator = builder(hp)
    assert sorted(spec.name for spec in hp.space) == ["max_depth", "n_estimators"]
    assert estimator.n_estimators == 300  # collection mode: the low bound
    assert estimator.max_depth is None    # the first choice


def test_bridge_builds_every_family() -> None:
    for model, task in [
        ("logistic_regression", "classification"),
        ("random_forest", "regression"),
        ("svm", "regression"),
        ("svm", "classification"),
        ("knn", "classification"),
        ("ann", "regression"),
        ("dnn", "regression"),
        ("hsvr", "regression"),
        ("ridge", "regression"),
    ]:
        builder = hypermodel_from_advice(model, task, preset="balanced")
        hp = HyperParameters()
        estimator = builder(hp)
        assert estimator is not None, (model, task)
        assert hp.space, (model, task)


def test_bridge_fails_incompatible_trials_loudly() -> None:
    """The per-trial guard: l1 + lbfgs never silently trains."""
    builder = hypermodel_from_advice(
        "logistic_regression", "classification", preset="exploratory"
    )
    hp = HyperParameters(values={"penalty": "l1", "solver": "lbfgs"})
    with pytest.raises(ModelError, match="does not support penalty"):
        builder(hp)


def test_bridge_refuses_spaces_with_validation_errors() -> None:
    bad = ParamAdvice(
        name="class_weight", kind="choice", priority="key", purpose="", choices=(None,)
    )
    issues = validate_space("random_forest", "regression", [bad])
    assert any(i.is_error for i in issues)
    # hypermodel_from_advice only sees its own generated spaces, which
    # carry no errors -- the guard is exercised through validate_space:
    builder = hypermodel_from_advice("random_forest", "regression")
    hp = HyperParameters()
    builder(hp)  # the generated space is clean by construction


def test_advised_random_search_runs_end_to_end() -> None:
    """A real search over an advised space (the whole point)."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    builder = hypermodel_from_advice(
        "random_forest", "classification", preset="conservative", n_samples=60
    )
    search = RandomSearch(
        builder,
        objective="accuracy",
        max_trials=4,
        cv=3,
        task="classification",
        seed=11,
    )
    results = search.search(X, y)
    assert len(results.trials) == 4
    assert results.best_score is not None
    # every drawn combination stayed inside the advised space
    for trial in results.trials:
        assert set(trial.hyperparameters) <= {"n_estimators", "max_depth"}
        assert 100 <= trial.hyperparameters["n_estimators"] <= 300


def test_advised_tpe_search_completes_cleanly() -> None:
    """TPE over the trap-free conservative advised space, all green."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] * 0.8 + X[:, 2] * 0.4 + rng.normal(scale=0.1, size=80) > 0).astype(int)
    builder = hypermodel_from_advice(
        "logistic_regression", "classification", preset="conservative"
    )
    tpe = TPESearch(
        builder, objective="accuracy", max_trials=10, cv=3, task="classification", seed=5
    )
    results = tpe.search(X, y)
    assert len(results.trials) == 10
    assert all(not t.failed for t in results.trials), [
        t.error for t in results.trials if t.failed
    ]
    assert results.best_score is not None
    for trial in results.trials:
        assert set(trial.hyperparameters) <= {"C", "penalty", "solver"}


def test_advised_tpe_search_end_to_end_balanced() -> None:
    """The balanced space (admitting the l1+lbfgs trap) still yields a
    best trial -- incompatible draws fail loudly, the search moves on."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] * 0.8 + X[:, 2] * 0.4 + rng.normal(scale=0.1, size=80) > 0).astype(int)
    builder = hypermodel_from_advice(
        "logistic_regression", "classification", preset="balanced"
    )
    tpe = TPESearch(
        builder, objective="accuracy", max_trials=8, cv=3, task="classification", seed=5
    )
    results = tpe.search(X, y)
    assert len(results.trials) == 8
    completed = [t for t in results.trials if not t.failed]
    assert completed and results.best_score is not None


def test_advice_survives_the_registry_surface() -> None:
    """The advised builder composes with the tuner's space collection."""
    from pharmaml.ml.tuner import _collect_space

    builder = hypermodel_from_advice("knn", "regression", preset="exploratory")
    hp = HyperParameters()
    builder(hp)
    specs = _collect_space(builder)
    assert {s.name for s in specs} == {"n_neighbors", "weights", "metric", "p"}

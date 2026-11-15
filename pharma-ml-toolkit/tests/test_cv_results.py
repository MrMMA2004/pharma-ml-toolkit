"""Tests for ``pharmaml.ml.cv_results`` -- the Part-3 §19 layer.

Pins the §19.3 splitter core (strategy selection, honest guards for
labels/groups/scaffolds, the limitations notes), the §19.2 display
contract (``mean ± std (k-fold CV)``), the §19.4/§19.5 honest metric
panels (binary-only drops, no-proba drops, the balanced-accuracy
fallback), the §19.6 deep-learning guarantee (a FRESH model object per
fold -- counted, not assumed), the §19.7 leakage-honesty of grouped and
scaffold folds (no group ever straddles train/validation), and the
§19.9 persistence round-trip (dataset fingerprint, fold records,
aggregates, execution metadata).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from pharmaml.exceptions import DataValidationError, JobCancelled, ModelError
from pharmaml.ml.cv_results import (
    CLASSIFICATION_CV_METRICS,
    METRIC_LABELS,
    REGRESSION_CV_METRICS,
    build_cv_splitter,
    compare_models_cv,
    cross_validate_model,
    cv_result_from_dict,
    cv_result_to_dict,
    dataset_fingerprint,
    default_folds,
    format_cv_score,
)
from pharmaml.ml.splits import scaffold_groups

# -- fixtures -------------------------------------------------------------- #


@pytest.fixture()
def toy_regression() -> tuple[pd.DataFrame, pd.Series]:
    """A learnable regression toy (y = 2x0 - x1 + 3)."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(48, 2)), columns=["x0", "x1"])
    y = 2.0 * X["x0"] - X["x1"] + 3.0
    return X, pd.Series(y.values)


@pytest.fixture()
def toy_classification() -> tuple[pd.DataFrame, pd.Series]:
    """A balanced binary classification toy (separable on x0)."""
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["x0", "x1", "x2"])
    y = pd.Series((X["x0"] > 0.2).astype(int).values)
    return X, y


RING_SMILES: list[str] = [
    "c1ccccc1O",  # benzene framework
    "c1ccncc1O",  # pyridine framework
    "c1ccc2ccccc2c1O",  # naphthalene framework
    "c1ccoc1C",  # furan framework
] * 12  # 48 rows, 4 distinct Bemis-Murcko scaffolds, 12 rows each


# -- §19.3: the splitter core --------------------------------------------- #


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("classification", StratifiedKFold),
        ("regression", KFold),
    ],
)
def test_strategy_defaults_by_task(task: str, expected: type) -> None:
    splitter, _notes = build_cv_splitter(task, n_folds=4, y=[0, 1] * 8)
    assert isinstance(splitter, expected)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("kfold", KFold),
        ("stratified", StratifiedKFold),
        ("group", GroupKFold),
        ("stratified_group", StratifiedGroupKFold),
        ("scaffold", GroupKFold),
    ],
)
def test_every_strategy_builds_its_splitter(strategy: str, expected: type) -> None:
    y = [0, 1] * 12
    splitter, notes = build_cv_splitter(
        "classification",
        strategy=strategy,
        n_folds=3,
        y=y,
        smiles=RING_SMILES[:len(y)],
        groups=["g0", "g1", "g2", "g3"] * 6,
    )
    assert isinstance(splitter, expected)
    assert notes, "every strategy must carry its limitations note (§19.3)"


def test_strategy_notes_document_their_limitations() -> None:
    _, notes = build_cv_splitter("regression", strategy="kfold", n_folds=3)
    assert any("straddle" in note for note in notes)
    _, notes = build_cv_splitter(
        "regression", strategy="scaffold", n_folds=3, smiles=RING_SMILES
    )
    assert any("Distinct scaffolds can still be similar" in note for note in notes)
    assert any("failed parses" in note for note in notes)


def test_scaffold_strategy_groups_by_bemis_murcko() -> None:
    # 48 rows / 4 distinct frameworks -> 4 foldable groups
    keys = scaffold_groups(RING_SMILES)
    assert len(set(keys)) == 4
    splitter, _ = build_cv_splitter(
        "regression", strategy="scaffold", n_folds=4, smiles=RING_SMILES
    )
    splits = list(splitter.split(np.zeros((48, 1)), groups=np.asarray(keys)))
    assert len(splits) == 4


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"task": "banana"}, "task must be"),
        ({"task": "regression", "strategy": "quantum"}, "unknown CV strategy"),
        ({"task": "regression", "n_folds": 1}, "n_folds must be >= 2"),
        ({"task": "regression", "n_folds": 5, "y": [1.0, 2.0]}, "exceeds the number of rows"),
        ({"task": "regression", "strategy": "stratified"}, "needs y"),
        ({"task": "regression", "strategy": "group"}, "needs groups"),
        ({"task": "regression", "strategy": "scaffold"}, "needs smiles"),
        (
            {"task": "regression", "strategy": "group", "groups": ["a"] * 8},
            "cannot fill 3 folds",
        ),
        (
            {
                "task": "regression",
                "strategy": "scaffold",
                "smiles": ["CCO"] * 8,  # one acyclic framework
            },
            "cannot fill 3 folds",
        ),
    ],
)
def test_splitter_guards_fail_loudly(kwargs: dict, fragment: str) -> None:
    kwargs = dict(kwargs)
    kwargs.setdefault("task", "regression")
    n_folds = kwargs.pop("n_folds", 3)
    with pytest.raises((DataValidationError, ModelError)) as excinfo:
        build_cv_splitter(n_folds=n_folds, **kwargs)
    assert fragment in str(excinfo.value)


def test_stratified_rare_class_guard() -> None:
    # rarest class has 2 members -- 3 stratified folds cannot form
    y = [0] * 10 + [1] * 2
    with pytest.raises(DataValidationError, match="rarest class"):
        build_cv_splitter("classification", strategy="stratified", n_folds=3, y=y)


# -- §19.2: the display contract ------------------------------------------- #


def test_format_cv_score_the_display_contract() -> None:
    assert format_cv_score(0.923, 0.018, 5) == "0.923 ± 0.018 (5-fold CV)"
    assert format_cv_score(-1.6613, 0.0, 4) == "-1.661 ± 0.000 (4-fold CV)"
    assert format_cv_score(float("nan"), 0.02, 5) == "nan ± 0.020 (5-fold CV)"
    assert format_cv_score(0.5, 0.25, 3, precision=4) == "0.5000 ± 0.2500 (3-fold CV)"


def test_default_folds_justify_themselves() -> None:
    folds, note = default_folds(400)
    assert folds == 5 and "default when feasible" in note
    folds, note = default_folds(30)
    assert folds == 3 and "limited data" in note
    folds, note = default_folds(400, dl=True)
    assert folds == 3 and "expensive" in note
    with pytest.raises(DataValidationError):
        default_folds(0)


# -- §19.2/§19.4/§19.5: cross_validate_model ------------------------------- #


def test_regression_result_records_everything_section_19_2_demands(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    seen: list[dict] = []
    result = cross_validate_model(
        X, y, "regression", "ridge", n_folds=4, seed=42, on_fold=seen.append
    )
    # strategy / folds / seed recorded verbatim
    assert (result.model, result.task, result.strategy, result.n_folds, result.seed) == (
        "ridge",
        "regression",
        "kfold",
        4,
        42,
    )
    # the per-fold heartbeat fired once per fold, in order
    assert [payload["fold"] for payload in seen] == [1, 2, 3, 4]
    assert all(payload["total"] == 4 for payload in seen)
    # individual fold scores + train-side mirrors + timings
    assert len(result.folds) == 4
    for record in result.folds:
        assert set(record.scores) == set(REGRESSION_CV_METRICS)
        assert set(record.train_scores) == set(REGRESSION_CV_METRICS)
        assert record.fit_seconds >= 0.0 and record.score_seconds >= 0.0
        assert record.n_train + record.n_val == len(y)
    # mean/std over the fold values (the §19.2 aggregate)
    fold_r2 = np.asarray([record.scores["r2"] for record in result.folds])
    assert result.metrics["r2"].mean == pytest.approx(float(fold_r2.mean()))
    assert result.metrics["r2"].std == pytest.approx(float(fold_r2.std()))
    assert result.metrics["r2"].folds == fold_r2.tolist()
    # the formatted summary
    assert result.summary() == format_cv_score(
        result.metrics["r2"].mean, result.metrics["r2"].std, 4
    )
    # execution metadata + fingerprint (§19.9)
    assert result.toolkit_version and result.created_at
    assert result.dataset_id == dataset_fingerprint(X, y)
    assert result.feature_columns == ["x0", "x1"]


def test_binary_classification_keeps_the_full_panel(
    toy_classification: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_classification
    result = cross_validate_model(X, y, "classification", "logistic_regression", n_folds=3)
    assert result.primary == "roc_auc_ovo"
    assert set(result.metrics) == set(CLASSIFICATION_CV_METRICS)
    assert result.summary().endswith("(3-fold CV)")


def test_multiclass_drops_binary_only_metrics_with_a_note() -> None:
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(90, 2)), columns=["a", "b"])
    y = pd.Series(rng.integers(0, 3, size=90))  # 3 classes
    result = cross_validate_model(X, y, "classification", "logistic_regression", n_folds=3)
    dropped = {"precision", "recall", "f1", "average_precision"}
    assert dropped.isdisjoint(result.metrics)
    assert "multiclass target" in " ".join(result.notes)
    assert {"f1_macro", "f1_weighted", "roc_auc_ovo", "neg_log_loss"}.issubset(
        result.metrics
    )


def test_no_proba_estimator_drops_proba_metrics_and_falls_back(
    toy_classification: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_classification
    result = cross_validate_model(X, y, "classification", "ridge_classifier", n_folds=3)
    proba = {"roc_auc_ovo", "average_precision", "neg_log_loss"}
    assert proba.isdisjoint(result.metrics)
    assert result.primary == "balanced_accuracy"  # the honest fallback
    assert any("predict_proba" in note for note in result.notes)


def test_metric_direction_labels_are_honest() -> None:
    for name in REGRESSION_CV_METRICS:
        label, lower_is_better = METRIC_LABELS[name]
        assert label and isinstance(lower_is_better, bool)
        if name.startswith("neg_"):
            # displayed magnitudes: error metrics read lower-is-better
            assert lower_is_better, name
    for name in CLASSIFICATION_CV_METRICS:
        assert name in METRIC_LABELS


def test_explicit_primary_must_be_in_the_panel(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    with pytest.raises(ModelError, match="not in the reportable panel"):
        cross_validate_model(X, y, "regression", "ridge", n_folds=3, primary="roc_auc_ovo")


def test_to_frame_is_the_fold_table(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    result = cross_validate_model(X, y, "regression", "ridge", n_folds=4)
    frame = result.to_frame()
    assert frame["fold"].tolist() == [1, 2, 3, 4]
    for column in ("fold_r2", "fold_rmse", "fold_mae", "fold_mse", "fold_median_ae"):
        assert column in frame.columns
    for column in ("train_r2", "train_rmse"):
        assert column in frame.columns
    assert frame["n_train"].sum() + frame["n_val"].sum() == 4 * len(y)


# -- §19.6: deep learning, fresh model per fold ---------------------------- #


def test_dnn_cross_validation_uses_a_fresh_model_per_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")  # the DNN estimator needs the optional deep extra
    from pharmaml.ml import neural

    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(36, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int).values)
    fitted: list[object] = []
    original = neural.DeepMLPClassifier.fit

    def counting_fit(self: object, *args: object, **kwargs: object) -> object:
        # Hold the instance itself, not id(self): a freed model's address
        # can be recycled for the NEXT fold's fresh model inside a long
        # pytest process, collapsing set(fitted_ids) and failing the
        # distinctness assert on a false positive (caught by the round-262
        # nine-chunk gate after the sandbox rollback re-laid the heap).
        fitted.append(self)
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(neural.DeepMLPClassifier, "fit", counting_fit)
    result = cross_validate_model(
        X, y, "classification", "dnn", n_folds=3, primary="accuracy"
    )
    # 3 folds -> 3 fit calls on 3 DISTINCT model objects (§19.6: never
    # reuse trained weights across folds; a fresh module carries a
    # fresh optimizer and scheduler by construction)
    assert len(fitted) == 3
    assert len({id(m) for m in fitted}) == 3
    assert result.summary().endswith("(3-fold CV)")
    assert result.metrics["accuracy"].mean > 0.5  # the toy is separable


def test_default_folds_guides_dl_budget() -> None:
    folds, note = default_folds(36, dl=True)
    assert folds == 3 and "expensive" in note


# -- §19.7: leakage honesty of grouped / scaffold folds ------------------- #


def test_group_folds_never_split_a_group() -> None:
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    y = pd.Series(rng.normal(size=60))
    groups = [f"g{i % 6}" for i in range(60)]  # 6 groups of 10 rows
    result = cross_validate_model(
        X, y, "regression", "ridge", strategy="group", n_folds=3, groups=groups
    )
    assert result.strategy == "group"
    # rebuild the same fold assignment and prove no group straddles
    keys = np.asarray(groups)
    splitter, _ = build_cv_splitter(
        "regression", strategy="group", n_folds=3, groups=groups
    )
    for train_idx, val_idx in splitter.split(X, y, groups=keys):
        assert set(keys[train_idx]).isdisjoint(set(keys[val_idx]))
    # fold sizes honoured in the records
    for record in result.folds:
        assert record.n_train + record.n_val == 60


def test_scaffold_folds_never_split_a_scaffold() -> None:
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(48, 2)), columns=["a", "b"])
    y = pd.Series(rng.normal(size=48))
    result = cross_validate_model(
        X, y, "regression", "ridge", strategy="scaffold", n_folds=4, smiles=RING_SMILES
    )
    assert result.strategy == "scaffold"
    keys = np.asarray(scaffold_groups(RING_SMILES))
    splitter, _ = build_cv_splitter(
        "regression", strategy="scaffold", n_folds=4, smiles=RING_SMILES
    )
    for train_idx, val_idx in splitter.split(X, y, groups=keys):
        assert set(keys[train_idx]).isdisjoint(set(keys[val_idx]))
    # the limitations note travelled with the result (§19.3)
    assert any("Distinct scaffolds can still be similar" in note for note in result.notes)


def test_pipelines_fit_preprocessing_per_fold(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The pipeline structure §19.7 demands: prep INSIDE, estimator out."""
    X, y = toy_regression
    result = cross_validate_model(X, y, "regression", "ridge", n_folds=3)
    assert result.feature_columns == ["x0", "x1"]  # the config snapshot
    # structural pin via the engine's own pipeline builder
    from pharmaml.ml.cv_results import _build_pipeline

    pipeline = _build_pipeline(X, "ridge", "regression")
    assert [name for name, _ in pipeline.steps] == ["prep", "est"]


# -- §19.8: the board ------------------------------------------------------ #


def test_compare_models_cv_board_and_heartbeat(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    beats: list[dict] = []
    results, board = compare_models_cv(
        X,
        y,
        "regression",
        ["ridge", "lasso", "random_forest"],
        n_folds=3,
        on_model=beats.append,
    )
    assert [beat["index"] for beat in beats] == [1, 2, 3]
    assert board["model"].tolist() == [
        r.model for r in sorted(
            results, key=lambda r: r.metrics["r2"].mean, reverse=True
        )
    ]
    # the formatted column + strategy/folds context on every row
    assert board["strategy"].eq("kfold").all() and board["folds"].eq(3).all()
    assert board["cv_score"].str.contains("±").all()
    assert (board["metric"] == "R²").all()
    # same folds for every model (comparable board): identical fold sizes
    sizes = [
        tuple(record.n_train for record in result.folds) for result in results
    ]
    assert len(set(sizes)) == 1


def test_compare_models_cv_sorts_lower_is_better_ascending(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    _results, board = compare_models_cv(
        X, y, "regression", ["ridge", "lasso"], n_folds=3, primary="neg_root_mean_squared_error"
    )
    # RMSE displayed positive (lower better) -> smallest error first
    assert board["rmse_mean"].is_monotonic_increasing


def test_compare_models_cv_collapses_aliases_and_stops_on_false(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    results, board = compare_models_cv(
        X, y, "regression", ["rf", "random_forest"], n_folds=3
    )
    assert len(results) == 1 and board["model"].tolist() == ["random_forest"]

    def stop_after_first(info: dict) -> bool:
        return info["index"] < 1  # False after model 1 -> cancel

    with pytest.raises(JobCancelled, match="cancelled after 1/2"):
        compare_models_cv(
            X, y, "regression", ["ridge", "lasso"], n_folds=3, on_model=stop_after_first
        )


def test_compare_models_cv_guards(toy_regression: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_regression
    with pytest.raises(ModelError, match="empty"):
        compare_models_cv(X, y, "regression", [], n_folds=3)
    with pytest.raises(ModelError, match="banana"):
        compare_models_cv(X, y, "regression", ["banana"], n_folds=3)


# -- §19.9: persistence ---------------------------------------------------- #


def test_cv_result_json_round_trip(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    result = cross_validate_model(
        X,
        y,
        "regression",
        "ridge",
        n_folds=4,
        best_params={"alpha": 1.0},
    )
    payload = cv_result_to_dict(result)
    # JSON-serialisable as-is
    encoded = json.dumps(payload)
    rebuilt = cv_result_from_dict(json.loads(encoded))
    assert rebuilt.model == result.model
    assert rebuilt.task == result.task
    assert rebuilt.strategy == result.strategy
    assert rebuilt.n_folds == result.n_folds
    assert rebuilt.seed == result.seed
    assert rebuilt.primary == result.primary
    assert rebuilt.best_params == {"alpha": 1.0}
    assert rebuilt.dataset_id == result.dataset_id
    assert rebuilt.feature_columns == result.feature_columns
    assert rebuilt.toolkit_version == result.toolkit_version
    assert rebuilt.created_at == result.created_at
    assert rebuilt.summary() == result.summary()
    assert [f.fold for f in rebuilt.folds] == [1, 2, 3, 4]
    assert rebuilt.folds[0].scores == pytest.approx(result.folds[0].scores)
    assert rebuilt.folds[0].train_scores == pytest.approx(
        {k: v for k, v in result.folds[0].train_scores.items() if v is not None}
    )
    # the §19.9 field inventory
    for key in (
        "model",
        "dataset_id",
        "feature_columns",
        "strategy",
        "folds",
        "metrics",
        "best_params",
        "notes",
        "duration_s",
        "toolkit_version",
        "created_at",
    ):
        assert key in payload


def test_cv_result_from_dict_rejects_foreign_payloads() -> None:
    with pytest.raises(DataValidationError, match="not a pharmaml CV result"):
        cv_result_from_dict({"schema": "something/else"})
    with pytest.raises(DataValidationError, match="not a pharmaml CV result"):
        cv_result_from_dict([1, 2, 3])  # type: ignore[arg-type]
    # right schema, missing mandatory field
    with pytest.raises(DataValidationError, match="missing field"):
        cv_result_from_dict({"schema": "pharmaml.cv_result/1", "model": "ridge"})


def test_dataset_fingerprint_stable_and_sensitive(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    assert dataset_fingerprint(X, y) == dataset_fingerprint(X.copy(), y.copy())
    # a different target -> a different fingerprint
    assert dataset_fingerprint(X, y) != dataset_fingerprint(X, y * 2.0)
    # a different feature block -> a different fingerprint
    X2 = X.copy()
    X2.iloc[0, 0] += 1.0
    assert dataset_fingerprint(X, y) != dataset_fingerprint(X2, y)
    # object-typed targets fall back to their string rendering
    assert len(dataset_fingerprint(X, ["a", "b"] * 24)) == 16


def test_board_frame_single_row(
    toy_regression: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_regression
    result = cross_validate_model(X, y, "regression", "ridge", n_folds=3)
    one = result.board_frame()
    assert len(one) == 1
    assert one.iloc[0]["model"] == "ridge"
    assert one.iloc[0]["cv_score"] == result.summary()
    assert one.iloc[0]["metric"] == "R²"
    assert one.iloc[0]["duration_s"] >= 0.0

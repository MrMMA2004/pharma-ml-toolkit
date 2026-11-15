"""Semi-supervised learning: validation, training, and honest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.semi_supervised import (
    _self_training_estimator,
    _self_training_kwarg_name,
    evaluate_semi_supervised,
    train_semi_supervised,
    validate_unlabeled,
)

_METHODS = ("self_training", "label_propagation", "label_spreading")


def _synthetic(n_samples: int = 200, mask_step: int = 5, seed: int = 0):
    """Well-clustered binary data with 1-in-``mask_step`` labels masked."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    y_masked = y.copy()
    y_masked[::mask_step] = -1
    return X, y, y_masked


# ---------------------------------------------------------------------------
# validate_unlabeled
# ---------------------------------------------------------------------------


def test_validate_counts_and_sentinel_conversion():
    y = np.array([0, 1, -1, 0, 1, -1, 1, 0])
    labels, info = validate_unlabeled(y)
    assert info["n_labeled"] == 6 and info["n_unlabeled"] == 2
    assert info["class_counts"] == {0: 3, 1: 3}
    assert (labels == y).all()  # already in convention, unchanged
    assert (labels == -1).sum() == 2


def test_validate_string_minus_one_class_is_not_sentinel():
    # A pandas Series whose ACTUAL class label is the string "-1":
    # the numeric default marker must not eat it.
    series = pd.Series(["-1", "active", "inactive", "-1", "active"])
    labels, info = validate_unlabeled(series)
    assert info["n_unlabeled"] == 0
    assert info["n_labeled"] == 5
    assert sorted(info["class_counts"]) == ["-1", "active", "inactive"]
    assert info["class_counts"]["-1"] == 2
    # Nothing was converted, so no sentinel appears in the labels.
    assert not (np.asarray(labels) == -1).any()


def test_validate_numeric_minus_one_class_survives_custom_marker():
    # Numeric class -1 IS a real class here; a custom string marker
    # means only "NA" converts.
    series = pd.Series([-1, 0, 1, -1, 0])
    _labels, info = validate_unlabeled(series, unlabeled_marker="NA")
    assert info["n_unlabeled"] == 0
    assert info["class_counts"] == {-1: 2, 0: 2, 1: 1}


def test_validate_custom_marker_string():
    series = pd.Series(["A", "B", "unlabeled", "A", "unlabeled", "B"])
    labels, info = validate_unlabeled(series, unlabeled_marker="unlabeled")
    assert info["n_labeled"] == 4 and info["n_unlabeled"] == 2
    assert info["class_counts"] == {"A": 2, "B": 2}
    assert (np.asarray(labels) == -1).sum() == 2


def test_validate_nan_marker():
    y = np.array([0.0, 1.0, np.nan, 0.0, np.nan, 1.0])
    _labels, info = validate_unlabeled(y, unlabeled_marker=np.nan)
    assert info["n_labeled"] == 4 and info["n_unlabeled"] == 2
    assert info["class_counts"] == {0.0: 2, 1.0: 2}


def test_validate_marker_none_keeps_everything():
    _labels, info = validate_unlabeled([5, 7, 5, 7], unlabeled_marker=None)
    assert info["n_labeled"] == 4 and info["n_unlabeled"] == 0
    assert info["class_counts"] == {5: 2, 7: 2}


@pytest.mark.parametrize(
    "y, marker, message",
    [
        ([-1, -1, -1], -1, "0 labeled sample"),
        (["A", "A", "A"], -1, "1 distinct real class"),
        ([], -1, "empty"),
    ],
)
def test_validate_error_cases(y, marker, message):
    with pytest.raises(DataValidationError, match=message):
        validate_unlabeled(y, unlabeled_marker=marker)


# ---------------------------------------------------------------------------
# train_semi_supervised — shape contracts, determinism, per-method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _METHODS)
def test_train_result_contract(method):
    X, y, y_masked = _synthetic()
    result = train_semi_supervised(X, y_masked, method=method, max_iter=30)
    n = X.shape[0]
    assert result.method == method
    assert result.labels_.shape == (n,)
    assert result.pseudo_label_mask.shape == (n,)
    assert result.pseudo_label_mask.dtype == bool
    assert int(result.pseudo_label_mask.sum()) == int((y_masked == -1).sum())
    assert result.n_labeled == int((y_masked != -1).sum())
    assert result.n_unlabeled == int((y_masked == -1).sum())
    assert -1 not in set(np.asarray(result.classes_).tolist())
    assert result.confidence is not None
    assert result.confidence.shape == (n,)
    assert np.isfinite(result.confidence).all()
    assert result.fit_seconds >= 0.0
    # Genuine labels are never overridden.
    kept = ~result.pseudo_label_mask
    assert (np.asarray(result.labels_[kept]) == y_masked[kept]).all()
    # The pseudo-labeled rows actually recover their true classes on
    # this well-clustered set — the whole point of the exercise.
    recovered = float(
        (np.asarray(result.labels_[result.pseudo_label_mask])
         == y[result.pseudo_label_mask]).mean()
    )
    assert recovered >= 0.8, f"{method} recovered only {recovered:.2f}"
    # params is the reproducible config.
    for key in ("method", "threshold", "k_best", "max_iter", "kernel",
                "n_neighbors", "gamma", "sklearn_version", "estimator_class"):
        assert key in result.params
    assert result.params["sklearn_version"]
    assert result.fit_seconds >= 0.0


@pytest.mark.parametrize("method", _METHODS)
def test_train_determinism_same_seed(method):
    X, _y, y_masked = _synthetic()
    first = train_semi_supervised(X, y_masked, method=method, max_iter=30,
                                  random_state=7)
    second = train_semi_supervised(X, y_masked, method=method, max_iter=30,
                                   random_state=7)
    assert np.array_equal(first.labels_, second.labels_)
    assert np.allclose(first.confidence, second.confidence)


def test_train_self_training_fitted_wrapper_is_the_model():
    X, _y, y_masked = _synthetic(n_samples=100)
    result = train_semi_supervised(X, y_masked, method="self_training")
    assert isinstance(result.model, SelfTrainingClassifier)
    # Inductive: the fitted wrapper predicts genuinely new rows.
    assert result.model.predict(X[:4]).shape == (4,)


def test_train_graph_methods_are_transductive_models():
    X, _y, y_masked = _synthetic(n_samples=100)
    lp = train_semi_supervised(X, y_masked, method="label_propagation",
                               max_iter=30)
    ls = train_semi_supervised(X, y_masked, method="label_spreading",
                               max_iter=30)
    assert isinstance(lp.model, LabelPropagation)
    assert isinstance(ls.model, LabelSpreading)


def test_train_string_classes_roundtrip():
    X, y, _ = _synthetic(n_samples=120)
    series = pd.Series(np.where(y == 0, "active", "inactive"))
    masked = series.copy()
    masked.iloc[::5] = "unlabeled"
    result = train_semi_supervised(X, masked, method="label_spreading",
                                   unlabeled_marker="unlabeled", max_iter=30)
    assert result.n_unlabeled == 24
    assert sorted(np.asarray(result.classes_).tolist()) == ["active", "inactive"]
    kept = ~result.pseudo_label_mask
    assert (np.asarray(result.labels_[kept])
            == np.asarray(series)[kept]).all()
    # Pseudo-labels speak the caller's class language too.
    assert set(np.asarray(result.labels_[result.pseudo_label_mask]).tolist()) \
        <= {"active", "inactive"}


def test_train_confidence_self_training_is_predict_proba_max():
    X, _y, y_masked = _synthetic(n_samples=100)
    result = train_semi_supervised(X, y_masked, method="self_training")
    expected = result.model.predict_proba(X).max(axis=1)
    assert np.allclose(result.confidence, expected)


def test_train_confidence_graph_is_distribution_max():
    X, _y, y_masked = _synthetic(n_samples=100)
    result = train_semi_supervised(X, y_masked, method="label_propagation",
                                   max_iter=30)
    expected = result.model.label_distributions_.max(axis=1)
    assert np.allclose(result.confidence, expected)


def test_train_dataframe_features_and_custom_estimator():
    X, _y, y_masked = _synthetic(n_samples=120)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    forest = RandomForestClassifier(n_estimators=25, random_state=0)
    result = train_semi_supervised(frame, y_masked, method="self_training",
                                   base_estimator=forest)
    assert result.params["estimator_class"] == "RandomForestClassifier"
    assert result.confidence is not None


def test_train_row_count_mismatch_rejected():
    X, _y, y_masked = _synthetic(n_samples=60)
    with pytest.raises(DataValidationError, match="y has 5 value"):
        train_semi_supervised(X, y_masked[:5])


# ---------------------------------------------------------------------------
# parameter validation (fail-closed before any fit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"threshold": 0.0}, "threshold must be in \\(0, 1\\)"),
        ({"threshold": 1.0}, "threshold must be in \\(0, 1\\)"),
        ({"threshold": 1.5}, "threshold must be in \\(0, 1\\)"),
        ({"k_best": -1}, "k_best must be >= 0"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"n_neighbors": 0}, "n_neighbors must be >= 1"),
        ({"gamma": 0.0}, "gamma must be a positive"),
    ],
)
def test_train_invalid_params(kwargs, message):
    X, _y, y_masked = _synthetic(n_samples=60)
    with pytest.raises(DataValidationError, match=message):
        train_semi_supervised(X, y_masked, **kwargs)


def test_train_unknown_method_rejected():
    X, _y, y_masked = _synthetic(n_samples=60)
    with pytest.raises(DataValidationError, match="method must be one of"):
        train_semi_supervised(X, y_masked, method="pseudo_magic")


def test_train_non_numeric_base_estimator_rejected():
    X, _y, y_masked = _synthetic(n_samples=60)
    from sklearn.tree import DecisionTreeRegressor

    with pytest.raises(DataValidationError, match="predict_proba"):
        train_semi_supervised(X, y_masked, base_estimator=DecisionTreeRegressor())


def test_train_non_numeric_features_rejected():
    X, y, _ = _synthetic(n_samples=40)
    frame = pd.DataFrame(X[:, :3], columns=["a", "b", "c"])
    frame["smiles"] = "CCO"
    with pytest.raises(DataValidationError, match="non-numeric column"):
        train_semi_supervised(frame, y)


def test_train_nan_features_rejected():
    X, y, _ = _synthetic(n_samples=40)
    X[0, 0] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        train_semi_supervised(X, y)


def test_train_k_best_zero_runs():
    X, _y, y_masked = _synthetic(n_samples=80)
    result = train_semi_supervised(X, y_masked, method="self_training",
                                   k_best=0)
    assert result.params["k_best"] == 0
    assert result.pseudo_label_mask.sum() == (y_masked == -1).sum()


# ---------------------------------------------------------------------------
# sklearn version compatibility
# ---------------------------------------------------------------------------


def test_self_training_estimator_compat_kwarg():
    name = _self_training_kwarg_name()
    assert name in ("estimator", "base_estimator")
    from sklearn.linear_model import LogisticRegression

    wrapper = _self_training_estimator(
        LogisticRegression(max_iter=100), threshold=0.8, max_iter=5
    )
    assert isinstance(wrapper, SelfTrainingClassifier)
    # Whichever spelling this sklearn accepts, the wrapper was built
    # with it and carries the estimator.
    assert wrapper.threshold == 0.8
    base = getattr(wrapper, "estimator", None) or getattr(
        wrapper, "base_estimator", None
    )
    assert isinstance(base, LogisticRegression)


# ---------------------------------------------------------------------------
# evaluate_semi_supervised — leakage-safe protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _METHODS)
def test_evaluate_heldout_metrics_beat_random(method):
    X, _y, y_masked = _synthetic()  # 200 rows, 40% masked
    evaluation = evaluate_semi_supervised(X, y_masked, method=method,
                                          n_repeats=2, max_iter=30)
    assert evaluation.method == method
    assert evaluation.heldout_accuracy is not None
    assert evaluation.heldout_f1_macro is not None
    assert evaluation.baseline_accuracy is not None
    assert evaluation.baseline_f1_macro is not None
    assert evaluation.train_accuracy_on_labeled is not None
    assert 0.0 <= evaluation.heldout_accuracy <= 1.0
    assert 0.0 <= evaluation.heldout_f1_macro <= 1.0
    # Well-structured semi-supervised learning beats random on this set.
    assert evaluation.heldout_accuracy >= 0.7
    assert evaluation.heldout_f1_macro >= 0.7
    # The summary is one honest line naming the method and both numbers.
    text = evaluation.summary()
    assert method in text and "accuracy" in text


def test_evaluate_baseline_is_supervised_on_labeled_only():
    X, _y, y_masked = _synthetic()
    evaluation = evaluate_semi_supervised(X, y_masked, method="self_training",
                                          n_repeats=1)
    assert evaluation.baseline_accuracy is not None
    # On this easy set the baseline is itself strong; both are sane.
    assert 0.5 <= evaluation.baseline_accuracy <= 1.0


def test_evaluate_honest_none_when_too_few_labeled():
    X, _y, _ = _synthetic(n_samples=100)
    y_few = np.full(100, -1)
    y_few[:3] = 0        # 6 labeled rows total, 3 per class
    y_few[50:53] = 1
    evaluation = evaluate_semi_supervised(X, y_few, n_repeats=1)
    assert evaluation.heldout_accuracy is None
    assert evaluation.heldout_f1_macro is None
    assert evaluation.baseline_accuracy is None
    assert evaluation.baseline_f1_macro is None
    assert evaluation.train_accuracy_on_labeled is None
    text = evaluation.summary()
    assert "not possible" in text


def test_evaluate_one_class_per_split_guarded():
    # 4 rows per class passes the count guard, but only 2 per class
    # must fall through to the honest-None path instead of crashing.
    X, _y, _ = _synthetic(n_samples=100)
    y_few = np.full(100, -1)
    y_few[:2] = 0
    y_few[50:52] = 1
    evaluation = evaluate_semi_supervised(X, y_few, n_repeats=1)
    assert evaluation.heldout_accuracy is None
    assert "not possible" in evaluation.summary()


def test_evaluate_invalid_repeats_and_test_size():
    X, _y, y_masked = _synthetic(n_samples=60)
    with pytest.raises(DataValidationError, match="n_repeats must be >= 1"):
        evaluate_semi_supervised(X, y_masked, n_repeats=0)
    with pytest.raises(DataValidationError, match="test_size must be in"):
        evaluate_semi_supervised(X, y_masked, test_size=0.0)
    with pytest.raises(DataValidationError, match="test_size must be in"):
        evaluate_semi_supervised(X, y_masked, test_size=1.0)


def test_evaluate_string_classes():
    X, y, _ = _synthetic(n_samples=150)
    series = pd.Series(np.where(y == 0, "active", "inactive"))
    masked = series.copy()
    masked.iloc[::5] = "unlabeled"
    evaluation = evaluate_semi_supervised(
        X, masked, method="label_spreading",
        unlabeled_marker="unlabeled", n_repeats=2, max_iter=30,
    )
    assert evaluation.heldout_accuracy is not None
    assert evaluation.heldout_accuracy >= 0.7
    assert evaluation.baseline_accuracy is not None


def test_evaluate_forwards_train_kwargs():
    X, _y, y_masked = _synthetic(n_samples=120)
    evaluation = evaluate_semi_supervised(
        X, y_masked, method="self_training", threshold=0.9, n_repeats=1,
    )
    assert evaluation.heldout_accuracy is not None


# ---------------------------------------------------------------------------
# to_frame contract
# ---------------------------------------------------------------------------


def test_to_frame_contract():
    X, _y, y_masked = _synthetic(n_samples=100)
    result = train_semi_supervised(X, y_masked, method="self_training")
    frame = result.to_frame()
    assert list(frame.columns) == [
        "name", "final_label", "confidence", "was_pseudo_labeled",
    ]
    assert len(frame) == 100
    assert (frame["final_label"].to_numpy() == np.asarray(result.labels_)).all()
    assert np.allclose(frame["confidence"].to_numpy(), result.confidence)
    assert (frame["was_pseudo_labeled"].to_numpy()
            == result.pseudo_label_mask).all()
    assert list(frame["name"][:3]) == ["sample_0", "sample_1", "sample_2"]


def test_to_frame_custom_names_padded():
    X, _y, y_masked = _synthetic(n_samples=50)
    result = train_semi_supervised(X, y_masked, method="label_propagation",
                                   max_iter=30)
    frame = result.to_frame(names=["cmp-1", "cmp-2"])
    assert list(frame["name"][:2]) == ["cmp-1", "cmp-2"]
    assert list(frame["name"][2:4]) == ["sample_2", "sample_3"]
    assert len(frame) == 50

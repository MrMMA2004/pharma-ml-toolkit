"""ROC-AUC must never guess around invalid class or probability contracts."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pharmaml.ml import evaluation
from pharmaml.ml.evaluation import evaluate_model


@pytest.fixture(autouse=True)
def warning_spy():
    # Other test modules may reconfigure logging, so avoid relying on caplog.
    with patch.object(evaluation._LOG, "warning") as warning:
        yield warning


class ProbabilityModel:
    def __init__(self, predicted, proba, classes=None):
        self.predicted = predicted
        self.proba = proba
        if classes is not None:
            self.classes_ = classes

    def predict(self, X):
        return np.asarray(self.predicted)

    def predict_proba(self, X):
        return self.proba


def evaluate(model, y):
    return evaluate_model(model, np.zeros((len(y), 1)), y, "classification")


@pytest.mark.parametrize("classes", [["x", "y"], ["a", "a"], ["a"], ["a", "b", "c"], [["a"], ["b"]]])
def test_inconsistent_class_metadata_never_falls_back(classes, warning_spy):
    y = ["a", "b", "a", "b"]
    model = ProbabilityModel(y, [[0.9, 0.1], [0.1, 0.9]] * 2, classes)
    report = evaluate(model, y)
    assert report.accuracy == 1.0
    assert report.roc_auc is None
    assert warning_spy.call_args.args[0] == "ROC-AUC could not be computed: %s"


@pytest.mark.parametrize("proba", [
    [[0.9, 0.1], [0.1, 0.9]],  # row count
    [0.1, 0.9, 0.1, 0.9],  # 1-D
    [[0.9, 0.1, 0.0]] * 4,  # column count
    [[0.9, np.nan]] * 4,
    [[0.9, np.inf]] * 4,
    [[1.1, -0.1]] * 4,
    [[0.2, 0.2]] * 4,
    [["bad", "data"]] * 4,
])
def test_invalid_probability_matrix_does_not_produce_auc(proba):
    y = [0, 1, 0, 1]
    report = evaluate(ProbabilityModel(y, proba, [0, 1]), y)
    assert report.accuracy == 1.0
    assert report.roc_auc is None


def test_single_class_has_no_auc_without_calling_proba():
    class Single:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            raise AssertionError("one class cannot define AUC")

    assert evaluate(Single(), [0, 0, 0]).roc_auc is None


def test_binary_reversed_classes_and_list_probabilities():
    y = ["low", "high", "low", "high"]
    # sorted last is "low"; reversed metadata puts it in column 0
    proba = [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]
    model = ProbabilityModel(y, proba, ["low", "high"])
    assert evaluate(model, y).roc_auc == pytest.approx(1.0)


def test_no_class_metadata_uses_documented_sorted_column_convention():
    y = [0, 1, 0, 1]
    proba = [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]
    assert evaluate(ProbabilityModel(y, proba), y).roc_auc == pytest.approx(1.0)


def test_multiclass_class_order_mapping_and_mismatch():
    y = ["a", "b", "c", "a", "b", "c"]
    classes = ["c", "a", "b"]
    proba = np.array([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]] * 2)
    report = evaluate(ProbabilityModel(y, proba, classes), y)
    assert report.roc_auc == pytest.approx(1.0)
    assert evaluate(ProbabilityModel(y, proba, ["c", "a", "other"]), y).roc_auc is None
    assert evaluate(ProbabilityModel(y, proba), y).roc_auc != pytest.approx(1.0)


def test_missing_training_class_is_not_silently_renormalized():
    y = [0, 1, 0, 1]
    proba = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]] * 2
    assert evaluate(ProbabilityModel(y, proba, [0, 1, 2]), y).roc_auc is None


def test_real_binary_and_multiclass_estimators_match_sklearn():
    for n_classes in (2, 3):
        X = np.arange(36, dtype=float).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 36 // n_classes)
        model = LogisticRegression(max_iter=500).fit(X, y)
        report = evaluate_model(model, X, y, "classification")
        proba = model.predict_proba(X)
        expected = (roc_auc_score(y, proba[:, 1]) if n_classes == 2 else
                    roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_))
        assert report.roc_auc == pytest.approx(expected)

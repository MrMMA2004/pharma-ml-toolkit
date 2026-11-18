"""Classification conformal scores, class mapping, coverage and validation."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import calibrate_split_conformal_classifier


class ProbabilityModel:
    def __init__(self, classes=("a", "b", "c")):
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        return np.asarray(X, dtype=float)


def test_exact_rank_sets_ties_and_empirical_coverage():
    X = [[0.9, 0.05, 0.05], [0.7, 0.15, 0.15],
         [0.4, 0.3, 0.3], [0.2, 0.4, 0.4]]
    cal = calibrate_split_conformal_classifier(ProbabilityModel(), X, ["a"] * 4, alpha=0.4)
    assert cal.rank == 3 and cal.n_calibration == 4
    assert cal.threshold == pytest.approx(0.6)
    query = [[0.4, 0.45, 0.15], [0.34, 0.33, 0.33], [1.0, 0.0, 0.0]]
    result = cal.predict_set(query, ["b", "a", "new"])
    assert result.prediction_sets == (("a", "b"), (), ("a",))
    np.testing.assert_array_equal(result.set_sizes, [2, 0, 1])
    np.testing.assert_array_equal(result.covered, [True, False, False])
    assert result.coverage == pytest.approx(1 / 3)
    assert result.n_covered == 1 and result.n_test == 3 and result.mean_set_size == 1
    assert list(result.to_dataframe().columns) == ["prediction_set", "set_size", "covered"]
    no_truth = cal.predict_set(query)
    assert no_truth.coverage is None and no_truth.n_covered is None and no_truth.covered is None
    assert list(no_truth.to_dataframe().columns) == ["prediction_set", "set_size"]


def test_small_calibration_returns_full_set_not_clipped_quantile():
    cal = calibrate_split_conformal_classifier(ProbabilityModel(), [[1, 0, 0]], ["a"], alpha=0.1)
    assert cal.rank == 2 and cal.threshold == 1
    result = cal.predict_set([[0, 1, 0], [0.2, 0.3, 0.5]], ["c", "b"])
    assert result.prediction_sets == (("a", "b", "c"), ("a", "b", "c"))
    assert result.coverage == 1


def test_class_order_and_label_types_are_preserved():
    model = ProbabilityModel((2, 0, 1))
    X = [[0.9, 0.05, 0.05]] * 9
    cal = calibrate_split_conformal_classifier(model, X, np.array([2] * 9), alpha=0.1)
    result = cal.predict_set([[0.9, 0.05, 0.05]], [2])
    assert result.classes == (2, 0, 1) and result.prediction_sets == ((2,),)
    assert result.coverage == 1
    model.classes_ = np.array([0, 1, 2])
    with pytest.raises(DataValidationError, match="classes_ changed"):
        cal.predict_set([[0.9, 0.05, 0.05]])


@pytest.mark.parametrize("alpha", [0, 1, -1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_alpha(alpha):
    with pytest.raises(DataValidationError, match="alpha"):
        calibrate_split_conformal_classifier(ProbabilityModel(), [[1, 0, 0]], ["a"], alpha=alpha)


@pytest.mark.parametrize("classes", [[], ["a"], ["a", "a"], ["a", None], ["a", np.nan], [["a"], ["b"]]])
def test_invalid_classes(classes):
    model = ProbabilityModel()
    model.classes_ = classes
    with pytest.raises(DataValidationError, match="classes_"):
        calibrate_split_conformal_classifier(model, [[1, 0, 0]], ["a"])


@pytest.mark.parametrize("truth", [[], [["a"]], [None], [float("nan")], [float("inf")], ["unknown"]])
def test_invalid_calibration_labels(truth):
    with pytest.raises(DataValidationError, match="y_cal"):
        calibrate_split_conformal_classifier(ProbabilityModel(), [[1, 0, 0]], truth)


@pytest.mark.parametrize("proba", [
    [0.9, 0.1], [[0.5, 0.5]], [[0.9, 0.1, float("nan")]],
    [[1.1, -0.1, 0]], [[0.3, 0.3, 0.3]], [[0.2, 0.3, float("inf")]],
    [["invalid", 0, 1]],
])
def test_invalid_probability_outputs(proba):
    class Bad(ProbabilityModel):
        def predict_proba(self, X):
            return proba

    with pytest.raises(DataValidationError, match="probab|predict_proba"):
        calibrate_split_conformal_classifier(Bad(), [[0]], ["a"])


def test_missing_model_methods_and_mismatched_rows():
    with pytest.raises(DataValidationError, match="predict_proba"):
        calibrate_split_conformal_classifier(object(), [[0]], ["a"])
    with pytest.raises(DataValidationError, match="same length"):
        calibrate_split_conformal_classifier(ProbabilityModel(), [[1, 0, 0]], ["a", "b"])
    cal = calibrate_split_conformal_classifier(ProbabilityModel(), [[1, 0, 0]], ["a"])
    with pytest.raises(DataValidationError, match="same length"):
        cal.predict_set([[1, 0, 0]], ["a", "b"])
    with pytest.raises(DataValidationError, match="at least one row"):
        cal.predict_set([])
    with pytest.raises(DataValidationError, match="y_true"):
        cal.predict_set([[1, 0, 0]], [None])


def test_real_multiclass_pipeline_dataframe_and_sparse_input():
    X, y = make_classification(n_samples=200, n_features=8, n_informative=6,
                               n_classes=3, random_state=2)
    X = pd.DataFrame(X)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(X.iloc[:100], y[:100])
    means = model[0].mean_.copy()
    cal = calibrate_split_conformal_classifier(model, X.iloc[100:150], y[100:150], alpha=0.1)
    result = cal.predict_set(X.iloc[150:], y[150:])
    np.testing.assert_array_equal(model[0].mean_, means)
    assert result.membership.shape == (50, 3)
    assert 0 <= result.coverage <= 1
    sparse_model = LogisticRegression(max_iter=300).fit(csr_matrix(X.iloc[:100].values), y[:100])
    sparse_cal = calibrate_split_conformal_classifier(sparse_model, csr_matrix(X.iloc[100:150].values),
                                                       y[100:150])
    assert sparse_cal.predict_set(csr_matrix(X.iloc[150:].values)).n_test == 50


def test_seeded_exchangeable_coverage_simulation():
    rng = np.random.default_rng(914)
    model = ProbabilityModel()
    covered = 0
    for _ in range(1200):
        X = rng.dirichlet([1, 1, 1], size=20)
        y = rng.choice(["a", "b", "c"], size=20)
        cal = calibrate_split_conformal_classifier(model, X[:19], y[:19], alpha=0.1)
        covered += cal.predict_set(X[19:], y[19:]).n_covered
    assert 0.87 < covered / 1200 < 0.94


def test_full_set_does_not_cover_unknown_class():
    cal = calibrate_split_conformal_classifier(ProbabilityModel(), [[1, 0, 0]], ["a"])
    result = cal.predict_set([[0, 1, 0]], ["unseen"])
    assert result.prediction_sets == (("a", "b", "c"),)
    assert result.coverage == 0


def test_invalid_model_probabilities_at_prediction_time_are_rejected():
    class Mutable(ProbabilityModel):
        def predict_proba(self, X):
            return np.asarray(X) if not getattr(self, "changed", False) else [[0, 0, 0]]

    model = Mutable()
    cal = calibrate_split_conformal_classifier(model, [[1, 0, 0]], ["a"])
    model.changed = True
    with pytest.raises(DataValidationError, match="probabilities"):
        cal.predict_set([[1, 0, 0]])


def test_scores_are_taken_from_the_true_label_columns():
    model = ProbabilityModel(("b", "a", "c"))
    X = [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8], [0.6, 0.1, 0.3]]
    cal = calibrate_split_conformal_classifier(model, X, ["a", "a", "c", "c"], alpha=0.4)
    # True-label scores .9, .3, .2, .7; rank 3 is .7, not a score
    # taken from the predicted class or from the sorted class order.
    assert cal.threshold == pytest.approx(0.7)

"""Label-conditional calibration ranks, fallbacks, coverage and integration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import calibrate_label_conditional_classifier, coverage_by_group


class Model:
    def __init__(self, classes=("a", "b")):
        self.classes_ = list(classes)

    def predict_proba(self, X):
        return np.asarray(X, dtype=float)


def test_class_ranks_thresholds_ties_and_empirical_coverage():
    # True label A scores .1, .2, .4, .6; B scores .1, .3, .7, .9.
    a = [0.9, 0.8, 0.6, 0.4]
    b = [0.9, 0.7, 0.3, 0.1]
    X = [[p, 1 - p] for p in a] + [[1 - p, p] for p in b]
    y = ["a"] * 4 + ["b"] * 4
    cal = calibrate_label_conditional_classifier(Model(), X, y, alpha=0.4)
    assert cal.counts == (4, 4) and cal.ranks == (3, 3)
    assert cal.thresholds == pytest.approx((0.4, 0.7))
    assert cal.n_calibration == 8
    query = [[0.3, 0.7], [0.6, 0.4], [0.9, 0.1]]
    result = cal.predict_set(query, ["b", "a", "new"])
    assert result.prediction_sets == (("b",), ("a", "b"), ("a",))
    np.testing.assert_array_equal(result.covered, [True, True, False])
    assert result.coverage == pytest.approx(2 / 3)
    assert result.class_thresholds == cal.thresholds and result.threshold is None
    assert result.mean_set_size == pytest.approx(4 / 3)
    frame = cal.calibration_summary()
    assert list(frame.columns) == ["class_label", "n_calibration", "rank", "threshold", "fallback_full_class"]
    assert frame.class_label.tolist() == ["a", "b"]
    assert not frame.fallback_full_class.any()
    assert coverage_by_group(result.covered, ["x", "x", "y"], sizes=result.set_sizes).overall.coverage == result.coverage
    assert cal.predict_set(query).covered is None


def test_empty_and_small_classes_are_always_included():
    # a has nine samples and a finite rank, b has one and c has none.
    X = [[0.9, 0.05, 0.05]] * 9 + [[0.05, 0.9, 0.05]]
    cal = calibrate_label_conditional_classifier(Model(("a", "b", "c")), X,
                                                 ["a"] * 9 + ["b"], alpha=0.1)
    assert cal.counts == (9, 1, 0)
    assert cal.ranks == (9, 2, 1)
    assert cal.thresholds == pytest.approx((0.1, 1.0, 1.0))
    assert cal.calibration_summary().fallback_full_class.tolist() == [False, True, True]
    sets = cal.predict_set([[1, 0, 0], [0, 1, 0]], ["c", "b"])
    assert sets.prediction_sets == (("a", "b", "c"), ("b", "c"))
    assert sets.coverage == 1
    assert cal.predict_set([[0, 1, 0]], ["outside"]).coverage == 0


def test_class_order_and_shuffled_calibration_rows():
    model = Model((2, 0, 1))
    X = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 10
    y = [2, 0, 1] * 10
    order = np.random.default_rng(9).permutation(30)
    X, y = np.asarray(X)[order], np.asarray(y)[order]
    cal = calibrate_label_conditional_classifier(model, X, y)
    assert cal.classes == (2, 0, 1) and cal.counts == (10, 10, 10)
    assert cal.thresholds == pytest.approx((0.2, 0.2, 0.2))
    assert cal.predict_set([[0.8, 0.1, 0.1]], [2]).prediction_sets == ((2,),)
    model.classes_ = [0, 1, 2]
    with pytest.raises(DataValidationError, match="classes_ changed"):
        cal.predict_set([[0.8, 0.1, 0.1]])


@pytest.mark.parametrize("alpha", [0, 1, True, -1, np.nan, np.inf, "0.1"])
def test_invalid_alpha(alpha):
    with pytest.raises(DataValidationError, match="alpha"):
        calibrate_label_conditional_classifier(Model(), [[1, 0]], ["a"], alpha=alpha)


@pytest.mark.parametrize("y", [[], [["a"]], [None], [np.nan], ["unknown"]])
def test_invalid_calibration_labels(y):
    with pytest.raises(DataValidationError, match="y_cal"):
        calibrate_label_conditional_classifier(Model(), [[1, 0]], y)


def test_invalid_models_probabilities_and_query_labels():
    with pytest.raises(DataValidationError, match="predict_proba"):
        calibrate_label_conditional_classifier(object(), [[1, 0]], ["a"])
    with pytest.raises(DataValidationError, match="same length"):
        calibrate_label_conditional_classifier(Model(), [[1, 0]], ["a", "b"])
    for bad in ([[0.4, 0.4]], [[np.nan, 1]], [[-0.1, 1.1]], [[1, 0, 0]]):
        with pytest.raises(DataValidationError, match="probab|predict_proba"):
            calibrate_label_conditional_classifier(Model(), bad, ["a"])
    cal = calibrate_label_conditional_classifier(Model(), [[1, 0]], ["a"])
    with pytest.raises(DataValidationError, match="same length"):
        cal.predict_set([[1, 0]], ["a", "b"])
    with pytest.raises(DataValidationError, match="y_true"):
        cal.predict_set([[1, 0]], [None])
    with pytest.raises(DataValidationError, match="at least one row"):
        cal.predict_set([])


def test_real_pipeline_and_pandas_input_keeps_model_fixed():
    X, y = make_classification(n_samples=300, n_features=8, n_informative=6,
                               n_classes=3, random_state=45)
    X = pd.DataFrame(X)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X.iloc[:150], y[:150])
    before = model[0].mean_.copy()
    cal = calibrate_label_conditional_classifier(model, X.iloc[150:225], y[150:225])
    result = cal.predict_set(X.iloc[225:], y[225:])
    np.testing.assert_array_equal(before, model[0].mean_)
    assert sum(cal.counts) == 75 and result.membership.shape == (75, 3)
    assert 0 <= result.coverage <= 1


def test_seeded_within_class_exchangeability_simulation():
    rng = np.random.default_rng(755)
    hits = [0, 0]
    for _ in range(1200):
        # Equal fixed class counts; query rows come from the same class-specific
        # distributions as their independent calibration counterparts.
        a = rng.dirichlet([4, 2], size=20)
        b = rng.dirichlet([2, 4], size=20)
        cal = calibrate_label_conditional_classifier(Model(), np.vstack((a[:19], b[:19])),
                                                    ["a"] * 19 + ["b"] * 19, alpha=0.1)
        result = cal.predict_set(np.vstack((a[19:], b[19:])), ["a", "b"])
        hits[0] += int(result.covered[0])
        hits[1] += int(result.covered[1])
    assert all(0.87 < value / 1200 < 0.94 for value in hits)


def test_threshold_one_from_observed_scores_is_not_mislabeled_as_fallback():
    cal = calibrate_label_conditional_classifier(Model(), [[0, 1]] * 9, ["a"] * 9)
    assert cal.thresholds == (1, 1)
    # Class a genuinely has rank 9 and a score of 1; class b has no data.
    assert cal.calibration_summary().fallback_full_class.tolist() == [False, True]
    assert cal.predict_set([[0, 1]]).prediction_sets == (("a", "b"),)

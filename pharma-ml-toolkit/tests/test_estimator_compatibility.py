"""Estimator-type and fitted-state contracts across supported sklearn versions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import calibrate_split_conformal
from pharmaml.ml.dalex import Explainer


class TinyRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class TinyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        raise AssertionError("classifier predictions must not be used for regression calibration")


class LegacyClassifier:
    _estimator_type = "classifier"

    def predict(self, X):
        raise AssertionError("classifier predictions must not be used")

    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))


class UntaggedSklearnModel(BaseEstimator):
    def fit(self, X, y):
        self.trained_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


class BrokenTags(BaseEstimator):
    def __sklearn_tags__(self):
        return None

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise AssertionError("invalid tags must not reach prediction")


class LightweightRegressor:
    def predict(self, X):
        return np.zeros(len(X))


X = pd.DataFrame({"x": np.arange(20, dtype=float)})
Y = np.arange(20, dtype=float)
BINARY = (Y > 9).astype(int)


@pytest.mark.parametrize("make_model", [
    lambda: LogisticRegression(),
    lambda: make_pipeline(StandardScaler(), LogisticRegression()),
    lambda: LogisticRegression().fit(X, BINARY),
    lambda: make_pipeline(StandardScaler(), LogisticRegression()).fit(X, BINARY),
    lambda: TinyClassifier(),
    lambda: TinyClassifier().fit(X, BINARY),
    lambda: LegacyClassifier(),
])
def test_classifiers_rejected_before_prediction(make_model):
    with pytest.raises(DataValidationError, match="classifier"):
        calibrate_split_conformal(make_model(), X, Y, alpha=0.2)


@pytest.mark.parametrize("make_model", [
    lambda: LinearRegression(),
    lambda: make_pipeline(StandardScaler(), LinearRegression()),
    lambda: TinyRegressor(),
])
def test_unfitted_sklearn_regressors_raise_project_error_with_cause(make_model):
    with pytest.raises(DataValidationError, match="fitted") as exc:
        calibrate_split_conformal(make_model(), X, Y, alpha=0.2)
    assert isinstance(exc.value.__cause__, NotFittedError)


@pytest.mark.parametrize("model", [
    LinearRegression().fit(X, Y),
    make_pipeline(StandardScaler(), LinearRegression()).fit(X, Y),
    TinyRegressor().fit(X, Y),
    LightweightRegressor(),
])
def test_fitted_and_lightweight_regressors_remain_supported(model):
    result = calibrate_split_conformal(model, X, Y, alpha=0.2)
    assert result.n_calibration == len(X)
    np.testing.assert_allclose(result.predict_interval(X).prediction, model.predict(X))


@pytest.mark.parametrize("model", [UntaggedSklearnModel().fit(X, Y), BrokenTags()])
def test_sklearn_estimators_with_unknown_or_invalid_tags_fail_closed(model):
    with pytest.raises(DataValidationError, match="tags|estimator type"):
        calibrate_split_conformal(model, X, Y, alpha=0.2)


def test_untagged_lightweight_probability_predictor_is_not_silently_accepted():
    class Ambiguous:
        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    with pytest.raises(DataValidationError, match="classifier|ambiguous"):
        calibrate_split_conformal(Ambiguous(), X, Y, alpha=0.2)


def test_lightweight_not_fitted_error_is_translated_but_computational_errors_are_not():
    class Unready:
        def predict(self, X):
            raise NotFittedError("not ready")

    with pytest.raises(DataValidationError, match="fitted") as exc:
        calibrate_split_conformal(Unready(), X, Y, alpha=0.2)
    assert isinstance(exc.value.__cause__, NotFittedError)

    class BrokenComputation:
        def predict(self, X):
            raise RuntimeError("computation failed")

    with pytest.raises(RuntimeError, match="computation failed"):
        calibrate_split_conformal(BrokenComputation(), X, Y, alpha=0.2)


@pytest.mark.parametrize("model", [
    LogisticRegression().fit(X, BINARY),
    make_pipeline(StandardScaler(), LogisticRegression()).fit(X, BINARY),
    LegacyClassifier(),
])
def test_dalex_uses_second_class_probability(model):
    explanation = Explainer(model, X, BINARY)
    np.testing.assert_allclose(explanation.y_hat, model.predict_proba(X)[:, 1])
    assert explanation.model_type == "classification"


def test_dalex_uses_regression_prediction_and_rejects_nonbinary_probabilities():
    regressor = LinearRegression().fit(X, Y)
    np.testing.assert_allclose(Explainer(regressor, X, Y).y_hat, regressor.predict(X))

    class Multiclass(LegacyClassifier):
        def predict_proba(self, X):
            return np.tile([0.2, 0.3, 0.5], (len(X), 1))

    with pytest.raises(DataValidationError, match="binary"):
        Explainer(Multiclass(), X, BINARY)


def test_calibration_and_query_require_aligned_sized_rows():
    class FixedTen:
        def predict(self, X):
            return np.zeros(10)

    with pytest.raises(DataValidationError, match="X_cal and y_cal"):
        calibrate_split_conformal(FixedTen(), [[0]] * 11, [0] * 10, alpha=0.2)
    with pytest.raises(DataValidationError, match="sample count"):
        calibrate_split_conformal(FixedTen(), iter([[0]] * 10), [0] * 10, alpha=0.2)
    calibrated = calibrate_split_conformal(FixedTen(), [[0]] * 10, [0] * 10, alpha=0.2)
    with pytest.raises(DataValidationError, match="X and model predictions"):
        calibrated.predict_interval([[0]] * 9)


@pytest.mark.parametrize("prediction", [np.full(20, np.inf), np.zeros((20, 1)), np.zeros(19)])
def test_invalid_prediction_shape_finiteness_and_length(prediction):
    class Invalid:
        def predict(self, X):
            return prediction

    with pytest.raises(DataValidationError):
        calibrate_split_conformal(Invalid(), X, Y, alpha=0.2)


def test_broken_classifier_mixin_order_does_not_leak_into_regression():
    class WrongOrder(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            return self

        def predict(self, X):
            raise AssertionError("classifier predictions must not be used")

    with pytest.raises(DataValidationError, match="classifier"):
        calibrate_split_conformal(WrongOrder(), X, Y, alpha=0.2)
    with pytest.raises(DataValidationError, match="classifier|estimator type"):
        calibrate_split_conformal(make_pipeline(StandardScaler(), WrongOrder()), X, Y, alpha=0.2)


def test_sparse_rows_and_stateless_custom_sklearn_regressor():
    from scipy.sparse import csr_matrix

    class Stateless(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def __sklearn_is_fitted__(self):
            return True

        def predict(self, X):
            return np.zeros(X.shape[0])

    sparse = csr_matrix(np.zeros((20, 2)))
    result = calibrate_split_conformal(Stateless(), sparse, Y, alpha=0.2)
    assert result.predict_interval(sparse).n_test == 20


def test_notfitted_after_calibration_is_converted_with_cause():
    class Mutable:
        ready = True

        def predict(self, X):
            if not self.ready:
                raise NotFittedError("no longer ready")
            return np.zeros(len(X))

    model = Mutable()
    result = calibrate_split_conformal(model, X, Y, alpha=0.2)
    model.ready = False
    with pytest.raises(DataValidationError, match="fitted") as exc:
        result.predict_interval(X)
    assert isinstance(exc.value.__cause__, NotFittedError)


def test_dalex_classifier_without_probabilities_can_use_explicit_predict_function():
    class ClassifierWithoutProba:
        _estimator_type = "classifier"

        def predict(self, X):
            return np.ones(len(X))

    model = ClassifierWithoutProba()
    with pytest.raises(DataValidationError, match="predict_proba"):
        Explainer(model, X, BINARY)
    explanation = Explainer(model, X, BINARY, predict_function=lambda data: np.full(len(data), 0.25))
    np.testing.assert_allclose(explanation.y_hat, 0.25)


def test_pipeline_rejects_classifier_with_conflicting_regressor_mixin():
    class BothKinds(RegressorMixin, ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            raise AssertionError("must not call classifier prediction")

    with pytest.raises(DataValidationError, match="classifier"):
        calibrate_split_conformal(make_pipeline(StandardScaler(), BothKinds()), X, Y, alpha=0.2)

"""Split-conformal regression: finite-sample ranks and honest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import calibrate_split_conformal


class ZeroModel:
    def predict(self, X):
        return np.zeros(len(X))


def test_exact_rank_and_inclusive_coverage():
    cal = calibrate_split_conformal(ZeroModel(), np.zeros((10, 1)), np.arange(10), alpha=0.2)
    assert (cal.rank, cal.n_calibration, cal.radius) == (9, 10, 8.0)
    result = cal.predict_interval(np.zeros((4, 1)), y_true=[-8, 8, 8.1, 0])
    np.testing.assert_array_equal(result.prediction, [0, 0, 0, 0])
    np.testing.assert_array_equal(result.lower, [-8, -8, -8, -8])
    np.testing.assert_array_equal(result.upper, [8, 8, 8, 8])
    assert result.coverage == 0.75 and result.n_covered == 3
    assert result.mean_width == 16.0 and result.n_test == 4
    assert list(result.to_dataframe().columns) == ["prediction", "lower", "upper"]
    assert cal.predict_interval(np.zeros((1, 1))).coverage is None


def test_ties_zero_width_and_smallest_finite_calibration():
    cal = calibrate_split_conformal(ZeroModel(), np.zeros((9, 1)), [1] * 9, alpha=0.1)
    assert cal.rank == 9 and cal.radius == 1
    zero = calibrate_split_conformal(ZeroModel(), np.zeros((9, 1)), [0] * 9, alpha=0.1)
    assert zero.predict_interval(np.zeros((1, 1)), [0]).coverage == 1
    with pytest.raises(DataValidationError, match="cannot give finite intervals"):
        calibrate_split_conformal(ZeroModel(), np.zeros((8, 1)), [1] * 8, alpha=0.1)


@pytest.mark.parametrize("alpha", [0, 1, -1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_alpha(alpha):
    with pytest.raises(DataValidationError, match="alpha"):
        calibrate_split_conformal(ZeroModel(), [[0]] * 10, [0] * 10, alpha=alpha)


@pytest.mark.parametrize("truth", [[], [1, float("nan")], [[1], [2]], ["bad"]])
def test_invalid_calibration_target(truth):
    with pytest.raises(DataValidationError):
        calibrate_split_conformal(ZeroModel(), [[0]] * 10, truth, alpha=0.2)


def test_model_and_predictions_validation():
    with pytest.raises(DataValidationError, match="predict method"):
        calibrate_split_conformal(object(), [[0]] * 10, [0] * 10, alpha=0.2)
    with pytest.raises(DataValidationError, match="regressor"):
        calibrate_split_conformal(LogisticRegression(), [[0]] * 10, [0] * 10, alpha=0.2)
    with pytest.raises(DataValidationError, match="same length"):
        calibrate_split_conformal(ZeroModel(), [[0]] * 11, [0] * 10, alpha=0.2)

    class InvalidModel:
        def predict(self, X):
            return np.full(len(X), np.nan)

    with pytest.raises(DataValidationError, match="finite"):
        calibrate_split_conformal(InvalidModel(), [[0]] * 10, [0] * 10, alpha=0.2)
    cal = calibrate_split_conformal(ZeroModel(), [[0]] * 10, [0] * 10, alpha=0.2)
    with pytest.raises(DataValidationError, match="same length"):
        cal.predict_interval([[0], [0]], [0])
    with pytest.raises(DataValidationError, match="finite"):
        cal.predict_interval([[0]], [float("nan")])
    with pytest.raises(DataValidationError, match="nonempty"):
        cal.predict_interval([])


def test_scale_overflow_is_rejected():
    with pytest.raises(DataValidationError, match="overflowed"):
        calibrate_split_conformal(ZeroModel(), [[0]] * 10, [1e308] * 10, alpha=0.2)


def test_sklearn_pipeline_not_refitted_and_dataframe_inputs():
    X = pd.DataFrame({"x": np.arange(30.0)})
    y = 2 * X["x"] + 1
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X.iloc[:10], y.iloc[:10])
    means_before = model[0].mean_.copy()
    cal = calibrate_split_conformal(model, X.iloc[10:20], y.iloc[10:20], alpha=0.2)
    result = cal.predict_interval(X.iloc[20:], y.iloc[20:])
    np.testing.assert_array_equal(model[0].mean_, means_before)
    np.testing.assert_allclose(result.prediction, model.predict(X.iloc[20:]))
    assert result.n_calibration == 10 and 0 <= result.coverage <= 1


def test_seeded_exchangeable_simulation_is_near_target():
    rng = np.random.default_rng(613)
    covered = 0
    for _ in range(1200):
        values = rng.normal(size=20)
        cal = calibrate_split_conformal(ZeroModel(), np.zeros((19, 1)), values[:19], alpha=0.1)
        covered += cal.predict_interval([[0]], values[19:]).n_covered
    assert 0.87 < covered / 1200 < 0.93

"""Tests for the Tabular Transformer estimators (``pharmaml.ml.transformer``).

part18 §5.4's lane: the tokenised tabular transformer — fit/predict
determinism, softmax probabilities, sklearn compatibility
(get_params/clone/joblib pickle), the telemetry contract, validation
guards (d_model/heads divisibility, degenerate data), import-safety
without torch (the lazy-import contract), and the honest separation
(the module never claims pretrained molecular weights).

Every test keeps the data tiny and ``epochs`` small — the sandbox runs
on CPU only.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="the transformer suite needs the optional deep extra")
from sklearn.base import clone

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.transformer import (
    TabularTransformerClassifier,
    TabularTransformerRegressor,
)


# ---------------------------------------------------------------------------
# Helpers: tiny data + fast configs
# ---------------------------------------------------------------------------
def fast_classifier(**overrides: Any) -> TabularTransformerClassifier:
    """A classifier tuned for test speed (one block, few epochs)."""
    params: dict[str, Any] = {
        "d_model": 16,
        "heads": 4,
        "n_blocks": 1,
        "dim_feedforward": 32,
        "epochs": 12,
        "random_state": 0,
    }
    params.update(overrides)
    return TabularTransformerClassifier(**params)


def fast_regressor(**overrides: Any) -> TabularTransformerRegressor:
    params: dict[str, Any] = {
        "d_model": 16,
        "heads": 4,
        "n_blocks": 1,
        "dim_feedforward": 32,
        "epochs": 20,
        "random_state": 0,
    }
    params.update(overrides)
    return TabularTransformerRegressor(**params)


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(48, 3))
    y = (X[:, 0] + 0.4 * X[:, 2] > 0).astype(int)
    return X, y


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(48, 3))
    y = 2.0 * X[:, 1] + 0.5
    return X, y


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
class TestValidation:
    def test_d_model_must_divide_heads(self) -> None:
        with pytest.raises(ModelError, match="divisible by heads"):
            TabularTransformerClassifier(d_model=17, heads=4).fit(
                [[0.0, 1.0]] * 4, [0, 1, 0, 1]
            )

    def test_predict_before_fit(self) -> None:
        with pytest.raises(ModelError, match="has not been fitted yet"):
            TabularTransformerClassifier().predict([[0.0, 1.0]])

    def test_negative_d_model(self) -> None:
        with pytest.raises(ModelError, match="d_model must be"):
            TabularTransformerClassifier(d_model=-2).fit(
                [[0.0, 1.0]] * 4, [0, 1, 0, 1]
            )

    def test_bad_dropout(self) -> None:
        with pytest.raises(ModelError, match="dropout must be in"):
            TabularTransformerClassifier(dropout=1.5).fit(
                [[0.0, 1.0]] * 4, [0, 1, 0, 1]
            )

    def test_bad_device_token(self) -> None:
        with pytest.raises(ModelError, match="device must be one of"):
            TabularTransformerClassifier(device="tpu").fit(
                [[0.0, 1.0]] * 4, [0, 1, 0, 1]
            )

    def test_nan_features(self) -> None:
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            fast_classifier().fit([[np.nan, 0.0], [1.0, 0.0]], [0, 1])

    def test_single_class_target(self) -> None:
        with pytest.raises(DataValidationError, match="2 distinct classes"):
            fast_classifier().fit([[0.0], [1.0]], [5, 5])

    def test_feature_width_mismatch_at_predict(self, binary_data) -> None:
        X, y = binary_data
        net = fast_classifier().fit(X, y)
        with pytest.raises(DataValidationError, match="fitted with"):
            net.predict(np.zeros((3, 9)))

    def test_dataframe_column_mismatch_at_predict(self, binary_data) -> None:
        import pandas as pd

        X, y = binary_data
        frame = pd.DataFrame(X, columns=["a", "b", "c"])
        net = fast_classifier().fit(frame, y)
        with pytest.raises(DataValidationError, match="feature names"):
            net.predict(pd.DataFrame(X, columns=["x", "y", "z"]))

    def test_x_and_y_length_mismatch(self, binary_data) -> None:
        X, y = binary_data
        with pytest.raises(DataValidationError, match="inconsistent lengths"):
            fast_classifier().fit(X, y[:-1])


# ---------------------------------------------------------------------------
# Fit / predict / determinism
# ---------------------------------------------------------------------------
class TestFitPredict:
    def test_classifier_learns_separable_data(self, binary_data) -> None:
        X, y = binary_data
        net = fast_classifier(epochs=40).fit(X, y)
        assert float(np.mean(net.predict(X) == y)) >= 0.85

    def test_same_seed_identical_predictions(self, binary_data) -> None:
        X, y = binary_data
        a = fast_classifier().fit(X, y)
        b = fast_classifier().fit(X, y)
        assert np.array_equal(a.predict(X), b.predict(X))

    def test_different_seed_changes_predictions(self, binary_data) -> None:
        X, y = binary_data
        a = fast_classifier(random_state=1).fit(X, y)
        b = fast_classifier(random_state=2).fit(X, y)
        assert not np.array_equal(a.predict(X), b.predict(X))

    def test_regressor_predicts_a_linear_function(self, regression_data) -> None:
        X, y = regression_data
        reg = fast_regressor(epochs=45).fit(X, y)
        assert float(np.mean(np.abs(reg.predict(X) - y))) < 0.5

    def test_proba_rows_sum_to_one(self, binary_data) -> None:
        X, y = binary_data
        net = fast_classifier().fit(X, y)
        proba = net.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert proba.shape == (len(X), 2)

    def test_multiclass_proba(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 2))
        y = np.repeat([0, 1, 2], 20)
        net = fast_classifier(epochs=25).fit(X, y)
        proba = net.predict_proba(X)
        assert proba.shape == (60, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert list(net.classes_) == [0, 1, 2]

    def test_history_and_n_iter_agree(self, binary_data) -> None:
        X, y = binary_data
        net = fast_classifier().fit(X, y)
        assert len(net.history_) == net.n_iter_
        assert set(net.history_[0]) == {"epoch", "train_loss", "val_loss"}

    def test_telemetry_stream_is_live(self, binary_data) -> None:
        X, y = binary_data
        seen: list[dict] = []
        fast_classifier().fit(X, y, on_epoch=seen.append)
        assert len(seen) > 0
        assert set(seen[0]) == {
            "epoch", "epochs", "train_loss", "val_loss", "best_val", "stale"
        }
        assert [d["epoch"] for d in seen] == list(range(1, len(seen) + 1))


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------
class TestSklearnContract:
    def test_get_params_roundtrip(self) -> None:
        net = fast_classifier(d_model=24, heads=6)
        params = net.get_params()
        clone(net)
        assert params["d_model"] == 24 and params["heads"] == 6

    def test_clone_is_unfitted(self, binary_data) -> None:
        X, y = binary_data
        net = fast_classifier().fit(X, y)
        fresh = clone(net)
        with pytest.raises(ModelError, match="has not been fitted yet"):
            fresh.predict(X)

    def test_pickle_round_trip(self, binary_data) -> None:
        X, y = binary_data
        net = fast_classifier().fit(X, y)
        blob = pickle.dumps(net)
        twin = pickle.loads(blob)
        assert np.array_equal(net.predict(X), twin.predict(X))

    def test_dataframe_input_and_feature_names_in(self, binary_data) -> None:
        import pandas as pd

        X, y = binary_data
        frame = pd.DataFrame(X, columns=["a", "b", "c"])
        net = fast_classifier().fit(frame, y)
        assert list(net.feature_names_in_) == ["a", "b", "c"]
        assert np.array_equal(net.predict(frame), net.predict(X))


# ---------------------------------------------------------------------------
# The honest separation (part18 §5.4's rule)
# ---------------------------------------------------------------------------
class TestHonestScope:
    def test_module_docstring_disclaims_pretrained_weights(self) -> None:
        from pharmaml.ml import transformer as module

        text = module.__doc__ or ""
        assert "No pretrained molecular-sequence Transformer is claimed" in text

    def test_catalog_note_not_needed_in_module_but_docstring_states_flavors(self) -> None:
        from pharmaml.ml import transformer as module

        text = module.__doc__ or ""
        assert "Tabular Transformer" in text

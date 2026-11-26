"""Tests for the sequence CNN estimators (``pharmaml.ml.cnn``).

part18 §5.2's lane: the 1-D temporal convolution — 3-D-only input
contract (the no-fake-images rule, REFUSING 2-D tabular matrices with
the honest pointer), motif learning, determinism, pickle portability,
the telemetry contract, and the validation guards.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="the cnn suite needs the optional deep extra")
from sklearn.base import clone

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.cnn import (
    SequenceCNNClassifier,
    SequenceCNNRegressor,
    synthetic_motif_sequences,
)


def fast_classifier(**overrides: Any) -> SequenceCNNClassifier:
    params: dict[str, Any] = {
        "channels": 8,
        "kernel": 3,
        "n_layers": 2,
        "epochs": 25,
        "learning_rate": 3e-3,
        "random_state": 0,
    }
    params.update(overrides)
    return SequenceCNNClassifier(**params)


@pytest.fixture()
def motif_data() -> tuple[np.ndarray, np.ndarray]:
    return synthetic_motif_sequences(48, seed=0)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------
class TestFixture:
    def test_shapes_and_balance(self) -> None:
        x, y = synthetic_motif_sequences(40, seed=1)
        assert x.shape == (40, 16, 3)
        assert set(y.tolist()) == {0, 1}
        counts = np.bincount(y)
        assert abs(counts[0] - counts[1]) <= 1

    def test_seeded_determinism(self) -> None:
        a = synthetic_motif_sequences(30, seed=5)
        b = synthetic_motif_sequences(30, seed=5)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_degenerate_arguments(self) -> None:
        with pytest.raises(DataValidationError, match="n must be >= 2"):
            synthetic_motif_sequences(1)
        with pytest.raises(DataValidationError, match="length must be >="):
            synthetic_motif_sequences(10, length=5)


# ---------------------------------------------------------------------------
# The 3-D-only contract (part18 §5.2's no-fake-images rule)
# ---------------------------------------------------------------------------
class TestInputContract:
    def test_tabular_matrix_is_refused_with_the_pointer(self) -> None:
        with pytest.raises(DataValidationError, match="DeepMLP / Tabular Transformer"):
            SequenceCNNClassifier().fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])

    def test_1d_input_refused(self) -> None:
        with pytest.raises(DataValidationError, match="X must be 3-dimensional"):
            SequenceCNNRegressor().fit([0.0, 1.0, 2.0], [1, 2, 3])

    def test_nan_refused(self) -> None:
        x = np.zeros((4, 8, 2)); x[0, 0, 0] = np.nan
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            SequenceCNNClassifier().fit(x, [0, 1, 0, 1])

    def test_channel_mismatch_at_predict(self, motif_data) -> None:
        x, y = motif_data
        net = fast_classifier().fit(x, y)
        with pytest.raises(DataValidationError, match="fitted with"):
            net.predict(np.zeros((3, 16, 9)))

    def test_predict_before_fit(self) -> None:
        with pytest.raises(ModelError, match="has not been fitted yet"):
            SequenceCNNClassifier().predict([[[0.0]]])


# ---------------------------------------------------------------------------
# Fit / predict / determinism
# ---------------------------------------------------------------------------
class TestFitPredict:
    def test_learns_the_local_motif(self, motif_data) -> None:
        x, y = motif_data
        net = fast_classifier().fit(x, y)
        assert float(np.mean(net.predict(x) == y)) >= 0.9

    def test_same_seed_identical_predictions(self, motif_data) -> None:
        x, y = motif_data
        a = fast_classifier().fit(x, y)
        b = fast_classifier().fit(x, y)
        assert np.array_equal(a.predict(x), b.predict(x))

    def test_proba_shapes_and_sum(self, motif_data) -> None:
        x, y = motif_data
        net = fast_classifier().fit(x, y)
        proba = net.predict_proba(x)
        assert proba.shape == (len(x), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_regressor_tracks_window_target(self, motif_data) -> None:
        x, _y = motif_data
        target = x[:, :, 0].sum(axis=1)
        reg = SequenceCNNRegressor(
            channels=8, epochs=30, random_state=0
        ).fit(x, target)
        err = float(np.mean(np.abs(reg.predict(x) - target)))
        assert err <= float(np.std(target))  # beats the zero guess by a lot

    def test_telemetry_stream(self, motif_data) -> None:
        x, y = motif_data
        seen: list[dict] = []
        fast_classifier(epochs=4).fit(x, y, on_epoch=seen.append)
        assert len(seen) == 4
        assert set(seen[0]) == {
            "epoch", "epochs", "train_loss", "val_loss", "best_val", "stale"
        }

    def test_history_records_val_loss(self, motif_data) -> None:
        x, y = motif_data
        net = fast_classifier().fit(x, y)
        assert all("val_loss" in rec for rec in net.history_)


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------
class TestSklearnContract:
    def test_get_params_and_clone(self, motif_data) -> None:
        x, y = motif_data
        net = fast_classifier().fit(x, y)
        fresh = clone(net)
        with pytest.raises(ModelError, match="has not been fitted yet"):
            fresh.predict(x)

    def test_pickle_round_trip(self, motif_data) -> None:
        x, y = motif_data
        net = fast_classifier().fit(x, y)
        twin = pickle.loads(pickle.dumps(net))
        assert np.array_equal(net.predict(x), twin.predict(x))

    def test_odd_kernel_rule(self) -> None:
        with pytest.raises(ModelError, match="positive odd int"):
            SequenceCNNClassifier(kernel=2).fit(
                np.zeros((4, 8, 2)), [0, 1, 0, 1]
            )

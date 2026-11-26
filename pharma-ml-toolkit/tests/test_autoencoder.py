"""Tests for the tabular autoencoder (``pharmaml.ml.autoencoder``).

part18 §5.5's lane: latent representation learning, reconstruction,
dimensionality reduction (a 2-D latent as the scatter-plotable
nonlinear projection), anomaly detection by reconstruction error
(an honest RANKING — never ground truth), determinism, pickle
portability, and the validation guards.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="the autoencoder suite needs the optional deep extra")
from sklearn.base import clone

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.autoencoder import TabularAutoencoder


def fast_ae(**overrides: Any) -> TabularAutoencoder:
    params: dict[str, Any] = {
        "hidden_layers": (16,),
        "latent_dim": 2,
        "epochs": 40,
        "random_state": 0,
    }
    params.update(overrides)
    return TabularAutoencoder(**params)


@pytest.fixture()
def gaussian_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=1.0, scale=0.4, size=(60, 5))


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
class TestValidation:
    def test_encode_before_fit(self) -> None:
        with pytest.raises(ModelError, match="has not been fitted yet"):
            TabularAutoencoder().encode([[0.0, 1.0]])

    def test_latent_dim_positive(self) -> None:
        with pytest.raises(ModelError, match="latent_dim must be a positive"):
            TabularAutoencoder(latent_dim=0).fit([[0.0, 1.0]] * 2)

    def test_nan_refused(self) -> None:
        X = np.zeros((4, 3)); X[0, 0] = np.nan
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            fast_ae().fit(X)

    def test_width_mismatch_at_encode(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        with pytest.raises(DataValidationError, match="fitted with"):
            ae.encode(np.zeros((3, 9)))

    def test_dataframe_names_checked(self, gaussian_data) -> None:
        frame = pd.DataFrame(gaussian_data, columns=list("abcde"))
        ae = fast_ae().fit(frame)
        with pytest.raises(DataValidationError, match="feature names"):
            ae.encode(pd.DataFrame(gaussian_data, columns=list("vwxyz")))

    def test_fit_accepts_y_none_and_ignores_y(self, gaussian_data) -> None:
        ae = fast_ae(epochs=3).fit(gaussian_data, y=None)
        assert ae.n_iter_ == 3


# ---------------------------------------------------------------------------
# Representation / reconstruction / dimensionality reduction
# ---------------------------------------------------------------------------
class TestRepresentations:
    def test_latent_shape(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        assert ae.encode(gaussian_data).shape == (60, 2)

    def test_reconstruction_shape_and_scale(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        recon = ae.reconstruct(gaussian_data)
        assert recon.shape == gaussian_data.shape
        # ORIGINAL scale: reconstructions live near the data's mean (1.0),
        # not near the standardised zero
        assert float(np.abs(recon.mean() - 1.0)) < 0.5

    def test_reconstruction_error_is_nonnegative_per_row(
        self, gaussian_data
    ) -> None:
        ae = fast_ae().fit(gaussian_data)
        err = ae.reconstruction_errors(gaussian_data)
        assert err.shape == (60,)
        assert np.all(err >= 0)

    def test_training_reduces_reconstruction_error(self, gaussian_data) -> None:
        early = fast_ae(epochs=2).fit(gaussian_data)
        late = fast_ae().fit(gaussian_data)
        assert late.reconstruction_errors(gaussian_data).mean() < (
            early.reconstruction_errors(gaussian_data).mean()
        )

    def test_latent_dim_3_for_3d_projection(self, gaussian_data) -> None:
        ae = fast_ae(latent_dim=3, epochs=3).fit(gaussian_data)
        assert ae.encode(gaussian_data).shape[1] == 3

    def test_telemetry_stream(self, gaussian_data) -> None:
        seen: list[dict] = []
        fast_ae(epochs=4).fit(gaussian_data, on_epoch=seen.append)
        assert len(seen) == 4
        assert set(seen[0]) == {
            "epoch", "epochs", "train_loss", "val_loss", "best_val", "stale"
        }


# ---------------------------------------------------------------------------
# Anomaly detection (reconstruction-error ranking)
# ---------------------------------------------------------------------------
class TestAnomalyRanking:
    def test_injected_outlier_ranks_worst(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        X2 = np.vstack([gaussian_data, np.array([12.0, -12.0, 12.0, -12.0, 12.0])])
        ranking = np.argsort(ae.reconstruction_errors(X2))[::-1]
        assert int(ranking[0]) == len(X2) - 1

    def test_ranking_is_a_ranking_not_a_label(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        errors = ae.reconstruction_errors(gaussian_data)
        assert len(np.unique(np.round(errors, 8))) > 1  # varied, not constant


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------
class TestSklearnContract:
    def test_clone_unfitted(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        fresh = clone(ae)
        with pytest.raises(ModelError, match="has not been fitted yet"):
            fresh.encode(gaussian_data)

    def test_pickle_round_trip(self, gaussian_data) -> None:
        ae = fast_ae().fit(gaussian_data)
        twin = pickle.loads(pickle.dumps(ae))
        assert np.array_equal(ae.encode(gaussian_data), twin.encode(gaussian_data))

    def test_same_seed_identical_latents(self, gaussian_data) -> None:
        a = fast_ae().fit(gaussian_data)
        b = fast_ae().fit(gaussian_data)
        assert np.array_equal(a.encode(gaussian_data), b.encode(gaussian_data))

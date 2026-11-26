"""Tests for the feature-space generative models (part18 §5.6).

:mod:`pharmaml.generative.gan` (``FeatureSpaceGAN``) and
:mod:`pharmaml.generative.diffusion`` (``FeatureSpaceDiffusion``):
the adversarial and DDPM lanes over numeric feature matrices —
seeded reproducibility, statistical-parity quality reports, the
telemetry contracts, pickle portability, validation guards, and the
HONEST SCOPE (feature-space, never molecules — pinned so a future
change cannot silently claim molecular generation).
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

pytest.importorskip(
    "torch", reason="the feature-space generative suite needs the optional deep extra"
)
from sklearn.base import clone

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.generative.diffusion import FeatureSpaceDiffusion
from pharmaml.generative.gan import FeatureSpaceGAN


def fast_gan(**overrides: Any) -> FeatureSpaceGAN:
    params: dict[str, Any] = {
        "latent_dim": 8,
        "hidden_layers": (32,),
        "epochs": 40,
        "random_state": 0,
    }
    params.update(overrides)
    return FeatureSpaceGAN(**params)


def fast_diffusion(**overrides: Any) -> FeatureSpaceDiffusion:
    params: dict[str, Any] = {
        "timesteps": 30,
        "hidden_layers": (32,),
        "epochs": 40,
        "random_state": 0,
    }
    params.update(overrides)
    return FeatureSpaceDiffusion(**params)


@pytest.fixture()
def feature_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=1.5, scale=0.3, size=(180, 3))


# ---------------------------------------------------------------------------
# Validation guards (shared shape)
# ---------------------------------------------------------------------------
class TestValidation:
    def test_sample_before_fit_gan(self) -> None:
        with pytest.raises(ModelError, match="has not been fitted yet"):
            FeatureSpaceGAN().sample(2)

    def test_sample_before_fit_diffusion(self) -> None:
        with pytest.raises(ModelError, match="has not been fitted yet"):
            FeatureSpaceDiffusion().sample(2)

    def test_gan_latent_dim_positive(self) -> None:
        with pytest.raises(ModelError, match="latent_dim must be a positive"):
            fast_gan(latent_dim=0).fit([[0.0, 1.0]] * 4)

    def test_diffusion_timesteps_at_least_two(self) -> None:
        with pytest.raises(ModelError, match="timesteps must be >= 2"):
            fast_diffusion(timesteps=1).fit([[0.0, 1.0]] * 4)

    def test_nan_refused(self) -> None:
        X = np.zeros((8, 3)); X[0, 0] = np.nan
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            fast_gan().fit(X)
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            fast_diffusion().fit(X)

    def test_bad_sample_n(self, feature_data) -> None:
        gan = fast_gan(epochs=2).fit(feature_data)
        with pytest.raises(ModelError, match="n must be a positive"):
            gan.sample(0)

    def test_bad_seed_type(self, feature_data) -> None:
        gan = fast_gan(epochs=2).fit(feature_data)
        with pytest.raises(ModelError, match="seed must be an integer"):
            gan.sample(2, seed="seven")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The honest scope (part18 §5.6 — pinned)
# ---------------------------------------------------------------------------
class TestHonestScope:
    def test_gan_docstring_disclaims_molecules(self) -> None:
        from pharmaml.generative import gan as module

        text = module.__doc__ or ""
        assert "NOT molecular generation" in text
        assert "feature-space" in text.lower()

    def test_diffusion_docstring_disclaims_molecules(self) -> None:
        from pharmaml.generative import diffusion as module

        text = module.__doc__ or ""
        assert "NOT molecular generation" in text

    def test_samples_are_feature_rows_not_strings(self, feature_data) -> None:
        gan = fast_gan(epochs=3).fit(feature_data)
        rows = gan.sample(4)
        assert rows.shape == (4, 3)
        assert np.issubdtype(rows.dtype, np.floating)


# ---------------------------------------------------------------------------
# GAN: sampling / parity / telemetry / portability
# ---------------------------------------------------------------------------
class TestGan:
    def test_seeded_sample_reproducible(self, feature_data) -> None:
        gan = fast_gan().fit(feature_data)
        a = gan.sample(6, seed=5)
        b = gan.sample(6, seed=5)
        assert np.array_equal(a, b)

    def test_seeded_sample_does_not_disturb_global_rng(self, feature_data) -> None:
        import torch

        gan = fast_gan(epochs=2).fit(feature_data)
        before = torch.get_rng_state().clone()
        gan.sample(4, seed=7)
        assert torch.equal(torch.get_rng_state(), before)

    def test_quality_report_parity(self, feature_data) -> None:
        gan = fast_gan().fit(feature_data)
        report = gan.quality_report()
        assert set(report) == {"n_ref", "mean_delta", "std_delta", "corr_l1"}
        assert report["n_ref"] == 180
        # a trained GAN's sampled means land near the training means
        assert float(np.max(np.abs(report["mean_delta"]))) <= 0.8

    def test_training_telemetry_contract(self, feature_data) -> None:
        seen: list[dict] = []
        fast_gan(epochs=4).fit(feature_data, on_epoch=seen.append)
        assert len(seen) == 4
        assert set(seen[0]) == {
            "epoch", "epochs", "g_loss", "d_loss", "d_real_acc", "d_fake_acc"
        }

    def test_history_is_the_telemetry(self, feature_data) -> None:
        gan = fast_gan(epochs=5).fit(feature_data)
        assert len(gan.history_) == 5
        assert all(np.isfinite(rec["g_loss"]) for rec in gan.history_)

    def test_pickle_round_trip(self, feature_data) -> None:
        gan = fast_gan().fit(feature_data)
        twin = pickle.loads(pickle.dumps(gan))
        assert np.array_equal(
            gan.sample(4, seed=3), twin.sample(4, seed=3)
        )

    def test_clone_unfitted(self, feature_data) -> None:
        gan = fast_gan().fit(feature_data)
        fresh = clone(gan)
        with pytest.raises(ModelError, match="has not been fitted yet"):
            fresh.sample(2)

    def test_alternating_updates_drive_d_loss(self, feature_data) -> None:
        # the adversarial game produces FINITE telemetry on both sides
        gan = fast_gan().fit(feature_data)
        losses = [rec["d_loss"] for rec in gan.history_]
        assert all(np.isfinite(v) for v in losses)
        assert any(v > 0.0 for v in losses)  # a real game, not a zero solver


# ---------------------------------------------------------------------------
# Diffusion: sampling / parity / telemetry / portability
# ---------------------------------------------------------------------------
class TestDiffusion:
    def test_seeded_sample_reproducible(self, feature_data) -> None:
        dpm = fast_diffusion().fit(feature_data)
        a = dpm.sample(6, seed=5)
        b = dpm.sample(6, seed=5)
        assert np.array_equal(a, b)

    def test_sample_shape_and_scale(self, feature_data) -> None:
        dpm = fast_diffusion().fit(feature_data)
        rows = dpm.sample(8, seed=1)
        assert rows.shape == (8, 3)
        # ORIGINAL scale: the reverse chain ends near the data's mean
        assert float(np.abs(rows.mean() - feature_data.mean())) < 1.0

    def test_quality_report_parity(self, feature_data) -> None:
        dpm = fast_diffusion().fit(feature_data)
        report = dpm.quality_report()
        assert set(report) == {"n_ref", "mean_delta", "std_delta", "corr_l1"}
        assert float(np.max(np.abs(report["mean_delta"]))) <= 0.9

    def test_training_telemetry_contract(self, feature_data) -> None:
        seen: list[dict] = []
        fast_diffusion(epochs=4).fit(feature_data, on_epoch=seen.append)
        assert len(seen) == 4
        assert set(seen[0]) == {"epoch", "epochs", "train_loss"}

    def test_noise_prediction_improves_with_training(self, feature_data) -> None:
        early = fast_diffusion(epochs=2).fit(feature_data)
        late = fast_diffusion().fit(feature_data)
        assert late.history_[-1]["train_loss"] < early.history_[-1]["train_loss"]

    def test_pickle_round_trip(self, feature_data) -> None:
        dpm = fast_diffusion().fit(feature_data)
        twin = pickle.loads(pickle.dumps(dpm))
        assert np.array_equal(
            dpm.sample(4, seed=3), twin.sample(4, seed=3)
        )

    def test_schedule_is_stored_and_portable(self, feature_data) -> None:
        dpm = fast_diffusion(timesteps=20).fit(feature_data)
        assert dpm._schedule_["timesteps"] == 20
        assert len(dpm._schedule_["beta"]) == 20
        beta = np.asarray(dpm._schedule_["beta"])
        # the linear schedule's endpoints are the DDPM paper's values
        assert beta[0] == pytest.approx(1e-4, rel=1e-6)
        assert beta[-1] == pytest.approx(0.02, rel=1e-3)

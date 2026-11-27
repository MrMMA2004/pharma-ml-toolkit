"""Feature-space GAN (PyTorch) — part18 §5.6.

The Deep Learning architecture expansion's adversarial lane, living
in the Generative AI family where the spec puts it (never the
supervised model registry).  The honest scope, stated once and
everywhere it matters:

* this is a **feature-space generative model**: the generator
  learns the joint distribution of a *numeric feature matrix* (the
  descriptor rows of a registered dataset) and samples SYNTHETIC
  FEATURE ROWS from it — statistical parity against the training
  distribution is the quality bar (per-feature mean/std deltas,
  correlation-matrix distance), reported honestly by
  :meth:`FeatureSpaceGAN.quality_report`.
* it is **NOT molecular generation**: no SMILES, no sanitisation, no
  drug-likeness claim.  Molecules come from the SMILES lanes of this
  package (the RNN prior, the genetic optimiser, MCTS) and nothing
  here supersedes them.  Nothing generated is experimentally
  validated data — synthetic rows are synthetic rows.

The machinery is the classic two-player game: a generator
``G(z) -> row`` over ``z ~ N(0, I)`` and a discriminator
``D(row) -> logit`` trained adversarially with BCE logits loss and
alternating updates (``d_steps`` discriminator steps per generator
step), minibatch Adam, per-epoch telemetry (``g_loss`` / ``d_loss`` /
discriminator accuracies) through the same ``on_epoch`` callback
convention :mod:`pharmaml.ml.neural` established.  Determinism is the
usual contract: seeds for NumPy and torch, the global torch RNG
snapshot-and-restored, portable NumPy weight dicts, honest typed
errors.  ``sample`` draws on the ORIGINAL feature scale (the GAN
trains on per-feature standardised rows; sampling inverts it).

Examples
--------
>>> from pharmaml.generative.gan import FeatureSpaceGAN
>>> FeatureSpaceGAN().sample(2)   # fit before sample
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: FeatureSpaceGAN has not been fitted yet — call fit(X) first

>>> FeatureSpaceGAN(latent_dim=0).fit([[0.0, 1.0]] * 4)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: latent_dim must be a positive int, got 0

>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(loc=2.0, scale=0.5, size=(200, 3))
>>> gan = FeatureSpaceGAN(latent_dim=8, hidden_layers=(32,), epochs=60,
...     random_state=0).fit(X)
>>> gan.sample(4).shape     # synthetic rows on the original scale
(4, 3)
>>> gan.sample(4, seed=7).shape   # a seeded draw
(4, 3)

Statistical parity — sampled means land near the training means
(loose tolerance: a GAN is a game, not an exact solver):

>>> report = gan.quality_report()
>>> round(float(np.max(np.abs(report['mean_delta']))), 1) <= 0.8
True
>>> sorted(report.keys())
['corr_l1', 'mean_delta', 'n_ref', 'std_delta']

Same seed, identical sample stream:

>>> twin = gan.sample(5, seed=3)
>>> int(np.array_equal(gan.sample(5, seed=3), twin))
1

The training telemetry contract (a live epoch stream, never
fabricated):

>>> seen = []
>>> _g = FeatureSpaceGAN(latent_dim=4, hidden_layers=(8,), epochs=3,
...     random_state=0).fit(X[:24], on_epoch=seen.append)
>>> [set(d) == {'epoch', 'epochs', 'g_loss', 'd_loss', 'd_real_acc',
...             'd_fake_acc'} for d in seen]
[True, True, True]
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.logger import get_logger

__all__ = ["FeatureSpaceGAN"]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when torch is missing.
_TORCH_HINT = (
    "deep learning models require torch — from the toolkit package root run: "
    "python -m pip install -e '.[deep]'"
)

#: Legal ``device`` tokens ('auto' resolves to cuda when available).
_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

#: The module-wide default seed (the toolkit's convention).
_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Lazy torch access
# ---------------------------------------------------------------------------
def _require_torch() -> ModuleType:
    """Import and return torch, or raise the actionable error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise DependencyError(f"{_TORCH_HINT} ({exc})") from exc
    return torch


def _build_gan(
    n_features: int,
    hidden: tuple[int, ...],
    latent_dim: int,
) -> Any:
    """The generator/discriminator pair (two torch ``Modules``).

    G: ``latent -> hidden (ReLU) -> n_features``; D: ``n_features ->
    hidden (LeakyReLU 0.2) -> 1 logit``.  Defined lazily; the methods
    take their ``torch`` references through local imports.
    """
    from torch import nn

    class _Generator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            width = latent_dim
            for h in hidden:
                layers.append(nn.Linear(width, h))
                layers.append(nn.ReLU())
                width = h
            layers.append(nn.Linear(width, n_features))
            self.net = nn.Sequential(*layers)

        def forward(self, z: Any) -> Any:
            return self.net(z)

    class _Discriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            width = n_features
            for h in hidden:
                layers.append(nn.Linear(width, h))
                layers.append(nn.LeakyReLU(0.2))
                width = h
            layers.append(nn.Linear(width, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return _Generator(), _Discriminator()


class FeatureSpaceGAN(BaseEstimator):
    """Feature-space generative adversarial network (PyTorch).

    Trains a generator/discriminator pair on the numeric feature
    matrix ``X`` (per-feature standardised at fit; ``sample`` returns
    ORIGINAL-scale rows).  Adversarial training: alternating
    discriminator (``d_steps`` per generator step) and generator
    updates with BCE logits loss, minibatch Adam, no early stopping
    (the game has no validation loss — every epoch is honest
    telemetry, the losses and the discriminator accuracies).

    Parameters
    ----------
    latent_dim:
        Generator noise width.
    hidden_layers:
        Hidden widths shared by G and D.
    d_steps:
        Discriminator updates per generator update.
    epochs, batch_size, learning_rate, weight_decay, random_state, device:
        The usual :mod:`pharmaml.ml.neural`-style contract.

    Attributes
    ----------
    n_features_in_:
        Feature width seen at fit.
    history_:
        Per-epoch dicts with ``g_loss`` / ``d_loss`` / ``d_real_acc``
        / ``d_fake_acc`` (adversarial telemetry, never a validation
        loss — GANs have none).

    Notes
    -----
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.  FEATURE-SPACE generative model: synthetic
    rows, not molecules (the SMILES lanes own molecular generation);
    nothing sampled is experimentally validated data.
    """

    def __init__(
        self,
        latent_dim=16,
        hidden_layers=(64,),
        d_steps=1,
        epochs=100,
        batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.0,
        random_state=_RANDOM_STATE,
        device="auto",
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.d_steps = d_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    # -- validation -----------------------------------------------------
    def _validate_hyperparameters(self) -> tuple[int, ...]:
        hidden = self.hidden_layers
        if isinstance(hidden, (str, bytes)) or not hasattr(hidden, "__iter__"):
            raise ModelError(
                f"hidden_layers must be a sequence of positive ints, got {hidden!r}"
            )
        hidden = tuple(hidden)
        for width in hidden:
            if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
                raise ModelError(
                    "hidden_layers entries must be positive ints, got "
                    f"{width!r} in {tuple(hidden)!r}"
                )
        if isinstance(self.latent_dim, bool) or not isinstance(
            self.latent_dim, (int, np.integer)
        ) or self.latent_dim < 1:
            raise ModelError(f"latent_dim must be a positive int, got {self.latent_dim!r}")
        if isinstance(self.d_steps, bool) or not isinstance(
            self.d_steps, (int, np.integer)
        ) or self.d_steps < 1:
            raise ModelError(f"d_steps must be a positive int, got {self.d_steps!r}")
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ModelError(f"{name} must be an integer >= 1, got {value!r}")
        if not np.isfinite(self.learning_rate) or float(self.learning_rate) <= 0:
            raise ModelError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not np.isfinite(self.weight_decay) or float(self.weight_decay) < 0:
            raise ModelError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if self.device not in _DEVICES:
            raise ModelError(f"device must be one of {_DEVICES}, got {self.device!r}")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ModelError(
                f"random_state must be an integer seed (determinism contract), "
                f"got {self.random_state!r}"
            )
        return hidden

    def _check_X(self, X: Any) -> np.ndarray:
        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"features must be numeric — conversion failed: {exc}"
            ) from exc
        if X_arr.ndim != 2:
            raise DataValidationError(
                "X must be 2-dimensional (n_samples, n_features), got "
                f"{X_arr.ndim} dimension(s)"
            )
        if X_arr.shape[0] == 0:
            raise DataValidationError(
                f"X has 0 rows — {type(self).__name__} needs at least one sample"
            )
        if X_arr.shape[1] == 0:
            raise DataValidationError(
                f"X has 0 feature columns — {type(self).__name__} needs at least "
                "one input feature"
            )
        if not np.isfinite(X_arr).all():
            raise DataValidationError(
                "X contains NaN or infinite values — impute or drop them first "
                "(pharmaml.ml.build_preprocessor does this inside a Pipeline)"
            )
        return X_arr

    def _check_fitted(self) -> None:
        if getattr(self, "_weights", None) is None:
            raise ModelError(
                f"{type(self).__name__} has not been fitted yet — call fit(X) "
                "first"
            )

    # -- fit ------------------------------------------------------------
    def fit(
        self, X: Any, y: Any = None, *, on_epoch: Callable[[dict[str, Any]], None] | None = None
    ) -> FeatureSpaceGAN:
        """Train the adversarial pair on ``X`` (unsupervised; ``y`` ignored).

        ``on_epoch`` receives ``{'epoch', 'epochs', 'g_loss',
        'd_loss', 'd_real_acc', 'd_fake_acc'}`` after every epoch —
        the adversarial telemetry (accuracies are the discriminator's
        fraction-correct on real / generated rows) — and must never
        raise.  No early stopping: the game has no validation loss to
        stop on (the honest GAN statement).
        """
        hidden = self._validate_hyperparameters()
        X_arr = self._check_X(X)

        _require_torch()  # before any fitted attribute is written
        device = self._resolve_device()

        self.n_features_in_ = int(X_arr.shape[1])
        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self._mean_ = mean
        self._std_ = std
        self._n_train_rows_ = int(X_arr.shape[0])
        # the training correlation matrix (standardised space), kept for
        # quality_report's parity check — the honest quality bar
        self._corr_ref_ = (
            np.corrcoef((X_arr - mean) / std, rowvar=False)
            if int(X_arr.shape[1]) > 1
            else np.zeros((1, 1))
        )
        X_scaled = (X_arr - mean) / std

        import torch

        rng_state = torch.get_rng_state()
        try:
            self._train_torch(X_scaled, device, hidden, on_epoch=on_epoch)
        finally:
            torch.set_rng_state(rng_state)
        return self

    def _resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ModelError(
                "device='cuda' requested but no CUDA device is visible to torch"
            )
        return self.device

    def _train_torch(
        self,
        X_scaled: np.ndarray,
        device: str,
        hidden: tuple[int, ...],
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """The alternating adversarial loop (minibatch Adam, BCE logits)."""
        import torch
        from torch import nn

        torch.manual_seed(int(self.random_state))
        rng = np.random.default_rng(int(self.random_state))

        self._net_spec_: dict[str, Any] = {
            "n_features": int(X_scaled.shape[1]),
            "hidden": hidden,
            "latent_dim": int(self.latent_dim),
        }
        generator, discriminator = _build_gan(
            self._net_spec_["n_features"],
            self._net_spec_["hidden"],
            self._net_spec_["latent_dim"],
        )
        generator.to(device)
        discriminator.to(device)

        bce: Any = nn.BCEWithLogitsLoss()
        g_opt = torch.optim.Adam(
            generator.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        d_opt = torch.optim.Adam(
            discriminator.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
        n = int(X_scaled.shape[0])
        epochs = int(self.epochs)
        batch_size = min(int(self.batch_size), n)
        d_steps = int(self.d_steps)
        latent_dim = int(self.latent_dim)

        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            g_loss_val = d_loss_val = float("nan")
            real_acc = fake_acc = float("nan")
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = torch.as_tensor(
                    perm[start : start + batch_size], dtype=torch.long, device=device
                )
                rows = X_t[batch]
                members = int(batch.numel())
                if members < 2:  # a lone tail row adds no batch statistics
                    continue
                # -- discriminator: real -> 1, generated (detached) -> 0
                for _ in range(d_steps):
                    z = torch.randn(members, latent_dim, device=device)
                    with torch.no_grad():
                        fake = generator(z)
                    real_logits = discriminator(rows)
                    fake_logits = discriminator(fake)
                    d_real = bce(real_logits, torch.ones(members, 1, device=device))
                    d_fake = bce(fake_logits, torch.zeros(members, 1, device=device))
                    d_loss = 0.5 * (d_real + d_fake)
                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()
                    d_loss_val = float(d_loss.item())
                    real_acc = float(
                        (real_logits.detach() > 0).float().mean().item()
                    )
                    fake_acc = float(
                        (fake_logits.detach() <= 0).float().mean().item()
                    )
                # -- generator: make D call the generated rows real
                z = torch.randn(members, latent_dim, device=device)
                g_loss = bce(discriminator(generator(z)), torch.ones(members, 1, device=device))
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                g_loss_val = float(g_loss.item())
            epoch_rec = {
                "epoch": epoch,
                "epochs": epochs,
                "g_loss": g_loss_val,
                "d_loss": d_loss_val,
                "d_real_acc": real_acc,
                "d_fake_acc": fake_acc,
            }
            history.append(epoch_rec)
            if on_epoch is not None:
                on_epoch(epoch_rec)

        weights = {}
        for prefix, module in (("g.", generator), ("d.", discriminator)):
            for key, value in module.state_dict().items():
                weights[f"{prefix}{key}"] = value.detach().cpu().numpy().copy()
        if not all(np.isfinite(value).all() for value in weights.values()):
            raise ModelError(
                "adversarial training diverged — the fitted weights contain "
                "NaN/inf values; lower learning_rate and refit"
            )
        self.history_ = history
        self._weights = weights
        self._generator = generator
        self._discriminator = discriminator

        _LOG.info(
            "%s fitted on %d rows x %d features: %d epochs, final g_loss %.4f "
            "d_loss %.4f (device=%s)",
            type(self).__name__,
            n,
            int(X_scaled.shape[1]),
            epochs,
            history[-1]["g_loss"] if history else float("nan"),
            history[-1]["d_loss"] if history else float("nan"),
            device,
        )

    # -- sampling -----------------------------------------------------------
    def _ensure_networks(self) -> Any:
        """Rebuild the live G/D modules from the portable weights."""
        generator = getattr(self, "_generator", None)
        if generator is None:
            _require_torch()
            import torch

            spec = self._net_spec_
            generator, discriminator = _build_gan(
                spec["n_features"], spec["hidden"], spec["latent_dim"]
            )
            g_state = {
                key[2:]: torch.from_numpy(value)
                for key, value in self._weights.items() if key.startswith("g.")
            }
            d_state = {
                key[2:]: torch.from_numpy(value)
                for key, value in self._weights.items() if key.startswith("d.")
            }
            generator.load_state_dict(g_state)
            discriminator.load_state_dict(d_state)
            self._generator = generator
            self._discriminator = discriminator
        return self._generator

    def sample(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """Draw ``n`` synthetic feature rows (ORIGINAL feature scale).

        ``seed`` makes the draw reproducible (the same fitted model +
        the same seed → the same rows); without it the draw rides the
        process RNG.
        """
        self._check_fitted()
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ModelError(f"n must be a positive int, got {n!r}")
        _require_torch()
        import torch

        generator = self._ensure_networks()
        generator.eval()
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise ModelError(f"seed must be an integer, got {seed!r}")
            gen_state = torch.get_rng_state()
            try:
                torch.manual_seed(int(seed))
                with torch.no_grad():
                    z = torch.randn(int(n), int(self._net_spec_["latent_dim"]))
                    rows = generator(z).numpy()
            finally:
                torch.set_rng_state(gen_state)
        else:
            with torch.no_grad():
                z = torch.randn(int(n), int(self._net_spec_["latent_dim"]))
                rows = generator(z).numpy()
        return rows * self._std_ + self._mean_

    def quality_report(self, X_ref: Any | None = None) -> dict[str, Any]:
        """Honest statistical parity of 1000 samples vs a reference.

        Without ``X_ref`` the reference is the estimator's own fitted
        training statistics (``_mean_`` / ``_std_`` / the stored
        training correlation matrix).  ``mean_delta`` / ``std_delta``:
        per-feature ``|sampled - reference|`` differences;
        ``corr_l1``: the element-wise L1 distance between the sampled
        and reference correlation matrices.  A distributional sanity
        check — a parity REPORT, never a claim that sampled rows are
        real data.
        """
        self._check_fitted()
        if X_ref is None:
            ref_mean = self._mean_
            ref_std = self._std_
            ref_corr = self._corr_ref_
            n_ref = int(self._n_train_rows_)
        else:
            ref = self._check_X(X_ref)
            ref_mean = ref.mean(axis=0)
            ref_std = np.where(ref.std(axis=0) > 0, ref.std(axis=0), 1.0)
            ref_corr = (
                np.corrcoef(ref, rowvar=False)
                if int(ref.shape[1]) > 1
                else np.zeros((1, 1))
            )
            n_ref = int(ref.shape[0])
        draw = self.sample(1000, seed=int(self.random_state))
        mean_delta = np.abs(draw.mean(axis=0) - ref_mean)
        std_delta = np.abs(draw.std(axis=0) - ref_std)
        if int(self.n_features_in_) > 1:
            corr_l1 = float(
                np.mean(np.abs(np.corrcoef(draw, rowvar=False) - ref_corr))
            )
        else:
            corr_l1 = 0.0
        return {
            "n_ref": n_ref,
            "mean_delta": mean_delta.tolist(),
            "std_delta": std_delta.tolist(),
            "corr_l1": corr_l1,
        }

    # -- pickle support ----------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Drop the live torch modules from pickles (rebuilt on demand)."""
        state = self.__dict__.copy()
        state["_generator"] = None
        state["_discriminator"] = None
        return state

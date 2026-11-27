"""Tabular autoencoder (PyTorch) — part18 §5.5.

The Deep Learning architecture expansion's representation lane:
an encoder/decoder pair trained to RECONSTRUCT its own (standardised)
input through a low-dimensional bottleneck.  Four honest uses, all
wired:

* **latent representation learning** — :meth:`TabularAutoencoder.encode`
  maps any fitted-width matrix into the ``latent_dim`` space;
* **reconstruction** — :meth:`~pharmaml.ml.autoencoder.TabularAutoencoder.reconstruct`
  round-trips rows through the bottleneck;
* **dimensionality reduction** — a fitted 2-D/3-D latent is a
  nonlinear projection the workspace can scatter-plot next to the
  linear PCA / manifold family;
* **anomaly detection** — :meth:`~pharmaml.ml.autoencoder.TabularAutoencoder.reconstruction_errors`
  ranks rows by how badly the bottleneck reproduces them (an
  honest reconstruction-based RANKING — the error is a heuristic
  outlier score, never ground truth, and the docstrings say so).

The unsupervised sklearn contract (``fit(X, y=None)``), lazily-imported
torch with the actionable :class:`~pharmaml.exceptions.DependencyError`,
``fit(X, on_epoch=...)`` per-epoch telemetry
(``train_loss``/``val_loss``/``best_val``/``stale`` — early stopping
on a plain random split, no labels exist), seeded determinism (the
global torch RNG snapshot-and-restored), portable NumPy weight dicts,
and honest typed errors follow :mod:`pharmaml.ml.neural` verbatim.

Examples
--------
>>> from pharmaml.ml.autoencoder import TabularAutoencoder
>>> TabularAutoencoder().encode([[0.0, 1.0]])  # fit before encode
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: TabularAutoencoder has not been fitted yet — call fit(X) first

>>> TabularAutoencoder(latent_dim=0).fit([[0.0, 1.0], [1.0, 0.0]])
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: latent_dim must be a positive int, got 0

>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(60, 5))
>>> ae = TabularAutoencoder(hidden_layers=(16,), latent_dim=2, epochs=60,
...     random_state=0).fit(X)
>>> ae.encode(X).shape        # the latent representation
(60, 2)
>>> ae.reconstruct(X).shape   # the reconstruction
(60, 5)
>>> err = ae.reconstruction_errors(X)
>>> err.shape, bool(np.all(err >= 0))
((60,), True)

Reconstruction loss falls with training (the honest 2-D bottleneck
keeps some error — that IS the point):

>>> early = TabularAutoencoder(hidden_layers=(16,), latent_dim=2, epochs=3,
...     random_state=0).fit(X)
>>> float(ae.reconstruction_errors(X).mean()) <= float(early.reconstruction_errors(X).mean())
True

Anomaly ranking: an injected outlier row reconstructs worst:

>>> X2 = np.vstack([X, np.array([8.0, -8.0, 8.0, -8.0, 8.0])])
>>> ae2 = TabularAutoencoder(hidden_layers=(16,), latent_dim=2, epochs=60,
...     random_state=0).fit(X)
>>> ranking = np.argsort(ae2.reconstruction_errors(X2))[::-1]
>>> int(ranking[0])   # the injected row (index 60) tops the ranking
60

Same seed, identical latents:

>>> twin = TabularAutoencoder(hidden_layers=(16,), latent_dim=2, epochs=60,
...     random_state=0).fit(X)
>>> int(np.array_equal(ae.encode(X), twin.encode(X)))
1
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.logger import get_logger

__all__ = ["TabularAutoencoder"]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when torch is missing.
_TORCH_HINT = (
    "deep learning models require torch — from the toolkit package root run: "
    "python -m pip install -e '.[deep]'"
)

#: Legal ``device`` tokens ('auto' resolves to cuda when available).
_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


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


def _build_autoencoder(
    n_features: int,
    hidden: tuple[int, ...],
    latent_dim: int,
) -> Any:
    """The encoder/decoder MLP pair (a torch ``Module``).

    Encoder ``in -> hidden... -> latent``; decoder the mirrored widths.
    Defined lazily; forward takes its ``torch`` references through a
    local import (method bodies resolve only module globals).
    """
    from torch import nn

    class _Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            enc: list[Any] = []
            width = n_features
            for h in hidden:
                enc.append(nn.Linear(width, h))
                enc.append(nn.ReLU())
                width = h
            enc.append(nn.Linear(width, latent_dim))
            self.encoder = nn.Sequential(*enc)
            dec: list[Any] = []
            width = latent_dim
            for h in reversed(hidden):
                dec.append(nn.Linear(width, h))
                dec.append(nn.ReLU())
                width = h
            dec.append(nn.Linear(width, n_features))
            self.decoder = nn.Sequential(*dec)

        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    return _Autoencoder()


class TabularAutoencoder(BaseEstimator):
    """Tabular autoencoder (PyTorch) — representation/reconstruction lane.

    Trains an encoder/decoder pair to reconstruct standardised rows
    through a ``latent_dim`` bottleneck.  ``fit`` is unsupervised
    (``y=None``); the validation split is a plain random split of the
    rows (no labels exist) and early stopping tracks the validation
    reconstruction loss.

    Parameters
    ----------
    hidden_layers:
        Widths of the encoder's hidden stack (the decoder mirrors
        them).
    latent_dim:
        Bottleneck width — 2 or 3 gives a scatter-plotable nonlinear
        projection (the dimensionality-reduction use).
    epochs, batch_size, learning_rate, weight_decay, patience, val_split,
    random_state, device:
        The same contract as :class:`~pharmaml.ml.neural.DeepMLPRegressor`.

    Attributes
    ----------
    n_features_in_:
        Number of input features seen at fit.
    history_:
        Per-epoch dicts ``{'epoch', 'train_loss', 'val_loss'}``.
    n_iter_:
        Epochs actually run (early stopping included).

    Notes
    -----
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.  ``reconstruction_errors`` is a heuristic
    outlier RANKING, not ground truth (the anomaly lane of
    :mod:`pharmaml.ml.anomaly` owns the dedicated detectors).
    """

    def __init__(
        self,
        hidden_layers=(64,),
        latent_dim=8,
        epochs=120,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=15,
        val_split=0.15,
        random_state=42,
        device="auto",
    ) -> None:
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
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
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ModelError(f"{name} must be an integer >= 1, got {value!r}")
        if not np.isfinite(self.learning_rate) or float(self.learning_rate) <= 0:
            raise ModelError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not np.isfinite(self.weight_decay) or float(self.weight_decay) < 0:
            raise ModelError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if not np.isfinite(self.val_split) or not 0.0 <= float(self.val_split) < 1.0:
            raise ModelError(f"val_split must be in [0, 1), got {self.val_split!r}")
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

    def _check_feature_names(self, X: Any) -> None:
        names = getattr(self, "feature_names_in_", None)
        if names is None or not isinstance(X, pd.DataFrame):
            return
        columns = np.asarray(X.columns, dtype=object)
        if not np.array_equal(columns, names):
            raise DataValidationError(
                f"X has feature names {columns.tolist()} that do not match the "
                f"feature names seen during fit: {names.tolist()} — pass the same "
                "columns in the fit order, or a plain (positional) array"
            )

    # -- fit ------------------------------------------------------------
    def fit(
        self, X: Any, y: Any = None, *, on_epoch: Callable[[dict[str, Any]], None] | None = None
    ) -> TabularAutoencoder:
        """Train the autoencoder on ``X`` (unsupervised; ``y`` ignored).

        The telemetry contract matches :mod:`pharmaml.ml.neural`:
        ``on_epoch`` receives ``{'epoch', 'epochs', 'train_loss',
        'val_loss', 'best_val', 'stale'}`` after every epoch (the
        losses are reconstruction MSE on the standardised scale) and
        must never raise.
        """
        hidden = self._validate_hyperparameters()
        X_arr = self._check_X(X)

        _require_torch()  # before any fitted attribute is written
        device = self._resolve_device()

        self.n_features_in_ = int(X_arr.shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        seed = int(self.random_state)
        rng = np.random.default_rng(seed)
        n = int(X_arr.shape[0])

        val_split = float(self.val_split)
        if val_split <= 0 or n < 2:
            train_idx, val_idx = np.arange(n), None
        else:
            perm = rng.permutation(n)
            val_idx = perm[: max(1, round(val_split * n))]
            if val_idx.size == 0 or val_idx.size >= n:
                train_idx, val_idx = np.arange(n), None
            else:
                mask = np.zeros(n, dtype=bool)
                mask[val_idx] = True
                train_idx, val_idx = np.flatnonzero(~mask), np.flatnonzero(mask)

        mean = X_arr[train_idx].mean(axis=0)
        std = X_arr[train_idx].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self._mean_ = mean
        self._std_ = std
        X_scaled = (X_arr - mean) / std

        import torch

        rng_state = torch.get_rng_state()
        try:
            self._train_torch(
                X_scaled, device, hidden, train_idx, val_idx, rng, on_epoch=on_epoch
            )
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
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
        rng: np.random.Generator,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Minibatch Adam reconstruction training with early stopping."""
        import torch
        from torch import nn

        torch.manual_seed(int(self.random_state))

        self._net_spec_: dict[str, Any] = {
            "n_features": int(X_scaled.shape[1]),
            "hidden": hidden,
            "latent_dim": int(self.latent_dim),
        }
        network = _build_autoencoder(
            self._net_spec_["n_features"],
            self._net_spec_["hidden"],
            self._net_spec_["latent_dim"],
        )
        network.to(device)

        X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
        criterion: Any = nn.MSELoss()
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        X_tr = X_t[train_t]
        X_val = None
        if val_idx is not None:
            val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
            X_val = X_t[val_t]

        epochs = int(self.epochs)
        batch_size = int(self.batch_size)
        patience = int(self.patience)

        history: list[dict[str, Any]] = []
        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0

        for epoch in range(1, epochs + 1):
            network.train()
            perm = torch.as_tensor(rng.permutation(train_idx.size), device=device)
            batch_losses: list[float] = []
            for start in range(0, perm.numel(), batch_size):
                batch = perm[start : start + batch_size]
                optimizer.zero_grad()
                rows = X_tr[batch]
                loss = criterion(network(rows), rows)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

            val_loss: float | None = None
            if X_val is not None:
                network.eval()
                with torch.no_grad():
                    val_loss = float(criterion(network(X_val), X_val).item())
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in network.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if on_epoch is not None:
                on_epoch({
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val": best_val if best_val != float("inf") else None,
                    "stale": stale,
                })
            if val_loss is not None and stale >= patience:
                _LOG.debug(
                    "early stopping at epoch %03d (no val improvement for %d epochs)",
                    epoch,
                    patience,
                )
                break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()

        weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in network.state_dict().items()
        }
        if not all(np.isfinite(value).all() for value in weights.values()):
            raise ModelError(
                "training diverged — the fitted weights contain NaN/inf "
                "values; lower learning_rate and refit"
            )
        self.history_ = history
        self.n_iter_ = len(history)
        self._weights = weights
        self._network = network

        n_params = sum(int(p.numel()) for p in network.parameters())
        _LOG.info(
            "%s fitted on %d samples (%d -> %d latent): %d epochs run, "
            "%d parameters, device=%s",
            type(self).__name__,
            int(X_scaled.shape[0]),
            int(X_scaled.shape[1]),
            int(self.latent_dim),
            self.n_iter_,
            n_params,
            device,
        )

    # -- transform methods -------------------------------------------------
    def _ensure_network(self) -> Any:
        """Rebuild the live torch network from the portable weights."""
        network = getattr(self, "_network", None)
        if network is None:
            _require_torch()
            import torch

            spec = self._net_spec_
            network = _build_autoencoder(
                spec["n_features"], spec["hidden"], spec["latent_dim"]
            )
            network.load_state_dict(
                {key: torch.from_numpy(value) for key, value in self._weights.items()}
            )
            self._network = network
        return network

    def _forward(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Standardise ``X``; return ``(latent, reconstruction)`` in eval mode."""
        self._check_fitted()
        self._check_feature_names(X)
        X_arr = self._check_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise DataValidationError(
                f"X has {X_arr.shape[1]} feature column(s), but "
                f"{type(self).__name__} was fitted with "
                f"{self.n_features_in_} — pass the same features as to fit()"
            )
        _require_torch()
        import torch

        network = self._ensure_network()
        network.eval()
        X_scaled = (X_arr - self._mean_) / self._std_
        with torch.no_grad():
            X_t = torch.as_tensor(X_scaled, dtype=torch.float32)
            latent = network.encoder(X_t).numpy()
            recon = network.decoder(torch.as_tensor(latent, dtype=torch.float32)).numpy()
        return latent, recon

    def encode(self, X: Any) -> np.ndarray:
        """Map ``X`` into the latent space (shape ``(n, latent_dim)``)."""
        latent, _ = self._forward(X)
        return latent

    def reconstruct(self, X: Any) -> np.ndarray:
        """Round-trip ``X`` through the bottleneck (original scale)."""
        _, recon = self._forward(X)
        return recon * self._std_ + self._mean_

    def reconstruction_errors(self, X: Any) -> np.ndarray:
        """Per-row reconstruction MSE on the ORIGINAL feature scale.

        An honest heuristic outlier RANKING (high error = the
        bottleneck could not reproduce the row) — not ground truth;
        the dedicated anomaly detectors live in
        :mod:`pharmaml.ml.anomaly`.
        """
        X_arr = self._check_X(X)
        recon = self.reconstruct(X_arr)
        return np.mean((X_arr - recon) ** 2, axis=1)

    # -- pickle support ----------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Drop the live torch network from pickles (rebuilt on demand)."""
        state = self.__dict__.copy()
        state["_network"] = None
        return state

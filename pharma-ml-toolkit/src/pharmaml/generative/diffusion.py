"""Feature-space diffusion model (PyTorch DDPM) — part18 §5.6.

The Deep Learning architecture expansion's diffusion lane, in the
Generative AI family where the spec puts it (never the supervised
model registry).  The honest scope, stated once and everywhere it
matters:

* this is a **feature-space generative model**: the denoiser learns
  the distribution of a *numeric feature matrix* (the descriptor rows
  of a registered dataset) and the reverse process samples SYNTHETIC
  FEATURE ROWS from pure Gaussian noise — statistical parity against
  the training distribution is the quality bar (per-feature mean/std
  deltas, correlation-matrix distance), reported honestly by
  :meth:`FeatureSpaceDiffusion.quality_report`.
* it is **NOT molecular generation**: no SMILES, no sanitisation, no
  drug-likeness claim.  Molecules come from the SMILES lanes of this
  package (the RNN prior, the genetic optimiser, MCTS) and nothing
  here supersedes them.  Nothing generated is experimentally
  validated data — synthetic rows are synthetic rows.

The machinery is the textbook DDPM (Ho et al., 2020) at tabular
scale: a fixed linear ``beta`` schedule, the forward process
``q(x_t | x_0) = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps``
sampled per training row with a per-row diffusion step ``t``, an
eps-predictor MLP over ``[x_t ; time-embedding(t)]`` trained with
plain MSE (noise-prediction parameterisation), and the reverse
process iterating the DDPM posterior mean
``mu_t = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta) /
sqrt(alpha_t)`` from ``x_T ~ N(0, I)`` down to ``x_0``.

Determinism and portability follow :mod:`pharmaml.ml.neural` verbatim
(seeded torch + NumPy, the global torch RNG snapshot-and-restored
around fit and around every seeded sample draw, portable NumPy weight
dicts, honest typed errors); the ``on_epoch`` telemetry carries the
noise-prediction MSE per epoch.

Examples
--------
>>> from pharmaml.generative.diffusion import FeatureSpaceDiffusion
>>> FeatureSpaceDiffusion().sample(2)   # fit before sample
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: FeatureSpaceDiffusion has not been fitted yet — call fit(X) first

>>> FeatureSpaceDiffusion(timesteps=1).fit([[0.0, 1.0]] * 4)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: timesteps must be >= 2, got 1

>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(loc=1.5, scale=0.3, size=(200, 3))
>>> dpm = FeatureSpaceDiffusion(timesteps=50, hidden_layers=(32,),
...     epochs=60, random_state=0).fit(X)
>>> dpm.sample(4).shape      # synthetic rows on the original scale
(4, 3)
>>> dpm.sample(4, seed=9).shape   # a seeded draw
(4, 3)

Statistical parity — sampled means land near the training means
(a generous tolerance: a small CPU DDPM approximates, it does not
solve the SDE exactly):

>>> report = dpm.quality_report()
>>> round(float(np.max(np.abs(report['mean_delta']))), 1) <= 0.9
True
>>> sorted(report.keys())
['corr_l1', 'mean_delta', 'n_ref', 'std_delta']

Same seed, identical sample stream:

>>> twin = dpm.sample(5, seed=3)
>>> int(np.array_equal(dpm.sample(5, seed=3), twin))
1

The training telemetry contract (a live epoch stream, never
fabricated):

>>> seen = []
>>> _d = FeatureSpaceDiffusion(timesteps=8, hidden_layers=(8,), epochs=3,
...     random_state=0).fit(X[:24], on_epoch=seen.append)
>>> [set(d) == {'epoch', 'epochs', 'train_loss'} for d in seen]
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

__all__ = ["FeatureSpaceDiffusion"]

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

#: The linear beta schedule's endpoints (the DDPM paper's values).
_BETA_START = 1e-4
_BETA_END = 0.02

#: Width of the sinusoidal time embedding fed to the denoiser.
_TIME_EMBED_DIM = 32


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


def _linear_schedule(timesteps: int) -> tuple[np.ndarray, ...]:
    """The DDPM linear beta/alpha/alpha_bar schedule (pure NumPy).

    Returns ``(beta, alpha, alpha_bar)`` with ``alpha_bar`` the
    exclusive cumulative product — ``alpha_bar[t] = prod(alpha[:t])``.
    """
    beta = np.linspace(_BETA_START, _BETA_END, int(timesteps))
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    return beta, alpha, alpha_bar


def _time_embedding(t: np.ndarray) -> np.ndarray:
    """The sinusoidal diffusion-step embedding (transformer-style).

    ``t`` is an int array of per-row diffusion steps; the embedding
    alternates sin/cos of ``t / 10000 ** (2i / d)`` across ``d``
    dimensions — the standard positional encoding applied to the
    scalar step index.
    """
    half = _TIME_EMBED_DIM // 2
    freqs = np.exp(
        -np.log(10000.0) * np.arange(half, dtype=np.float64) / max(1, half)
    )
    angles = np.asarray(t, dtype=np.float64)[:, None] * freqs[None, :]
    return np.concatenate(
        [np.sin(angles), np.cos(angles)], axis=1
    )[:, :_TIME_EMBED_DIM]


def _build_denoiser(n_features: int, hidden: tuple[int, ...]) -> Any:
    """The eps-predictor MLP over ``[x_t ; time_embedding]``.

    Defined lazily; the forward method takes its ``torch`` references
    through a local import (method bodies resolve only module globals).
    """
    from torch import nn

    in_dim = n_features + _TIME_EMBED_DIM

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            width = in_dim
            for h in hidden:
                layers.append(nn.Linear(width, h))
                layers.append(nn.SiLU())
                width = h
            layers.append(nn.Linear(width, n_features))
            self.net = nn.Sequential(*layers)

        def forward(self, x_t: Any, t_emb: Any) -> Any:
            import torch  # local: method bodies see module globals only

            return self.net(torch.cat([x_t, t_emb], dim=1))

    return _Denoiser()


class FeatureSpaceDiffusion(BaseEstimator):
    """Feature-space denoising diffusion model (PyTorch DDPM).

    Trains an eps-predictor on the numeric feature matrix ``X``
    (per-feature standardised at fit; ``sample`` returns ORIGINAL-scale
    rows): each training row is noised to a random diffusion step
    ``t`` via the forward process and the denoiser regresses the
    injected noise (plain MSE — the DDPM objective).  ``sample`` runs
    the full reverse chain from ``x_T ~ N(0, I)``.

    Parameters
    ----------
    timesteps:
        Number of diffusion steps in the schedule (>= 2).
    hidden_layers:
        Hidden widths of the eps-predictor.
    epochs, batch_size, learning_rate, weight_decay, random_state, device:
        The usual :mod:`pharmaml.ml.neural`-style contract.

    Attributes
    ----------
    n_features_in_:
        Feature width seen at fit.
    history_:
        Per-epoch dicts ``{'epoch', 'epochs', 'train_loss'}`` — the
        noise-prediction MSE (a diffusion model has no validation
        loss; every epoch is honest telemetry).

    Notes
    -----
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.  FEATURE-SPACE generative model: synthetic
    rows, not molecules (the SMILES lanes own molecular generation);
    nothing sampled is experimentally validated data.
    """

    def __init__(
        self,
        timesteps=100,
        hidden_layers=(64,),
        epochs=120,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        random_state=_RANDOM_STATE,
        device="auto",
    ) -> None:
        self.timesteps = timesteps
        self.hidden_layers = hidden_layers
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
        if isinstance(self.timesteps, bool) or not isinstance(
            self.timesteps, (int, np.integer)
        ) or self.timesteps < 2:
            raise ModelError(f"timesteps must be >= 2, got {self.timesteps!r}")
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
    ) -> FeatureSpaceDiffusion:
        """Train the eps-predictor on ``X`` (unsupervised; ``y`` ignored).

        ``on_epoch`` receives ``{'epoch', 'epochs', 'train_loss'}``
        after every epoch (noise-prediction MSE on the standardised
        scale) and must never raise.
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
        # the training correlation matrix, kept for quality_report's
        # parity check — the honest quality bar
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
        """The DDPM noise-prediction loop (minibatch Adam, plain MSE)."""
        import torch
        from torch import nn

        torch.manual_seed(int(self.random_state))
        rng = np.random.default_rng(int(self.random_state))

        timesteps = int(self.timesteps)
        beta, alpha, alpha_bar = _linear_schedule(timesteps)
        self._schedule_: dict[str, Any] = {
            "timesteps": timesteps,
            "beta": beta.tolist(),
            "alpha": alpha.tolist(),
            "alpha_bar": alpha_bar.tolist(),
        }

        self._net_spec_: dict[str, Any] = {
            "n_features": int(X_scaled.shape[1]),
            "hidden": hidden,
        }
        denoiser = _build_denoiser(
            self._net_spec_["n_features"], self._net_spec_["hidden"]
        )
        denoiser.to(device)

        criterion: Any = nn.MSELoss()
        optimizer = torch.optim.Adam(
            denoiser.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
        n = int(X_scaled.shape[0])
        epochs = int(self.epochs)
        batch_size = min(int(self.batch_size), n)

        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            denoiser.train()
            perm = rng.permutation(n)
            batch_losses: list[float] = []
            for start in range(0, n, batch_size):
                idx = torch.as_tensor(
                    perm[start : start + batch_size], dtype=torch.long, device=device
                )
                members = int(idx.numel())
                if members < 1:  # pragma: no cover - empty tail impossible
                    continue
                # the forward process: a per-row diffusion step + noise
                steps = rng.integers(1, timesteps + 1, size=members)
                noise = rng.normal(size=(members, X_scaled.shape[1]))
                t_e = torch.as_tensor(
                    _time_embedding(steps - 1), dtype=torch.float32, device=device
                )
                ab = alpha_bar[steps - 1][:, None]
                x0 = X_t[idx]
                x_noisy = (
                    np.sqrt(ab) * x0.detach().cpu().numpy()
                    + np.sqrt(1.0 - ab) * noise
                )
                x_in = torch.as_tensor(x_noisy, dtype=torch.float32, device=device)
                eps_true = torch.as_tensor(noise, dtype=torch.float32, device=device)
                optimizer.zero_grad()
                pred = denoiser(x_in, t_e)
                loss = criterion(pred, eps_true)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            rec = {"epoch": epoch, "epochs": epochs, "train_loss": train_loss}
            history.append(rec)
            if on_epoch is not None:
                on_epoch(rec)

        denoiser.eval()
        weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in denoiser.state_dict().items()
        }
        if not all(np.isfinite(value).all() for value in weights.values()):
            raise ModelError(
                "training diverged — the fitted weights contain NaN/inf "
                "values; lower learning_rate and refit"
            )
        self.history_ = history
        self._weights = weights
        self._denoiser = denoiser

        n_params = sum(int(p.numel()) for p in denoiser.parameters())
        _LOG.info(
            "%s fitted on %d rows x %d features (%d timesteps): %d epochs, "
            "final noise-MSE %.4f, %d parameters, device=%s",
            type(self).__name__,
            n,
            int(X_scaled.shape[1]),
            timesteps,
            len(history),
            history[-1]["train_loss"] if history else float("nan"),
            n_params,
            device,
        )

    # -- sampling -----------------------------------------------------------
    def _ensure_network(self) -> Any:
        """Rebuild the live denoiser from the portable weights."""
        denoiser = getattr(self, "_denoiser", None)
        if denoiser is None:
            _require_torch()
            import torch

            spec = self._net_spec_
            denoiser = _build_denoiser(spec["n_features"], spec["hidden"])
            denoiser.load_state_dict(
                {key: torch.from_numpy(value) for key, value in self._weights.items()}
            )
            self._denoiser = denoiser
        return denoiser

    def sample(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """Run the reverse chain: ``n`` synthetic rows, original scale.

        ``seed`` makes the draw reproducible (the same fitted model +
        the same seed → the same rows); without it the draw rides the
        process RNG.
        """
        self._check_fitted()
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ModelError(f"n must be a positive int, got {n!r}")
        _require_torch()
        import torch

        denoiser = self._ensure_network()
        denoiser.eval()
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
        ):
            raise ModelError(f"seed must be an integer, got {seed!r}")

        gen_state = torch.get_rng_state()
        try:
            if seed is not None:
                torch.manual_seed(int(seed))
            rng = np.random.default_rng(int(seed) if seed is not None else int(self.random_state) + 1)
            beta = np.asarray(self._schedule_["beta"])
            alpha = np.asarray(self._schedule_["alpha"])
            alpha_bar = np.asarray(self._schedule_["alpha_bar"])
            timesteps = int(self._schedule_["timesteps"])

            x = rng.normal(size=(int(n), self.n_features_in_))  # x_T
            with torch.no_grad():
                for step in range(timesteps, 0, -1):
                    t_idx = step - 1
                    t_e = torch.as_tensor(
                        _time_embedding(np.full(int(n), t_idx)),
                        dtype=torch.float32,
                    )
                    x_t = torch.as_tensor(x, dtype=torch.float32)
                    eps = denoiser(x_t, t_e).numpy()
                    coef = beta[t_idx] / np.sqrt(1.0 - alpha_bar[t_idx])
                    mean = (x - coef * eps) / np.sqrt(alpha[t_idx])
                    if step > 1:
                        var = beta[t_idx]
                        x = mean + rng.normal(
                            scale=np.sqrt(var), size=x.shape
                        )
                    else:
                        x = mean
            return x * self._std_ + self._mean_
        finally:
            torch.set_rng_state(gen_state)

    def quality_report(self, X_ref: Any | None = None) -> dict[str, Any]:
        """Honest statistical parity of 1000 samples vs a reference.

        Without ``X_ref`` the reference is the estimator's own fitted
        training statistics.  ``mean_delta`` / ``std_delta``:
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
        """Drop the live torch denoiser from pickles (rebuilt on demand)."""
        state = self.__dict__.copy()
        state["_denoiser"] = None
        return state

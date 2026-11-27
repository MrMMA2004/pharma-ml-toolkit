"""Sequence CNN estimators (PyTorch) — part18 §5.2.

The Deep Learning architecture expansion's convolutional lane, on
the only input representation where a convolution is *scientifically
meaningful* for this toolkit (the spec's own rule: never reshape
arbitrary tabular descriptors into artificial images):

* **3-D sequence inputs** ``(n_samples, seq_len, n_features)`` — the
  ``sliding_windows`` layout of :mod:`pharmaml.ml.sequences` and the
  time-step × channel layout of the timeseries lane.  A plain 2-D
  tabular matrix is REJECTED with a typed error that points at the
  DeepMLP / Tabular Transformer families — the same honesty the spec
  demands, on the receiving end.
* **molecular sequence representations** (SMILES token sequences) can
  be embedded as per-token feature vectors and trained through this
  family; graph-structured molecules already have their own lane
  (:mod:`pharmaml.ml.gnn` — the graph convolution IS a convolution).

The network is a 1-D temporal conv stack (``Conv1d(kernel) ->
BatchNorm1d -> ReLU`` per layer over the time axis, channels carried
across), a concatenated average+maximum pool over time (the average
preserves magnitude, the max answers "did the kernel fire anywhere"),
and a linear head — the classic locality detector: a kernel of 3+
spans the motif, the pooling makes the response
translation-invariant.

Everything else is the :mod:`pharmaml.ml.neural` contract verbatim:
sklearn estimator surface, lazily-imported torch (the actionable
:class:`~pharmaml.exceptions.DependencyError`), ``fit(X, y,
on_epoch=...)`` per-epoch telemetry, early stopping on a stratified
validation split, seeded determinism (the global torch RNG
snapshot-and-restored), portable NumPy weight dicts, and honest
typed errors for the degenerate cases (mixed sequence lengths,
non-3-D input, NaN features, width mismatch at predict time).

:func:`synthetic_motif_sequences` is the lane's deterministic fixture
(the GNN lane's ``synthetic_motif_graphs`` analogue): class-1 windows
carry a LOCAL amplitude bump on a contiguous block of steps — a
pattern only a local receptive field can find cheaply, which is
exactly what a conv kernel detects.

Examples
--------
>>> from pharmaml.ml.cnn import SequenceCNNClassifier, synthetic_motif_sequences
>>> SequenceCNNClassifier().predict([[[0.0, 1.0]]])  # fit before predict
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: SequenceCNNClassifier has not been fitted yet — call fit(X, y) first

A tabular matrix is honestly refused, with the pointer to the
families that fit it:

>>> SequenceCNNClassifier().fit([[1.0, 2.0], [3.0, 4.0]], [0, 1])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: X must be 3-dimensional (n_samples, seq_len, n_features) — a 2-D tabular matrix belongs to the DeepMLP / Tabular Transformer families (pharmaml.ml.neural / pharmaml.ml.transformer)

>>> x, y = synthetic_motif_sequences(40, seed=0)
>>> x.shape, sorted(set(y.tolist()))
((40, 16, 3), [0, 1])
>>> x2, y2 = synthetic_motif_sequences(40, seed=0)
>>> int(np.array_equal(x, x2) * np.array_equal(y, y2))   # seeded fixture
1

The classifier learns the local motif (kernel spans it):

>>> import numpy as np
>>> net = SequenceCNNClassifier(channels=8, kernel=3, epochs=25,
...     learning_rate=3e-3, random_state=0).fit(x, y)
>>> float((net.predict(x) == y).mean()) >= 0.9
True
>>> twin = SequenceCNNClassifier(channels=8, kernel=3, epochs=25,
...     learning_rate=3e-3, random_state=0).fit(x, y)
>>> int(np.array_equal(net.predict(x), twin.predict(x)))   # same seed
1

Mixed / ragged sequence lengths are an honest typed error (NumPy
refuses the inhomogeneous batch before the estimator sees it):

>>> from pharmaml.exceptions import DataValidationError
>>> try:
...     SequenceCNNClassifier().fit([[[0.0]] * 4, [[0.0]] * 5], [0, 1])
... except DataValidationError as exc:
...     str(exc).startswith("sequences must be numeric")
True
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.logger import get_logger

__all__ = [
    "SequenceCNNClassifier",
    "SequenceCNNRegressor",
    "synthetic_motif_sequences",
]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when torch is missing.
_TORCH_HINT = (
    "deep learning models require torch — from the toolkit package root run: "
    "python -m pip install -e '.[deep]'"
)

#: Legal ``device`` tokens ('auto' resolves to cuda when available).
_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

#: The fixture's motif width (steps of the amplitude bump) and height.
_MOTIF_WIDTH = 4
_MOTIF_AMPLITUDE = 3.0

#: The fixture's default generator (the toolkit's module-wide seed).
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


def _build_cnn(
    n_features: int,
    n_out: int,
    channels: int,
    kernel: int,
    n_layers: int,
) -> Any:
    """The 1-D temporal conv stack + adaptive pool + linear head.

    Defined lazily (never at module import); the forward method takes
    its ``torch`` references through a local import (method bodies
    resolve only module globals, never the builder's locals).
    """
    from torch import nn

    class _SequenceCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_ch = n_features
            for _ in range(n_layers):
                layers.append(
                    nn.Conv1d(in_ch, channels, kernel_size=kernel, padding="same")
                )
                layers.append(nn.BatchNorm1d(channels))
                layers.append(nn.ReLU())
                in_ch = channels
            self.conv = nn.Sequential(*layers)
            # both time-pools, concatenated: the average preserves
            # magnitude (smooth/regression signals), the max answers
            # "did the kernel fire anywhere" (local motifs) — the head
            # reads whichever the data supports.
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Linear(2 * channels, n_out)

        def forward(self, x: Any) -> Any:
            import torch  # local: method bodies see module globals only

            # x arrives as (n, seq_len, features); Conv1d wants the
            # channel axis second — transpose, convolve over time, pool.
            out = self.conv(x.transpose(1, 2))
            pooled = torch.cat(
                [self.avg_pool(out), self.max_pool(out)], dim=1
            ).squeeze(-1)
            return self.head(pooled)

    return _SequenceCNN()


def synthetic_motif_sequences(
    n: int,
    *,
    length: int = 16,
    features: int = 3,
    seed: int = _RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic binary motif/sequence fixture.

    Class-1 windows carry a LOCAL amplitude bump
    (``+_MOTIF_AMPLITUDE`` on feature 0) on one contiguous block of
    ``_MOTIF_WIDTH`` steps at a random position; class-0 windows are
    standard normal noise throughout.  Labels are balanced (even
    ``n``) or as close as the integers allow.  The bump is local — a
    receptive field of ``kernel >= _MOTIF_WIDTH - 1`` steps detects it
    while per-feature pooling alone cannot tell where it sits.

    Returns
    -------
    (x, y) : tuple of numpy.ndarray
        ``x`` has shape ``(n, length, features)``, ``y`` shape
        ``(n,)`` with labels in {0, 1}.

    Raises
    ------
    DataValidationError
        When ``n < 2``, ``length < _MOTIF_WIDTH + 2`` or
        ``features < 1``.
    """
    if n < 2:
        raise DataValidationError(f"n must be >= 2, got {n!r}")
    if length < _MOTIF_WIDTH + 2:
        raise DataValidationError(
            f"length must be >= {_MOTIF_WIDTH + 2} for the motif to sit inside "
            f"the window, got {length!r}"
        )
    if features < 1:
        raise DataValidationError(f"features must be >= 1, got {features!r}")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(int(n), int(length), int(features)))
    y = np.zeros(int(n), dtype=int)
    y[::2] = 1  # balanced labels
    for i in np.flatnonzero(y == 1):
        start = int(rng.integers(0, length - _MOTIF_WIDTH + 1))
        x[i, start : start + _MOTIF_WIDTH, 0] += _MOTIF_AMPLITUDE
    return x, y


class _BaseSequenceCNN(BaseEstimator):
    """Private fit/predict machinery shared by the two public estimators."""

    _is_classifier: bool = False

    def _validate_hyperparameters(self) -> None:
        if isinstance(self.channels, bool) or not isinstance(
            self.channels, (int, np.integer)
        ) or self.channels < 1:
            raise ModelError(f"channels must be a positive int, got {self.channels!r}")
        if isinstance(self.kernel, bool) or not isinstance(
            self.kernel, (int, np.integer)
        ) or self.kernel < 1 or self.kernel % 2 == 0:
            raise ModelError(
                f"kernel must be a positive odd int (same padding keeps the "
                f"sequence length), got {self.kernel!r}"
            )
        if isinstance(self.n_layers, bool) or not isinstance(
            self.n_layers, (int, np.integer)
        ) or self.n_layers < 1:
            raise ModelError(f"n_layers must be a positive int, got {self.n_layers!r}")
        if not isinstance(self.batchnorm, bool):
            raise ModelError(f"batchnorm must be a bool, got {self.batchnorm!r}")
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
        if self.n_layers > 1 and int(self.channels) < 2 and self.batchnorm:
            raise ModelError("batchnorm=True needs channels >= 2 and batch_size >= 2")

    def _check_X(self, X: Any) -> np.ndarray:
        """Validate ``X`` as a finite float64 3-D sequence batch."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"sequences must be numeric — conversion failed: {exc}"
            ) from exc
        if X_arr.ndim != 3:
            hint = (
                "X must be 3-dimensional (n_samples, seq_len, n_features) — a "
                "2-D tabular matrix belongs to the DeepMLP / Tabular Transformer "
                "families (pharmaml.ml.neural / pharmaml.ml.transformer)"
                if X_arr.ndim == 2
                else f"X must be 3-dimensional (n_samples, seq_len, n_features), "
                f"got {X_arr.ndim} dimension(s)"
            )
            raise DataValidationError(hint)
        if X_arr.shape[0] == 0:
            raise DataValidationError(
                f"X has 0 samples — {type(self).__name__} needs at least one sequence"
            )
        if X_arr.shape[1] == 0:
            raise DataValidationError(
                f"X has 0 time steps — {type(self).__name__} needs a non-empty "
                "seq_len axis"
            )
        if X_arr.shape[2] == 0:
            raise DataValidationError(
                f"X has 0 channels — {type(self).__name__} needs at least one "
                "feature channel"
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
                f"{type(self).__name__} has not been fitted yet — call fit(X, y) "
                "first"
            )

    # -- fit ------------------------------------------------------------
    def fit(
        self, X: Any, y: Any, *, on_epoch: Callable[[dict[str, Any]], None] | None = None
    ) -> _BaseSequenceCNN:
        """Train the temporal CNN on 3-D sequences ``(X, y)``.

        Every sample must share one ``seq_len`` and one channel width;
        the telemetry contract matches :mod:`pharmaml.ml.neural`
        (``on_epoch`` gets ``{'epoch', 'epochs', 'train_loss',
        'val_loss', 'best_val', 'stale'}`` and must never raise).
        """
        self._validate_hyperparameters()
        X_arr = self._check_X(X)

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise DataValidationError(
                f"y must be 1-dimensional, got {y_arr.ndim} dimension(s)"
            )
        if y_arr.size == 0:
            raise DataValidationError(
                f"y is empty — {type(self).__name__} needs at least one target value"
            )
        if y_arr.shape[0] != X_arr.shape[0]:
            raise DataValidationError(
                f"X and y have inconsistent lengths: {X_arr.shape[0]} samples vs "
                f"{y_arr.shape[0]} targets"
            )

        if self._is_classifier:
            try:
                classes, y_enc = np.unique(y_arr, return_inverse=True)
            except TypeError as exc:
                raise DataValidationError(
                    "classification targets could not be label-encoded — they "
                    f"contain unorderable values (e.g. None): {exc}"
                ) from exc
            if classes.size < 2:
                raise DataValidationError(
                    "classification needs at least 2 distinct classes, got "
                    f"{classes.size}: {classes.tolist()[:5]}"
                )
            n_out = int(classes.size)
        else:
            try:
                y_enc = np.asarray(y_arr, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"regression targets must be numeric — conversion failed: {exc}"
                ) from exc
            if not np.isfinite(y_enc).all():
                raise DataValidationError(
                    "regression targets contain NaN or infinite values — drop or "
                    "impute those rows first"
                )
            n_out = 1

        _require_torch()  # before any fitted attribute is written
        device = self._resolve_device()

        if self._is_classifier:
            self.classes_ = classes
        self.n_features_in_ = int(X_arr.shape[2])
        self.seq_len_in_ = int(X_arr.shape[1])

        seed = int(self.random_state)
        rng = np.random.default_rng(seed)
        n = int(X_arr.shape[0])
        train_idx, val_idx = self._train_val_split(y_enc, n, rng)

        # per-channel standardisation, fitted on the training rows only
        mean = X_arr[train_idx].reshape(-1, X_arr.shape[2]).mean(axis=0)
        std = X_arr[train_idx].reshape(-1, X_arr.shape[2]).std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self._mean_ = mean
        self._std_ = std
        X_scaled = (X_arr - mean) / std

        if self._is_classifier:
            self._y_mean_: float | None = None
            self._y_std_: float | None = None
        else:
            y_train = y_enc[train_idx]
            self._y_mean_ = float(y_train.mean())
            y_std = float(y_train.std())
            self._y_std_ = y_std if y_std > 0 else 1.0
            y_enc = (y_enc - self._y_mean_) / self._y_std_

        import torch

        rng_state = torch.get_rng_state()
        try:
            self._train_torch(
                X_scaled, y_enc, n_out, device, train_idx, val_idx, rng,
                on_epoch=on_epoch,
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

    def _train_val_split(
        self, y_enc: np.ndarray, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Stratified (classifier) / random (regressor) validation split."""
        val_split = float(self.val_split)
        if val_split <= 0 or n < 2:
            return np.arange(n), None
        if self._is_classifier:
            parts: list[np.ndarray] = []
            for cls in np.unique(y_enc):
                members = np.flatnonzero(y_enc == cls)
                members = members[rng.permutation(members.size)]
                k = round(val_split * members.size)
                k = min(k, max(members.size - 1, 0))
                parts.append(members[:k])
            val_idx = (
                np.concatenate(parts).astype(int) if parts else np.array([], dtype=int)
            )
        else:
            perm = rng.permutation(n)
            val_idx = perm[: max(1, round(val_split * n))]
        if val_idx.size == 0 or val_idx.size >= n:
            return np.arange(n), None
        mask = np.zeros(n, dtype=bool)
        mask[val_idx] = True
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    def _train_torch(
        self,
        X_scaled: np.ndarray,
        y_enc: np.ndarray,
        n_out: int,
        device: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
        rng: np.random.Generator,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Minibatch Adam training with early stopping (the neural.py loop)."""
        import torch
        from torch import nn

        torch.manual_seed(int(self.random_state))

        self._net_spec_: dict[str, Any] = {
            "n_features": int(X_scaled.shape[2]),
            "n_out": int(n_out),
            "channels": int(self.channels),
            "kernel": int(self.kernel),
            "n_layers": int(self.n_layers),
        }
        network = _build_cnn(
            self._net_spec_["n_features"],
            self._net_spec_["n_out"],
            self._net_spec_["channels"],
            self._net_spec_["kernel"],
            self._net_spec_["n_layers"],
        )
        network.to(device)

        X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
        if self._is_classifier:
            y_t = torch.as_tensor(y_enc, dtype=torch.long, device=device)
            criterion: Any = nn.CrossEntropyLoss()
        else:
            y_t = torch.as_tensor(y_enc, dtype=torch.float32, device=device).reshape(-1, 1)
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        X_tr, y_tr = X_t[train_t], y_t[train_t]
        X_val = y_val = None
        if val_idx is not None:
            val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
            X_val, y_val = X_t[val_t], y_t[val_t]

        n_train = int(train_idx.size)
        if self.batchnorm and n_train < 2:
            raise ModelError(
                "batchnorm=True needs at least 2 training rows — got "
                f"{n_train} (lower val_split or disable batchnorm)"
            )

        epochs = int(self.epochs)
        batch_size = int(self.batch_size)
        patience = int(self.patience)

        history: list[dict[str, Any]] = []
        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0

        for epoch in range(1, epochs + 1):
            network.train()
            perm = torch.as_tensor(rng.permutation(n_train), device=device)
            batch_losses: list[float] = []
            for start in range(0, perm.numel(), batch_size):
                batch = perm[start : start + batch_size]
                members = int(batch.numel())
                if members < 2 and self.batchnorm:
                    # BatchNorm1d cannot compute batch statistics from a
                    # single sample (the partial last batch); skip it.
                    continue
                optimizer.zero_grad()
                loss = criterion(network(X_tr[batch]), y_tr[batch])
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

            val_loss: float | None = None
            if X_val is not None:
                network.eval()
                with torch.no_grad():
                    val_loss = float(criterion(network(X_val), y_val).item())
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
            "%s fitted on %d sequences (%d steps x %d channels): %d epochs run, "
            "%d parameters, device=%s",
            type(self).__name__,
            int(X_scaled.shape[0]),
            int(X_scaled.shape[1]),
            int(X_scaled.shape[2]),
            self.n_iter_,
            n_params,
            device,
        )

    # -- predict ----------------------------------------------------------
    def _ensure_network(self) -> Any:
        """Rebuild the live torch network from the portable weights."""
        network = getattr(self, "_network", None)
        if network is None:
            _require_torch()
            import torch

            spec = self._net_spec_
            network = _build_cnn(
                spec["n_features"],
                spec["n_out"],
                spec["channels"],
                spec["kernel"],
                spec["n_layers"],
            )
            network.load_state_dict(
                {key: torch.from_numpy(value) for key, value in self._weights.items()}
            )
            self._network = network
        return network

    def _forward(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = self._check_X(X)
        if X_arr.shape[2] != self.n_features_in_:
            raise DataValidationError(
                f"X has {X_arr.shape[2]} feature channel(s), but "
                f"{type(self).__name__} was fitted with "
                f"{self.n_features_in_} — pass the same channels as to fit()"
            )
        _require_torch()
        import torch

        network = self._ensure_network()
        network.eval()
        X_scaled = (X_arr - self._mean_) / self._std_
        with torch.no_grad():
            out = network(torch.as_tensor(X_scaled, dtype=torch.float32))
        return out.numpy()

    # -- pickle support ----------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Drop the live torch network from pickles (rebuilt on demand)."""
        state = self.__dict__.copy()
        state["_network"] = None
        return state


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------
class SequenceCNNClassifier(ClassifierMixin, _BaseSequenceCNN):
    """1-D temporal convolutional classifier (PyTorch).

    A stack of ``Conv1d(kernel) -> BatchNorm1d -> ReLU`` layers over
    the time axis of 3-D sequence batches ``(n, seq_len, n_features)``
    (the :func:`~pharmaml.ml.sequences.sliding_windows` layout), a
    concatenated average+max pool over time, and a linear head.
    Locality and translation invariance are the point: a motif
    anywhere in the window lights up the same kernels.  Multiclass
    (K > 2) supported.

    Parameters
    ----------
    channels:
        Conv width (kept across layers).
    kernel:
        Convolution kernel size — a positive ODD int (same-padding
        keeps ``seq_len``; a kernel spanning the motif detects it).
    n_layers:
        Number of conv blocks.
    batchnorm:
        ``BatchNorm1d`` after every conv map (``True`` by default — the
        conv lane's stabiliser; single-sample partial tail batches are
        skipped, the neural.py precedent).

    Notes
    -----
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.  A 2-D tabular matrix is rejected up front
    with the pointer to the DeepMLP / Tabular Transformer families
    (part18 §5.2's no-fake-images rule).
    """

    _is_classifier = True

    def __init__(
        self,
        channels=16,
        kernel=3,
        n_layers=2,
        epochs=120,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=15,
        val_split=0.15,
        random_state=42,
        device="auto",
        batchnorm=True,
    ) -> None:
        self.channels = channels
        self.kernel = kernel
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device
        self.batchnorm = batchnorm

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for 3-D sequence batches ``X``."""
        logits = self._forward(X)
        return self.classes_[np.argmax(logits, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities (softmax, columns == ``classes_``)."""
        logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


class SequenceCNNRegressor(RegressorMixin, _BaseSequenceCNN):
    """1-D temporal convolutional regressor (PyTorch).

    The same conv stack with a single linear output; the target is
    z-scaled on the training rows and ``predict`` inverts the scaling
    (the neural.py contract).
    """

    _is_classifier = False

    def __init__(
        self,
        channels=16,
        kernel=3,
        n_layers=2,
        epochs=120,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=15,
        val_split=0.15,
        random_state=42,
        device="auto",
        batchnorm=True,
    ) -> None:
        self.channels = channels
        self.kernel = kernel
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device
        self.batchnorm = batchnorm

    def predict(self, X: Any) -> np.ndarray:
        """Predict numeric targets for 3-D sequence batches ``X``."""
        self._check_fitted()
        y_mean, y_std = self._y_mean_, self._y_std_
        if y_mean is None or y_std is None:
            raise ModelError(
                f"{type(self).__name__} is missing its target scaling -- "
                "was it fitted as a regressor?"
            )
        raw = self._forward(X).reshape(-1)
        return raw * y_std + y_mean

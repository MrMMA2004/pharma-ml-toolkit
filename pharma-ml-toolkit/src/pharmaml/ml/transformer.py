"""Tabular Transformer estimators (PyTorch) — part18 §5.4.

The Deep Learning architecture expansion's tabular-Transformer lane:
an attention network over the FEATURE TOKENS of a tabular matrix,
in the FT-Transformer spirit (Gorishniy et al., 2021) — every
feature column ``j`` becomes one token (its scaled value projected
into ``d_model`` dimensions plus a learned per-feature embedding),
a learned CLS token is prepended, a stack of self-attention blocks
mixes the tokens, and the CLS output feeds the prediction head.

The honest scope, stated once (part18 §5.4's own rule):

* this is a **Tabular Transformer** — one of the spec's three
  distinguished flavors, trained FROM SCRATCH on the caller's own
  matrix.  No pretrained molecular-sequence Transformer is claimed
  anywhere in this toolkit: molecular sequences are served by the
  SMILES RNN prior (:mod:`pharmaml.generative.rnn`) and graphs by the
  GNN lane (:mod:`pharmaml.ml.gnn`), and a pretrained-weights claim
  would be false until weights actually ship.
* inputs are the ordinary tabular matrix ``(n_rows, n_features)``
  the rest of the toolkit already standardises; the tokenisation is
  internal.  Sequences and graphs have their own lanes.

Everything else follows :mod:`pharmaml.ml.neural`'s established
contract verbatim so the DL route treats the family exactly like the
DeepMLP/ResidualMLP ones: sklearn estimator contract (``fit`` /
``predict`` / ``predict_proba``, clonable), lazily-imported torch
with the actionable :class:`~pharmaml.exceptions.DependencyError`,
``fit(X, y, on_epoch=...)`` per-epoch telemetry
(``train_loss``/``val_loss``/``best_val``/``stale``), early stopping
on a stratified validation split, per-feature standardisation fitted
on training rows only, seeded determinism (torch + NumPy, the global
torch RNG snapshot-and-restored), portable NumPy weight dicts so
pickles never embed torch classes, and honest typed errors for every
degenerate input.

Examples
--------
>>> from pharmaml.ml.transformer import TabularTransformerClassifier
>>> TabularTransformerClassifier().predict([[0.0]])  # fit before predict
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: TabularTransformerClassifier has not been fitted yet — call fit(X, y) first

>>> TabularTransformerClassifier(d_model=17, heads=4).fit([[0.0]] * 4, [0, 1, 0, 1])
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: d_model (17) must be divisible by heads (4) — the attention blocks split d_model evenly across the heads

>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(40, 3))
>>> y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(int)
>>> net = TabularTransformerClassifier(d_model=16, heads=4, n_blocks=1,
...     dim_feedforward=32, epochs=15, random_state=0).fit(X, y)
>>> twin = TabularTransformerClassifier(d_model=16, heads=4, n_blocks=1,
...     dim_feedforward=32, epochs=15, random_state=0).fit(X, y)
>>> int(np.array_equal(net.predict(X), twin.predict(X)))   # same seed
1
>>> len(net.history_) == net.n_iter_   # the history and the epoch count agree
True

The regressor z-scales its target (the neural.py contract —
pharmaceutical endpoints live far from zero):

>>> reg = TabularTransformerRegressor(d_model=16, heads=4, n_blocks=1,
...     dim_feedforward=32, epochs=25, random_state=1).fit(X, X[:, 1])
>>> float(np.abs(reg.predict(X) - X[:, 1]).mean()) < 1.0
True
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

__all__ = [
    "TabularTransformerClassifier",
    "TabularTransformerRegressor",
]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when torch is missing.
_TORCH_HINT = "deep learning models require torch — pip install 'pharmaml[deep]'"

#: Legal ``device`` tokens ('auto' resolves to cuda when available).
_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


# ---------------------------------------------------------------------------
# Lazy torch access
# ---------------------------------------------------------------------------
def _require_torch() -> ModuleType:
    """Import and return torch, or raise the actionable error.

    Raises
    ------
    DependencyError
        When torch is not installed (message names the ``pharmaml[deep]``
        extra that installs it).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise DependencyError(f"{_TORCH_HINT} ({exc})") from exc
    return torch


def _build_transformer(
    n_features: int,
    n_out: int,
    d_model: int,
    heads: int,
    n_blocks: int,
    dim_feedforward: int,
    dropout: float,
) -> Any:
    """The tokenised tabular transformer (a torch ``Module``).

    Tokenisation: per-feature ``Linear(1 -> d_model)`` projections of
    the (already standardised) values + a learned per-feature position
    embedding + a learned CLS token; the encoder is a stack of torch's
    own ``TransformerEncoderLayer`` blocks (batch-first, GELU); the
    head reads the CLS token output through a final LayerNorm.
    The class is defined lazily (never at module import) so importing
    :mod:`pharmaml.ml.transformer` works on machines without torch;
    the methods take their ``torch`` references through local imports
    (method bodies resolve only module globals, never the builder's
    locals — the neural.py custom-network rule).
    """
    import torch
    from torch import nn

    class _TabularTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_proj = nn.ModuleList(
                [nn.Linear(1, d_model) for _ in range(n_features)]
            )
            self.feature_embed = nn.Embedding(n_features, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                block, num_layers=n_blocks, enable_nested_tensor=False
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, n_out),
            )

        def forward(self, x: Any) -> Any:
            import torch  # local: method bodies see module globals only

            n = x.shape[0]
            tokens = [
                proj(x[:, j : j + 1]).unsqueeze(1)  # (n, 1, d_model)
                for j, proj in enumerate(self.value_proj)
            ]
            positions = self.feature_embed.weight.unsqueeze(0)  # (1, F, d)
            tokens = torch.cat(tokens, dim=1) + positions
            cls = self.cls.expand(1, n, d_model).transpose(0, 1)  # (n, 1, d)
            tokens = torch.cat([cls, tokens], dim=1)
            encoded = self.encoder(tokens)
            return self.head(encoded[:, 0])

    return _TabularTransformer()


class _BaseTabularTransformer(BaseEstimator):
    """Private fit/predict machinery shared by the two public estimators.

    Mirrors :mod:`pharmaml.ml.neural`'s ``_BaseDeepEstimator`` contract
    (validation, standardisation, seeded training loop, early stopping,
    portable weights) with the transformer's tokenised network.
    """

    _is_classifier: bool = False

    def _validate_hyperparameters(self) -> None:
        if isinstance(self.d_model, bool) or not isinstance(
            self.d_model, (int, np.integer)
        ) or self.d_model < 1:
            raise ModelError(f"d_model must be a positive int, got {self.d_model!r}")
        if isinstance(self.heads, bool) or not isinstance(
            self.heads, (int, np.integer)
        ) or self.heads < 1:
            raise ModelError(f"heads must be a positive int, got {self.heads!r}")
        if int(self.d_model) % int(self.heads) != 0:
            raise ModelError(
                f"d_model ({self.d_model}) must be divisible by heads "
                f"({self.heads}) — the attention blocks split d_model "
                "evenly across the heads"
            )
        if isinstance(self.n_blocks, bool) or not isinstance(
            self.n_blocks, (int, np.integer)
        ) or self.n_blocks < 1:
            raise ModelError(f"n_blocks must be a positive int, got {self.n_blocks!r}")
        if isinstance(self.dim_feedforward, bool) or not isinstance(
            self.dim_feedforward, (int, np.integer)
        ) or self.dim_feedforward < 1:
            raise ModelError(
                f"dim_feedforward must be a positive int, got {self.dim_feedforward!r}"
            )
        if isinstance(self.dropout, bool) or not np.isfinite(self.dropout) or not (
            0.0 <= float(self.dropout) < 1.0
        ):
            raise ModelError(f"dropout must be in [0, 1), got {self.dropout!r}")
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

    def _check_X(self, X: Any) -> np.ndarray:
        """Validate and convert ``X`` to a finite float64 matrix."""
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
                f"{type(self).__name__} has not been fitted yet — call fit(X, y) "
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
        self, X: Any, y: Any, *, on_epoch: Callable[[dict[str, Any]], None] | None = None
    ) -> _BaseTabularTransformer:
        """Train the tabular transformer on ``(X, y)`` with early stopping.

        The telemetry contract matches :mod:`pharmaml.ml.neural`:
        ``on_epoch`` receives ``{'epoch', 'epochs', 'train_loss',
        'val_loss', 'best_val', 'stale'}`` after every epoch and must
        never raise.
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
                f"X and y have inconsistent lengths: {X_arr.shape[0]} rows vs "
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
        self.n_features_in_ = int(X_arr.shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        seed = int(self.random_state)
        rng = np.random.default_rng(seed)
        n = int(X_arr.shape[0])
        train_idx, val_idx = self._train_val_split(y_enc, n, rng)

        mean = X_arr[train_idx].mean(axis=0)
        std = X_arr[train_idx].std(axis=0)
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
            "n_features": int(X_scaled.shape[1]),
            "n_out": int(n_out),
            "d_model": int(self.d_model),
            "heads": int(self.heads),
            "n_blocks": int(self.n_blocks),
            "dim_feedforward": int(self.dim_feedforward),
            "dropout": float(self.dropout),
        }
        network = _build_transformer(
            self._net_spec_["n_features"],
            self._net_spec_["n_out"],
            self._net_spec_["d_model"],
            self._net_spec_["heads"],
            self._net_spec_["n_blocks"],
            self._net_spec_["dim_feedforward"],
            self._net_spec_["dropout"],
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
            "%s fitted on %d samples: %d epochs run, %d parameters, device=%s",
            type(self).__name__,
            int(X_scaled.shape[0]),
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
            network = _build_transformer(
                spec["n_features"],
                spec["n_out"],
                spec["d_model"],
                spec["heads"],
                spec["n_blocks"],
                spec["dim_feedforward"],
                spec["dropout"],
            )
            network.load_state_dict(
                {key: torch.from_numpy(value) for key, value in self._weights.items()}
            )
            self._network = network
        return network

    def _forward(self, X: Any) -> np.ndarray:
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
class TabularTransformerClassifier(ClassifierMixin, _BaseTabularTransformer):
    """Tabular Transformer classifier (PyTorch, from scratch).

    FT-Transformer-style attention over the feature tokens of a
    standardised tabular matrix: per-value projection + learned
    per-feature embedding + CLS token, ``n_blocks`` self-attention
    blocks (``heads`` heads of size ``d_model / heads``), CLS head.
    Trained with minibatch Adam, weight decay and early stopping on a
    stratified validation split; multiclass (K > 2) fully supported.

    Parameters
    ----------
    d_model:
        Token width of the transformer; must be divisible by ``heads``.
    heads:
        Attention heads per block.
    n_blocks:
        Number of stacked self-attention blocks.
    dim_feedforward:
        Width of each block's position-wise feed-forward layer.
    dropout:
        Attention/dropout probability inside the blocks (``0`` disables).
    epochs, batch_size, learning_rate, weight_decay, patience, val_split,
    random_state, device:
        The same contract as :class:`~pharmaml.ml.neural.DeepMLPClassifier`.

    Notes
    -----
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.  A *pretrained* molecular-sequence transformer
    is NOT claimed anywhere in this toolkit (part18 §5.4's rule).
    """

    _is_classifier = True

    def __init__(
        self,
        d_model=32,
        heads=4,
        n_blocks=2,
        dim_feedforward=64,
        dropout=0.1,
        epochs=120,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-4,
        patience=15,
        val_split=0.15,
        random_state=42,
        device="auto",
    ) -> None:
        self.d_model = d_model
        self.heads = heads
        self.n_blocks = n_blocks
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for ``X`` (argmax of the softmax)."""
        logits = self._forward(X)
        return self.classes_[np.argmax(logits, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities for ``X`` (softmax, columns == ``classes_``)."""
        logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


class TabularTransformerRegressor(RegressorMixin, _BaseTabularTransformer):
    """Tabular Transformer regressor (PyTorch, from scratch).

    The same tokenised architecture as
    :class:`TabularTransformerClassifier` with a single linear output;
    the target is z-scaled on the training rows and ``predict``
    inverts the scaling (the neural.py contract).
    """

    _is_classifier = False

    def __init__(
        self,
        d_model=32,
        heads=4,
        n_blocks=2,
        dim_feedforward=64,
        dropout=0.1,
        epochs=120,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-4,
        patience=15,
        val_split=0.15,
        random_state=42,
        device="auto",
    ) -> None:
        self.d_model = d_model
        self.heads = heads
        self.n_blocks = n_blocks
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def predict(self, X: Any) -> np.ndarray:
        """Predict numeric targets for ``X`` (de-standardised)."""
        raw = self._forward(X).reshape(-1)
        return raw * self._y_std_ + self._y_mean_

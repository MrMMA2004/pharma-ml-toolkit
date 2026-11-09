"""Deep-learning estimators for QSAR on the toolkit's own autograd engine.

:mod:`pharmaml.ml.neural` brings PyTorch deep learning to the toolkit —
behind an optional dependency.  This module brings the *numpy* path:
:class:`NumpyMLPClassifier` / :class:`NumpyMLPRegressor` are
scikit-learn-contract feed-forward networks trained by the toolkit's
own reverse-mode autodiff engine (:mod:`pharmaml.ml.autograd`), with
the **optimizer as a constructor parameter**: ``sgd`` (momentum,
Nesterov, weight decay), ``adagrad``, ``rmsprop``, ``adam`` (bias
correction, AMSGrad) and ``adamw`` (decoupled decay) — the same five
classes the engine ships, finally driven by a real trainer.

Why this matters beyond the torch-absent machine: the optimizer choice
is a *first-class experimental knob* here, not a pip install.  The
registry key is ``'dnn_numpy'`` (no ``requires`` flag — pure numpy),
so ``train_model('dnn_numpy', ...)`` and the playground's Deep
Learning tab work everywhere, and the optimizer comparison the field
runs before believing a network ("SGD vs Adam on this dataset") is
one ``optimizer=`` swap away.

The training loop mirrors the torch estimators' contract exactly, so
GUI/CLI code treats the two families interchangeably: minibatch
training, early stopping on a held-out validation fraction with
best-weights restore, per-feature standardisation fitted on the
training data, an ``on_epoch`` progress callback, the per-epoch
``history_`` records (``{'epoch', 'train_loss', 'val_loss'}`` with
``val_loss=None`` when ``val_split=0``), ``n_iter_`` epochs-run
count, string class labels out of the box, and a hard
:class:`~pharmaml.exceptions.ModelError` when training diverges to
NaN/inf weights rather than leaving a silently-broken estimator
behind.  Determinism: the same ``random_state`` twice produces
byte-identical predictions.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.models import MODEL_REGISTRY
>>> spec = MODEL_REGISTRY["dnn_numpy"]    # registry key of the numpy family
>>> spec.task, spec.requires
('both', None)
>>> sorted(spec.param_grid)
['hidden_layers', 'learning_rate', 'optimizer']

>>> from pharmaml.ml.numpy_neural import NumpyMLPRegressor
>>> X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
>>> y = 2.0 * X[:, 0] + 1.0
>>> net = NumpyMLPRegressor(hidden_layers=(32,), epochs=250, random_state=0).fit(X, y)
>>> net.score(X, y) > 0.98   # R^2 on a 40-point linear function (seed 0, Adam)
True
>>> twin = NumpyMLPRegressor(hidden_layers=(32,), epochs=250, random_state=0).fit(X, y)
>>> np.array_equal(net.predict(X), twin.predict(X))   # same seed, identical net
True
>>> net.n_iter_ >= 1 and set(net.history_[0]) == {"epoch", "train_loss", "val_loss"}
True

The optimizer is the knob (the engine's own SGD/Adagrad/RMSprop/
Adam/AdamW, selected by name):

>>> sgd_net = NumpyMLPRegressor(
...     hidden_layers=(16,), optimizer="sgd", learning_rate=0.05,
...     epochs=200, random_state=0).fit(X, y)
>>> sgd_net.score(X, y) > 0.9
True
>>> adamw_net = NumpyMLPRegressor(
...     hidden_layers=(16,), optimizer="adamw", weight_decay=1e-4,
...     epochs=200, random_state=0).fit(X, y)
>>> adamw_net.score(X, y) > 0.9
True

String class labels, softmax probabilities, the sklearn contract:

>>> from pharmaml.ml.numpy_neural import NumpyMLPClassifier
>>> Xc = np.concatenate(
...     [np.linspace(-1.0, 0.0, 20), np.linspace(1.0, 2.0, 20)]).reshape(-1, 1)
>>> yc = np.array(["low"] * 20 + ["high"] * 20)
>>> clf = NumpyMLPClassifier(hidden_layers=(8,), epochs=120, random_state=0).fit(Xc, yc)
>>> clf.score(Xc, yc) >= 0.95
True
>>> clf.classes_.tolist()
['high', 'low']
>>> probs = clf.predict_proba(Xc[:2])
>>> bool(np.allclose(probs.sum(axis=1), 1.0))
True

Errors are named, not crashed on:

>>> NumpyMLPRegressor().predict([[0.0]])  # sklearn contract: fit before predict
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: NumpyMLPRegressor has not been fitted yet — call fit(X, y) first
>>> NumpyMLPRegressor(optimizer="nope").fit([[0.0], [1.0]], [0.0, 1.0])
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: unknown optimizer 'nope' -- valid optimizers: sgd, adagrad, rmsprop, adam, adamw
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger
from pharmaml.utils.checkpoints import CheckpointManager, PersistenceError

__all__ = [
    "NumpyMLPClassifier",
    "NumpyMLPRegressor",
]

_LOG = get_logger(__name__)

#: Optimizer tokens accepted by the ``optimizer`` constructor parameter --
#: exactly the classes :mod:`pharmaml.ml.autograd` ships.
OPTIMIZERS: tuple[str, ...] = ("sgd", "adagrad", "rmsprop", "adam", "adamw")

#: Activation tokens accepted by the ``activation`` parameter.  ``gelu``
#: is the tanh approximation (numpy has no erf); the torch estimators'
#: exact erf form differs in the fourth decimal.
_ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "sigmoid", "gelu")


# ---------------------------------------------------------------------------
# the activation pair (Tensor path for training, numpy path for predict)
# ---------------------------------------------------------------------------


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _gelu_np(z: np.ndarray) -> np.ndarray:
    """tanh-approximation GELU (the missing-erf numpy form)."""
    inner = z + 0.044715 * z * z * z
    return 0.5 * z * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * inner))


def _gelu_tensor(x):  # Tensor, typed loosely to keep the engine lazy
    inner = x + 0.044715 * x * x * x
    return 0.5 * x * (1.0 + math.sqrt(2.0 / math.pi) * inner.tanh())


#: activation name -> (Tensor path, numpy path)
_ACTIVATION_FNS: dict[str, tuple[Callable, Callable]] = {
    "relu": (lambda t: t.relu(), lambda z: np.maximum(z, 0.0)),
    "tanh": (lambda t: t.tanh(), np.tanh),
    "sigmoid": (lambda t: t.sigmoid(), _sigmoid_np),
    "gelu": (_gelu_tensor, _gelu_np),
}


def _build_optimizer(
    name: str,
    params: list,
    lr: float,
    weight_decay: float,
    momentum: float,
):
    """Map an ``optimizer`` token onto an autograd optimizer instance."""
    from pharmaml.ml.autograd import SGD, Adagrad, Adam, AdamW, RMSprop

    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adagrad":
        return Adagrad(params, lr=lr)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ModelError(
        f"unknown optimizer {name!r} -- valid optimizers: " + ", ".join(OPTIMIZERS)
    )


# ---------------------------------------------------------------------------
# SS20.4 checkpoint payload helpers -- JSON + base64, never pickle (a
# malicious .json cannot execute on load; arrays ride as base64).
# ---------------------------------------------------------------------------


def _b64_pack(array: np.ndarray) -> dict[str, object]:
    """One array -> a JSON-safe ``{dtype, shape, data}`` payload."""
    arr = np.ascontiguousarray(np.asarray(array))
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _b64_unpack(payload: object) -> np.ndarray:
    """Rebuild one :func:`_b64_pack` payload into a fresh array."""
    if not isinstance(payload, dict):
        raise TypeError("array payload must be a dict")
    arr = np.frombuffer(
        base64.b64decode(payload["data"]), dtype=np.dtype(payload["dtype"])
    )
    return arr.reshape([int(s) for s in payload["shape"]]).copy()


def _config_fingerprint(estimator: _BaseNumpyMLP) -> str:
    """SHA-256 over EVERY trajectory-shaping hyperparameter.

    ``epochs``/``patience`` stay budgets -- free to differ between the
    interrupted and the resuming call -- so they are deliberately NOT
    fingerprinted (SS20.5).
    """
    payload = {
        "hidden_layers": [int(h) for h in estimator.hidden_layers],
        "activation": str(estimator.activation),
        "optimizer": str(estimator.optimizer),
        "learning_rate": float(estimator.learning_rate),
        "weight_decay": float(estimator.weight_decay),
        "momentum": float(estimator.momentum),
        "batch_size": int(estimator.batch_size),
        "val_split": float(estimator.val_split),
        "dropout": float(estimator.dropout),
        "random_state": int(estimator.random_state),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _dataset_fingerprint(matrix: np.ndarray, encoded: np.ndarray) -> str:
    """SHA-256 over X and the encoded y (dtype + shape + raw bytes)."""
    digest = hashlib.sha256()
    for arr in (matrix, encoded):
        a = np.ascontiguousarray(arr)
        digest.update(str(a.dtype).encode("utf-8"))
        digest.update(str(a.shape).encode("utf-8"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _encode_optimizer_state(state: dict[str, object]) -> dict[str, object]:
    """``Optimizer.state_dict()`` -> JSON-safe (arrays become base64)."""
    raw_count = state["step_count"]
    if not isinstance(raw_count, int) or isinstance(raw_count, bool):
        raise TypeError(
            f"optimizer state step_count must be an int, got {type(raw_count).__name__}"
        )
    encoded: dict[str, object] = {"step_count": raw_count}
    for key, value in state.items():
        if key == "step_count":
            continue
        if not isinstance(value, list):
            raise TypeError(f"optimizer buffer {key!r} must be a list")
        encoded[key] = [_b64_pack(buf) for buf in value]
    return encoded


def _decode_optimizer_state(payload: dict[str, object]) -> dict[str, object]:
    """Inverse of :func:`_encode_optimizer_state` (loads raise ModelError)."""
    decoded: dict[str, object] = {"step_count": payload["step_count"]}
    for key, value in payload.items():
        if key == "step_count":
            continue
        if not isinstance(value, list):
            raise TypeError(f"optimizer buffer {key!r} must be a list")
        decoded[key] = [_b64_unpack(item) for item in value]
    return decoded


def _encode_best_weights(
    best_state: list[tuple[np.ndarray, np.ndarray | None]] | None,
) -> list[dict[str, object]] | None:
    """The early-stopping best snapshot -> JSON-safe layer records."""
    if best_state is None:
        return None
    return [
        {"w": _b64_pack(w), "b": None if b is None else _b64_pack(b)}
        for w, b in best_state
    ]


def _decode_best_weights(
    payload: list[dict[str, object]] | None,
) -> list[tuple[np.ndarray, np.ndarray | None]] | None:
    """Inverse of :func:`_encode_best_weights`."""
    if payload is None:
        return None
    restored: list[tuple[np.ndarray, np.ndarray | None]] = []
    for layer in payload:
        bias = layer.get("b")
        restored.append(
            (_b64_unpack(layer["w"]), None if bias is None else _b64_unpack(bias))
        )
    return restored


class _MLPNetwork:
    """The layers: autograd ``Linear`` stack + activation + dropout mask.

    Two forward paths share the same weights: the :class:`~pharmaml.ml.autograd.Tensor`
    path (training, dropout active) and the numpy path (inference,
    dropout always off — exactly the torch ``model.eval()`` semantics).
    """

    def __init__(
        self,
        dims: Sequence[int],
        activation: str,
        dropout: float,
        rng: np.random.Generator,
    ) -> None:
        from pharmaml.ml.autograd import Linear

        self.training = True
        self._dropout = float(dropout)
        self._rng = rng
        self._act_tensor, self._act_np = _ACTIVATION_FNS[activation]
        self.linears = [
            Linear(dims[i], dims[i + 1], seed=int(rng.integers(2**31)))
            for i in range(len(dims) - 1)
        ]

    def parameters(self) -> list:
        params: list = []
        for linear in self.linears:
            params.extend(p for p in (linear.weight, linear.bias) if p is not None)
        return params

    def forward_tensor(self, x):  # Tensor in, Tensor out
        for linear in self.linears[:-1]:
            x = linear.forward(x)
            x = self._act_tensor(x)
            if self.training and self._dropout > 0.0:
                keep = (self._rng.random(x.data.shape) >= self._dropout).astype(float)
                x = x * type(x)(keep / (1.0 - self._dropout))
        return self.linears[-1].forward(x)

    def forward_numpy(self, X: np.ndarray) -> np.ndarray:
        out = X
        for linear in self.linears[:-1]:
            W = np.asarray(linear.weight.data, dtype=float)
            b = (
                np.asarray(linear.bias.data, dtype=float)
                if linear.bias is not None
                else 0.0
            )
            out = self._act_np(out @ W.T + b)
        last = self.linears[-1]
        W = np.asarray(last.weight.data, dtype=float)
        b = (
            np.asarray(last.bias.data, dtype=float) if last.bias is not None else 0.0
        )
        return out @ W.T + b

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [
            (
                np.asarray(linear.weight.data, dtype=float).copy(),
                (
                    np.asarray(linear.bias.data, dtype=float).copy()
                    if linear.bias is not None
                    else None
                ),
            )
            for linear in self.linears
        ]

    def restore(self, state: list[tuple[np.ndarray, np.ndarray | None]]) -> None:
        for linear, (weight, bias) in zip(self.linears, state, strict=True):
            linear.weight.data = weight.copy()
            if bias is not None and linear.bias is not None:
                linear.bias.data = bias.copy()


# ---------------------------------------------------------------------------
# the shared estimator core
# ---------------------------------------------------------------------------


class _BaseNumpyMLP(BaseEstimator):
    """Shared fit core; subclasses define the loss and the prediction head."""

    _is_classifier = False

    def __init__(
        self,
        hidden_layers: Sequence[int] = (64, 32),
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 15,
        val_split: float = 0.15,
        dropout: float = 0.0,
        random_state: int = 42,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 0,
        checkpoint_resume: bool = True,
    ) -> None:
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_split = val_split
        self.dropout = dropout
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.checkpoint_resume = checkpoint_resume

    # -- validation ------------------------------------------------------ #

    def _validate_params(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ModelError(
                f"unknown optimizer {self.optimizer!r} -- valid optimizers: "
                + ", ".join(OPTIMIZERS)
            )
        if self.activation not in _ACTIVATIONS:
            raise ModelError(
                f"unknown activation {self.activation!r} -- valid activations: "
                + ", ".join(_ACTIVATIONS)
            )
        if not self.hidden_layers or any(
            not isinstance(w, (int, np.integer))
            or isinstance(w, bool)
            or int(w) <= 0
            for w in self.hidden_layers
        ):
            raise ModelError(
                f"hidden_layers must be positive widths, got {self.hidden_layers!r}"
            )
        if not (isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0):
            raise ModelError(
                f"learning_rate must be > 0, got {self.learning_rate!r}"
            )
        if not (
            isinstance(self.weight_decay, (int, float)) and self.weight_decay >= 0
        ):
            raise ModelError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if not (isinstance(self.momentum, (int, float)) and 0 <= self.momentum < 1):
            raise ModelError(f"momentum must be in [0, 1), got {self.momentum!r}")
        if not (isinstance(self.batch_size, (int, np.integer)) and int(self.batch_size) >= 1):
            raise ModelError(f"batch_size must be >= 1, got {self.batch_size!r}")
        if not (isinstance(self.epochs, (int, np.integer)) and int(self.epochs) >= 1):
            raise ModelError(f"epochs must be >= 1, got {self.epochs!r}")
        if not (isinstance(self.patience, (int, np.integer)) and int(self.patience) >= 1):
            raise ModelError(f"patience must be >= 1, got {self.patience!r}")
        if not (isinstance(self.val_split, (int, float)) and 0 <= float(self.val_split) < 0.5):
            raise ModelError(
                f"val_split must be in [0, 0.5), got {self.val_split!r}"
            )
        if not (isinstance(self.dropout, (int, float)) and 0 <= float(self.dropout) < 1):
            raise ModelError(f"dropout must be in [0, 1), got {self.dropout!r}")

    def _require_fitted(self) -> None:
        if getattr(self, "_network", None) is None:
            raise ModelError(
                f"{type(self).__name__} has not been fitted yet — call fit(X, y) first"
            )

    def _prepare_X(self, X: object) -> np.ndarray:
        matrix = np.asarray(
            X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else X, dtype=float
        )
        if matrix.ndim != 2:
            raise DataValidationError(
                f"X must be 2-D (n_samples, n_features), got shape {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise DataValidationError(
                "X contains NaN/inf values -- impute before fitting "
                "(the train_model pipeline imputes for you)"
            )
        return matrix

    def _prepare_y(self, y: object, n: int) -> np.ndarray:
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != n:
            raise DataValidationError(
                f"y must be 1-D with {n} entries, got shape {target.shape}"
            )
        if bool(pd.isna(target).any()):
            raise DataValidationError(
                "y contains NaN values -- drop or impute the target before fitting"
            )
        return target

    # -- the training loop ------------------------------------------------ #

    def fit(self, X: object, y: object, on_epoch: Callable | None = None):
        """Minibatch training with early stopping (the torch contract).

        Parameters
        ----------
        X, y:
            Numeric feature matrix and target vector (string class
            labels are fine for the classifier).
        on_epoch:
            Optional ``callable(dict)`` progress hook, called once per
            epoch with ``{'epoch', 'epochs', 'train_loss', 'val_loss',
            'best_val', 'stale'}`` — the playground's Deep Learning tab
            drives its progress bar from it.

        The estimator's ``checkpoint_dir`` / ``checkpoint_every`` /
        ``checkpoint_resume`` parameters wire the SS20.4 depth: after
        every ``checkpoint_every`` epochs AND the final epoch the
        COMPLETE resumable state (weights, optimizer moments + step
        count, training RNG, early-stopping state, history) is written
        through the round-229 :class:`CheckpointManager` — atomic,
        sha-256 self-verified, JSON + base64, never pickle.  A fit
        interrupted after N epochs and resumed from its checkpoint
        reaches BIT-IDENTICAL predictions vs the uninterrupted run
        (``np.array_equal`` — ``allclose`` would hide a rewound Adam
        bias correction or a shifted batch permutation).  Incompatible
        or unreadable checkpoints are discarded with a warning and
        training starts fresh; writes are best-effort and degrade to
        warnings; ``epochs``/``patience`` stay free budgets.

        Checkpoint round-trip — the interrupted + resumed pair equals
        the uninterrupted run exactly:

        >>> import numpy as np, tempfile
        >>> rng = np.random.default_rng(3)
        >>> Xc = rng.normal(size=(40, 3))
        >>> yc = Xc @ np.array([1.5, -2.0, 0.5]) + 0.3
        >>> with tempfile.TemporaryDirectory() as home:
        ...     part = NumpyMLPRegressor(hidden_layers=(12,), epochs=5,
        ...         random_state=5, checkpoint_dir=home,
        ...         checkpoint_every=3).fit(Xc, yc)
        ...     full = NumpyMLPRegressor(hidden_layers=(12,), epochs=8,
        ...         random_state=5, checkpoint_dir=home,
        ...         checkpoint_every=3).fit(Xc, yc)
        ...     rest = NumpyMLPRegressor(hidden_layers=(12,), epochs=8,
        ...         random_state=5, checkpoint_dir=home).fit(Xc, yc)
        ...     bit_exact = bool(np.array_equal(full.predict(Xc), rest.predict(Xc)))
        >>> bit_exact
        True
        """
        from pharmaml.ml.autograd import Tensor, cross_entropy, mse_loss

        self._validate_params()
        matrix = self._prepare_X(X)
        n, n_features = matrix.shape
        target = self._prepare_y(y, n)
        if self._is_classifier:
            classes = np.unique(target)
            if len(classes) < 2:
                raise ModelError(
                    f"classification needs >= 2 classes, got {len(classes)}"
                )
            self.classes_ = classes
            encoded = np.searchsorted(classes, target).astype(float)
        else:
            encoded = target.astype(float)

        # the sklearn fitted-attribute contract: n_features_in_ (what
        # check-CSV style audits read) and feature_names_in_ when the
        # caller passed a DataFrame.
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns])

        rng = np.random.default_rng(self.random_state)
        use_val = float(self.val_split) > 0.0 and n >= 4
        if use_val:
            perm = rng.permutation(n)
            n_val = max(1, math.ceil(float(self.val_split) * n))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            val_idx = np.asarray([], dtype=int)
            train_idx = np.arange(n)

        # standardisation fitted on the TRAINING rows only
        self._mean = matrix[train_idx].mean(axis=0)
        spread = matrix[train_idx].std(axis=0)
        spread[spread == 0.0] = 1.0
        self._scale = spread
        Z = (matrix - self._mean) / self._scale

        out_dim = len(self.classes_) if self._is_classifier else 1
        dims = [n_features, *map(int, self.hidden_layers), out_dim]
        network = _MLPNetwork(dims, self.activation, float(self.dropout), rng)
        params = network.parameters()
        optimizer = _build_optimizer(
            self.optimizer,
            params,
            float(self.learning_rate),
            float(self.weight_decay),
            float(self.momentum),
        )

        batch = max(1, int(self.batch_size))
        y_train_col = encoded[train_idx].reshape(-1, 1)

        # -- SS20.4/SS20.5: resumable checkpoints through CheckpointManager --
        manager: CheckpointManager | None = None
        if self.checkpoint_dir is not None:
            try:
                manager = CheckpointManager(Path(self.checkpoint_dir), name="numpy-mlp")
            except PersistenceError as exc:
                _LOG.warning("checkpoint home unusable (%s); training without checkpoints", exc)
        cfg_fp = _config_fingerprint(self)
        ds_fp = _dataset_fingerprint(matrix, encoded)

        history: list[dict[str, object]] = []
        best_val = float("inf")
        best_state: list[tuple[np.ndarray, np.ndarray | None]] | None = None
        stale = 0
        resumed_epoch = 0
        if manager is not None and self.checkpoint_resume:
            try:
                latest = manager.latest()
            except OSError as exc:
                latest = None
                _LOG.warning("checkpoint scan failed (%s); starting fresh", exc)
            if latest is not None:
                state = latest.state
                if state.get("config_fp") != cfg_fp or state.get("dataset_fp") != ds_fp:
                    _LOG.warning(
                        "checkpoint at step %d is for a different config/dataset; "
                        "starting fresh",
                        latest.step,
                    )
                else:
                    try:
                        network.restore(
                            [
                                (
                                    _b64_unpack(layer["w"]),
                                    None if layer.get("b") is None else _b64_unpack(layer["b"]),
                                )
                                for layer in state["weights"]
                            ]
                        )
                        optimizer.load_state_dict(
                            _decode_optimizer_state(state["optimizer"])
                        )
                        rng.bit_generator.state = state["rng_state"]
                        history = [dict(h) for h in state.get("history", [])]
                        saved_val = state.get("best_val")
                        best_val = float("inf") if saved_val is None else float(saved_val)
                        stale = int(state.get("stale", 0))
                        best_state = _decode_best_weights(state.get("best_weights"))
                        resumed_epoch = int(latest.step)
                        _LOG.info("resumed %s from checkpoint step %d", type(self).__name__, resumed_epoch)
                    except (KeyError, TypeError, ValueError, ModelError) as exc:
                        _LOG.warning("checkpoint unreadable (%s); starting fresh", exc)
                        history = []
                        best_val = float("inf")
                        best_state = None
                        stale = 0
                        resumed_epoch = 0
        for epoch in range(resumed_epoch + 1, int(self.epochs) + 1):
            network.training = True
            order = rng.permutation(len(train_idx))
            losses: list[float] = []
            for start in range(0, len(order), batch):
                pick = order[start : start + batch]
                x_batch = Tensor(Z[train_idx][pick])
                logits = network.forward_tensor(x_batch)
                if self._is_classifier:
                    loss = cross_entropy(logits, encoded[train_idx][pick])
                else:
                    loss = mse_loss(logits, y_train_col[pick])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()) * len(pick))
            train_loss = float(np.sum(losses) / max(1, len(train_idx)))

            val_loss: float | None = None
            if use_val:
                network.training = False
                val_logits = network.forward_numpy(Z[val_idx])
                if self._is_classifier:
                    val_loss = float(
                        _numpy_cross_entropy(val_logits, encoded[val_idx])
                    )
                else:
                    residual = val_logits - encoded[val_idx].reshape(-1, 1)
                    val_loss = float(np.mean(residual * residual))
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = network.snapshot()
                    stale = 0
                else:
                    stale += 1
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            if on_epoch is not None:
                on_epoch(
                    {
                        "epoch": epoch,
                        "epochs": int(self.epochs),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val": best_val if best_val != float("inf") else None,
                        "stale": stale,
                    }
                )
            if manager is not None:
                every = int(self.checkpoint_every)
                if (every > 0 and epoch % every == 0) or epoch == int(self.epochs):
                    _write_checkpoint(
                        manager,
                        epoch,
                        int(self.epochs),
                        type(self).__name__,
                        network,
                        optimizer,
                        rng,
                        history,
                        best_val,
                        best_state,
                        stale,
                        cfg_fp,
                        ds_fp,
                    )
            if use_val and stale >= int(self.patience):
                if manager is not None:
                    _write_checkpoint(
                        manager,
                        epoch,
                        int(self.epochs),
                        type(self).__name__,
                        network,
                        optimizer,
                        rng,
                        history,
                        best_val,
                        best_state,
                        stale,
                        cfg_fp,
                        ds_fp,
                    )
                _LOG.debug(
                    "early stopping at epoch %03d (no val improvement for %d epochs)",
                    epoch,
                    int(self.patience),
                )
                break

        if best_state is not None:
            network.restore(best_state)
        network.training = False
        weights = network.snapshot()
        if not all(
            np.all(np.isfinite(w)) and (b is None or np.all(np.isfinite(b)))
            for w, b in weights
        ):
            raise ModelError(
                "training diverged — the fitted weights contain NaN/inf "
                "values; lower learning_rate (e.g. 1e-3) and refit"
            )
        self._network = network
        self.history_ = history
        self.n_iter_ = len(history)
        last = history[-1]
        _LOG.info(
            "%s fitted on %d samples: epoch %03d train_loss %.4f val_loss %s "
            "(%d epochs run, optimizer=%s)",
            type(self).__name__,
            n,
            last["epoch"],
            last["train_loss"],
            "n/a" if last["val_loss"] is None else f"{last['val_loss']:.4f}",
            self.n_iter_,
            self.optimizer,
        )
        return self

    # -- inference -------------------------------------------------------- #

    def _forward(self, X: object) -> np.ndarray:
        self._require_fitted()
        Z = (self._prepare_X(X) - self._mean) / self._scale
        return self._network.forward_numpy(Z)


def _write_checkpoint(
    manager: CheckpointManager,
    epoch: int,
    total: int,
    estimator_name: str,
    network: _MLPNetwork,
    optimizer: object,
    rng: np.random.Generator,
    history: list[dict[str, object]],
    best_val: float,
    best_state: list[tuple[np.ndarray, np.ndarray | None]] | None,
    stale: int,
    cfg_fp: str,
    ds_fp: str,
) -> None:
    """Persist the COMPLETE resumable state (SS20.4) -- best-effort.

    Any write failure (unusable home, full disk, non-serialisable
    state) degrades to a log warning; training itself never stops
    because of a checkpoint problem.
    """
    payload = {
        "schema_note": "numpy-mlp complete resumable state (Part-3 SS20.4)",
        "config_fp": cfg_fp,
        "dataset_fp": ds_fp,
        "weights": _encode_best_weights(network.snapshot()),
        "optimizer": _encode_optimizer_state(optimizer.state_dict()),  # type: ignore[attr-defined]
        "rng_state": rng.bit_generator.state,
        "history": [dict(h) for h in history],
        "best_val": None if not math.isfinite(best_val) else float(best_val),
        "stale": int(stale),
        "best_weights": _encode_best_weights(best_state),
    }
    try:
        manager.save(
            step=epoch,
            total=total,
            state=payload,
            note=f"{estimator_name} epoch {epoch}",
        )
    except (PersistenceError, OSError, TypeError, ValueError, KeyError) as exc:
        _LOG.warning(
            "checkpoint write failed at epoch %d (%s); continuing without it",
            epoch,
            exc,
        )


def _numpy_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Softmax cross-entropy in plain numpy (the validation-loss path)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(labels)), labels.astype(int)]))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, max-shifted for stability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class NumpyMLPClassifier(ClassifierMixin, _BaseNumpyMLP):
    """Feed-forward classifier trained on the autograd engine.

    Softmax cross-entropy loss over ``np.unique(y)`` classes (string
    labels included), Adam by default, any of :data:`OPTIMIZERS` by
    name.  ``predict_proba`` returns the softmax; ``score`` is the
    accuracy.
    """

    _is_classifier = True

    def predict(self, X: object) -> np.ndarray:
        """Class labels (argmax of the softmax)."""
        logits = self._forward(X)
        return self.classes_[np.argmax(logits, axis=1)]

    def predict_proba(self, X: object) -> np.ndarray:
        """Class-conditional probabilities (softmax, rows sum to 1)."""
        return _softmax_np(self._forward(X))


class NumpyMLPRegressor(RegressorMixin, _BaseNumpyMLP):
    """Feed-forward regressor trained on the autograd engine.

    Mean-squared-error loss on a single output unit, Adam by default,
    any of :data:`OPTIMIZERS` by name.  ``score`` is the R².
    """

    _is_classifier = False

    def predict(self, X: object) -> np.ndarray:
        """Continuous predictions, shape ``(n_samples,)``."""
        return self._forward(X).reshape(-1)

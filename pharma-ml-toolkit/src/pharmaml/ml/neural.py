"""Deep-learning estimators for QSAR (PyTorch, scikit-learn compatible).

The legacy codebase shipped one shallow ``MLPClassifier`` recipe under
the name "ANN"; this module brings *real* deep learning to the
toolkit: feed-forward networks (:class:`DeepMLPClassifier` /
:class:`DeepMLPRegressor`) and residual networks
(:class:`ResidualMLPClassifier` / :class:`ResidualMLPRegressor`) built
on PyTorch, with minibatch Adam training (weight decay), early
stopping on a held-out validation split and per-feature
standardisation fitted on the training data.

The estimators follow the scikit-learn estimator contract (``fit`` /
``predict`` / ``predict_proba``, ``get_params`` / ``set_params``,
clonable by :class:`~sklearn.model_selection.GridSearchCV`), so they
drop into :func:`~pharmaml.ml.training.train_model`,
:func:`~pharmaml.ml.training.compare_models` and
:func:`~pharmaml.ml.evaluation.evaluate_model` like any registry
model — the registry keys are ``'dnn'`` and ``'dnn_residual'``.

PyTorch is an *optional* dependency imported lazily, mirroring how
:mod:`pharmaml.ml.models` treats XGBoost/LightGBM/CatBoost: importing
this module (or ``pharmaml`` itself) never requires torch; the import
happens at :meth:`fit` time and raises an actionable
:class:`~pharmaml.exceptions.DependencyError` when missing::

    pip install 'pharmaml[deep]'

Determinism: ``fit`` seeds both the PyTorch and NumPy random number
generators from ``random_state``, so two estimators constructed with
the same seed and trained on the same data produce identical
predictions (CPU).  Fitted weights are always stored as plain NumPy
arrays, so pickled estimators stay portable across torch versions —
and even across machines without torch installed (torch is only
needed again to *predict*).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

__all__ = [
    "DeepMLPClassifier",
    "DeepMLPRegressor",
    "ResidualMLPClassifier",
    "ResidualMLPRegressor",
]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when torch is missing.
_TORCH_HINT = "deep learning models require torch — pip install 'pharmaml[deep]'"

#: Activation tokens accepted by the ``activation`` constructor parameter.
_ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "gelu", "sigmoid")

#: Legal ``device`` tokens ('auto' resolves to cuda when available).
_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


# ---------------------------------------------------------------------------
# Lazy torch access
# ---------------------------------------------------------------------------
def _require_torch() -> "torch":
    """Import and return torch, or raise the actionable error.

    Raises
    ------
    DependencyError
        When torch is not installed (message names the ``pharmaml[deep]``
        extra that installs it).
    """
    try:
        import torch  # noqa: PLC0415 - lazy import is the whole point
    except ImportError as exc:
        raise DependencyError(_TORCH_HINT) from exc
    return torch


def _make_activation(activation: str) -> Any:
    """Return a fresh ``torch.nn`` activation module for a token."""
    from torch import nn  # noqa: PLC0415 - lazy import

    factory = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    return factory[activation]()


def _build_network(
    residual: bool,
    n_in: int,
    hidden: tuple[int, ...],
    n_out: int,
    activation: str,
    dropout: float,
) -> Any:
    """Construct an *untrained* torch network for the given architecture.

    ``residual=False`` builds a plain feed-forward stack (Linear ->
    activation -> dropout between hidden layers); ``residual=True``
    builds :class:`_ResidualNet` blocks with skip connections.  An
    empty ``hidden`` tuple yields a single linear layer (a generalised
    linear model) in both cases.
    """
    if residual:
        return _torch_networks()[2](n_in, hidden, n_out, activation, dropout)

    from torch import nn  # noqa: PLC0415 - lazy import

    layers: list[Any] = []
    widths = [n_in, *hidden, n_out]
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:  # hidden layer -> activation + dropout
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


#: Lazily-created torch network classes, cached so state-dict keys are
#: stable across every build within a process.
_TORCH_NETS: tuple[Any, type, type] | None = None


def _torch_networks() -> tuple[Any, type, type]:
    """Define (once) and return ``(torch, ResidualBlock, ResidualNet)``.

    The ``torch.nn.Module`` subclasses are defined lazily — never at
    module import — so ``import pharmaml.ml.neural`` works on machines
    without torch; caching them keeps ``state_dict`` keys identical
    between the training build and the rebuild-on-predict.
    """
    global _TORCH_NETS
    if _TORCH_NETS is None:
        import torch
        from torch import nn

        class ResidualBlock(nn.Module):
            """Linear -> activation -> dropout -> Linear with an input skip.

            The skip is an identity when input and output widths match,
            else a learned linear projection (the dense analogue of a
            1x1 convolution).
            """

            def __init__(
                self, width_in: int, width_out: int, activation: str, dropout: float
            ) -> None:
                super().__init__()
                self.fc1 = nn.Linear(width_in, width_out)
                self.fc2 = nn.Linear(width_out, width_out)
                self.act = _make_activation(activation)
                self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                self.skip = (
                    nn.Identity() if width_in == width_out else nn.Linear(width_in, width_out)
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out = self.fc2(self.drop(self.act(self.fc1(x))))
                return self.act(out + self.skip(x))

        class ResidualNet(nn.Module):
            """A stack of :class:`ResidualBlock` layers plus a linear head."""

            def __init__(
                self,
                n_in: int,
                hidden: tuple[int, ...],
                n_out: int,
                activation: str,
                dropout: float,
            ) -> None:
                super().__init__()
                widths = [n_in, *hidden]
                self.blocks = nn.ModuleList(
                    ResidualBlock(widths[i], widths[i + 1], activation, dropout)
                    for i in range(len(widths) - 1)
                )
                self.head = nn.Linear(widths[-1], n_out)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        _TORCH_NETS = (torch, ResidualBlock, ResidualNet)
    return _TORCH_NETS


# ---------------------------------------------------------------------------
# Shared estimator machinery
# ---------------------------------------------------------------------------
class _BaseDeepEstimator(BaseEstimator):
    """Private base with the fit/predict machinery shared by the four
    public deep estimators (never instantiated directly).

    Subclasses set two class flags and define the sklearn-visible
    ``__init__`` (all constructor parameters verbatim, so
    ``get_params``/``set_params``/``clone`` work); everything else —
    validation, standardisation, the torch training loop, early
    stopping, weight (de)serialisation — lives here.
    """

    _is_classifier: bool = False
    _residual: bool = False

    # -- validation -----------------------------------------------------
    def _validate_hyperparameters(self) -> None:
        """Raise :class:`ModelError` early for bad constructor values.

        Raises
        ------
        ModelError
            When ``hidden_layers`` holds a non-positive/non-integer
            width, ``dropout``/``val_split`` fall outside ``[0, 1)``,
            ``activation``/``device`` are unknown tokens, ``epochs``/
            ``batch_size``/``patience`` are below 1, ``learning_rate``
            is not positive-finite, ``weight_decay`` is negative, or
            ``random_state`` is not an integer.
        """
        hidden = self.hidden_layers
        if isinstance(hidden, (str, bytes)) or not hasattr(hidden, "__iter__"):
            raise ModelError(
                f"hidden_layers must be a sequence of positive ints, got {hidden!r}"
            )
        for width in hidden:
            if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
                raise ModelError(
                    "hidden_layers entries must be positive ints, got "
                    f"{width!r} in {tuple(hidden)!r}"
                )
        if isinstance(self.dropout, bool) or not np.isfinite(self.dropout) or not (
            0.0 <= float(self.dropout) < 1.0
        ):
            raise ModelError(f"dropout must be in [0, 1), got {self.dropout!r}")
        if self.activation not in _ACTIVATIONS:
            raise ModelError(
                f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}"
            )
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
        """Validate and convert ``X`` to a finite float64 matrix.

        Raises
        ------
        DataValidationError
            When ``X`` is not 2-dimensional, non-numeric, empty, has no
            feature columns, or contains NaN/infinite values.
        """
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
        """Raise :class:`ModelError` when the estimator is used before fit."""
        if getattr(self, "_weights", None) is None:
            raise ModelError(
                f"{type(self).__name__} has not been fitted yet — call fit(X, y) "
                "first"
            )

    # -- fit ------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "_BaseDeepEstimator":
        """Train the network on ``(X, y)`` with early stopping.

        Parameters
        ----------
        X:
            Feature matrix — DataFrame or 2-D array-like; must be
            numeric and finite (the standard ``pharmaml`` pipeline
            imputes/scales upstream).
        y:
            Targets — class labels (any hashable, e.g. strings or ints)
            for the classifiers, numeric values for the regressors.

        Returns
        -------
        self
            Fitted estimator (``classes_``, ``n_features_in_``,
            ``history_``, ``n_iter_`` and the standardisation
            attributes are now available).

        Raises
        ------
        DependencyError
            When torch is not installed (see ``pip install
            'pharmaml[deep]'``).
        ModelError
            For invalid hyperparameters (validated before any work) or
            when predicting before fitting.
        DataValidationError
            For empty/mismatched/non-numeric ``X``/``y``, NaN targets,
            or a single-class classification target.

        Notes
        -----
        Determinism: PyTorch and NumPy are seeded from
        ``random_state`` at fit start, so the same seed + data always
        yields the same fitted network on CPU.
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
            classes, y_enc = np.unique(y_arr, return_inverse=True)
            if classes.size < 2:
                raise DataValidationError(
                    "classification needs at least 2 distinct classes, got "
                    f"{classes.size}: {classes.tolist()[:5]}"
                )
            if classes.dtype.kind in "fc" and np.isnan(classes.astype(float)).any():
                raise DataValidationError(
                    "classification targets contain NaN — drop those rows first"
                )
            self.classes_ = classes
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

        self.n_features_in_ = int(X_arr.shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        # Per-feature standardisation, fitted on the training block only
        # (constant columns get std 1 to avoid division by zero).
        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self._mean_ = mean
        self._std_ = std
        X_scaled = (X_arr - mean) / std

        torch = _require_torch()
        device = self._resolve_device(torch)
        self._train_torch(X_scaled, y_enc, n_out, device, torch)
        return self

    def _resolve_device(self, torch: "torch") -> str:
        """Map the ``device`` token to a concrete torch device string.

        Raises
        ------
        ModelError
            When ``device='cuda'`` is requested but no CUDA device is
            visible to torch.
        """
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
        """Split row indices into (train, validation) index arrays.

        Classification splits are *stratified* (every class keeps at
        least one training row); regression uses a plain random split.
        Returns ``(all_rows, None)`` when ``val_split`` is 0, the data
        is a single row, or the split would be degenerate — in that
        case training runs for all ``epochs`` with no early stopping.
        """
        val_split = float(self.val_split)
        if val_split <= 0 or n < 2:
            return np.arange(n), None
        if self._is_classifier:
            parts: list[np.ndarray] = []
            for cls in np.unique(y_enc):
                members = np.flatnonzero(y_enc == cls)
                members = members[rng.permutation(members.size)]
                k = int(round(val_split * members.size))
                if members.size > 1:
                    k = min(k, members.size - 1)  # keep >= 1 training row per class
                parts.append(members[:k])
            val_idx = np.concatenate(parts).astype(int) if parts else np.array([], dtype=int)
        else:
            perm = rng.permutation(n)
            val_idx = perm[: max(1, int(round(val_split * n)))]
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
        torch: "torch",
    ) -> None:
        """Run the minibatch Adam training loop with early stopping."""
        from torch import nn  # noqa: PLC0415 - lazy import

        seed = int(self.random_state)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        hidden = tuple(int(w) for w in self.hidden_layers)
        dropout = float(self.dropout)
        # Snapshot of the *fitted* architecture — later set_params calls
        # cannot desynchronise the weight rebuild.
        self._net_spec_: dict[str, Any] = {
            "residual": self._residual,
            "n_in": int(X_scaled.shape[1]),
            "hidden": hidden,
            "n_out": int(n_out),
            "activation": self.activation,
            "dropout": dropout,
        }
        network = _build_network(
            self._residual, self._net_spec_["n_in"], hidden, n_out, self.activation, dropout
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

        n = X_scaled.shape[0]
        train_idx, val_idx = self._train_val_split(y_enc, n, rng)
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
            _LOG.debug(
                "epoch %03d train_loss %.3f val_loss %s",
                epoch,
                train_loss,
                "n/a" if val_loss is None else f"{val_loss:.3f}",
            )
            if val_loss is not None and stale >= patience:
                _LOG.debug(
                    "early stopping at epoch %03d (no val improvement for %d epochs)",
                    epoch,
                    patience,
                )
                break

        if best_state is not None:  # restore the best (lowest val loss) weights
            network.load_state_dict(best_state)
        network.eval()

        self.history_ = history
        self.n_iter_ = len(history)
        # Portable CPU weights — pickles never embed torch classes.
        self._weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in network.state_dict().items()
        }
        self._network = network

        last = history[-1]
        val_text = "n/a" if last["val_loss"] is None else f"{last['val_loss']:.4f}"
        n_params = sum(int(p.numel()) for p in network.parameters())
        _LOG.info(
            "%s fitted on %d samples: epoch %03d train_loss %.4f val_loss %s "
            "(%d epochs run, %d parameters, device=%s)",
            type(self).__name__,
            n,
            last["epoch"],
            last["train_loss"],
            val_text,
            self.n_iter_,
            n_params,
            device,
        )

    # -- predict ----------------------------------------------------------
    def _ensure_network(self) -> Any:
        """Return the live torch network, rebuilding it when needed.

        After unpickling (or a ``set_params`` mutation) the live module
        is ``None``; it is reconstructed from the fitted architecture
        snapshot and the stored NumPy weights — deterministic, because
        ``load_state_dict`` overwrites every parameter.
        """
        network = getattr(self, "_network", None)
        if network is None:
            torch = _require_torch()
            spec = self._net_spec_
            network = _build_network(
                spec["residual"], spec["n_in"], spec["hidden"], spec["n_out"],
                spec["activation"], spec["dropout"],
            )
            network.load_state_dict(
                {key: torch.from_numpy(value) for key, value in self._weights.items()}
            )
            self._network = network
        return network

    def _forward(self, X: Any) -> np.ndarray:
        """Standardise ``X`` and forward it through the network (eval mode)."""
        self._check_fitted()
        X_arr = self._check_X(X)
        torch = _require_torch()
        network = self._ensure_network()
        network.eval()  # disable dropout at inference time
        X_scaled = (X_arr - self._mean_) / self._std_
        with torch.no_grad():
            out = network(torch.as_tensor(X_scaled, dtype=torch.float32))
        return out.numpy()

    # -- pickle support ----------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Drop the live torch network from pickles (rebuilt on demand).

        Only the NumPy weight dict is serialised, so pickled estimators
        stay portable across torch versions — and loadable even where
        torch is absent (torch is required again only to predict).
        """
        state = self.__dict__.copy()
        state["_network"] = None
        return state


# ---------------------------------------------------------------------------
# Public estimators — feed-forward
# ---------------------------------------------------------------------------
class DeepMLPClassifier(ClassifierMixin, _BaseDeepEstimator):
    """Deep feed-forward neural-network classifier (PyTorch).

    A fully connected network trained with minibatch Adam and
    weight-decay regularisation, early-stopped on a stratified
    validation split.  Multiclass (K > 2) is fully supported: the
    network has K outputs and ``predict_proba`` returns a full softmax
    matrix aligned with ``classes_``.

    Parameters
    ----------
    hidden_layers:
        Widths of the hidden layers, e.g. ``(64, 32)``.  An empty tuple
        builds a single linear layer — i.e. a (regularised) multinomial
        logistic regression.
    dropout:
        Dropout probability applied after every hidden activation
        (``0`` disables it).
    activation:
        Hidden activation token: ``'relu'``, ``'tanh'``, ``'gelu'`` or
        ``'sigmoid'``.
    epochs:
        Maximum number of training epochs (>= 1).
    batch_size:
        Minibatch size (>= 1).
    learning_rate:
        Adam step size (> 0).
    weight_decay:
        Adam L2 penalty (>= 0).
    patience:
        Early-stopping patience: stop when the validation loss has not
        improved for this many epochs (>= 1).
    val_split:
        Fraction of the training rows held out for early stopping
        (stratified).  ``0`` trains on everything and disables early
        stopping.
    random_state:
        Integer seed — same seed + same data ⇒ identical fitted
        predictions on CPU.
    device:
        ``'auto'`` (cuda if available else cpu), ``'cpu'`` or
        ``'cuda'``.  Training only: fitted weights are always stored
        on CPU.

    Attributes
    ----------
    classes_:
        Sorted class labels (strings label-encoded internally).
    n_features_in_:
        Number of input features seen at fit.
    feature_names_in_:
        Column names when ``X`` was a DataFrame.
    history_:
        List of per-epoch dicts ``{'epoch', 'train_loss', 'val_loss'}``
        (``val_loss`` is ``None`` without a validation split).
    n_iter_:
        Number of epochs actually run (early stopping included).

    Raises
    ------
    DependencyError
        At fit/predict time when torch is missing.
    ModelError
        For invalid hyperparameters or use before ``fit``.
    DataValidationError
        For empty/non-numeric/mismatched data or single-class targets.

    See Also
    --------
    DeepMLPRegressor, ResidualMLPClassifier : siblings in this module.

    Notes
    -----
    Registry alias: ``build_model('dnn', 'classification')``.  Requires
    the optional torch dependency — ``pip install 'pharmaml[deep]'``.
    """

    _is_classifier = True
    _residual = False

    def __init__(
        self,
        hidden_layers=(64, 32),
        dropout=0.1,
        activation="relu",
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
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for ``X`` (argmax of the softmax).

        Parameters
        ----------
        X:
            Feature matrix — DataFrame or 2-D array-like, numeric.

        Returns
        -------
        numpy.ndarray
            Labels in the original ``classes_`` dtype (strings stay
            strings).
        """
        logits = self._forward(X)
        return self.classes_[np.argmax(logits, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities for ``X`` (softmax, columns == classes_).

        Parameters
        ----------
        X:
            Feature matrix — DataFrame or 2-D array-like, numeric.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, len(classes_))``; rows sum to 1.
        """
        logits = self._forward(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


class DeepMLPRegressor(RegressorMixin, _BaseDeepEstimator):
    """Deep feed-forward neural-network regressor (PyTorch).

    Same architecture and training protocol as
    :class:`DeepMLPClassifier` (see its docstring for the full
    parameter/attribute reference) with a single linear output and
    mean-squared-error loss; ``score`` returns R².

    Registry alias: ``build_model('dnn', 'regression')``.  Requires
    the optional torch dependency — ``pip install 'pharmaml[deep]'``.
    """

    _is_classifier = False
    _residual = False

    def __init__(
        self,
        hidden_layers=(64, 32),
        dropout=0.1,
        activation="relu",
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
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def predict(self, X: Any) -> np.ndarray:
        """Predict continuous targets for ``X``.

        Parameters
        ----------
        X:
            Feature matrix — DataFrame or 2-D array-like, numeric.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)``.
        """
        return self._forward(X).reshape(-1)


# ---------------------------------------------------------------------------
# Public estimators — residual
# ---------------------------------------------------------------------------
class ResidualMLPClassifier(ClassifierMixin, _BaseDeepEstimator):
    """Residual (skip-connection) deep-network classifier (PyTorch).

    Each hidden block is Linear -> activation -> dropout -> Linear with
    an additive skip connection: identity when the block's input and
    output widths match, else a learned linear projection (the dense
    analogue of a 1x1 convolution).  This lets gradients flow through
    deeper stacks than a plain feed-forward network; the default
    ``(64, 64, 64)`` stack exercises the identity-skip path.

    The constructor parameters, attributes, error contract and
    determinism guarantee are identical to :class:`DeepMLPClassifier`
    (see its docstring) — only the block architecture differs.

    Registry alias: ``build_model('dnn_residual', 'classification')``.
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.
    """

    _is_classifier = True
    _residual = True

    def __init__(
        self,
        hidden_layers=(64, 64, 64),
        dropout=0.1,
        activation="relu",
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
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for ``X`` (argmax of the softmax).

        Returns
        -------
        numpy.ndarray
            Labels in the original ``classes_`` dtype.
        """
        logits = self._forward(X)
        return self.classes_[np.argmax(logits, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities for ``X`` (softmax, columns == classes_).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, len(classes_))``; rows sum to 1.
        """
        logits = self._forward(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


class ResidualMLPRegressor(RegressorMixin, _BaseDeepEstimator):
    """Residual (skip-connection) deep-network regressor (PyTorch).

    The residual-block counterpart of :class:`DeepMLPRegressor`; see
    :class:`DeepMLPClassifier` for the full parameter/attribute
    reference and :class:`ResidualMLPClassifier` for the block
    architecture.  ``score`` returns R².

    Registry alias: ``build_model('dnn_residual', 'regression')``.
    Requires the optional torch dependency — ``pip install
    'pharmaml[deep]'``.
    """

    _is_classifier = False
    _residual = True

    def __init__(
        self,
        hidden_layers=(64, 64, 64),
        dropout=0.1,
        activation="relu",
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
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def predict(self, X: Any) -> np.ndarray:
        """Predict continuous targets for ``X``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)``.
        """
        return self._forward(X).reshape(-1)

"""Deep ensembles the Precily way: diverse members, mean and spread.

``SmritiChawla/Precily`` trains its drug-response predictors as an
ensemble of five Keras MLPs (``Model1.hdf5`` ... ``Model5.hdf5``, one
per cell-line cross-validation fold, each hyperband-tuned
independently by ``keras-tuner``) and predicts with the five models'
mean -- the R code averages ``prediction1..5`` in ``DrugsPred.R`` and
the shipped ``Drugs_means_sd.csv`` tables carry the mean *and sd*.
That is the Deep Ensembles recipe (Lakshminarayanan et al. 2017):
independent members disagree where the data cannot constrain the
model, and the disagreement -- the per-sample standard deviation
across members -- is the uncertainty estimate.

TensorFlow/Keras is not in the venv and is not reimplemented.  What
lands is the recipe itself, in NumPy: :func:`deep_ensemble` trains
``n_members`` small single-hidden-layer MLPs whose diversity axes are
the ones that survive without a tuning engine -- member-specific
widths (the stand-in for Precily's hyperband-tuned architectures),
member-specific seeds (independent initialisations, the original Deep
Ensembles axis) and optional row bagging (the CV-fold stand-in) --
then reports the members' mean prediction and their per-sample
standard deviation, the honest "how much do the models disagree"
number.  :class:`DeepEnsembleResult` keeps every member's weights, so
``.predict`` extends the ensemble to new rows.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.deepensemble import deep_ensemble
>>> x = np.linspace(-3.0, 3.0, 80).reshape(-1, 1)
>>> rng = np.random.default_rng(42)
>>> y = np.sin(x[:, 0]) + 0.05 * rng.normal(size=x.shape[0])
>>> result = deep_ensemble(x, y, n_members=5, epochs=300, seed=42)
>>> result.mean.shape == (80,)
True
>>> float(np.max(result.sd)) >= 0.0 and result.predictions.shape == (80, 5)
True
>>> rmse = float(np.sqrt(np.mean((result.mean - y) ** 2)))
>>> rmse < 0.1   # five weak learners agree on a sine
True
>>> spread = result.predict(np.array([[-6.0]]))
>>> bool(np.all(np.isfinite(spread)))
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError

__all__ = [  # noqa: RUF022 (kept grouped: classes, then functions)
    "DeepEnsembleResult",
    "EnsembleMember",
    "deepensemble_default_widths",
    "deep_ensemble",
]

#: Precily's five models, widths as the diversity axis (24..96 hidden
#: units stand in for its hyperband-tuned 128-256-unit stacks).
deepensemble_default_widths: tuple[int, ...] = (24, 32, 48, 64, 96)

_ADAM_B1 = 0.9
_ADAM_B2 = 0.999
_ADAM_EPS = 1e-8


@dataclass(frozen=True)
class EnsembleMember:
    """One member's identity and fit quality.

    Attributes
    ----------
    width:
        Hidden-layer width the member was trained with.
    seed:
        The member's own RNG seed (initialisation diversity).
    rows:
        Row count the member saw (bagged members see the full count
        *with repeats*).
    train_rmse:
        Root-mean-square error of this member alone on its own rows.
    """

    width: int
    seed: int
    rows: int
    train_rmse: float


@dataclass(frozen=True)
class DeepEnsembleResult:
    """The fitted ensemble: members, their mean, their disagreement.

    Attributes
    ----------
    members:
        Per-member identity/quality records (the honest "five
        hyperband-tuned models" table Precily ships as HDF5 metadata).
    predictions:
        ``(n_rows, n_members)`` matrix of the members' predictions on
        the training rows.
    mean / sd:
        The ensemble prediction and its per-sample standard deviation
        across members -- the uncertainty estimate.
    weights:
        Per-member ``(w1, b1, w2, b2)`` tuples; :meth:`predict` uses
        them to extend to new rows.
    y_mean / y_std:
        The target standardisation constants (predictions are always
        returned on the ORIGINAL y scale).
    """

    members: tuple[EnsembleMember, ...]
    predictions: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    weights: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]
    y_mean: float
    y_std: float
    feature_names: tuple[str, ...]

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """The ensemble mean on new rows (original y scale).

        Examples
        --------
        >>> import numpy as np
        >>> from pharmaml.ml.deepensemble import deep_ensemble
        >>> x = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
        >>> result = deep_ensemble(x, x[:, 0] ** 2, n_members=3, epochs=120,
        ...                        seed=0)
        >>> result.predict(np.array([[0.5]])).shape
        (1,)
        """
        matrix = _as_matrix(x, self.feature_names)
        outputs = np.stack(
            [_forward(matrix, *member).reshape(-1) for member in self.weights], axis=1
        )
        return outputs.mean(axis=1) * self.y_std + self.y_mean

    def member_frame(self) -> pd.DataFrame:
        """The per-member table (width / seed / rows / train RMSE)."""
        return pd.DataFrame(
            [member.__dict__ for member in self.members]
        )


def _as_matrix(
    x: np.ndarray | pd.DataFrame, feature_names: tuple[str, ...]
) -> np.ndarray:
    """Coerce input to a float matrix with the trained column order."""
    if isinstance(x, pd.DataFrame):
        missing = [name for name in feature_names if name not in x.columns]
        if missing:
            raise DataValidationError(f"x is missing columns: {missing[:3]}")
        matrix = x[list(feature_names)].apply(pd.to_numeric, errors="coerce")
        if matrix.isna().any().any():
            raise DataValidationError("x contains non-numeric/NaN values")
        return matrix.to_numpy(dtype=float)
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise DataValidationError(f"x must be 2-D, got shape {matrix.shape}")
    if feature_names and matrix.shape[1] != len(feature_names):
        raise DataValidationError(
            f"x has {matrix.shape[1]} columns, the ensemble was trained "
            f"on {len(feature_names)}"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError("x contains non-finite values")
    return matrix


def _forward(
    matrix: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """One member's forward pass: ReLU MLP, linear output."""
    hidden = np.maximum(matrix @ w1 + b1, 0.0)
    return hidden @ w2 + b2


def _train_member(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Full-batch Adam on one small MLP; returns (w1, b1, w2, b2, rmse)."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w1 = rng.normal(0.0, np.sqrt(2.0 / d), (d, width))
    b1 = np.zeros(width)
    w2 = rng.normal(0.0, np.sqrt(2.0 / width), (width, 1))
    b2 = np.zeros(1)
    m = [np.zeros_like(w1), np.zeros_like(b1), np.zeros_like(w2), np.zeros_like(b2)]
    v = [np.zeros_like(w1), np.zeros_like(b1), np.zeros_like(w2), np.zeros_like(b2)]
    params = [w1, b1, w2, b2]
    y_col = y.reshape(-1, 1)
    for epoch in range(1, epochs + 1):
        hidden = np.maximum(x @ params[0] + params[1], 0.0)
        pred = hidden @ params[2] + params[3]
        error = pred - y_col
        grad_w2 = hidden.T @ error / n
        grad_b2 = error.mean(axis=0)
        back = (error @ params[2].T) * (hidden > 0.0)
        grad_w1 = x.T @ back / n
        grad_b1 = back.mean(axis=0)
        grads = [grad_w1, grad_b1, grad_w2, grad_b2]
        for k in range(4):
            m[k] = _ADAM_B1 * m[k] + (1.0 - _ADAM_B1) * grads[k]
            v[k] = _ADAM_B2 * v[k] + (1.0 - _ADAM_B2) * grads[k] ** 2
            step = lr * (m[k] / (1.0 - _ADAM_B1**epoch)) / (
                np.sqrt(v[k] / (1.0 - _ADAM_B2**epoch)) + _ADAM_EPS
            )
            params[k] = params[k] - step
    hidden = np.maximum(x @ params[0] + params[1], 0.0)
    pred = (hidden @ params[2] + params[3]).reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return params[0], params[1], params[2], params[3], rmse


def deep_ensemble(
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_members: int = 5,
    hidden_widths: tuple[int, ...] = deepensemble_default_widths,
    epochs: int = 300,
    lr: float = 0.01,
    bagging: bool = True,
    seed: int = 42,
) -> DeepEnsembleResult:
    """Train ``n_members`` diverse NumPy MLPs; predict mean ± sd.

    Precily's ``Model1..5.hdf5`` + ``DrugsPred.R`` mean aggregation,
    without TensorFlow: each member gets its own hidden width (cycled
    from ``hidden_widths`` -- the architecture-diversity stand-in),
    its own seed (the initialisation diversity the Deep Ensembles
    paper prescribes) and, with ``bagging=True``, its own bootstrap
    resample of the rows (the CV-fold stand-in).  Targets are
    z-standardised internally; every reported prediction is on the
    original scale.

    Parameters
    ----------
    x / y:
        Feature matrix and target vector (any sklearn-style shapes;
        a DataFrame keeps its column order for later ``predict``).
    n_members:
        How many members to train (Precily ships five).
    hidden_widths:
        The width pool members cycle through.
    epochs / lr:
        Full-batch Adam epochs and learning rate per member.
    bagging:
        Bootstrap the rows per member (True) or train every member on
        all rows (False -- pure initialisation diversity).
    seed:
        Base seed; member ``m`` trains with ``seed + m``.

    Returns
    -------
    DeepEnsembleResult
        With ``.mean`` (the ensemble prediction), ``.sd`` (the
        per-sample member disagreement -- the uncertainty), the
        members' individual predictions, and ``.predict`` for new
        rows.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-3.0, 3.0, 60).reshape(-1, 1)
    >>> y = np.sin(x[:, 0])
    >>> result = deep_ensemble(x, y, n_members=4, epochs=250, seed=42)
    >>> len(result.members)
    4
    >>> rmse = float(np.sqrt(np.mean((result.mean - y) ** 2)))
    >>> rmse < 0.2
    True
    """
    if n_members < 2:
        raise DataValidationError(f"n_members must be >= 2, got {n_members}")
    if epochs < 10:
        raise DataValidationError(f"epochs must be >= 10, got {epochs}")
    if lr <= 0.0:
        raise DataValidationError(f"lr must be > 0, got {lr}")
    if not hidden_widths or min(hidden_widths) < 2:
        raise DataValidationError("hidden_widths needs widths >= 2")

    if isinstance(x, pd.DataFrame):
        feature_names = tuple(str(c) for c in x.columns)
        matrix = x.apply(pd.to_numeric, errors="coerce")
        if matrix.isna().any().any():
            raise DataValidationError("x contains non-numeric/NaN values")
        values = matrix.to_numpy(dtype=float)
    else:
        feature_names = ()
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] < 8:
        raise DataValidationError(
            f"x must be a 2-D matrix with >= 8 rows, got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise DataValidationError("x contains non-finite values")

    target = np.asarray(y, dtype=float).reshape(-1)
    if target.size != values.shape[0]:
        raise DataValidationError(
            f"x and y row mismatch: {values.shape[0]} vs {target.size}"
        )
    if not np.isfinite(target).all():
        raise DataValidationError("y contains non-finite values")
    y_mean = float(target.mean())
    y_std = float(target.std())
    if y_std <= 0.0:
        raise DataValidationError("y is constant -- nothing to ensemble")
    z_target = (target - y_mean) / y_std

    members: list[EnsembleMember] = []
    weights: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    predictions = np.zeros((values.shape[0], n_members), dtype=float)
    for member_index in range(n_members):
        member_seed = seed + member_index
        width = int(hidden_widths[member_index % len(hidden_widths)])
        rng = np.random.default_rng(1000 + member_seed)
        if bagging:
            rows = rng.integers(0, values.shape[0], values.shape[0])
        else:
            rows = np.arange(values.shape[0])
        w1, b1, w2, b2, _z_rmse = _train_member(
            values[rows], z_target[rows], width, member_seed, epochs, lr
        )
        z_pred = _forward(values, w1, b1, w2, b2).reshape(-1)
        predictions[:, member_index] = z_pred * y_std + y_mean
        members.append(
            EnsembleMember(
                width=width,
                seed=member_seed,
                rows=int(rows.size),
                train_rmse=float(np.sqrt(np.mean((z_pred - z_target) ** 2))),
            )
        )
        weights.append((w1, b1, w2, b2))

    return DeepEnsembleResult(
        members=tuple(members),
        predictions=predictions,
        mean=predictions.mean(axis=1),
        sd=predictions.std(axis=1, ddof=1) if n_members > 1 else np.zeros(values.shape[0]),
        weights=tuple(weights),
        y_mean=y_mean,
        y_std=y_std,
        feature_names=feature_names,
    )

"""Tensors and the from-scratch training loop (the Chollet foundations).

François Chollet's *Deep Learning with Python* notebooks
(``fchollet/deep-learning-with-python-notebooks``) were reviewed file by
file for this round.  Chapter 2 ("mathematical building blocks") and
chapter 5 ("fundamentals of machine learning") carry the parts of the
book every later chapter stands on -- the tensor vocabulary, the
broadcasting contract, the hand-written ``NaiveDense`` /
``NaiveSequential`` / ``BatchGenerator`` trio with its manual gradient
descent loop, and the overfitting-vs-capacity experiment -- and none of
them had a surface in this toolkit: the Deep Learning tab trains
``pharmaml.ml.neural`` estimators through torch (when installed) and
never shows the moving parts.  This module is those moving parts, in
plain NumPy, plus the gradient-descent linear regression that the
SENATOROVAI *Data Science For Beginners* curriculum puts at the centre
of its machine-learning track.

What landed, chapter by chapter:

* :func:`describe_tensor` / :func:`tensor_primer_report` -- chapter 2's
  "data representations" section as an audit: rank, shape, dtype, size,
  per-tensor statistics and the batch arithmetic (``num_batches`` at a
  given batch size, the ragged tail batch included in the count).
* :func:`broadcast_report` -- chapter 2's broadcasting rules as pure
  shape arithmetic: which axes stretch, from what size to what size,
  and the NumPy error text when the shapes are incompatible.
* :class:`NaiveDense`, :class:`NaiveSequential`, :class:`BatchGenerator`
  -- the chapter's own classes, NumPy editions.  ``NaiveDense`` keeps
  Chollet's uniform ``W`` / zero ``b`` initialisation (seeded -- the
  toolkit's ``random_state=42`` pin) and adds the one thing the book's
  autodiff backend hides: :meth:`NaiveDense.backward`, the hand-derived
  ``dW = xᵀ·δ``, ``db = Σδ``, ``dx = δ·Wᵀ`` with the activation
  derivative chained in.
* :func:`fit_scratch` / :func:`predict_scratch` -- the "full training
  loop" cell (batch generator → forward → loss → backward → SGD update)
  with the evaluation the chapter ends on.  Binary classification fuses
  the sigmoid + binary-cross-entropy gradient into the textbook
  ``δ = (p − y)/B`` (the numerically stable form; the unfused
  ``1/(p(1−p))`` blows up on confident predictions).
* :func:`linear_regression_gd` -- the *Data Science For Beginners*
  from-scratch track: batch gradient descent on the squared error, run
  on standardised features for the conditioning GD needs, reported back
  in the original feature space.
* :func:`capacity_sweep` -- chapter 5's headline experiment: the same
  data through growing hidden widths, train loss against validation
  loss, the "where does overfitting start?" curve.

The honest scope, stated once: chapters 8-12 and 17 (convnets,
segmentation, detection, image generation -- the whole image domain)
and chapter 15's trainable Transformer stay out of a NumPy toolkit;
chapter 15's *mathematics* (scaled dot-product attention, positional
encoding) lands in :mod:`pharmaml.ml.attention`, chapter 13's
recurrent models in :mod:`pharmaml.ml.sequences`, and chapter 16's
sampling strategies in :meth:`pharmaml.generative.rnn.SmilesRnn.sample`
(``top_k``).  Chapters 4, 7 and 18 (the classification/regression
workflows, callbacks, tuning and ensembling) were already absorbed by
the Data & ML tab, the estimators' early stopping and
:mod:`pharmaml.ml.tuner`.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.tensors import (
...     BatchGenerator, NaiveDense, NaiveSequential, broadcast_report,
... )
>>> report = broadcast_report((80, 4), (4,))
>>> report.result_shape
(80, 4)
>>> report.stretched[0]                       # the (4,) pads to (1, 4), axis -2 stretches
(-2, 1, 80)

>>> gen = BatchGenerator(np.arange(10).reshape(5, 2), np.arange(5), batch_size=2)
>>> gen.num_batches
3
>>> x_batch, y_batch = gen.next()
>>> x_batch.shape
(2, 2)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through every initialisation (the toolkit's pin).
RANDOM_STATE = 42

#: Activation tokens :class:`NaiveDense` understands (chapter 2 uses
#: relu and sigmoid; tanh joins because the recurrent modules lean on
#: the same derivative table).
SCRATCH_ACTIVATIONS: tuple[str, ...] = ("relu", "sigmoid", "tanh")

#: Training-loop tasks :func:`fit_scratch` understands.
SCRATCH_TASKS: tuple[str, ...] = ("classification", "regression")


# ---------------------------------------------------------------------------
# Chapter 2 -- data representations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSummary:
    """Chapter 2's "key attributes" list for one array."""

    name: str
    dtype: str
    rank: int
    shape: tuple[int, ...]
    size: int
    n_missing: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    batch_size: int
    num_batches: int
    tail_batch_size: int

    def as_row(self) -> list[str]:
        """The primer table's row for this tensor."""
        stats = (
            "—"
            if self.mean is None
            else f"{self.minimum:.3g} … {self.maximum:.3g} · μ {self.mean:.3g} · σ {self.std:.3g}"
        )
        return [
            self.name,
            f"rank {self.rank} · {self.shape}",
            self.dtype,
            f"{self.size:,}",
            stats,
            f"{self.num_batches} × {self.batch_size} (+{self.tail_batch_size} tail)"
            if self.tail_batch_size
            else f"{self.num_batches} × {self.batch_size}",
        ]


def _finite_stats(values: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    """min/max/mean/std of the finite entries (``None`` when empty)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, None, None
    return (
        float(finite.min()),
        float(finite.max()),
        float(finite.mean()),
        float(finite.std()),
    )


def describe_tensor(
    array: Any,
    name: str = "tensor",
    *,
    batch_size: int = 32,
) -> TensorSummary:
    """Audit one array with chapter 2's "key attributes" vocabulary.

    Parameters
    ----------
    array:
        Anything :func:`numpy.asarray` understands (a DataFrame column,
        a feature matrix, a window stack).  Non-numeric arrays raise
        ``DataValidationError`` -- the primer teaches *numeric* tensors.
    name:
        Label carried onto the report row.
    batch_size:
        Batch size for the "notion of data batches" arithmetic.

    Raises
    ------
    DataValidationError
        If ``array`` is empty, ragged, non-numeric, or ``batch_size``
        is out of range.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise DataValidationError(f"batch_size must be a positive int, got {batch_size!r}")
    try:
        values = np.asarray(array)
    except Exception as exc:
        raise DataValidationError(f"{name}: cannot convert to an array ({exc})") from exc
    if values.size == 0:
        raise DataValidationError(f"{name}: empty tensor")
    if values.dtype.kind not in "biufc":
        raise DataValidationError(
            f"{name}: non-numeric dtype {values.dtype!r} -- the tensor primer "
            "describes numeric data"
        )
    flat = values.astype(float, copy=False).ravel()
    n_missing = int(np.count_nonzero(~np.isfinite(flat)))
    minimum, maximum, mean, std = _finite_stats(flat)
    num_batches = math.ceil(flat.size / batch_size) if values.size else 0
    tail = flat.size - (num_batches - 1) * batch_size if num_batches else 0
    return TensorSummary(
        name=name,
        dtype=str(values.dtype),
        rank=int(values.ndim),
        shape=tuple(int(dim) for dim in values.shape),
        size=int(values.size),
        n_missing=n_missing,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        batch_size=batch_size,
        num_batches=num_batches,
        tail_batch_size=int(tail if tail < batch_size else 0),
    )


@dataclass(frozen=True)
class BroadcastReport:
    """NumPy's broadcasting verdict for two shapes.

    ``stretched`` lists ``(axis, from_size, to_size)`` for every axis
    NumPy stretches (size-1 axes only, counted numpy-style -- negative
    for the left-pad axes), because the rule chapter 2 states as
    "smaller arrays are transparently *broadcast*" is exactly this.
    """

    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    stretched: tuple[tuple[int, int, int], ...]
    compatible: bool = True

    def as_text(self) -> str:
        """One-line verdict for the report card."""
        if not self.compatible:
            return f"{self.a_shape} ⊗ {self.b_shape} — incompatible"
        if not self.stretched:
            return f"{self.a_shape} ⊗ {self.b_shape} → {self.result_shape} (no stretching)"
        notes = ", ".join(
            f"axis {axis}: {frm} → {to}" for axis, frm, to in self.stretched
        )
        return f"{self.a_shape} ⊗ {self.b_shape} → {self.result_shape} (stretched: {notes})"


def _parse_shape(text: str) -> tuple[int, ...]:
    """Parse ``(3, 4)`` / ``"3, 4"`` / ``"4"`` into a shape tuple."""
    cleaned = text.strip().strip("()[]").replace("x", ",").replace("×", ",")
    if not cleaned:
        return ()
    try:
        dims = tuple(int(part) for part in cleaned.split(",") if part.strip())
    except ValueError as exc:
        raise DataValidationError(f"cannot parse shape {text!r}") from exc
    if any(dim < 0 for dim in dims):
        raise DataValidationError(f"negative dimension in shape {text!r}")
    return dims


def broadcast_report(a_shape: Any, b_shape: Any) -> BroadcastReport:
    """Chapter 2's broadcasting rules, computed on shapes alone.

    Parameters
    ----------
    a_shape, b_shape:
        Shape tuples, or their ``(3, 4)`` / ``"3,4"`` text forms (the
        GUI passes entry text).

    Raises
    ------
    DataValidationError
        On unparseable shapes.  *Incompatible* shapes do NOT raise --
        the report carries ``compatible=False`` and the verdict text,
        because the lesson is precisely which pairs fail.
    """
    a = a_shape if isinstance(a_shape, tuple) else _parse_shape(str(a_shape))
    b = b_shape if isinstance(b_shape, tuple) else _parse_shape(str(b_shape))
    if not a and not b:
        raise DataValidationError("both shapes are empty")
    max_len = max(len(a), len(b))
    padded_a = (1,) * (max_len - len(a)) + tuple(a)
    padded_b = (1,) * (max_len - len(b)) + tuple(b)
    result: list[int] = []
    stretched: list[tuple[int, int, int]] = []
    compatible = True
    for axis, (dim_a, dim_b) in enumerate(zip(padded_a, padded_b)):
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
            stretched.append((axis - max_len, 1, dim_b))
        elif dim_b == 1:
            result.append(dim_a)
            stretched.append((axis - max_len, 1, dim_a))
        else:
            compatible = False
            result.append(max(dim_a, dim_b))
    return BroadcastReport(
        a_shape=tuple(int(dim) for dim in a),
        b_shape=tuple(int(dim) for dim in b),
        result_shape=tuple(int(dim) for dim in result),
        stretched=tuple(stretched),
        compatible=compatible,
    )


@dataclass(frozen=True)
class PrimerReport:
    """The full chapter 2 walkthrough for one dataset table.

    ``columns`` is the per-column audit; ``matrix`` the feature-matrix
    audit; the demo fields are the chapter's remaining sections as
    single verdict lines (tensor product, reshaping, batching).
    """

    columns: tuple[TensorSummary, ...]
    matrix: TensorSummary
    target: TensorSummary | None
    tensor_product: str
    reshape: str
    batches: str
    batch_size: int

    @property
    def rows(self) -> list[list[str]]:
        """Every audit row (columns first, matrix/target last)."""
        rows = [summary.as_row() for summary in self.columns]
        rows.append(self.matrix.as_row())
        if self.target is not None:
            rows.append(self.target.as_row())
        return rows


def tensor_primer_report(
    frame: Any,
    *,
    target: str | None = None,
    batch_size: int = 32,
    max_columns: int = 12,
) -> PrimerReport:
    """Walk a table through chapter 2's tensor vocabulary.

    Every numeric column is one rank-1 tensor; the numeric block is the
    rank-2 feature matrix (or rank-3 with a batch axis, the "real-world
    examples" section's shape); the tensor-product and reshape demos
    run on that matrix.

    Raises
    ------
    DataValidationError
        If the frame has no numeric columns or fewer than two rows.
    """
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    numeric = [column for column in frame.columns if frame[column].dtype.kind in "biufc"]
    if not numeric:
        raise DataValidationError("no numeric columns -- the tensor primer needs numbers")
    if len(frame) < 2:
        raise DataValidationError("the tensor primer needs at least two rows")
    shown = numeric[:max_columns]
    column_summaries = tuple(
        describe_tensor(frame[column].to_numpy(), name=str(column), batch_size=batch_size)
        for column in shown
    )
    matrix_values = frame[shown].to_numpy(dtype=float)
    matrix = describe_tensor(matrix_values, name=f"[{len(frame)} rows × {len(shown)} numeric cols]", batch_size=batch_size)
    target_summary = None
    if target is not None and target in frame.columns:
        if frame[target].dtype.kind in "biufc":
            target_summary = describe_tensor(
                frame[target].to_numpy(), name=f"target: {target}", batch_size=batch_size
            )
        else:
            _LOG.debug(
                "tensor_primer_report: target %r is non-numeric (%s) -- the "
                "class-label column stays out of the numeric audit",
                target,
                frame[target].dtype,
            )
    n_features = matrix_values.shape[1]
    tensor_product = (
        f"Xᵀ·X → ({n_features}, {n_features}) -- the feature co-occurrence "
        "matrix every second-moment statistic reads from"
        if n_features
        else "no features"
    )
    reshape = (
        f"reshape ({matrix_values.shape[0]}, {n_features}) → "
        f"({matrix_values.shape[0]}, 1, {n_features}) -- the batch-axis view "
        "a conv1d/rnn layer consumes"
    )
    n_batches = math.ceil(len(frame) / batch_size)
    batches = (
        f"{n_batches} batches of {batch_size} at batch_size={batch_size} "
        f"({'ragged tail' if len(frame) % batch_size else 'even'})"
    )
    return PrimerReport(
        columns=column_summaries,
        matrix=matrix,
        target=target_summary,
        tensor_product=tensor_product,
        reshape=reshape,
        batches=batches,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Chapter 2 -- the from-scratch classes
# ---------------------------------------------------------------------------


def _activation_forward(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    if name == "tanh":
        return np.tanh(z)
    if name == "linear":
        return z
    raise DataValidationError(f"unknown activation {name!r}")


def _activation_backward(name: str, z: np.ndarray, out: np.ndarray) -> np.ndarray:
    """The activation derivative, expressed against the forward output."""
    if name == "relu":
        return (z > 0).astype(float)
    if name == "sigmoid":
        return out * (1.0 - out)
    if name == "tanh":
        return 1.0 - out * out
    if name == "linear":
        return np.ones_like(out)
    raise DataValidationError(f"unknown activation {name!r}")


class NaiveDense:
    """Chollet's ``NaiveDense`` (chapter 2), plain NumPy.

    The forward pass is the book's exact ``activation(x·W + b)``.  The
    backward pass is what the book's ``GradientTape`` hides:
    ``δ ← grad_out · f'(z)``, then ``dW = xᵀ·δ``, ``db = Σδ``,
    ``dx = δ·Wᵀ``.  Gradients accumulate on ``dW`` / ``db`` between
    :meth:`backward` and :meth:`apply_gradients` calls (the SGD step
    clears them, so one layer safely serves many batch updates).

    Parameters
    ----------
    input_size, output_size:
        Fan-in and fan-out of the linear map.
    activation:
        One of :data:`SCRATCH_ACTIVATIONS` plus ``"linear"`` (the
        output layer's identity).
    seed:
        Initialisation seed; ``W`` is uniform like the book's, scaled
        by ``1/√fan-in`` so standardised inputs actually converge.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        *,
        seed: int = RANDOM_STATE,
    ) -> None:
        if input_size < 1 or output_size < 1:
            raise DataValidationError(
                f"layer sizes must be positive, got {input_size} → {output_size}"
            )
        if activation not in (*SCRATCH_ACTIVATIONS, "linear"):
            raise DataValidationError(
                f"unknown activation {activation!r} (one of {(*SCRATCH_ACTIVATIONS, 'linear')})"
            )
        rng = np.random.default_rng(seed)
        limit = 1.0 / math.sqrt(input_size)
        self.W = rng.uniform(-limit, limit, size=(input_size, output_size))
        self.b = np.zeros(output_size)
        self.activation = activation
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._out: np.ndarray | None = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(inputs, dtype=float))
        self._x = x
        self._z = x @ self.W + self.b
        self._out = _activation_forward(self.activation, self._z)
        return self._out

    @property
    def weights(self) -> list[np.ndarray]:
        """The book's ``[W, b]`` view (SGD updates walk this list)."""
        return [self.W, self.b]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop ``dL/d(output)``; accumulate ``dW``/``db``; return ``dL/d(input)``."""
        if self._x is None or self._z is None or self._out is None:
            raise DataValidationError("backward() before a forward __call__()")
        delta = np.asarray(grad_out, dtype=float) * _activation_backward(
            self.activation, self._z, self._out
        )
        return self.backward_from_pre(delta)

    def backward_from_pre(self, delta: np.ndarray) -> np.ndarray:
        """Backprop a gradient that already includes the activation term.

        This is the fused path :func:`fit_scratch` uses for the
        sigmoid+binary-cross-entropy head: the fused delta is the
        textbook ``(p − y)/B``, numerically stable where the unfused
        ``1/(p(1−p))`` overflows.
        """
        if self._x is None:
            raise DataValidationError("backward_from_pre() before a forward __call__()")
        delta = np.atleast_2d(np.asarray(delta, dtype=float))
        self.dW += self._x.T @ delta
        self.db += delta.sum(axis=0)
        return delta @ self.W.T

    def apply_gradients(self, learning_rate: float) -> None:
        """One plain SGD step (the book's ``update_weights``)."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    @property
    def n_parameters(self) -> int:
        """Parameter count (the capacity sweep's x-axis)."""
        return int(self.W.size + self.b.size)


class NaiveSequential:
    """Chollet's ``NaiveSequential``: a stack called like a function."""

    def __init__(self, layers: list[NaiveDense]) -> None:
        if not layers:
            raise DataValidationError("NaiveSequential needs at least one layer")
        self.layers = layers

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        for layer in self.layers:
            weights.extend(layer.weights)
        return weights

    def backward(self, grad: np.ndarray) -> None:
        """Chain the backward pass through the stack (right to left)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def backward_fused_head(self, grad_pre: np.ndarray) -> None:
        """Backprop a gradient whose head-activation term is fused in.

        The sigmoid + binary-cross-entropy pair simplifies to
        ``δ = (p − y)/B`` *before* the activation derivative, so the
        head layer must skip its own derivative (the
        :meth:`NaiveDense.backward_from_pre` path) while the earlier
        layers chain normally.
        """
        grad = self.layers[-1].backward_from_pre(grad_pre)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def apply_gradients(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.apply_gradients(learning_rate)

    @property
    def n_parameters(self) -> int:
        return int(sum(layer.n_parameters for layer in self.layers))


class BatchGenerator:
    """Chollet's ``BatchGenerator`` -- sequential, ``num_batches`` counted.

    The only addition over the book's class is the optional seeded
    ``shuffle`` (Keras' ``fit`` shuffles by default; the book's own
    generator does not, and :func:`fit_scratch` passes ``shuffle=True``
    so both behaviours are visible).

    ``next()`` wraps around: one epoch's worth of calls plus one more
    restarts the sweep (the book re-creates the generator per epoch;
    wrapping makes a long-running GUI loop simpler without changing
    the batch arithmetic).
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 128,
        *,
        shuffle: bool = False,
        seed: int = RANDOM_STATE,
    ) -> None:
        if len(images) != len(labels):
            raise DataValidationError(
                f"images and labels disagree ({len(images)} vs {len(labels)})"
            )
        if batch_size < 1:
            raise DataValidationError(f"batch_size must be >= 1, got {batch_size}")
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)
        self._order = (
            np.random.default_rng(seed).permutation(len(images)) if shuffle else None
        )

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        """The next ``(images_batch, labels_batch)`` slice."""
        if self._order is None:
            images = self.images[self.index : self.index + self.batch_size]
            labels = self.labels[self.index : self.index + self.batch_size]
        else:
            take = self._order[self.index : self.index + self.batch_size]
            images = self.images[take]
            labels = self.labels[take]
        self.index += self.batch_size
        if self.index >= len(self.images):
            self.index = 0
        return images, labels


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


@dataclass
class ScratchHistory:
    """Everything the scratch training loop observed, per epoch."""

    task: str
    epochs_run: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_metric: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    learning_rate: float = 0.01
    batch_size: int = 32
    n_parameters: int = 0
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None
    model: NaiveSequential | None = None
    val_x: np.ndarray | None = None
    val_y: np.ndarray | None = None

    @property
    def metric_name(self) -> str:
        return "accuracy" if self.task == "classification" else "R²"

    def final_line(self) -> str:
        """One verdict line for the report card."""
        tail = (
            f" · val loss {self.val_loss[-1]:.4f} · val {self.metric_name} "
            f"{self.val_metric[-1]:.3f}"
            if self.val_loss
            else ""
        )
        return (
            f"{self.epochs_run} epoch(s) · train loss {self.train_loss[-1]:.4f} · "
            f"train {self.metric_name} {self.train_metric[-1]:.3f}{tail} · "
            f"{self.n_parameters:,} parameters"
        )


def _as_xy(x: Any, y: Any, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate and shape-check the fit inputs."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float).ravel()
    if x_values.ndim != 2:
        raise DataValidationError(f"x must be 2-D (n, features), got shape {x_values.shape}")
    if x_values.shape[0] != y_values.shape[0]:
        raise DataValidationError(
            f"x and y disagree ({x_values.shape[0]} vs {y_values.shape[0]} rows)"
        )
    if x_values.shape[0] < 2:
        raise DataValidationError("the scratch trainer needs at least two rows")
    if not np.isfinite(x_values).all():
        raise DataValidationError("x contains NaN/inf -- clean the table first")
    if task == "classification":
        classes = np.unique(y_values[np.isfinite(y_values)])
        if classes.size != 2 or not set(classes.tolist()) <= {0.0, 1.0}:
            raise DataValidationError(
                "classification needs a 0/1 target (two classes) -- "
                "regression accepts anything numeric"
            )
    elif not np.isfinite(y_values).all():
        raise DataValidationError("y contains NaN/inf")
    return x_values, y_values


def _forward_metric(task: str, pred: np.ndarray, y: np.ndarray) -> float:
    if task == "classification":
        return float(np.mean((pred.ravel() > 0.5) == (y > 0.5)))
    ss_res = float(np.sum((pred.ravel() - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _bce(pred: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(pred.ravel(), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred.ravel() - y) ** 2))


def _split_rows(n_rows: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Seeded chronological split (row order kept -- the split only cuts)."""
    if not 0.0 <= val_fraction < 0.5:
        raise DataValidationError(f"val_fraction must be in [0, 0.5), got {val_fraction}")
    n_val = round(n_rows * val_fraction)
    n_train = n_rows - n_val
    if n_train < 1 or (n_val and n_val < 1):
        raise DataValidationError("the split left an empty side -- load more rows")
    return np.arange(n_train), np.arange(n_train, n_rows)


def fit_scratch(
    model: NaiveSequential,
    x: Any,
    y: Any,
    *,
    task: str = "classification",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    val_fraction: float = 0.2,
    seed: int = RANDOM_STATE,
    standardize: bool = True,
    progress: Callable[[int, int | None, str], None] | None = None,
) -> ScratchHistory:
    """Chapter 2's full training loop, mouse-free edition.

    Batch generator → forward → loss → backward → SGD update, once per
    batch, once per epoch; the epoch's aggregate train loss and metric
    (accuracy for the binary task, R² for regression) land in the
    history together with the validation side.  The inputs are
    standardised per feature when ``standardize`` (the means/scales
    ride on the returned :class:`ScratchHistory` so
    :func:`predict_scratch` applies the identical transform) -- batch
    gradient descent on raw descriptors is the divergence lesson the
    *Data Science For Beginners* course teaches before scaling.

    The classification head must end in a sigmoid layer (the fused
    ``δ = (p − y)/B`` gradient assumes it); the regression head must be
    linear.

    Raises
    ------
    DataValidationError
        On shape/task/head mismatches or out-of-range hyperparameters.
    """
    if task not in SCRATCH_TASKS:
        raise DataValidationError(f"unknown task {task!r} (one of {SCRATCH_TASKS})")
    if epochs < 1:
        raise DataValidationError(f"epochs must be >= 1, got {epochs}")
    if not 0.0 < learning_rate <= 10.0:
        raise DataValidationError(f"learning_rate must be in (0, 10], got {learning_rate}")
    x_values, y_values = _as_xy(x, y, task)
    if not model.layers:
        raise DataValidationError("the model has no layers")
    head = model.layers[-1]
    if task == "classification" and head.activation != "sigmoid":
        raise DataValidationError(
            "classification needs a sigmoid output layer (the fused BCE gradient)"
        )
    if task == "regression" and head.activation != "linear":
        raise DataValidationError("regression needs a linear output layer")

    x_mean = x_values.mean(axis=0) if standardize else np.zeros(x_values.shape[1])
    x_std = x_values.std(axis=0) if standardize else np.ones(x_values.shape[1])
    x_std = np.where(x_std > 0, x_std, 1.0)
    x_scaled = (x_values - x_mean) / x_std

    train_idx, val_idx = _split_rows(len(x_scaled), val_fraction)
    x_train, y_train = x_scaled[train_idx], y_values[train_idx]
    x_val, y_val = (
        (x_scaled[val_idx], y_values[val_idx]) if val_idx.size else (None, None)
    )

    history = ScratchHistory(
        task=task,
        epochs_run=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_parameters=model.n_parameters,
        x_mean=x_mean,
        x_std=x_std,
        model=model,
        val_x=x_val,
        val_y=y_val,
    )
    generator = BatchGenerator(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for _ in range(generator.num_batches):
            x_batch, y_batch = generator.next()
            pred = model(x_batch)
            if task == "classification":
                epoch_loss += _bce(pred, y_batch) * len(x_batch)
                delta = (pred - y_batch.reshape(-1, 1)) / len(x_batch)
                model.backward_fused_head(delta)
            else:
                epoch_loss += _mse(pred, y_batch) * len(x_batch)
                grad = 2.0 * (pred - y_batch.reshape(-1, 1)) / len(x_batch)
                model.backward(grad)
            model.apply_gradients(learning_rate)
        train_loss = epoch_loss / len(x_train)
        train_pred = model(x_train)
        history.train_loss.append(train_loss)
        history.train_metric.append(_forward_metric(task, train_pred, y_train))
        if x_val is not None and y_val is not None:
            val_pred = model(x_val)
            history.val_loss.append(
                _bce(val_pred, y_val) if task == "classification" else _mse(val_pred, y_val)
            )
            history.val_metric.append(_forward_metric(task, val_pred, y_val))
        history.epochs_run = epoch
        if progress is not None:
            progress(epoch, epochs, f"epoch {epoch}/{epochs} · loss {train_loss:.4f}")
    return history


def predict_scratch(history: ScratchHistory, x: Any) -> np.ndarray:
    """Predict with a trained scratch model (standardisation reapplied)."""
    if history.model is None or history.x_mean is None or history.x_std is None:
        raise DataValidationError("the history carries no trained model")
    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 2:
        raise DataValidationError(f"x must be 2-D, got shape {x_values.shape}")
    if x_values.shape[1] != len(history.x_mean):
        raise DataValidationError(
            f"x has {x_values.shape[1]} features, the model was fit on {len(history.x_mean)}"
        )
    return history.model((x_values - history.x_mean) / history.x_std).ravel()


# ---------------------------------------------------------------------------
# Data Science For Beginners -- gradient-descent linear regression
# ---------------------------------------------------------------------------


@dataclass
class LinearGDResult:
    """Outcome of the from-scratch linear regression.

    ``coefficients`` and ``intercept`` live in the ORIGINAL feature
    space (the standardisation the descent ran on is undone), so
    ``x @ coefficients + intercept`` predicts raw targets.
    """

    coefficients: np.ndarray
    intercept: float
    mse_history: list[float]
    learning_rate: float
    epochs_run: int
    r2: float

    def final_line(self) -> str:
        terms = " + ".join(
            f"{coef:.4g}·x{i}" for i, coef in enumerate(self.coefficients, start=1)
        )
        return (
            f"y = {self.intercept:.4g} + {terms} · GD {self.epochs_run} epochs @ "
            f"lr {self.learning_rate} · final MSE {self.mse_history[-1]:.4g} · "
            f"R² {self.r2:.3f}"
        )


def linear_regression_gd(
    x: Any,
    y: Any,
    *,
    learning_rate: float = 0.05,
    epochs: int = 200,
    seed: int = RANDOM_STATE,
) -> LinearGDResult:
    """Batch gradient descent on the squared error (DSBC's centrepiece).

    The descent itself is the course's three-liner -- ``w ← w − lr ·
    ∇(‖Xw + b − y‖²)/n`` -- run on standardised features (the course's
    own scaling lecture: unscaled descriptors make the bowls
    ill-conditioned valleys).  The returned coefficients are translated
    back to the raw feature space, and the closed-form least-squares
    solution is logged alongside as the convergence cross-check.

    Raises
    ------
    DataValidationError
        On shape mismatches, NaN input or out-of-range hyperparameters.
    """
    if not 0.0 < learning_rate <= 10.0:
        raise DataValidationError(f"learning_rate must be in (0, 10], got {learning_rate}")
    if epochs < 1:
        raise DataValidationError(f"epochs must be >= 1, got {epochs}")
    x_values, y_values = _as_xy(x, y, "regression")
    rng = np.random.default_rng(seed)
    x_mean = x_values.mean(axis=0)
    x_std = np.where(x_values.std(axis=0) > 0, x_values.std(axis=0), 1.0)
    x_scaled = (x_values - x_mean) / x_std
    n_rows, n_features = x_scaled.shape
    w = rng.normal(scale=0.01, size=n_features)
    b = 0.0
    mse_history: list[float] = []
    for _ in range(epochs):
        residual = x_scaled @ w + b - y_values
        grad_w = 2.0 * (x_scaled.T @ residual) / n_rows
        grad_b = 2.0 * float(residual.mean())
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        mse_history.append(float(np.mean(residual**2)))
    # translate back: y = Σ wᵢ(xᵢ - μᵢ)/σᵢ + b = Σ (wᵢ/σᵢ)xᵢ + (b - Σ wᵢμᵢ/σᵢ)
    coefficients = w / x_std
    intercept = float(b - np.sum(w * x_mean / x_std))
    predictions = x_values @ coefficients + intercept
    ss_tot = float(np.sum((y_values - y_values.mean()) ** 2))
    r2 = 1.0 - float(np.sum((predictions - y_values) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    try:  # the closed-form cross-check (logged, never fatal)
        least_squares, *_ = np.linalg.lstsq(
            np.column_stack([x_values, np.ones(n_rows)]), y_values, rcond=None
        )
        _LOG.debug(
            "linear_regression_gd: GD intercept %.4g vs lstsq %.4g",
            intercept,
            least_squares[-1],
        )
    except np.linalg.LinAlgError:  # pragma: no cover - lstsq on degenerate X
        pass
    return LinearGDResult(
        coefficients=coefficients,
        intercept=intercept,
        mse_history=mse_history,
        learning_rate=learning_rate,
        epochs_run=epochs,
        r2=r2,
    )


# ---------------------------------------------------------------------------
# Chapter 5 -- the capacity sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityPoint:
    """One (width, train loss, validation loss) sample of the sweep."""

    hidden_width: int
    n_parameters: int
    train_loss: float
    val_loss: float | None
    train_metric: float
    val_metric: float | None

    def as_row(self) -> list[str]:
        val = "—" if self.val_loss is None else f"{self.val_loss:.4f}"
        val_metric = "—" if self.val_metric is None else f"{self.val_metric:.3f}"
        return [
            str(self.hidden_width) if self.hidden_width else "linear",
            f"{self.n_parameters:,}",
            f"{self.train_loss:.4f}",
            val,
            f"{self.train_metric:.3f}",
            val_metric,
        ]


@dataclass
class CapacitySweepResult:
    """Chapter 5's "reducing the network's size" figure, as data.

    ``best_width`` is the validation-loss minimiser (``0`` = the linear
    baseline won -- underfitting territory); ``verdict`` says which
    regime the curve ended in.
    """

    task: str
    points: list[CapacityPoint]
    histories: dict[int, ScratchHistory]
    best_width: int
    verdict: str

    def table_rows(self) -> list[list[str]]:
        return [point.as_row() for point in self.points]


def capacity_sweep(
    x: Any,
    y: Any,
    *,
    task: str = "classification",
    widths: tuple[int, ...] = (0, 4, 8, 16, 32, 64),
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    val_fraction: float = 0.25,
    activation: str = "relu",
    seed: int = RANDOM_STATE,
    progress: Callable[[int, int | None, str], None] | None = None,
) -> CapacitySweepResult:
    """Train the same data through growing hidden widths (chapter 5).

    Width ``0`` is the linear baseline (no hidden layer -- the size
    chapter 5 starts from); every other width builds
    ``Dense(width, activation) → Dense(out, head)`` with the SAME seed,
    split and standardisation so the only moving part is capacity.
    The verdict line reads the classic shape: validation loss falling
    then rising = overfitting on the right; never falling = everything
    underfits; flat parity = capacity stopped mattering.

    Raises
    ------
    DataValidationError
        Via :func:`fit_scratch` (bad task/target/hyperparameters), or
        on duplicate/negative widths.
    """
    if not widths:
        raise DataValidationError("widths must not be empty")
    if any(width < 0 for width in widths):
        raise DataValidationError(f"widths must be >= 0, got {widths}")
    if len(set(widths)) != len(widths):
        raise DataValidationError("widths must be unique")
    x_values, y_values = _as_xy(x, y, task)
    n_features = x_values.shape[1]
    points: list[CapacityPoint] = []
    histories: dict[int, ScratchHistory] = {}
    for position, width in enumerate(widths, start=1):
        if width == 0:
            layers = [NaiveDense(n_features, 1, "linear" if task == "regression" else "sigmoid", seed=seed)]
        else:
            layers = [
                NaiveDense(n_features, width, activation, seed=seed),
                NaiveDense(width, 1, "linear" if task == "regression" else "sigmoid", seed=seed + 1),
            ]
        model = NaiveSequential(layers)
        history = fit_scratch(
            model,
            x_values,
            y_values,
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_fraction=val_fraction,
            seed=seed,
        )
        histories[width] = history
        points.append(
            CapacityPoint(
                hidden_width=width,
                n_parameters=history.n_parameters,
                train_loss=history.train_loss[-1],
                val_loss=history.val_loss[-1] if history.val_loss else None,
                train_metric=history.train_metric[-1],
                val_metric=history.val_metric[-1] if history.val_metric else None,
            )
        )
        if progress is not None:
            if history.val_loss:
                note = f"width {width or 'linear'} · val loss {history.val_loss[-1]:.4f}"
            else:
                note = f"width {width or 'linear'} done"
            progress(position, len(widths), note)
    scored = [(point.val_loss, point.hidden_width) for point in points if point.val_loss is not None]
    best_width = min(scored)[1] if scored else widths[0]
    if scored:
        val_values = [value for value, _ in scored]
        best_index = val_values.index(min(val_values))
        rising_after = best_index < len(val_values) - 1 and val_values[-1] > min(val_values) * 1.10
        if rising_after:
            verdict = (
                f"overfitting regime: validation loss bottoms at width {best_width} "
                "then climbs -- chapter 5's right-hand side"
            )
        elif max(val_values) - min(val_values) <= 1e-9:
            verdict = "capacity stopped mattering -- every width validates the same"
        else:
            verdict = f"no overfitting yet: width {best_width} wins and the curve stays flat"
    else:
        verdict = "no validation split -- the sweep compared train loss only"
    return CapacitySweepResult(
        task=task,
        points=points,
        histories=histories,
        best_width=best_width,
        verdict=verdict,
    )


__all__ = [
    "RANDOM_STATE",
    "SCRATCH_ACTIVATIONS",
    "SCRATCH_TASKS",
    "BatchGenerator",
    "BroadcastReport",
    "CapacityPoint",
    "CapacitySweepResult",
    "LinearGDResult",
    "NaiveDense",
    "NaiveSequential",
    "PrimerReport",
    "ScratchHistory",
    "TensorSummary",
    "broadcast_report",
    "capacity_sweep",
    "describe_tensor",
    "fit_scratch",
    "linear_regression_gd",
    "predict_scratch",
    "tensor_primer_report",
]

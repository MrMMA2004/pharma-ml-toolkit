r"""PyTorch's autograd concept, absorbed natively: the reverse tape.

``pytorch/pytorch`` (reviewed in this round's original pass and
re-derived here from its published core) is above all a
*differentiable tensor library*: values carry their computation
history, ``backward()`` walks it in reverse topological order, and
the optimisers/NN containers sit on top of that one idea.  The CUDA
kernels, the distributed launcher, the C++ dispatcher graph and the
``.pth`` serialization are exactly what this toolkit does not
vendor — the *mathematics* of autodiff transfers natively:

* :class:`Tensor` — a NumPy-backed value with ``requires_grad``,
  ``grad``, ``backward()`` and the operator set
  (``+ - * / ** @`` unary functions, indexing, reshape/transpose,
  sum/mean) whose every node knows how to push gradients to its
  parents — including **broadcasting-aware** gradients (an
  ``(3, 1)`` + ``(4,)`` addition sums the incoming gradient back
  over the broadcast axes, exactly like ``expand``-reduction).
* :func:`no_grad` / ``Tensor.detach`` — the inference-mode switch
  and the graph cut.
* :func:`gradient_check` — central finite differences against the
  reverse pass, the same contract ``torch.autograd.gradcheck``
  enforces, used by every doctest below.
* The optimiser family — :class:`SGD` (momentum, Nesterov, weight
  decay), :class:`Adagrad`, :class:`RMSprop`, :class:`Adam` /
  :class:`AdamW` (bias correction, optional AMSGrad) — operating on
  parameter lists through the exact ``param.grad`` / ``step()`` /
  ``zero_grad()`` protocol PyTorch standardised.
* The ``torch.nn`` layer — :class:`Parameter`, :class:`Module`
  (recursive ``parameters()``), :class:`Linear` (PyTorch's default
  Kaiming-uniform ``1/sqrt(fan_in)`` bound init), :class:`Sequential`
  and the stable losses (:func:`mse_loss`, :func:`bce_with_logits`,
  :func:`cross_entropy` with log-softmax).

The honest scope, stated once: CPU float64, eager, no graph
checkpointing (the tape is re-built every forward — retraining a
node's parents mid-``backward`` is undefined, as it is in PyTorch's
default mode), no ``torch.compile``/vmap/persistent buffers, and a
teaching-scale training loop instead of the DataParallel fleet.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.autograd import (
...     Adam, Linear, MSELoss, Module, Sequential, SGD, Tensor,
...     gradient_check, no_grad,
... )
>>> x = Tensor(np.array([[0.0], [1.0], [2.0]]), requires_grad=True)
>>> y = (x * x).sum()
>>> y.backward()
>>> np.allclose(x.grad, np.array([[0.0], [2.0], [4.0]]))
True
>>> def f(a):
...     return ((a * 3.0 - 1.0) ** 2).sum()
>>> bool(gradient_check(f, np.array([1.0, -2.0, 0.5])) < 1e-8)
True
>>> with no_grad():
...     z = x * 2
>>> bool(z.requires_grad)
False
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError, ModelError

__all__ = [
    "SGD",
    "Adagrad",
    "Adam",
    "AdamW",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "Linear",
    "MSELoss",
    "Module",
    "Parameter",
    "RMSprop",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tensor",
    "bce_with_logits",
    "cross_entropy",
    "gradient_check",
    "log_softmax",
    "mse_loss",
    "no_grad",
    "sigmoid",
    "softmax",
    "tanh",
]


# --------------------------------------------------------------------------
# the tensor
# --------------------------------------------------------------------------


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` over the axes a broadcast added (PyTorch's expand-reduce)."""

    grad = np.asarray(grad, dtype=float)
    if grad.shape == shape:
        return grad
    # leading axes added entirely
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # trailing axes of size 1 that were expanded
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A differentiable value: NumPy data plus reverse-mode history.

    Parameters
    ----------
    data : array-like
        The value (copied to a float ``np.ndarray``).
    requires_grad : bool
        Whether this value participates in / accumulates gradients.

    Attributes
    ----------
    grad : np.ndarray | None
        Accumulated gradient (``None`` until a ``backward()`` touches
        this leaf; accumulation is additive, PyTorch-style).
    """

    _grad_enabled: bool = True

    def __init__(self, data: object, requires_grad: bool = False) -> None:
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim == 0:
            self.data = self.data.reshape(1)
        self.requires_grad = requires_grad and Tensor._grad_enabled
        self.grad: np.ndarray | None = None
        self._parents: list[Tensor] = []
        self._grad_fns: list[Callable[[np.ndarray], np.ndarray]] = []
        self.name: str = ""
        if self.data.dtype == object:
            raise DataValidationError("Tensor data must be numeric")

    # -- basic protocol ---------------------------------------------------- #

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        grad_flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{grad_flag})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:  # identity semantics for graph code
        return self is other

    def item(self) -> float:
        if self.data.size != 1:
            raise DataValidationError(f"item() on a {self.data.shape} tensor")
        return float(self.data.reshape(()))

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def detach(self) -> Tensor:
        """The same value, cut from the graph."""

        return Tensor(self.data.copy())

    def zero_grad(self) -> None:
        self.grad = None

    # -- graph plumbing ---------------------------------------------------- #

    @classmethod
    def _from_op(
        cls,
        data: np.ndarray,
        parents: Sequence[Tensor],
        grad_fns: Sequence[Callable[[np.ndarray], np.ndarray]],
    ) -> Tensor:
        out = cls(data)
        if Tensor._grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = list(parents)
            out._grad_fns = list(grad_fns)
        return out

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Backpropagate from this value (the tape entry point)."""

        if not self.requires_grad:
            raise ModelError("backward() on a tensor that does not require grad")
        seed = (
            np.ones_like(self.data)
            if grad is None
            else np.asarray(grad, dtype=float).reshape(self.data.shape)
        )
        order: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited or not node.requires_grad:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        flow: dict[int, np.ndarray] = {id(self): seed}
        for node in reversed(order):
            incoming = flow.get(id(node))
            if incoming is None:
                continue
            if node._parents:
                for parent, fn in zip(node._parents, node._grad_fns):
                    contribution = fn(incoming)
                    if contribution is None:
                        continue
                    contribution = _unbroadcast(
                        np.asarray(contribution, dtype=float), parent.shape
                    )
                    flow[id(parent)] = (
                        flow[id(parent)] + contribution
                        if id(parent) in flow
                        else contribution
                    )
            else:  # leaf
                node.grad = incoming if node.grad is None else node.grad + incoming

    # -- arithmetic --------------------------------------------------------- #

    def _wrap(self, other: object) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=float))

    def __add__(self, other: object) -> Tensor:
        rhs = self._wrap(other)
        data = self.data + rhs.data

        def dx(g: np.ndarray) -> np.ndarray:
            return g

        return Tensor._from_op(data, [self, rhs], [dx, dx])

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        data = -self.data
        return Tensor._from_op(data, [self], [lambda g: -g])

    def __sub__(self, other: object) -> Tensor:
        rhs = self._wrap(other)
        data = self.data - rhs.data
        return Tensor._from_op(data, [self, rhs], [lambda g: g, lambda g: -g])

    def __rsub__(self, other: object) -> Tensor:
        return self._wrap(other).__sub__(self)

    def __mul__(self, other: object) -> Tensor:
        rhs = self._wrap(other)
        data = self.data * rhs.data
        return Tensor._from_op(
            data, [self, rhs], [lambda g: g * rhs.data, lambda g: g * self.data]
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Tensor:
        rhs = self._wrap(other)
        data = self.data / rhs.data
        return Tensor._from_op(
            data,
            [self, rhs],
            [lambda g: g / rhs.data, lambda g: -g * self.data / (rhs.data**2)],
        )

    def __rtruediv__(self, other: object) -> Tensor:
        return self._wrap(other).__truediv__(self)

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor):
            raise DataValidationError("Tensor ** Tensor is not supported (scalar exponents only)")
        data = self.data**exponent
        return Tensor._from_op(
            data, [self], [lambda g: g * exponent * self.data ** (exponent - 1)]
        )

    def __matmul__(self, other: object) -> Tensor:
        rhs = self._wrap(other)
        left, right = self.data, rhs.data
        if left.ndim == 1 and right.ndim == 1:
            data = np.array([float(left @ right)])

            def dl(g: np.ndarray) -> np.ndarray:
                return np.asarray(g).reshape(-1)[0] * right

            def dr(g: np.ndarray) -> np.ndarray:
                return np.asarray(g).reshape(-1)[0] * left

            return Tensor._from_op(data, [self, rhs], [dl, dr])
        if left.ndim == 2 and right.ndim == 1:
            data = left @ right

            def dl2(g: np.ndarray) -> np.ndarray:
                return np.outer(np.asarray(g).reshape(-1), right)

            def dr2(g: np.ndarray) -> np.ndarray:
                return left.T @ np.asarray(g).reshape(-1)

            return Tensor._from_op(data, [self, rhs], [dl2, dr2])
        if left.ndim == 1 and right.ndim == 2:
            data = left @ right

            def dl1(g: np.ndarray) -> np.ndarray:
                return np.asarray(g).reshape(-1) @ right.T

            def dr1(g: np.ndarray) -> np.ndarray:
                return np.outer(left, np.asarray(g).reshape(-1))

            return Tensor._from_op(data, [self, rhs], [dl1, dr1])
        data = left @ right
        return Tensor._from_op(
            data, [self, rhs], [lambda g: g @ right.T, lambda g: left.T @ g]
        )

    # -- elementwise functions ------------------------------------------------ #

    def exp(self) -> Tensor:
        data = np.exp(self.data)
        return Tensor._from_op(data, [self], [lambda g: g * data])

    def log(self) -> Tensor:
        if np.any(self.data <= 0):
            raise DataValidationError("log of a non-positive value")
        data = np.log(self.data)
        return Tensor._from_op(data, [self], [lambda g: g / self.data])

    def sqrt(self) -> Tensor:
        if np.any(self.data < 0):
            raise DataValidationError("sqrt of a negative value")
        data = np.sqrt(self.data)
        return Tensor._from_op(data, [self], [lambda g: g / (2.0 * data)])

    def abs(self) -> Tensor:
        data = np.abs(self.data)
        return Tensor._from_op(data, [self], [lambda g: g * np.sign(self.data)])

    def sin(self) -> Tensor:
        return Tensor._from_op(
            np.sin(self.data), [self], [lambda g: g * np.cos(self.data)]
        )

    def cos(self) -> Tensor:
        return Tensor._from_op(
            np.cos(self.data), [self], [lambda g: -g * np.sin(self.data)]
        )

    def tanh(self) -> Tensor:
        data = np.tanh(self.data)
        return Tensor._from_op(data, [self], [lambda g: g * (1.0 - data * data)])

    def sigmoid(self) -> Tensor:
        data = 1.0 / (1.0 + np.exp(-self.data))
        return Tensor._from_op(data, [self], [lambda g: g * data * (1.0 - data)])

    def relu(self) -> Tensor:
        data = np.maximum(self.data, 0.0)
        return Tensor._from_op(data, [self], [lambda g: g * (self.data > 0)])

    def leaky_relu(self, slope: float = 0.01) -> Tensor:
        data = np.where(self.data > 0, self.data, slope * self.data)
        gate = np.where(self.data > 0, 1.0, slope)
        return Tensor._from_op(data, [self], [lambda g: g * gate])

    # -- shape / reductions ----------------------------------------------------- #

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        data = self.data.sum(axis=axis, keepdims=keepdims)
        in_shape = self.data.shape

        def dx(g: np.ndarray) -> np.ndarray:
            grad = np.asarray(g, dtype=float)
            if axis is None:
                return np.broadcast_to(grad, in_shape).astype(float) * np.ones(in_shape)
            expanded = (
                grad
                if keepdims
                else np.expand_dims(grad, axis=axis if isinstance(axis, int) else tuple(axis))
            )
            return np.broadcast_to(expanded, in_shape).astype(float) * np.ones(in_shape)

        return Tensor._from_op(data, [self], [dx])

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        n = float(self.data.size) if axis is None else float(
            np.prod([self.data.shape[a] for a in np.atleast_1d(axis).tolist()])
        )
        summed = self.sum(axis=axis, keepdims=keepdims)
        return summed * (1.0 / n)

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        target: tuple[int, ...]
        if len(shape) == 1 and isinstance(shape[0], tuple):
            target = tuple(int(s) for s in shape[0] if not isinstance(s, tuple))
        else:
            target = tuple(int(s) for s in shape if not isinstance(s, tuple))
        old_shape = self.data.shape
        data = self.data.reshape(target)
        return Tensor._from_op(
            data, [self], [lambda g: np.asarray(g).reshape(old_shape)]
        )

    def transpose(self, *axes: int) -> Tensor:
        permutation = tuple(axes) if axes else tuple(reversed(range(self.data.ndim)))
        data = self.data.transpose(permutation)
        inverse = tuple(np.argsort(permutation))
        return Tensor._from_op(
            data, [self], [lambda g: np.asarray(g).transpose(inverse)]
        )

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __getitem__(self, key: object) -> Tensor:
        if not isinstance(key, tuple):
            key = (key,)
        if any(isinstance(k, Tensor) for k in key):
            raise DataValidationError("advanced indexing is not supported")
        old_shape = self.data.shape
        data = self.data[key]

        def dx(g: np.ndarray) -> np.ndarray:
            full = np.zeros(old_shape, dtype=float)
            np.add.at(full, key, np.asarray(g, dtype=float))
            return full

        return Tensor._from_op(data, [self], [dx])


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction (the inference-mode switch)."""

    previous = Tensor._grad_enabled
    Tensor._grad_enabled = False
    try:
        yield
    finally:
        Tensor._grad_enabled = previous


# module-level functional aliases ------------------------------------------------ #


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Row-stable softmax (subtract-max)."""

    shifted = x - Tensor(np.max(x.data, axis=axis, keepdims=True))
    exp = shifted.exp()
    return exp * (1.0 / exp.sum(axis=axis, keepdims=True))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log-softmax, the stable way (x − logsumexp)."""

    shifted = x - Tensor(np.max(x.data, axis=axis, keepdims=True))
    exp = shifted.exp()
    return shifted - exp.sum(axis=axis, keepdims=True).log()


def scatter_add(src: Tensor, index: object, size: int, dim: int = 0) -> Tensor:
    """Segment-sum aggregation — ``torch_geometric.utils.scatter(reduce="add")``.

    ``index`` holds the destination segment of every ``src`` row along
    ``dim`` (0 only — the message-passing axis); the result has ``size``
    rows.  The backward is the gather ``g[index]`` — exactly the
    transpose of the forward's ``np.add.at``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.autograd import Tensor, gradient_check, scatter_add
    >>> out = scatter_add(Tensor([[1.0], [2.0], [3.0]]), np.array([0, 2, 2]), 4)
    >>> out.numpy().ravel().tolist()
    [1.0, 0.0, 5.0, 0.0]
    >>> check = gradient_check(
    ...     lambda t: scatter_add(t, np.array([0, 1, 1]), 2).sum() ** 2,
    ...     np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -0.5]]),
    ... )
    >>> bool(check < 1e-7)
    True
    """

    if dim != 0:
        raise DataValidationError("scatter_*: only the node axis (dim=0) is supported")
    idx = np.asarray(index, dtype=int).reshape(-1)
    if idx.shape[0] != src.data.shape[0]:
        raise DataValidationError("scatter_add: index must name every src row")
    out = np.zeros((size,) + tuple(src.data.shape[1:]), dtype=float)
    np.add.at(out, idx, src.data)

    def dx(g: np.ndarray) -> np.ndarray:
        return np.asarray(g, dtype=float)[idx]

    return Tensor._from_op(out, [src], [dx])


def scatter_max(src: Tensor, index: object, size: int, dim: int = 0) -> Tensor:
    """Segment-max aggregation, first-occurrence tie-break.

    The gradient routes to the FIRST row achieving each segment's max
    (``torch_geometric.utils.scatter(reduce="max")``'s convention) and
    is zero elsewhere — the forward's ``np.maximum.at`` keeps cells the
    backward cannot split.  Segments with no row stay ``-inf``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.autograd import Tensor, gradient_check, scatter_max
    >>> out = scatter_max(Tensor([[1.0], [4.0], [4.0], [2.0]]), np.array([0, 2, 2, 2]), 3)
    >>> out.numpy().ravel().tolist()
    [1.0, -inf, 4.0]
    >>> check = gradient_check(
    ...     lambda t: scatter_max(t, np.array([0, 1, 1, 0]), 2).sum(),
    ...     np.array([[3.0], [1.0], [2.0], [1.5]]),
    ... )
    >>> bool(check < 1e-8)
    True
    """

    if dim != 0:
        raise DataValidationError("scatter_*: only the node axis (dim=0) is supported")
    idx = np.asarray(index, dtype=int).reshape(-1)
    if idx.shape[0] != src.data.shape[0]:
        raise DataValidationError("scatter_max: index must name every src row")
    shape = (size,) + tuple(src.data.shape[1:])
    out = np.full(shape, -np.inf)
    np.maximum.at(out, idx, src.data)
    claims = src.data == out[idx]
    keep = np.zeros_like(claims)
    claimed = np.zeros(shape, dtype=bool)
    for row in range(idx.shape[0]):  # first occurrence per cell wins
        take = claims[row] & ~claimed[idx[row]]
        keep[row] = take
        claimed[idx[row]] |= take

    def dx(g: np.ndarray) -> np.ndarray:
        return np.asarray(g, dtype=float)[idx] * keep

    return Tensor._from_op(out, [src], [dx])


def scatter_softmax(src: Tensor, index: object, size: int, dim: int = 0) -> Tensor:
    """Per-segment softmax — ``torch_geometric.utils.softmax``.

    The segment max is subtracted for stability and DETACHED (PyG's
    own choice): the max's gradient routes through the exponential
    path, not through the comparison.  Composed entirely of tape ops,
    so the backward is automatic.  Requires every named segment to be
    non-empty (self-loops guarantee that in the message-passing use).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.autograd import Tensor, gradient_check, scatter_softmax
    >>> out = scatter_softmax(Tensor([[0.0], [1.0], [2.0]]), np.array([0, 1, 1]), 2)
    >>> out.numpy().ravel().round(3).tolist()
    [1.0, 0.269, 0.731]
    >>> check = gradient_check(
    ...     lambda t: (scatter_softmax(t, np.array([0, 0, 1, 1]), 2)
    ...                * Tensor(np.array([[0.5], [1.5], [2.5], [3.5]]))).sum(),
    ...     np.array([[0.3], [-0.2], [1.1], [0.7]]),
    ... )
    >>> bool(check < 1e-8)
    True
    """

    if dim != 0:
        raise DataValidationError("scatter_*: only the node axis (dim=0) is supported")
    idx = np.asarray(index, dtype=int).reshape(-1)
    if idx.shape[0] != src.data.shape[0]:
        raise DataValidationError("scatter_softmax: index must name every src row")
    m = np.full((size,) + tuple(src.data.shape[1:]), -np.inf)
    np.maximum.at(m, idx, src.data)
    shifted = src - Tensor(m[idx])
    weights = shifted.exp()
    denom = scatter_add(weights, idx, size)
    return weights / denom[idx]


def gradient_check(
    function: Callable[[Tensor], Tensor],
    x: object,
    *,
    eps: float = 1e-6,
) -> float:
    """Central-difference check of ``function``'s reverse-mode gradient.

    The ``torch.autograd.gradcheck`` contract: the analytic gradient
    at ``x`` must match ``(f(x+εe_i) − f(x−εe_i)) / 2ε`` for every
    coordinate.  Returns the maximum absolute deviation (float64
    gives ~1e-9 for well-conditioned functions).

    Parameters
    ----------
    function : callable
        Maps one :class:`Tensor` to a scalar :class:`Tensor`.
    x : array-like
        The point to check at (made a leaf tensor inside).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.autograd import Tensor, gradient_check
    >>> def f(a):
    ...     return ((a * a).sum() - (a * 3.0).sum()) * 2.0
    >>> bool(gradient_check(f, np.array([1.0, -1.0, 2.0])) < 1e-8)
    True
    """

    array = np.asarray(x, dtype=float)
    leaf = Tensor(array.copy(), requires_grad=True)
    out = function(leaf)
    if out.data.size != 1:
        raise DataValidationError("gradient_check: the function must return a scalar")
    out.backward()
    analytic = np.zeros_like(array)
    analytic[...] = leaf.grad if leaf.grad is not None else 0.0
    numeric = np.zeros_like(array)
    flat = array.reshape(-1)
    numeric_flat = numeric.reshape(-1)
    for i in range(flat.size):
        bumped = array.copy()
        bumped.reshape(-1)[i] += eps
        high = function(Tensor(bumped)).item()
        bumped.reshape(-1)[i] -= 2 * eps
        low = function(Tensor(bumped)).item()
        numeric_flat[i] = (high - low) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


# --------------------------------------------------------------------------
# the optimisers (the exact PyTorch update formulas)
# --------------------------------------------------------------------------


class Optimizer:
    """The ``param.grad`` / ``step()`` / ``zero_grad()`` protocol."""

    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params = [p for p in params if isinstance(p, Tensor)]
        if not self.params:
            raise ModelError("optimizer: no parameter tensors given")
        for param in self.params:
            if not param.requires_grad:
                param.requires_grad = True
        self._step_count = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        self._step_count += 1
        for param in self.params:
            if param.grad is None:
                continue
            self._update(param, np.asarray(param.grad, dtype=float))

    def _update(self, param: Tensor, grad: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with momentum, Nesterov and (L2) weight decay.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.autograd import SGD, Tensor
    >>> w = Tensor(np.array([5.0]), requires_grad=True)
    >>> opt = SGD([w], lr=0.1)
    >>> for _ in range(60):
    ...     opt.zero_grad()
    ...     loss = ((w - 1.0) ** 2).sum()
    ...     loss.backward()
    ...     opt.step()
    >>> float(round(w.item(), 3))
    1.0
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params)
        if lr <= 0:
            raise DataValidationError(f"SGD: lr must be positive, got {lr}")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def _update(self, param: Tensor, grad: np.ndarray) -> None:
        index = self.params.index(param)
        adjusted = grad + self.weight_decay * param.data
        if self.momentum:
            self._velocity[index] = self.momentum * self._velocity[index] + adjusted
            direction = (
                adjusted + self.momentum * self._velocity[index]
                if self.nesterov
                else self._velocity[index]
            )
        else:
            direction = adjusted
        param.data = param.data - self.lr * direction


class Adagrad(Optimizer):
    """Adagrad: the per-coordinate accumulator ``G += g²``."""

    def __init__(self, params: Iterable[Tensor], lr: float = 0.01, eps: float = 1e-10) -> None:
        super().__init__(params)
        self.lr, self.eps = lr, eps
        self._state = [np.zeros_like(p.data) for p in self.params]

    def _update(self, param: Tensor, grad: np.ndarray) -> None:
        index = self.params.index(param)
        self._state[index] = self._state[index] + grad * grad
        param.data = param.data - self.lr * grad / (
            np.sqrt(self._state[index]) + self.eps
        )


class RMSprop(Optimizer):
    """RMSprop: the running quadratic mean with ``alpha`` decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr, self.alpha, self.eps, self.weight_decay = lr, alpha, eps, weight_decay
        self._state = [np.zeros_like(p.data) for p in self.params]

    def _update(self, param: Tensor, grad: np.ndarray) -> None:
        index = self.params.index(param)
        adjusted = grad + self.weight_decay * param.data
        self._state[index] = self.alpha * self._state[index] + (1 - self.alpha) * (
            adjusted * adjusted
        )
        param.data = param.data - self.lr * adjusted / (
            np.sqrt(self._state[index]) + self.eps
        )


class Adam(Optimizer):
    """Adam with bias correction (and optional AMSGrad).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.autograd import Adam, Tensor
    >>> w = Tensor(np.array([8.0]), requires_grad=True)
    >>> opt = Adam([w], lr=0.2)
    >>> for _ in range(300):
    ...     opt.zero_grad()
    ...     ((w * w - 4.0) ** 2).sum().backward()
    ...     opt.step()
    >>> bool(abs(w.item() - 2.0) < 1e-2)
    True
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay, self.amsgrad = weight_decay, amsgrad
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._vmax = [np.zeros_like(p.data) for p in self.params]

    def _update(self, param: Tensor, grad: np.ndarray) -> None:
        index = self.params.index(param)
        beta1, beta2 = self.betas
        adjusted = grad + self.weight_decay * param.data
        self._m[index] = beta1 * self._m[index] + (1 - beta1) * adjusted
        self._v[index] = beta2 * self._v[index] + (1 - beta2) * (adjusted * adjusted)
        t = self._step_count
        m_hat = self._m[index] / (1 - beta1**t)
        v_hat = self._v[index] / (1 - beta2**t)
        if self.amsgrad:
            self._vmax[index] = np.maximum(self._vmax[index], v_hat)
            v_hat = self._vmax[index]
        param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (the Loshchilov-Hutter fix)."""

    def _update(self, param: Tensor, grad: np.ndarray) -> None:
        index = self.params.index(param)
        beta1, beta2 = self.betas
        self._m[index] = beta1 * self._m[index] + (1 - beta1) * grad
        self._v[index] = beta2 * self._v[index] + (1 - beta2) * (grad * grad)
        t = self._step_count
        m_hat = self._m[index] / (1 - beta1**t)
        v_hat = self._v[index] / (1 - beta2**t)
        if self.amsgrad:
            self._vmax[index] = np.maximum(self._vmax[index], v_hat)
            v_hat = self._vmax[index]
        param.data = (
            param.data
            - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            - self.lr * self.weight_decay * param.data
        )


# --------------------------------------------------------------------------
# the torch.nn layer
# --------------------------------------------------------------------------


class Parameter(Tensor):
    """A leaf tensor that always requires grad."""

    def __init__(self, data: object) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """The container protocol: recursive ``parameters()``, ``__call__``."""

    def parameters(self) -> list[Parameter]:
        found: list[Parameter] = []
        for value in vars(self).values():
            if isinstance(value, Parameter):
                found.append(value)
            elif isinstance(value, Module):
                found.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        found.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        found.append(item)
        return found

    def named_parameters(self) -> dict[str, Parameter]:
        found: dict[str, Parameter] = {}

        def walk(module: Module, prefix: str) -> None:
            for name, value in vars(module).items():
                if isinstance(value, Parameter):
                    found[f"{prefix}{name}"] = value
                elif isinstance(value, Module):
                    walk(value, f"{prefix}{name}.")
                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        if isinstance(item, Module):
                            walk(item, f"{prefix}{name}.{index}.")

        walk(self, "")
        return found

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class Linear(Module):
    """``y = x @ W.T + b`` with PyTorch's default init bounds.

    ``W ~ U(−1/√fan_in, +1/√fan_in)`` (the Kaiming-uniform ``a=√5``
    reduction) and the same bound for ``b``, seeded for
    reproducibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise DataValidationError("Linear: feature counts must be positive")
        rng = np.random.default_rng(seed)
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            rng.uniform(-bound, bound, size=(out_features, in_features))
        )
        self.bias = Parameter(rng.uniform(-bound, bound, size=(out_features,))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(Module):
    """Modules applied in order."""

    def __init__(self, *modules: Module) -> None:
        self.layers = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, self.axis)


# --------------------------------------------------------------------------
# the losses (stable formulations)
# --------------------------------------------------------------------------


def mse_loss(prediction: Tensor, target: object) -> Tensor:
    """Mean squared error (no gradient through the target)."""

    y = np.asarray(target, dtype=float)
    if y.shape != prediction.shape:
        raise DataValidationError(
            f"mse_loss: shape mismatch {prediction.shape} vs {y.shape}"
        )
    diff = prediction - y
    return (diff * diff).mean()


def bce_with_logits(logits: Tensor, target: object) -> Tensor:
    """Binary cross-entropy with logits, the numerically stable form."""

    y = np.asarray(target, dtype=float)
    if y.shape != logits.shape:
        raise DataValidationError(
            f"bce_with_logits: shape mismatch {logits.shape} vs {y.shape}"
        )
    positive = np.maximum(logits.data, 0.0)
    loss = (
        Tensor(positive - logits.data * y + np.log1p(np.exp(-np.abs(logits.data))))
    ).mean()
    return loss


def cross_entropy(logits: Tensor, labels: object) -> Tensor:
    """Softmax cross-entropy: ``mean(−log_softmax[i, label_i])``."""

    y = np.asarray(labels).reshape(-1)
    if logits.data.ndim != 2 or y.size != logits.data.shape[0]:
        raise DataValidationError(
            f"cross_entropy: logits {logits.shape} vs {y.size} labels"
        )
    if np.any(y < 0) or np.any(y >= logits.data.shape[1]):
        raise DataValidationError("cross_entropy: label out of class range")
    log_probs = log_softmax(logits, axis=1)
    picked = log_probs[
        np.arange(y.size), y.astype(int)
    ]
    return -(picked.mean())


# class wrappers so the losses are nn-usable ------------------------------------ #


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: object) -> Tensor:
        return mse_loss(prediction, target)


class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, target: object) -> Tensor:
        return bce_with_logits(logits, target)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, target: object) -> Tensor:
        return cross_entropy(logits, target)

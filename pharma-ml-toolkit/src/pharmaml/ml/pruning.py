"""Offline weight and channel pruning -- the torch.nn.utils.prune lineage.

Rounds 188-189 checked PyTorch's pruning utilities and the gradual-
pruning literature (Zhu & Gupta 2017) file by file and lifted what
survives without a torch module tree:

* **Magnitude pruning** (Han et al.) -- the top-|w| mask at a target
  sparsity, global (one threshold over all layers) or local (per-layer
  thresholds).  Ties break toward the smaller index so the cut is
  deterministic.
* **Structured channel pruning** -- whole output channels (rows of a
  weight matrix) dropped by L1 norm, the pruning that actually shrinks
  a matmul.  The paired bias entries go with the rows.
* **The gradual sparsity schedule** -- Zhu & Gupta's cubic ramp from
  an initial to a final sparsity across ``n`` steps (the lottery-
  ticket-style iterative retrain loop's metronome).
* **The iterative prune-and-retrain loop** -- prune to the schedule's
  next level, retrain, repeat: the "rewind and retrain" half that
  keeps accuracy while the mask monotonically tightens.
* **The pruning report** -- per-layer weights/masked/sparsity, the
  global counts, and the params actually freed (the number the GUI
  card prints).

The honest scope, stated once: torch's ``torch.nn.utils.prune`` method
zoo (random/l1/ln structured + the RemovableHandle bookkeeping) is
reduced to magnitude + channel, the two the toolkit's dense MLP
estimators can act on; movement pruning (powerful-but-exotic) and
fine-grained N:M sparsity (2:4 tensor-core formats) are what this
toolkit does not vendor.  What transfers is the mask arithmetic, the
schedule, and the loop.

The engines are pure NumPy and operate on weight matrices directly or
on the autograd tape's :class:`~pharmaml.ml.autograd.Module` objects
(and sklearn coef_ blocks via :func:`prune_linear_model`).

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.pruning import magnitude_mask
>>> w = np.array([[0.1, -0.9, 0.2], [0.05, 0.3, -0.8]])
>>> mask = magnitude_mask(w, sparsity=0.5, scope="local")
>>> mask.astype(int).sum(axis=1)  # two of three kept per row
array([2, 2])
>>> (w * mask == 0).sum()  # the weakest entry per row is zeroed
2

>>> from pharmaml.ml.pruning import sparsity_schedule
>>> [round(s, 3) for s in sparsity_schedule(0.0, 0.8, 3)]
[0.0, 0.7, 0.8]
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pharmaml.ml.autograd import Linear, Module

__all__ = [
    "LayerPruning",
    "PruningReport",
    "apply_magnitude_pruning",
    "count_sparsity",
    "iterative_prune_and_retrain",
    "magnitude_mask",
    "prune_linear_layer",
    "prune_linear_model",
    "sparsity_schedule",
    "structured_channel_mask",
]


# -- magnitude pruning ------------------------------------------------------ #


def magnitude_mask(
    weights: np.ndarray, sparsity: float, scope: str = "local"
) -> np.ndarray:
    """The keep-mask that cuts the smallest |w| to reach ``sparsity``.

    Parameters
    ----------
    weights:
        Any-shape weight block.
    sparsity:
        Fraction of entries to ZERO (0 = keep everything, 1 = drop all
        but a single survivor per scope unit).
    scope:
        ``"local"`` thresholds each row (each channel) independently;
        ``"global"`` takes one threshold over the whole block.

    Ties break toward keeping the earlier entry, so the mask is a
    deterministic function of ``weights``.

    >>> import numpy as np
    >>> w = np.array([0.1, -0.5, 0.3, -0.05])
    >>> magnitude_mask(w, 0.5).astype(int).tolist()
    [0, 1, 1, 0]

    Raises
    ------
    ValueError
        Sparsity outside [0, 1) or an unknown scope.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if scope not in {"local", "global"}:
        raise ValueError(f"scope must be 'local' or 'global', got {scope!r}")
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return np.ones_like(w, dtype=bool)
    if w.ndim < 2:
        return _topk_keep_mask(w.ravel(), sparsity)
    if scope == "global":
        return _topk_keep_mask(w.ravel(), sparsity).reshape(w.shape)
    mask = np.ones_like(w, dtype=bool)
    flat_view = w.reshape(w.shape[0], -1)
    mask_view = mask.reshape(flat_view.shape)
    for row in range(flat_view.shape[0]):
        mask_view[row] = _topk_keep_mask(flat_view[row], sparsity)
    return mask


def _topk_keep_mask(flat: np.ndarray, sparsity: float) -> np.ndarray:
    """Keep the ceil((1-s)·n) largest |entries| of a 1-D block."""
    n = flat.size
    keep = max(1, math.ceil((1.0 - sparsity) * n))
    if keep >= n:
        return np.ones(n, dtype=bool)
    magnitudes = np.abs(flat)
    # argsort ascending, stable (kind='stable') -> ties keep earlier index
    order = np.argsort(-magnitudes, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:keep]] = True
    return mask


def count_sparsity(weights: np.ndarray) -> float:
    """The achieved zero-fraction of a (possibly already pruned) block.

    >>> import numpy as np
    >>> count_sparsity(np.array([[0.0, 1.0], [0.0, 0.0]]))
    0.75
    """
    w = np.asarray(weights)
    if w.size == 0:
        return 0.0
    return float(np.mean(w == 0.0))


# -- the module-facing API --------------------------------------------------- #


@dataclass
class LayerPruning:
    """One layer's outcome: the mask, the pruned weights, the counts."""

    name: str
    shape: tuple[int, ...]
    n_weights: int
    n_zeroed: int
    sparsity: float

    @property
    def n_kept(self) -> int:
        return self.n_weights - self.n_zeroed


@dataclass
class PruningReport:
    """What the GUI card renders: per-layer rows plus the global cut."""

    layers: list[LayerPruning] = field(default_factory=list)
    scope: str = "local"
    target_sparsity: float = 0.0

    @property
    def n_weights(self) -> int:
        return sum(l.n_weights for l in self.layers)

    @property
    def n_zeroed(self) -> int:
        return sum(l.n_zeroed for l in self.layers)

    @property
    def achieved_sparsity(self) -> float:
        total = self.n_weights
        return self.n_zeroed / total if total else 0.0

    @property
    def n_kept(self) -> int:
        return self.n_weights - self.n_zeroed

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "target_sparsity": self.target_sparsity,
            "achieved_sparsity": self.achieved_sparsity,
            "n_weights": self.n_weights,
            "n_zeroed": self.n_zeroed,
            "n_kept": self.n_kept,
            "layers": [
                {
                    "name": l.name,
                    "shape": list(l.shape),
                    "n_weights": l.n_weights,
                    "n_zeroed": l.n_zeroed,
                    "sparsity": l.sparsity,
                }
                for l in self.layers
            ],
        }


def apply_magnitude_pruning(
    module: Module,
    sparsity: float,
    scope: str = "local",
    *,
    in_place: bool = True,
) -> PruningReport:
    """Magnitude-prune every :class:`Linear` layer of a tape module.

    The mask zeroes the weight entries in place (the torch.prune
    semantic: weights become their own masked view); with
    ``in_place=False`` the module is left untouched and only the report
    is computed.

    >>> import numpy as np
    >>> from pharmaml.ml.autograd import Linear, Sequential, ReLU
    >>> net = Sequential(Linear(4, 3, seed=1), ReLU(), Linear(3, 1, seed=2))
    >>> report = apply_magnitude_pruning(net, 0.5, scope="global")
    >>> 0.4 <= report.achieved_sparsity <= 0.6
    True
    >>> len(report.layers)
    2
    """
    report = PruningReport(scope=scope, target_sparsity=sparsity)
    for name, layer in _linear_layers(module):
        w = np.asarray(layer.weight.data, dtype=float)
        mask = magnitude_mask(w, sparsity, scope=scope)
        if in_place:
            layer.weight.data = (w * mask).astype(layer.weight.data.dtype)
        pruned = w * mask if not in_place else np.asarray(layer.weight.data)
        report.layers.append(
            LayerPruning(
                name=name,
                shape=tuple(w.shape),
                n_weights=w.size,
                n_zeroed=int(np.sum(pruned == 0.0)),
                sparsity=count_sparsity(pruned),
            )
        )
    return report


def _linear_layers(module: Module) -> list[tuple[str, Linear]]:
    """The named Linear children (depth-first, tape walk order)."""
    found: list[tuple[str, Linear]] = []

    def walk(node: Any, prefix: str) -> None:
        for name, value in vars(node).items():
            if isinstance(value, Linear):
                found.append((f"{prefix}{name}", value))
            elif isinstance(value, Module):
                walk(value, f"{prefix}{name}.")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Linear):
                        found.append((f"{prefix}{name}[{index}]", item))
                    elif isinstance(item, Module):
                        walk(item, f"{prefix}{name}[{index}].")

    walk(module, "")
    return found


# -- structured channel pruning ----------------------------------------------- #


def structured_channel_mask(
    weights: np.ndarray, sparsity: float
) -> np.ndarray:
    """The row-keep mask cutting whole output channels by L1 norm.

    For a ``[out, in]`` weight this drops entire rows -- the pruning
    that makes the next forward genuinely cheaper.  Rows tie-break
    toward the earlier channel.

    >>> import numpy as np
    >>> w = np.array([[1.0, 1.0], [0.1, 0.1], [0.5, 0.5]])
    >>> structured_channel_mask(w, 1 / 3).astype(int).tolist()
    [[1, 1], [0, 0], [1, 1]]

    Raises
    ------
    ValueError
        Weights not 2-D or sparsity outside [0, 1).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError(f"channel pruning needs a 2-D weight, got {w.ndim}-D")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    n_channels = w.shape[0]
    keep = max(1, math.ceil((1.0 - sparsity) * n_channels))
    if keep >= n_channels:
        return np.ones_like(w, dtype=bool)
    norms = np.abs(w).sum(axis=1)
    order = np.argsort(-norms, kind="stable")
    mask = np.zeros(n_channels, dtype=bool)
    mask[order[:keep]] = True
    return np.repeat(mask[:, None], w.shape[1], axis=1)


def prune_linear_layer(
    layer: Linear, sparsity: float, *, structured: bool = False
) -> LayerPruning:
    """Prune one tape :class:`Linear` in place (magnitude or channel).

    Channel pruning also zeroes the paired bias entries and compacts
    nothing (shape is preserved; the zeros are the pruning) -- the
    honest statement of a masked channel.
    """
    w = np.asarray(layer.weight.data, dtype=float)
    if structured:
        mask = structured_channel_mask(w, sparsity)
    else:
        mask = magnitude_mask(w, sparsity, scope="local")
    layer.weight.data = (w * mask).astype(layer.weight.data.dtype)
    bias_param = getattr(layer, "bias", None)
    if structured and bias_param is not None:
        bias = np.asarray(bias_param.data, dtype=float)
        if bias.ndim == 1 and bias.shape[0] == w.shape[0]:
            keep_rows = mask[:, 0]
            bias_param.data = (bias * keep_rows).astype(bias_param.data.dtype)
    pruned = np.asarray(layer.weight.data)
    return LayerPruning(
        name="linear",
        shape=tuple(w.shape),
        n_weights=w.size,
        n_zeroed=int(np.sum(pruned == 0.0)),
        sparsity=count_sparsity(pruned),
    )


def prune_linear_model(
    coefficients: np.ndarray, sparsity: float
) -> tuple[np.ndarray, float]:
    """Magnitude-prune a sklearn-style flat ``coef_`` block.

    Returns the pruned coefficients and the achieved sparsity -- the
    linear models' one-shot entry into pruning.

    >>> import numpy as np
    >>> coef, s = prune_linear_model(np.array([0.1, -0.7, 0.4]), 1 / 3)
    >>> coef.astype(float).tolist()
    [0.0, -0.7, 0.4]
    >>> round(s, 4)
    0.3333
    """
    w = np.asarray(coefficients, dtype=float)
    mask = magnitude_mask(w, sparsity, scope="global")
    return w * mask, count_sparsity(w * mask)


# -- the gradual schedule ------------------------------------------------------ #


def sparsity_schedule(
    initial: float, final: float, steps: int
) -> list[float]:
    """Zhu & Gupta's cubic gradual-sparsity ramp over ``steps`` levels.

    The ramp is ``s_i = s_final + (s_init - s_final) * (1 - i/(steps-1))^3``
    for ``i = 0..steps-1`` -- flat at the start, steep in the middle,
    landing exactly on ``final``; ``steps <= 1`` short-circuits to the
    final value (a one-shot prune).

    >>> sparsity_schedule(0.0, 0.8, 1)
    [0.8]
    >>> sparsity_schedule(0.0, 0.8, 2)
    [0.0, 0.8]
    """
    if steps <= 1:
        return [final]
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        out.append(final + (initial - final) * (1.0 - t) ** 3)
    return out


# -- the iterative loop --------------------------------------------------------- #


def iterative_prune_and_retrain(
    module: Module,
    retrain: Callable[[Module], float],
    *,
    target_sparsity: float = 0.9,
    steps: int = 4,
    initial_sparsity: float = 0.0,
    scope: str = "local",
) -> tuple[Module, PruningReport, list[float]]:
    """The gradual prune-retrain loop over a tape module.

    Each iteration prunes to the schedule's next level (masks only
    tighten: :func:`magnitude_mask` re-ranks the survivors, and the
    previously zeroed entries stay zero because their magnitude is 0)
    and calls ``retrain(module)``, whose return value (a loss/score)
    is recorded.  Returns the module, the final report, and the retrain
    values in loop order.

    >>> from pharmaml.ml.autograd import Linear, Sequential, ReLU
    >>> net = Sequential(Linear(6, 4, seed=3), ReLU(), Linear(4, 2, seed=4))
    >>> losses = []
    >>> net, report, losses = iterative_prune_and_retrain(
    ...     net, lambda m: 0.5, target_sparsity=0.5, steps=2)
    >>> len(losses) == 2 and 0.4 <= report.achieved_sparsity <= 0.6
    True
    """
    schedule = sparsity_schedule(initial_sparsity, target_sparsity, steps)
    retrain_values: list[float] = []
    report = PruningReport(scope=scope, target_sparsity=target_sparsity)
    for level in schedule:
        report = apply_magnitude_pruning(module, level, scope=scope)
        retrain_values.append(float(retrain(module)))
    return module, report, retrain_values

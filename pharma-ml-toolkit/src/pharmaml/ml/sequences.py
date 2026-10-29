"""Timeseries forecasting: sliding windows and from-scratch RNNs.

Chapter 13 of François Chollet's *Deep Learning with Python* notebooks
(``fchollet/deep-learning-with-python-notebooks``) is the one whole ML
domain the toolkit had no surface for: sequences in *time* -- the
Jena-weather style workflow of sliding windows, the commonsense
non-ML baseline every model must beat, recurrent layers trained
through backpropagation-through-time, and the honesty of evaluating
strictly on the future.  This module is that chapter in plain NumPy:

* :func:`sliding_windows` -- the chapter's
  ``timeseries_dataset_from_array`` equivalent: one window per row
  (shape ``(n, window, n_features)``), the target ``horizon`` steps
  past the window's end, every window/target alignment spelled out.
* :func:`commonsense_baseline` -- the ``evaluate_naive_method`` cell:
  predict "the last observed value of the target channel" and measure
  it, because the chapter's first result is how hard that is to beat.
* :class:`RecurrentForecaster` -- the recurrent layer itself, from
  scratch, with the two cells the book teaches: the Elman ``tanh``
  layer (``h_t = tanh(Wx·x_t + Wh·h_{t-1} + b)``) and the full LSTM
  (input/forget/output/cell gates, ``c_t = f⊙c_{t-1} + i⊙g``,
  ``h_t = o⊙tanh(c_t)``).  Training is exact backpropagation through
  time over the whole window (windows are short by design), minibatch
  SGD, and three of the chapter's own recipes: **recurrent dropout**
  (a variational mask on the recurrent input -- one mask per sequence,
  the Gal & Ghahramani approximation the book's ``recurrent_dropout``
  rides), **stacking** (a second layer consuming the first layer's
  hidden sequence, gradients flowing back through every timestep) and
  **bidirectionality** (a second stack reading the window backwards;
  the book's caveat -- bidirectional helps density estimation, *not*
  forecasting -- rides along in the docstring).  Training clips the
  global gradient norm (the chapter's exploding-gradient recipe),
  because backpropagation through time without it diverges on the
  first random walk it meets.
* :func:`forecast_timeseries` -- the end-to-end run: standardise on
  the train segment only (the no-leakage rule -- future statistics
  never touch training inputs), build windows, fit the model, evaluate
  the baseline AND the model on the held-out future windows, and hand
  back everything the plot needs.
* :func:`synthetic_series` -- three seeded generators (seasonal sine,
  AR(2), random walk) so the workflow runs with zero data loaded.

What stayed out, stated once: the 1D-convolution model of the same
chapter (a stride trick over windows, less useful than the recurrence
it replaces at this scale) and Keras' dataset shuffling (windows are
shuffled for *training* here -- the chapter does the same -- but the
split itself stays chronological, always).

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.sequences import (
...     commonsense_baseline, forecast_timeseries, sliding_windows,
...     synthetic_series,
... )
>>> series = synthetic_series("ar2", 240, seed=42)
>>> windows = sliding_windows(series, window=12)
>>> windows.x.shape
(228, 12, 1)
>>> baseline = commonsense_baseline(windows)
>>> round(baseline.mae, 3) > 0          # the naive forecast has real error
True
>>> result = forecast_timeseries(          # a tiny LSTM beats it on AR(2)
...     series, window=12, model="lstm", hidden=6, epochs=30, seed=42,
... )
>>> round(result.model_mae, 3) < round(result.baseline.mae, 3)
True
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

#: Recurrent cell kinds :class:`RecurrentForecaster` implements.
SEQUENCE_MODELS: tuple[str, ...] = ("lstm", "elman")

#: Synthetic series kinds :func:`synthetic_series` generates.
SYNTHETIC_KINDS: tuple[str, ...] = ("sine", "ar2", "walk")


# ---------------------------------------------------------------------------
# Chapter 13 -- windows and the commonsense baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowDataset:
    """A windowed timeseries: the chapter's dataset object, NumPy edition."""

    x: np.ndarray
    y: np.ndarray
    window: int
    horizon: int
    target_index: int
    feature_names: tuple[str, ...]
    target_name: str
    y_index: np.ndarray

    @property
    def n_windows(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[2])


def _as_series(values: Any, *, caller: str) -> np.ndarray:
    """Flatten a Series / column / 1-D array into a float 1-D array."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise DataValidationError(
                f"{caller}: a DataFrame needs exactly one column, got {values.shape[1]}"
            )
        values = values.iloc[:, 0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise DataValidationError(f"{caller}: expected a 1-D series, got shape {array.shape}")
    if array.size < 3:
        raise DataValidationError(f"{caller}: the series is too short ({array.size} points)")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{caller}: the series contains NaN/inf")
    return array


def sliding_windows(
    data: Any,
    window: int,
    *,
    horizon: int = 1,
    target_index: int = 0,
    feature_names: tuple[str, ...] | None = None,
    target_name: str = "target",
) -> WindowDataset:
    """Cut a series (or channel matrix) into overlapping windows.

    Window ``i`` covers ``data[i : i + window]``; its target is the
    value ``horizon`` steps past the window end
    (``data[i + window + horizon - 1]``) of channel ``target_index`` --
    the exact ``timeseries_dataset_from_array`` alignment (the
    chapter's dummy ``range(10)`` example, checked by the demo card).

    Raises
    ------
    DataValidationError
        On non-2-D channel matrices, non-positive window/horizon, a
        target index outside the channels, or a series too short for
        even one window.
    """
    if window < 1:
        raise DataValidationError(f"window must be >= 1, got {window}")
    if horizon < 1:
        raise DataValidationError(f"horizon must be >= 1, got {horizon}")
    if isinstance(data, pd.Series):
        matrix = data.to_numpy(dtype=float).reshape(-1, 1)
        names: tuple[str, ...] = (str(data.name) if data.name else "series",)
    elif isinstance(data, pd.DataFrame):
        matrix = data.to_numpy(dtype=float)
        names = tuple(str(column) for column in data.columns)
    else:
        array = np.asarray(data, dtype=float)
        matrix = array.reshape(-1, 1) if array.ndim == 1 else array
        names = tuple(str(i) for i in range(matrix.shape[1]))
    if matrix.ndim != 2:
        raise DataValidationError(f"expected a series or channel matrix, got shape {matrix.shape}")
    if not 0 <= target_index < matrix.shape[1]:
        raise DataValidationError(
            f"target_index {target_index} outside {matrix.shape[1]} channel(s)"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError("the series contains NaN/inf")
    if feature_names is not None:
        names = tuple(feature_names)
    n_rows = matrix.shape[0]
    n_windows = n_rows - window - horizon + 1
    if n_windows < 1:
        raise DataValidationError(
            f"series of {n_rows} points is too short for window={window} + horizon={horizon}"
        )
    x = np.stack([matrix[i : i + window] for i in range(n_windows)], axis=0)
    y = matrix[window + horizon - 1 : window + horizon - 1 + n_windows, target_index].copy()
    y_index = np.arange(window + horizon - 1, window + horizon - 1 + n_windows)
    return WindowDataset(
        x=x,
        y=y,
        window=window,
        horizon=horizon,
        target_index=target_index,
        feature_names=names,
        target_name=target_name,
        y_index=y_index,
    )


@dataclass(frozen=True)
class BaselineResult:
    """The commonsense baseline's scorecard (the chapter's first table)."""

    mae: float
    rmse: float
    n_windows: int

    def line(self) -> str:
        return f"commonsense baseline · MAE {self.mae:.4g} · RMSE {self.rmse:.4g} · {self.n_windows} window(s)"


def commonsense_baseline(dataset: WindowDataset) -> BaselineResult:
    """Predict "the last observed value" (``evaluate_naive_method``).

    The chapter's own non-ML baseline: the forecast for a window is
    that window's final target-channel reading.  Every model the module
    trains is judged against this number first -- a recurrent model
    that cannot beat "tomorrow equals today" has learned nothing.
    """
    preds = dataset.x[:, -1, dataset.target_index]
    residual = preds - dataset.y
    return BaselineResult(
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual**2))),
        n_windows=dataset.n_windows,
    )


def synthetic_series(kind: str, n: int, *, seed: int = RANDOM_STATE) -> np.ndarray:
    """Seeded synthetic series: the no-data fallback for the whole card.

    ``sine`` is the seasonal signal plus trend and noise (the Jena
    template in miniature), ``ar2`` an autoregressive AR(2) process
    (``x_t = 0.6·x_{t-1} − 0.3·x_{t-2} + ε`` -- recurrence with an
    exact answer), ``walk`` a random walk (the baseline's home turf:
    "tomorrow = today" is near-optimal there, and the report says so).

    Raises
    ------
    DataValidationError
        On an unknown kind or a non-positive length.
    """
    if kind not in SYNTHETIC_KINDS:
        raise DataValidationError(f"unknown series kind {kind!r} (one of {SYNTHETIC_KINDS})")
    if n < 3:
        raise DataValidationError(f"n must be >= 3, got {n}")
    rng = np.random.default_rng(seed)
    if kind == "sine":
        t = np.arange(n, dtype=float)
        return (
            0.4 * np.sin(2.0 * np.pi * t / 24.0)
            + 0.2 * np.sin(2.0 * np.pi * t / 6.0)
            + 0.001 * t
            + rng.normal(0.0, 0.05, n)
        )
    if kind == "ar2":
        values = np.zeros(n)
        noise = rng.normal(0.0, 0.2, n)
        values[0], values[1] = noise[0], noise[1]
        for t in range(2, n):
            values[t] = 0.6 * values[t - 1] - 0.3 * values[t - 2] + noise[t]
        return values
    steps = rng.normal(0.0, 1.0, n)
    return np.cumsum(steps)


# ---------------------------------------------------------------------------
# The recurrent forecaster (Elman + LSTM, backpropagation through time)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


LayerCache = dict[str, np.ndarray]


class _RecurrentStack:
    """One direction of stacked recurrent layers over a window batch.

    Weights per layer: ``Wx`` (features → hidden), ``Wh`` (hidden →
    hidden), ``b``.  LSTM packs the four gates along the output axis in
    ``i, f, g, o`` order.  All math is batch-vectorised: a time step is
    one ``(B, k) @ (k, h)`` product, never a Python loop over rows.

    :meth:`backward` is exact BPTT: the incoming gradient may name the
    layer stack's last state (the output head) AND every timestep of
    the top layer's output (an upper layer's input sequence, when the
    stacks are deep); the per-timestep input gradients it returns are
    what a lower layer consumes, so a two-stack's gradients flow
    through the full sequence, not a last-state shortcut.
    """

    def __init__(
        self,
        kind: str,
        n_input: int,
        hidden: int,
        layers: int,
        *,
        seed: int = RANDOM_STATE,
    ) -> None:
        if kind not in SEQUENCE_MODELS:
            raise DataValidationError(f"unknown recurrent kind {kind!r}")
        self.kind = kind
        self.hidden = hidden
        self.layers = layers
        self.n_input = n_input
        rng = np.random.default_rng(seed)
        self.Wx: list[np.ndarray] = []
        self.Wh: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        fan_in = n_input
        for _ in range(layers):
            gate = 4 if kind == "lstm" else 1
            limit = 1.0 / math.sqrt(fan_in)
            self.Wx.append(rng.uniform(-limit, limit, size=(fan_in, hidden * gate)))
            limit_h = 1.0 / math.sqrt(hidden)
            self.Wh.append(rng.uniform(-limit_h, limit_h, size=(hidden, hidden * gate)))
            bias = np.zeros(hidden * gate)
            if kind == "lstm":
                # the forget gate's bias starts at +1 (Jozefowicz et
                # al.'s finding, the init every LSTM stack ships with):
                # a forget gate at σ(0)=0.5 halves the cell every step
                # and short windows never learn to keep anything
                bias[hidden : 2 * hidden] = 1.0
            self.b.append(bias)
            fan_in = hidden
        self.dWx = [np.zeros_like(w) for w in self.Wx]
        self.dWh = [np.zeros_like(w) for w in self.Wh]
        self.db = [np.zeros_like(b) for b in self.b]

    def forward(
        self, x: np.ndarray, *, dropout_masks: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, list[list[LayerCache]]]:
        """Run ``(B, T, k)`` through the stack; return the last hidden state.

        ``dropout_masks`` (one ``(B, hidden)`` inverted mask per layer)
        applies the variational recurrent dropout: the SAME mask scales
        ``h_{t-1}`` at every step of the sequence (training only).
        """
        batch, steps, _ = x.shape
        caches: list[list[LayerCache]] = []
        current = x
        for layer in range(self.layers):
            mask = dropout_masks[layer] if dropout_masks is not None else None
            Wx, Wh, b = self.Wx[layer], self.Wh[layer], self.b[layer]
            h_prev = np.zeros((batch, self.hidden))
            c_prev = np.zeros((batch, self.hidden))
            layer_cache: list[LayerCache] = []
            h_seq = np.zeros((batch, steps, self.hidden))
            for t in range(steps):
                x_t = current[:, t, :]
                h_input = h_prev * mask if mask is not None else h_prev
                z = x_t @ Wx + h_input @ Wh + b
                if self.kind == "elman":
                    h_t = np.tanh(z)
                    layer_cache.append({"x_t": x_t, "h_input": h_input, "h_t": h_t, "mask": mask})
                else:
                    i, f, g, o = np.split(z, 4, axis=1)
                    i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
                    g = np.tanh(g)
                    c_t = f * c_prev + i * g
                    tanh_c = np.tanh(c_t)
                    h_t = o * tanh_c
                    layer_cache.append(
                        {
                            "x_t": x_t,
                            "h_input": h_input,
                            "mask": mask,
                            "i": i,
                            "f": f,
                            "g": g,
                            "o": o,
                            "c_prev": c_prev,
                            "c_t": c_t,
                            "tanh_c": tanh_c,
                            "h_t": h_t,
                        }
                    )
                    c_prev = c_t
                h_seq[:, t, :] = h_t
                h_prev = h_t
            caches.append(layer_cache)
            current = h_seq
        return h_prev, caches

    def backward(
        self,
        d_seq: np.ndarray | None,
        d_last: np.ndarray | None,
        caches: list[list[LayerCache]],
    ) -> np.ndarray:
        """BPTT into the stack; return the per-timestep INPUT gradients.

        ``d_seq`` (``(B, T, hidden)``) is the gradient on the top
        layer's output sequence (``None`` when nothing above consumes
        it), ``d_last`` the gradient on the top layer's last state (the
        output head).  Returns ``(B, T, n_input)`` -- the gradient wrt
        the stack's input sequence, which a lower stack (or nothing,
        for the bottom) receives.
        """
        batch = caches[0][0]["x_t"].shape[0]
        steps = len(caches[0])
        current_d_seq = d_seq
        if current_d_seq is None:
            current_d_seq = np.zeros((batch, steps, self.hidden))
        elif d_last is not None:
            current_d_seq = current_d_seq.copy()
        if d_last is not None:
            current_d_seq[:, steps - 1, :] += d_last
        for layer in range(self.layers - 1, -1, -1):
            layer_cache = caches[layer]
            dWx, dWh, db = self.dWx[layer], self.dWh[layer], self.db[layer]
            Wx, Wh = self.Wx[layer], self.Wh[layer]
            d_h = np.zeros((batch, self.hidden))
            d_c = np.zeros((batch, self.hidden))
            n_in = Wx.shape[0]
            d_x_seq = np.zeros((batch, steps, n_in))
            for t in range(steps - 1, -1, -1):
                cache = layer_cache[t]
                mask = cache["mask"]
                d_h_t = current_d_seq[:, t, :] + d_h
                if self.kind == "elman":
                    d_pre = d_h_t * (1.0 - cache["h_t"] ** 2)
                    dWx += cache["x_t"].T @ d_pre
                    dWh += cache["h_input"].T @ d_pre
                    db += d_pre.sum(axis=0)
                    d_h = d_pre @ Wh.T
                    if mask is not None:
                        d_h = d_h * mask
                    d_x_seq[:, t, :] = d_pre @ Wx.T
                else:
                    d_o_pre = d_h_t * cache["tanh_c"] * cache["o"] * (1.0 - cache["o"])
                    d_c_total = d_h_t * cache["o"] * (1.0 - cache["tanh_c"] ** 2) + d_c
                    d_f_pre = d_c_total * cache["c_prev"] * cache["f"] * (1.0 - cache["f"])
                    d_i_pre = d_c_total * cache["g"] * cache["i"] * (1.0 - cache["i"])
                    d_g_pre = d_c_total * cache["i"] * (1.0 - cache["g"] ** 2)
                    dz = np.concatenate([d_i_pre, d_f_pre, d_g_pre, d_o_pre], axis=1)
                    dWx += cache["x_t"].T @ dz
                    dWh += cache["h_input"].T @ dz
                    db += dz.sum(axis=0)
                    d_h = dz @ Wh.T
                    if mask is not None:
                        d_h = d_h * mask
                    d_c = d_c_total * cache["f"]
                    d_x_seq[:, t, :] = dz @ Wx.T
            # the input-sequence gradient of THIS layer is the output-
            # sequence gradient of the layer BELOW
            current_d_seq = d_x_seq
        return current_d_seq

    def gradient_norm(self) -> float:
        """Global L2 norm of the accumulated gradients (the clip input)."""
        total = 0.0
        for layer in range(self.layers):
            total += float(np.sum(self.dWx[layer] ** 2))
            total += float(np.sum(self.dWh[layer] ** 2))
            total += float(np.sum(self.db[layer] ** 2))
        return math.sqrt(total)

    def apply_gradients(self, learning_rate: float, scale: float = 1.0) -> None:
        """One SGD step (optionally rescaled by the clip factor)."""
        step = learning_rate * scale
        for layer in range(self.layers):
            self.Wx[layer] -= step * self.dWx[layer]
            self.Wh[layer] -= step * self.dWh[layer]
            self.b[layer] -= step * self.db[layer]
            self.dWx[layer] = np.zeros_like(self.Wx[layer])
            self.dWh[layer] = np.zeros_like(self.Wh[layer])
            self.db[layer] = np.zeros_like(self.b[layer])

    @property
    def n_parameters(self) -> int:
        return int(
            sum(w.size for w in self.Wx)
            + sum(w.size for w in self.Wh)
            + sum(b.size for b in self.b)
        )


@dataclass
class SequenceHistory:
    """Per-epoch training telemetry for one recurrent fit."""

    train_mae: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epochs_run: int = 0

    def final_line(self) -> str:
        val = f" · val MAE {self.val_mae[-1]:.4g}" if self.val_mae else ""
        return f"{self.epochs_run} epoch(s) · train MAE {self.train_mae[-1]:.4g}{val}"


class RecurrentForecaster:
    """A from-scratch recurrent regressor over fixed-length windows.

    Parameters
    ----------
    model:
        ``"lstm"`` (the gated cell) or ``"elman"`` (the plain tanh
        recurrence) -- chapter 13's two recurrent layers.
    hidden:
        Hidden width per layer per direction.
    layers:
        Stack depth (1 or 2; the chapter's "stacking recurrent layers"
        section).  Gradients flow between stacks through the full
        timestep sequence (exact BPTT, no last-state shortcut).
    bidirectional:
        Run a second stack over the reversed window and concatenate the
        two final states.  The book's caveat applies verbatim: this
        helps when the whole sequence is known (classification) and
        buys nothing for forecasting -- offered so the experiment can
        be run, not recommended.
    recurrent_dropout:
        Variational recurrent dropout: one Bernoulli mask per sequence
        per layer scales ``h_{t-1}`` (inverted, ``/(1-p)``) during
        training only -- the approximation Keras' own
        ``recurrent_dropout`` rides.
    clip_norm:
        Global gradient-norm clip (the chapter's own recipe for the
        exploding-gradient problem BPTT invites: every rescaling the
        book's text describes).  The norm is taken over EVERY
        parameter's gradient at once; above the cap the whole step is
        rescaled, never clipped per-entry.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.sequences import RecurrentForecaster, sliding_windows, synthetic_series
    >>> series = synthetic_series("ar2", 160, seed=7)
    >>> windows = sliding_windows(series, window=10)
    >>> forecaster = RecurrentForecaster("lstm", hidden=6, seed=7)
    >>> history = forecaster.fit(windows.x[:100], windows.y[:100], epochs=6)
    >>> predictions = forecaster.predict(windows.x[100:])
    >>> len(predictions) == windows.n_windows - 100
    True
    """

    def __init__(
        self,
        model: str = "lstm",
        *,
        hidden: int = 16,
        layers: int = 1,
        bidirectional: bool = False,
        recurrent_dropout: float = 0.0,
        learning_rate: float = 0.02,
        clip_norm: float = 5.0,
        seed: int = RANDOM_STATE,
    ) -> None:
        if model not in SEQUENCE_MODELS:
            raise DataValidationError(f"unknown model {model!r} (one of {SEQUENCE_MODELS})")
        if hidden < 1 or hidden > 512:
            raise DataValidationError(f"hidden must be in [1, 512], got {hidden}")
        if layers not in (1, 2):
            raise DataValidationError(f"layers must be 1 or 2, got {layers}")
        if not 0.0 <= recurrent_dropout < 0.8:
            raise DataValidationError(
                f"recurrent_dropout must be in [0, 0.8), got {recurrent_dropout}"
            )
        if not 0.0 < learning_rate <= 1.0:
            raise DataValidationError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not 0.0 < clip_norm <= 1e6:
            raise DataValidationError(f"clip_norm must be positive, got {clip_norm}")
        self.model_kind = model
        self.hidden = hidden
        self.layers = layers
        self.bidirectional = bidirectional
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.seed = seed
        self._forward_stack: _RecurrentStack | None = None
        self._backward_stack: _RecurrentStack | None = None
        self._Wout: np.ndarray | None = None
        self._bout: np.ndarray | None = None
        self._dWout: np.ndarray | None = None
        self._dbout: np.ndarray | None = None
        self._n_input: int | None = None

    def _ensure_built(self, n_input: int) -> None:
        if self._forward_stack is not None:
            return
        rng = np.random.default_rng(self.seed)
        self._forward_stack = _RecurrentStack(
            self.model_kind, n_input, self.hidden, self.layers, seed=self.seed
        )
        if self.bidirectional:
            self._backward_stack = _RecurrentStack(
                self.model_kind, n_input, self.hidden, self.layers, seed=self.seed + 1
            )
        out_width = self.hidden * (2 if self.bidirectional else 1)
        limit = 1.0 / math.sqrt(out_width)
        self._Wout = rng.uniform(-limit, limit, size=(out_width, 1))
        self._bout = np.zeros(1)
        self._dWout = np.zeros_like(self._Wout)
        self._dbout = np.zeros_like(self._bout)
        self._n_input = n_input

    def _dropout_masks(
        self, batch: int, rng: np.random.Generator
    ) -> list[list[np.ndarray]] | None:
        """One inverted ``(B, hidden)`` mask per layer per direction."""
        if self.recurrent_dropout <= 0.0:
            return None
        keep = 1.0 - self.recurrent_dropout
        masks: list[list[np.ndarray]] = []
        for _stack in range(2 if self.bidirectional else 1):
            stack_masks = [
                (rng.random((batch, self.hidden)) < keep).astype(float) / keep
                for _ in range(self.layers)
            ]
            masks.append(stack_masks)
        return masks

    def _forward_pass(
        self, x: np.ndarray, masks: list[list[np.ndarray]] | None
    ) -> tuple[
        np.ndarray,
        list[list[LayerCache]],
        list[list[LayerCache]] | None,
    ]:
        """Combined final states plus both directions' caches (one pass)."""
        if self._forward_stack is None:
            raise DataValidationError("the forecaster is untrained -- call fit() first")
        forward, forward_caches = self._forward_stack.forward(
            x, dropout_masks=masks[0] if masks else None
        )
        if not self.bidirectional or self._backward_stack is None:
            return forward, forward_caches, None
        backward, backward_caches = self._backward_stack.forward(
            x[:, ::-1, :], dropout_masks=masks[1] if masks else None
        )
        combined = np.concatenate([forward, backward], axis=1)
        return combined, forward_caches, backward_caches

    def _combined(self, x: np.ndarray) -> np.ndarray:
        """Prediction-time combination (no dropout masks)."""
        combined, _fwd, _bwd = self._forward_pass(x, None)
        return combined

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        epochs: int = 40,
        batch_size: int = 16,
        val_fraction: float = 0.2,
        progress: Callable[[int, int | None, str], None] | None = None,
    ) -> SequenceHistory:
        """Train on windowed data with minibatch SGD + full BPTT.

        The validation split is chronological (the LAST windows), never
        shuffled -- chapter 13's rule that timeseries models are judged
        on the future only.  Training windows themselves ARE shuffled
        (seeded), exactly like the chapter's ``shuffle=True`` datasets.

        Raises
        ------
        DataValidationError
            On bad shapes, NaN input or out-of-range hyperparameters.
        """
        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if not 1 <= batch_size <= 4096:
            raise DataValidationError(f"batch_size must be in [1, 4096], got {batch_size}")
        x_values = np.asarray(x, dtype=float)
        y_values = np.asarray(y, dtype=float).ravel()
        if x_values.ndim != 3:
            raise DataValidationError(f"x must be (n, window, features), got {x_values.shape}")
        if x_values.shape[0] != y_values.shape[0]:
            raise DataValidationError(
                f"x and y disagree ({x_values.shape[0]} vs {y_values.shape[0]} windows)"
            )
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise DataValidationError("the windowed data contains NaN/inf")
        if x_values.shape[0] < 4:
            raise DataValidationError("too few windows to train (need >= 4)")
        if not 0.0 <= val_fraction < 0.5:
            raise DataValidationError(f"val_fraction must be in [0, 0.5), got {val_fraction}")
        n_windows = x_values.shape[0]
        n_train = n_windows - round(n_windows * val_fraction)
        if n_train < 2 or n_train >= n_windows:
            raise DataValidationError("the chronological split left an empty side")
        self._ensure_built(x_values.shape[2])
        if (
            self._Wout is None
            or self._bout is None
            or self._dWout is None
            or self._dbout is None
            or self._forward_stack is None
        ):  # pragma: no cover - _ensure_built just built them
            raise DataValidationError("internal: the forecaster failed to build")
        Wout, bout = self._Wout, self._bout
        dWout, dbout = self._dWout, self._dbout
        forward_stack = self._forward_stack
        backward_stack = self._backward_stack
        rng = np.random.default_rng(self.seed + 2)
        history = SequenceHistory()
        x_train, y_train = x_values[:n_train], y_values[:n_train]
        x_val, y_val = x_values[n_train:], y_values[n_train:]
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n_train)
            epoch_loss = 0.0
            for start in range(0, n_train, batch_size):
                take = order[start : start + batch_size]
                x_batch = x_train[take]
                y_batch = y_train[take]
                masks = self._dropout_masks(len(x_batch), rng)
                combined, forward_caches, backward_caches = self._forward_pass(x_batch, masks)
                pred = combined @ Wout + bout
                residual = pred.ravel() - y_batch
                epoch_loss += float(np.mean(residual**2)) * len(x_batch)
                d_pred = (2.0 * residual / len(x_batch)).reshape(-1, 1)
                dWout += combined.T @ d_pred
                dbout += d_pred.sum(axis=0)
                d_combined = d_pred @ Wout.T
                if backward_stack is not None and backward_caches is not None:
                    d_forward = d_combined[:, : self.hidden]
                    d_backward = d_combined[:, self.hidden :]
                    forward_stack.backward(None, d_forward, forward_caches)
                    backward_stack.backward(None, d_backward, backward_caches)
                else:
                    forward_stack.backward(None, d_combined, forward_caches)
                # the chapter's exploding-gradient recipe: ONE global
                # norm over every gradient, the whole step rescaled
                # together when it exceeds the cap
                total_sq = float(np.sum(dWout**2)) + float(np.sum(dbout**2))
                total_sq += forward_stack.gradient_norm() ** 2
                if backward_stack is not None:
                    total_sq += backward_stack.gradient_norm() ** 2
                norm = math.sqrt(total_sq)
                scale = min(1.0, self.clip_norm / norm) if norm > 0 else 1.0
                # one SGD step over everything the batch touched
                Wout = Wout - self.learning_rate * scale * dWout
                bout = bout - self.learning_rate * scale * dbout
                dWout = np.zeros_like(Wout)
                dbout = np.zeros_like(bout)
                forward_stack.apply_gradients(self.learning_rate, scale)
                if backward_stack is not None:
                    backward_stack.apply_gradients(self.learning_rate, scale)
            self._Wout, self._bout = Wout, bout
            self._dWout, self._dbout = dWout, dbout
            train_pred = self._combined(x_train) @ Wout + bout
            history.train_loss.append(epoch_loss / n_train)
            history.train_mae.append(float(np.mean(np.abs(train_pred.ravel() - y_train))))
            if len(x_val):
                val_pred = self._combined(x_val) @ Wout + bout
                history.val_loss.append(float(np.mean((val_pred.ravel() - y_val) ** 2)))
                history.val_mae.append(float(np.mean(np.abs(val_pred.ravel() - y_val))))
            history.epochs_run = epoch
            if progress is not None:
                progress(
                    epoch,
                    epochs,
                    f"epoch {epoch}/{epochs} · train MAE {history.train_mae[-1]:.4g}",
                )
        return history

    def predict(self, x: Any) -> np.ndarray:
        """Forecast each window (the target, ``horizon`` steps ahead)."""
        if self._forward_stack is None or self._Wout is None or self._bout is None:
            raise DataValidationError("the forecaster is untrained -- call fit() first")
        x_values = np.asarray(x, dtype=float)
        if x_values.ndim != 3:
            raise DataValidationError(f"x must be (n, window, features), got {x_values.shape}")
        if x_values.shape[2] != self._n_input:
            raise DataValidationError(
                f"x has {x_values.shape[2]} features, the model was fit on {self._n_input}"
            )
        return (self._combined(x_values) @ self._Wout + self._bout).ravel()

    @property
    def n_parameters(self) -> int:
        """Parameter count of the built network (0 before fit)."""
        total = 0
        if self._forward_stack is not None:
            total += self._forward_stack.n_parameters
        if self._backward_stack is not None:
            total += self._backward_stack.n_parameters
        if self._Wout is not None and self._bout is not None:
            total += int(self._Wout.size + self._bout.size)
        return total


def _regression_metrics(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    residual = pred - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


@dataclass
class ForecastResult:
    """Everything the sequence card reports, after one run."""

    model: str
    window: int
    horizon: int
    n_series: int
    n_train_windows: int
    n_val_windows: int
    baseline: BaselineResult
    model_mae: float
    model_rmse: float
    model_r2: float
    improvement_pct: float
    history: SequenceHistory
    n_parameters: int
    series_tail: np.ndarray
    series_tail_index: np.ndarray
    val_index: np.ndarray
    val_actual: np.ndarray
    val_pred: np.ndarray
    val_baseline_pred: np.ndarray
    target_name: str

    def headline(self) -> str:
        verdict = (
            "the recurrent model beats the commonsense baseline"
            if self.improvement_pct > 0
            else "the commonsense baseline wins -- the model learned nothing"
        )
        return (
            f"{self.model.upper()} · window {self.window} · horizon {self.horizon} · "
            f"{self.n_train_windows} train / {self.n_val_windows} validation windows · "
            f"baseline MAE {self.baseline.mae:.4g} vs model MAE {self.model_mae:.4g} "
            f"({self.improvement_pct:+.1f}%) · {verdict}"
        )


def forecast_timeseries(
    series: Any,
    *,
    target_name: str = "series",
    window: int = 24,
    horizon: int = 1,
    model: str = "lstm",
    hidden: int = 16,
    layers: int = 1,
    bidirectional: bool = False,
    recurrent_dropout: float = 0.0,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 0.02,
    clip_norm: float = 5.0,
    val_fraction: float = 0.2,
    seed: int = RANDOM_STATE,
    progress: Callable[[int, int | None, str], None] | None = None,
) -> ForecastResult:
    """The chapter's end-to-end workflow on one series.

    The series is standardised with the TRAIN segment's statistics
    only (future values never touch the training inputs -- the
    toolkit's standing no-leakage rule), cut into windows, split
    chronologically (validation windows are the LAST ones), fitted,
    and scored against :func:`commonsense_baseline` on the same
    held-out future.  The returned record carries the raw-space
    validation predictions, the baseline's predictions and the series
    tail so the caller can draw the chapter's comparison plot without
    recomputing anything.

    Raises
    ------
    DataValidationError
        Via :func:`sliding_windows` / the forecaster (short series,
        bad hyperparameters).
    """
    values = _as_series(series, caller="forecast_timeseries")
    n_series = values.size
    n_train_points = max(window + horizon, round(n_series * (1.0 - val_fraction)))
    n_train_points = min(n_train_points, n_series - 1)
    mean = float(values[:n_train_points].mean())
    std = float(values[:n_train_points].std())
    std = std if std > 1e-12 else 1.0
    scaled = (values - mean) / std
    dataset = sliding_windows(scaled, window, horizon=horizon, target_name=target_name)
    n_windows = dataset.n_windows
    if n_windows < 6:
        raise DataValidationError(
            f"{n_windows} window(s) is too few for the train/validation split"
        )
    n_train_windows = max(2, n_windows - round(n_windows * val_fraction))
    n_train_windows = min(n_train_windows, n_windows - 2)
    n_val_windows = n_windows - n_train_windows
    forecaster = RecurrentForecaster(
        model,
        hidden=hidden,
        layers=layers,
        bidirectional=bidirectional,
        recurrent_dropout=recurrent_dropout,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        seed=seed,
    )
    val_fraction_windows = n_val_windows / n_windows if n_val_windows else 0.0
    history = forecaster.fit(
        dataset.x[:n_train_windows],
        dataset.y[:n_train_windows],
        epochs=epochs,
        batch_size=batch_size,
        val_fraction=val_fraction_windows,
        progress=progress,
    )
    val_x = dataset.x[n_train_windows:]
    val_y = dataset.y[n_train_windows:]
    val_pred = forecaster.predict(val_x) * std + mean
    val_actual = val_y * std + mean
    val_baseline_pred = val_x[:, -1, dataset.target_index] * std + mean
    residual = val_baseline_pred - val_actual
    baseline = BaselineResult(
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual**2))),
        n_windows=len(val_x),
    )
    mae, rmse, r2 = _regression_metrics(val_pred, val_actual)
    improvement = (baseline.mae - mae) / baseline.mae * 100.0 if baseline.mae > 0 else 0.0
    tail_points = min(240, n_series)
    return ForecastResult(
        model=model,
        window=window,
        horizon=horizon,
        n_series=n_series,
        n_train_windows=n_train_windows,
        n_val_windows=n_val_windows,
        baseline=baseline,
        model_mae=mae,
        model_rmse=rmse,
        model_r2=r2,
        improvement_pct=improvement,
        history=history,
        n_parameters=forecaster.n_parameters,
        series_tail=values[-tail_points:].copy(),
        series_tail_index=np.arange(n_series - tail_points, n_series),
        val_index=dataset.y_index[n_train_windows:].copy(),
        val_actual=val_actual,
        val_pred=val_pred,
        val_baseline_pred=val_baseline_pred,
        target_name=target_name,
    )


__all__ = [
    "RANDOM_STATE",
    "SEQUENCE_MODELS",
    "SYNTHETIC_KINDS",
    "BaselineResult",
    "ForecastResult",
    "RecurrentForecaster",
    "SequenceHistory",
    "WindowDataset",
    "commonsense_baseline",
    "forecast_timeseries",
    "sliding_windows",
    "synthetic_series",
]

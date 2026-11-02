"""The offline attribution engine -- four interpretability repos, absorbed.

Round 185 checked, one by one, the contents of four explainability
repositories and lifted what survives without an engine dependency:

* **pytorch/captum** -- the attribution protocol itself: one ``attribute``
  call per method, optional ``baselines``, the integrated-gradients
  ``convergence_delta`` sanity number, the InputXGradient / Saliency
  (vanilla gradients) / NoiseTunnel (SmoothGrad) / Occlusion /
  FeatureAblation family, and the positive/negative sign split that
  ``visualize_image_attr`` paints with.
* **keisen/tf-keras-vis** -- the "explain ONE trained network on ONE
  input" card shape (score + seed_input) and the averaged-noise Saliency
  reading of SmoothGrad.
* **PAIR-code/saliency** -- the mask API's two visualizers,
  ``VisualizeImageGrayscale`` (percentile-clipped monochrome) and
  ``VisualizeImageDiverging`` (blue negative / red positive), here
  reduced to Tkinter-ready hex colors.
* **jessevig/bertviz** -- the layer x head attention grid and the
  query/key token-pair digest, rebuilt on the toolkit's own
  :mod:`pharmaml.ml.attention` (numpy multi-head attention).

Everything is numpy-only: no torch, no tensorflow, no network.  The
gradient methods need a layer stack -- the numpy engine's estimators
(``NumpyMLP*``) expose ``_network`` and the stack is read from its
``linears``; occlusion needs only ``predict``/``predict_proba`` and
therefore works on every engine, torch included.  Gradients are exact
analytic backprop through the stack (the finite-difference check in the
test suite pins them), so the numbers do not depend on torch either.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.exceptions import PharmaMLError

__all__ = [
    "ATTRIBUTION_METHODS",
    "AttributionResult",
    "LayerStack",
    "XaiError",
    "attribution_methods",
    "demo_attention_layers",
    "diverging_hex",
    "grayscale_hex",
    "input_x_gradient",
    "integrated_gradients",
    "layer_activation_summary",
    "layer_activation_trace",
    "numpy_layer_stack",
    "occlusion_attributions",
    "positive_negative_split",
    "smoothgrad_attributions",
    "token_pairs",
]


class XaiError(PharmaMLError):
    """Raised on attribution requests the offline engine cannot honour."""


#: The four-method registry (the GUI combobox order).  Each entry is a
#: plain str->str dict -- kind drives the render, label names the card
#: row, description explains the math, origin credits the absorbed repo.
ATTRIBUTION_METHODS: dict[str, dict[str, str]] = {
    "input_x_gradient": {
        "kind": "gradient",
        "label": "Input×Gradient (captum Saliency)",
        "description": (
            "The exact analytic gradient of one output wrt the input "
            "row -- the raw push, sign kept, so the positive/negative "
            "split survives to the render.  Needs the numpy engine's "
            "layer stack."
        ),
        "origin": "pytorch/captum (Saliency / InputXGradient)",
    },
    "integrated_gradients": {
        "kind": "gradient",
        "label": "Integrated Gradients (captum IG)",
        "description": (
            "Midpoint Riemann sum over steps scaled inputs from a zero "
            "(or given) baseline; convergence_delta is captum's "
            "completeness residual f(x) - f(baseline) - sum(attributions)."
        ),
        "origin": "pytorch/captum (IntegratedGradients)",
    },
    "smoothgrad": {
        "kind": "gradient",
        "label": "SmoothGrad (captum NoiseTunnel)",
        "description": (
            "Input gradients averaged over n_samples noisy copies of "
            "the row (std = stdev_ratio x the row's own scale); "
            "sharpens into the plain gradient as the noise vanishes."
        ),
        "origin": "pytorch/captum (NoiseTunnel SmoothGrad); "
        "keisen/tf-keras-vis averaged-noise Saliency",
    },
    "occlusion": {
        "kind": "perturbation",
        "label": "Occlusion (captum Occlusion / FeatureAblation)",
        "description": (
            "The score drop when one feature is replaced by the "
            "baseline value.  Predict-only, so it runs on EVERY "
            "engine -- torch included."
        ),
        "origin": "pytorch/captum (Occlusion / FeatureAblation)",
    },
}


def attribution_methods() -> tuple[str, ...]:
    """The registry keys, in the order the GUI combobox offers them.

    >>> names = attribution_methods()
    >>> names[0]
    'input_x_gradient'
    >>> names[-1]
    'occlusion'
    """
    return tuple(ATTRIBUTION_METHODS)


@dataclass
class AttributionResult:
    """One row's attribution: values plus the method's own metadata.

    ``convergence_delta`` is integrated gradients' completeness
    residual (``None`` for the other methods); ``note`` names the
    baseline or the averaging settings so a saved table can be read
    back honestly.
    """

    attributions: np.ndarray
    method: str
    kind: str
    convergence_delta: float | None = None
    note: str = ""


@dataclass
class LayerStack:
    """The estimator's linear maps + activation, snapshot as plain numpy.

    ``layers[i]`` is ``(weight, bias | None)`` with weight shaped
    ``(out, in)`` -- the numpy engine's own forward convention
    (``x @ W.T + b``).  ``activation`` is the engine token between the
    hidden maps (the backprop needs its derivative).
    """

    layers: tuple[tuple[np.ndarray, np.ndarray | None], ...]
    activation: str = "relu"

    @property
    def n_features(self) -> int:
        return int(self.layers[0][0].shape[1])


def numpy_layer_stack(estimator: Any) -> LayerStack:
    """Snapshot a numpy-engine estimator's layers for offline math.

    Accepts the estimator (``NumpyMLPRegressor`` / ``NumpyMLPClassifier``
    expose ``_network`` and ``activation``) or the ``_MLPNetwork``
    itself.  Anything else -- a torch estimator, a sklearn pipeline --
    raises :class:`XaiError` naming the occlusion sibling that does not
    need the stack.
    """
    network = getattr(estimator, "_network", estimator)
    linears = getattr(network, "linears", None)
    if linears is None:
        raise XaiError(
            "the gradient methods need the numpy engine's layer stack "
            "(a NumpyMLP* estimator, whose _network exposes .linears); "
            "occlusion_attributions() needs only predict/predict_proba "
            "and runs on every engine"
        )
    if not linears:
        raise XaiError("the estimator's layer stack is empty")
    activation = str(
        getattr(estimator, "activation", None) or getattr(network, "activation", None)
        or _probe_activation_name(getattr(network, "_act_np", None)) or "relu"
    )
    layers = []
    for linear in linears:
        weight = np.asarray(linear.weight.data, dtype=float)
        bias = (
            np.asarray(linear.bias.data, dtype=float)
            if getattr(linear, "bias", None) is not None
            else None
        )
        layers.append((weight, bias))
    return LayerStack(layers=tuple(layers), activation=activation)


def _np_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """The numpy forward activation, shared with ml/numpy_neural."""
    from pharmaml.ml.numpy_neural import _ACTIVATION_FNS

    try:
        _tensor_fn, np_fn = _ACTIVATION_FNS[name]
    except KeyError as exc:
        raise XaiError(f"unknown activation {name!r}") from exc
    return np_fn


def _probe_activation_name(probe: Callable[[np.ndarray], np.ndarray] | None) -> str:
    """Name the network's numpy activation by probing it on one value.

    ``_MLPNetwork`` stores the activation as the CALLABLE pair
    (``_act_np``), not a token -- so the bare-network path recovers the
    name empirically instead of silently defaulting to ``relu`` (which
    would corrupt every tanh/sigmoid network's backprop).
    """
    if probe is None:
        return ""
    from pharmaml.ml.numpy_neural import _ACTIVATION_FNS

    sample = np.array([1.0, -1.0, 0.5])
    try:
        out = np.asarray(probe(sample), dtype=float)
    except Exception:  # noqa: BLE001 -- unknown callable, caller defaults
        return ""
    for name, (_tensor_fn, np_fn) in _ACTIVATION_FNS.items():
        if np.allclose(out, np.asarray(np_fn(sample), dtype=float)):
            return str(name)
    return ""


def _activation_derivative(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """d act/dz for the four tokens the engine accepts."""
    if name == "relu":
        return (z > 0).astype(float)
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "gelu":
        # d/dz of the tanh-approximation GELU the engine runs
        inner = z + 0.044715 * z * z * z
        d_inner = 1.0 + 3.0 * 0.044715 * z * z
        t = np.tanh(math.sqrt(2.0 / math.pi) * inner)
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * math.sqrt(
            2.0 / math.pi
        ) * d_inner
    raise XaiError(f"unknown activation {name!r}")


def _as_row(value: Any) -> np.ndarray:
    """Validate + flatten one input row (finite, non-empty, 1-D)."""
    row = np.asarray(value, dtype=float).reshape(-1)
    if row.size == 0:
        raise XaiError("the input row is empty")
    if not np.all(np.isfinite(row)):
        raise XaiError("the input row carries NaN/inf -- attribute a finite row")
    return row


def _forward_trace(
    stack: LayerStack, x: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Numpy forward; returns the logits and the hidden ``(z, a)`` pairs."""
    activation = _np_activation(stack.activation)
    out = x
    trace: list[tuple[np.ndarray, np.ndarray]] = []
    for weight, bias in stack.layers[:-1]:
        z = out @ weight.T + (bias if bias is not None else 0.0)
        a = activation(z)
        trace.append((z, a))
        out = a
    weight, bias = stack.layers[-1]
    logits = out @ weight.T + (bias if bias is not None else 0.0)
    return logits, trace


def _stack_score(stack: LayerStack, row: np.ndarray, output_index: int) -> float:
    """The scalar output the gradient methods differentiate."""
    logits, _trace = _forward_trace(stack, row)
    values = np.asarray(logits, dtype=float).reshape(-1)
    idx = int(output_index)
    if idx < 0 or idx >= values.size:
        raise XaiError(
            f"output_index {idx} out of range -- this row produces "
            f"{values.size} output(s)"
        )
    return float(values[idx])


def _input_gradient(
    stack: LayerStack, x: np.ndarray, output_index: int
) -> tuple[np.ndarray, float]:
    """Exact analytic d output/output_index / d input via backprop.

    Returns ``(gradient, output_value)``.  The chain rule walks the
    snapshot backwards: ``d h = delta @ W`` through each linear map,
    ``d z = d h * act'(z)`` through each activation.
    """
    activation_name = stack.activation
    activation = _np_activation(activation_name)
    value = _stack_score(stack, x, output_index)

    # forward, keeping the hidden pre/post activations
    zs: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    out = x
    for weight, bias in stack.layers[:-1]:
        z = out @ weight.T + (bias if bias is not None else 0.0)
        a = activation(z)
        zs.append(z)
        acts.append(a)
        out = a

    # backward from one-hot on the chosen output
    delta = np.zeros(stack.layers[-1][0].shape[0])
    delta[int(output_index)] = 1.0
    for i in range(len(stack.layers) - 1, 0, -1):
        weight_i, _bias_i = stack.layers[i]
        # d a_i = delta @ W_i   (logits/hidden = a_i @ W_i.T)
        delta = delta @ weight_i
        # through activation i -> d z_i (zs[i-1] IS z_i, 0-indexed list)
        deriv = _activation_derivative(activation_name, zs[i - 1], acts[i - 1])
        delta = delta * deriv.reshape(-1)
    grad = delta @ stack.layers[0][0]
    return grad, value


def input_x_gradient(
    estimator: Any, row: Any, output_index: int = 0
) -> AttributionResult:
    """Vanilla input gradients of one output wrt the row (captum Saliency).

    The raw analytic gradient -- captum's Saliency without the abs, so
    the push SIGN survives to the GUI's positive/negative split.
    Pinned against finite differences in the test suite.

    >>> net = _demo_network()
    >>> result = input_x_gradient(net, [0.5, -1.0, 2.0])
    >>> result.method, result.kind, result.convergence_delta
    ('input_x_gradient', 'gradient', None)
    """
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    grad, _value = _input_gradient(stack, x, output_index)
    return AttributionResult(
        attributions=grad,
        method="input_x_gradient",
        kind="gradient",
    )


def integrated_gradients(
    estimator: Any,
    row: Any,
    steps: int = 50,
    baseline: Any = None,
    output_index: int = 0,
) -> AttributionResult:
    """Integrated gradients from a zero (or given) baseline (captum IG).

    Midpoint Riemann sum over ``steps`` scaled inputs; the returned
    ``convergence_delta`` is ``f(x) - f(baseline) - sum(attributions)``
    -- captum's completeness residual, near zero when the step count
    has bought convergence.
    """
    if int(steps) < 1:
        raise XaiError("integrated gradients needs steps >= 1")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    if baseline is None:
        base = np.zeros_like(x)
        note = "zero baseline"
    else:
        base = _as_row(baseline)
        if base.size != x.size:
            raise XaiError(
                f"the baseline has {base.size} features, the row {x.size}"
            )
        note = "given baseline"
    total = np.zeros_like(x)
    for step in range(int(steps)):
        alpha_lo = step / int(steps)
        alpha_hi = (step + 1) / int(steps)
        point = base + (x - base) * (0.5 * (alpha_lo + alpha_hi))
        grad, _value = _input_gradient(stack, point, output_index)
        total += grad
    attributions = (x - base) * total / int(steps)
    f_x = _stack_score(stack, x, output_index)
    f_base = _stack_score(stack, base, output_index)
    delta = f_x - f_base - float(attributions.sum())
    return AttributionResult(
        attributions=attributions,
        method="integrated_gradients",
        kind="gradient",
        convergence_delta=delta,
        note=note,
    )


def smoothgrad_attributions(
    estimator: Any,
    row: Any,
    n_samples: int = 32,
    stdev_ratio: float = 0.2,
    seed: int = 0,
    output_index: int = 0,
) -> AttributionResult:
    """SmoothGrad: input gradients averaged over noisy copies.

    ``n_samples`` perturbed rows (per-feature std = ``stdev_ratio`` x
    the row's own std, never zero) go through the same analytic
    backprop and the attributions average -- captum's NoiseTunnel
    reading, and tf-keras-vis' averaged-noise Saliency.  As the noise
    vanishes the average sharpens into the plain gradient (the test
    suite pins that limit).
    """
    if int(n_samples) < 1:
        raise XaiError("smoothgrad needs n_samples >= 1")
    if float(stdev_ratio) < 0.0:
        raise XaiError("stdev_ratio must be >= 0")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    rng = np.random.default_rng(int(seed))
    spread = float(stdev_ratio) * float(np.std(x)) if np.std(x) > 0 else 1e-6
    total = np.zeros_like(x)
    for _sample in range(int(n_samples)):
        noisy = x + rng.normal(0.0, spread, size=x.shape)
        grad, _value = _input_gradient(stack, _as_row(noisy), output_index)
        total += grad
    attributions = total / int(n_samples)
    return AttributionResult(
        attributions=attributions,
        method="smoothgrad",
        kind="gradient",
        note=(
            f"smoothgrad: {int(n_samples)} samples, "
            f"stdev_ratio={float(stdev_ratio):g}, seed={int(seed)}"
        ),
    )


def _occlusion_score(
    estimator: Any, row: np.ndarray, output_index: int
) -> tuple[Callable[[np.ndarray], float], int]:
    """A scalar score function for occlusion, from stack or predictions.

    The numpy engine scores through its own forward; every other
    engine scores through ``predict`` (regression) or
    ``predict_proba`` (classification, ``output_index`` picks the
    class column) -- the predict-only path is why occlusion runs on
    every engine, torch included.
    """
    try:
        stack = numpy_layer_stack(estimator)
    except XaiError:
        stack = None
    if stack is not None:
        if row.size != stack.n_features:
            raise XaiError(
                f"the row has {row.size} features but the network "
                f"expects {stack.n_features}"
            )
        n_features = stack.n_features

        def score_via_stack(values: np.ndarray) -> float:
            return _stack_score(stack, values, output_index)

        return score_via_stack, n_features

    row2d = row.reshape(1, -1)
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(row2d), dtype=float).reshape(-1)
        idx = int(output_index)
        if idx < 0 or idx >= proba.size:
            raise XaiError(
                f"output_index {idx} out of range -- predict_proba "
                f"offers {proba.size} class column(s)"
            )

        def score_via_proba(values: np.ndarray) -> float:
            return float(np.asarray(estimator.predict_proba(values.reshape(1, -1)), dtype=float).reshape(-1)[idx])

        return score_via_proba, int(row.size)

    if hasattr(estimator, "predict"):
        idx = int(output_index)
        if idx != 0:
            raise XaiError(
                "a regressor offers one output -- output_index must be 0"
            )

        def score_via_predict(values: np.ndarray) -> float:
            return float(
                np.asarray(estimator.predict(values.reshape(1, -1)), dtype=float)
                .reshape(-1)[0]
            )

        return score_via_predict, int(row.size)

    raise XaiError(
        "occlusion needs a numpy-engine estimator or something exposing "
        "predict/predict_proba"
    )


def occlusion_attributions(
    estimator: Any, row: Any, baseline: Any = None, output_index: int = 0
) -> AttributionResult:
    """One-feature occlusion: the score drop per replaced feature.

    Each feature is swapped to the baseline value (zero unless given)
    and the score drop ``f(full) - f(occluded)`` recorded -- captum's
    Occlusion / FeatureAblation reading of the perturbation family.
    """
    x = _as_row(row)
    score, n_features = _occlusion_score(estimator, x, output_index)
    if baseline is None:
        base = np.zeros_like(x)
        note = "zero baseline"
    else:
        base = _as_row(baseline)
        if base.size != x.size:
            raise XaiError(
                f"the baseline has {base.size} features, the row {x.size}"
            )
        note = "given baseline"
    full = score(x)
    drops = np.empty(n_features, dtype=float)
    for i in range(n_features):
        occluded = x.copy()
        occluded[i] = base[i]
        drops[i] = full - score(occluded)
    return AttributionResult(
        attributions=drops,
        method="occlusion",
        kind="perturbation",
        note=note,
    )


def positive_negative_split(attributions: Any) -> tuple[np.ndarray, np.ndarray]:
    """captum's sign split: (positive part, negative part), signs kept.

    The convention ``visualize_image_attr`` paints with -- red the
    positive push, blue the negative -- reduced to the two masks the
    GUI's per-feature table renders.

    >>> values = np.array([0.5, -1.0, 0.0])
    >>> pos, neg = positive_negative_split(values)
    >>> pos.tolist(), neg.tolist()
    ([0.5, 0.0, 0.0], [0.0, -1.0, 0.0])
    """
    values = np.asarray(attributions, dtype=float)
    positive = np.where(values > 0.0, values, 0.0)
    negative = np.where(values < 0.0, values, 0.0)
    return positive, negative


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _to_hex(r: float, g: float, b: float) -> str:
    """One Tkinter hex color from 0-1 floats (clamped, 2-digit each)."""
    return "#{:02x}{:02x}{:02x}".format(
        int(round(_clamp01(r) * 255)),
        int(round(_clamp01(g) * 255)),
        int(round(_clamp01(b) * 255)),
    )


def diverging_hex(value: Any, vmax: Any) -> str:
    """PAIR-code/saliency's diverging palette as one Tkinter hex color.

    ``VisualizeImageDiverging`` maps positive values towards red and
    negative towards blue over ``[-vmax, vmax]``; zero stays white.

    >>> diverging_hex(0.0, 1.0)
    '#ffffff'
    >>> int(diverging_hex(-1.0, 1.0)[5:7], 16) > int(diverging_hex(-1.0, 1.0)[1:3], 16)
    True
    """
    v = float(np.asarray(value, dtype=float).reshape(-1)[0])
    limit = abs(float(np.asarray(vmax, dtype=float).reshape(-1)[0]))
    if limit <= 0.0:
        return "#ffffff"
    ratio = _clamp01(abs(v) / limit)
    if v >= 0.0:
        return _to_hex(1.0, 1.0 - ratio, 1.0 - ratio)
    return _to_hex(1.0 - ratio, 1.0 - ratio, 1.0)


def grayscale_hex(value: Any, values: Any, percentile: float = 99.0) -> str:
    """PAIR-code/saliency's grayscale palette as one Tkinter hex color.

    ``VisualizeImageGrayscale`` caps the magnitude at the given
    percentile of ``values`` (exactly the reference implementation's
    outlier clamp), then maps |value|/cap onto white -> black.
    """
    v = abs(float(np.asarray(value, dtype=float).reshape(-1)[0]))
    magnitude = np.abs(np.asarray(values, dtype=float).reshape(-1))
    if magnitude.size == 0:
        raise XaiError("grayscale_hex needs a non-empty values array")
    cap = float(np.percentile(magnitude, float(percentile)))
    if cap <= 0.0:
        cap = float(magnitude.max()) if magnitude.max() > 0 else 1.0
    shade = _clamp01(v / cap)
    return _to_hex(1.0 - shade, 1.0 - shade, 1.0 - shade)


def layer_activation_trace(estimator: Any, row: Any) -> list[np.ndarray]:
    """Per-hidden-layer activations for one row (captum LayerActivation).

    Returns the post-activation hidden values layer by layer -- the
    offline twin of captum's ``LayerActivation`` for the numpy engine.
    """
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    _logits, trace = _forward_trace(stack, x)
    return [np.asarray(a, dtype=float).reshape(-1) for _z, a in trace]


def layer_activation_summary(trace: Any) -> list[dict[str, float | int]]:
    """Per-layer readout: units, min/mean/max and the dead-unit share.

    >>> summary = layer_activation_summary([np.array([0.0, 1.5, -0.5])])
    >>> summary[0]["units"], summary[0]["dead_fraction"]
    (3, 0.3333333333333333)
    """
    summary: list[dict[str, float | int]] = []
    for index, values in enumerate(trace, start=1):
        layer = np.asarray(values, dtype=float).reshape(-1)
        if layer.size == 0:
            raise XaiError("the trace carries an empty layer")
        summary.append(
            {
                "layer": int(index),
                "units": int(layer.size),
                "min": float(layer.min()),
                "mean": float(layer.mean()),
                "max": float(layer.max()),
                "dead_fraction": float(np.mean(layer == 0.0)),
            }
        )
    return summary


def demo_attention_layers(
    tokens: Any, n_layers: int = 2, n_heads: int = 2, seed: int = 0
) -> list[Any]:
    """A seeded multi-layer multi-head attention stack (bertviz's data).

    Every layer gets its own seeded embedding matrix and runs the
    toolkit's own numpy multi-head attention
    (:func:`pharmaml.ml.attention.self_attention_report`), so the
    layer x head grid shows genuinely different patterns per layer and
    head -- analysis, not training, exactly like the attention module's
    own stance on learned projections.
    """
    from pharmaml.ml.attention import self_attention_report

    names = [str(token).strip() for token in tokens if str(token).strip()]
    if len(names) < 2:
        raise XaiError("the attention demo needs at least 2 tokens")
    if n_layers < 1 or n_heads < 1:
        raise XaiError("n_layers and n_heads must both be >= 1")
    rng = np.random.default_rng(int(seed))
    dim = max(4, 2 * n_heads)
    reports = []
    for layer in range(n_layers):
        matrix = rng.normal(size=(len(names), dim))
        matrix += np.arange(len(names), dtype=float)[:, None] * 0.37
        matrix += float(layer) * 1.7
        reports.append(
            self_attention_report(
                matrix,
                n_heads=n_heads,
                causal=False,
                seed=int(seed) + 1000 * (layer + 1),
            )
        )
    return reports


def token_pairs(matrix: Any, tokens: Any, top_k: int = 8) -> list[tuple[float, str, str]]:
    """The strongest query→key pairs of one head (bertviz's digest).

    >>> matrix = [[0.1, 0.9], [0.2, 0.8]]
    >>> token_pairs(matrix, ["a", "b"], top_k=1)
    [(0.9, 'a', 'b')]
    """
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise XaiError("an attention head matrix must be square (n, n)")
    n = int(weights.shape[0])
    names = [str(token) for token in tokens]
    if len(names) != n:
        raise XaiError(f"{len(names)} token labels for an {n}x{n} attention matrix")
    k = max(1, min(top_k, n * n))
    flat = np.argsort(weights, axis=None)[::-1][:k]
    return [
        (float(weights.reshape(-1)[i]), names[int(i) // n], names[int(i) % n])
        for i in flat
    ]


def _demo_network() -> Any:
    """A tiny trained numpy-engine network for the doctests."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.1
    from pharmaml.ml.numpy_neural import NumpyMLPRegressor

    return NumpyMLPRegressor(hidden_layers=(4,), epochs=30, random_state=0).fit(x, y)

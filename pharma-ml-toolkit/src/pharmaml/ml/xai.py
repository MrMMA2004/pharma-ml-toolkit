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
    "attribution_effectiveness",
    "attribution_methods",
    "cav_directional_derivative",
    "deep_lift",
    "demo_attention_layers",
    "diverging_hex",
    "feature_permutation",
    "fgsm_attack",
    "gradient_shap",
    "grayscale_hex",
    "infidelity",
    "input_x_gradient",
    "integrated_gradients",
    "kernel_shap",
    "layer_activation_summary",
    "layer_activation_trace",
    "layer_conductance",
    "lime_tabular",
    "lrp",
    "noise_tunnel",
    "numpy_layer_stack",
    "occlusion_attributions",
    "pgd_attack",
    "positive_negative_split",
    "sensitivity_max",
    "shapley_value",
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
    return (
        f"#{round(_clamp01(r) * 255):02x}"
        f"{round(_clamp01(g) * 255):02x}"
        f"{round(_clamp01(b) * 255):02x}"
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


# ---------------------------------------------------------------------------
# Round 187 -- the captum extensions (scratch/gh187_review.md section 1).
# Everything below is a pure-numpy, teaching-scale port of the captum
# surfaces round 185 left out: gradient SHAP, DeepLIFT's rescale rule,
# the SHAP-kernel and LIME-kernel regressions, Shapley sampling,
# feature permutation, the noise tunnel, LRP's epsilon rule,
# per-neuron conductance, TCAV's CAV probes, the FGSM/PGD attackers
# and the attribution validators.  The torch plumbing is gone; the
# ALGORITHMS are kept, on the file's own LayerStack machinery.
# ---------------------------------------------------------------------------

#: captum's ``inf_weight``: the surrogate weight pinned on the empty and
#: full coalitions so the kernel regression satisfies both endpoint
#: constraints exactly.
_SHAP_ENDPOINT_WEIGHT: float = 1000000.0

#: The three smoothing readings captum's NoiseTunnel accepts.
_NOISE_TUNNEL_TYPES: tuple[str, ...] = ("smoothgrad", "smoothgrad_sq", "vargrad")


def _row_noise_spread(row: np.ndarray, ratio: float) -> float:
    """Gaussian spread for input noise: ``ratio`` x the row's own std.

    Mirrors :func:`smoothgrad_attributions` (never zero, so a constant
    row still perturbs).
    """
    scale = float(np.std(row))
    return ratio * scale if scale > 0.0 else 1e-6


def _background_matrix(X: Any, n_features: int) -> np.ndarray:
    """Validate a 2-D background matrix against one row's width."""
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise XaiError("the background matrix must be 2-D with at least one row")
    if matrix.shape[1] != n_features:
        raise XaiError(
            f"the background has {matrix.shape[1]} features but the row has "
            f"{n_features}"
        )
    if not np.all(np.isfinite(matrix)):
        raise XaiError("the background matrix carries NaN/inf")
    return matrix


def _concept_input_matrix(X: Any, n_features: int) -> np.ndarray:
    """Validate a 2-D concept/random activation-probe set (>= 2 rows)."""
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise XaiError("the concept/random sets need at least 2 rows each (2-D)")
    if matrix.shape[1] != n_features:
        raise XaiError(
            f"the set has {matrix.shape[1]} features but the row has {n_features}"
        )
    if not np.all(np.isfinite(matrix)):
        raise XaiError("the concept/random sets carry NaN/inf")
    return matrix


def _batch_scorer(
    estimator: Any, row_size: int, output_index: int
) -> Callable[[np.ndarray], float]:
    """A mean-per-row scalar scorer over a batch -- occlusion's dual contract, batched.

    The numpy engine scores through its own forward; every other engine
    scores through ``predict`` (regression) or ``predict_proba``
    (classification, ``output_index`` picks the class column) -- the
    same ladder :func:`_occlusion_score` walks, lifted one dimension so
    the permutation family can score whole datasets.
    """
    try:
        stack = numpy_layer_stack(estimator)
    except XaiError:
        stack = None
    if stack is not None:
        if row_size != stack.n_features:
            raise XaiError(
                f"the row has {row_size} features but the network expects "
                f"{stack.n_features}"
            )
        idx = int(output_index)
        width = int(np.asarray(_forward_trace(stack, np.zeros((1, row_size)))[0]).size)
        if idx < 0 or idx >= width:
            raise XaiError(
                f"output_index {idx} out of range -- this row produces "
                f"{width} output(s)"
            )

        def score_via_stack(batch: np.ndarray) -> float:
            logits, _trace = _forward_trace(stack, batch)
            return float(np.mean(np.asarray(logits, dtype=float)[:, idx]))

        return score_via_stack

    if hasattr(estimator, "predict_proba"):
        idx = int(output_index)
        probe = np.asarray(
            estimator.predict_proba(np.zeros((1, row_size))), dtype=float
        ).reshape(-1)
        if idx < 0 or idx >= probe.size:
            raise XaiError(
                f"output_index {idx} out of range -- predict_proba "
                f"offers {probe.size} class column(s)"
            )

        def score_via_proba(batch: np.ndarray) -> float:
            return float(
                np.mean(np.asarray(estimator.predict_proba(batch), dtype=float)[:, idx])
            )

        return score_via_proba

    if hasattr(estimator, "predict"):
        idx = int(output_index)
        if idx != 0:
            raise XaiError("a regressor offers one output -- output_index must be 0")

        def score_via_predict(batch: np.ndarray) -> float:
            values = np.asarray(estimator.predict(batch), dtype=float).reshape(
                len(batch), -1
            )
            return float(np.mean(values[:, 0]))

        return score_via_predict

    raise XaiError(
        "feature permutation needs a numpy-engine estimator or something exposing "
        "predict/predict_proba"
    )


def _normalize_hidden_index(stack: LayerStack, layer_index: int) -> int:
    """Python-style hidden-layer indexing (negatives wrap) with a typed refusal."""
    hidden = len(stack.layers) - 1
    if hidden < 1:
        raise XaiError("this needs a stack with at least one hidden layer")
    idx = int(layer_index) + hidden if int(layer_index) < 0 else int(layer_index)
    if idx < 0 or idx >= hidden:
        raise XaiError(
            f"layer_index {int(layer_index)} out of range -- the stack has "
            f"{hidden} hidden layer(s)"
        )
    return idx


def _tail_forward(
    stack: LayerStack, layer_index: int, activations: np.ndarray
) -> np.ndarray:
    """Forward from the post-activation of hidden ``layer_index`` to the logits.

    ``activations`` may be 1-D (one row) or 2-D (a batch of perturbed
    activations -- the finite-difference probes stack one perturbation
    per row).
    """
    out = activations
    remaining = stack.layers[layer_index + 1 :]
    activation = _np_activation(stack.activation)
    for position, (weight, bias) in enumerate(remaining):
        out = out @ weight.T + (bias if bias is not None else 0.0)
        if position < len(remaining) - 1:
            out = activation(out)
    return out


def _activation_output_gradient(
    stack: LayerStack, batch: np.ndarray, layer_index: int, output_index: int
) -> np.ndarray:
    """Exact d output/output_index / d (activation of hidden ``layer_index``).

    The analytic backward walk stops at the chosen layer instead of the
    input -- the per-neuron sensitivities TCAV contracts against its
    CAV.  One output row per batch row.
    """
    idx = _normalize_hidden_index(stack, layer_index)
    logits, trace = _forward_trace(stack, batch)
    values = np.asarray(logits, dtype=float)
    out_idx = int(output_index)
    if out_idx < 0 or out_idx >= values.shape[-1]:
        raise XaiError(
            f"output_index {out_idx} out of range -- this row produces "
            f"{values.shape[-1]} output(s)"
        )
    delta = np.zeros_like(values)
    delta[..., out_idx] = 1.0
    for i in range(len(stack.layers) - 1, idx, -1):
        weight, _bias = stack.layers[i]
        delta = delta @ weight
        if i - 1 > idx:
            z, a = trace[i - 1]
            delta = delta * _activation_derivative(stack.activation, z, a)
    return delta


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Competition-averaged 1-based ranks (tie-safe), pure numpy."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    index = 0
    while index < values.size:
        stop = index
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman rank correlation: Pearson on the average ranks (0.0 if flat)."""
    ranks_first = _average_ranks(first)
    ranks_second = _average_ranks(second)
    if float(np.std(ranks_first)) == 0.0 or float(np.std(ranks_second)) == 0.0:
        return 0.0
    return float(np.corrcoef(ranks_first, ranks_second)[0, 1])


def gradient_shap(
    estimator: Any,
    row: Any,
    n_baseline: int = 16,
    n_samples: int = 16,
    seed: int | None = None,
    output_index: int = 0,
) -> AttributionResult:
    """Gradient SHAP: the expectation of (input - baseline) x gradient (captum).

    ``n_baseline`` baselines are drawn from a zero-centred gaussian
    whose spread follows the row's own scale (captum's
    ``draw_baseline_from_distrib`` reading of the zero base); each of
    ``n_samples`` rounds then draws a fresh gaussian perturbation of
    the input, a uniform point ``alpha`` along the baseline->input
    path, and accumulates ``(x_noisy - baseline) * grad`` evaluated at
    the interpolated point -- the SmoothGrad-style expectation captum's
    GradientSHAP runs.

    >>> net = _demo_network()
    >>> result = gradient_shap(net, [0.5, -1.0, 2.0], n_baseline=4, n_samples=8, seed=0)
    >>> result.method, result.kind, result.convergence_delta
    ('gradient_shap', 'gradient', None)
    >>> again = gradient_shap(net, [0.5, -1.0, 2.0], n_baseline=4, n_samples=8, seed=0)
    >>> np.allclose(result.attributions, again.attributions)
    True
    """
    n_baseline = int(n_baseline)
    n_samples = int(n_samples)
    if n_baseline < 1:
        raise XaiError("gradient shap needs n_baseline >= 1")
    if n_samples < 1:
        raise XaiError("gradient shap needs n_samples >= 1")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    rng = np.random.default_rng(seed)
    spread = _row_noise_spread(x, 0.1)
    baselines = rng.normal(0.0, spread, size=(n_baseline, x.size))
    total = np.zeros_like(x)
    for sample in range(n_samples):
        noisy = x + rng.normal(0.0, spread, size=x.shape)
        base = baselines[sample % n_baseline]
        alpha = float(rng.uniform(0.0, 1.0))
        point = base + alpha * (noisy - base)
        grad, _value = _input_gradient(stack, _as_row(point), output_index)
        total += (noisy - base) * grad
    return AttributionResult(
        attributions=total / n_samples,
        method="gradient_shap",
        kind="gradient",
        note=(
            f"gradient shap: {n_samples} samples over {n_baseline} "
            f"baselines, seed={seed}"
        ),
    )


def deep_lift(
    estimator: Any,
    row: Any,
    baseline: Any = None,
    output_index: int = 0,
) -> AttributionResult:
    """DeepLIFT rescale rule over the layer stack (captum DeepLiftRescale).

    Every hidden activation contributes its rescale multiplier
    ``(a(x) - a(baseline)) / (z(x) - z(baseline))`` and the linear maps
    chain it backwards, so the attributions sum to ``f(x) - f(baseline)``
    by construction -- reported as ``convergence_delta``.  Where the
    rescale denominator vanishes the multiplier falls back to the
    activation's local derivative at ``x`` (captum's stabilised delta
    reading): the activation is handled through its derivative at the
    rescale points, exactly like captum's linear-layer rules.

    >>> net = _demo_network()
    >>> result = deep_lift(net, [0.5, -1.0, 2.0])
    >>> result.method, result.kind
    ('deep_lift', 'gradient')
    >>> abs(result.convergence_delta) < 1e-8
    True
    """
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    if baseline is None:
        base = np.zeros_like(x)
    else:
        base = _as_row(baseline)
        if base.size != x.size:
            raise XaiError(
                f"the baseline has {base.size} features, the row {x.size}"
            )
    logits_x, trace_x = _forward_trace(stack, x)
    logits_b, trace_b = _forward_trace(stack, base)
    values_x = np.asarray(logits_x, dtype=float).reshape(-1)
    values_b = np.asarray(logits_b, dtype=float).reshape(-1)
    idx = int(output_index)
    if idx < 0 or idx >= values_x.size:
        raise XaiError(
            f"output_index {idx} out of range -- this row produces "
            f"{values_x.size} output(s)"
        )
    # rescale multipliers, hidden layer by hidden layer
    multipliers = []
    for (z_x, a_x), (z_b, a_b) in zip(trace_x, trace_b):
        delta_z = np.asarray(z_x, dtype=float) - np.asarray(z_b, dtype=float)
        delta_a = np.asarray(a_x, dtype=float) - np.asarray(a_b, dtype=float)
        usable = np.abs(delta_z) > 1e-9
        local = _activation_derivative(stack.activation, z_x, a_x)
        multipliers.append(
            np.where(usable, delta_a / np.where(usable, delta_z, 1.0), local)
        )
    # backward: the output multiplier walks the maps down to the input
    backward = np.zeros(values_x.size)
    backward[idx] = 1.0
    backward = backward @ stack.layers[-1][0]  # wrt the last hidden activation
    for i in range(len(stack.layers) - 2, -1, -1):
        backward = backward * multipliers[i]  # wrt the hidden pre-activation z_i
        backward = backward @ stack.layers[i][0]  # wrt a_{i-1} (the input when i == 0)
    attributions = (x - base) * backward
    delta = float(values_x[idx] - values_b[idx] - attributions.sum())
    return AttributionResult(
        attributions=attributions,
        method="deep_lift",
        kind="gradient",
        convergence_delta=delta,
        note="deep lift rescale rule",
    )


def kernel_shap(
    estimator: Any,
    X: Any,
    row: Any,
    n_coalitions: int | None = None,
    output_index: int = 0,
    seed: int | None = None,
) -> AttributionResult:
    """Kernel SHAP: Shapley-kernel weighted least squares over coalitions (captum).

    Coalitions ``z`` in {0,1}^M mask the row onto the background (the
    per-column means of ``X``) and are scored; the surrogate
    ``f(z) ~ phi_0 + phi . z`` is fit by weighted least squares with the
    Shapley kernel ``k(|z|) = (M-1) / (C(M,|z|) |z| (M-|z|))``.  The
    empty and full coalitions always ride along with captum's
    ``inf_weight`` (1e6), pinning ``phi_0 = f(background)`` and
    ``phi_0 + sum(phi) = f(x)`` -- so ``convergence_delta`` collapses
    towards zero.  Coalition sizes are sampled from captum's size
    distribution ``p(k) ~ (M-1) / (k (M-k))``; runs on every engine
    through the occlusion scorer's predict/predict_proba contract.

    >>> net = _demo_network()
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(12, 3))
    >>> result = kernel_shap(net, X, [0.5, -1.0, 2.0], n_coalitions=8, seed=0)
    >>> result.method, result.kind
    ('kernel_shap', 'perturbation')
    >>> result.attributions.shape
    (3,)
    >>> abs(result.convergence_delta) < 1e-4
    True
    """
    x = _as_row(row)
    background = _background_matrix(X, x.size)
    score, _n_features = _occlusion_score(estimator, x, output_index)
    M = int(x.size)
    if n_coalitions is None:
        n_coalitions = max(min(2**M - 2, 32), 0)
    n_coalitions = int(n_coalitions)
    if n_coalitions < 1 and M > 1:
        raise XaiError("kernel shap needs n_coalitions >= 1 beyond the two endpoints")
    rng = np.random.default_rng(seed)
    column_means = background.mean(axis=0)
    f_full = score(x)
    f_empty = score(column_means)
    if M == 1:
        return AttributionResult(
            attributions=np.array([f_full - f_empty]),
            method="kernel_shap",
            kind="perturbation",
            convergence_delta=0.0,
            note="kernel shap: single feature, endpoints only",
        )
    sizes = np.arange(1, M)
    size_p = (M - 1.0) / (sizes * (M - sizes))
    size_p = size_p / size_p.sum()
    rows = [np.ones(M, dtype=float), np.zeros(M, dtype=float)]
    values = [f_full, f_empty]
    weights = [_SHAP_ENDPOINT_WEIGHT, _SHAP_ENDPOINT_WEIGHT]
    for _coalition in range(n_coalitions):
        k = int(rng.choice(sizes, p=size_p))
        picked = rng.choice(M, size=k, replace=False)
        z = np.zeros(M)
        z[picked] = 1.0
        masked = x.copy()
        masked[z == 0.0] = column_means[z == 0.0]
        rows.append(z)
        values.append(score(masked))
        weights.append((M - 1.0) / (math.comb(M, k) * k * (M - k)))
    design = np.column_stack([np.ones(len(rows)), np.asarray(rows, dtype=float)])
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))
    coef, *_residuals = np.linalg.lstsq(
        design * sqrt_weights[:, None], np.asarray(values) * sqrt_weights, rcond=None
    )
    attributions = coef[1:]
    delta = f_full - f_empty - float(attributions.sum())
    return AttributionResult(
        attributions=attributions,
        method="kernel_shap",
        kind="perturbation",
        convergence_delta=delta,
        note=(
            f"kernel shap: {n_coalitions} sampled coalitions + endpoints, "
            f"background=column means, seed={seed}"
        ),
    )


def lime_tabular(
    estimator: Any,
    X: Any,
    row: Any,
    n_samples: int = 64,
    kernel_width: float = 0.75,
    seed: int | None = None,
    output_index: int = 0,
) -> AttributionResult:
    """LIME: the local linear surrogate with an exponential kernel (captum).

    ``n_samples`` gaussian perturbations of the row (per-feature std =
    the background's own column stds) are scored, weighted by the
    similarity kernel ``exp(-d^2 / kernel_width^2)`` with ``d`` the
    euclidean distance to the row, and the weighted least squares of
    the scores on the displacement matrix gives the per-feature local
    slopes.  The displacement matrix is mean-centred before the fit --
    captum's NormLayer reading -- which stabilises the intercept
    without touching the slopes.  Runs on every engine through the
    occlusion scorer's predict/predict_proba contract.

    >>> net = _demo_network()
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(16, 3))
    >>> result = lime_tabular(net, X, [0.5, -1.0, 2.0], n_samples=24, seed=0)
    >>> result.method, result.kind
    ('lime_tabular', 'perturbation')
    >>> result.attributions.shape
    (3,)
    >>> bool(np.all(np.isfinite(result.attributions)))
    True
    """
    n_samples = int(n_samples)
    if n_samples < 2:
        raise XaiError("lime needs n_samples >= 2 to fit the local surrogate")
    if float(kernel_width) <= 0.0:
        raise XaiError("kernel_width must be > 0")
    x = _as_row(row)
    background = _background_matrix(X, x.size)
    score, _n_features = _occlusion_score(estimator, x, output_index)
    rng = np.random.default_rng(seed)
    column_std = background.std(axis=0)
    column_std = np.where(column_std > 0.0, column_std, 1.0)
    perturbed = x + rng.normal(size=(n_samples, x.size)) * column_std
    distances = np.linalg.norm(perturbed - x, axis=1)
    sample_weights = np.exp(-((distances**2) / float(kernel_width) ** 2))
    targets = np.array([score(sample) for sample in perturbed])
    displacements = perturbed - x
    centred = displacements - displacements.mean(axis=0)
    sqrt_weights = np.sqrt(sample_weights)
    design = np.column_stack([np.ones(n_samples), centred])
    coef, *_residuals = np.linalg.lstsq(
        design * sqrt_weights[:, None], targets * sqrt_weights, rcond=None
    )
    return AttributionResult(
        attributions=coef[1:],
        method="lime_tabular",
        kind="perturbation",
        note=(
            f"lime: {n_samples} samples, kernel_width={float(kernel_width):g}, "
            f"seed={seed}"
        ),
    )


def shapley_value(
    estimator: Any,
    X: Any,
    row: Any,
    n_permutations: int = 32,
    seed: int | None = None,
    output_index: int = 0,
) -> AttributionResult:
    """Shapley values by permutation sampling (captum ShapleyValueSampling).

    ``n_permutations`` random feature orderings walk the row up from
    the background (the per-column means of ``X``); each feature
    collects its marginal contribution at the position the ordering
    gives it, and the per-permutation sums telescope to
    ``f(x) - f(background)`` -- so ``convergence_delta`` is exact up to
    float error regardless of the sample count.  Runs on every engine
    through the occlusion scorer's predict/predict_proba contract.

    >>> net = _demo_network()
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(12, 3))
    >>> result = shapley_value(net, X, [0.5, -1.0, 2.0], n_permutations=8, seed=0)
    >>> result.method, result.kind
    ('shapley_value', 'perturbation')
    >>> abs(result.convergence_delta) < 1e-9
    True
    """
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise XaiError("shapley value needs n_permutations >= 1")
    x = _as_row(row)
    background = _background_matrix(X, x.size)
    score, M = _occlusion_score(estimator, x, output_index)
    rng = np.random.default_rng(seed)
    masked_row = background.mean(axis=0)
    f_empty = score(masked_row)
    f_full = score(x)
    totals = np.zeros(M)
    for _permutation in range(n_permutations):
        order = rng.permutation(M)
        current = masked_row.copy()
        previous = f_empty
        for feature in order:
            current[feature] = x[feature]
            value = score(current)
            totals[feature] += value - previous
            previous = value
    attributions = totals / n_permutations
    delta = f_full - f_empty - float(attributions.sum())
    return AttributionResult(
        attributions=attributions,
        method="shapley_value",
        kind="perturbation",
        convergence_delta=delta,
        note=(
            f"shapley sampling: {n_permutations} permutations, "
            f"background=column means, seed={seed}"
        ),
    )


def feature_permutation(
    estimator: Any,
    X: Any,
    row: Any,
    n_repeats: int = 16,
    seed: int | None = None,
    output_index: int = 0,
) -> AttributionResult:
    """Permutation importance: the score drop when a column is shuffled (captum).

    Each feature is permuted across ``X``'s rows ``n_repeats`` times
    (captum redraws the identity permutation) and the mean drop of the
    mean-per-row score is recorded -- the unsupervised reading of
    captum's FeaturePermutation (no labels needed: the model's own
    score drops instead of a labelled metric).  The estimator goes
    through the same dual contract occlusion uses: the numpy engine's
    layer stack when available, otherwise any ``predict`` /
    ``predict_proba`` object.  ``row`` fixes the feature count and the
    AttributionResult's shape; the values are dataset-level.

    >>> net = _demo_network()
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(20, 3))
    >>> result = feature_permutation(net, X, [0.5, -1.0, 2.0], n_repeats=4, seed=0)
    >>> result.method, result.kind
    ('feature_permutation', 'perturbation')
    >>> result.attributions.shape
    (3,)
    """
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise XaiError("feature permutation needs n_repeats >= 1")
    x = _as_row(row)
    matrix = _background_matrix(X, x.size)
    if matrix.shape[0] < 2:
        raise XaiError("feature permutation needs at least 2 rows to permute")
    score = _batch_scorer(estimator, x.size, output_index)
    baseline_score = score(matrix)
    rng = np.random.default_rng(seed)
    identity = np.arange(matrix.shape[0])
    drops = np.zeros(x.size)
    for feature in range(x.size):
        for _repeat in range(n_repeats):
            order = rng.permutation(matrix.shape[0])
            while np.array_equal(order, identity):
                order = rng.permutation(matrix.shape[0])
            permuted = matrix.copy()
            permuted[:, feature] = matrix[order, feature]
            drops[feature] += baseline_score - score(permuted)
    return AttributionResult(
        attributions=drops / n_repeats,
        method="feature_permutation",
        kind="perturbation",
        note=(
            f"feature permutation: {n_repeats} repeats over "
            f"{matrix.shape[0]} rows, seed={seed}"
        ),
    )


def noise_tunnel(
    estimator: Any,
    row: Any,
    method: str = "smoothgrad",
    n_noise: int = 16,
    noise_scale: float = 0.1,
    seed: int | None = None,
    output_index: int = 0,
) -> AttributionResult:
    """captum's NoiseTunnel: gaussian noise around the three gradient readings.

    ``n_noise`` noisy copies of the row (per-feature std =
    ``noise_scale`` x the row's own scale) go through the exact
    analytic backprop, then ``method`` aggregates: ``smoothgrad`` the
    mean attribution, ``smoothgrad_sq`` the mean of squares,
    ``vargrad`` the variance.  With ``method="smoothgrad"`` this is
    exactly :func:`smoothgrad_attributions` under the same seed.

    >>> net = _demo_network()
    >>> result = noise_tunnel(
    ...     net, [0.5, -1.0, 2.0], n_noise=8, noise_scale=0.2, seed=0
    ... )
    >>> result.method, result.kind
    ('noise_tunnel', 'gradient')
    >>> smooth = smoothgrad_attributions(net, [0.5, -1.0, 2.0], n_samples=8, seed=0)
    >>> np.allclose(result.attributions, smooth.attributions)
    True
    """
    if method not in _NOISE_TUNNEL_TYPES:
        raise XaiError(
            f"unknown noise tunnel type {method!r} -- expected one of "
            f"{_NOISE_TUNNEL_TYPES}"
        )
    n_noise = int(n_noise)
    if n_noise < 1:
        raise XaiError("noise tunnel needs n_noise >= 1")
    if float(noise_scale) < 0.0:
        raise XaiError("noise_scale must be >= 0")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    rng = np.random.default_rng(seed)
    spread = _row_noise_spread(x, float(noise_scale))
    samples = np.empty((n_noise, x.size))
    squares = np.empty((n_noise, x.size))
    for sample in range(n_noise):
        noisy = x + rng.normal(0.0, spread, size=x.shape)
        grad, _value = _input_gradient(stack, _as_row(noisy), output_index)
        samples[sample] = grad
        squares[sample] = grad * grad
    mean = samples.mean(axis=0)
    mean_square = squares.mean(axis=0)
    if method == "smoothgrad":
        attributions = mean
    elif method == "smoothgrad_sq":
        attributions = mean_square
    else:
        attributions = mean_square - mean * mean
    return AttributionResult(
        attributions=attributions,
        method="noise_tunnel",
        kind="gradient",
        note=(
            f"noise tunnel {method}: {n_noise} samples, "
            f"noise_scale={float(noise_scale):g}, seed={seed}"
        ),
    )


def lrp(
    estimator: Any, row: Any, eps: float = 1e-6, output_index: int = 0
) -> AttributionResult:
    """Layer-wise relevance propagation with the epsilon rule (captum LRP).

    Relevance starts at the chosen output (the prediction itself) and
    each linear layer redistributes it by ``r_in = x_in * (W^T (r / z))``
    where ``z`` is the layer's pre-activation, stabilised to
    ``z + eps`` for ``z >= 0`` and ``z - eps`` otherwise (the epsilon
    rule: the denominator never gets closer to zero than ``eps``).
    Relevance leaking through the bias terms is dropped, so the sum of
    the attributions conserves ``f(x)`` up to that leak -- reported as
    ``convergence_delta``, captum's own LRP completeness residual.  The
    activation between layers is treated as pass-through for relevance
    routing (exact on ReLU's active branch, the standard composite).

    >>> net = _demo_network()
    >>> result = lrp(net, [0.5, -1.0, 2.0])
    >>> result.method, result.kind
    ('lrp', 'gradient')
    >>> result.attributions.shape
    (3,)
    >>> bool(np.all(np.isfinite(result.attributions)))
    True
    """
    if float(eps) <= 0.0:
        raise XaiError("lrp needs eps > 0 for the epsilon rule")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    logits, trace = _forward_trace(stack, x)
    values = np.asarray(logits, dtype=float).reshape(-1)
    idx = int(output_index)
    if idx < 0 or idx >= values.size:
        raise XaiError(
            f"output_index {idx} out of range -- this row produces "
            f"{values.size} output(s)"
        )
    relevance = np.zeros(values.size)
    relevance[idx] = values[idx]
    n_layers = len(stack.layers)
    for layer in range(n_layers - 1, -1, -1):
        weight, _bias = stack.layers[layer]
        z = (
            values
            if layer == n_layers - 1
            else np.asarray(trace[layer][0], dtype=float).reshape(-1)
        )
        x_in = (
            x if layer == 0 else np.asarray(trace[layer - 1][1], dtype=float).reshape(-1)
        )
        denominator = np.where(z >= 0.0, z + float(eps), z - float(eps))
        relevance = x_in * ((relevance / denominator) @ weight)
    delta = float(values[idx] - relevance.sum())
    return AttributionResult(
        attributions=relevance,
        method="lrp",
        kind="gradient",
        convergence_delta=delta,
        note=f"lrp epsilon rule, eps={float(eps):g}",
    )


def layer_conductance(
    estimator: Any,
    row: Any,
    layer_index: int = 0,
    baseline: Any = None,
    n_samples: int = 16,
    seed: int | None = None,
    output_index: int = 0,
) -> list[dict[str, float]]:
    """Per-neuron conductance on one hidden layer (captum LayerConductance).

    Midpoint interpolation from the baseline (zero unless given) to the
    row; each step contributes ``(act(x_step) - act(baseline)) *
    d(output)/d(z_neuron)`` with the pre-activation derivative taken by
    central finite differences through :func:`_forward_trace` (the tail
    forward is re-run with the layer's ``z`` nudged one neuron at a
    time).  Returns the per-neuron list
    ``{"neuron", "conductance", "mean_activation", "mean_gradient"}``
    in the shape :func:`layer_activation_summary` returns -- position
    in the list is the neuron index.  ``seed`` is accepted for
    signature symmetry with the noisy siblings; the midpoint
    interpolation is deterministic.

    >>> net = _demo_network()
    >>> neurons = layer_conductance(net, [0.5, -1.0, 2.0], n_samples=8)
    >>> len(neurons)
    4
    >>> neurons[0]["neuron"]
    0.0
    >>> bool(np.isfinite(neurons[0]["conductance"]))
    True
    """
    n_samples = int(n_samples)
    if n_samples < 1:
        raise XaiError("layer conductance needs n_samples >= 1")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    idx = _normalize_hidden_index(stack, layer_index)
    if baseline is None:
        base = np.zeros_like(x)
    else:
        base = _as_row(baseline)
        if base.size != x.size:
            raise XaiError(
                f"the baseline has {base.size} features, the row {x.size}"
            )
    values = np.asarray(_forward_trace(stack, x)[0], dtype=float).reshape(-1)
    out_idx = int(output_index)
    if out_idx < 0 or out_idx >= values.size:
        raise XaiError(
            f"output_index {out_idx} out of range -- this row produces "
            f"{values.size} output(s)"
        )
    activation = _np_activation(stack.activation)
    a_base = np.asarray(_forward_trace(stack, base)[1][idx][1], dtype=float).reshape(-1)
    units = int(a_base.size)
    probes = np.eye(units) * 1e-6
    total = np.zeros(units)
    total_activation = np.zeros(units)
    total_gradient = np.zeros(units)
    for step in range(n_samples):
        alpha = (step + 0.5) / n_samples
        point = base + alpha * (x - base)
        _logits, trace = _forward_trace(stack, point)
        z_h = np.asarray(trace[idx][0], dtype=float).reshape(-1)
        a_h = np.asarray(trace[idx][1], dtype=float).reshape(-1)
        plus = _tail_forward(stack, idx, activation(z_h[None, :] + probes))[:, out_idx]
        minus = _tail_forward(stack, idx, activation(z_h[None, :] - probes))[:, out_idx]
        dy_dz = (plus - minus) / 2e-6
        total += (a_h - a_base) * dy_dz
        total_activation += a_h
        total_gradient += dy_dz
    return [
        {
            "neuron": float(neuron),
            "conductance": float(total[neuron] / n_samples),
            "mean_activation": float(total_activation[neuron] / n_samples),
            "mean_gradient": float(total_gradient[neuron] / n_samples),
        }
        for neuron in range(units)
    ]


def cav_directional_derivative(
    estimator: Any,
    X_concept: Any,
    X_random: Any,
    row: Any,
    layer_index: int = -1,
    output_index: int = 0,
) -> dict[str, float | list[float]]:
    """TCAV's CAV probe: separating direction + directional derivative (captum.concept).

    The CAV is the Fisher linear discriminant separating the concept
    activations from the random activations at the chosen hidden layer
    (numpy closed form -- captum trains a logistic probe; the
    discriminant is the same separating direction without the iterative
    fit).  Returns ``{"cav": <unit direction as a list>,
    "directional_derivative": <v . d(output)/d(activation) for the row>,
    "tcav_score": <fraction of RANDOM-set rows whose directional
    derivative along the CAV is negative>}``.  The scalar sensitivity is
    the classic TCAV reading of the input-space ``v . d(activations)/d
    input`` contracted with the output gradient through the chain rule.

    >>> net = _demo_network()
    >>> rng = np.random.default_rng(3)
    >>> concept = rng.normal(size=(12, 3)) + np.array([2.0, 0.0, 0.0])
    >>> random_set = rng.normal(size=(12, 3))
    >>> report = cav_directional_derivative(net, concept, random_set, [0.5, -1.0, 2.0])
    >>> sorted(report)
    ['cav', 'directional_derivative', 'tcav_score']
    >>> len(report["cav"])
    4
    >>> 0.0 <= report["tcav_score"] <= 1.0
    True
    """
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    idx = _normalize_hidden_index(stack, layer_index)
    concept = _concept_input_matrix(X_concept, x.size)
    random_rows = _concept_input_matrix(X_random, x.size)
    activations_concept = np.asarray(
        _forward_trace(stack, concept)[1][idx][1], dtype=float
    )
    activations_random = np.asarray(
        _forward_trace(stack, random_rows)[1][idx][1], dtype=float
    )
    units = int(activations_concept.shape[1])
    within = np.cov(activations_concept, rowvar=False) + np.cov(
        activations_random, rowvar=False
    )
    ridge = 1e-6 * max(1.0, float(np.trace(within)) / units)
    try:
        cav = np.linalg.solve(
            within + ridge * np.eye(units),
            activations_concept.mean(axis=0) - activations_random.mean(axis=0),
        )
    except np.linalg.LinAlgError as exc:
        raise XaiError(
            "the CAV system is singular -- the activations are degenerate"
        ) from exc
    norm = float(np.linalg.norm(cav))
    if norm <= 0.0:
        raise XaiError("the CAV collapsed to zero -- indistinguishable activation sets")
    cav = cav / norm
    gradient_row = _activation_output_gradient(
        stack, x, idx, output_index
    ).reshape(-1)
    gradients_random = _activation_output_gradient(stack, random_rows, idx, output_index)
    return {
        "cav": [float(component) for component in cav],
        "directional_derivative": float(cav @ gradient_row),
        "tcav_score": float(np.mean((gradients_random @ cav) < 0.0)),
    }


def fgsm_attack(
    estimator: Any, row: Any, eps: float = 0.1, output_index: int = 0
) -> np.ndarray:
    """FGSM: one signed-gradient step inside the L-infinity ball (captum.robust).

    ``x' = clip(x + eps * sign(grad))`` with the clip projecting onto
    the ``eps``-ball around the row (captum's clamp to the attack
    bounds, with the ball the attack is defined on).  The estimator
    goes through the numpy stack's analytic :func:`_input_gradient`.

    >>> net = _demo_network()
    >>> row = np.array([0.5, -1.0, 2.0])
    >>> adversarial = fgsm_attack(net, row, eps=0.1)
    >>> bool(np.all(np.isfinite(adversarial)))
    True
    >>> float(np.max(np.abs(adversarial - row))) <= 0.1 + 1e-12
    True
    """
    if float(eps) <= 0.0:
        raise XaiError("fgsm needs eps > 0")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    grad, _value = _input_gradient(stack, x, output_index)
    radius = float(eps)
    return np.clip(x + radius * np.sign(grad), x - radius, x + radius)


def pgd_attack(
    estimator: Any,
    row: Any,
    eps: float = 0.1,
    alpha: float = 0.02,
    n_steps: int = 20,
    output_index: int = 0,
) -> np.ndarray:
    """PGD: iterated signed-gradient steps, projected each step (captum.robust).

    ``x_{t+1} = clip(x_t + alpha * sign(grad), x - eps, x + eps)`` for
    ``n_steps`` steps from the row itself (deterministic -- captum's
    random ball start is the only stochastic piece and the signature
    carries no seed).  The clip is the L-infinity projection.

    >>> net = _demo_network()
    >>> row = np.array([0.5, -1.0, 2.0])
    >>> adversarial = pgd_attack(net, row, eps=0.1, alpha=0.02, n_steps=5)
    >>> bool(np.all(np.isfinite(adversarial)))
    True
    >>> float(np.max(np.abs(adversarial - row))) <= 0.1 + 1e-12
    True
    """
    if float(eps) <= 0.0:
        raise XaiError("pgd needs eps > 0")
    if float(alpha) <= 0.0:
        raise XaiError("pgd needs alpha > 0")
    if int(n_steps) < 1:
        raise XaiError("pgd needs n_steps >= 1")
    stack = numpy_layer_stack(estimator)
    x = _as_row(row)
    if x.size != stack.n_features:
        raise XaiError(
            f"the row has {x.size} features but the network expects "
            f"{stack.n_features}"
        )
    radius = float(eps)
    step = float(alpha)
    adversarial = x.copy()
    for _step_index in range(int(n_steps)):
        grad, _value = _input_gradient(stack, adversarial, output_index)
        adversarial = np.clip(
            adversarial + step * np.sign(grad), x - radius, x + radius
        )
    return adversarial


def sensitivity_max(
    attribution_fn: Callable[[np.ndarray], np.ndarray],
    row: Any,
    n_noise: int = 16,
    noise_scale: float = 0.1,
    seed: int | None = None,
) -> float:
    """Max sensitivity: the largest attribution change under noise (captum.metrics).

    Monte-Carlo over ``n_noise`` gaussian perturbations with spread
    ``noise_scale`` x the row's own scale (captum samples an
    L-infinity ball with an absolute radius; the scale-relative
    gaussian matches the file's smoothgrad convention).  Each sample
    contributes the 2-norm of the attribution change normalised by the
    reference attribution's 2-norm (a zero reference norm counts as 1),
    and the maximum is returned -- a constant attribution function
    scores exactly 0.0.

    >>> sensitivity_max(lambda values: np.ones(3), [0.5, -1.0, 2.0], n_noise=4, seed=0)
    0.0
    """
    x = _as_row(row)
    n_noise = int(n_noise)
    if n_noise < 1:
        raise XaiError("sensitivity_max needs n_noise >= 1")
    if float(noise_scale) < 0.0:
        raise XaiError("noise_scale must be >= 0")
    reference = np.asarray(attribution_fn(x), dtype=float).reshape(-1)
    if reference.size != x.size:
        raise XaiError(
            f"the attribution_fn returned {reference.size} values for a "
            f"{x.size}-feature row"
        )
    norm = float(np.linalg.norm(reference))
    if norm == 0.0:
        norm = 1.0
    rng = np.random.default_rng(seed)
    spread = _row_noise_spread(x, float(noise_scale))
    worst = 0.0
    for _sample in range(n_noise):
        noisy = x + rng.normal(0.0, spread, size=x.shape)
        attr = np.asarray(attribution_fn(noisy), dtype=float).reshape(-1)
        if attr.size != x.size:
            raise XaiError(
                f"the attribution_fn returned {attr.size} values for a "
                f"{x.size}-feature row"
            )
        worst = max(worst, float(np.linalg.norm(attr - reference)) / norm)
    return worst


def infidelity(
    attribution_fn: Callable[[np.ndarray], np.ndarray],
    estimator: Any,
    row: Any,
    n_noise: int = 16,
    noise_scale: float = 0.1,
    seed: int | None = None,
    output_index: int = 0,
) -> float:
    """Infidelity: E[(I . (x - x') - (f(x) - f(x'))^2)] (captum.metrics).

    The expected squared mismatch between the attribution's predicted
    output change under a perturbation and the model's actual change --
    the completeness property turned into a validator.  The model is
    scored through the occlusion contract (numpy stack, else
    predict/predict_proba); the noise spread follows the file's
    smoothgrad convention (``noise_scale`` x the row's own scale).  An
    attribution that equals the exact gradient of an affine region
    scores ~0.

    >>> net = _demo_network()
    >>> row = [0.5, -1.0, 2.0]
    >>> grads = input_x_gradient(net, row).attributions
    >>> value = infidelity(lambda values: grads, net, row, n_noise=8, seed=0)
    >>> bool(np.isfinite(value) and value >= 0.0)
    True
    """
    x = _as_row(row)
    n_noise = int(n_noise)
    if n_noise < 1:
        raise XaiError("infidelity needs n_noise >= 1")
    if float(noise_scale) < 0.0:
        raise XaiError("noise_scale must be >= 0")
    score, _n_features = _occlusion_score(estimator, x, output_index)
    attributions = np.asarray(attribution_fn(x), dtype=float).reshape(-1)
    if attributions.size != x.size:
        raise XaiError(
            f"the attribution_fn returned {attributions.size} values for a "
            f"{x.size}-feature row"
        )
    f_x = score(x)
    rng = np.random.default_rng(seed)
    spread = _row_noise_spread(x, float(noise_scale))
    total = 0.0
    for _sample in range(n_noise):
        noisy = x + rng.normal(0.0, spread, size=x.shape)
        f_noisy = score(noisy)
        residual = float(attributions @ (x - noisy)) - (f_x - f_noisy)
        total += residual * residual
    return total / n_noise


def attribution_effectiveness(
    estimator: Any,
    row: Any,
    attribution: Any,
    n_alternatives: int = 16,
    seed: int | None = None,
    output_index: int = 0,
) -> float:
    """Effectiveness: rank agreement between attributions and deletion drops.

    The randomized-deletion validator of Bhatt, Weller & Moura 2020 --
    captum.metrics ships sensitivity/infidelity but not this one, so it
    completes the file's validator trio.  ``n_alternatives`` alternative
    baselines are drawn from the file's zero-centred gaussian (the
    gradient-shap convention, spread = 0.1 x the row's own scale); each
    gives a per-feature deletion ranking ``f(x) - f(x with feature i
    replaced by the alternative's i-th value)`` (scored through the
    occlusion contract, so every engine works), and the returned score
    is the average Spearman rank correlation between the
    ``|attribution|`` ranking and those deletion rankings.  A constant
    attribution ranks flatly and scores exactly 0.0.

    >>> net = _demo_network()
    >>> row = [0.5, -1.0, 2.0]
    >>> grads = input_x_gradient(net, row).attributions
    >>> value = attribution_effectiveness(net, row, grads, n_alternatives=8, seed=0)
    >>> -1.0 <= value <= 1.0
    True
    >>> flat = attribution_effectiveness(net, row, np.zeros(3))
    >>> flat
    0.0
    """
    x = _as_row(row)
    attributions = _as_row(attribution)
    if attributions.size != x.size:
        raise XaiError(
            f"the attribution carries {attributions.size} values, the row "
            f"{x.size}"
        )
    n_alternatives = int(n_alternatives)
    if n_alternatives < 1:
        raise XaiError("attribution effectiveness needs n_alternatives >= 1")
    score, _n_features = _occlusion_score(estimator, x, output_index)
    f_full = score(x)
    rng = np.random.default_rng(seed)
    spread = _row_noise_spread(x, 0.1)
    total = 0.0
    for _alternative in range(n_alternatives):
        values = rng.normal(0.0, spread, size=x.shape)
        drops = np.empty(x.size)
        for feature in range(x.size):
            masked = x.copy()
            masked[feature] = values[feature]
            drops[feature] = f_full - score(masked)
        total += _spearman(np.abs(attributions), drops)
    return total / n_alternatives

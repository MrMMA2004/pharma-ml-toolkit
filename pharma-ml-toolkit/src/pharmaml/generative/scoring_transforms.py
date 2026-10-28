"""REINVENT4's score-transform DSL, ported to the toolkit's stack.

The instruction (round 100x-164): *"Check the contents and files in
this GitHub link completely and one by one and add the missing items to
playground.py: github.com/MolecularAI/REINVENT4."*  The repository was
cloned and walked file by file (4.8.x, ``reinvent/scoring/transforms/``
and ``reinvent/scoring/aggregators/means.py`` in particular).  What
REINVENT4 computes and this toolkit lacked:

* the **eight score transforms** -- every one reproduced with its exact
  formula: :class:`SigmoidTransform` / :class:`ReverseSigmoidTransform`
  (REINVENT4's base-10 *numerically stable* sigmoid with the ``k``
  rescaling ``10k/(high-low)`` and the hard-step degenerate case
  ``high == low``), :class:`DoubleSigmoidTransform` (the two-sided
  window with ``coef_div`` / ``coef_si`` / ``coef_se``), the three step
  functions (:class:`RightStepTransform`, :class:`LeftStepTransform`,
  :class:`StepTransform`), :class:`ExponentialDecayTransform` (the
  rectified ``exp(-k x)``) and :class:`ValueMappingTransform` (category
  names to floats, unmapped keys to NaN with a warning -- REINVENT4's
  MMP companion);
* the **NaN-aware weighted aggregators** (``arithmetic_mean`` /
  ``geometric_mean`` over ``(scores, weight)`` pairs with NaN-masked
  weight renormalisation -- a NaN component drops out of the
  *denominator* too, exactly ``reinvent/scoring/aggregators/means.py``);
* the **transformable raw components** -- REINVENT4's plugin components
  (``reinvent_plugins/components/RDKit/comp_physchem.py`` and friends)
  exposed as objectives that map SMILES to a *raw* descriptor value and
  push it through any transform: :class:`RawComponentObjective` with the
  :data:`RAW_COMPONENTS` registry (QED, molecular weight, SlogP, TPSA,
  hydrogen-bond donors/acceptors, rotatable bonds, ring counts, heavy
  and hetero atoms, Csp3 fraction, largest ring, graph length, atom
  stereo centres) plus :class:`GroupCountObjective` (``comp_group_count.py``);
* the **filter / penalty semantics** -- :class:`CustomAlertsObjective`
  (``comp_custom_alerts.py``: score 1.0 when no blocklist SMARTS
  matches, 0.0 otherwise -- inside the toolkit's geometric-mean
  :class:`~pharmaml.generative.MultiObjective` a hit vetoes the
  molecule outright) and :class:`MatchingSubstructureObjective`
  (``comp_matching_substructure.py``: ``0.5 * (1 + match)`` -- a
  *penalty* in REINVENT4, product-multiplied after aggregation; inside
  the toolkit's geometric mean the same product semantics fall out
  naturally, a non-match halves the total).

What deliberately did **not** land (the house rule from the SHAP/Optuna
and SELFIES rounds: re-implement in typed library surfaces, optional-gate
only what truly needs the third-party package):

* the plugin **namespace-package discovery** (``comp_*`` modules with
  ``@add_tag`` decorators) -- the toolkit's registry is an explicit
  typed dict, importable without importlib magic;
* components that shell out to external CLIs or licensed SDKs
  (SynthSense/AiZynthFinder, RingPrecedence databases, ROCS/OpenEye,
  ChemProp, Qptuna, Maize, DockStream, Icolos, REST predictors) and the
  pydantic TOML config schema that drives them;
* the Ertl SA score (ships a pickled feature-count model, ~200 KB of
  binary provenance the toolkit refuses to vendor).

REINVENT4 keeps every transform's output in float32; this port computes
in float64 (the toolkit's numpy convention) which is strictly more
precise along the same curve.

Examples
--------
>>> import numpy as np
>>> from pharmaml.generative.scoring_transforms import (
...     SigmoidTransform, StepTransform, make_transform)
>>> sig = SigmoidTransform(low=200, high=400, k=0.5)
>>> float(sig([300.0])[0])          # the window centre maps to 0.5
0.5
>>> float(sig([250.0])[0]) < 0.5 < float(sig([350.0])[0])
True
>>> step = StepTransform(low=0, high=5)
>>> step([2.0, 5.0, 7.0]).tolist()
[1.0, 1.0, 0.0]
>>> make_transform("sigmoid", low=0.0, high=1.0, k=1.0)([0.5])[0]
0.5

The aggregators -- NaN-aware weighting (a NaN component drops out of
the denominator too, so the total stays a *mean over what is known*):

>>> from pharmaml.generative.scoring_transforms import (
...     weighted_arithmetic_mean, weighted_geometric_mean)
>>> weighted_arithmetic_mean(
...     [([0.2, 0.8], 1.0), ([0.6, 0.4], 3.0)]).round(10).tolist()
[0.5, 0.5]
>>> weighted_geometric_mean(
...     [([0.5, 0.5], 1.0), ([float("nan"), 0.5], 2.0)]).round(10).tolist()
[0.5, 0.5]

The transformable components -- molecular weight pushed through the
REINVENT4 double sigmoid (a drug-like window centred on 350 Da):

>>> from pharmaml.generative.scoring_transforms import RawComponentObjective
>>> obj = RawComponentObjective(
...     "molecular_weight",
...     DoubleSigmoidTransform(low=250, high=450, k=100.0))
>>> 0.0 < obj.score("CCO") < 0.5        # ethanol (46 Da) sits far below the window
True
>>> obj.score("not-a-smiles")           # invalid SMILES: 0.0, counted
0.0
>>> obj.n_invalid
1

Custom alerts -- the blocklist filter (1.0 clean, 0.0 when any SMARTS
matches, so a geometric-mean combinator vetoes the molecule):

>>> from pharmaml.generative.scoring_transforms import CustomAlertsObjective
>>> alerts = CustomAlertsObjective(["[N+]=O", "C(F)(F)F"])
>>> alerts.score("CCO")
1.0
>>> alerts.score("Cc1ccc(N)cc1")        # an aniline... not on this blocklist
1.0
>>> alerts.score("C(F)(F)F")            # a CF3 hits the alert
0.0
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from .scoring import ScoringObjective, require_rdkit

_LOG = get_logger(__name__)

__all__ = [
    "RAW_COMPONENTS",
    "TRANSFORM_REGISTRY",
    "CustomAlertsObjective",
    "DoubleSigmoidTransform",
    "ExponentialDecayTransform",
    "GroupCountObjective",
    "LeftStepTransform",
    "MatchingSubstructureObjective",
    "RawComponentObjective",
    "ReverseSigmoidTransform",
    "RightStepTransform",
    "ScoreTransform",
    "SigmoidTransform",
    "StepTransform",
    "ValueMappingTransform",
    "make_transform",
    "weighted_arithmetic_mean",
    "weighted_geometric_mean",
]


# ---------------------------------------------------------------------------
# The sigmoid primitives (reinvent/scoring/transforms/sigmoid_functions.py)
# ---------------------------------------------------------------------------


def _hard_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    """REINVENT4's degenerate sigmoid: the hard step ``1[k x > 0]``."""
    return (k * x > 0).astype(np.float64)


def _stable_sigmoid(x: np.ndarray, k: float, base_10: bool = True) -> np.ndarray:
    """REINVENT4's numerically stable sigmoid.

    ``h = ln(10) . k . x`` (base-10 steepness, the REINVENT convention);
    the split form never exponentiates a positive argument, so no
    ``k`` however extreme overflows.
    """
    h = k * x
    if base_10:
        h = h * math.log(10.0)
    y = np.zeros_like(x, dtype=np.float64)
    positive = h >= 0
    y[positive] = 1.0 / (1.0 + np.exp(-h[positive]))
    exp_h = np.exp(h[~positive])
    y[~positive] = exp_h / (1.0 + exp_h)
    return y


def _double_sigmoid(
    x: np.ndarray,
    x_left: float,
    x_right: float,
    k: float,
    k_left: float,
    k_right: float,
) -> np.ndarray:
    """REINVENT4's double sigmoid (``sigmoid_functions.py`` verbatim).

    Two mirrored sigmoids sharing the centre ``x_c = (x_right - x_left)
    / 2 + x_left``: the left leg rises through ``x_left`` (output 0.5 at
    the boundary), the right leg falls through ``x_right``.  ``k == 0``
    switches both legs to the hard-step degenerate case.
    """
    x = np.asarray(x, dtype=np.float64)
    x_center = (x_right - x_left) / 2.0 + x_left

    left_mask = x < x_center
    xl = x[left_mask] - x_left
    xr = x[~left_mask] - x_right

    if k == 0:
        sigmoid_left = _hard_sigmoid(xl, k_left)
        sigmoid_right = 1.0 - _hard_sigmoid(xr, k_right)
    else:
        sigmoid_left = _stable_sigmoid(xl, k_left / k)
        sigmoid_right = 1.0 - _stable_sigmoid(xr, k_right / k)

    out = np.zeros_like(x, dtype=np.float64)
    out[left_mask] = sigmoid_left
    out[~left_mask] = sigmoid_right
    return out


# ---------------------------------------------------------------------------
# The transform base + registry
# ---------------------------------------------------------------------------


class ScoreTransform(ABC):
    """Base class of the REINVENT4 transform DSL.

    A transform maps an array of *raw* component values (molecular
    weights, logP values, group counts ...) to desirability scores in
    ``[0, 1]``.  Subclasses implement :meth:`__call__`; instances are
    created either directly or through :func:`make_transform` by their
    registry name (``"sigmoid"``, ``"double_sigmoid"``, ``"right_step"``
    ... -- dashes and underscores are stripped, mirroring REINVENT4's
    ``get_transform`` name normalisation).

    Every constructor validates its parameters and raises
    :class:`~pharmaml.exceptions.DataValidationError` on nonsense --
    the toolkit's typed surface where REINVENT4 relies on pydantic.
    """

    #: Registry name (the lowercase class name, REINVENT4's convention).
    name: str = "transform"

    @abstractmethod
    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Transform ``values`` to desirability scores in ``[0, 1]``."""

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        params = ", ".join(
            f"{key}={getattr(self, key)!r}"
            for key in self._parameter_names()
        )
        return f"{type(self).__name__}({params})"

    def _parameter_names(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in vars(self)
            if not key.startswith("_") and key not in {"params"}
        )


class SigmoidTransform(ScoreTransform):
    """REINVENT4's sigmoid (``transforms/sigmoids.py``).

    ``x = v - (high + low) / 2`` centred on the window; the steepness
    ``k`` is rescaled to ``10 k / (high - low)`` (REINVENT4's own
    normalisation, so ``k`` is window-width free).  ``high == low``
    degenerates to the hard step ``1[k x > 0]``.

    Parameters
    ----------
    low, high:
        The window whose centre maps to 0.5.
    k:
        Steepness (REINVENT4's TOML default is 5 for LogP-style
        components).

    Example
    -------
    >>> sig = SigmoidTransform(low=200, high=400, k=0.5)
    >>> [round(float(v), 3) for v in sig([200.0, 300.0, 400.0])]
    [0.003, 0.5, 0.997]
    """

    name = "sigmoid"

    def __init__(self, low: float, high: float, k: float = 1.0) -> None:
        for label, value in (("low", low), ("high", high), ("k", k)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DataValidationError(
                    f"SigmoidTransform {label} must be a finite number, got {value!r}"
                )
        self.low = float(low)
        self.high = float(high)
        self.k = float(k)

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        x = values - (self.high + self.low) / 2.0
        if (self.high - self.low) == 0:
            return _hard_sigmoid(x, 10.0 * self.k)
        return _stable_sigmoid(x, 10.0 * self.k / (self.high - self.low))


class ReverseSigmoidTransform(ScoreTransform):
    """REINVENT4's reverse sigmoid: ``1 - Sigmoid(values)``.

    The mirrored desirability -- *low* raw values score high -- with
    identical parameter semantics.

    Example
    -------
    >>> rev = ReverseSigmoidTransform(low=2.0, high=4.0, k=1.0)
    >>> float(rev([3.0])[0])
    0.5
    """

    name = "reverse_sigmoid"

    def __init__(self, low: float, high: float, k: float = 1.0) -> None:
        self._sigmoid = SigmoidTransform(low, high, k)
        self.low = self._sigmoid.low
        self.high = self._sigmoid.high
        self.k = self._sigmoid.k

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return 1.0 - self._sigmoid(values)


class DoubleSigmoidTransform(ScoreTransform):
    """REINVENT4's double sigmoid (``transforms/double_sigmoid.py``).

    A *band pass*: values inside ``[low, high]`` score towards 1.0,
    both tails decay.  The three coefficients control the leg steepness
    relative to the common divisor -- REINVENT4's defaults
    (``coef_div=100``, ``coef_si=150``, ``coef_se=150``) make both legs
    equally steep.  ``coef_div == 0`` switches to hard steps.

    Example
    -------
    >>> dbl = DoubleSigmoidTransform(low=250, high=450, k=100.0)
    >>> float(dbl([350.0])[0])            # window centre: the maximum
    1.0
    >>> float(dbl([100.0])[0]) < 0.01     # far below the window
    True
    """

    name = "double_sigmoid"

    def __init__(
        self,
        low: float,
        high: float,
        k: float = 100.0,
        coef_si: float = 150.0,
        coef_se: float = 150.0,
    ) -> None:
        # REINVENT4 spells the common divisor "coef_div"; the toolkit's
        # constructor names it k to match every other transform here.
        for label, value in (
            ("low", low),
            ("high", high),
            ("k", k),
            ("coef_si", coef_si),
            ("coef_se", coef_se),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DataValidationError(
                    f"DoubleSigmoidTransform {label} must be a finite number, "
                    f"got {value!r}"
                )
        if low > high:
            raise DataValidationError(
                f"DoubleSigmoidTransform low ({low}) exceeds high ({high})"
            )
        self.low = float(low)
        self.high = float(high)
        self.k = float(k)
        self.coef_si = float(coef_si)
        self.coef_se = float(coef_se)

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return _double_sigmoid(
            values, self.low, self.high, self.k, self.coef_si, self.coef_se
        )


class RightStepTransform(ScoreTransform):
    """REINVENT4's right step: ``1[v >= high]`` (a floor constraint).

    Example
    -------
    >>> RightStepTransform(high=0.7)([0.7, 0.6]).tolist()
    [1.0, 0.0]
    """

    name = "right_step"

    def __init__(self, high: float) -> None:
        if not isinstance(high, (int, float)) or not math.isfinite(float(high)):
            raise DataValidationError(
                f"RightStepTransform high must be a finite number, got {high!r}"
            )
        self.high = float(high)

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values >= self.high).astype(np.float64)


class LeftStepTransform(ScoreTransform):
    """REINVENT4's left step: ``1[v <= low]`` (a ceiling constraint).

    Example
    -------
    >>> LeftStepTransform(low=0.3)([0.3, 0.4]).tolist()
    [1.0, 0.0]
    """

    name = "left_step"

    def __init__(self, low: float) -> None:
        if not isinstance(low, (int, float)) or not math.isfinite(float(low)):
            raise DataValidationError(
                f"LeftStepTransform low must be a finite number, got {low!r}"
            )
        self.low = float(low)

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values <= self.low).astype(np.float64)


class StepTransform(ScoreTransform):
    """REINVENT4's window step: ``1[low <= v <= high]``.

    Example
    -------
    >>> StepTransform(low=0.2, high=0.8)([0.5, 0.9]).tolist()
    [1.0, 0.0]
    """

    name = "step"

    def __init__(self, low: float, high: float) -> None:
        for label, value in (("low", low), ("high", high)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DataValidationError(
                    f"StepTransform {label} must be a finite number, got {value!r}"
                )
        if low > high:
            raise DataValidationError(
                f"StepTransform low ({low}) exceeds high ({high})"
            )
        self.low = float(low)
        self.high = float(high)

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return ((values >= self.low) & (values <= self.high)).astype(np.float64)


class ExponentialDecayTransform(ScoreTransform):
    """REINVENT4's rectified exponential decay: ``exp(-k v)`` for
    ``v >= 0``, flat 1.0 below zero.

    Example
    -------
    >>> decay = ExponentialDecayTransform(k=0.5)
    >>> round(float(decay([2.0])[0]), 6)
    0.367879
    >>> decay([-3.0])[0]                  # negative values are clamped
    1.0
    """

    name = "exponential_decay"

    def __init__(self, k: float) -> None:
        if not isinstance(k, (int, float)) or not math.isfinite(float(k)):
            raise DataValidationError(
                f"ExponentialDecayTransform k must be a finite number, got {k!r}"
            )
        if k <= 0:
            raise DataValidationError(
                f"ExponentialDecayTransform k must be > 0, got {k!r} "
                "(REINVENT4 raises the same refusal)"
            )
        self.k = float(k)

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.where(values < 0, 1.0, np.exp(-self.k * values))


class ValueMappingTransform(ScoreTransform):
    """REINVENT4's category mapping: ``mapping[str(v)]``.

    Unmapped keys become NaN with a warning (never silently zero --
    REINVENT4 pairs this transform with the MMP component whose raw
    values are the strings ``"MMP"`` / ``"No MMP"``).

    Example
    -------
    >>> vm = ValueMappingTransform({"MMP": 1.0, "No MMP": 0.0})
    >>> vm(["MMP", "No MMP"]).tolist()
    [1.0, 0.0]
    >>> import math
    >>> math.isnan(vm(["unexpected"])[0])
    True
    """

    name = "value_mapping"

    def __init__(self, mapping: Mapping[str, float]) -> None:
        if not isinstance(mapping, Mapping):
            raise DataValidationError(
                "ValueMappingTransform mapping must be a dict of "
                "{category: score}, got "
                f"{type(mapping).__name__}"
            )
        if not mapping:
            raise DataValidationError(
                "ValueMappingTransform mapping must not be empty -- there "
                "would be no category to map"
            )
        clean: dict[str, float] = {}
        for key, value in mapping.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DataValidationError(
                    f"ValueMappingTransform score for {key!r} must be a finite "
                    f"number, got {value!r}"
                )
            clean[str(key)] = float(value)
        self.mapping = clean

    def __call__(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        out: list[float] = []
        unmapped: set[str] = set()
        for value in values:
            key = str(value)
            if key in self.mapping:
                out.append(self.mapping[key])
            else:
                out.append(float("nan"))
                unmapped.add(key)
        if unmapped:
            _LOG.warning(
                "ValueMappingTransform: %d unmapped categor(ies) %s substituted "
                "with NaN",
                len(unmapped),
                sorted(unmapped),
            )
        return np.asarray(out, dtype=np.float64)


#: The registry, keyed exactly like REINVENT4's ``get_transform`` lookup:
#: the lowercase *class name* (with the toolkit's ``Transform`` suffix
#: stripped, so ``DoubleSigmoidTransform`` registers as ``"doublesigmoid"``
#: just like REINVENT4's ``DoubleSigmoid`` class); the lookup in
#: :func:`make_transform` strips underscores and dashes the same way.
def _registry_key(transform_type: type[ScoreTransform]) -> str:
    return transform_type.__name__.lower().removesuffix("transform")


TRANSFORM_REGISTRY: dict[str, type[ScoreTransform]] = {
    _registry_key(transform): transform
    for transform in (
        SigmoidTransform,
        ReverseSigmoidTransform,
        DoubleSigmoidTransform,
        RightStepTransform,
        LeftStepTransform,
        StepTransform,
        ExponentialDecayTransform,
        ValueMappingTransform,
    )
}


def make_transform(name: str, **params: Any) -> ScoreTransform:
    """Build a transform by registry name (REINVENT4's ``get_transform``).

    ``name`` is normalised exactly like REINVENT4's lookup: lowercased
    with underscores and dashes stripped, so ``"DoubleSigmoid"``,
    ``"double_sigmoid"`` and ``"double-sigmoid"`` all resolve.

    Example
    -------
    >>> make_transform("DoubleSigmoid", low=250, high=450, k=100)([350.0])[0]
    1.0
    >>> make_transform("double-sigmoid", low=250, high=450, k=100)([350.0])[0]
    1.0
    >>> make_transform("no-such-transform")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: no score transform named 'no-such-transform' -- the registry holds: doublesigmoid, exponentialdecay, leftstep, reversesigmoid, rightstep, sigmoid, step, valuemapping
    """

    if not isinstance(name, str) or not name.strip():
        raise DataValidationError("transform name must be a non-empty string")
    key = name.lower().replace("_", "").replace("-", "")
    if key not in TRANSFORM_REGISTRY:
        known = ", ".join(sorted(TRANSFORM_REGISTRY))
        raise DataValidationError(
            f"no score transform named {name!r} -- the registry holds: {known}"
        )
    transform_type = TRANSFORM_REGISTRY[key]
    try:
        return transform_type(**params)  # type: ignore[arg-type]
    except TypeError as exc:
        raise DataValidationError(
            f"transform {name!r} rejected its parameters: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# The NaN-aware weighted aggregators (aggregators/means.py)
# ---------------------------------------------------------------------------


def _validated_score_weight_pairs(
    all_scores: Sequence[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Check shape agreement and return the stacked (scores, weights)."""
    if not all_scores:
        raise DataValidationError(
            "weighted aggregation needs at least one (scores, weight) pair"
        )
    sizes = {len(np.asarray(scores)) for scores, _ in all_scores}
    if len(sizes) > 1:
        raise DataValidationError(
            f"component score arrays disagree in length: {sorted(sizes)}"
        )
    scores = np.asarray(
        [np.asarray(scores, dtype=np.float64) for scores, _ in all_scores]
    )
    weights = np.asarray([float(weight) for _, weight in all_scores])
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise DataValidationError(
            "component weights must be finite, non-negative numbers "
            f"(got {weights.tolist()})"
        )
    return scores, weights


def weighted_arithmetic_mean(
    all_scores: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    """REINVENT4's weighted arithmetic mean (``custom_sum`` alias).

    ``result = nansum(scores . w_normalised)`` where the weights of NaN
    components are masked to NaN *before* normalisation -- a NaN
    component drops out of both the numerator and the denominator.

    Example
    -------
    >>> import numpy as np
    >>> weighted_arithmetic_mean(
    ...     [([0.2, 0.8], 1.0), ([0.6, 0.4], 3.0)]).round(10).tolist()
    [0.5, 0.5]
    >>> weighted_arithmetic_mean([([0.2], 1.0), ([float("nan")], 2.0)]).tolist()
    [0.2]
    """

    scores, weights = _validated_score_weight_pairs(all_scores)
    weight_matrix = np.broadcast_to(
        weights.reshape(-1, 1), scores.shape
    ).astype(np.float64)
    weight_matrix = np.array(weight_matrix, dtype=np.float64)
    weight_matrix[np.isnan(scores)] = np.nan
    with np.errstate(invalid="ignore"):
        sum_weights = np.nansum(weight_matrix, axis=0)
        result = np.nansum(scores * (weight_matrix / sum_weights), axis=0)
    return result


def weighted_geometric_mean(
    all_scores: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    """REINVENT4's weighted geometric mean (``custom_product`` alias).

    Scores are clamped to ``>= 1e-8`` (REINVENT4's guard), weights
    NaN-masked and normalised exactly like the arithmetic variant; the
    product runs through ``nanprod`` so a NaN component is skipped, not
    zeroed.

    Example
    -------
    >>> weighted_geometric_mean(
    ...     [([0.5, 0.5], 1.0), ([0.5, 0.5], 3.0)]).round(10).tolist()
    [0.5, 0.5]
    >>> weighted_geometric_mean([([0.5], 1.0), ([float("nan")], 2.0)]).tolist()
    [0.5]
    """

    scores, weights = _validated_score_weight_pairs(all_scores)
    scores = np.maximum(scores, 1e-8)
    weight_matrix = np.broadcast_to(
        weights.reshape(-1, 1), scores.shape
    ).astype(np.float64)
    weight_matrix = np.array(weight_matrix, dtype=np.float64)
    weight_matrix[np.isnan(scores)] = np.nan
    with np.errstate(invalid="ignore"):
        sum_weights = np.maximum(np.nansum(weight_matrix, axis=0), 1e-8)
        result = np.nanprod(scores ** (weight_matrix / sum_weights), axis=0)
    return result


# ---------------------------------------------------------------------------
# The raw physicochemical components (comp_physchem.py / comp_tpsa.py / ...)
# ---------------------------------------------------------------------------

#: Every raw component one line of provenance: (label, RDKit callable,
#: unit).  The value domain feeds the transform; the unit string rides
#: along into GUI tables.  Keys are stable identifiers.
_RAW_COMPONENT_SPECS: dict[str, tuple[str, Callable[[Any], float], str]] = {}


def _register_raw_component(
    key: str,
    label: str,
    computer: Callable[[Any], float],
    unit: str,
) -> None:
    _RAW_COMPONENT_SPECS[key] = (label, computer, unit)


def _rdkit_component(key: str, label: str, rdkit_attr: str, unit: str) -> None:
    """Register a component computed off a single ``Descriptors`` name."""

    def computer(mol: Any, _attr: str = rdkit_attr) -> float:
        from rdkit.Chem import Descriptors

        return float(getattr(Descriptors, _attr)(mol))

    _register_raw_component(key, label, computer, unit)


# The physchem set (comp_physchem.py, one entry each)
_rdkit_component("molecular_weight", "Molecular weight", "MolWt", "Da")
_rdkit_component("logp", "Crippen SlogP", "MolLogP", "")
_rdkit_component("h_bond_acceptors", "H-bond acceptors", "NumHAcceptors", "")
_rdkit_component("h_bond_donors", "H-bond donors", "NumHDonors", "")
_rdkit_component("rotatable_bonds", "Rotatable bonds", "NumRotatableBonds", "")
_rdkit_component("num_rings", "Rings", "RingCount", "")
_rdkit_component("aromatic_rings", "Aromatic rings", "NumAromaticRings", "")
_rdkit_component("aliphatic_rings", "Aliphatic rings", "NumAliphaticRings", "")
_rdkit_component("heavy_atoms", "Heavy atoms", "HeavyAtomCount", "")
_rdkit_component("heteroatoms", "Hetero atoms", "NumHeteroatoms", "")
_rdkit_component("csp3_fraction", "Csp3 fraction", "FractionCSP3", "")


def _largest_ring_computer(mol: Any) -> float:
    """``LargestRingSize`` (0 when acyclic) -- comp_physchem.py."""
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return 0.0
    return float(max(len(ring) for ring in ring_info.AtomRings()))


_register_raw_component(
    "largest_ring", "Largest ring size", _largest_ring_computer, "atoms"
)


def _qed_computer(mol: Any) -> float:
    from rdkit.Chem.QED import qed

    return float(qed(mol))


_register_raw_component("qed", "QED drug-likeness", _qed_computer, "")


def _tpsa_computer(mol: Any) -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcTPSA(mol))


_register_raw_component("tpsa", "Topological polar surface area", _tpsa_computer, "Å²")


def _graph_length_computer(mol: Any) -> float:
    """``GraphLength`` -- the maximum topological distance (0/1 atoms)."""
    from rdkit.Chem import rdmolops

    distances = rdmolops.GetDistanceMatrix(mol)
    if distances.size == 0:
        return 0.0
    return float(distances.max())


_register_raw_component(
    "graph_length", "Graph length (max distance)", _graph_length_computer, "bonds"
)


def _num_atom_stereo_centers_computer(mol: Any) -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))


_register_raw_component(
    "num_atom_stereo_centers",
    "Atom stereo centres",
    _num_atom_stereo_centers_computer,
    "",
)


#: The raw-component registry: key -> (label, ``mol -> float`` unit).
#: A ``MappingProxyType`` so GUI layers cannot mutate the library's
#: vocabulary by accident.
RAW_COMPONENTS: Mapping[str, tuple[str, Callable[[Any], float], str]] = MappingProxyType(
    _RAW_COMPONENT_SPECS
)


class RawComponentObjective(ScoringObjective):
    """A REINVENT4 component: raw descriptor value pushed through a transform.

    REINVENT4 wires ``[[scoring]] [[components]] [[Parameters]]`` blocks
    of component + transform + weight; the toolkit's equivalent is this
    objective -- ``score = clip(transform(raw_value), 0, 1)`` -- which
    drops straight into every optimiser (GA / RL / MCTS) and into
    :class:`~pharmaml.generative.MultiObjective` as a weighted member.

    Invalid SMILES score 0.0 with the standard ``n_invalid`` accounting.

    Parameters
    ----------
    component:
        A :data:`RAW_COMPONENTS` key (``"molecular_weight"``,
        ``"logp"``, ``"tpsa"``, ``"qed"``, ...).
    transform:
        Any :class:`ScoreTransform` instance.

    Raises
    ------
    DataValidationError
        On an unknown component key or a missing/invalid transform.

    Example
    -------
    >>> obj = RawComponentObjective(
    ...     "molecular_weight",
    ...     StepTransform(low=150, high=450))
    >>> obj.score("CCO")                       # ethanol: 46 Da, below
    0.0
    >>> obj.score("CC(=O)Oc1ccccc1C(=O)O")     # aspirin: 180 Da, inside
    1.0
    """

    def __init__(
        self, component: str, transform: ScoreTransform | None = None
    ) -> None:
        super().__init__()
        if not isinstance(component, str) or component not in RAW_COMPONENTS:
            known = ", ".join(sorted(RAW_COMPONENTS))
            raise DataValidationError(
                f"unknown raw component {component!r} -- the registry holds: "
                f"{known}"
            )
        if transform is None:
            raise DataValidationError(
                "RawComponentObjective needs a ScoreTransform instance "
                "(the component produces raw values, the transform makes "
                "them scores)"
            )
        if not isinstance(transform, ScoreTransform):
            raise DataValidationError(
                f"transform must be a ScoreTransform instance, got "
                f"{type(transform).__name__}"
            )
        self.component = component
        self.transform = transform
        self.name = f"{component}"

    def raw_value(self, smiles: str) -> float:
        """The untransformed component value (NaN for invalid SMILES)."""
        mol = self._parse(smiles)
        if mol is None:
            return float("nan")
        _, computer, _ = RAW_COMPONENTS[self.component]
        try:
            return computer(mol)
        except Exception:  # noqa: BLE001 - RDKit can raise on hostile input
            _LOG.warning(
                "raw component %r failed on %r -- scored as NaN",
                self.component,
                smiles,
            )
            return float("nan")

    def score(self, smiles: str) -> float:
        raw = self.raw_value(smiles)
        if math.isnan(raw):
            return 0.0
        transformed = float(self.transform(np.asarray([raw]))[0])
        if not math.isfinite(transformed):
            return 0.0
        return min(max(transformed, 0.0), 1.0)


class GroupCountObjective(ScoringObjective):
    """REINVENT4's ``GroupCount`` component (``comp_group_count.py``).

    The raw value is the number of SMARTS matches in the molecule; the
    transform maps it to desirability (REINVENT4's own pairing is a
    ``LeftStep`` on halogen counts or a ``Sigmoid`` on donor tallies).

    Example
    -------
    >>> from pharmaml.generative.scoring_transforms import LeftStepTransform
    >>> fluorines = GroupCountObjective("[F]", LeftStepTransform(low=1))
    >>> fluorines.score("C(F)(F)F")            # three F: over the ceiling
    0.0
    >>> fluorines.score("CCF")                 # one F: at the ceiling
    1.0
    """

    def __init__(self, smarts: str, transform: ScoreTransform) -> None:
        super().__init__()
        Chem = require_rdkit()
        from rdkit.rdBase import DisableLog, EnableLog

        if not isinstance(smarts, str) or not smarts.strip():
            raise DataValidationError("smarts must be a non-empty string")
        if not isinstance(transform, ScoreTransform):
            raise DataValidationError(
                f"transform must be a ScoreTransform instance, got "
                f"{type(transform).__name__}"
            )
        DisableLog("rdApp.error")
        try:
            query = Chem.MolFromSmarts(smarts)
        finally:
            EnableLog("rdApp.error")
        if query is None:
            raise DataValidationError(f"smarts {smarts!r} is not a parseable SMARTS")
        self.smarts = smarts
        self.transform = transform
        self._query = query
        self.name = "group_count"

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        count = float(len(mol.GetSubstructMatches(self._query)))
        if count == 0.0:
            # zero matches still flow through the transform (a LeftStep
            # ceiling of 1 scores 1.0 on zero groups, 0.0 on two)
            raw = 0.0
        else:
            raw = count
        transformed = float(self.transform(np.asarray([raw]))[0])
        if not math.isfinite(transformed):
            return 0.0
        return min(max(transformed, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Filter / penalty semantics (comp_custom_alerts.py, comp_matching_substructure.py)
# ---------------------------------------------------------------------------


class CustomAlertsObjective(ScoringObjective):
    """REINVENT4's ``CustomAlerts`` blocklist filter.

    Score ``1.0`` when **none** of the SMARTS alerts matches the
    molecule, ``0.0`` when any does.  REINVENT4 ANDs filters into the
    validity mask *before* aggregation; inside the toolkit's
    geometric-mean :class:`~pharmaml.generative.MultiObjective` the same
    semantics hold -- a zero member vetoes the product outright.

    Parameters
    ----------
    smarts:
        One or more SMARTS alert patterns.  Every entry must parse.

    Example
    -------
    >>> alerts = CustomAlertsObjective(["[N+](=O)[O-]", "C(=O)Cl"])
    >>> alerts.score("CCO")
    1.0
    >>> alerts.score("Cc1ccc([N+](=O)[O-])cc1")   # a nitro group
    0.0
    """

    def __init__(self, smarts: Sequence[str]) -> None:
        super().__init__()
        Chem = require_rdkit()
        from rdkit.rdBase import DisableLog, EnableLog

        if isinstance(smarts, str):
            raise DataValidationError(
                "smarts must be a list of SMARTS strings, not a single "
                f"string (got {smarts!r})"
            )
        try:
            patterns = list(smarts)
        except TypeError as exc:
            raise DataValidationError(
                "smarts must be a sequence of SMARTS strings"
            ) from exc
        if not patterns:
            raise DataValidationError(
                "the alerts list is empty -- a blocklist with no patterns "
                "filters nothing (pass the patterns you want to reject)"
            )
        queries: list[Any] = []
        clean: list[str] = []
        DisableLog("rdApp.error")
        try:
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern.strip():
                    raise DataValidationError(
                        "every alerts entry must be a non-empty SMARTS string"
                    )
                query = Chem.MolFromSmarts(pattern)
                if query is None:
                    raise DataValidationError(
                        f"alerts entry {pattern!r} is not a parseable SMARTS"
                    )
                queries.append(query)
                clean.append(pattern)
        finally:
            EnableLog("rdApp.error")
        self.smarts = tuple(clean)
        self._queries = tuple(queries)
        self.name = "custom_alerts"

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        for query in self._queries:
            if mol.HasSubstructMatch(query):
                return 0.0
        return 1.0


class MatchingSubstructureObjective(ScoringObjective):
    """REINVENT4's ``MatchingSubstructure`` penalty (``0.5 * (1 + match)``).

    In REINVENT4 penalties are **product-multiplied into the final
    score after aggregation**; the toolkit's geometric-mean
    :class:`~pharmaml.generative.MultiObjective` computes exactly that
    product, so a member returning ``1.0`` (the SMARTS matches -- no
    penalty) or ``0.5`` (no match -- the total is halved) reproduces
    the semantics.  A matching molecule is *rewarded* relative to a
    non-matching one, which is the point: the SMARTS describes what
    you want present.

    Example
    -------
    >>> penalty = MatchingSubstructureObjective(["c1ccccc1"])
    >>> penalty.score("CC(=O)Oc1ccccc1C(=O)O")   # aspirin carries benzene
    1.0
    >>> penalty.score("CCO")                     # ethanol does not
    0.5
    """

    def __init__(self, smarts: Sequence[str]) -> None:
        super().__init__()
        Chem = require_rdkit()
        from rdkit.rdBase import DisableLog, EnableLog

        if isinstance(smarts, str):
            raise DataValidationError(
                "smarts must be a list of SMARTS strings, not a single "
                f"string (got {smarts!r})"
            )
        try:
            patterns = list(smarts)
        except TypeError as exc:
            raise DataValidationError(
                "smarts must be a sequence of SMARTS strings"
            ) from exc
        if not patterns:
            raise DataValidationError(
                "the smarts list is empty -- a penalty with no patterns "
                "matches nothing (pass the SMARTS you want rewarded)"
            )
        queries: list[Any] = []
        clean: list[str] = []
        DisableLog("rdApp.error")
        try:
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern.strip():
                    raise DataValidationError(
                        "every smarts entry must be a non-empty SMARTS string"
                    )
                query = Chem.MolFromSmarts(pattern)
                if query is None:
                    raise DataValidationError(
                        f"smarts entry {pattern!r} is not a parseable SMARTS"
                    )
                queries.append(query)
                clean.append(pattern)
        finally:
            EnableLog("rdApp.error")
        self.smarts = tuple(clean)
        self._queries = tuple(queries)
        self.name = "matching_substructure"

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            # REINVENT4 invalid molecules score 0 before penalties; the
            # toolkit contract keeps invalid SMILES at the floor.
            return 0.0
        matched = any(mol.HasSubstructMatch(query) for query in self._queries)
        return 0.5 * (1.0 + float(matched))

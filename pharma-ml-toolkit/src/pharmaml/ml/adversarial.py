"""Adversarial robustness without gradients: the zeroth-order audit.

The best-of-ml-python *Adversarial Robustness* category (9 projects:
foolbox, cleverhans, the Adversarial Robustness Toolbox, adversarial
...) was reviewed entry by entry for this module.  Every library
there ships gradient-guided attacks -- FGSM, PGD, DeepFool -- that
need differentiable models on an autodiff graph.  But the *audit*
those attacks serve needs no gradient at all, and that audit is what
a model-agnostic desktop tool can run on every estimator it owns:

* **The sign-pattern family** -- FGSM's structure without its
  gradients: the all-plus and all-minus L∞ balls, every
  single-coordinate bump, and seeded random sign patterns.  If *any*
  pattern in the family flips a sample's verdict within radius ``r``,
  the sample is broken at ``r`` -- a lower bound on what a
  gradient-guided attacker could do (they can only do better), which
  is the honest direction for a robustness *audit*.
* **The minimal radius** -- bisection on that family: the smallest
  radius at which some pattern breaks the sample, i.e. the sample's
  adversarial margin.
* **The audit itself** -- :func:`adversarial_audit` runs the family
  across a radius grid for classification (label flips) *and*
  regression (error beyond tolerance) verdicts, reports the
  broken-rate curve, and :func:`adversarial_augmentation` ships the
  training recipe (noise the training set inside the radius, keep the
  labels -- adversarial training's closed-form half).

The pharma reading, stated once: a QSAR model that flips its call on
a descriptor vector inside realistic measurement noise
(±5 % of each descriptor's spread) is not a model, it is a coin --
and this audit is how the toolkit says so with numbers.

The honest scope: PGD with backtracked gradients, Carlini-Wagner
optimisation and certified defences need the autodiff engines the
toolkit refuses to vendor; the sign-pattern family here is the
gradient-free floor every one of those attacks stands on.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.adversarial import (
...     adversarial_audit, adversarial_augmentation, minimal_attack_radius,
... )
>>> rng = np.random.default_rng(42)
>>> X = np.vstack([rng.normal(size=(40, 4)) + 2.0,
...                rng.normal(size=(40, 4)) - 2.0])
>>> y = np.array([1] * 40 + [0] * 40)
>>> def predict_fn(rows):            # a hard-margin linear classifier
...     return (rows.sum(axis=1) > 0.0).astype(int)
>>> audit = adversarial_audit(predict_fn, X, y, radii=(0.5, 1.0, 2.0))
>>> audit.clean_accuracy == 1.0
True
>>> audit.broken_rate(2.0) >= audit.broken_rate(0.5)   # monotone attack
True
>>> margin = minimal_attack_radius(predict_fn, X[0], y[0], 4.0)
>>> 0.0 < margin < 4.0
True
>>> X_aug, y_aug = adversarial_augmentation(X, y, radius=0.1, per_sample=2)
>>> X_aug.shape[0], y_aug.shape[0]
(240, 240)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through the sign-pattern family (the pin).
RANDOM_STATE = 42

#: Seeded random sign patterns in the family, beyond the 2 + 2d exact ones.
_N_RANDOM_PATTERNS = 8


def _check_predict_fn(predict_fn: object) -> None:
    if not callable(predict_fn):
        raise DataValidationError("predict_fn must be callable: X (n, d) -> y (n,)")


def _check_matrix(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise DataValidationError(f"X must be 2-D (samples x features), got {array.ndim}-D")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise DataValidationError(f"X must be non-empty, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise DataValidationError("X must all be finite (no NaN/inf)")
    return array


def _check_radii(radii: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    checked = tuple(float(r) for r in radii)
    if not checked:
        raise DataValidationError("radii must not be empty")
    if any(not np.isfinite(r) or r <= 0 for r in checked):
        raise DataValidationError(f"radii must all be finite and > 0, got {radii!r}")
    return tuple(sorted(checked))


def _check_scales(scales: np.ndarray, n_features: int) -> np.ndarray:
    array = np.asarray(scales, dtype=float).reshape(-1)
    if array.size != n_features:
        raise DataValidationError(
            f"feature_scales has {array.size} entries, X has {n_features} features"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise DataValidationError("feature_scales must all be finite and > 0")
    return array


# ---------------------------------------------------------------------------
# The sign-pattern family (FGSM's structure, no gradients)
# ---------------------------------------------------------------------------


def _pattern_family(n_features: int, seed: int) -> np.ndarray:
    """The (2 + 2d + k, d) sign matrix: ±1 entries, one row per pattern.

    Row 0/1: the all-plus/all-minus vectors (FGSM's two signs, both
    tried -- without gradients you cannot know which one helps).
    Rows 2..2d+1: every single-coordinate bump (the coordinate attack).
    The last ``k`` rows: seeded random sign patterns (the family's
    cheap diversity, covering sign combinations the exact rows miss).
    """
    family = np.zeros((2 + 2 * n_features + _N_RANDOM_PATTERNS, n_features))
    family[0, :] = 1.0
    family[1, :] = -1.0
    for j in range(n_features):
        family[2 + 2 * j, j] = 1.0
        family[3 + 2 * j, j] = -1.0
    rng = np.random.default_rng(seed)
    family[2 + 2 * n_features :, :] = rng.choice([-1.0, 1.0], size=(_N_RANDOM_PATTERNS, n_features))
    return family


def _verdict_broken(
    predictions: np.ndarray, y_true: float, task: str, tolerance: float
) -> np.ndarray:
    """Which perturbed predictions count as broken, per the task."""
    if task == "classification":
        return predictions != y_true
    return np.abs(predictions - y_true) > tolerance


def _family_breaks(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_row: np.ndarray,
    y_true: float,
    radius: float,
    scales: np.ndarray,
    family: np.ndarray,
    task: str,
    tolerance: float,
) -> bool:
    """Does any pattern in the family break the sample at this radius?"""
    perturbed = x_row.reshape(1, -1) + radius * scales.reshape(1, -1) * family
    predictions = np.asarray(predict_fn(perturbed)).reshape(-1)
    if predictions.size != perturbed.shape[0]:
        raise DataValidationError(
            f"predict_fn returned {predictions.size} values for {perturbed.shape[0]} rows"
        )
    return bool(np.any(_verdict_broken(predictions, y_true, task, tolerance)))


# ---------------------------------------------------------------------------
# The minimal radius (the sample's adversarial margin)
# ---------------------------------------------------------------------------


def minimal_attack_radius(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_row: np.ndarray,
    y_true: float,
    radius_max: float,
    *,
    task: str = "classification",
    tolerance: float = 1.0,
    feature_scales: np.ndarray | None = None,
    tolerance_radius: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> float | None:
    """Bisect the smallest radius at which some pattern breaks the sample.

    Returns ``None`` when even ``radius_max`` breaks nothing (the
    sample is robust across the whole audited range).  The bisection
    keeps the *smallest breaking* radius, so the answer is an upper
    bound on the true margin: the family might flip the verdict at a
    smaller radius only a gradient-guided attack would find.

    Raises
    ------
    DataValidationError
        On a non-callable predict_fn, malformed inputs, unknown task,
        or a non-positive radius_max/tolerance.
    """
    _check_predict_fn(predict_fn)
    if task not in ("classification", "regression"):
        raise DataValidationError(f"unknown task {task!r}; valid: classification, regression")
    if not np.isfinite(radius_max) or radius_max <= 0:
        raise DataValidationError(f"radius_max must be finite and > 0, got {radius_max!r}")
    if tolerance_radius <= 0:
        raise DataValidationError(f"tolerance_radius must be > 0, got {tolerance_radius!r}")
    row = np.asarray(x_row, dtype=float).reshape(-1)
    if not np.all(np.isfinite(row)):
        raise DataValidationError("x_row must be finite (no NaN/inf)")
    scales = (
        _check_scales(feature_scales, row.size) if feature_scales is not None else np.ones(row.size)
    )
    family = _pattern_family(row.size, seed)
    if not _family_breaks(
        predict_fn, row, float(y_true), radius_max, scales, family, task, tolerance
    ):
        return None
    lo, hi = 0.0, float(radius_max)  # lo never breaks, hi always breaks
    while hi - lo > tolerance_radius:
        mid = 0.5 * (lo + hi)
        if _family_breaks(predict_fn, row, float(y_true), mid, scales, family, task, tolerance):
            hi = mid
        else:
            lo = mid
    _LOG.debug(
        "minimal attack radius: task=%s margin=%.4g (<= %.4g)",
        task,
        hi,
        radius_max,
    )
    return float(hi)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiusResult:
    """One radius of the audit grid: how many samples broke."""

    radius: float
    n_broken: int
    broken_rate: float


@dataclass(frozen=True)
class RobustnessAudit:
    """The verdict curve of a model under the sign-pattern family.

    ``clean_accuracy`` is the un-attacked agreement (classification) or
    within-tolerance rate (regression); ``radii_results`` is the
    broken-rate curve; ``margins`` the per-sample minimal radii
    (``inf`` for samples nothing in the family could break within
    ``radius_max``).  ``median_margin`` is the honest headline: half
    the samples flip before it, half after.
    """

    task: str
    n_samples: int
    n_features: int
    clean_accuracy: float
    tolerance: float
    radius_max: float
    radii_results: tuple[RadiusResult, ...]
    margins: tuple[float, ...]

    def broken_rate(self, radius: float) -> float:
        """The broken rate at (or below) the closest audited radius."""
        for result in self.radii_results:
            if result.radius >= radius:
                return result.broken_rate
        return self.radii_results[-1].broken_rate

    @property
    def median_margin(self) -> float:
        """The median per-sample minimal radius (inf-safe)."""
        finite = [m for m in self.margins if np.isfinite(m)]
        if not finite:
            return float("inf")
        return float(np.median(finite))


def adversarial_audit(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    radii: tuple[float, ...] | list[float],
    *,
    task: str = "classification",
    tolerance: float = 1.0,
    feature_scales: np.ndarray | None = None,
    seed: int = RANDOM_STATE,
) -> RobustnessAudit:
    """Run the sign-pattern family across a radius grid.

    For every sample and every radius, the family (2 + 2d + 8 seeded
    patterns, one batched ``predict_fn`` call per sample per radius)
    asks whether *any* perturbation within the L∞ ball breaks the
    verdict: a flipped label (classification) or an error beyond
    ``tolerance`` (regression).  ``feature_scales`` scales each
    feature's perturbation separately (pass the per-feature spreads to
    audit "±5 % of realistic measurement noise" rather than raw
    absolute units); the per-sample margins are then reported in the
    same scaled units.

    Raises
    ------
    DataValidationError
        On malformed inputs, an unknown task, empty radii, or a
        predict_fn that violates its batch contract.
    """
    _check_predict_fn(predict_fn)
    if task not in ("classification", "regression"):
        raise DataValidationError(f"unknown task {task!r}; valid: classification, regression")
    array = _check_matrix(X)
    labels = np.asarray(y, dtype=float).reshape(-1)
    if labels.size != array.shape[0]:
        raise DataValidationError(f"y has {labels.size} entries, X has {array.shape[0]} rows")
    if not np.all(np.isfinite(labels)):
        raise DataValidationError("y must all be finite (no NaN/inf)")
    if task == "classification":
        verdicts = np.asarray(predict_fn(array)).reshape(-1)
        if verdicts.size != array.shape[0]:
            raise DataValidationError(
                f"predict_fn returned {verdicts.size} values for {array.shape[0]} rows"
            )
        clean = float(np.mean(verdicts == labels))
    else:
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise DataValidationError(f"tolerance must be finite and > 0, got {tolerance!r}")
        predictions = np.asarray(predict_fn(array)).reshape(-1)
        if predictions.size != array.shape[0]:
            raise DataValidationError(
                f"predict_fn returned {predictions.size} values for {array.shape[0]} rows"
            )
        clean = float(np.mean(np.abs(predictions - labels) <= tolerance))
    grid = _check_radii(radii)
    radius_max = grid[-1]
    scales = (
        _check_scales(feature_scales, array.shape[1])
        if feature_scales is not None
        else np.ones(array.shape[1])
    )
    family = _pattern_family(array.shape[1], seed)
    margins: list[float] = []
    broken_counts = {radius: 0 for radius in grid}
    for row_idx in range(array.shape[0]):
        row, y_true = array[row_idx], float(labels[row_idx])
        margin: float | None = None
        for radius in grid:
            perturbed = row.reshape(1, -1) + radius * scales.reshape(1, -1) * family
            predictions = np.asarray(predict_fn(perturbed)).reshape(-1)
            if predictions.size != perturbed.shape[0]:
                raise DataValidationError(
                    f"predict_fn returned {predictions.size} values for {perturbed.shape[0]} rows"
                )
            if np.any(_verdict_broken(predictions, y_true, task, tolerance)):
                broken_counts[radius] += 1
                if margin is None:
                    margin = radius
        margins.append(float("inf") if margin is None else margin)
    results = tuple(
        RadiusResult(
            radius=float(radius),
            n_broken=int(broken_counts[radius]),
            broken_rate=float(broken_counts[radius]) / array.shape[0],
        )
        for radius in grid
    )
    audit = RobustnessAudit(
        task=task,
        n_samples=int(array.shape[0]),
        n_features=int(array.shape[1]),
        clean_accuracy=clean,
        tolerance=float(tolerance),
        radius_max=float(radius_max),
        radii_results=results,
        margins=tuple(margins),
    )
    _LOG.debug(
        "adversarial audit: task=%s clean=%.3f broken@max=%.3f median margin=%.4g",
        task,
        audit.clean_accuracy,
        results[-1].broken_rate,
        audit.median_margin,
    )
    return audit


# ---------------------------------------------------------------------------
# The training recipe
# ---------------------------------------------------------------------------


def adversarial_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    radius: float,
    per_sample: int = 2,
    *,
    feature_scales: np.ndarray | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Adversarial training's closed-form half: in-ball noisy copies.

    Every row gets ``per_sample`` copies perturbed uniformly inside
    the L∞ ball of ``radius`` (scaled per feature), labels unchanged
    -- the standard smooth-label augmentation that widens the margins
    the audit above measures.  The output stacks original rows first.

    Raises
    ------
    DataValidationError
        On malformed inputs or a non-positive radius/per_sample.
    """
    array = _check_matrix(X)
    labels = np.asarray(y).reshape(-1)
    if labels.size != array.shape[0]:
        raise DataValidationError(f"y has {labels.size} entries, X has {array.shape[0]} rows")
    if not np.isfinite(radius) or radius <= 0:
        raise DataValidationError(f"radius must be finite and > 0, got {radius!r}")
    if per_sample < 1:
        raise DataValidationError(f"per_sample must be >= 1, got {per_sample}")
    scales = (
        _check_scales(feature_scales, array.shape[1])
        if feature_scales is not None
        else np.ones(array.shape[1])
    )
    rng = np.random.default_rng(seed)
    n = array.shape[0]
    noise = rng.uniform(-radius, radius, size=(n * per_sample, array.shape[1]))
    copies = np.repeat(array, per_sample, axis=0) + noise * scales.reshape(1, -1)
    X_aug = np.vstack([array, copies])
    y_aug = np.concatenate([labels, np.repeat(labels, per_sample)])
    _LOG.debug(
        "adversarial augmentation: %d rows -> %d (radius=%.3g, per_sample=%d)",
        n,
        X_aug.shape[0],
        radius,
        per_sample,
    )
    return X_aug, y_aug

"""The offline glass box -- InterpretML, absorbed.

Round 187 checked interpretml/interpret 0.7.8 (the ``glassbox`` EBM
family and the ``blackbox`` partial-dependence surface) and lifted what
survives without the libebm C++ core:

* **The EBM quantile binning** -- :func:`ebm_bins` cuts every column at
  interior quantiles, reserves bin 0 for missing values and keeps the
  value bins lower-bound-inclusive
  (``searchsorted(cuts, x, "right") + 1``), the preprocessing every
  libebm term consumes before boosting.
* **The cyclic-greedy boosting schedule** -- per outer bag: stump
  updates (single-split, two-sided Newton leaf deltas with L1
  soft-thresholding and XGBoost-style split gains), one full cyclic
  sweep that computes every term's best gain WITHOUT applying, then
  ``ceil(greedy_ratio * n_terms)`` greedy pops that apply the cached
  updates (recomputed fresh when the popped term was touched since its
  gain was cached), ``smoothing_rounds`` opening steps per term at
  RANDOM split positions, and validation early stopping with
  best-state restore.
* **The FAST pair screen** -- every column re-binned at 64 bins, per
  bag a 2-D gradient/hessian histogram per candidate pair ranked by a
  depth-2 axis-aligned Newton gain, ranks averaged across bags, and the
  top ``interactions`` pairs boosted in their own stage on the mains
  residuals (init scores with the intercept removed, no intercept
  rounds; the postprocess centering folds whatever global bias the
  pair tensors absorb back into the intercept).
* **The EBM postprocess** -- per-bag term centering with the weighted
  means folded into the intercept, zero-weight bins zeroed, bag-count
  weighted averaging of the term tensors and per-bin standard
  deviations across bags.
* **Partial dependence + ICE** -- the ``interpret.blackbox`` sweep:
  percentile grids, one mutated column per grid point over ALL rows,
  the full ICE matrix plus mean/std/bounds and a Doane-histogram
  density.  One public function (:func:`partial_dependence`) returns
  the ICE curves inside the record -- no separate ``ice_curves`` entry
  point (documented here instead).

libebm is C++ with proprietary RNGs, so bit-exactness with upstream is
impossible: this port matches the algorithm, not the bitstream.  The
honest deviations from the upstream 0.7.8 defaults and algorithm:

* ``max_rounds`` defaults to **2000** (upstream 50000) -- validation
  early stopping triggers far earlier on toolkit-sized data; pass more
  to override.
* Above **40 features** the FAST screen evaluates a seeded subsample of
  300 pairs instead of every C(n, 2) combination.
* Pair terms boost **single axis-aligned splits** only (libebm searches
  richer cut geometry); cross-structure still emerges over rounds of
  alternating row/column stumps.
* The FAST strength is a **depth-2** axis-aligned gain (split one axis,
  then the other within each side): a single split's gain depends only
  on the shared feature's marginal gradient prefix, so a true pair ties
  exactly with any pair sharing its more informative feature -- the
  depth-2 gain is the minimal measure that separates them.
* ``inner_bags > 0`` and ``max_leaves != 2`` are refused with typed
  errors (the outer bags carry the variance reduction here; stumps are
  the upstream default geometry), and multiclass targets are refused
  in favour of the toolkit's binary/regression QSAR focus.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.interpret import EBMClassifier, ebm_bins, discretize
>>> bins = ebm_bins(np.array([0.0, 1.0, 2.0, 3.0]), max_bins=6)
>>> bins.cuts
(1.0, 2.0)
>>> discretize(np.array([-1.0, 0.5, 1.0, 2.5, np.nan]), bins).tolist()
[1, 1, 2, 3, 0]
>>> cuts = ebm_bins(np.linspace(0.0, 1.0, 101), max_bins=8).cuts
>>> len(cuts), all(a < b for a, b in zip(cuts, cuts[1:]))
(4, True)
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import PharmaMLError


class InterpretError(PharmaMLError):
    """Raised for invalid interpret inputs or unported upstream surfaces."""


#: Above this many features the FAST pair screen stops evaluating every
#: C(n, 2) combination and samples pairs instead (see the deviations
#: list in the module docstring).
FAST_FEATURE_CAP = 40

#: Pair candidates evaluated per bag when the feature count exceeds
#: :data:`FAST_FEATURE_CAP`.
FAST_PAIR_SAMPLE = 300


# --------------------------------------------------------------------------- #
# Binning -- the preprocessing every EBM term consumes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EbmBin:
    """The binning record for one column: its strictly increasing cuts.

    Value bins are lower-bound-inclusive: ``bin(x) =
    searchsorted(cuts, x, "right") + 1`` (below the first cut -> 1,
    ``c_k <= x < c_{k+1}`` -> ``k + 1``, at/above the last cut ->
    ``#cuts + 1``).  Bin 0 is reserved for missing values.
    """

    cuts: tuple[float, ...]


def ebm_bins(x: object, max_bins: int = 1024, min_samples_bin: int = 1) -> EbmBin:
    """Quantile-cut one column, the EBM preprocessing step.

    Non-finite values are dropped before placing cuts (NaN maps to the
    missing bin 0 at discretize time; +/-inf is treated the same way,
    the honest lumping for a column libebm would refuse).  The cut
    points are the unique interior quantiles at
    ``np.linspace(0, 1, max_bins - 2)[1:-1]`` -- interior so the data
    min/max become the outer value bins -- deduplicated to a strictly
    increasing sequence.  ``min_samples_bin`` then merges value bins
    that hold fewer rows by dropping the adjacent cut (the cut between
    the thin bin and its thinner neighbour).

    >>> ebm_bins(np.array([0.0, 1.0, 2.0, 3.0]), max_bins=6).cuts
    (1.0, 2.0)
    >>> ebm_bins(np.array([0.0] * 4 + [5.0] * 4 + [7.0]), max_bins=8,
    ...          min_samples_bin=4).cuts
    (5.0,)

    :param x: one column of values (anything ``np.asarray`` reads as
        1-D floats).
    :param max_bins: total bin budget (missing bin + value bins); at
        least 4.
    :param min_samples_bin: merge value bins holding fewer rows by
        dropping the adjacent cut; 1 merges only empty bins.
    :returns: the :class:`EbmBin` record.
    :raises InterpretError: on a non-2-D-friendly empty column, bad
        ``max_bins``/``min_samples_bin``, or a column with no finite
        value at all.
    """
    if int(max_bins) < 4:
        raise InterpretError(f"max_bins must be >= 4, got {max_bins}")
    if int(min_samples_bin) < 1:
        raise InterpretError(f"min_samples_bin must be >= 1, got {min_samples_bin}")
    array = np.asarray(x, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise InterpretError(
            "ebm_bins needs at least one finite value (the column is all-NaN/inf)"
        )
    points = np.linspace(0.0, 1.0, int(max_bins) - 2)[1:-1]
    raw = np.quantile(finite, points) if points.size else np.zeros(0)
    cuts = [float(c) for c in np.unique(raw)]
    cuts = _merge_thin_bins(cuts, finite, int(min_samples_bin))
    return EbmBin(cuts=tuple(cuts))


def _merge_thin_bins(
    cuts: list[float], values: np.ndarray, min_samples_bin: int
) -> list[float]:
    """Drop cuts until every value bin holds ``min_samples_bin`` rows.

    The thin bin merges with its thinner neighbour (ties -> left), so
    the loop always removes one cut and always terminates.
    """
    out = list(cuts)
    while out:
        bins_ = _bin_column(values, tuple(out))
        counts = np.bincount(bins_, minlength=len(out) + 2)[1:]  # value bins only
        thin = np.nonzero(counts < min_samples_bin)[0]
        if thin.size == 0:
            return out
        bin_index = int(thin[np.argmin(counts[thin])])  # 0-based among value bins
        n_bins = len(out)
        if bin_index == 0:
            del out[0]
        elif bin_index == n_bins:
            del out[n_bins - 1]
        elif counts[bin_index - 1] <= counts[bin_index + 1]:
            del out[bin_index - 1]
        else:
            del out[bin_index]
    return out


def _bin_column(values: np.ndarray, cuts: tuple[float, ...]) -> np.ndarray:
    """Map one column to bin indices (0 = missing/non-finite, 1..n)."""
    array = np.asarray(values, dtype=float).reshape(-1)
    cut_array = np.asarray(cuts, dtype=float)
    bins = np.searchsorted(cut_array, array, side="right").astype(np.int64) + 1
    bins[~np.isfinite(array)] = 0
    return bins


def discretize(x: object, bins: EbmBin) -> np.ndarray:
    """Bin one column against an :class:`EbmBin` record.

    Lower-bound-inclusive value bins; NaN (and +/-inf, the documented
    lumping) map to the missing bin 0.

    >>> discretize(np.array([0.5, 1.0, 2.5]), ebm_bins(
    ...     np.array([0.0, 1.0, 2.0, 3.0]), max_bins=6)).tolist()
    [1, 2, 3]
    """
    return _bin_column(np.asarray(x, dtype=float).reshape(-1), bins.cuts)


# --------------------------------------------------------------------------- #
# Stump machinery -- Newton leaves, XGBoost-style split gains
# --------------------------------------------------------------------------- #
def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty(scores.shape, dtype=float)
    positive = scores >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_negative = np.exp(scores[~positive])
    out[~positive] = exp_negative / (1.0 + exp_negative)
    return out


def _soft(value: float, alpha: float) -> float:
    """L1 soft threshold: ``sign(g) * max(|g| - alpha, 0)``."""
    if value > alpha:
        return value - alpha
    if value < -alpha:
        return value + alpha
    return 0.0


def _sq_over(num: float, den: float) -> float:
    """``num^2 / den`` with a zero-safe denominator."""
    return num * num / den if den > 0.0 else 0.0


def _sq_over_array(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_sq_over` (zero-safe elementwise)."""
    safe = np.where(den > 0.0, den, 1.0)
    return np.where(den > 0.0, num * num / safe, 0.0)


def _leaf_delta(
    grad: float, hess: float, lr: float, alpha: float, hess_floor: float
) -> float:
    """One side's Newton leaf update with L1 soft-thresholding."""
    denom = hess + hess_floor
    if denom <= 0.0:
        return 0.0
    return -lr * _soft(grad, alpha) / denom


def _split_scan(
    g: np.ndarray,
    h: np.ndarray,
    counts: np.ndarray,
    *,
    min_samples_leaf: int,
    hess_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid stump boundaries and their XGBoost-style gains (1-D).

    Boundary ``k`` splits ``left = bins <= k`` from ``right = bins > k``
    along the given 1-D marginal arrays; both sides must hold at least
    ``min_samples_leaf`` rows.  Returns ``(ks, gains)``, both empty when
    no boundary is valid.
    """
    size = int(g.size)
    if size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0)
    prefix_g = np.concatenate(([0.0], np.cumsum(g)))
    prefix_h = np.concatenate(([0.0], np.cumsum(h)))
    prefix_c = np.concatenate(([0], np.cumsum(counts)))
    total = int(prefix_c[-1])
    left_n = prefix_c[1:size]
    valid = (left_n >= min_samples_leaf) & ((total - left_n) >= min_samples_leaf)
    ks = np.nonzero(valid)[0].astype(np.int64)
    if ks.size == 0:
        return ks, np.empty(0)
    gl = prefix_g[1:size][valid]
    hl = prefix_h[1:size][valid]
    gt = float(prefix_g[-1])
    ht = float(prefix_h[-1])
    gains = _sq_over_array(gl, hl + hess_floor) + _sq_over_array(
        gt - gl, ht - hl + hess_floor
    ) - _sq_over(gt, ht + hess_floor)
    return ks, gains


def _best_gain_1d(
    g: np.ndarray,
    h: np.ndarray,
    counts: np.ndarray,
    *,
    min_samples_leaf: int,
    hess_floor: float,
) -> float:
    """Best single-split gain of a 1-D marginal (0.0 when unsplittable)."""
    ks, gains = _split_scan(
        g, h, counts, min_samples_leaf=min_samples_leaf, hess_floor=hess_floor
    )
    return float(gains.max()) if ks.size else 0.0


def _deltas_at(
    g: np.ndarray,
    h: np.ndarray,
    k: int,
    *,
    lr: float,
    alpha: float,
    hess_floor: float,
) -> tuple[float, float]:
    """Two-sided Newton deltas for boundary ``k`` on a 1-D marginal."""
    prefix_g = np.concatenate(([0.0], np.cumsum(g)))
    prefix_h = np.concatenate(([0.0], np.cumsum(h)))
    gl = float(prefix_g[k + 1])
    hl = float(prefix_h[k + 1])
    gr = float(prefix_g[-1]) - gl
    hr = float(prefix_h[-1]) - hl
    return (
        _leaf_delta(gl, hl, lr, alpha, hess_floor),
        _leaf_delta(gr, hr, lr, alpha, hess_floor),
    )


def _stump_1d(
    g: np.ndarray,
    h: np.ndarray,
    counts: np.ndarray,
    *,
    min_samples_leaf: int,
    hess_floor: float,
    lr: float,
    alpha: float,
    random_split: bool,
    rng: np.random.Generator,
) -> tuple[float, int, int, float, float] | None:
    """Best (or random, for smoothing rounds) stump over per-bin sums.

    Returns ``(gain, axis=0, k, delta_left, delta_right)`` or ``None``
    when no boundary satisfies ``min_samples_leaf`` (permanent for a
    term: per-bin row counts never change during boosting).
    """
    ks, gains = _split_scan(
        g, h, counts, min_samples_leaf=min_samples_leaf, hess_floor=hess_floor
    )
    if ks.size == 0:
        return None
    if random_split:
        pos = int(rng.integers(ks.size))
    else:
        pos = int(np.argmax(gains))
    k = int(ks[pos])
    dl, dr = _deltas_at(
        g, h, k, lr=lr, alpha=alpha, hess_floor=hess_floor
    )
    return float(gains[pos]), 0, k, dl, dr


def _stump_2d(
    g2: np.ndarray,
    h2: np.ndarray,
    c2: np.ndarray,
    *,
    min_samples_leaf: int,
    hess_floor: float,
    lr: float,
    alpha: float,
    random_split: bool,
    rng: np.random.Generator,
) -> tuple[float, int, int, float, float] | None:
    """Best axis-aligned single split of a pair's 2-D G/H grid.

    Row-threshold or column-threshold (rows win ties); the chosen
    side's deltas come from that axis's marginals.  Returns
    ``(gain, axis, k, delta_left, delta_right)`` with axis 0 = the
    first feature's bins, or ``None`` when neither axis can split.
    """
    row_g, row_h, row_c = g2.sum(axis=1), h2.sum(axis=1), c2.sum(axis=1)
    col_g, col_h, col_c = g2.sum(axis=0), h2.sum(axis=0), c2.sum(axis=0)
    row_ks, row_gains = _split_scan(
        row_g, row_h, row_c,
        min_samples_leaf=min_samples_leaf, hess_floor=hess_floor,
    )
    col_ks, col_gains = _split_scan(
        col_g, col_h, col_c,
        min_samples_leaf=min_samples_leaf, hess_floor=hess_floor,
    )
    if random_split:
        options = [
            (0, row_ks, row_gains, row_g, row_h),
            (1, col_ks, col_gains, col_g, col_h),
        ]
        options = [(o) for o in options if o[1].size > 0]
        if not options:
            return None
        axis, ks, gains, mg, mh = options[int(rng.integers(len(options)))]
        pos = int(rng.integers(ks.size))
    else:
        if row_ks.size == 0 and col_ks.size == 0:
            return None
        if col_ks.size == 0 or (
            row_ks.size > 0 and float(row_gains.max()) >= float(col_gains.max())
        ):
            axis, ks, gains, mg, mh = 0, row_ks, row_gains, row_g, row_h
        else:
            axis, ks, gains, mg, mh = 1, col_ks, col_gains, col_g, col_h
        pos = int(np.argmax(gains))
    k = int(ks[pos])
    dl, dr = _deltas_at(mg, mh, k, lr=lr, alpha=alpha, hess_floor=hess_floor)
    return float(gains[pos]), axis, k, dl, dr


def _pair_strength(
    g2: np.ndarray,
    h2: np.ndarray,
    c2: np.ndarray,
    *,
    min_samples_leaf: int,
    hess_floor: float,
) -> float:
    """Depth-2 axis-aligned Newton gain of a pair's gradient grid.

    Split one axis at a threshold, then the OTHER axis within each side
    (both orders tried); the total is the four-leaf gain minus the root
    term.  The depth-2 form is the documented FAST deviation: a single
    split's gain depends only on the shared feature's marginal gradient
    prefix, so a true pair ties exactly with any pair sharing its more
    informative feature.  Returns 0.0 when nothing valid can split.
    """
    rows, cols = g2.shape
    prefix_g = np.zeros((rows + 1, cols + 1))
    prefix_g[1:, 1:] = g2.cumsum(axis=0).cumsum(axis=1)
    prefix_h = np.zeros((rows + 1, cols + 1))
    prefix_h[1:, 1:] = h2.cumsum(axis=0).cumsum(axis=1)
    prefix_c = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    prefix_c[1:, 1:] = c2.cumsum(axis=0).cumsum(axis=1)
    gt = float(prefix_g[rows, cols])
    ht = float(prefix_h[rows, cols])
    root = _sq_over(gt, ht + hess_floor)
    best = 0.0
    for axis in (0, 1):
        outer = rows if axis == 0 else cols
        for cut in range(outer - 1):
            if axis == 0:
                n_left = int(prefix_c[cut + 1, cols])
                n_total = int(prefix_c[rows, cols])
                gl = float(prefix_g[cut + 1, cols])
                hl = float(prefix_h[cut + 1, cols])
                left_g = prefix_g[cut + 1, 1:]
                left_h = prefix_h[cut + 1, 1:]
                left_c = prefix_c[cut + 1, 1:]
            else:
                n_left = int(prefix_c[rows, cut + 1])
                n_total = int(prefix_c[rows, cols])
                gl = float(prefix_g[rows, cut + 1])
                hl = float(prefix_h[rows, cut + 1])
                left_g = prefix_g[1:, cut + 1]
                left_h = prefix_h[1:, cut + 1]
                left_c = prefix_c[1:, cut + 1]
            n_right = n_total - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            gr = gt - gl
            hr = ht - hl
            first = (
                _sq_over(gl, hl + hess_floor)
                + _sq_over(gr, hr + hess_floor)
                - root
            )
            right_g = prefix_g[rows, 1:] - left_g if axis == 0 else (
                prefix_g[1:, cols] - left_g
            )
            right_h = prefix_h[rows, 1:] - left_h if axis == 0 else (
                prefix_h[1:, cols] - left_h
            )
            right_c = prefix_c[rows, 1:] - left_c if axis == 0 else (
                prefix_c[1:, cols] - left_c
            )
            second_left = _best_gain_1d(
                left_g, left_h, left_c,
                min_samples_leaf=min_samples_leaf, hess_floor=hess_floor,
            )
            second_right = _best_gain_1d(
                right_g, right_h, right_c,
                min_samples_leaf=min_samples_leaf, hess_floor=hess_floor,
            )
            total = first + max(0.0, second_left) + max(0.0, second_right)
            best = max(best, total)
    return best


# --------------------------------------------------------------------------- #
# Term records -- what a fitted model stores and what the GUI plots
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EbmTermData:
    """One boosted term as stored on a fitted model.

    ``tensor`` is indexed by bin INCLUDING the missing bin 0 (1-D mains
    of length ``#cuts + 2``, 2-D pair grids of ``(#cuts_i + 2) x
    (#cuts_j + 2)``); ``bin_weights`` are the full-matrix row counts per
    bin and ``std`` the bag-weighted standard deviation per bin.
    """

    features: tuple[int, ...]
    cuts: tuple[tuple[float, ...], ...]
    tensor: np.ndarray
    bin_weights: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class EbmTerm:
    """One term in plot-ready form (:meth:`EBMClassifier.term_functions`).

    ``bin_centers`` holds the value-bin centers per feature (midpoints
    of the cut intervals, with the outer edges taken from the data
    min/max); ``scores``/``scores_std``/``weights`` are value-bins-only
    views (the missing bin is dropped): 1-D for mains, 2-D for pairs.
    """

    features: tuple[int, ...]
    kind: str
    bin_centers: tuple[tuple[float, ...], ...]
    scores: np.ndarray
    scores_std: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class _TermSpec:
    """Internal fit-time description of one boosted term."""

    features: tuple[int, ...]
    cuts: tuple[tuple[float, ...], ...]
    bins: tuple[np.ndarray, ...]
    shape: tuple[int, ...]
    weights: np.ndarray


def _term_scores(
    specs: Sequence[_TermSpec],
    tensors: Sequence[np.ndarray],
    rows: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """intercept + summed term lookups over the given row indices."""
    scores = np.full(int(rows.size), float(intercept), dtype=float)
    for spec, tensor in zip(specs, tensors):
        if len(spec.features) == 1:
            scores += tensor[spec.bins[0][rows]]
        else:
            scores += tensor[spec.bins[0][rows], spec.bins[1][rows]]
    return scores


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
class _BaseEBM:
    """Shared machinery for the EBM classifier and regressor."""

    _IS_CLASSIFIER: bool = True

    def __init__(
        self,
        feature_names: Sequence[str] | None,
        max_bins: int,
        max_interaction_bins: int,
        interactions: int | str | Sequence[Sequence[int]],
        exclude: Sequence[int] | None,
        validation_size: float,
        outer_bags: int,
        inner_bags: int,
        learning_rate: float,
        greedy_ratio: float,
        cyclic_progress: bool,
        smoothing_rounds: int,
        interaction_smoothing_rounds: int,
        early_stopping_rounds: int,
        early_stopping_tolerance: float,
        min_samples_leaf: int,
        min_hessian: float,
        reg_alpha: float,
        reg_lambda: float,
        max_leaves: int,
        max_rounds: int,
        random_state: int,
    ) -> None:
        self.feature_names = feature_names
        self.max_bins = int(max_bins)
        self.max_interaction_bins = int(max_interaction_bins)
        self.interactions = interactions
        self.exclude = exclude
        self.validation_size = float(validation_size)
        self.outer_bags = int(outer_bags)
        self.inner_bags = int(inner_bags)
        self.learning_rate = float(learning_rate)
        self.greedy_ratio = float(greedy_ratio)
        self.cyclic_progress = bool(cyclic_progress)
        self.smoothing_rounds = int(smoothing_rounds)
        self.interaction_smoothing_rounds = int(interaction_smoothing_rounds)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.early_stopping_tolerance = float(early_stopping_tolerance)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_hessian = float(min_hessian)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.max_leaves = int(max_leaves)
        self.max_rounds = int(max_rounds)
        self.random_state = int(random_state)

    # -- validation ----------------------------------------------------- #
    def _validate(self) -> None:
        """Typed parameter guards (the sklearn-convention fit-time check)."""
        if self.inner_bags != 0:
            raise InterpretError(
                "inner_bags > 0 is not ported: the outer bags carry the "
                "variance reduction in this offline port (upstream "
                "inner bagging is a libebm-side replication detail)"
            )
        if self.max_leaves != 2:
            raise InterpretError(
                f"max_leaves must be 2 (single-split stumps, the upstream "
                f"default geometry); got {self.max_leaves}"
            )
        checks: tuple[tuple[bool, str], ...] = (
            (self.max_bins >= 4, "max_bins must be >= 4"),
            (self.max_interaction_bins >= 4, "max_interaction_bins must be >= 4"),
            (0.0 < self.validation_size < 1.0, "validation_size must be in (0, 1)"),
            (self.outer_bags >= 1, "outer_bags must be >= 1"),
            (self.learning_rate > 0.0, "learning_rate must be > 0"),
            (self.greedy_ratio > 0.0, "greedy_ratio must be > 0"),
            (self.smoothing_rounds >= 0, "smoothing_rounds must be >= 0"),
            (
                self.interaction_smoothing_rounds >= 0,
                "interaction_smoothing_rounds must be >= 0",
            ),
            (self.early_stopping_rounds >= 1, "early_stopping_rounds must be >= 1"),
            (
                self.early_stopping_tolerance > 0.0,
                "early_stopping_tolerance must be > 0",
            ),
            (self.min_samples_leaf >= 1, "min_samples_leaf must be >= 1"),
            (self.min_hessian >= 0.0, "min_hessian must be >= 0"),
            (self.reg_alpha >= 0.0, "reg_alpha must be >= 0"),
            (self.reg_lambda >= 0.0, "reg_lambda must be >= 0"),
            (self.max_rounds >= 1, "max_rounds must be >= 1"),
        )
        for ok, message in checks:
            if not ok:
                raise InterpretError(message)

    # -- fit ------------------------------------------------------------- #
    def _fit_matrix(self, X: object, y: object) -> tuple[np.ndarray, np.ndarray]:
        """Validate and coerce X/y; returns the (X, y) float matrices."""
        try:
            matrix = np.asarray(X, dtype=float)
            target = np.asarray(y, dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise InterpretError(f"X and y must be numeric: {error}") from error
        if matrix.ndim != 2:
            raise InterpretError(f"X must be 2-D, got ndim={matrix.ndim}")
        if matrix.shape[0] != target.size:
            raise InterpretError(
                f"X has {matrix.shape[0]} rows but y has {target.size} values"
            )
        if matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise InterpretError(f"X must have at least 2 rows, got {matrix.shape}")
        if not np.isfinite(target).all():
            raise InterpretError("y must be finite everywhere")
        return matrix, target

    def _validate_target(self, target: np.ndarray) -> None:
        """Binary {0, 1} guard for the classifier (multiclass refused)."""
        values = np.unique(target)
        if values.size > 2 or not np.isin(values, (0.0, 1.0)).all():
            raise InterpretError(
                "EBMClassifier is binary-only: y must contain exactly the "
                "values {0, 1}. Upstream supports multiclass via per-class "
                "term tensors + softmax; this offline port refuses it -- the "
                "toolkit's QSAR focus is binary classification and "
                "regression (use EBMRegressor for continuous targets)"
            )
        if values.size < 2:
            raise InterpretError("y must contain both classes 0 and 1")

    def _resolve_interactions(
        self, n_features: int
    ) -> tuple[list[tuple[int, int]] | None, int]:
        """Parse the ``interactions`` spec.

        Returns ``(explicit_pairs_or_None, pair_count)``: a list means
        exact pairs (FAST skipped), a count means FAST selects that
        many.  ``"3x"`` -> 3 * n_features, capped at C(n, 2).
        """
        spec = self.interactions
        total_pairs = n_features * (n_features - 1) // 2
        if isinstance(spec, str):
            text = spec.strip().lower()
            if not text.endswith("x") or not text[:-1].isdigit():
                raise InterpretError(
                    f"interactions string must look like '3x', got {spec!r}"
                )
            wanted = int(text[:-1]) * n_features
        elif isinstance(spec, (int, np.integer)):
            wanted = int(spec)
            if wanted < 0:
                raise InterpretError(
                    f"interactions count must be >= 0, got {wanted}"
                )
        else:
            pairs: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for item in spec:
                pair = tuple(int(v) for v in item)
                if len(pair) != 2 or pair[0] == pair[1]:
                    raise InterpretError(
                        f"interactions list items must be two distinct "
                        f"feature indices, got {item!r}"
                    )
                if not (0 <= pair[0] < n_features and 0 <= pair[1] < n_features):
                    raise InterpretError(
                        f"interactions pair {pair} outside [0, {n_features})"
                    )
                key = (min(pair), max(pair))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
            return pairs, len(pairs)
        wanted = min(wanted, total_pairs)
        return None, max(wanted, 0)

    def _make_bags(
        self, target: np.ndarray, rng: np.random.Generator
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Outer bags: stratified (binary) or plain (regressor) splits."""
        n_rows = int(target.size)
        bags: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(self.outer_bags):
            if self._IS_CLASSIFIER:
                train_parts: list[np.ndarray] = []
                val_parts: list[np.ndarray] = []
                for cls in (0.0, 1.0):
                    idx = np.nonzero(target == cls)[0]
                    perm = rng.permutation(idx)
                    k = math.floor(idx.size * self.validation_size)
                    val_parts.append(perm[:k])
                    train_parts.append(perm[k:])
                train_idx = np.concatenate(train_parts)
                val_idx = np.concatenate(val_parts)
            else:
                perm = rng.permutation(n_rows)
                k = math.floor(n_rows * self.validation_size)
                val_idx = perm[:k]
                train_idx = perm[k:]
            if val_idx.size == 0 or train_idx.size == 0:
                raise InterpretError(
                    f"validation_size={self.validation_size} leaves an empty "
                    f"train/validation split on {n_rows} rows"
                )
            bags.append((train_idx, val_idx))
        return bags

    def _boost_stage(
        self,
        specs: list[_TermSpec],
        tensors: list[np.ndarray],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        base_train: np.ndarray,
        base_val: np.ndarray,
        rng: np.random.Generator,
        smoothing_rounds: int,
    ) -> int:
        """One bag's boosting over the given terms; mutates ``tensors``.

        Runs the cyclic-greedy schedule (sweep for gains, greedy pops of
        cached updates, fresh recompute for touched terms, random
        smoothing splits, validation early stopping with best-state
        restore).  The intercept is NOT boosted here (closed-form init;
        the postprocess folds the term means into it).  Returns the
        number of applied steps.
        """
        n_terms = len(specs)
        train_scores = np.array(base_train, dtype=float, copy=True)
        val_scores = np.array(base_val, dtype=float, copy=True)
        train_bins = [tuple(b[train_idx] for b in spec.bins) for spec in specs]
        val_bins = [tuple(b[val_idx] for b in spec.bins) for spec in specs]
        applied = np.zeros(n_terms, dtype=np.int64)
        version = np.zeros(n_terms, dtype=np.int64)
        cap = self.max_rounds * n_terms
        greedy_steps = math.ceil(self.greedy_ratio * n_terms)
        window = self.early_stopping_rounds * n_terms
        metrics: list[float] = []
        window_values: deque[float] = deque()
        min_outside = math.inf
        best_metric = math.inf
        best_tensors = [t.copy() for t in tensors]
        heap: list[tuple[float, int, int, int, int, float, float, int]] = []
        counter = 0
        total = 0

        def gradients() -> tuple[np.ndarray, np.ndarray]:
            if self._IS_CLASSIFIER:
                p = _sigmoid(train_scores)
                return p - y_train, p * (1.0 - p)
            return train_scores - y_train, np.ones_like(train_scores)

        def compute_stump(
            term: int, random_split: bool
        ) -> tuple[float, int, int, float, float] | None:
            g, h = gradients()
            spec = specs[term]
            if len(spec.features) == 1:
                length = spec.shape[0]
                bins = train_bins[term][0]
                g_bins = np.bincount(bins, weights=g, minlength=length)
                h_bins = np.bincount(bins, weights=h, minlength=length)
                c_bins = np.bincount(bins, minlength=length)
                return _stump_1d(
                    g_bins,
                    h_bins,
                    c_bins,
                    min_samples_leaf=self.min_samples_leaf,
                    hess_floor=self.reg_lambda + self.min_hessian,
                    lr=self.learning_rate,
                    alpha=self.reg_alpha,
                    random_split=random_split,
                    rng=rng,
                )
            rows_n, cols_n = spec.shape
            flat = train_bins[term][0] * cols_n + train_bins[term][1]
            size = rows_n * cols_n
            g_grid = np.bincount(flat, weights=g, minlength=size).reshape(
                rows_n, cols_n
            )
            h_grid = np.bincount(flat, weights=h, minlength=size).reshape(
                rows_n, cols_n
            )
            c_grid = np.bincount(flat, minlength=size).reshape(rows_n, cols_n)
            return _stump_2d(
                g_grid,
                h_grid,
                c_grid,
                min_samples_leaf=self.min_samples_leaf,
                hess_floor=self.reg_lambda + self.min_hessian,
                lr=self.learning_rate,
                alpha=self.reg_alpha,
                random_split=random_split,
                rng=rng,
            )

        def apply_step(
            term: int, axis: int, k: int, dl: float, dr: float
        ) -> None:
            tensor = tensors[term]
            bins = train_bins[term]
            if len(specs[term].features) == 1 or axis == 0:
                tensor[: k + 1] += dl
                tensor[k + 1 :] += dr
                left = bins[0] <= k
                train_scores[left] += dl
                train_scores[~left] += dr
                left_val = val_bins[term][0] <= k
                val_scores[left_val] += dl
                val_scores[~left_val] += dr
            else:
                tensor[:, : k + 1] += dl
                tensor[:, k + 1 :] += dr
                left = bins[1] <= k
                train_scores[left] += dl
                train_scores[~left] += dr
                left_val = val_bins[term][1] <= k
                val_scores[left_val] += dl
                val_scores[~left_val] += dr

        def validation_metric() -> float:
            if self._IS_CLASSIFIER:
                p = np.clip(_sigmoid(val_scores), 1e-15, 1.0 - 1e-15)
                return float(-np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p)))
            return float(np.sqrt(np.mean((val_scores - y_val) ** 2)))

        def bookkeep() -> bool:
            """Metric, best-state snapshot, early stopping; True = stop."""
            nonlocal min_outside, best_metric, best_tensors
            metric = validation_metric()
            metrics.append(metric)
            if len(window_values) == window:
                min_outside = min(min_outside, window_values.popleft())
            window_values.append(metric)
            if metric < best_metric:
                best_metric = metric
                best_tensors = [t.copy() for t in tensors]
            if (
                min_outside < math.inf
                and int(applied.min()) >= smoothing_rounds
            ):
                window_min = min(window_values)
                tolerance = (
                    min(abs(min_outside), abs(window_min))
                    * self.early_stopping_tolerance
                )
                if min_outside - tolerance <= window_min:
                    return True
            return False

        def push(term: int) -> None:
            nonlocal counter
            stump = compute_stump(term, int(applied[term]) < smoothing_rounds)
            if stump is None:
                heapq.heappush(
                    heap, (math.inf, term, counter, 0, 0, 0.0, 0.0, int(version[term]))
                )
            else:
                gain, axis, k, dl, dr = stump
                heapq.heappush(
                    heap, (-gain, term, counter, axis, k, dl, dr, int(version[term]))
                )
            counter += 1

        while total < cap:
            if self.cyclic_progress:
                progressed = False
                for term in range(n_terms):
                    if total >= cap:
                        break
                    stump = compute_stump(
                        term, int(applied[term]) < smoothing_rounds
                    )
                    if stump is None:
                        continue
                    _, axis, k, dl, dr = stump
                    apply_step(term, axis, k, dl, dr)
                    total += 1
                    applied[term] += 1
                    version[term] += 1
                    progressed = True
                    if bookkeep():
                        for t in range(n_terms):
                            tensors[t][...] = best_tensors[t]
                        return total
                if not progressed:
                    break
            else:
                heap.clear()
                for term in range(n_terms):
                    push(term)
                stop = False
                for _ in range(greedy_steps):
                    if total >= cap or not heap:
                        break
                    neg_gain, term, _, axis, k, dl, dr, ver = heapq.heappop(heap)
                    if neg_gain == math.inf:
                        stop = True  # nothing can split (permanent)
                        break
                    if ver != int(version[term]):
                        stump = compute_stump(
                            term, int(applied[term]) < smoothing_rounds
                        )
                        if stump is None:
                            stop = True
                            break
                        _, axis, k, dl, dr = stump
                    apply_step(term, axis, k, dl, dr)
                    total += 1
                    applied[term] += 1
                    version[term] += 1
                    if bookkeep():
                        stop = True
                        break
                    push(term)  # fresh re-push for the touched term
                if stop:
                    break
        for t in range(n_terms):
            tensors[t][...] = best_tensors[t]
        return total

    def _fit(self, X: object, y: object) -> None:
        """The full EBM fit (shared by both estimators)."""
        self._validate()
        matrix, target = self._fit_matrix(X, y)
        if self._IS_CLASSIFIER:
            self._validate_target(target)
        n_rows, n_features = matrix.shape
        if self.feature_names is not None and len(self.feature_names) != n_features:
            raise InterpretError(
                f"feature_names has {len(self.feature_names)} entries "
                f"but X has {n_features} columns"
            )
        exclude = {int(v) for v in (self.exclude or ())}
        for j in exclude:
            if not 0 <= j < n_features:
                raise InterpretError(f"exclude index {j} outside [0, {n_features})")
        # Parse the interactions spec up front (fail fast -- a bad spec
        # must refuse before any boosting runs).
        explicit, pair_count = self._resolve_interactions(n_features)

        rng = np.random.default_rng(self.random_state)
        bags = self._make_bags(target, rng)
        main_features = [j for j in range(n_features) if j not in exclude]
        if not main_features:
            raise InterpretError("every feature is excluded: nothing to boost")

        main_bins = [ebm_bins(matrix[:, j], max_bins=self.max_bins) for j in main_features]
        main_arrays = [
            discretize(matrix[:, j], b) for j, b in zip(main_features, main_bins)
        ]
        main_specs = [
            _TermSpec(
                features=(j,),
                cuts=(b.cuts,),
                bins=(a,),
                shape=(len(b.cuts) + 2,),
                weights=np.bincount(a, minlength=len(b.cuts) + 2),
            )
            for j, b, a in zip(main_features, main_bins, main_arrays)
        ]

        bag_intercepts: list[float] = []
        bag_tensors: list[list[np.ndarray]] = []
        total_steps = 0
        for bag, (train_idx, val_idx) in enumerate(bags):
            y_train = target[train_idx]
            if self._IS_CLASSIFIER:
                p_pos = min(max(float(np.mean(y_train)), 1e-6), 1.0 - 1e-6)
                intercept = math.log(p_pos / (1.0 - p_pos))
            else:
                intercept = float(np.mean(y_train))
            tensors = [np.zeros(spec.shape) for spec in main_specs]
            base_train = np.full(int(train_idx.size), intercept)
            base_val = np.full(int(val_idx.size), intercept)
            bag_rng = np.random.default_rng([self.random_state, bag])
            total_steps += self._boost_stage(
                main_specs,
                tensors,
                train_idx,
                val_idx,
                y_train,
                target[val_idx],
                base_train,
                base_val,
                bag_rng,
                self.smoothing_rounds,
            )
            bag_intercepts.append(intercept)
            bag_tensors.append(tensors)

        # -- pair stage: re-bin at max_interaction_bins, FAST screen, boost #
        selected: list[tuple[int, int]] = []
        pair_specs: list[_TermSpec] = []
        if pair_count > 0 and n_features >= 2:
            pair_bins = [
                ebm_bins(matrix[:, j], max_bins=self.max_interaction_bins)
                for j in range(n_features)
            ]
            pair_arrays = [
                discretize(matrix[:, j], b) for j, b in enumerate(pair_bins)
            ]
            if explicit is not None:
                selected = list(explicit)
            else:
                selected = self._fast_pairs(
                    main_specs, bag_tensors, bags, target,
                    pair_arrays, pair_bins, n_features, pair_count,
                )
            for i, j in selected:
                rows_n = len(pair_bins[i].cuts) + 2
                cols_n = len(pair_bins[j].cuts) + 2
                flat = pair_arrays[i] * cols_n + pair_arrays[j]
                pair_specs.append(
                    _TermSpec(
                        features=(i, j),
                        cuts=(pair_bins[i].cuts, pair_bins[j].cuts),
                        bins=(pair_arrays[i], pair_arrays[j]),
                        shape=(rows_n, cols_n),
                        weights=np.bincount(
                            flat, minlength=rows_n * cols_n
                        ).reshape(rows_n, cols_n),
                    )
                )
            for bag, (train_idx, val_idx) in enumerate(bags):
                mains = bag_tensors[bag][: len(main_specs)]
                base_train = _term_scores(main_specs, mains, train_idx, 0.0)
                base_val = _term_scores(main_specs, mains, val_idx, 0.0)
                tensors = [np.zeros(spec.shape) for spec in pair_specs]
                bag_rng = np.random.default_rng(
                    [self.random_state, self.outer_bags + bag]
                )
                total_steps += self._boost_stage(
                    pair_specs,
                    tensors,
                    train_idx,
                    val_idx,
                    target[train_idx],
                    target[val_idx],
                    base_train,
                    base_val,
                    bag_rng,
                    self.interaction_smoothing_rounds,
                )
                bag_tensors[bag].extend(tensors)

        # -- postprocess: center per bag, zero empty bins, average bags ---- #
        all_specs = main_specs + pair_specs
        bag_weights = np.asarray(
            [float(train_idx.size) for train_idx, _ in bags], dtype=float
        )
        weight_sum = float(bag_weights.sum())
        averaged: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        intercept = float(
            np.average(np.asarray(bag_intercepts), weights=bag_weights)
        )
        for term_index, spec in enumerate(all_specs):
            total_weight = float(spec.weights.sum())
            centered: list[np.ndarray] = []
            for bag in range(len(bags)):
                tensor = bag_tensors[bag][term_index]
                if total_weight > 0.0:
                    shift = float(
                        np.average(tensor, weights=spec.weights)
                    )
                else:
                    shift = 0.0
                bag_intercepts[bag] += shift
                moved = tensor - shift
                moved[spec.weights == 0] = 0.0
                centered.append(moved)
            stack = np.stack(centered)
            mean_tensor = np.tensordot(bag_weights, stack, axes=(0, 0)) / weight_sum
            variance = np.tensordot(
                bag_weights, (stack - mean_tensor) ** 2, axes=(0, 0)
            ) / weight_sum
            averaged.append(mean_tensor)
            stds.append(np.sqrt(variance))
        intercept = float(
            np.average(np.asarray(bag_intercepts), weights=bag_weights)
        )

        terms = [
            EbmTermData(
                features=spec.features,
                cuts=spec.cuts,
                tensor=mean_tensor,
                bin_weights=spec.weights,
                std=std,
            )
            for spec, mean_tensor, std in zip(all_specs, averaged, stds)
        ]
        self.terms_ = terms
        self.intercept_ = intercept
        self.n_features_in_ = n_features
        self.feature_names_ = (
            [str(name) for name in self.feature_names]
            if self.feature_names is not None
            else None
        )
        self.n_steps_ = int(total_steps)
        self._data_min = np.nanmin(matrix, axis=0)
        self._data_max = np.nanmax(matrix, axis=0)
        if self._IS_CLASSIFIER:
            self.classes_ = np.asarray([0, 1], dtype=np.int64)
        else:
            # residual mean polish: intercept += mean(y - predictions)
            scores = intercept
            predictions = np.full(n_rows, scores)
            for term in terms:
                spec_bins = [
                    _bin_column(matrix[:, f], cuts) for f, cuts in zip(term.features, term.cuts)
                ]
                if len(term.features) == 1:
                    predictions += term.tensor[spec_bins[0]]
                else:
                    predictions += term.tensor[spec_bins[0], spec_bins[1]]
            self.intercept_ = intercept + float(np.mean(target - predictions))

    def _fast_pairs(
        self,
        main_specs: list[_TermSpec],
        bag_tensors: list[list[np.ndarray]],
        bags: list[tuple[np.ndarray, np.ndarray]],
        target: np.ndarray,
        pair_arrays: list[np.ndarray],
        pair_bins: list[EbmBin],
        n_features: int,
        pair_count: int,
    ) -> list[tuple[int, int]]:
        """Rank candidate pairs per bag by depth-2 gain; average ranks.

        Gradients read the mains-only residual scores (the intercept
        removed, the upstream FAST reading): the screen ranks pairs on
        what the mains left behind, not on the global class prior.
        """
        all_pairs = [
            (i, j) for i in range(n_features) for j in range(i + 1, n_features)
        ]
        fast_rng = np.random.default_rng([self.random_state, 987654321])
        if n_features > FAST_FEATURE_CAP and len(all_pairs) > FAST_PAIR_SAMPLE:
            chosen = fast_rng.choice(
                len(all_pairs), size=FAST_PAIR_SAMPLE, replace=False
            )
            candidates = [all_pairs[int(c)] for c in sorted(int(v) for v in chosen)]
        else:
            candidates = all_pairs
        if not candidates:
            return []
        rank_sums: dict[tuple[int, int], float] = {pair: 0.0 for pair in candidates}
        for bag, (train_idx, val_idx) in enumerate(bags):
            scores = _term_scores(main_specs, bag_tensors[bag], train_idx, 0.0)
            y_train = target[train_idx]
            if self._IS_CLASSIFIER:
                p = _sigmoid(scores)
                g = p - y_train
                h = p * (1.0 - p)
            else:
                g = scores - y_train
                h = np.ones_like(scores)
            strengths: dict[tuple[int, int], float] = {}
            for i, j in candidates:
                rows_n = len(pair_bins[i].cuts) + 2
                cols_n = len(pair_bins[j].cuts) + 2
                flat = pair_arrays[i][train_idx] * cols_n + pair_arrays[j][train_idx]
                size = rows_n * cols_n
                g_grid = np.bincount(flat, weights=g, minlength=size).reshape(
                    rows_n, cols_n
                )
                h_grid = np.bincount(flat, weights=h, minlength=size).reshape(
                    rows_n, cols_n
                )
                c_grid = np.bincount(flat, minlength=size).reshape(rows_n, cols_n)
                strength = _pair_strength(
                    g_grid,
                    h_grid,
                    c_grid,
                    min_samples_leaf=self.min_samples_leaf,
                    hess_floor=self.reg_lambda + self.min_hessian,
                ) / float(train_idx.size)
                strengths[(i, j)] = strength
            order = sorted(candidates, key=lambda pair: (-strengths[pair], pair))
            for rank, pair in enumerate(order):
                rank_sums[pair] += rank
        ranked = sorted(rank_sums.items(), key=lambda item: (item[1], item[0]))
        return [pair for pair, _ in ranked[:pair_count]]

    # -- inference surfaces ---------------------------------------------- #
    def _check_fitted(self) -> None:
        if getattr(self, "terms_", None) is None:
            raise InterpretError("this EBM is not fitted yet; call fit(X, y) first")

    def _validate_matrix(self, X: object) -> np.ndarray:
        self._check_fitted()
        try:
            matrix = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise InterpretError(f"X must be numeric: {error}") from error
        if matrix.ndim != 2:
            raise InterpretError(f"X must be 2-D, got ndim={matrix.ndim}")
        if matrix.shape[1] != self.n_features_in_:
            raise InterpretError(
                f"X has {matrix.shape[1]} columns, expected {self.n_features_in_}"
            )
        return matrix

    def eval_terms(self, X: object) -> np.ndarray:
        """Per-term contributions: an ``(n_rows, n_terms)`` matrix.

        Exact additivity holds by construction:
        ``eval_terms(X).sum(axis=1) + intercept_ == decision_function(X)``.

        >>> import numpy as np
        >>> from pharmaml.ml.interpret import EBMClassifier
        >>> rng = np.random.default_rng(7)
        >>> X = rng.random((20, 3))
        >>> y = (X[:, 0] > 0.5).astype(int)
        >>> model = EBMClassifier(outer_bags=2, max_rounds=10, smoothing_rounds=2,
        ...     interactions=0, early_stopping_rounds=5, random_state=0).fit(X, y)
        >>> model.eval_terms(X).shape
        (20, 3)
        >>> bool(np.array_equal(model.eval_terms(X).sum(axis=1) + model.intercept_,
        ...                     model.decision_function(X)))
        True
        """
        matrix = self._validate_matrix(X)
        out = np.zeros((matrix.shape[0], len(self.terms_)), dtype=float)
        for index, term in enumerate(self.terms_):
            bins = [
                _bin_column(matrix[:, f], cuts)
                for f, cuts in zip(term.features, term.cuts)
            ]
            if len(term.features) == 1:
                out[:, index] = term.tensor[bins[0]]
            else:
                out[:, index] = term.tensor[bins[0], bins[1]]
        return out

    def decision_function(self, X: object) -> np.ndarray:
        """The raw additive score: intercept + summed term lookups."""
        self._check_fitted()
        return self.intercept_ + self.eval_terms(X).sum(axis=1)

    def term_importances(self) -> np.ndarray:
        """Per-term importance: the weighted mean of ``|tensor|``.

        ``avg_weight`` (the mean of the absolute scores weighted by the
        bin row counts) is the default; the upstream's alternative
        ``min_max`` normalization (max - min of the term's scores) is
        deliberately not offered here -- the weighted form is stable
        for zero-weight bins, which the postprocess zeroes.

        >>> import numpy as np
        >>> from pharmaml.ml.interpret import EBMClassifier
        >>> rng = np.random.default_rng(7)
        >>> X = rng.random((20, 3))
        >>> y = (X[:, 0] > 0.5).astype(int)
        >>> model = EBMClassifier(outer_bags=2, max_rounds=10, smoothing_rounds=2,
        ...     interactions=0, early_stopping_rounds=5, random_state=0).fit(X, y)
        >>> model.term_importances().shape
        (3,)
        >>> bool((model.term_importances() >= 0).all())
        True
        """
        self._check_fitted()
        importances = np.zeros(len(self.terms_), dtype=float)
        for index, term in enumerate(self.terms_):
            weight = float(term.bin_weights.sum())
            if weight > 0.0:
                importances[index] = float(
                    np.average(np.abs(term.tensor), weights=term.bin_weights)
                )
        return importances

    @property
    def feature_importances_(self) -> np.ndarray:
        """Main-term importances mapped to feature order (pairs excluded).

        Pair terms are excluded by design (upstream reports mains-only
        here too); use :meth:`term_importances` for the full per-term
        view.  Excluded features score 0.
        """
        self._check_fitted()
        importances = np.zeros(self.n_features_in_, dtype=float)
        values = self.term_importances()
        for index, term in enumerate(self.terms_):
            if len(term.features) == 1:
                importances[term.features[0]] = values[index]
        return importances

    def term_functions(self) -> list[EbmTerm]:
        """The term surfaces ready for GUI plotting.

        Mains get 1-D ``scores`` over the value bins; pairs get the 2-D
        grid; ``bin_centers`` are the value-bin centers per feature (cut
        interval midpoints with outer edges from the data min/max).
        """
        self._check_fitted()
        out: list[EbmTerm] = []
        for term in self.terms_:
            bin_centers = tuple(
                _bin_centers(cuts, float(self._data_min[f]), float(self._data_max[f]))
                for f, cuts in zip(term.features, term.cuts)
            )
            if len(term.features) == 1:
                scores = term.tensor[1:]
                std = term.std[1:]
                weights = term.bin_weights[1:]
            else:
                scores = term.tensor[1:, 1:]
                std = term.std[1:, 1:]
                weights = term.bin_weights[1:, 1:]
            out.append(
                EbmTerm(
                    features=term.features,
                    kind="main" if len(term.features) == 1 else "pair",
                    bin_centers=bin_centers,
                    scores=scores,
                    scores_std=std,
                    weights=weights,
                )
            )
        return out


def _bin_centers(cuts: tuple[float, ...], low: float, high: float) -> tuple[float, ...]:
    """Value-bin centers: cut-interval midpoints with outer data edges."""
    edges = [low, *(float(c) for c in cuts), high]
    return tuple(
        (edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)
    )


class EBMClassifier(_BaseEBM):
    """The Explainable Boosting Classifier, binary -- offline port.

    Faithful to the upstream 0.7.8 default table (see ``__init__``);
    the documented deviations from upstream are listed in the module
    docstring: ``max_rounds`` 2000 (upstream 50000), subsampled FAST
    above 40 features, single axis-aligned pair splits, the depth-2
    FAST gain, refused ``inner_bags > 0`` / ``max_leaves != 2``, and
    multiclass refusal (decode y to {0, 1}; use :class:`EBMRegressor`
    for continuous targets).

    ``cyclic_progress=True`` switches the schedule from
    sweep-then-greedy to classic cyclic boosting (each sweep step
    applies its update immediately).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(7)
    >>> X = rng.random((20, 3))
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> model = EBMClassifier(outer_bags=2, max_rounds=10, smoothing_rounds=2,
    ...     interactions=0, early_stopping_rounds=5, random_state=0).fit(X, y)
    >>> probs = model.predict_proba(X)
    >>> bool(np.allclose(probs.sum(axis=1), 1.0))
    True
    """

    _IS_CLASSIFIER = True

    def __init__(
        self,
        feature_names: Sequence[str] | None = None,
        max_bins: int = 1024,
        max_interaction_bins: int = 64,
        interactions: int | str | Sequence[Sequence[int]] = "3x",
        exclude: Sequence[int] | None = None,
        validation_size: float = 0.15,
        outer_bags: int = 14,
        inner_bags: int = 0,
        learning_rate: float = 0.015,
        greedy_ratio: float = 10.0,
        cyclic_progress: bool = False,
        smoothing_rounds: int = 75,
        interaction_smoothing_rounds: int = 75,
        early_stopping_rounds: int = 100,
        early_stopping_tolerance: float = 1e-5,
        min_samples_leaf: int = 4,
        min_hessian: float = 1e-4,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        max_leaves: int = 2,
        max_rounds: int = 2000,
        random_state: int = 42,
    ) -> None:
        """Store the upstream default table verbatim (see the class docs)."""
        super().__init__(
            feature_names=feature_names,
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            greedy_ratio=greedy_ratio,
            cyclic_progress=cyclic_progress,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            min_samples_leaf=min_samples_leaf,
            min_hessian=min_hessian,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_leaves=max_leaves,
            max_rounds=max_rounds,
            random_state=random_state,
        )

    def fit(self, X: object, y: object) -> EBMClassifier:
        """Fit on a binary {0, 1} target; returns ``self``.

        :raises InterpretError: on multiclass/invalid y, bad parameters,
            all-NaN columns, or empty bag splits.
        """
        self._fit(X, y)
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Class probabilities ``[1 - sigma, sigma]`` per row."""
        scores = self.decision_function(X)
        positive = _sigmoid(scores)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: object) -> np.ndarray:
        """Class labels at the 0.5 probability threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


class EBMRegressor(_BaseEBM):
    """The Explainable Boosting Regressor -- offline port (RMSE objective).

    Same machinery and the same documented deviations as
    :class:`EBMClassifier`, with the upstream regressor defaults
    (``interactions="5x"``, ``learning_rate=0.04``,
    ``smoothing_rounds=500``, ``interaction_smoothing_rounds=100``,
    ``min_hessian=0.0``).  ``objective`` accepts only ``"rmse"``.

    The fit ends with the upstream's residual mean polish: the final
    intercept absorbs ``mean(y - predictions)`` so the mean residual is
    zero by construction.
    """

    _IS_CLASSIFIER = False

    def __init__(
        self,
        feature_names: Sequence[str] | None = None,
        max_bins: int = 1024,
        max_interaction_bins: int = 64,
        interactions: int | str | Sequence[Sequence[int]] = "5x",
        exclude: Sequence[int] | None = None,
        validation_size: float = 0.15,
        outer_bags: int = 14,
        inner_bags: int = 0,
        learning_rate: float = 0.04,
        greedy_ratio: float = 10.0,
        cyclic_progress: bool = False,
        smoothing_rounds: int = 500,
        interaction_smoothing_rounds: int = 100,
        early_stopping_rounds: int = 100,
        early_stopping_tolerance: float = 1e-5,
        min_samples_leaf: int = 4,
        min_hessian: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        max_leaves: int = 2,
        max_rounds: int = 2000,
        random_state: int = 42,
        objective: str = "rmse",
    ) -> None:
        """Store the upstream regressor default table verbatim."""
        if objective != "rmse":
            raise InterpretError(
                f"objective must be 'rmse', got {objective!r} (the only "
                f"objective ported offline)"
            )
        super().__init__(
            feature_names=feature_names,
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            greedy_ratio=greedy_ratio,
            cyclic_progress=cyclic_progress,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            min_samples_leaf=min_samples_leaf,
            min_hessian=min_hessian,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_leaves=max_leaves,
            max_rounds=max_rounds,
            random_state=random_state,
        )

    def fit(self, X: object, y: object) -> EBMRegressor:
        """Fit on a finite float target; returns ``self``."""
        self._fit(X, y)
        return self

    def predict(self, X: object) -> np.ndarray:
        """The raw additive score per row (the regression output)."""
        return self.decision_function(X)


# --------------------------------------------------------------------------- #
# Partial dependence + ICE -- interpret.blackbox, offline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PartialDependenceRecord:
    """One partial-dependence sweep with its ICE curves.

    ``grid`` are the evaluated feature values; ``mean``/``std`` average
    the ``ice`` matrix (one row per input row, one column per grid
    point) over rows; ``bounds`` stacks ``mean - std`` and
    ``mean + std`` (``std_coef=1.0``); ``density_counts`` /
    ``density_edges`` are the Doane histogram of the original column.
    """

    feature: int
    feature_name: str
    grid: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    bounds: np.ndarray
    ice: np.ndarray
    density_counts: np.ndarray
    density_edges: np.ndarray


def _pd_feature_index(model: object, feature: object, n_features: int) -> tuple[int, str]:
    """Resolve a feature spec (int index or name) to (index, name)."""
    names = getattr(model, "feature_names_", None)
    if isinstance(feature, (int, np.integer)):
        index = int(feature)
        if not 0 <= index < n_features:
            raise InterpretError(
                f"feature index {index} outside [0, {n_features})"
            )
        name = str(names[index]) if names is not None else str(index)
        return index, name
    if isinstance(feature, str):
        if names is None or feature not in list(names):
            raise InterpretError(
                f"feature name {feature!r} not found on the model "
                f"(feature_names_ is absent or does not contain it)"
            )
        return list(names).index(feature), feature
    raise InterpretError(
        f"feature must be an int index or a str name, got {feature!r}"
    )


def _pd_predictions(
    values: object, n_rows: int, *, explicit: bool, what: str
) -> np.ndarray:
    """Validate/normalize one predict call's output to shape (n,)."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise InterpretError(f"{what} returned non-numeric output: {error}") from error
    if array.ndim == 1:
        column = array
    elif array.ndim == 2 and array.shape[1] == 1:
        column = array[:, 0]
    elif array.ndim == 2 and array.shape[1] == 2 and not explicit:
        column = array[:, 1]
    elif array.ndim == 2 and array.shape[1] > 2 and not explicit:
        raise InterpretError(
            f"{what} returned {array.shape[1]} probability columns: "
            f"partial_dependence is binary-only (the upstream refuses "
            f"multiclass too) -- pass predict_fn to pick a column"
        )
    else:
        raise InterpretError(
            f"{what} must return one value per row, got shape {array.shape}"
        )
    if column.shape[0] != n_rows:
        raise InterpretError(
            f"{what} returned {column.shape[0]} values for {n_rows} rows"
        )
    return np.asarray(column, dtype=float)


def partial_dependence(
    model: object,
    X: object,
    feature: object,
    num_points: int = 10,
    predict_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PartialDependenceRecord:
    """One feature's partial dependence with the full ICE matrix.

    The grid is the sorted unique values when there are at most
    ``num_points`` of them, else ``np.percentile(col, np.linspace(0,
    100, num_points))`` deduplicated.  Each grid point replaces the
    feature's column in a copy of X and predicts over ALL rows (the
    ICE curves); the partial dependence is the per-grid-point mean.

    ``predict_fn`` resolution: an explicit callable overrides
    everything; otherwise ``model.predict_proba(X)[:, 1]`` when present
    (a >2-column probability matrix raises the same typed refusal the
    upstream gives for multiclass), else ``model.predict(X)``.  There
    is no separate ``ice_curves`` entry point: the ICE matrix rides in
    the record (``record.ice``).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.interpret import partial_dependence
    >>> class _Line:
    ...     def __init__(self, coef):
    ...         self.coef = coef
    ...     def predict(self, X):
    ...         return np.asarray(X, dtype=float) @ self.coef
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> rec = partial_dependence(_Line(np.array([2.0, -1.0])), X,
    ...     feature=0, num_points=3)
    >>> [round(v, 4) for v in rec.grid.tolist()]
    [0.0, 1.0, 2.0]
    >>> [round(v, 4) for v in rec.mean.tolist()]
    [-1.0, 1.0, 3.0]

    :param model: any fitted model with ``predict_proba`` or ``predict``
        (or supply ``predict_fn``).
    :param X: the background matrix ``(n_rows, n_features)``.
    :param feature: int column index or str feature name (resolved via
        the model's ``feature_names_`` when present).
    :param num_points: grid size (at least 2).
    :param predict_fn: optional explicit ``(n, k) -> (n,)`` override.
    :returns: the :class:`PartialDependenceRecord` (grid, mean, std,
        bounds, ice, density, feature index/name).
    :raises InterpretError: on bad shapes, unknown features, a
        multiclass probability matrix, or a model with no prediction
        surface.
    """
    try:
        matrix = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as error:
        raise InterpretError(f"X must be numeric: {error}") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise InterpretError(
            f"X must be a non-empty 2-D matrix, got shape {matrix.shape}"
        )
    n_rows, n_features = matrix.shape
    if int(num_points) < 2:
        raise InterpretError(f"num_points must be >= 2, got {num_points}")
    index, name = _pd_feature_index(model, feature, n_features)

    if predict_fn is not None:
        if not callable(predict_fn):
            raise InterpretError("predict_fn must be callable")

        def evaluate(mutated: np.ndarray) -> np.ndarray:
            return _pd_predictions(
                predict_fn(mutated), n_rows, explicit=True, what="predict_fn"
            )

    elif hasattr(model, "predict_proba"):

        def evaluate(mutated: np.ndarray) -> np.ndarray:
            return _pd_predictions(
                model.predict_proba(mutated),  # type: ignore[attr-defined]
                n_rows,
                explicit=False,
                what="model.predict_proba",
            )

    elif hasattr(model, "predict"):

        def evaluate(mutated: np.ndarray) -> np.ndarray:
            return _pd_predictions(
                model.predict(mutated),  # type: ignore[attr-defined]
                n_rows,
                explicit=True,
                what="model.predict",
            )

    else:
        raise InterpretError(
            "model needs predict_proba or predict, or an explicit predict_fn"
        )

    column = matrix[:, index]
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        raise InterpretError(
            f"feature {name!r} has no finite values to build a grid from"
        )
    uniques = np.unique(finite)
    if uniques.size <= int(num_points):
        grid = uniques
    else:
        grid = np.unique(
            np.percentile(finite, np.linspace(0.0, 100.0, int(num_points)))
        )
    ice = np.empty((n_rows, int(grid.size)), dtype=float)
    for point in range(int(grid.size)):
        mutated = matrix.copy()
        mutated[:, index] = grid[point]
        ice[:, point] = evaluate(mutated)
    mean = ice.mean(axis=0)
    std = ice.std(axis=0)
    bounds = np.vstack([mean - std, mean + std])
    density_counts, density_edges = np.histogram(finite, bins="doane")
    return PartialDependenceRecord(
        feature=index,
        feature_name=name,
        grid=np.asarray(grid, dtype=float),
        mean=mean,
        std=std,
        bounds=bounds,
        ice=ice,
        density_counts=np.asarray(density_counts, dtype=np.int64),
        density_edges=np.asarray(density_edges, dtype=float),
    )

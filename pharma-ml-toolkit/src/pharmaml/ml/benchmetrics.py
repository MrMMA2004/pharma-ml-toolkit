"""Chemistry benchmark metrics, hand-rolled (100x-175).

Round 175 reviewed four molecular-ML repositories file by file --
``chemprop/chemprop`` (``nn/metrics.py``), ``deepchem/deepchem``
(``metrics/`` + ``utils/evaluate.py``), ``mims-harvard/TDC``
(``evaluator.py`` + ``chem_utils/evaluator.py``) and
``swansonk14/admet_ai`` (the per-endpoint AUPRC/AUROC/R^2/MAE table
that drives the ADMET card).  All four rank models with the same small
set of scorers, and all four either delegate to sklearn or hand-roll
them; this module is the toolkit's own hand-rolled copy of that
surface, so benchmark numbers stay computable without sklearn and stay
comparable with it:

* AUROC -- the rank form of the Mann-Whitney U statistic with
  mid-ranks for ties, ``(sum(ranks_pos) - n_pos(n_pos+1)/2) /
  (n_pos*n_neg)`` (TDC/chemprop's ``roc`` / ``roc-auc``).
* AUPRC -- average-precision semantics: descending unique thresholds,
  ``sum((R_k - R_{k-1}) * P_k)`` (TDC's ``pr-auc``; sklearn's
  ``average_precision_score`` agrees to 1e-12, verified below).
* MCC, Cohen's kappa, balanced accuracy, specificity, precision /
  recall / F1 at threshold 0.5 -- the confusion-matrix family none of
  which existed in the toolkit before this round.
* BEDROC -- the Truchon-Bayly (2007) early-recognition score, ported
  from RDKit's ``ML/Scoring/Scoring.py`` ``CalcBEDROC`` (RIE with the
  exponential rank decay ``exp(-alpha*(i+1)/N)``, then the
  ``(RIE - RIEmin)/(RIEmax - RIEmin)`` normalisation).
* ``range_logAUC`` -- TDC's ``evaluator.py`` verbatim logic: the area
  under the ROC curve on a ``log10(FPR)`` axis restricted to
  ``FPR in [0.001, 0.1]`` (a perfect classifier scores 1, a random one
  ~0.0215).
* ``recall_at_precision_k`` / ``precision_at_recall_k`` -- TDC's
  PR-curve reachability queries at the default 0.9 threshold.
* concordance index -- deepchem's ``concordance_index`` pairwise
  ordering score with 0.5 credit for ties (the PK/PD metric GCPNE and
  survival pipelines share).
* the multitask wrapper :func:`multitask_metric` -- deepchem's
  ``Metric.compute_metric`` contract (per-task scores on the
  ``w != 0``/non-NaN mask, averaged over tasks) and chemprop's masked
  accumulator convention (``total / num_samples`` over finite targets).

The ROC and PR curves themselves are built by the same threshold walk
sklearn uses (unique descending scores, ties share one threshold), and
the doctests pin the hand-rolled AUROC/AUPRC/MCC/kappa to sklearn's
reference implementations so any drift shows up immediately.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_trapezoid: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
# numpy 1.x spells it trapz, 2.x renamed it trapezoid -- both supported

from pharmaml.exceptions import DataValidationError

__all__ = [
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "bedroc",
    "cohens_kappa",
    "concordance_index",
    "f1_score",
    "mae",
    "mcc",
    "multitask_metric",
    "pearson_pcc",
    "pr_curve_points",
    "precision",
    "precision_at_recall_k",
    "r2_score",
    "range_log_auc",
    "recall",
    "recall_at_precision_k",
    "rmse",
    "roc_curve_points",
    "spearman_rho",
    "specificity",
]


# ------------------------------------------------------------------ #
# Input hygiene                                                       #
# ------------------------------------------------------------------ #


def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Flatten to a float 1-D array, rejecting empty input."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise DataValidationError(f"{name} is empty -- nothing to score")
    return arr


def _binary_labels(y_true: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate 0/1 labels (tolerating float encodings)."""
    labels = _as_1d_float(y_true, "y_true")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise DataValidationError(
            "y_true must contain only 0/1 labels for classification metrics"
        )
    return labels


def _align(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = _binary_labels(y_true)
    score = _as_1d_float(y_score, "y_score")
    if truth.size != score.size:
        raise DataValidationError(
            f"y_true and y_score lengths differ ({truth.size} vs {score.size})"
        )
    return truth, score


# ------------------------------------------------------------------ #
# Curve construction (shared by AUROC / AUPRC / BEDROC / logAUC)      #
# ------------------------------------------------------------------ #


def roc_curve_points(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr)`` arrays for the ROC curve, ties grouped.

    One point per *unique* descending score threshold, prepended with
    the (0, 0) origin -- sklearn's ``roc_curve`` layout without its
    ``drop_intermediate`` thinning.

    Example
    -------
    >>> fpr, tpr = roc_curve_points([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> [round(float(v), 3) for v in fpr]
    [0.0, 0.0, 0.5, 0.5, 1.0]
    >>> [round(float(v), 3) for v in tpr]
    [0.0, 0.5, 0.5, 1.0, 1.0]
    """
    truth, score = _align(y_true, y_score)
    n_pos = float(truth.sum())
    n_neg = float(truth.size - truth.sum())
    if n_pos == 0 or n_neg == 0:
        raise DataValidationError(
            "ROC needs at least one positive AND one negative sample"
        )
    order = np.argsort(-score, kind="stable")
    truth_sorted = truth[order]
    score_sorted = score[order]
    # Group ties: a new threshold starts wherever the score changes.
    new_thresh = np.empty(truth.size, dtype=bool)
    new_thresh[0] = True
    np.not_equal(score_sorted[1:], score_sorted[:-1], out=new_thresh[1:])
    group_id = np.cumsum(new_thresh) - 1
    n_groups = int(group_id[-1]) + 1
    group_tp = np.zeros(n_groups)
    group_count = np.zeros(n_groups)
    np.add.at(group_tp, group_id, truth_sorted)
    np.add.at(group_count, group_id, 1.0)
    cum_tp = np.cumsum(group_tp)
    cum_fp = np.cumsum(group_count) - cum_tp
    fpr = np.concatenate(([0.0], cum_fp / n_neg))
    tpr = np.concatenate(([0.0], cum_tp / n_pos))
    return fpr, tpr


def pr_curve_points(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(precision, recall)`` arrays, ties grouped.

    sklearn's ``precision_recall_curve`` convention: descending unique
    thresholds, ending with ``precision=1, recall=0`` (the empty
    prediction set), *without* that sentinel being reachable.

    Example
    -------
    >>> pr, rc = pr_curve_points([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> [round(float(v), 3) for v in pr]
    [1.0, 0.5, 0.667, 0.5, 1.0]
    >>> [round(float(v), 3) for v in rc]
    [0.5, 0.5, 1.0, 1.0, 0.0]
    """
    truth, score = _align(y_true, y_score)
    n_pos = float(truth.sum())
    if n_pos == 0:
        raise DataValidationError("PR curve needs at least one positive sample")
    order = np.argsort(-score, kind="stable")
    truth_sorted = truth[order]
    score_sorted = score[order]
    new_thresh = np.empty(truth.size, dtype=bool)
    new_thresh[0] = True
    np.not_equal(score_sorted[1:], score_sorted[:-1], out=new_thresh[1:])
    group_id = np.cumsum(new_thresh) - 1
    n_groups = int(group_id[-1]) + 1
    group_tp = np.zeros(n_groups)
    group_count = np.zeros(n_groups)
    np.add.at(group_tp, group_id, truth_sorted)
    np.add.at(group_count, group_id, 1.0)
    cum_tp = np.cumsum(group_tp)
    cum_count = np.cumsum(group_count)
    cum_count - cum_tp
    recall = cum_tp / n_pos
    precision = cum_tp / cum_count  # every threshold group is non-empty
    # sklearn's terminal sentinel: the empty set has precision 1.
    precision = np.concatenate((precision, [1.0]))
    recall = np.concatenate((recall, [0.0]))
    return precision, recall


# ------------------------------------------------------------------ #
# Ranking metrics                                                      #
# ------------------------------------------------------------------ #


def auroc(y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic.

    Ties contribute a half win each (mid-ranks), which reproduces
    sklearn's ``roc_auc_score`` exactly, including on constant scores.

    Example
    -------
    >>> y = [0, 0, 1, 1]
    >>> round(auroc(y, [0.1, 0.4, 0.35, 0.8]), 6)
    0.75
    >>> round(auroc(y, [0.2, 0.2, 0.2, 0.2]), 6)   # ties -> random
    0.5
    >>> from sklearn.metrics import roc_auc_score
    >>> scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.5, 0.3, 0.2, 0.1]
    >>> truth = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    >>> bool(auroc(truth, scores) == roc_auc_score(truth, scores))
    True
    """
    truth, score = _align(y_true, y_score)
    n_pos = float(truth.sum())
    n_neg = float(truth.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise DataValidationError(
            "AUROC needs at least one positive AND one negative sample"
        )
    # Mid-ranks (average ranks) handle ties the way sklearn does.
    order = np.argsort(score, kind="stable")
    ranks = np.empty(truth.size, dtype=float)
    sorted_scores = score[order]
    i = 0
    n = truth.size
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        mid = (i + j) / 2.0 + 1.0  # 1-based average rank of the tie block
        ranks[order[i : j + 1]] = mid
        i = j + 1
    sum_pos = float(ranks[truth == 1.0].sum())
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def auprc(y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Average precision (area under the step-wise PR curve).

    Matches ``sklearn.metrics.average_precision_score`` on unique
    thresholds; the terminal ``(recall=0, precision=1)`` sentinel
    contributes nothing.

    Example
    -------
    >>> y = [0, 0, 1, 1]
    >>> round(auprc(y, [0.1, 0.4, 0.35, 0.8]), 6)
    0.833333
    >>> from sklearn.metrics import average_precision_score
    >>> scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.5, 0.3, 0.2, 0.1]
    >>> truth = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    >>> bool(abs(auprc(truth, scores) - average_precision_score(truth, scores)) < 1e-12)
    True
    """
    precision, recall = pr_curve_points(y_true, y_score)
    if precision.size < 2:
        raise DataValidationError("AUPRC needs at least two threshold points")
    # sklearn's step convention: the recall gained at point k is credited
    # with point k's precision; the leading jump from recall 0 uses the
    # highest threshold's precision (diff with a prepended 0).
    recall_real = recall[:-1]
    precision_real = precision[:-1]
    dr = np.diff(recall_real, prepend=0.0)
    return float(np.sum(dr * precision_real))


# ------------------------------------------------------------------ #
# Confusion-matrix family                                              #
# ------------------------------------------------------------------ #


def _confusion(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    """(TP, FP, TN, FN) at ``threshold`` on the positive-class score."""
    truth, score = _align(y_true, y_score)
    pred = score >= threshold
    tp = float(np.sum((truth == 1.0) & pred))
    fp = float(np.sum((truth == 0.0) & pred))
    tn = float(np.sum((truth == 0.0) & ~pred))
    fn = float(np.sum((truth == 1.0) & ~pred))
    return tp, fp, tn, fn


def mcc(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Matthews correlation coefficient at threshold 0.5.

    Example
    -------
    >>> round(mcc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 6)
    0.57735
    >>> from sklearn.metrics import matthews_corrcoef
    >>> truth = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    >>> scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.5, 0.3, 0.2, 0.1]
    >>> bool(mcc(truth, scores) == matthews_corrcoef(
    ...     truth, [s >= 0.5 for s in scores]))
    True
    """
    tp, fp, tn, fn = _confusion(y_true, y_score)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def cohens_kappa(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Cohen's kappa at threshold 0.5 (TDC's ``kappa``).

    Example
    -------
    >>> from sklearn.metrics import cohen_kappa_score
    >>> truth = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    >>> scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.5, 0.3, 0.2, 0.1]
    >>> bool(cohens_kappa(truth, scores) == cohen_kappa_score(
    ...     truth, [s >= 0.5 for s in scores]))
    True
    """
    truth, score = _align(y_true, y_score)
    pred = score >= 0.5
    n = float(truth.size)
    po = float(np.sum(truth == pred)) / n
    p_yes = float(np.sum(truth == 1.0)) / n
    q_yes = float(np.sum(pred)) / n
    pe = p_yes * q_yes + (1.0 - p_yes) * (1.0 - q_yes)
    if pe == 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def accuracy(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Fraction correct at threshold 0.5."""
    truth, score = _align(y_true, y_score)
    return float(np.mean(truth == (score >= 0.5)))


def balanced_accuracy(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """(sensitivity + specificity) / 2 at threshold 0.5."""
    tp, fp, tn, fn = _confusion(y_true, y_score)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return float((sens + spec) / 2.0)


def specificity(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """TN / (TN + FP) at threshold 0.5."""
    _tp, fp, tn, _fn = _confusion(y_true, y_score)
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def precision(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """TP / (TP + FP) at threshold 0.5 (0.0 when no predictions)."""
    tp, fp, _tn, _fn = _confusion(y_true, y_score)
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """TP / (TP + FN) at threshold 0.5 -- TDC's ``recall``."""
    tp, _fp, _tn, fn = _confusion(y_true, y_score)
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(
    y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Harmonic mean of precision and recall at threshold 0.5."""
    p = precision(y_true, y_score)
    r = recall(y_true, y_score)
    return float(2.0 * p * r / (p + r)) if (p + r) else 0.0


# ------------------------------------------------------------------ #
# Regression metrics                                                   #
# ------------------------------------------------------------------ #


def rmse(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Root mean squared error (chemprop's default tracker)."""
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.size != pred.size:
        raise DataValidationError("y_true and y_pred lengths differ")
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mae(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Mean absolute error (the ADMET-AI regression score)."""
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.size != pred.size:
        raise DataValidationError("y_true and y_pred lengths differ")
    return float(np.mean(np.abs(truth - pred)))


def r2_score(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Coefficient of determination, 1 - SS_res/SS_tot."""
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.size != pred.size:
        raise DataValidationError("y_true and y_pred lengths differ")
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - np.sum((truth - pred) ** 2) / ss_tot)


def pearson_pcc(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Pearson correlation (TDC's ``pcc``); 0.0 when either side is constant."""
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.size != pred.size:
        raise DataValidationError("y_true and y_pred lengths differ")
    if truth.size < 2:
        return 0.0
    if np.all(truth == truth[0]) or np.all(pred == pred[0]):
        return 0.0
    return float(np.corrcoef(truth, pred)[1, 0])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based mid-ranks of ``values``."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_vals = values[order]
    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Spearman rank correlation (TDC's ``spearman``)."""
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.size != pred.size:
        raise DataValidationError("y_true and y_pred lengths differ")
    if truth.size < 2:
        return 0.0
    rt = _average_ranks(truth)
    rp = _average_ranks(pred)
    if np.all(rt == rt[0]) or np.all(rp == rp[0]):
        return 0.0
    return float(np.corrcoef(rt, rp)[1, 0])


# ------------------------------------------------------------------ #
# Early recognition                                                    #
# ------------------------------------------------------------------ #


def bedroc(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    alpha: float = 20.0,
) -> float:
    """BEDROC early-recognition score (Truchon & Bayly 2007), alpha=20.

    Ported from RDKit's ``ML/Scoring/Scoring.py``: RIE over the
    descending-ranked actives with the ``exp(-alpha*(i+1)/N)`` decay,
    then the affine map onto ``[0, 1]`` via ``RIEmax``/``RIEmin``.  A
    perfect ranking scores 1.0, a reversed one 0.0.

    Example
    -------
    >>> round(bedroc([1, 1, 0, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1]), 6)
    1.0
    >>> round(bedroc([0, 0, 0, 1, 1], [0.9, 0.8, 0.3, 0.2, 0.1]), 6)
    0.0
    >>> round(bedroc([1, 0, 1, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1]), 4)
    0.9823
    """
    truth, score = _align(y_true, y_score)
    if alpha <= 0.0:
        raise DataValidationError("alpha must be > 0")
    n = float(truth.size)
    order = np.argsort(-score, kind="stable")
    truth_sorted = truth[order]
    n_act = float(truth_sorted.sum())
    if n_act == 0.0:
        return 0.0
    if n_act == n:
        return 1.0
    positions = np.nonzero(truth_sorted == 1.0)[0] + 1  # 1-based ranks
    sum_exp = float(np.sum(np.exp(-alpha * positions / n)))
    denom = (1.0 / n) * ((1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0))
    rie = sum_exp / (n_act * denom)
    ratio = n_act / n
    rie_max = (1.0 - math.exp(-alpha * ratio)) / (ratio * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * ratio)) / (ratio * (1.0 - math.exp(alpha)))
    if rie_max == rie_min:
        return 1.0
    return float((rie - rie_min) / (rie_max - rie_min))


def range_log_auc(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    fpr_range: tuple[float, float] = (0.001, 0.1),
) -> float:
    """TDC's ``range_logAUC``: ROC area on a log10(FPR) axis.

    The curve is trimmed to ``FPR in [lower, upper]`` (default
    ``[0.001, 0.1]``, the virtual-screening regime where only the top
    predictions are experimentally testable), interpolated at both
    bounds, and the trapezoid area is divided by the log-width of the
    window.  Perfect ranking -> 1.0; random -> ~0.0215.

    Example
    -------
    >>> round(range_log_auc([1, 1, 0, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1]), 6)
    1.0
    >>> 0.0 < range_log_auc([1, 0, 1, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1]) < 0.6
    True
    """
    truth, score = _align(y_true, y_score)
    lower, upper = float(fpr_range[0]), float(fpr_range[1])
    if not 0.0 < lower < upper <= 1.0:
        raise DataValidationError(
            "fpr_range must satisfy 0 < lower < upper <= 1"
        )
    fpr, tpr = roc_curve_points(truth, score)
    # Interpolate TPR at both bounds and append (TDC's exact recipe).
    tpr = np.append(tpr, np.interp([lower, upper], fpr, tpr))
    fpr = np.append(fpr, [lower, upper])
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)
    with np.errstate(divide="ignore"):
        x = np.log10(fpr)
    log_lo = math.log10(lower)
    log_hi = math.log10(upper)
    keep = (x >= log_lo - 1e-12) & (x <= log_hi + 1e-12)
    trim_x = x[keep]
    trim_y = tpr[keep]
    if trim_x.size < 2:
        return 0.0
    # Trapezoid on the log axis (duplicated bound points collapse).
    area = float(_trapezoid(trim_y, trim_x))
    return area / (log_hi - log_lo)


def recall_at_precision_k(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.9,
) -> float:
    """First recall reached at ``precision >= threshold`` (TDC's ``rp@k``)."""
    precision, recall = pr_curve_points(y_true, y_score)
    hits = np.nonzero(precision >= threshold)[0]
    if hits.size == 0:
        return 0.0
    return float(recall[hits[0]])


def precision_at_recall_k(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.9,
) -> float:
    """Last precision sustained at ``recall >= threshold`` (TDC's ``pr@k``)."""
    precision, recall = pr_curve_points(y_true, y_score)
    hits = np.nonzero(recall >= threshold)[0]
    if hits.size == 0:
        return 0.0
    return float(precision[hits[-1]])


def concordance_index(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Pairwise ordering score with 0.5 credit for ties (deepchem's port).

    The fraction of comparable pairs whose predicted order agrees with
    the true order; identical predictions count half.

    Example
    -------
    >>> round(concordance_index([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]), 6)
    1.0
    >>> round(concordance_index([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]), 6)
    0.5
    >>> round(concordance_index([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), 6)
    0.0
    """
    truth = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if truth.size != pred.size:
        raise DataValidationError("y_true and y_pred lengths differ")
    concordant = 0.0
    total = 0.0
    for i in range(truth.size):
        for j in range(i + 1, truth.size):
            dt = truth[j] - truth[i]
            dp = pred[j] - pred[i]
            if dt == 0.0:
                continue  # not comparable in truth
            total += 1.0
            if dp == 0.0:
                concordant += 0.5
            elif (dt > 0.0) == (dp > 0.0):
                concordant += 1.0
    if total == 0.0:
        raise DataValidationError("no comparable pairs -- y_true is constant")
    return float(concordant / total)


# ------------------------------------------------------------------ #
# Multitask wrapper (deepchem Metric.compute_metric contract)          #
# ------------------------------------------------------------------ #


def multitask_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    w: np.ndarray | None = None,
    per_task: bool = False,
) -> float | tuple[float, list[float]]:
    """Per-task scores on the ``w != 0``/finite mask, averaged over tasks.

    This is deepchem's ``Metric.compute_metric`` contract: one score per
    task computed on the labelled samples only (``w == 0`` marks a
    missing label, NaN does too), then the unweighted mean over tasks --
    the same number chemprop's masked accumulators report.

    Parameters
    ----------
    y_true, y_pred:
        ``(n_samples, n_tasks)`` arrays (1-D input is treated as one
        task).
    metric:
        A two-argument scorer such as :func:`auroc` or :func:`rmse`.
    w:
        Optional sample weights / label mask, same shape; ``0`` drops
        the sample.
    per_task:
        Return ``(mean, per_task_list)`` instead of the mean alone.

    Example
    -------
    >>> truth = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> pred = np.array([[0.9, 0.2], [0.4, 0.8], [0.6, 0.7]])
    >>> weights = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    >>> mean, per = multitask_metric(truth, pred, auroc, w=weights, per_task=True)
    >>> [round(v, 3) for v in per]   # task 1 masked to two samples
    [1.0, 1.0]
    >>> round(float(mean), 3)
    1.0
    >>> multitask_metric(truth, pred, rmse)  # doctest: +ELLIPSIS
    0.28...
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
        pred = pred[:, None]
    if truth.shape != pred.shape:
        raise DataValidationError(
            f"y_true {truth.shape} and y_pred {pred.shape} shapes differ"
        )
    if w is None:
        weights = np.ones_like(truth)
    else:
        weights = np.asarray(w, dtype=float)
        if weights.ndim == 1:
            weights = weights[:, None]
        if weights.shape != truth.shape:
            raise DataValidationError("w must match y_true's shape")
    scores: list[float] = []
    for task in range(truth.shape[1]):
        mask = (weights[:, task] != 0.0) & np.isfinite(truth[:, task])
        if mask.sum() == 0:
            continue  # entirely missing task: excluded from the average
        scores.append(float(metric(truth[mask, task], pred[mask, task])))
    if not scores:
        raise DataValidationError("no task has any labelled sample")
    mean = float(np.mean(scores))
    if per_task:
        return mean, scores
    return mean

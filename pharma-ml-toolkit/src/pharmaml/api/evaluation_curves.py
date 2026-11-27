"""part21 §3 — the evaluation-sample curve and diagnostic computations.

Everything here derives from the evaluation block's own bounded sample
rows (the aligned actual/predicted pairs the parity lane already
plots) and, for classification, the probability rows aligned to them
by position over the same prediction pass. These are DISPLAY
diagnostics for the same sample the parity chart draws — never a
second evaluation of the model's authoritative prediction set (the
Export stage registers the full table; the cross-validated metrics
live in the train answer's own ``cv_scores`` block, which stays the
authoritative model-quality lane).

Design rails (MASTER_PROMPT part21 §4 — scientific validity):

* no fabricated values — every curve is computed from real aligned
  data the prediction pass already produced;
* a chart whose required data is absent is simply not attached (the
  frontend renders an explanatory state, never an invented curve);
* the statistics stay in the backend (the frontend renders what this
  module computed — no sklearn math duplicated in TypeScript);
* a class with no positives or no negatives in the sample has no ROC
  or PR curve for that class — the honest omission, never a degenerate
  line;
* zero-division metrics (a class the sample never predicted, or one
  with no actuals) report 0.0 with the support counts exposed so the
  zero is legible as "no samples in the sample", not as performance;
* the sample bound (``_MAX_EVALUATION_ROWS``) applies before this
  module sees the rows — these curves describe the plotted sample,
  exactly like the parity chart does today, and the block's own
  ``n_rows_evaluated``/``n_rows_available`` pair keeps that scoping
  honest on the surface.
"""

from __future__ import annotations

import math
from statistics import NormalDist, mean, median, pstdev
from typing import Any

__all__ = ["enrich_evaluation_block"]

#: Decimation bound for large curve point lists (rendering bound only;
#: the underlying rows keep their full fidelity in the block itself).
_MAX_CURVE_POINTS = 200

#: Equal-width bins over the residual range (the same count the
#: matplotlib residual recipe in ``viz/ml_plots.py`` draws).
_HISTOGRAM_BINS = 20

#: Reliability-diagram bins over [0, 1] predicted probability.
_CALIBRATION_BINS = 10


def _finite(x: float) -> float:
    """Coerce to a JSON-safe finite float (NaN/inf never leave here)."""
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return float(x)


def _decimate(values: list[float]) -> list[float]:
    """Rendering bound: keep at most ``_MAX_CURVE_POINTS`` points."""
    if len(values) <= _MAX_CURVE_POINTS:
        return values
    step = len(values) / _MAX_CURVE_POINTS
    kept: list[float] = []
    i = 0.0
    while i < len(values):
        kept.append(values[int(i)])
        i += step
    if kept[-1] != values[-1]:
        kept.append(values[-1])
    return kept


# ---------------------------------------------------------- regression


def residual_diagnostics(residuals: list[float]) -> dict[str, Any] | None:
    """The residual histogram, summary stats, and normal Q–Q points.

    Over the sample's own numeric residuals (``actual - predicted``,
    exactly the values the evaluation rows already carry). Requires
    at least four residuals — fewer makes neither a histogram nor a
    quantile plot honest. A constant residual set (zero variance) has
    no Q–Q plot (every theoretical quantile maps to the same point —
    the honest null).
    """
    n = len(residuals)
    if n < 4:
        return None
    mean_v = mean(residuals)
    median_v = median(residuals)
    std_v = pstdev(residuals) if n > 1 else 0.0
    lo, hi = min(residuals), max(residuals)
    if hi > lo:
        width = (hi - lo) / _HISTOGRAM_BINS
        edges = [lo + width * i for i in range(_HISTOGRAM_BINS + 1)]
        counts = [0] * _HISTOGRAM_BINS
        for r in residuals:
            idx = min(int((r - lo) / width), _HISTOGRAM_BINS - 1)
            counts[idx] += 1
    else:
        # a constant residual: one bin either side of the value
        edges = [lo - 0.5, lo + 0.5]
        counts = [n]
    qq: dict[str, Any] | None = None
    if std_v > 0:
        dist = NormalDist(0.0, 1.0)
        ordered = sorted(residuals)
        theoretical: list[float] = []
        empirical: list[float] = []
        for i, r in enumerate(ordered):
            p = (i + 1.0 / 3.0) / (n + 1.0 / 3.0)  # Blom positions
            theoretical.append(dist.inv_cdf(p))
            empirical.append((r - mean_v) / std_v)
        qq = {
            "theoretical": _decimate(theoretical),
            "empirical": _decimate(empirical),
        }
    return {
        "histogram": {
            "bin_edges": [_finite(e) for e in edges],
            "counts": counts,
        },
        "stats": {
            "mean": _finite(mean_v),
            "median": _finite(median_v),
            "std": _finite(std_v),
            "min": _finite(lo),
            "max": _finite(hi),
            "n": n,
        },
        "qq": qq,
    }


# ------------------------------------------------------- classification


def class_metrics(
    labels_true: list[str],
    labels_pred: list[str],
    classes: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Per-class precision/recall/F1/support + macro/accuracy.

    The class ORDER follows the estimator's own ``classes_`` (the
    model's original ordering — part21 §4's class-ordering rail)
    when the probability block carried it, else the sorted label
    set (the confusion chart's own derivation). Zero-division
    metrics report 0.0; the support counts stay exposed so the zero
    reads as "no samples in the sample".
    """
    if not labels_true:
        return None
    if classes:
        seen = {str(c) for c in labels_true} | {str(c) for c in labels_pred}
        ordered = [str(c) for c in classes] + sorted(
            seen - {str(c) for c in classes}
        )
    else:
        ordered = sorted(
            {str(c) for c in labels_true} | {str(c) for c in labels_pred}
        )
    rows: list[dict[str, Any]] = []
    tp_total = 0
    for label in ordered:
        tp = sum(
            1 for t, p in zip(labels_true, labels_pred) if t == label and p == label
        )
        fp = sum(1 for t, p in zip(labels_true, labels_pred)
                 if t != label and p == label)
        fn = sum(1 for t, p in zip(labels_true, labels_pred)
                 if t == label and p != label)
        support = sum(1 for t in labels_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "label": label,
                "precision": _finite(precision),
                "recall": _finite(recall),
                "f1": _finite(f1),
                "support": support,
            }
        )
        tp_total += tp
    n = len(labels_true)
    macro = {
        key: sum(r[key] for r in rows) / len(rows)
        for key in ("precision", "recall", "f1")
    }
    return {
        "rows": rows,
        "macro": {k: _finite(v) for k, v in macro.items()},
        "accuracy": _finite(tp_total / n) if n else 0.0,
        "n": n,
    }


def _binary_roc(
    scores: list[float], flags: list[bool]
) -> dict[str, Any] | None:
    """The ROC step walk for one class (OvR: flag = is-this-class).

    Points at each DISTINCT score threshold (ties grouped — the
    standard curve), a leading (0, 0) anchor, AUC by the step
    trapezoid over the grouped walk (equal to the tie-aware
    Mann–Whitney U / the Gini form). No curve when the class has no
    positives or no negatives in the sample.
    """
    n_pos = sum(1 for f in flags if f)
    n_neg = len(flags) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [scores[order[0]]]
    tp = 0
    fp = 0
    i = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    while i < len(order):
        score = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == score:
            if flags[order[j]]:
                tp += 1
            else:
                fp += 1
            j += 1
        cur_fpr = fp / n_neg
        cur_tpr = tp / n_pos
        auc += (cur_fpr - prev_fpr) * (cur_tpr + prev_tpr) / 2.0
        fpr.append(cur_fpr)
        tpr.append(cur_tpr)
        thresholds.append(score)
        prev_fpr = cur_fpr
        prev_tpr = cur_tpr
        i = j
    return {
        "fpr": _decimate(fpr),
        "tpr": _decimate(tpr),
        "thresholds": _decimate(thresholds),
        "auc": _finite(auc),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def _binary_pr(
    scores: list[float], flags: list[bool]
) -> dict[str, Any] | None:
    """The PR step walk for one class (OvR).

    Recall is monotone by construction; precision at each grouped
    threshold; average precision by the step integral over recall
    gains (the tie-aware AP). No curve for a class without positives.
    """
    n_pos = sum(1 for f in flags if f)
    if n_pos == 0:
        return None
    order = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )
    recall = [0.0]
    precision = [1.0]
    thresholds = [scores[order[0]]]
    tp = 0
    i = 0
    ap = 0.0
    prev_recall = 0.0
    while i < len(order):
        score = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == score:
            if flags[order[j]]:
                tp += 1
            j += 1
        cur_recall = tp / n_pos
        cur_precision = tp / j
        ap += (cur_recall - prev_recall) * cur_precision
        recall.append(cur_recall)
        precision.append(cur_precision)
        thresholds.append(score)
        prev_recall = cur_recall
        i = j
    return {
        "recall": _decimate(recall),
        "precision": _decimate(precision),
        "thresholds": _decimate(thresholds),
        "average_precision": _finite(ap),
        "prevalence": n_pos / len(flags),
        "n_pos": n_pos,
    }


def _calibration(
    scores: list[float], flags: list[bool]
) -> dict[str, Any] | None:
    """The reliability diagram bins for one binary curve.

    Ten equal-width bins over the predicted probability of the
    positive class; each bin reports its mean predicted probability,
    the observed positive fraction, and its own count — an empty bin
    stays a count-0 row (an honest gap, never dropped to fake density).
    """
    if len(scores) == 0:
        return None
    width = 1.0 / _CALIBRATION_BINS
    sums = [0.0] * _CALIBRATION_BINS
    hits = [0] * _CALIBRATION_BINS
    counts = [0] * _CALIBRATION_BINS
    for score, flag in zip(scores, flags):
        idx = min(int(score / width), _CALIBRATION_BINS - 1)
        if score < 0:
            idx = 0
        sums[idx] += score
        if flag:
            hits[idx] += 1
        counts[idx] += 1
    bins = []
    for idx in range(_CALIBRATION_BINS):
        count = counts[idx]
        bins.append(
            {
                "mean_predicted": _finite(sums[idx] / count) if count else 0.0,
                "observed": _finite(hits[idx] / count) if count else 0.0,
                "count": count,
            }
        )
    return {"bins": bins, "n": len(scores)}


def _class_column(classes: list[Any], label: str) -> int | None:
    for i, c in enumerate(classes):
        if str(c) == label:
            return i
    return None


# ------------------------------------------------------------ the entry


def enrich_evaluation_block(block: dict[str, Any], proba: dict[str, Any] | None) -> None:
    """Attach the sample-derived diagnostics to the block (in place).

    ``block`` is the evaluation block from the SAME prediction pass;
    ``proba`` (optional) is the pass's own probability block — the
    class list and per-row probability vectors, aligned to the
    evaluation rows by position (both walk the same X frame in the
    same order). Regression rows gain the residual histogram, summary
    stats, and Q–Q points; classification rows gain per-class
    metrics, and — only when the pass carried probabilities — the
    per-class ROC/PR curves and the binary reliability diagram.
    Every attachment is conditional on its real data being present.
    """
    rows = block.get("rows") or []
    task = block.get("task")
    if task == "regression":
        residuals = [
            float(r["residual"])
            for r in rows
            if isinstance(r.get("residual"), (int, float))
        ]
        diag = residual_diagnostics(residuals)
        if diag is not None:
            block["residual_diagnostics"] = diag
        return
    if task != "classification" or not rows:
        return
    labels_true = [str(r.get("actual")) for r in rows]
    labels_pred = [str(r.get("predicted")) for r in rows]
    classes: list[Any] | None = None
    proba_rows: list[list[float]] | None = None
    if proba is not None:
        raw_classes = proba.get("classes")
        raw_rows = proba.get("rows")
        if (
            isinstance(raw_classes, list)
            and isinstance(raw_rows, list)
            and len(raw_rows) >= len(rows)
        ):
            # position alignment: the evaluation rows are the FIRST
            # len(rows) rows of the prediction set; the probability
            # rows cover the whole set in the same order
            classes = raw_classes
            proba_rows = [
                [float(v) for v in row] for row in raw_rows[: len(rows)]
            ]
    metrics = class_metrics(labels_true, labels_pred, classes)
    if metrics is not None:
        block["class_metrics"] = metrics
    if proba_rows is None or not classes:
        return
    width = len(classes)
    if any(len(row) != width for row in proba_rows):
        return  # a malformed probability row never feeds a curve
    roc_series: list[dict[str, Any]] = []
    pr_series: list[dict[str, Any]] = []
    for i, cls in enumerate(classes):
        label = str(cls)
        flags = [t == label for t in labels_true]
        scores = [row[i] for row in proba_rows]
        roc = _binary_roc(scores, flags)
        if roc is not None:
            roc_series.append({"label": label, **roc})
        pr = _binary_pr(scores, flags)
        if pr is not None:
            pr_series.append({"label": label, **pr})
    if roc_series:
        block["roc_curves"] = roc_series
    if pr_series:
        block["pr_curves"] = pr_series
    if len(classes) == 2:
        # the reliability diagram is the binary case's own honest lane
        positive = classes[-1]
        pos_label = str(positive)
        pos_idx = _class_column(classes, pos_label)
        if pos_idx is not None:
            flags = [t == pos_label for t in labels_true]
            scores = [row[pos_idx] for row in proba_rows]
            cal = _calibration(scores, flags)
            if cal is not None:
                cal["positive_class"] = pos_label
                block["calibration"] = cal

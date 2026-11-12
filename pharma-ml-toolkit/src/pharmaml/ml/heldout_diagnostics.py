"""Descriptive held-out probability and distance diagnostics.

These diagnostics do not fit or calibrate a prediction model, select a
threshold on evaluation data, or certify predictions as safe.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.applicability import knn_applicability_domain


def classification_probability_diagnostics(
    probabilities: Any, labels: Any, classes: list[str], *, bins: int = 10,
) -> dict[str, Any]:
    """Compute multiclass Brier score and one-vs-rest fixed-bin reliability.

    The Brier score is the mean, per observation, of the *sum* of squared
    probability errors across classes (range 0–2, including binary tasks).
    Bins are [lower, upper), except the last includes 1. Empty bins are omitted.
    These held-out summaries do not recalibrate probabilities; bins can be
    noisy, especially with small or dependent evaluation samples.
    """
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 100:
        raise DataValidationError("bins must be an integer between 2 and 100")
    if (not isinstance(classes, list) or len(classes) < 2
            or any(not isinstance(c, str) or not c.strip() for c in classes)
            or len(set(classes)) != len(classes)):
        raise DataValidationError("classes must be unique nonempty strings")
    try:
        p = np.asarray(probabilities, dtype=float)
        truth = np.asarray(labels, dtype=object)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("invalid probabilities or labels") from exc
    if (p.ndim != 2 or p.shape[1] != len(classes) or p.shape[0] == 0
            or truth.ndim != 1 or len(truth) != len(p)):
        raise DataValidationError("probabilities and labels must be aligned and nonempty")
    if (not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1.0)):
        raise DataValidationError("probabilities must be finite, in [0, 1], and sum to 1")
    if any(not isinstance(label, str) or label not in classes for label in truth):
        raise DataValidationError("evaluation labels must belong to the trained class space")
    one_hot = np.column_stack([truth == label for label in classes]).astype(float)
    per_class = []
    for index, label in enumerate(classes):
        values = p[:, index]
        # Multiplication may round a probability just below 1 to exactly bins.
        indices = np.minimum((values * bins).astype(int), bins - 1)
        rows = []
        for bin_id in range(bins):
            selected = indices == bin_id
            if selected.any():
                rows.append({"lower_inclusive": bin_id / bins, "upper": (bin_id + 1) / bins,
                             "upper_inclusive": bin_id == bins - 1, "n": int(selected.sum()),
                             "mean_probability": float(values[selected].mean()),
                             "observed_fraction": float(one_hot[selected, index].mean())})
        per_class.append({"class": label, "bins": rows})
    return {"n_test": len(truth), "n_bins": bins,
            "multiclass_brier_sum": float(np.square(p - one_hot).sum(axis=1).mean()),
            "per_class": per_class,
            "limitations": ("Descriptive fixed-bin held-out reliability, not probability recalibration. "
                            "Small, correlated or selectively chosen evaluation samples can be misleading; "
                            "do not tune on this test set.")}


def distance_domain_diagnostics(
    model: Pipeline, train: pd.DataFrame, test: pd.DataFrame, features: list[str], *,
    covered: Any, k: int = 5, percentile: float = 95.0,
) -> dict[str, Any]:
    """Summarize train-referenced kNN distance and held-out empirical coverage.

    Use the already-fitted model's train-only median imputer, and fit the kNN
    scaler/index and percentile threshold exclusively on imputed training
    rows. Coverage by distance slice is descriptive. No individual is deemed
    reliable, and structural similarity or out-of-distribution behavior is not
    established. For fewer than k+1 train rows, report unavailable instead of
    changing k silently.
    """
    if len(train) <= k:
        return {"status": "unavailable", "reason": f"k={k} requires at least {k + 1} training rows"}
    imputer = model.named_steps["prep"].named_steps["impute"]
    train_values = imputer.transform(train[features])
    test_values = imputer.transform(test[features])
    domain = knn_applicability_domain(train_values, test_values, k=k, percentile=percentile)
    hits = np.asarray(covered)
    if hits.ndim != 1 or len(hits) != len(test) or hits.dtype.kind != "b":
        raise DataValidationError("coverage must be a boolean vector aligned with test rows")
    slices = []
    for name, mask in (("inside", domain.inside_ad), ("outside", ~domain.inside_ad)):
        n = int(mask.sum())
        count = int(hits[mask].sum())
        slices.append({"distance_slice": name, "n": n, "n_covered": count,
                       "coverage": count / n if n else None})
    return {"status": "computed", "k": k, "training_percentile": percentile,
            "distance_threshold": domain.distance_threshold,
            "n_test": len(test), "n_inside": int(domain.inside_ad.sum()), "coverage_by_distance": slices,
            "limitations": ("Training-referenced descriptor-distance heuristic only; not a chemical "
                            "similarity, safety, reliability or coverage guarantee. Distance-slice "
                            "coverage is descriptive and can be unstable or biased.")}

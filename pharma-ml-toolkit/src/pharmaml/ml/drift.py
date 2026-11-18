"""Distribution shift: has a new batch stopped looking like the training data?

The applicability box (:mod:`pharmaml.ml.applicability`) answers a
per-compound question — *is this row inside the training range?*  It
cannot answer the batch-level question a retraining decision actually
turns on: *has the whole distribution moved?*  A thousand new compounds
can each sit inside the box while their mean and spread drift far enough
that the model is quietly extrapolating on the population, not the point.

This module compares a **reference** distribution (the training frame, or
the per-feature quantile sketch a bundle can carry) against a **current**
batch with two standard, complementary statistics:

* **Population Stability Index (PSI)** — bin the reference into deciles,
  measure how the current batch's bin proportions differ:
  ``sum((cur% - ref%) * ln(cur% / ref%))``.  The credit-risk convention:
  ``< 0.1`` stable, ``0.1 - 0.2`` moderate, ``> 0.2`` a shift worth
  acting on.  PSI needs only the reference *bins*, so a bundle can carry
  it as a small sketch and detect drift with no training data on hand.
* **Kolmogorov-Smirnov** — the maximum gap between the two empirical CDFs
  with its p-value, sensitive to location *and* shape shifts PSI's fixed
  bins can miss.  This one needs the raw reference values.

Both are reported per feature; the module never decides to retrain, it
surfaces which features moved and by how much, so a human (or a
pipeline) can.  The direction of the shift is stated too — a mean that
climbed is a different problem from one that fell.

Examples
--------
>>> import numpy as np, pandas as pd
>>> rng = np.random.default_rng(0)
>>> ref = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(5, 2, 400)})
>>> same = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(5, 2, 400)})
>>> moved = pd.DataFrame({"a": rng.normal(3, 1, 400), "b": rng.normal(5, 2, 400)})

A stable batch reads as stable:

>>> report = detect_drift(ref, same)
>>> report.n_drifted
0
>>> bool(report.drift_share < 0.5)
True

A shifted feature is named, with its direction:

>>> report = detect_drift(ref, moved)
>>> [f.name for f in report.drifted]
['a']
>>> float(report.worst.psi) > 0.2
True

The PSI alone, for one pair of arrays:

>>> round(psi(ref["a"], ref["a"]), 6)      # identical -> 0
0.0
>>> float(psi(ref["a"], moved["a"])) > 0.2
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError

__all__ = [
    "DriftReport",
    "FeatureDrift",
    "detect_drift",
    "drift_from_metadata",
    "feature_distributions_from_frame",
    "ks_statistic",
    "psi",
]

#: PSI below this is read as "stable".
DEFAULT_WARN_PSI = 0.1
#: PSI at or above this is read as "shifted".
DEFAULT_SHIFT_PSI = 0.2
#: Floor added to zero proportions before the PSI logarithm.
_PSI_EPSILON = 1e-6


def _as_1d(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a finite 1-D float array.

    Raises
    ------
    DataValidationError
        When the input is empty, not 1-D, or carries NaN/±inf (a drift
        statistic over a poisoned vector is meaningless).
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise DataValidationError(f"{name} is empty — nothing to compare")
    if not np.isfinite(array).all():
        n_bad = int((~np.isfinite(array)).sum())
        raise DataValidationError(
            f"{name} contains {n_bad} non-finite value(s) — drift needs "
            "finite values (drop or impute first)"
        )
    return array


def _quantile_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Monotone bin edges from ``reference`` deciles, deduplicated.

    A near-constant reference collapses several quantiles onto one value;
    the duplicate edges are dropped so the histogram has no empty bin and
    the PSI stays finite.
    """
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        # A constant reference: a single degenerate bin around the value.
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(counts.size, dtype=float)
    return counts.astype(float) / float(total)


def psi(reference: Any, current: Any, *, bins: int = 10) -> float:
    """Population Stability Index of ``current`` against ``reference``.

    Bins the reference into ``bins`` quantile bins, then sums
    ``(cur% - ref%) * ln(cur% / ref%)`` over the bins.  ``0`` means the
    two distributions land identically in the reference's bins; the usual
    reading is ``< 0.1`` stable, ``0.1 - 0.2`` moderate, ``> 0.2``
    shifted.  A degenerate (constant) reference falls back to one bin.
    """
    ref = _as_1d(reference, "reference")
    cur = _as_1d(current, "current")
    if bins < 2:
        raise DataValidationError(f"bins must be >= 2, got {bins}")
    edges = _quantile_edges(ref, bins)
    ref_props = _proportions(ref, edges)
    cur_props = _proportions(cur, edges)
    total = 0.0
    for ref_p, cur_p in zip(ref_props, cur_props, strict=True):
        r = max(ref_p, _PSI_EPSILON)
        c = max(cur_p, _PSI_EPSILON)
        total += (c - r) * math.log(c / r)
    return float(total)


def ks_statistic(reference: Any, current: Any) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov ``(statistic, p_value)``.

    The maximum gap between the two empirical CDFs and its asymptotic
    p-value, via SciPy.  A small p-value is evidence the two samples do
    not come from the same distribution.
    """
    from scipy.stats import ks_2samp

    ref = _as_1d(reference, "reference")
    cur = _as_1d(current, "current")
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


@dataclass(slots=True)
class FeatureDrift:
    """Per-feature drift readout."""

    name: str
    psi: float
    ks_statistic: float | None
    ks_pvalue: float | None
    reference_mean: float
    current_mean: float
    verdict: str  # "stable" | "moderate" | "shifted"

    @property
    def mean_direction(self) -> str:
        """``"up"`` / ``"down"`` / ``"flat"`` — which way the mean moved."""
        delta = self.current_mean - self.reference_mean
        if delta > 0:
            return "up"
        if delta < 0:
            return "down"
        return "flat"


@dataclass(slots=True)
class DriftReport:
    """Outcome of comparing a current batch against a reference.

    Attributes
    ----------
    features:
        One :class:`FeatureDrift` per compared feature, in the order the
        features were supplied.
    n_rows:
        Number of rows in the current batch.
    source:
        Where the reference came from — ``"frame"`` (raw training frame)
        or ``"bundle"`` (the quantile sketch a bundle recorded, PSI only).
    """

    features: tuple[FeatureDrift, ...]
    n_rows: int
    source: str
    warn_psi: float
    shift_psi: float

    @property
    def drifted(self) -> list[FeatureDrift]:
        """Features at or above the shift threshold, worst first."""
        return sorted(
            (f for f in self.features if f.verdict == "shifted"),
            key=lambda f: -f.psi,
        )

    @property
    def n_drifted(self) -> int:
        return len(self.drifted)

    @property
    def worst(self) -> FeatureDrift:
        """The feature with the highest PSI (undefined when no features)."""
        if not self.features:
            raise ValueError("no features were compared")
        return max(self.features, key=lambda f: f.psi)

    @property
    def drift_share(self) -> float:
        """Fraction of compared features flagged as shifted."""
        if not self.features:
            return 0.0
        return self.n_drifted / len(self.features)

    def to_frame(self) -> Any:
        """The per-feature table as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "feature": f.name,
                    "psi": f.psi,
                    "ks_statistic": f.ks_statistic,
                    "ks_pvalue": f.ks_pvalue,
                    "reference_mean": f.reference_mean,
                    "current_mean": f.current_mean,
                    "mean_direction": f.mean_direction,
                    "verdict": f.verdict,
                }
                for f in self.features
            ]
        )

    def __str__(self) -> str:
        if not self.features:
            return "drift: no numeric features compared"
        source = "the training frame" if self.source == "frame" else "the bundle's sketch"
        if self.n_drifted == 0:
            return (
                f"drift: no feature shifted vs {source} "
                f"({len(self.features)} compared, max PSI "
                f"{self.worst.psi:.3f})"
            )
        head = (
            f"drift: {self.n_drifted}/{len(self.features)} feature(s) shifted "
            f"vs {source} (PSI >= {self.shift_psi:g})"
        )
        lines = []
        for f in self.drifted[:8]:
            lines.append(
                f"{f.name} (PSI {f.psi:.3f}, mean {f.mean_direction})"
            )
        return head + "\n  " + "\n  ".join(lines)


def _numeric_features(
    reference: Any, features: list[str] | None
) -> list[str]:
    """Numeric columns to compare, honouring an explicit order."""
    import pandas as pd

    if not isinstance(reference, pd.DataFrame):
        return []
    columns = list(features) if features else list(reference.columns)
    out = []
    for name in columns:
        if name not in reference.columns:
            continue
        if getattr(reference[name].dtype, "kind", "") in "iufb":
            out.append(str(name))
    return out


def detect_drift(
    reference: Any,
    current: Any,
    *,
    features: list[str] | None = None,
    bins: int = 10,
    warn_psi: float = DEFAULT_WARN_PSI,
    shift_psi: float = DEFAULT_SHIFT_PSI,
    with_ks: bool = True,
) -> DriftReport:
    """Compare ``current`` against ``reference`` feature by feature.

    Parameters
    ----------
    reference:
        Training frame (a DataFrame with numeric feature columns).
    current:
        New batch; features absent from it are skipped, matching the
        applicability box's behaviour.
    features:
        Restrict the comparison to these columns (in this order);
        defaults to every shared numeric column.
    bins:
        PSI quantile bins over the reference.
    warn_psi, shift_psi:
        PSI thresholds for the ``"moderate"`` / ``"shifted"`` verdicts.
    with_ks:
        Also compute the two-sample KS statistic (needs the raw
        reference values; ignored by the metadata path).

    Returns
    -------
    DriftReport
        Per-feature PSI / KS and the overall verdict.
    """
    import pandas as pd

    if not isinstance(reference, pd.DataFrame):
        raise DataValidationError(
            "reference must be a DataFrame of training features, got "
            f"{type(reference).__name__}"
        )
    if not isinstance(current, pd.DataFrame):
        raise DataValidationError(
            f"current must be a DataFrame, got {type(current).__name__}"
        )
    names = _numeric_features(reference, features)
    feature_drifts: list[FeatureDrift] = []
    for name in names:
        if name not in current.columns:
            continue
        ref_col = pd.to_numeric(reference[name], errors="coerce").to_numpy(dtype=float)
        cur_col = pd.to_numeric(current[name], errors="coerce").to_numpy(dtype=float)
        ref_col = ref_col[np.isfinite(ref_col)]
        cur_col = cur_col[np.isfinite(cur_col)]
        if ref_col.size == 0 or cur_col.size == 0:
            continue
        score = psi(ref_col, cur_col, bins=bins)
        ks_stat = ks_p = None
        if with_ks:
            ks_stat, ks_p = ks_statistic(ref_col, cur_col)
        if score >= shift_psi:
            verdict = "shifted"
        elif score >= warn_psi:
            verdict = "moderate"
        else:
            verdict = "stable"
        feature_drifts.append(
            FeatureDrift(
                name=name,
                psi=score,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                reference_mean=float(ref_col.mean()),
                current_mean=float(cur_col.mean()),
                verdict=verdict,
            )
        )
    return DriftReport(
        features=tuple(feature_drifts),
        n_rows=len(current),
        source="frame",
        warn_psi=warn_psi,
        shift_psi=shift_psi,
    )


def feature_distributions_from_frame(
    frame: Any, features: list[str] | None = None, *, bins: int = 10
) -> dict[str, dict[str, Any]]:
    """Compact per-feature distribution sketch for a bundle's metadata.

    For each numeric feature records the quantile bin edges, the
    reference bin proportions and the mean/std — enough to compute a PSI
    against a later batch without shipping the training rows.  Non-numeric
    columns are skipped (one-hot encoded inside the pipeline, so their raw
    distribution is not meaningful).
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(
            f"frame must be a DataFrame, got {type(frame).__name__}"
        )
    names = _numeric_features(frame, features)
    sketch: dict[str, dict[str, Any]] = {}
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        edges = _quantile_edges(values, bins)
        props = _proportions(values, edges)
        sketch[name] = {
            "edges": [float(e) for e in edges],
            "proportions": [float(p) for p in props],
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return sketch


def drift_from_metadata(
    metadata: dict[str, Any],
    current: Any,
    *,
    features: list[str] | None = None,
    warn_psi: float = DEFAULT_WARN_PSI,
    shift_psi: float = DEFAULT_SHIFT_PSI,
) -> DriftReport | None:
    """PSI drift of ``current`` against a bundle's recorded sketch.

    Returns ``None`` when the bundle predates sketch recording (no
    ``feature_distributions`` in its metadata) — the same honesty the
    applicability box keeps: an absent check is never a pass.  KS is not
    available here (the sketch holds bins, not raw values), so
    ``ks_statistic`` / ``ks_pvalue`` are ``None``.
    """
    import pandas as pd

    sketch = metadata.get("feature_distributions")
    if not isinstance(sketch, dict) or not sketch:
        return None
    if not isinstance(current, pd.DataFrame):
        raise DataValidationError(
            f"current must be a DataFrame, got {type(current).__name__}"
        )
    names = list(features) if features else list(sketch)
    feature_drifts: list[FeatureDrift] = []
    for name in names:
        entry = sketch.get(name)
        if not isinstance(entry, dict) or name not in current.columns:
            continue
        edges = np.asarray(entry.get("edges", []), dtype=float)
        ref_props = np.asarray(entry.get("proportions", []), dtype=float)
        if edges.size < 2 or ref_props.size != edges.size - 1:
            continue
        cur = pd.to_numeric(current[name], errors="coerce").to_numpy(dtype=float)
        cur = cur[np.isfinite(cur)]
        if cur.size == 0:
            continue
        cur_props = _proportions(cur, edges)
        score = 0.0
        for ref_p, cur_p in zip(ref_props, cur_props, strict=True):
            r = max(float(ref_p), _PSI_EPSILON)
            c = max(float(cur_p), _PSI_EPSILON)
            score += (c - r) * math.log(c / r)
        if score >= shift_psi:
            verdict = "shifted"
        elif score >= warn_psi:
            verdict = "moderate"
        else:
            verdict = "stable"
        feature_drifts.append(
            FeatureDrift(
                name=str(name),
                psi=float(score),
                ks_statistic=None,
                ks_pvalue=None,
                reference_mean=float(entry.get("mean", float("nan"))),
                current_mean=float(cur.mean()),
                verdict=verdict,
            )
        )
    return DriftReport(
        features=tuple(feature_drifts),
        n_rows=len(current),
        source="bundle",
        warn_psi=warn_psi,
        shift_psi=shift_psi,
    )

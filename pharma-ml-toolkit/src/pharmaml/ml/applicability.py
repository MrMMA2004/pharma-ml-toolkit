"""Applicability domain analysis — is a prediction trustworthy?

The user's uploaded ``Applicability_domain_analysis.py`` (2026-10) asked
the two questions every QSAR readout deserves before anyone acts on it:

1. **Distance applicability domain** — is the query compound inside the
   training descriptor space at all?  Measured as the mean k-nearest-
   neighbour distance in the standardised descriptor space, thresholded
   at a percentile of the training set's own distances (the script's
   ``k = 5`` / 95th percentile convention).
2. **Prediction uncertainty** — how much does the model's answer wobble
   when its own training set is bootstrap-resampled?  The script
   re-fitted 30 fresh ``CatBoostRegressor`` models (30 more again for
   the training-side threshold — 60 fits total) and took the ensemble's
   per-compound standard deviation, thresholded at the 95th percentile
   of the training uncertainty.

This module ships both halves as composable, estimator-agnostic
functions with three deliberate engineering upgrades over the upload:

* **any estimator, not just CatBoost** — the bootstrap ensemble is
  ``clone(model)`` of whatever fitted pipeline the caller hands in (the
  toolkit's own trained model, a loaded persistence bundle, ...), so a
  ridge, a random forest or a stacking ensemble all work unchanged;
* **half the fits** — every bootstrap model predicts on the test set
  AND the training set in one pass, so the training-side uncertainty
  threshold costs zero extra fits (the upload ran 2×30 CatBoost fits);
* **deterministic** — ``resample`` is seeded through ``random_state``
  (the upload sampled unseeded: two runs of the "same" analysis
  produced different reliability verdicts).

A fourth upgrade is behavioural, not structural: the training kNN
distances **exclude each point's own zero self-distance** (the upload
queried the fitted neighbour index with the training matrix itself,
which always counts the point as its own first neighbour and deflates
the mean by roughly ``distance/k`` — every threshold was slightly too
small, every domain slightly too tight).

A third uncertainty readout joins the two domain questions in
100x-151: :func:`quantile_prediction_intervals` — the ``D2.py``
upload's prediction-interval block, ported honestly (its shipped code
fit three IDENTICAL plain forests, so the advertised 90% interval had
width zero; this module fits the quantile-regression trio the upload
meant).  Where the bootstrap spread asks how much the *model* wobbles
under resampled training data, the quantile bounds bracket where the
*residual noise* puts the true value.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from sklearn.linear_model import Ridge
>>> x = pd.DataFrame({"a": np.linspace(0.0, 10.0, 60)})
>>> y = pd.Series(2.0 * x["a"] + 1.0)
>>> model = Ridge().fit(x, y)
>>> inside = applicability_domain(model, x, y, x.iloc[::4], k=5, n_models=8)
>>> bool(inside.inside_ad.all())           # interpolation stays inside
True
>>> far = pd.DataFrame({"a": [1000.0]})    # wild extrapolation
>>> out = applicability_domain(model, x, y, far, k=5, n_models=4)
>>> bool(out.inside_ad[0]), bool(out.reliable[0])
(False, False)
>>> out.distance_threshold > 0.0 and out.uncertainty_threshold >= 0.0
True
>>> frame = out.to_frame(names=["far-away compound"])
>>> list(frame.columns)                    # the upload's reliability table
['name', 'predicted', 'prediction_sd', 'knn_distance', 'inside_ad', 'low_uncertainty', 'reliable']
>>> float(frame.loc[0, "knn_distance"]) > float(out.distance_threshold)
True

Descriptor alignment: when both inputs are DataFrames, their unique column
labels must match exactly in the same order; mismatches are rejected rather
than silently reordering. When either input is an array, columns are matched
by position and the caller must ensure their meanings and order agree. This
check does not verify descriptor units, preprocessing or provenance.

The kNN half alone (no bootstrap fits) is the quick pass:

>>> quick = knn_applicability_domain(x, far, k=5)
>>> bool(quick.inside_ad[0])
False
>>> quick.pred_mean is None and quick.reliable is None
True
>>> bool(quick.to_frame(names=["far-away compound"])["reliable"].isna().all())
True

Classification models get their honest path too — the ensemble spread
of the positive-class probability:

>>> from sklearn.linear_model import LogisticRegression
>>> xc = pd.DataFrame({"a": [0.0, 0.1, 0.2, 0.3, 9.7, 9.8, 9.9, 10.0]})
>>> yc = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
>>> clf = LogisticRegression().fit(xc, yc)
>>> res = applicability_domain(clf, xc, yc, pd.DataFrame({"a": [0.15, 5.0]}),
...                            k=3, n_models=6)
>>> res.uncertainty_kind
'positive-class probability'
>>> bool(res.inside_ad[0]), bool(res.inside_ad[1])   # mid-gap is far from both clusters
(True, False)

Error contracts (the house style):

>>> knn_applicability_domain(x, far, k=0)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: k must be >= 1, got 0
>>> knn_applicability_domain(x, far, k=60)            # more neighbours than training rows
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: k=60 needs at least 61 training rows to exclude each point's own self-distance, got 60
>>> knn_applicability_domain(x, far, percentile=101.0)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: percentile must be in (0, 100], got 101.0
>>> two = pd.DataFrame({"a": np.linspace(0.0, 1.0, 40),
...                     "b": np.linspace(1.0, 2.0, 40)})
>>> knn_applicability_domain(two, far)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: X_test has 1 feature(s) but X_train has 2 — the descriptor spaces must match
>>> bad = pd.DataFrame({"a": [0.0, np.nan]})
>>> knn_applicability_domain(bad, far)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: X_train contains 1 NaN value(s) — applicability domain needs finite descriptors
>>> applicability_domain(model, x, y, far, n_models=0)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: n_models must be >= 1, got 0 — pass uncertainty=False for a kNN-only pass
>>> xm = pd.DataFrame({"a": [0.0, 5.0, 10.0] * 4})
>>> ym = pd.Series([0, 1, 2] * 4)
>>> three = LogisticRegression().fit(xm, ym)          # multiclass proba: no honest std
>>> applicability_domain(three, xm, ym, pd.DataFrame({"a": [1.0]}), k=3)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: bootstrap_uncertainty needs numeric predictions or a binary predict_proba — this estimator exposes 3 classes
>>> applicability_domain("not a model", x, y, far)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: model must be a fitted estimator with predict(), got str
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "ApplicabilityDomainResult",
    "BootstrapUncertainty",
    "QuantileIntervals",
    "applicability_domain",
    "bootstrap_uncertainty",
    "knn_applicability_domain",
    "quantile_prediction_intervals",
]

#: Progress callback contract: ``report(done, total)`` after each
#: bootstrap fit (``done`` counts fitted ensemble members).
ProgressCallback = Callable[[int, int], None]

_UNCERTAINTY_SURFACE = (
    "bootstrap_uncertainty needs numeric predictions or a binary "
    "predict_proba"
)


def _numeric_matrix(X: pd.DataFrame | np.ndarray, name: str) -> np.ndarray:
    """Coerce a feature frame/array to a finite float matrix.

    Raises
    ------
    DataValidationError
        When ``X`` carries non-numeric columns (DataFrame) or values
        that refuse to float, or contains NaN/±inf entries — the
        applicability domain needs finite descriptors, and a NaN
        distance would silently poison every threshold downstream.
    """
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise DataValidationError(f"{name} has duplicate feature names; columns must be unique")
        bad = [str(col) for col in X.columns
               if X[col].dtype.kind not in "fiub"]
        if bad:
            raise DataValidationError(
                f"{name} carries non-numeric column(s) {', '.join(bad)} — "
                "compute descriptors / clean the frame first "
                "(the ML tab's numeric cleanup)"
            )
        try:
            matrix = X.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{name} could not be converted to a numeric matrix: {exc}"
            ) from exc
    else:
        try:
            matrix = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{name} could not be converted to a numeric matrix: {exc}"
            ) from exc
    if matrix.ndim != 2:
        raise DataValidationError(
            f"{name} must be a 2-D feature matrix, got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise DataValidationError(f"{name} is empty — nothing to analyse")
    if matrix.shape[1] == 0:
        raise DataValidationError(f"{name} has zero feature columns")
    n_nan = int(np.isnan(matrix).sum())
    if n_nan:
        raise DataValidationError(
            f"{name} contains {n_nan} NaN value(s) — applicability domain "
            "needs finite descriptors"
        )
    n_inf = int(np.isinf(matrix).sum())
    if n_inf:
        raise DataValidationError(
            f"{name} contains {n_inf} infinite value(s) — applicability "
            "domain needs finite descriptors"
        )
    return matrix


def _validate_descriptor_schema(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> None:
    """Require matching dimensions and, when available, matching column order."""
    if train.shape[1] != test.shape[1]:
        raise ModelError(
            f"X_test has {test.shape[1]} feature(s) but X_train has "
            f"{train.shape[1]} — the descriptor spaces must match"
        )
    if (
        isinstance(X_train, pd.DataFrame)
        and isinstance(X_test, pd.DataFrame)
        and not X_train.columns.equals(X_test.columns)
    ):
        raise ModelError(
            "X_train and X_test feature names and order must match exactly; "
            "align query columns to the training descriptors"
        )


def _feature_names(X: pd.DataFrame | np.ndarray) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return [str(col) for col in X.columns]
    return [f"f{i}" for i in range(np.asarray(X).shape[1])]


@dataclass(slots=True)
class ApplicabilityDomainResult:
    """Outcome of a full applicability-domain analysis.

    Attributes
    ----------
    k:
        Neighbour count of the distance domain.
    percentile:
        The percentile both thresholds are taken at.
    distance_threshold:
        Training-set kNN-distance cut-off (in standardised units).
    train_knn_mean / test_knn_mean:
        Per-row mean kNN distance of the training / query rows
        (training distances exclude each point's own self-match).
    inside_ad:
        Boolean mask — query rows inside the distance domain.
    feature_names:
        The descriptor columns the domain was measured in.
    n_models:
        Bootstrap ensemble size (``None`` when skipped).
    pred_mean / pred_std:
        Ensemble mean / per-row standard deviation on the query rows.
    train_pred_std:
        Per-row ensemble std on the TRAINING rows — the yardstick the
        uncertainty threshold is cut from.
    uncertainty_threshold:
        Percentile of ``train_pred_std`` (``None`` when skipped).
    low_uncertainty:
        Boolean mask — query rows under the uncertainty threshold.
    reliable:
        ``inside_ad & low_uncertainty`` — the upload's final verdict
        (``None`` when the bootstrap half was skipped).
    uncertainty_kind:
        ``'predictions'`` or ``'positive-class probability'``.
    fit_seconds:
        Wall-clock cost of the bootstrap fits (0.0 when skipped).
    """

    k: int
    percentile: float
    distance_threshold: float
    train_knn_mean: np.ndarray
    test_knn_mean: np.ndarray
    inside_ad: np.ndarray
    feature_names: list[str]
    n_models: int | None = None
    pred_mean: np.ndarray | None = None
    pred_std: np.ndarray | None = None
    train_pred_std: np.ndarray | None = None
    uncertainty_threshold: float | None = None
    low_uncertainty: np.ndarray | None = None
    reliable: np.ndarray | None = None
    uncertainty_kind: str | None = None
    fit_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat dict summary (no per-row arrays) for logs and reports."""
        return {
            "k": self.k,
            "percentile": self.percentile,
            "distance_threshold": self.distance_threshold,
            "n_train": int(self.train_knn_mean.size),
            "n_test": int(self.test_knn_mean.size),
            "inside_ad": int(self.inside_ad.sum()),
            "n_models": self.n_models,
            "uncertainty_threshold": self.uncertainty_threshold,
            "low_uncertainty": None if self.low_uncertainty is None
            else int(self.low_uncertainty.sum()),
            "reliable": None if self.reliable is None else int(self.reliable.sum()),
            "uncertainty_kind": self.uncertainty_kind,
            "fit_seconds": self.fit_seconds,
        }

    def to_frame(self, names: list[str] | pd.Series | None = None) -> pd.DataFrame:
        """The upload's reliability table, one row per query compound.

        Columns: ``name, predicted, prediction_sd, knn_distance,
        inside_ad, low_uncertainty, reliable`` — the uncertainty
        columns carry ``NaN``/``NA`` when the bootstrap half was
        skipped (the verdict honestly degrades to distance-only).
        """
        n = int(self.test_knn_mean.size)
        if names is None:
            labels = [f"compound_{i}" for i in range(n)]
        else:
            labels = [str(v) for v in list(names)[:n]]
            labels += [f"compound_{i}" for i in range(len(labels), n)]
        data: dict[str, Any] = {
            "name": labels,
            "predicted": self.pred_mean if self.pred_mean is not None
            else np.full(n, np.nan),
            "prediction_sd": self.pred_std if self.pred_std is not None
            else np.full(n, np.nan),
            "knn_distance": self.test_knn_mean,
            "inside_ad": self.inside_ad,
        }
        if self.low_uncertainty is not None and self.reliable is not None:
            data["low_uncertainty"] = self.low_uncertainty
            data["reliable"] = self.reliable
        else:
            data["low_uncertainty"] = pd.array([pd.NA] * n, dtype="boolean")
            data["reliable"] = pd.array([pd.NA] * n, dtype="boolean")
        return pd.DataFrame(data)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        inside = int(self.inside_ad.sum())
        n = int(self.inside_ad.size)
        text = (f"k={self.k} distance_threshold={self.distance_threshold:.4g} "
                f"inside_ad={inside}/{n}")
        if self.reliable is not None:
            text += (f" uncertainty_threshold={self.uncertainty_threshold:.4g} "
                     f"reliable={int(self.reliable.sum())}/{n}")
        return text


@dataclass(slots=True)
class BootstrapUncertainty:
    """Bootstrap-ensemble uncertainty of a fitted estimator.

    Attributes
    ----------
    pred_mean / pred_std:
        Ensemble mean / per-row std on the query rows.
    train_pred_std:
        Per-row std on the training rows (the threshold yardstick).
    n_models:
        Ensemble size actually fitted.
    kind:
        ``'predictions'`` (numeric regressor output) or
        ``'positive-class probability'`` (binary classifier proba).
    fit_seconds:
        Wall-clock cost of the whole ensemble.
    """

    pred_mean: np.ndarray
    pred_std: np.ndarray
    train_pred_std: np.ndarray
    n_models: int
    kind: str
    fit_seconds: float = 0.0


def knn_applicability_domain(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    *,
    k: int = 5,
    percentile: float = 95.0,
    metric: str = "euclidean",
) -> ApplicabilityDomainResult:
    """kNN-distance applicability domain (the upload's step 4-5).

    The descriptor space is standardised (``StandardScaler`` fitted on
    the training matrix — the upload's own convention), a
    ``NearestNeighbors`` index with ``k`` neighbours is fitted on it,
    and a row counts as **inside the domain** when its mean neighbour
    distance stays under the ``percentile``-th percentile of the
    training rows' own mean distances.

    Training distances exclude each point's own self-match: the
    training matrix is queried with ``k + 1`` neighbours and the zero
    first column dropped (see the module docstring for why the
    upload's thresholds were slightly too tight).

    Parameters
    ----------
    X_train:
        Training descriptor matrix (numeric, finite, unique DataFrame columns).
    X_test:
        Query descriptor matrix — same columns in the same order as
        ``X_train``. If either input is an array, alignment is positional.
    k:
        Neighbour count (default 5, the upload's value).
    percentile:
        Distance cut-off percentile in ``(0, 100]`` (default 95).
    metric:
        NearestNeighbors metric (default ``'euclidean'``, the upload's).

    Returns
    -------
    ApplicabilityDomainResult
        The distance half only — ``n_models`` is ``None`` and the
        uncertainty fields stay unset.

    Raises
    ------
    ModelError
        On invalid ``k`` / ``percentile``, a feature-count mismatch,
        or too few training rows for the self-exclusion query.
    DataValidationError
        On non-numeric / non-finite descriptor matrices.
    """
    train = _numeric_matrix(X_train, "X_train")
    test = _numeric_matrix(X_test, "X_test")
    if k < 1:
        raise ModelError(f"k must be >= 1, got {k}")
    if k > train.shape[0] - 1:
        raise ModelError(
            f"k={k} needs at least {k + 1} training rows to exclude each "
            f"point's own self-distance, got {train.shape[0]}"
        )
    if not 0.0 < percentile <= 100.0:
        raise ModelError(f"percentile must be in (0, 100], got {percentile}")
    _validate_descriptor_schema(X_train, X_test, train, test)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(train_scaled)

    # training distances WITHOUT the self-match: query k+1, drop col 0
    train_distances, _ = nbrs.kneighbors(train_scaled, n_neighbors=k + 1)
    train_knn_mean = train_distances[:, 1:].mean(axis=1)

    threshold = float(np.percentile(train_knn_mean, percentile))
    test_distances, _ = nbrs.kneighbors(test_scaled)
    test_knn_mean = test_distances.mean(axis=1)
    inside = test_knn_mean <= threshold
    _LOG.debug(
        "kNN AD: k=%d threshold=%.4g inside=%d/%d",
        k, threshold, int(inside.sum()), int(inside.size),
    )
    return ApplicabilityDomainResult(
        k=k,
        percentile=percentile,
        distance_threshold=threshold,
        train_knn_mean=train_knn_mean,
        test_knn_mean=test_knn_mean,
        inside_ad=inside,
        feature_names=_feature_names(X_train),
    )


def _probe_prediction_kind(model: Any, X_train: pd.DataFrame | np.ndarray) -> str:
    """Classify the estimator's prediction surface for bootstrapping.

    Returns ``'predictions'`` for float-valued ``predict()`` output
    (regressors), ``'positive-class probability'`` for binary
    classifiers with ``predict_proba``, and raises otherwise — integer
    or string class labels have no honest per-row standard deviation.
    """
    if not callable(getattr(model, "predict", None)):
        raise ModelError(
            "model must be a fitted estimator with predict(), got "
            f"{type(model).__name__}"
        )
    probe = np.asarray(model.predict(X_train[:1])).reshape(-1)
    if probe.dtype.kind == "f":
        return "predictions"
    proba = getattr(model, "predict_proba", None)
    if callable(proba):
        width = np.asarray(proba(X_train[:1])).shape[1]
        if width == 2:
            return "positive-class probability"
        raise ModelError(
            f"{_UNCERTAINTY_SURFACE} — this estimator exposes {width} classes"
        )
    raise ModelError(
        f"{_UNCERTAINTY_SURFACE} — this estimator predicts class labels"
    )


def bootstrap_uncertainty(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray | list,
    X_test: pd.DataFrame | np.ndarray,
    *,
    n_models: int = 30,
    random_state: int = 0,
    progress: ProgressCallback | None = None,
) -> BootstrapUncertainty:
    """Bootstrap-ensemble prediction uncertainty (the upload's step 6).

    ``n_models`` clones of ``model`` are fitted on bootstrap resamples
    of the training data and predict on BOTH the query rows and the
    training rows — the per-row standard deviation across the ensemble
    is the uncertainty readout, and the training-side std comes from
    the same fits (the upload spent a second, separate 30-model round
    on it).

    Deterministic: resample ``i`` is seeded with ``random_state + i``
    (the upload sampled unseeded).

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator — pipelines included.
    X_train, y_train:
        The training data ``model`` was fitted on.
    X_test:
        Query rows for the uncertainty readout. DataFrame columns must match
        training names and order; arrays are matched by position.
    n_models:
        Ensemble size (default 30, the upload's value).
    random_state:
        Seed of the first resample.
    progress:
        Optional ``report(done, total)`` callback fired after each fit
        (a broken reporter is logged and swallowed — it must never be
        able to kill a running analysis).

    Returns
    -------
    BootstrapUncertainty

    Raises
    ------
    ModelError
        When ``n_models < 1``, the row counts disagree, or the
        estimator's prediction surface has no honest per-row std.
    DataValidationError
        On non-numeric / non-finite descriptor matrices.
    """
    if n_models < 1:
        raise ModelError(
            f"n_models must be >= 1, got {n_models} — pass uncertainty=False "
            "for a kNN-only pass"
        )
    train = _numeric_matrix(X_train, "X_train")
    test = _numeric_matrix(X_test, "X_test")
    _validate_descriptor_schema(X_train, X_test, train, test)
    y = np.asarray(y_train)
    if y.shape[0] != train.shape[0]:
        raise ModelError(
            f"y_train has {y.shape[0]} value(s) but X_train has "
            f"{train.shape[0]} rows"
        )
    kind = _probe_prediction_kind(model, X_train)

    started = time.perf_counter()
    test_preds: list[np.ndarray] = []
    train_preds: list[np.ndarray] = []
    for index in range(n_models):
        X_boot, y_boot = resample(X_train, y_train,
                                  random_state=random_state + index)
        member = clone(model)
        member.fit(X_boot, y_boot)
        if kind == "predictions":
            test_preds.append(
                np.asarray(member.predict(X_test), dtype=float).reshape(-1))
            train_preds.append(
                np.asarray(member.predict(X_train), dtype=float).reshape(-1))
        else:
            test_preds.append(
                np.asarray(member.predict_proba(X_test), dtype=float)[:, 1])
            train_preds.append(
                np.asarray(member.predict_proba(X_train), dtype=float)[:, 1])
        if progress is not None:
            try:
                progress(index + 1, n_models)
            except Exception:  # noqa: BLE001 — a broken reporter must not kill the run
                _LOG.debug("progress callback failed", exc_info=True)
    fit_seconds = time.perf_counter() - started

    test_matrix = np.vstack(test_preds)
    train_matrix = np.vstack(train_preds)
    _LOG.debug(
        "bootstrap uncertainty: %d member(s) in %.2fs (%s)",
        n_models, fit_seconds, kind,
    )
    return BootstrapUncertainty(
        pred_mean=test_matrix.mean(axis=0),
        pred_std=test_matrix.std(axis=0),
        train_pred_std=train_matrix.std(axis=0),
        n_models=n_models,
        kind=kind,
        fit_seconds=fit_seconds,
    )


@dataclass(slots=True)
class QuantileIntervals:
    """Prediction intervals from gradient-boosted quantile regression.

    The third uncertainty readout (the ``D2.py`` upload's
    ``generate_uncertainty_quantification``, ported honestly in
    100x-151): two quantile-regression fits bracket every prediction
    with a central-``1 - alpha`` interval, and the median fit carries
    the point estimate.

    Attributes
    ----------
    alpha:
        The interval's tail mass — the bounds target the
        ``alpha/2`` and ``1 - alpha/2`` quantiles (default 0.1, the
        upload's 90% interval).
    lower / median / upper:
        Per-row interval bound / median prediction / upper bound on
        the query rows.  The bounds are the CROSSING-REPAIRED pair:
        separately-fit quantile models can cross on individual rows
        (``lower > upper``), every crossing is counted in
        ``n_crossing`` and the exported pair is swapped so
        ``lower <= upper`` always holds.
    coverage:
        Share of query rows whose true value falls inside the interval
        (``None`` when no ``y_test`` was supplied).
    mean_width:
        Mean of ``upper - lower`` across the query rows.
    n_test:
        Number of query rows.
    n_covered:
        Rows inside the interval (``None`` without ``y_test``).
    n_crossing:
        Rows where the raw quantile pair crossed (swapped in the
        exported bounds — quantile crossing is a known pathology of
        independently fit quantile models, reported not hidden).
    fit_seconds:
        Wall-clock cost of the three fits.
    """

    alpha: float
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    n_test: int
    mean_width: float
    n_crossing: int
    coverage: float | None = None
    n_covered: int | None = None
    fit_seconds: float = 0.0

    def as_frame(self, y_test: object = None) -> pd.DataFrame:
        """The upload's interval table, one row per query compound.

        Columns: ``actual, pred_median, pred_lower, pred_upper,
        interval_width`` — the ``actual`` column appears only when the
        true values are supplied (mirrors the coverage being optional).
        """
        data: dict[str, Any] = {}
        if y_test is not None:
            data["actual"] = np.asarray(y_test, dtype=float)
        data["pred_median"] = self.median
        data["pred_lower"] = self.lower
        data["pred_upper"] = self.upper
        data["interval_width"] = self.upper - self.lower
        return pd.DataFrame(data)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        text = (f"alpha={self.alpha:g} n_test={self.n_test} "
                f"mean_width={self.mean_width:.4g} "
                f"crossing={self.n_crossing}")
        if self.coverage is not None:
            text += f" coverage={self.coverage:.1%}"
        return text


def quantile_prediction_intervals(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray | list,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray | list | None = None,
    *,
    alpha: float = 0.1,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int | None = 3,
    subsample: float = 1.0,
    random_state: int = 42,
    progress: ProgressCallback | None = None,
) -> QuantileIntervals:
    """Central-``1 - alpha`` prediction intervals via quantile regression.

    The honest port of ``D2.py``'s uncertainty block, whose shipped
    code was structurally broken: it fit THREE plain
    ``RandomForestRegressor`` instances on the same data with no
    quantile loss, so the three forests were bit-for-bit identical —
    interval width 0.0 on every row, empirical coverage 0.0%, and the
    "90% Prediction Interval" the plot legend advertised did not exist
    (``RandomForestRegressor`` has no quantile mode in scikit-learn at
    all — probed on 1.5.2).  This function fits what the upload
    *meant*: ``GradientBoostingRegressor(loss="quantile")`` at the
    ``alpha/2`` / ``0.5`` / ``1 - alpha/2`` quantiles — the classic
    scikit-learn quantile-regression route (probed: a 5/50/95% trio
    returns ~90% empirical coverage on noisy synthetic data, with
    non-degenerate widths and correct ordering).

    The ensemble-std sibling of this readout is
    :func:`bootstrap_uncertainty`; the two answer different questions —
    the bootstrap spread measures how much the *model* wobbles under
    resampled training data, the quantile bounds bracket where the
    *residual noise* puts the true value.

    Deterministic by default: with ``subsample=1.0`` (the default)
    the gradient-boosting fits carry no stochastic element at all, so
    two runs agree bit-for-bit whatever the seed; a stochastic
    configuration (``subsample < 1.0``) is seeded through
    ``random_state`` (the upload's forests were seeded but
    identical-by-construction, a stronger and less useful guarantee).

    Parameters
    ----------
    X_train, y_train:
        Training descriptors and target (finite; ``y_train`` may not
        carry NaN — a quantile loss has no honest answer for a missing
        target).
    X_test:
        Query rows — DataFrame columns must match training names and order;
        arrays are matched by position.
    y_test:
        Optional true values for the coverage statistic.
    alpha:
        Tail mass (default 0.1 — the upload's 90% interval); must lie
        in ``(0, 0.5)``.
    n_estimators / learning_rate / max_depth / subsample:
        The gradient-boosting knobs shared by all three quantile fits
        (``subsample < 1.0`` turns on stochastic gradient boosting —
        seeded through ``random_state``).
    random_state:
        Seed of every fit (the knob behind the stochastic
        configuration's determinism).
    progress:
        Optional ``report(done, total)`` fired after each of the three
        fits (a broken reporter is logged and swallowed — it must
        never be able to kill a running analysis).

    Returns
    -------
    QuantileIntervals

    Raises
    ------
    ModelError
        On an out-of-range ``alpha``, non-positive ``n_estimators`` /
        ``learning_rate``, or disagreeing row / feature counts.
    DataValidationError
        On non-numeric / non-finite descriptor matrices or targets.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> x = pd.DataFrame({"a": rng.uniform(0.0, 10.0, 80)})
    >>> y = pd.Series(2.0 * x["a"] + 1.0 + rng.normal(0.0, 0.5, 80))
    >>> res = quantile_prediction_intervals(
    ...     x, y, x.iloc[::4], y.iloc[::4], alpha=0.2, n_estimators=60)
    >>> bool((res.lower <= res.upper).all())   # bounds never cross
    True
    >>> 0.0 <= res.coverage <= 1.0
    True
    >>> res.n_test, res.mean_width > 0.0
    (20, True)
    >>> frame = res.as_frame(y.iloc[::4])
    >>> list(frame.columns)
    ['actual', 'pred_median', 'pred_lower', 'pred_upper', 'interval_width']
    >>> no_truth = quantile_prediction_intervals(x, y, x.iloc[::4],
    ...     alpha=0.2, n_estimators=60)
    >>> no_truth.coverage is None and no_truth.n_covered is None
    True

    Error contracts (the house style):

    >>> quantile_prediction_intervals(x, y, x.iloc[:2], alpha=0.5)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: alpha must be in (0, 0.5) -- 0.5 would collapse the interval onto the median, got 0.5
    >>> quantile_prediction_intervals(x, y, x.iloc[:2], n_estimators=0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: n_estimators must be >= 1, got 0
    >>> x2 = pd.DataFrame({"a": x["a"], "b": 10.0 - x["a"]})
    >>> quantile_prediction_intervals(x2, y, pd.DataFrame({"b": [1.0]}))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: X_test has 1 feature(s) but X_train has 2 — the descriptor spaces must match
    >>> quantile_prediction_intervals(x, y, x.iloc[:2], y.iloc[:3])
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: y_test has 3 value(s) but X_test has 2 rows
    """
    if not 0.0 < alpha < 0.5:
        raise ModelError(
            f"alpha must be in (0, 0.5) -- 0.5 would collapse the "
            f"interval onto the median, got {alpha!r}"
        )
    if n_estimators < 1:
        raise ModelError(f"n_estimators must be >= 1, got {n_estimators}")
    if not (learning_rate > 0.0 and np.isfinite(learning_rate)):
        raise ModelError(
            f"learning_rate must be a positive finite number, got "
            f"{learning_rate!r}"
        )
    if not 0.0 < subsample <= 1.0:
        raise ModelError(
            f"subsample must be in (0, 1], got {subsample!r}"
        )
    train = _numeric_matrix(X_train, "X_train")
    test = _numeric_matrix(X_test, "X_test")
    _validate_descriptor_schema(X_train, X_test, train, test)
    y = _numeric_vector(y_train, "y_train", finite=True)
    if y.shape[0] != train.shape[0]:
        raise ModelError(
            f"y_train has {y.shape[0]} value(s) but X_train has "
            f"{train.shape[0]} rows"
        )
    truth: np.ndarray | None = None
    if y_test is not None:
        truth = _numeric_vector(y_test, "y_test", finite=True)
        if truth.shape[0] != test.shape[0]:
            raise ModelError(
                f"y_test has {truth.shape[0]} value(s) but X_test has "
                f"{test.shape[0]} rows"
            )

    started = time.perf_counter()
    lower = median = upper = np.zeros(0)
    for index, quantile in enumerate((alpha / 2.0, 0.5, 1.0 - alpha / 2.0)):
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )
        model.fit(train, y)
        prediction = np.asarray(model.predict(test), dtype=float).reshape(-1)
        if index == 0:
            lower = prediction
        elif index == 1:
            median = prediction
        else:
            upper = prediction
        if progress is not None:
            try:
                progress(index + 1, 3)
            except Exception:  # noqa: BLE001 — a broken reporter must not kill the run
                _LOG.debug("progress callback failed", exc_info=True)
    fit_seconds = time.perf_counter() - started

    # Quantile crossing (a known pathology of separately fit quantile
    # models): count it, then swap the crossed pairs so the exported
    # bounds always satisfy lower <= upper.  minimum/maximum (not a
    # sequential pair of np.where swaps) so the repair can never
    # reference its own output.
    crossed = lower > upper
    n_crossing = int(crossed.sum())
    if n_crossing:
        _LOG.debug(
            "quantile crossing on %d/%d row(s) -- bounds swapped",
            n_crossing, int(lower.size),
        )
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    coverage: float | None = None
    n_covered: int | None = None
    if truth is not None:
        inside = (truth >= lower) & (truth <= upper)
        n_covered = int(inside.sum())
        coverage = float(inside.mean())
    mean_width = float((upper - lower).mean())
    _LOG.debug(
        "quantile intervals: alpha=%g n_test=%d coverage=%s width=%.4g "
        "crossing=%d in %.2fs",
        alpha, int(lower.size),
        "n/a" if coverage is None else f"{coverage:.1%}",
        mean_width, n_crossing, fit_seconds,
    )
    return QuantileIntervals(
        alpha=float(alpha),
        lower=lower,
        median=median,
        upper=upper,
        n_test=int(lower.size),
        mean_width=mean_width,
        n_crossing=n_crossing,
        coverage=coverage,
        n_covered=n_covered,
        fit_seconds=fit_seconds,
    )


def _numeric_vector(
    values: object, name: str, *, finite: bool
) -> np.ndarray:
    """Coerce a target sequence to a flat float array (house guards).

    ``ValueError``/``TypeError`` from the coercion become
    :class:`DataValidationError`; when ``finite`` is set the result may
    not carry NaN either (a quantile loss has no honest answer for a
    missing target).
    """
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            f"{name} must be numeric, got non-numeric value(s): {error}"
        ) from error
    if finite and array.size and not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise DataValidationError(
            f"{name} contains {bad} NaN/inf value(s) -- quantile "
            "regression needs finite targets"
        )
    return array


def applicability_domain(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray | list,
    X_test: pd.DataFrame | np.ndarray,
    *,
    k: int = 5,
    percentile: float = 95.0,
    n_models: int = 30,
    random_state: int = 0,
    uncertainty: bool = True,
    progress: ProgressCallback | None = None,
) -> ApplicabilityDomainResult:
    """The uploaded workflow end-to-end, one call.

    Composes :func:`knn_applicability_domain` (steps 4-5 of the upload:
    standardised descriptor space, mean kNN distance, percentile
    threshold) with :func:`bootstrap_uncertainty` (step 6: ensemble
    spread over bootstrap resamples, its own percentile threshold on
    the training-side std), then folds both into the final reliability
    verdict per query row (step 7 of the upload):
    ``reliable = inside_ad & low_uncertainty``.

    Parameters
    ----------
    model:
        A fitted estimator — the toolkit's own ``train_model`` result,
        a persistence bundle's model, or any sklearn-compatible fit.
    X_train, y_train:
        The training data (descriptors + target).
    X_test:
        The query compounds, same descriptor columns.
    k, percentile:
        Distance-domain knobs (defaults 5 / 95 — the upload's).
    n_models:
        Bootstrap ensemble size (default 30 — the upload's); pass
        ``uncertainty=False`` to skip the bootstrap half entirely.
    random_state:
        Bootstrap seeding (determinism the upload lacked).
    uncertainty:
        Run the bootstrap half at all.
    progress:
        Optional ``report(done, total)`` fired per bootstrap fit.

    Returns
    -------
    ApplicabilityDomainResult
        With the reliability table available as ``result.to_frame()``.

    Raises
    ------
    ModelError
        From either half — invalid knobs, feature mismatches, or an
        estimator without an honest uncertainty surface.
    DataValidationError
        On non-numeric / non-finite descriptor matrices.
    """
    result = knn_applicability_domain(X_train, X_test, k=k, percentile=percentile)
    if not uncertainty:
        return result
    boot = bootstrap_uncertainty(
        model, X_train, y_train, X_test,
        n_models=n_models, random_state=random_state, progress=progress,
    )
    uncertainty_threshold = float(np.percentile(boot.train_pred_std, percentile))
    low = boot.pred_std <= uncertainty_threshold
    result.n_models = boot.n_models
    result.pred_mean = boot.pred_mean
    result.pred_std = boot.pred_std
    result.train_pred_std = boot.train_pred_std
    result.uncertainty_threshold = uncertainty_threshold
    result.low_uncertainty = low
    result.reliable = result.inside_ad & low
    result.uncertainty_kind = boot.kind
    result.fit_seconds = boot.fit_seconds
    return result

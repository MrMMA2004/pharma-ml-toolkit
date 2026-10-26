"""Model evaluation with structured, report-friendly result objects.

The legacy scripts printed ``classification_report`` to stdout and
immediately dumped PNG/PDF files; this module returns typed report
objects (dataclasses) that callers can print, serialise, or feed into
report generators -- printing is a one-line convenience.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.evaluation import evaluate_model
>>> class EchoClassifier:  # toy fitted estimator echoing a fixed vector
...     def __init__(self, predictions):
...         self._preds = np.asarray(predictions)
...     def predict(self, X):
...         return self._preds[: len(X)]
...     def predict_proba(self, X):
...         pos = (self._preds[: len(X)] == 1).astype(float)
...         return np.column_stack([1 - pos, pos])
>>> y_true = [0, 0, 1, 1]
>>> X_test = [[0.0], [1.0], [2.0], [3.0]]
>>> report = evaluate_model(EchoClassifier(y_true), X_test, y_true, "classification")
>>> report.accuracy, report.balanced_accuracy, report.mcc, report.roc_auc
(1.0, 1.0, 1.0, 1.0)
>>> report.confusion.tolist()               # rows: true label, columns: predicted
[[2, 0], [0, 2]]
>>> [int(v) for v in report.labels]         # union of true and predicted labels
[0, 1]
>>> str(report)                             # one-line presentation
'accuracy=1.0000  balanced=1.0000  f1(macro)=1.0000  mcc=1.0000  roc_auc=1.0000  n=4'

>>> report = evaluate_model(EchoClassifier([0, 1, 1, 1]), X_test, y_true, "classification")
>>> report.accuracy, report.balanced_accuracy, round(report.mcc, 4), report.roc_auc
(0.75, 0.75, 0.5774, 0.75)
>>> report.confusion.tolist()               # one false positive (row 0, column 1)
[[1, 1], [0, 2]]

>>> class HardClassifier:  # toy estimator without predict_proba
...     def __init__(self, predictions):
...         self._preds = np.asarray(predictions)
...     def predict(self, X):
...         return self._preds[: len(X)]
>>> evaluate_model(HardClassifier(y_true), X_test, y_true, "classification").roc_auc is None
True

>>> class OffsetRegressor:  # toy fitted estimator: predictions = features + offset
...     def __init__(self, offset=1.0):
...         self.offset = offset
...     def predict(self, X):
...         return np.asarray(X, dtype=float).ravel() + self.offset
>>> y_reg = [1.0, 2.0, 3.0]
>>> evaluate_model(OffsetRegressor(0.0), [[1.0], [2.0], [3.0]], y_reg, "regression").as_dict()
{'rmse': 0.0, 'mae': 0.0, 'r2': 1.0, 'max_error': 0.0, 'n_samples': 3}
>>> evaluate_model(OffsetRegressor(1.0), [[1.0], [2.0], [3.0]], y_reg, "regression").as_dict()
{'rmse': 1.0, 'mae': 1.0, 'r2': -0.5, 'max_error': 1.0, 'n_samples': 3}
>>> fitted = evaluate_model(OffsetRegressor(1.0), [[1.0], [2.0], [3.0]], y_reg,
...                         "regression", keep_predictions=True)
>>> fitted.predictions.tolist()             # kept for parity plots
[2.0, 3.0, 4.0]

>>> evaluate_model(OffsetRegressor(0.0), [[1.0]], [1.0], "banana")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: task must be 'classification' or 'regression', got 'banana'
>>> evaluate_model(OffsetRegressor(0.0), [[1.0], [2.0]], [1.0, 2.0, 3.0], "regression")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: prediction length 2 != target length 3
>>> evaluate_model(OffsetRegressor(0.0), [[]], [], "regression")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: y_test is empty — evaluate_model needs at least one held-out sample
>>> evaluate_model(EchoClassifier([0, 1, 0, 1]), X_test,
...                [0.0, 1.0, float("nan"), 1.0], "classification")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: y_test contains 1 NaN value(s) — drop or impute the target before evaluating
>>> evaluate_model(OffsetRegressor(0.0), [[1.0], [2.0], [3.0]], [1.0, float("inf"), 3.0],
...                "regression")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: y_test contains 1 infinite value(s) — regression metrics need finite targets
>>> evaluate_model(EchoClassifier([0, 1, 0, 1]), X_test,
...                [0.0, 1.0, float("inf"), 1.0], "classification")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: y_test contains 1 infinite value(s) — classification metrics need finite targets
>>> evaluate_model(EchoClassifier([0.0, 1.0, 0.0, float("-inf")]), X_test,
...                [0.0, 1.0, 0.0, 1.0], "classification")
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: model predictions contain 1 infinite value(s) — the estimator returned non-finite values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: numpy dtype "kind" characters that count as numeric labels (signed /
#: unsigned int, float, bool).  ``evaluate_model``'s label-type guard:
#: a numeric y_test crossed with string predictions (or vice versa)
#: would otherwise surface as a raw sklearn "Mix of label input types"
#: ValueError from deep inside the metric calls (100x-36).
_NUMERIC_KINDS = frozenset("iufb")

__all__ = [
    "ClassificationReport",
    "PermutationImportance",
    "RegressionReport",
    "evaluate_model",
    "permutation_feature_importance",
    "predict_with_uncertainty",
]


@dataclass(slots=True)
class ClassificationReport:
    """Structured classification metrics on a held-out set."""

    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    mcc: float
    roc_auc: float | None
    confusion: np.ndarray
    labels: list[Any]
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for DataFrame/JSON consumption."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "mcc": self.mcc,
            "roc_auc": self.roc_auc,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        auc_text = "n/a" if self.roc_auc is None else f"{self.roc_auc:.4f}"
        return (
            f"accuracy={self.accuracy:.4f}  balanced={self.balanced_accuracy:.4f}  "
            f"f1(macro)={self.f1_macro:.4f}  mcc={self.mcc:.4f}  roc_auc={auc_text}  "
            f"n={self.n_samples}"
        )


@dataclass(slots=True)
class RegressionReport:
    """Structured regression metrics on a held-out set."""

    rmse: float
    mae: float
    r2: float
    max_error: float
    n_samples: int
    predictions: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "max_error": self.max_error,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"rmse={self.rmse:.4f}  mae={self.mae:.4f}  r2={self.r2:.4f}  "
            f"n={self.n_samples}"
        )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray | list,
    task: str,
    keep_predictions: bool = False,
) -> ClassificationReport | RegressionReport:
    """Evaluate a fitted model on held-out data.

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator.
    X_test, y_test:
        Held-out features and targets.
    task:
        ``'classification'`` or ``'regression'``.
    keep_predictions:
        For regression: attach predictions to the report (useful for
        scatter/parity plots downstream).

    Returns
    -------
    ClassificationReport or RegressionReport

    Raises
    ------
    ModelError
        On invalid task, empty ``y_test``, prediction-length
        mismatch, NaN or ±inf values in ``y_test`` or the model
        predictions (both tasks), or a label-type mismatch (a
        numeric ``y_test`` with string predictions, or vice versa).
    """
    y_true = np.asarray(y_test)
    if y_true.size == 0:
        raise ModelError(
            "y_test is empty — evaluate_model needs at least one held-out "
            "sample"
        )
    # Some backends (notably CatBoost) return predictions shaped (n, 1)
    # instead of (n,) — flatten so label concatenation always works.
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)
    if len(y_pred) != len(y_true):
        raise ModelError(
            f"prediction length {len(y_pred)} != target length {len(y_true)}"
        )

    # 100x-36: a custom estimator that predicts string labels for a
    # numeric y_test (or vice versa) used to surface as a RAW sklearn
    # ``ValueError: Mix of label input types`` from deep inside the
    # metric calls (fail-first /tmp/task36 probe) -- name the mismatch
    # instead, the house error contract.  The registry never hits this
    # (its models are fitted on y itself); the guard is for direct
    # evaluate_model callers.  100x-41-F hoisted it above the task
    # branch: the mismatch is task-independent, and the regression
    # metrics used to leak their own raw sklearn ``ValueError:
    # dtype='numeric' is not compatible with arrays of bytes/strings``
    # for the very same mistake (fail-first
    # /tmp/task41f/probe_eval_labeltype.py).
    if (y_true.dtype.kind in _NUMERIC_KINDS) != (y_pred.dtype.kind in _NUMERIC_KINDS):
        raise ModelError(
            "y_test labels and model predictions carry different label "
            f"types ({y_true.dtype} vs {y_pred.dtype}): the estimator's "
            "predict() must return the same label type it was fitted on"
        )

    # 100x-41-F: NaN in either vector used to surface as RAW sklearn
    # ValueErrors in BOTH task paths -- ``Input y_true contains NaN.``
    # / ``Input y_pred contains NaN.`` from the classification metrics,
    # ``Input contains NaN`` and ``Input contains infinity or a value
    # too large for dtype('float64')`` from the regression metrics
    # (fail-first /tmp/task41f/probe_eval_nan.py) -- mirroring the NaN
    # guard make_train_test_split already carries (preprocessing.py).
    # Only float-kind arrays can hold NaN; string/int/bool targets are
    # structurally immune.
    if y_true.dtype.kind == "f":
        n_nan = int(np.isnan(y_true).sum())
        if n_nan:
            raise ModelError(
                f"y_test contains {n_nan} NaN value(s) — drop or impute the "
                "target before evaluating"
            )
    if y_pred.dtype.kind == "f":
        n_nan = int(np.isnan(y_pred).sum())
        if n_nan:
            raise ModelError(
                f"model predictions contain {n_nan} NaN value(s) — the "
                "estimator returned NaN"
            )
    # ±inf guard, both tasks.  100x-41-F scoped it to the regression
    # task (rmse/mae/r2/max_error all die with the raw sklearn
    # infinity ValueError); 100x-42-B closed the carried
    # classification half: a float-kind y_test/y_pred with ±inf also
    # trips sklearn's own infinity check inside type_of_target, so
    # accuracy_score's _check_targets raises the raw ``Input y_true
    # contains infinity or a value too large for dtype('float64')``
    # before any pharmaml code can name it (fail-first
    # /tmp/task42b/probe_eval_inf.py).  The task name interpolates
    # into the first message, keeping 41-F's regression wording
    # byte-identical; the prediction message is task-independent.
    if task in ("classification", "regression"):
        if y_true.dtype.kind == "f":
            n_inf = int(np.isinf(y_true).sum())
            if n_inf:
                raise ModelError(
                    f"y_test contains {n_inf} infinite value(s) — {task} "
                    "metrics need finite targets"
                )
        if y_pred.dtype.kind == "f":
            n_inf = int(np.isinf(y_pred).sum())
            if n_inf:
                raise ModelError(
                    f"model predictions contain {n_inf} infinite value(s) — the "
                    "estimator returned non-finite values"
                )

    if task == "classification":
        # .tolist() lives OUTSIDE np.unique so labels are plain python
        # scalars -- inside, np.unique re-wraps its input as an ndarray
        # and sorted() then yields np.int64/np.str_ entries (100x-34,
        # 33-b finding #4)
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    # 100x-34: locate the positive-class column instead of
                    # assuming column 1.  roc_auc_score treats the
                    # sorted-last label of y_true as positive; a conformant
                    # sklearn estimator lays predict_proba columns out in
                    # its own (sorted) classes_ order, so column 1 happens
                    # to match -- but a custom estimator is free to expose
                    # classes_ in its own order, and then a PERFECT
                    # classifier reported AUC 0.0 (silently inverted; the
                    # 100x-34 fail-first probe).  Index the column through
                    # the estimator's own classes_ when it exposes one;
                    # without classes_ the sklearn convention (column 1) is
                    # the only information available.
                    pos_col = 1
                    model_classes = getattr(model, "classes_", None)
                    if model_classes is not None and len(model_classes) == 2:
                        pos_label = np.unique(y_true)[-1]
                        for col, cls in enumerate(model_classes):
                            if cls == pos_label:
                                pos_col = col
                                break
                    roc_auc = float(roc_auc_score(y_true, proba[:, pos_col]))
                else:
                    # 100x-35: the multiclass twin of the 100x-34 binary
                    # fix.  roc_auc_score(labels=labels) maps y_score
                    # column j onto labels[j] (sorted-union order).  A
                    # conformant estimator lays proba columns out in its
                    # own sorted classes_ order, so the mapping coincides
                    # -- but a custom estimator is free to expose classes_
                    # in any order, and then score columns get silently
                    # attached to the wrong classes: a PERFECT 3-class
                    # classifier with classes_=['c','a','b'] reported
                    # roc_auc = 0.25 instead of 1.0 (the 100x-35
                    # fail-first probe).  Permute the columns into the
                    # labels order through the estimator's own classes_
                    # when it exposes a matching set; without classes_ the
                    # sklearn convention (sorted) is the only information
                    # available.
                    proba_ovr = proba
                    model_classes = getattr(model, "classes_", None)
                    if (
                        model_classes is not None
                        and len(model_classes) == proba.shape[1]
                        and set(model_classes.tolist()) == set(labels)
                    ):
                        col_of = {cls: col for col, cls in enumerate(model_classes)}
                        order = [col_of[cls] for cls in labels]
                        if order != list(range(len(labels))):
                            proba_ovr = proba[:, order]
                    roc_auc = float(
                        roc_auc_score(y_true, proba_ovr, multi_class="ovr", labels=labels)
                    )
            else:
                roc_auc = None
        except Exception as exc:  # noqa: BLE001 — AUC undefined for some label setups
            _LOG.warning("ROC-AUC could not be computed: %s", exc)
            roc_auc = None

        return ClassificationReport(
            accuracy=float(accuracy_score(y_true, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro")),
            f1_weighted=float(f1_score(y_true, y_pred, average="weighted")),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            mcc=float(matthews_corrcoef(y_true, y_pred)),
            roc_auc=roc_auc,
            confusion=confusion_matrix(y_true, y_pred, labels=labels),
            labels=labels,
            n_samples=len(y_true),
        )

    if task == "regression":
        return RegressionReport(
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
            max_error=float(np.max(np.abs(np.asarray(y_true, dtype=float) - y_pred))),
            n_samples=len(y_true),
            predictions=np.asarray(y_pred) if keep_predictions else None,
        )

    raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")


def _gp_noise_std(gp: GaussianProcessRegressor) -> float:
    """White-noise std of a fitted GP, in RAW target units.

    Every ``noise_level`` in the fitted kernel (WhiteKernel variances,
    normalised-y space) is summed and un-scaled by the estimator's own
    ``_y_train_std`` (1.0 when ``normalize_y=False``). Kernels without a
    white component honestly report 0.0.
    """
    try:
        params = gp.kernel_.get_params(deep=True)
        noise_var = sum(
            float(v)
            for key, v in params.items()
            if key.endswith("noise_level") and not key.endswith("noise_level_bounds")
        )
        y_std = float(np.ravel(gp._y_train_std)[0])
        return float(np.sqrt(noise_var) * y_std)
    except Exception:  # noqa: BLE001 - exotic kernels: latent sigma only
        return 0.0


def predict_with_uncertainty(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and per-row standard deviation of a GP model.

    The uncertainty readout for the registry's ``gaussian_process``
    family (and any other estimator whose ``predict`` accepts
    ``return_std=True``): the pipeline's preprocessing steps transform
    ``X`` and the final estimator returns ``(mean, sigma)`` in the
    ORIGINAL target units -- no manual unwrapping of ``named_steps``.

    Parameters
    ----------
    model:
        A fitted ``gaussian_process`` pipeline (as returned by
        :func:`~pharmaml.ml.training.train_model`) or a bare fitted
        :class:`~sklearn.gaussian_process.GaussianProcessRegressor`.
    X:
        Feature rows to predict (raw, unpreprocessed -- the pipeline's
        own impute/scale steps run first).

    Returns
    -------
    tuple of np.ndarray
        ``(y_mean, y_sigma)`` — the predictive mean and its per-row
        standard deviation, both shaped ``(n_rows,)``. ``y_sigma`` is the
        PREDICTION interval for a new observation — the latent posterior
        std plus the kernel's learned white-noise std (``Var(y*) =
        Var(f*) + σn²``), the uncertainty a QSAR readout actually wants;
        a GP whose kernel carries no white component reports the latent
        std alone.

    Raises
    ------
    ModelError
        When ``model``'s final estimator is not a
        ``GaussianProcessRegressor`` — every other family predicts a
        point estimate only, and pretending otherwise would fabricate
        uncertainty.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.ml import train_model, predict_with_uncertainty
    >>> rng = np.random.default_rng(7)
    >>> x = pd.DataFrame({"a": np.linspace(0.0, 10.0, 40)})
    >>> y = pd.Series(2.0 * x["a"] + rng.normal(0.0, 0.2, 40))
    >>> trained = train_model("gaussian_process", "regression", x, y, tune=False, cv=2)
    >>> mean, sigma = predict_with_uncertainty(trained.model, x)
    >>> mean.shape == sigma.shape == (40,)
    True
    >>> bool((sigma >= 0.0).all())          # a std is never negative
    True
    >>> far = pd.DataFrame({"a": [100.0]})   # extrapolation: sigma grows
    >>> bool(predict_with_uncertainty(trained.model, far)[1][0] > sigma.min())
    True
    >>> from pharmaml.ml import train_model as tm
    >>> ridge = tm("ridge", "regression", x, y, tune=False, cv=2)
    >>> predict_with_uncertainty(ridge.model, x)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: predict_with_uncertainty needs a GaussianProcessRegressor as the final step (the model's final step is Ridge) — only the gaussian_process family carries predictive sigma
    """
    estimator = model
    X_used: Any = X
    if hasattr(model, "steps"):  # a fitted sklearn Pipeline (prep + reg)
        prep = model[:-1]
        X_used = prep.transform(X) if len(prep.steps) else X
        estimator = model.steps[-1][1]
    if not isinstance(estimator, GaussianProcessRegressor):
        final_name = type(estimator).__name__
        raise ModelError(
            "predict_with_uncertainty needs a GaussianProcessRegressor as "
            f"the final step (the model's final step is {final_name}) — only "
            "the gaussian_process family carries predictive sigma"
        )
    mean, sigma = estimator.predict(X_used, return_std=True)
    mean = np.asarray(mean)
    sigma = np.asarray(sigma)
    noise = _gp_noise_std(estimator)
    if noise > 0.0:
        sigma = np.sqrt(sigma**2 + noise**2)
    return mean, sigma


@dataclass(slots=True)
class PermutationImportance:
    """Permutation feature importances on held-out data.

    The typed upgrade of ``D2.py``'s inline
    ``sklearn.inspection.permutation_importance`` call: the original
    fished ``importances_mean`` out of the raw result and lost the
    spread; this record keeps both, plus the feature names, so callers
    can print, serialise, or plot without re-deriving anything.

    Attributes
    ----------
    names:
        Feature labels (column names or positional ``x0, x1, ...``).
    means:
        Mean importance per feature (a score DROP when the column is
        shuffled -- negative values mean the shuffle helped, a classic
        sign of a noise feature).
    stds:
        Standard deviation of the importance across the repeats.
    scoring:
        The scoring rule that was used.
    n_repeats:
        How many shuffles per feature were run.
    n_samples:
        Rows in the evaluation set.
    """

    names: list[str]
    means: np.ndarray
    stds: np.ndarray
    scoring: str | None
    n_repeats: int
    n_samples: int

    def as_frame(self) -> pd.DataFrame:
        """The importances as a DataFrame (name, mean, std columns)."""
        return pd.DataFrame(
            {
                "feature": self.names,
                "importance_mean": np.asarray(self.means, dtype=float),
                "importance_std": np.asarray(self.stds, dtype=float),
            }
        )

    def ranking(self, top_n: int | None = None) -> list[tuple[str, float]]:
        """``[(feature, mean_importance), ...]`` descending by |mean|.

        Ties resolve first-declared-wins (a stable argsort on the
        negated absolute means), matching the viz layer's convention.
        """
        means = np.asarray(self.means, dtype=float)
        order = np.argsort(-np.abs(means), kind="stable")
        keep = len(means) if top_n is None else min(int(top_n), len(means))
        return [(self.names[int(index)], float(means[int(index)])) for index in order[:keep]]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        top = self.ranking(top_n=5)
        rendered = ", ".join(f"{name}={value:+.4f}" for name, value in top)
        return (
            f"permutation importance (scoring={self.scoring}, "
            f"repeats={self.n_repeats}, n={self.n_samples}): {rendered}"
        )


def permutation_feature_importance(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list,
    task: str = "regression",
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str | None = None,
    feature_names: list[str] | None = None,
) -> PermutationImportance:
    """Permutation importances of a fitted model on held-out data.

    Shuffles one feature column at a time and measures how much the
    score drops -- the importance metric ``D2.py`` computed for every
    base model and meta-learner before drawing its heatmaps.  Works
    with any fitted sklearn-compatible estimator or Pipeline (the
    pipeline's preprocessing reruns on every shuffle, so the measure
    reflects the WHOLE model, not just the final step).

    Parameters
    ----------
    model:
        Fitted estimator or sklearn Pipeline.
    X:
        Feature rows (raw, unpreprocessed -- a pipeline transforms them).
    y:
        True targets for the same rows.
    task:
        ``"regression"`` or ``"classification"`` -- picks the default
        scoring when ``scoring`` is ``None`` (``"r2"`` / ``"accuracy"``,
        D2.py's own choices).
    n_repeats:
        Shuffles per feature (D2.py used 5).
    random_state:
        Seed for the shuffler.
    scoring:
        Explicit sklearn scoring string; ``None`` uses the task default.
    feature_names:
        Labels for the columns; ``None`` reads ``X.columns`` when ``X``
        is a DataFrame, else ``x0, x1, ...``.

    Returns
    -------
    PermutationImportance

    Raises
    ------
    ModelError
        If ``task`` is unknown, ``n_repeats`` is not a positive
        integer, or ``X`` and ``y`` disagree on the row count.
    DependencyError
        If scikit-learn is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 3.0 * X[:, 0] - 1.0 * X[:, 2]          # x1 is pure noise
    >>> model = LinearRegression().fit(X, y)
    >>> imp = permutation_feature_importance(model, X, y, n_repeats=3)
    >>> imp.names
    ['x0', 'x1', 'x2']
    >>> imp.ranking()[0][0], round(imp.ranking()[0][1], 1)
    ('x0', 1.9)
    >>> imp.means.shape, imp.stds.shape
    ((3,), (3,))
    """
    if task not in ("regression", "classification"):
        raise ModelError(
            f"task must be 'regression' or 'classification', got {task!r}"
        )
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ModelError(f"n_repeats must be a positive integer, got {n_repeats!r}")
    try:
        from sklearn.inspection import permutation_importance as _sk_perm
    except ImportError as exc:  # pragma: no cover - environment guard
        from pharmaml.exceptions import DependencyError

        raise DependencyError(
            "permutation_feature_importance needs scikit-learn"
        ) from exc

    x_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_values = np.asarray(y).ravel()
    if len(x_values) != len(y_values):
        raise ModelError(
            f"X has {len(x_values)} rows but y carries {len(y_values)} values"
        )
    if feature_names is not None and len(feature_names) != x_values.shape[1]:
        raise ModelError(
            f"feature_names has {len(feature_names)} entries but X carries "
            f"{x_values.shape[1]} columns"
        )
    if feature_names is None:
        feature_names = (
            [str(name) for name in X.columns] if isinstance(X, pd.DataFrame) else
            [f"x{index}" for index in range(x_values.shape[1])]
        )
    if scoring is None:
        scoring = "r2" if task == "regression" else "accuracy"

    result = _sk_perm(
        model,
        X if isinstance(X, pd.DataFrame) else x_values,
        y_values,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,  # nested parallelism broke D2.py's runs; the honest fix
    )
    means = np.asarray(result.importances_mean, dtype=float)
    stds = np.asarray(result.importances_std, dtype=float)
    _LOG.info(
        "permutation importance over %d features (%s, %d repeats, n=%d)",
        len(feature_names),
        scoring,
        n_repeats,
        len(y_values),
    )
    return PermutationImportance(
        names=feature_names,
        means=means,
        stds=stds,
        scoring=scoring,
        n_repeats=int(n_repeats),
        n_samples=len(y_values),
    )

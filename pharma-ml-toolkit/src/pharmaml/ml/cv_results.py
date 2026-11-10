"""Cross-validation as a first-class result -- the Part-3 §19 layer.

``pharmaml.ml.training.compare_models`` already answers the compact
question ("which model wins?") with a mean/std leaderboard.  §19 of the
Part-3 directive promotes the SAME numbers to a reportable result:
every evaluated model must carry its CV strategy, fold count, seed,
individual fold scores, mean, standard deviation, train-side scores,
best hyperparameters and execution time -- presented as
``CV Score = Mean ± Standard Deviation`` (e.g. ``0.923 ± 0.018
(5-fold CV)``), persisted with a dataset fingerprint, and produced by
leakage-safe pipelines under an honestly documented splitter.

The engine here is an explicit fold loop rather than a bare
:func:`sklearn.model_selection.cross_validate` call, because §19.6
demands a *fresh* estimator (and, for the torch ``dnn`` family, a fresh
optimizer and learning-rate scheduler) per fold -- cloning the whole
pipeline per fold makes that guarantee structural instead of implicit,
and gives the GUI a per-fold progress heartbeat that the vectorised
call cannot.

Strategies (§19.3)
------------------
``kfold`` / ``stratified`` / ``group`` / ``stratified_group`` /
``scaffold``.  The molecular strategy groups rows by their Bemis–Murcko
scaffold (:func:`pharmaml.ml.splits.scaffold_groups`) so identical
scaffold keys do not straddle a train/validation boundary. Distinct
scaffolds can still be chemically similar; a scaffold split cannot prove
independence or generalization to all distinct compounds. Every strategy
carries a limitations note that travels with the result.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.ml.cv_results import (
...     cross_validate_model, format_cv_score, default_folds)
>>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
>>> y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
>>> result = cross_validate_model(X, y, "regression", "ridge", n_folds=4)
>>> result.model, result.task, result.strategy, result.n_folds
('ridge', 'regression', 'kfold', 4)
>>> result.primary
'r2'
>>> result.summary()          # the §19.2 display contract (ridge's honest L2 shrinkage)
'0.973 ± 0.006 (4-fold CV)'
>>> format_cv_score(0.923, 0.018, 5)
'0.923 ± 0.018 (5-fold CV)'
>>> default_folds(400)
(5, 'n=400 rows: 5-fold CV (the default when feasible)')
>>> default_folds(30)
(3, 'n=30 rows: 3-fold CV (limited data -- smaller validation folds)')
>>> default_folds(400, dl=True)
(3, 'deep model: 3-fold CV (fold-for-fold retraining is expensive)')

The fold table keeps every individual score (§19.2's "individual fold
scores") plus the train-side mirror where the metric allows it:

>>> frame = result.to_frame()
>>> frame.index.tolist()
[0, 1, 2, 3]
>>> sorted(frame.columns)
['fold', 'fold_mae', 'fold_median_ae', 'fold_mse', 'fold_r2', 'fold_rmse', 'n_train', 'n_val', 'train_mae', 'train_median_ae', 'train_mse', 'train_r2', 'train_rmse']
>>> round(float(frame["fold_r2"].mean()), 3), round(result.metrics["r2"].mean, 3)
(0.973, 0.973)
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from pharmaml import __version__ as TOOLKIT_VERSION
from pharmaml.exceptions import DataValidationError, JobCancelled, ModelError
from pharmaml.ml.models import (
    MODEL_REGISTRY,
    NO_SCALE_MODELS,
    build_model,
    resolve_model_name,
)
from pharmaml.ml.preprocessing import build_preprocessor, has_categorical_columns
from pharmaml.ml.splits import scaffold_groups

__all__ = [
    "CLASSIFICATION_CV_METRICS",
    "CV_STRATEGIES",
    "METRIC_LABELS",
    "REGRESSION_CV_METRICS",
    "CVResult",
    "FoldRecord",
    "MetricSummary",
    "build_cv_splitter",
    "compare_models_cv",
    "cross_validate_model",
    "cv_result_from_dict",
    "cv_result_to_dict",
    "dataset_fingerprint",
    "default_folds",
    "format_cv_score",
]

#: Splitters a CV report can name (§19.3).  ``stratified`` /
#: ``stratified_group`` need a class label; ``group`` /
#: ``stratified_group`` need ``groups``; ``scaffold`` needs ``smiles``.
CV_STRATEGIES: tuple[str, ...] = (
    "kfold",
    "stratified",
    "group",
    "stratified_group",
    "scaffold",
)

#: §19.4 panel for classification -- sklearn scorer names.  The binary
#: rows (``precision`` / ``recall`` / ``f1`` / ``average_precision``)
#: are dropped for multiclass targets and proba-based rows are dropped
#: for estimators without ``predict_proba``; every drop leaves a note.
CLASSIFICATION_CV_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "f1_weighted",
    "roc_auc_ovo",
    "average_precision",
    "neg_log_loss",
)

#: §19.5 panel for regression -- sklearn scorer names (all higher-is-
#: better in sklearn convention; the display labels carry the honest
#: direction, e.g. RMSE is reported as a positive magnitude).
REGRESSION_CV_METRICS: tuple[str, ...] = (
    "r2",
    "neg_root_mean_squared_error",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_median_absolute_error",
)

#: Display labels + honest direction for every reportable metric.
#: ``lower_is_better`` refers to the DISPLAYED magnitude (RMSE / MAE /
#: MSE / median AE are shown positive, so lower is better).
METRIC_LABELS: dict[str, tuple[str, bool]] = {
    "accuracy": ("Accuracy", False),
    "balanced_accuracy": ("Balanced Accuracy", False),
    "precision": ("Precision", False),
    "recall": ("Recall", False),
    "f1": ("F1", False),
    "f1_macro": ("Macro F1", False),
    "f1_weighted": ("Weighted F1", False),
    "roc_auc_ovo": ("ROC-AUC (one-vs-one)", False),
    "average_precision": ("PR-AUC (average precision)", False),
    "neg_log_loss": ("Log Loss", True),
    "r2": ("R²", False),
    "neg_root_mean_squared_error": ("RMSE", True),
    "neg_mean_absolute_error": ("MAE", True),
    "neg_mean_squared_error": ("MSE", True),
    "neg_median_absolute_error": ("Median Absolute Error", True),
}

#: Scorers that need ``predict_proba`` on the estimator.
_PROBA_METRICS = frozenset(
    {"roc_auc_ovo", "average_precision", "neg_log_loss"}
)

#: Scorers whose sklearn default averaging is binary-only.
_BINARY_ONLY_METRICS = frozenset({"precision", "recall", "f1", "average_precision"})

_PRIMARY_DEFAULTS: dict[str, str] = {
    "classification": "roc_auc_ovo",
    "regression": "r2",
}

#: The classification fallback when no compared model exposes
#: ``predict_proba`` (mirrors ``compare_models``'s board rule).
_CLASSIFICATION_FALLBACK = "balanced_accuracy"


# ---------------------------------------------------------------------------
# the display contract (§19.2)
# ---------------------------------------------------------------------------


def format_cv_score(mean: float, std: float, n_folds: int, precision: int = 3) -> str:
    """Render ``CV Score = Mean ± Standard Deviation`` honestly.

    Examples
    --------
    >>> format_cv_score(0.923, 0.018, 5)
    '0.923 ± 0.018 (5-fold CV)'
    >>> format_cv_score(-1.6613, 0.0, 4)
    '-1.661 ± 0.000 (4-fold CV)'
    >>> format_cv_score(float("nan"), 0.018, 5)
    'nan ± 0.018 (5-fold CV)'
    """
    return f"{float(mean):.{precision}f} ± {float(std):.{precision}f} ({int(n_folds)}-fold CV)"


# ---------------------------------------------------------------------------
# fold-count guidance (§19.3)
# ---------------------------------------------------------------------------


def default_folds(
    n_samples: int, *, dl: bool = False, strategy: str | None = None
) -> tuple[int, str]:
    """Fold-count default with its justification (§19.3).

    5-fold when feasible; 3-fold for limited datasets (< 50 rows) or
    deep models (each fold retrains from scratch, §19.6's cost note).
    Grouped strategies keep the same numbers but the note names the
    grouping, whose minimum is *distinct groups*, not rows.
    """
    if n_samples < 1:
        raise DataValidationError(f"n_samples must be >= 1, got {n_samples}")
    if dl:
        return 3, "deep model: 3-fold CV (fold-for-fold retraining is expensive)"
    if n_samples < 50:
        return 3, f"n={n_samples} rows: 3-fold CV (limited data -- smaller validation folds)"
    if strategy in ("group", "stratified_group", "scaffold"):
        return (
            5,
            f"n={n_samples} rows: 5-fold CV over whole groups (never split a group)",
        )
    return 5, f"n={n_samples} rows: 5-fold CV (the default when feasible)"


# ---------------------------------------------------------------------------
# the splitter core (§19.3)
# ---------------------------------------------------------------------------


def build_cv_splitter(
    task: str,
    *,
    strategy: str | None = None,
    n_folds: int = 5,
    seed: int = 42,
    y: Sequence[Any] | None = None,
    smiles: Sequence[str] | None = None,
    groups: Sequence[object] | None = None,
) -> tuple[Any, list[str]]:
    """Build the CV splitter plus its limitations notes (§19.3).

    Parameters
    ----------
    task:
        ``'classification'`` or ``'regression'`` -- decides the default
        strategy (``stratified`` vs ``kfold``).
    strategy:
        One of :data:`CV_STRATEGIES`; ``None`` picks the task default.
    n_folds:
        Fold count (>= 2; also <= class count for stratified variants
        and <= group count for grouped variants -- honest errors, not
        doomed all-NaN boards).
    seed:
        Shuffle seed (``GroupKFold`` is deterministic and unshuffled by
        sklearn's own contract -- noted, not silently ignored).
    y:
        Target vector; required by ``stratified`` / ``stratified_group``.
    smiles:
        One SMILES per row; required by ``scaffold`` (Bemis–Murcko keys
        via :func:`pharmaml.ml.splits.scaffold_groups`). When y is supplied,
        the lengths must match.
    groups:
        One group label per row; required by ``group`` /
        ``stratified_group``. Not used by ``scaffold``. The returned sklearn
        splitter is not bound to these labels; callers invoking its ``split``
        method directly must supply the appropriate aligned groups themselves.

    Returns
    -------
    tuple[splitter, list[str]]
        The sklearn splitter and the honest limitations notes that
        travel with every result produced under it.

    Raises
    ------
    DataValidationError, ModelError
        On an unknown task/strategy, ``n_folds < 2``, or a missing /
        degenerate prerequisite (no labels, too few classes or groups).
    """
    if task not in {"classification", "regression"}:
        raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")
    if strategy is None:
        strategy = "stratified" if task == "classification" else "kfold"
    if strategy not in CV_STRATEGIES:
        raise DataValidationError(
            f"unknown CV strategy {strategy!r} -- valid strategies: "
            + ", ".join(CV_STRATEGIES)
        )
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ModelError(f"n_folds must be >= 2, got {n_folds!r}")

    notes: list[str] = []
    n_rows: int | None = None
    if y is not None:
        n_rows = len(y)
    elif smiles is not None:
        n_rows = len(smiles)
    elif groups is not None:
        n_rows = len(groups)
    if n_rows is not None and n_folds > n_rows:
        raise ModelError(
            f"n_folds={n_folds} exceeds the number of rows ({n_rows}) -- "
            "no fold can be built"
        )

    if strategy == "kfold":
        notes.append(
            "KFold: plain row-level folds -- related rows (replicates, "
            "sister compounds) can straddle train/validation; use a "
            "grouped strategy when that would leak."
        )
        return KFold(n_splits=n_folds, shuffle=True, random_state=seed), notes

    if strategy == "stratified":
        if y is None:
            raise DataValidationError("strategy 'stratified' needs y (the class labels)")
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        if len(classes) < 2:
            raise DataValidationError(
                "stratified CV needs >= 2 classes; the target has one"
            )
        if int(counts.min()) < n_folds:
            raise DataValidationError(
                f"stratified {n_folds}-fold needs >= {n_folds} rows in the "
                f"rarest class (found {int(counts.min())}) -- lower n_folds "
                "or merge rare classes"
            )
        notes.append(
            "StratifiedKFold: class proportions preserved per fold; "
            "scaffold similarity is NOT controlled -- pair with the "
            "scaffold strategy when structural generalisation matters."
        )
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), notes

    if strategy in ("group", "stratified_group"):
        if groups is None:
            raise DataValidationError(
                f"strategy {strategy!r} needs groups (one label per row)"
            )
        group_values = np.asarray(list(groups))
        if group_values.ndim != 1:
            raise DataValidationError("groups must be a 1-D sequence of labels")
        n_groups = len(set(group_values.tolist()))
        if n_groups < n_folds:
            raise DataValidationError(
                f"{n_groups} distinct groups cannot fill {n_folds} folds -- "
                "lower n_folds or review the grouping"
            )
        notes.append(
            "GroupKFold: whole groups never straddle train/validation "
            "(leakage-honest for replicates / repeated samples); sklearn's "
            "GroupKFold is deterministic and unshuffled -- fold order "
            "follows group order, the seed does not reorder it."
        )
        if strategy == "group":
            return GroupKFold(n_splits=n_folds), notes
        if y is None:
            raise DataValidationError(
                "strategy 'stratified_group' needs y (the class labels)"
            )
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        if len(classes) < 2:
            raise DataValidationError(
                "stratified_group CV needs >= 2 classes; the target has one"
            )
        if int(counts.min()) < n_folds:
            raise DataValidationError(
                f"stratified_group {n_folds}-fold needs >= {n_folds} rows "
                f"in the rarest class (found {int(counts.min())})"
            )
        notes[-1] = (
            "StratifiedGroupKFold: class proportions balanced AND whole "
            "groups never straddle train/validation -- the honest default "
            "when both replicate leakage and class skew are concerns."
        )
        return (
            StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed),
            notes,
        )

    # strategy == "scaffold"
    if smiles is None:
        raise DataValidationError(
            "strategy 'scaffold' needs smiles (one SMILES per row) -- "
            "the Bemis-Murcko scaffold key defines the group"
        )
    if y is not None and len(smiles) != len(y):
        raise DataValidationError(
            "strategy 'scaffold' needs one SMILES per target row"
        )
    keys = scaffold_groups([str(item) for item in smiles])
    n_groups = len(set(keys))
    if n_groups < n_folds:
        raise DataValidationError(
            f"{n_groups} distinct Bemis-Murcko scaffolds cannot fill "
            f"{n_folds} folds -- lower n_folds or review the dataset's "
            "structural diversity"
        )
    notes.append(
        "Scaffold CV (GroupKFold over Bemis-Murcko keys): identical keys "
        "never straddle train/validation. Distinct scaffolds can still be "
        "similar, so this does not prove independence or generalisation. "
        "Acyclic molecules and failed parses share an empty key; validate "
        "SMILES and audit other leakage units separately."
    )
    return GroupKFold(n_splits=n_folds), notes


# ---------------------------------------------------------------------------
# the dataset fingerprint (§19.9)
# ---------------------------------------------------------------------------


def dataset_fingerprint(X: pd.DataFrame | np.ndarray, y: Sequence[Any]) -> str:
    """Stable sha256 fingerprint of the data a CV result belongs to.

    Covers shape, column names, the y vector and the X block's bytes --
    enough that results from different datasets / preprocessing
    configurations never silently mix (§19.9).  Object columns fall
    back to their string rendering (deterministic for pandas data).
    """
    hasher = hashlib.sha256()
    arr = np.asarray(X)
    hasher.update(f"shape={arr.shape}".encode())
    if isinstance(X, pd.DataFrame):
        hasher.update(("columns=" + "|".join(map(str, X.columns))).encode())
    else:
        hasher.update(f"columns={arr.shape[1] if arr.ndim == 2 else 0}".encode())
    y_arr = np.asarray(y)
    hasher.update(f"y-shape={y_arr.shape}".encode())
    try:
        hasher.update(np.ascontiguousarray(y_arr, dtype=float).tobytes())
    except (TypeError, ValueError):
        hasher.update(("y-values=" + "|".join(map(str, y_arr.tolist()))).encode())
    try:
        hasher.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    except (TypeError, ValueError):
        flat = arr.reshape(-1) if arr.ndim > 1 else arr
        hasher.update(("x-values=" + "|".join(map(str, flat.tolist()))).encode())
    return hasher.hexdigest()[:16]


# ---------------------------------------------------------------------------
# the result records (§19.2 / §19.9)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FoldRecord:
    """One fold's honest bookkeeping (§19.2 "individual fold scores").

    ``scores`` holds validation-side metric values; ``train_scores``
    the train-side mirror (``None`` entries where the metric cannot be
    computed on the train side); ``fit_seconds`` the estimator fit time
    for this fold, ``score_seconds`` the two-sided scoring time.
    """

    fold: int
    n_train: int
    n_val: int
    scores: dict[str, float] = field(default_factory=dict)
    train_scores: dict[str, float | None] = field(default_factory=dict)
    fit_seconds: float = 0.0
    score_seconds: float = 0.0


@dataclass(slots=True)
class MetricSummary:
    """One metric aggregated across folds (mean, std, every fold value)."""

    name: str
    mean: float
    std: float
    folds: list[float]
    higher_is_better: bool
    lower_is_better: bool

    @property
    def label(self) -> str:
        """Display label, e.g. ``'RMSE'``."""
        return METRIC_LABELS.get(self.name, (self.name, False))[0]

    def summary(self, precision: int = 3) -> str:
        """``mean ± std (k-fold CV)`` for THIS metric."""
        return format_cv_score(self.mean, self.std, len(self.folds), precision)


@dataclass(slots=True)
class CVResult:
    """A complete §19 CV report for one model on one dataset.

    Attributes
    ----------
    model, task:
        Registry name and ``'classification'`` / ``'regression'``.
    strategy, n_folds, seed:
        The splitter identity (§19.2's first three requirements).
    metrics:
        Every reportable metric as a :class:`MetricSummary`
        (validation side).
    primary:
        The ranking metric's scorer name.
    duration_s:
        Wall-clock total for the whole CV run.
    notes:
        Strategy limitations + every honest metric-drop / NaN note.
    best_params:
        Hyperparameters the model ran at (defaults unless the caller
        tuned first -- §19.2 "best hyperparameters, where applicable").
    folds:
        The per-fold records, fold order 0..n-1.
    test_metrics:
        Optional independent-test metrics (kept VISUALLY and
        structurally separate from the CV numbers, §19.7's "clearly
        distinguish" rule -- never merged into ``metrics``).
    dataset_id, feature_columns:
        §19.9's dataset fingerprint + feature configuration snapshot.
    toolkit_version, created_at:
        Execution metadata.
    """

    model: str
    task: str
    strategy: str
    n_folds: int
    seed: int
    metrics: dict[str, MetricSummary]
    primary: str
    duration_s: float
    notes: list[str] = field(default_factory=list)
    best_params: dict[str, Any] | None = None
    folds: list[FoldRecord] = field(default_factory=list)
    test_metrics: dict[str, float] | None = None
    dataset_id: str = ""
    feature_columns: list[str] = field(default_factory=list)
    toolkit_version: str = ""
    created_at: str = ""

    # -- display ----------------------------------------------------------- #

    def summary(self, precision: int = 3) -> str:
        """``CV Score = Mean ± Std (k-fold CV)`` for the primary metric."""
        metric = self.metrics.get(self.primary)
        if metric is None:
            return f"n/a (no {self.primary} score -- see notes)"
        return metric.summary(precision)

    def headline(self) -> str:
        """``'<model>: 0.923 ± 0.018 (5-fold CV) <label>'`` for boards."""
        metric = self.metrics.get(self.primary)
        label = METRIC_LABELS.get(self.primary, (self.primary, False))[0]
        if metric is None:
            return f"{self.model}: n/a ({label})"
        return f"{self.model}: {self.summary()} {label}"

    def to_frame(self) -> pd.DataFrame:
        """Fold-by-fold table: one row per fold, one column per metric.

        Validation columns are ``fold_<short>``, train-side mirrors
        ``train_<short>`` (the metrics that computed), plus ``fold``,
        ``n_train``, ``n_val`` and the fold timings.
        """
        rows: list[dict[str, Any]] = []
        for record in self.folds:
            row: dict[str, Any] = {
                "fold": record.fold,
                "n_train": record.n_train,
                "n_val": record.n_val,
            }
            for name, value in record.scores.items():
                row[f"fold_{_short_name(name)}"] = value
            for train_name, train_value in record.train_scores.items():
                if train_value is not None:
                    row[f"train_{_short_name(train_name)}"] = train_value
            rows.append(row)
        return pd.DataFrame(rows)

    def board_frame(self) -> pd.DataFrame:
        """One-row leaderboard frame (model, mean, std, summary, notes)."""
        metric = self.metrics.get(self.primary)
        mean = float(metric.mean) if metric else float("nan")
        std = float(metric.std) if metric else float("nan")
        label = METRIC_LABELS.get(self.primary, (self.primary, False))[0]
        return pd.DataFrame(
            [
                {
                    "model": self.model,
                    "task": self.task,
                    "strategy": self.strategy,
                    "folds": self.n_folds,
                    f"{_short_name(self.primary)}_mean": mean,
                    f"{_short_name(self.primary)}_std": std,
                    "cv_score": self.summary(),
                    "metric": label,
                    "duration_s": round(self.duration_s, 3),
                }
            ]
        )

    # -- persistence (§19.9) ----------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict preserving every §19.9 field."""
        return {
            "schema": "pharmaml.cv_result/1",
            "model": self.model,
            "task": self.task,
            "strategy": self.strategy,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "primary": self.primary,
            "metrics": {
                name: {
                    "label": summary.label,
                    "mean": summary.mean,
                    "std": summary.std,
                    "folds": list(summary.folds),
                    "higher_is_better": summary.higher_is_better,
                }
                for name, summary in self.metrics.items()
            },
            "folds": [
                {
                    "fold": record.fold,
                    "n_train": record.n_train,
                    "n_val": record.n_val,
                    "scores": dict(record.scores),
                    "train_scores": {
                        k: v for k, v in record.train_scores.items() if v is not None
                    },
                    "fit_seconds": record.fit_seconds,
                    "score_seconds": record.score_seconds,
                }
                for record in self.folds
            ],
            "duration_s": self.duration_s,
            "notes": list(self.notes),
            "best_params": self.best_params,
            "test_metrics": self.test_metrics,
            "dataset_id": self.dataset_id,
            "feature_columns": list(self.feature_columns),
            "toolkit_version": self.toolkit_version,
            "created_at": self.created_at,
        }


def cv_result_to_dict(result: CVResult) -> dict[str, Any]:
    """§19.9 interchange: :class:`CVResult` → JSON-ready dict."""
    return result.to_dict()


def cv_result_from_dict(data: dict[str, Any]) -> CVResult:
    """Rebuild a :class:`CVResult` from :meth:`CVResult.to_dict` output.

    Raises
    ------
    DataValidationError
        On a foreign schema or missing mandatory fields -- round-trips
        must fail loudly rather than half-materialise.
    """
    if not isinstance(data, dict) or data.get("schema") != "pharmaml.cv_result/1":
        raise DataValidationError(
            "not a pharmaml CV result dict (schema "
            f"{data.get('schema') if isinstance(data, dict) else type(data).__name__!r})"
        )
    try:
        metrics = {
            name: MetricSummary(
                name=name,
                mean=float(payload["mean"]),
                std=float(payload["std"]),
                folds=[float(value) for value in payload["folds"]],
                higher_is_better=bool(payload["higher_is_better"]),
                lower_is_better=not bool(payload["higher_is_better"]),
            )
            for name, payload in data["metrics"].items()
        }
        folds = [
            FoldRecord(
                fold=int(payload["fold"]),
                n_train=int(payload["n_train"]),
                n_val=int(payload["n_val"]),
                scores={k: float(v) for k, v in payload.get("scores", {}).items()},
                train_scores={
                    k: float(v) for k, v in payload.get("train_scores", {}).items()
                },
                fit_seconds=float(payload.get("fit_seconds", 0.0)),
                score_seconds=float(payload.get("score_seconds", 0.0)),
            )
            for payload in data.get("folds", [])
        ]
        best_params = data.get("best_params")
        return CVResult(
            model=str(data["model"]),
            task=str(data["task"]),
            strategy=str(data["strategy"]),
            n_folds=int(data["n_folds"]),
            seed=int(data["seed"]),
            metrics=metrics,
            primary=str(data["primary"]),
            duration_s=float(data.get("duration_s", 0.0)),
            notes=[str(note) for note in data.get("notes", [])],
            best_params=dict(best_params) if best_params else None,
            folds=folds,
            test_metrics=(
                {k: float(v) for k, v in data["test_metrics"].items()}
                if data.get("test_metrics")
                else None
            ),
            dataset_id=str(data.get("dataset_id", "")),
            feature_columns=[str(name) for name in data.get("feature_columns", [])],
            toolkit_version=str(data.get("toolkit_version", "")),
            created_at=str(data.get("created_at", "")),
        )
    except KeyError as exc:
        raise DataValidationError(f"CV result dict is missing field {exc}") from exc


def _short_name(scorer_name: str) -> str:
    """Compact column name: ``neg_root_mean_squared_error`` → ``rmse``."""
    replacements = {
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_squared_error": "mse",
        "neg_mean_absolute_error": "mae",
        "neg_median_absolute_error": "median_ae",
        "roc_auc_ovo": "roc_auc",
        "average_precision": "pr_auc",
        "neg_log_loss": "log_loss",
        "balanced_accuracy": "balanced_acc",
    }
    return replacements.get(scorer_name, scorer_name)


# ---------------------------------------------------------------------------
# the engine (§19.2 / §19.6 / §19.7)
# ---------------------------------------------------------------------------


def _select_metrics(
    task: str,
    y: Sequence[Any],
    pipeline: Pipeline,
    metrics: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Honest metric list for this task/target/estimator (§19.4/§19.5).

    Returns ``(names, notes)`` -- the panel plus a note for every drop
    (binary-only metric on a multiclass target; proba metric on an
    estimator without ``predict_proba``).  Never reports a metric that
    cannot be meaningfully calculated.
    """
    notes: list[str] = []
    if metrics is not None:
        panel = [str(name) for name in metrics]
    else:
        panel = list(
            CLASSIFICATION_CV_METRICS if task == "classification" else REGRESSION_CV_METRICS
        )
    if task == "classification":
        n_classes = len(np.unique(np.asarray(y)))
        if n_classes > 2:
            dropped = [name for name in panel if name in _BINARY_ONLY_METRICS]
            if dropped:
                panel = [name for name in panel if name not in _BINARY_ONLY_METRICS]
                notes.append(
                    f"multiclass target ({n_classes} classes): dropped "
                    f"{', '.join(dropped)} (binary averaging) -- macro/weighted "
                    "variants and roc_auc_ovo carry the panel"
                )
        has_proba = hasattr(pipeline, "predict_proba")
        if not has_proba:
            dropped = [name for name in panel if name in _PROBA_METRICS]
            if dropped:
                panel = [name for name in panel if name not in _PROBA_METRICS]
                notes.append(
                    f"{_last_step_name(pipeline)} exposes no predict_proba: dropped "
                    f"{', '.join(dropped)} (probabilities unavailable -- §19.4 "
                    "'never report a metric that cannot be calculated')"
                )
    return panel, notes


def _last_step_name(pipeline: Pipeline) -> str:
    """Registry-ish name of the pipeline's final estimator step."""
    estimator = pipeline.steps[-1][1]
    return type(estimator).__name__


def _build_pipeline(
    X: pd.DataFrame | np.ndarray, model: str, task: str
) -> Pipeline:
    """Leakage-safe pipeline: preprocessing INSIDE, estimator outside."""
    encode = has_categorical_columns(X)
    key = resolve_model_name(model)
    return Pipeline(
        [
            (
                "prep",
                build_preprocessor(
                    encode_categorical=encode, scale=key not in NO_SCALE_MODELS
                ),
            ),
            ("est", build_model(model, task)),
        ]
    )


def cross_validate_model(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    task: str,
    model: str,
    *,
    strategy: str | None = None,
    n_folds: int = 5,
    seed: int = 42,
    smiles: Sequence[str] | None = None,
    groups: Sequence[object] | None = None,
    metrics: Sequence[str] | None = None,
    primary: str | None = None,
    best_params: dict[str, Any] | None = None,
    return_train_scores: bool = True,
    on_fold: Callable[[dict[str, Any]], None] | None = None,  # heartbeat only
) -> CVResult:
    """Cross-validate ONE registry model into a full §19 CVResult.

    The fold loop is explicit: every fold clones the whole pipeline
    (preprocessor + estimator), so imputation / scaling / encoding are
    fit on the fold's training rows only (§19.7's leakage rule) and a
    deep model gets a fresh module -- and with it a fresh optimizer and
    learning-rate scheduler -- every fold (§19.6).

    Parameters
    ----------
    X, y, task, model:
        Dataset plus registry model name (aliases resolve).
    strategy, n_folds, seed, smiles, groups:
        Splitter configuration, see :func:`build_cv_splitter`.
    metrics:
        Explicit sklearn scorer names, or ``None`` for the honest
        §19.4/§19.5 panel (task- and estimator-aware).
    primary:
        Ranking metric; defaults to ROC-AUC (ovo) for classification
        (``balanced_accuracy`` when no proba metric survives) and R²
        for regression.
    best_params:
        Hyperparameters the model ran at (``{}`` = registry defaults);
        recorded verbatim for §19.9 persistence.
    return_train_scores:
        Include train-side metric mirrors (§19.2 "training scores
        where appropriate").
    on_fold:
        Optional per-fold progress callback
        ``({'fold': i, 'total': k})`` -- the GUI heartbeat.

    Raises
    ------
    DataValidationError, ModelError
        Bad task/strategy/fold configuration, unknown model, or a
        model-task mismatch (the registry's own error, verbatim).
    """
    if task not in {"classification", "regression"}:
        raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")
    key = resolve_model_name(model)  # ModelError on unknown / task clash
    started = time.perf_counter()

    pipeline = _build_pipeline(X, key, task)
    splitter, notes = build_cv_splitter(
        task,
        strategy=strategy,
        n_folds=n_folds,
        seed=seed,
        y=y,
        smiles=smiles,
        groups=groups,
    )
    panel, metric_notes = _select_metrics(task, y, pipeline, metrics)
    notes = notes + metric_notes

    if primary is None:
        primary = _PRIMARY_DEFAULTS[task]
        if task == "classification" and primary not in panel:
            primary = _CLASSIFICATION_FALLBACK
            notes.append(
                "primary metric falls back to balanced_accuracy (no "
                "probability metric is reportable for this panel)"
            )
    elif primary not in panel:
        raise ModelError(
            f"primary metric {primary!r} is not in the reportable panel "
            f"({', '.join(panel)})"
        )

    scorers = {name: get_scorer(name) for name in panel}
    y_arr = np.asarray(y)
    X_frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    fold_groups = None
    if strategy == "scaffold":
        # Do not allow an unrelated groups argument to override the very
        # scaffold keys used to validate and describe this splitter.
        assert smiles is not None  # validated by build_cv_splitter
        fold_groups = np.asarray(scaffold_groups([str(item) for item in smiles]))
        if groups is not None:
            notes.append("Scaffold CV ignores groups; only scaffold keys define the folds.")
    elif (splitter.__class__.__name__ in ("GroupKFold", "StratifiedGroupKFold")
          and groups is not None):
        fold_groups = np.asarray(list(groups))

    fold_records: list[FoldRecord] = []
    split_iter = (
        splitter.split(X_frame, y_arr, groups=fold_groups)
        if fold_groups is not None
        else splitter.split(X_frame, y_arr)
    )
    for fold_index, (train_idx, val_idx) in enumerate(split_iter, start=1):
        X_train, X_val = X_frame.iloc[train_idx], X_frame.iloc[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        # fresh pipeline per fold: fresh preprocessing fit (§19.7) and a
        # fresh estimator -- for torch models a fresh module + optimizer
        # + scheduler (§19.6).  clone() re-instantiates from the
        # constructor params, never copies fitted state.
        fresh = clone(pipeline)
        fit_started = time.perf_counter()
        fresh.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started
        score_started = time.perf_counter()
        fold_scores: dict[str, float] = {}
        train_scores: dict[str, float | None] = {}
        for name, scorer in scorers.items():
            try:
                fold_scores[name] = float(scorer(fresh, X_val, y_val))
            except Exception as exc:  # noqa: BLE001 — one bad metric must not kill the fold
                fold_scores[name] = float("nan")
                notes.append(
                    f"fold {fold_index}: metric {name} failed ({exc}) -- kept as NaN"
                )
            if return_train_scores:
                try:
                    train_scores[name] = float(scorer(fresh, X_train, y_train))
                except Exception:  # noqa: BLE001 — train mirror is best-effort
                    train_scores[name] = None
        score_seconds = time.perf_counter() - score_started
        fold_records.append(
            FoldRecord(
                fold=fold_index,
                n_train=len(train_idx),
                n_val=len(val_idx),
                scores=fold_scores,
                train_scores=train_scores,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
        )
        if on_fold is not None:
            on_fold({"fold": fold_index, "total": n_folds, "model": key})

    summaries: dict[str, MetricSummary] = {}
    for name in panel:
        values = [record.scores.get(name, float("nan")) for record in fold_records]
        array = np.asarray(values, dtype=float)
        label, lower_is_better = METRIC_LABELS.get(name, (name, False))
        summaries[name] = MetricSummary(
            name=name,
            mean=float(np.nanmean(array)) if not np.isnan(array).all() else float("nan"),
            std=float(np.nanstd(array)) if not np.isnan(array).all() else float("nan"),
            folds=[float(value) for value in values],
            higher_is_better=not lower_is_better,
            lower_is_better=lower_is_better,
        )
        if np.isnan(array).any():
            notes.append(
                f"metric {label} has NaN fold(s) -- degenerate fold for this "
                "scorer; the mean/std skip them honestly (nan-aware)"
            )

    duration = time.perf_counter() - started
    return CVResult(
        model=key,
        task=task,
        strategy=(strategy or ("stratified" if task == "classification" else "kfold")),
        n_folds=n_folds,
        seed=seed,
        metrics=summaries,
        primary=primary,
        duration_s=duration,
        notes=notes,
        best_params=dict(best_params) if best_params is not None else {},
        folds=fold_records,
        dataset_id=dataset_fingerprint(X, y),
        feature_columns=[str(name) for name in X_frame.columns],
        toolkit_version=TOOLKIT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def compare_models_cv(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    task: str,
    models: list[str] | None = None,
    *,
    strategy: str | None = None,
    n_folds: int = 5,
    seed: int = 42,
    smiles: Sequence[str] | None = None,
    groups: Sequence[object] | None = None,
    metrics: Sequence[str] | None = None,
    primary: str | None = None,
    return_train_scores: bool = True,
    on_model: Callable[[dict[str, Any]], None | bool] | None = None,
) -> tuple[list[CVResult], pd.DataFrame]:
    """Cross-validate a model panel into full §19 reports + the board.

    Every model runs under the SAME splitter (same folds -- comparable
    boards), each as its own :class:`CVResult`; the returned DataFrame
    is the §19.8 leaderboard: ``model / task / strategy / folds /
    mean / std / cv_score (formatted) / metric / duration_s``, sorted
    by the primary metric mean (descending; lower-is-better metrics
    sort ascending so the best row is always first).

    ``on_model`` mirrors :func:`pharmaml.ml.training.compare_models`'s
    progress contract: one call per completed row
    (``{'model', 'index', 'total'}``) for GUI heartbeats; a ``False``
    answer raises :class:`pharmaml.exceptions.JobCancelled` after that
    row -- the cooperative Stop channel.
    """
    if models is None:
        models = [
            spec.name
            for spec in MODEL_REGISTRY.values()
            if spec.available and spec.task in {task, "both"}
        ]
    else:
        resolved: list[str] = []
        for name in models:
            key = resolve_model_name(name)  # ModelError on unknown names
            resolved.append(MODEL_REGISTRY[key].name)
        # aliases collapse ('rf' and 'random_forest' → one row)
        models = list(dict.fromkeys(resolved))
    if not models:
        raise ModelError(
            "models list is empty — pass registry names (see "
            "pharmaml.ml.list_models()) or None to compare every available "
            "model for the task"
        )
    results: list[CVResult] = []
    for position, name in enumerate(models, start=1):
        result = cross_validate_model(
            X,
            y,
            task,
            name,
            strategy=strategy,
            n_folds=n_folds,
            seed=seed,
            smiles=smiles,
            groups=groups,
            metrics=metrics,
            primary=primary,
            return_train_scores=return_train_scores,
        )
        results.append(result)
        if on_model is not None and on_model(
            {"model": result.model, "index": position, "total": len(models)}
        ) is False:
            raise JobCancelled(
                f"model comparison cancelled after {position}/{len(models)} "
                "models -- stopped at a checkpoint by the caller"
            )
    if not results:
        raise ModelError("models list is empty -- pass registry names or None")
    board = pd.concat([result.board_frame() for result in results], ignore_index=True)
    short = _short_name(results[0].primary)
    column = f"{short}_mean"
    lower_is_better = METRIC_LABELS.get(
        results[0].primary, (results[0].primary, False)
    )[1]
    board = board.sort_values(column, ascending=lower_is_better).reset_index(drop=True)
    return results, board

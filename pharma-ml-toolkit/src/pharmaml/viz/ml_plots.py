"""Visual diagnostics for machine-learning results.

Restores and upgrades the figures the legacy scripts hard-coded:

* ``Classification/RF.py``  → ``plt.savefig('confusion_matrix.png')`` and
  one ROC curve per class — now :func:`plot_confusion_matrix` and
  :func:`plot_roc_curves`, which share one code path for binary and
  multiclass targets.
* ``Regression/RF.py``      → ``Predicted vs Actual`` scatter — now
  :func:`plot_predicted_vs_actual`, plus a residual companion.

New chart types the originals never had: model-leaderboard bars with
CV error bars (:func:`plot_model_comparison`), class balance
(:func:`plot_class_distribution`), feature importance
(:func:`plot_feature_importance`) and per-fold CV scores
(:func:`plot_cv_scores`).

Every function returns a matplotlib ``Figure`` (Agg backend, safe on
headless machines); persist it with
:func:`pharmaml.viz.save_figure` in png/svg/pdf.

Examples
--------
>>> from pharmaml.viz import close_figure
>>> from pharmaml.viz.ml_plots import plot_confusion_matrix
>>> fig = plot_confusion_matrix([[3, 1], [0, 4]], labels=["active", "inactive"])
>>> len(fig.axes)            # heat-map axes + colour bar
2
>>> ax = fig.axes[0]
>>> (ax.get_xlabel(), ax.get_ylabel(), ax.get_title())
('Predicted label', 'True label', 'Confusion matrix')
>>> plot_confusion_matrix([[1, 2, 3], [4, 5, 6]])   # non-square -> typed error
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: confusion matrix must be square, got shape (2, 3)
>>> close_figure(fig)

>>> from pharmaml.viz.ml_plots import plot_predicted_vs_actual
>>> fig = plot_predicted_vs_actual([1.0, 2.0, 3.0], [0.9, 2.2, 2.8])
>>> len(fig.axes), fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()
(1, 'actual', 'predicted (ŷ)')
>>> plot_predicted_vs_actual([1.0, 2.0], [1.0, 2.0, 3.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y_true (2,) and y_pred (3,) must match
>>> close_figure(fig)

>>> from pharmaml.viz import figure_to_base64, save_figure
>>> from pharmaml.viz.ml_plots import plot_class_distribution
>>> fig = plot_class_distribution({"active": 30, "inactive": 10})
>>> fig.axes[0].get_ylabel(), fig.axes[0].get_title()
('samples', 'Class distribution — 40 sample(s), 2 class(es)')
>>> out = save_figure(fig, "/tmp/pharmaml_doctest_class_distribution.png")
>>> out.as_posix()
'/tmp/pharmaml_doctest_class_distribution.png'
>>> figure_to_base64(fig, fmt="gif")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Unsupported figure format 'gif'; supported: png, svg, pdf
>>> close_figure(fig); out.unlink()

>>> from pharmaml.viz.ml_plots import plot_feature_importance
>>> fig = plot_feature_importance({"logp": 0.8, "mw": 0.4, "tpsa": 0.1})
>>> len(fig.axes[0].patches)                 # every feature by default
3
>>> close_figure(fig)
>>> plot_feature_importance({"logp": 0.8, "mw": 0.4}, top_n=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: top_n must be a positive integer or None, got 0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = [
    "plot_calibration_curve",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_cv_scores",
    "plot_feature_importance",
    "plot_learning_curve",
    "plot_model_comparison",
    "plot_predicted_vs_actual",
    "plot_residuals",
    "plot_roc_curves",
]

_LOG = get_logger(__name__)


def _check_top_n(top_n: int | None) -> None:
    """Typed guard for the shared ``top_n`` contract.

    Mirrors ``docking_plots._check_top_n``: ``None`` -> all rows, a
    positive integer -> that many rows, anything else (``0``/negative/
    non-int/bool) raises.  Pre-fix ``plot_feature_importance`` read
    ``top_n=0`` as "empty figure" and ``-1`` as "all but one" while
    ``plot_top_ligands`` read both as "all rows" (100x-41-B).
    """
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1
    ):
        raise DataValidationError(
            f"top_n must be a positive integer or None, got {top_n!r}"
        )


def _unwrap_confusion(confusion, labels: list[Any] | None):
    """Accept a raw matrix, a ``ClassificationReport`` or a ``Dataset``."""
    if hasattr(confusion, "confusion") and hasattr(confusion, "labels"):
        # ClassificationReport
        return np.asarray(confusion.confusion), list(confusion.labels)
    matrix = np.asarray(confusion)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DataValidationError(
            f"confusion matrix must be square, got shape {matrix.shape}"
        )
    if labels is None:
        labels = [str(i) for i in range(matrix.shape[0])]
    if len(labels) != matrix.shape[0]:
        raise DataValidationError(
            f"{len(labels)} labels do not match a {matrix.shape[0]}x{matrix.shape[0]} matrix"
        )
    return matrix, list(labels)


def plot_confusion_matrix(
    confusion,
    labels: list[Any] | None = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> Any:
    """Heat-map of predicted-vs-true class counts.

    Mirrors the legacy ``Classification/*.py`` confusion-matrix PNG but
    annotates each cell with both the count and its row-share, works for
    binary **and multiclass** problems, and colour-scales by correctness
    when ``normalize=True``.

    Parameters
    ----------
    confusion:
        A square count matrix, a
        :class:`~pharmaml.ml.evaluation.ClassificationReport` (labels
        are read automatically), or any object exposing
        ``.confusion``/``.labels``.
    labels:
        Class labels in matrix order (optional when a report is passed).
    normalize:
        Scale colours by row-normalized shares instead of raw counts.

    Returns
    -------
    matplotlib.figure.Figure
    """
    matrix, labels = _unwrap_confusion(confusion, labels)
    n = matrix.shape[0]
    display = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix

    fig, ax = new_figure(figsize=(2.2 + 1.1 * n, 2.2 + 1.0 * n))
    image = ax.imshow(display, cmap="YlGnBu", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    threshold = display.max() / 2.0 if display.max() else 0.0
    for i in range(n):
        for j in range(n):
            share = matrix[i, j] / matrix[i].sum() if matrix[i].sum() else 0.0
            # a float-dtype matrix (e.g. re-plotted normalized counts) must
            # not hit the ":d" format code — cast integer annotations
            cell = f"{display[i, j]:.2f}" if normalize else f"{round(float(display[i, j])):d}"
            annotation = cell + f"\n({share:.0%})"
            ax.text(
                j, i, annotation, ha="center", va="center", fontsize=9,
                color="white" if display[i, j] > threshold else "#1F2937",
            )
    ax.set_xticks(range(n), [str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticks(range(n), [str(label) for label in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title + (" (row-normalized)" if normalize else ""))
    ax.grid(False)
    return fig


def plot_roc_curves(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series | list,
    title: str = "ROC curves",
) -> Any:
    """Receiver-operating curves — binary or one-vs-rest multiclass.

    Reproduces the legacy per-class ROC PNGs (``AUC {cls}={auc:.3f}``
    legend entries) for any classifier with ``predict_proba``:

    * binary targets    → one curve for the positive class;
    * 3+ class targets  → one curve per class (one-vs-rest), each in a
      different house colour, sharing the diagonal reference.

    Parameters
    ----------
    model:
        Fitted estimator or Pipeline exposing ``predict_proba``.
    X_test, y_test:
        Held-out features and labels.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ModelError
        If the model lacks ``predict_proba`` or the fitted ``classes_``
        attribute (``classes_`` is set during ``fit`` — a stub exposing
        only ``predict_proba`` used to die with a raw ``AttributeError``).
    """
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_curve

    if not hasattr(model, "predict_proba"):
        raise ModelError(
            "plot_roc_curves needs a model with predict_proba; "
            "fit a probabilistic classifier (logistic_regression, random_forest, ...)"
        )
    if not hasattr(model, "classes_"):
        raise ModelError(
            "plot_roc_curves needs a FITTED model: predict_proba is present "
            "but classes_ is not (classes_ is set during fit)"
        )

    proba = np.asarray(model.predict_proba(X_test))
    classes = list(model.classes_)
    y_true = np.asarray(y_test)

    fig, ax = new_figure(figsize=(7.0, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", label="chance")

    if len(classes) == 2:
        scores = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=classes[1])
        area = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[0], lw=2.2,
                label=f"AUC {classes[1]}={area:.3f}")
    else:
        for index, cls in enumerate(classes):
            binary_truth = (y_true == cls).astype(int)
            scores = proba[:, index]
            try:
                fpr, tpr, _ = roc_curve(binary_truth, scores)
            except ValueError as exc:
                _LOG.warning("ROC for class %r skipped: %s", cls, exc)
                continue
            area = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2.0, color=PALETTE[index % len(PALETTE)],
                    label=f"AUC {cls}={area:.3f}")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_calibration_curve(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series | list,
    n_bins: int = 10,
    title: str = "Calibration (reliability diagram)",
) -> Any:
    """Reliability diagram — are the predicted probabilities honest?

    A diagnostic the legacy scripts never drew: discrimination (ROC)
    says *whether* the model ranks actives above inactives, calibration
    says whether a predicted ``0.8`` really means 80 % positives. Poorly
    calibrated probabilities quietly corrupt threshold choices, expected
    cost calculations and consensus rescoring.

    * binary targets    → one curve (positive class) with its Brier score;
    * 3+ class targets  → one one-vs-rest curve per class, each with its
      Brier score, sharing the diagonal reference.

    Parameters
    ----------
    model:
        Fitted estimator or Pipeline exposing ``predict_proba``.
    X_test, y_test:
        Held-out features and labels.
    n_bins:
        Probability bins for the diagram (classic: 10).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ModelError
        If the model lacks ``predict_proba`` or the fitted ``classes_``
        attribute (``classes_`` is set during ``fit`` — a stub exposing
        only ``predict_proba`` used to die with a raw ``AttributeError``).
    DataValidationError
        If ``n_bins < 2``.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if not hasattr(model, "predict_proba"):
        raise ModelError(
            "plot_calibration_curve needs a model with predict_proba; "
            "fit a probabilistic classifier (logistic_regression, random_forest, ...)"
        )
    if not hasattr(model, "classes_"):
        raise ModelError(
            "plot_calibration_curve needs a FITTED model: predict_proba is "
            "present but classes_ is not (classes_ is set during fit)"
        )
    if n_bins < 2:
        raise DataValidationError(f"n_bins must be >= 2, got {n_bins}")

    proba = np.asarray(model.predict_proba(X_test))
    classes = list(model.classes_)
    y_true = np.asarray(y_test)

    if len(classes) == 2:
        curves = [(classes[1], (y_true == classes[1]).astype(int), proba[:, 1])]
    else:
        curves = [
            (cls, (y_true == cls).astype(int), proba[:, index])
            for index, cls in enumerate(classes)
        ]

    fig, ax = new_figure(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF",
            label="perfectly calibrated")

    for index, (cls, truth, scores) in enumerate(curves):
        fraction_positive, mean_predicted = calibration_curve(
            truth, scores, n_bins=n_bins, strategy="uniform"
        )
        brier = brier_score_loss(truth, scores)
        ax.plot(
            mean_predicted, fraction_positive, marker="o", lw=1.9,
            color=PALETTE[index % len(PALETTE)],
            label=f"{cls} — Brier {brier:.3f}",
        )

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction of positives")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_learning_curve(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series | list,
    cv: int = 5,
    scoring: str | None = None,
    train_sizes: np.ndarray | list[float] | None = None,
    title: str = "Learning curve",
) -> Any:
    """Training-set size vs CV score — is more data still buying accuracy?

    The honest answer to "should we assay 500 more compounds?" that a
    single CV number cannot give:

    * both curves converging to a plateau → more data will not help much;
    * validation still climbing at the right edge → more data *will* help;
    * a persistent gap between the curves → variance/overfitting, not a
      data problem.

    Parameters
    ----------
    model:
        Any scikit-learn compatible estimator (cloned per fold).
    X, y:
        Full training features and labels.
    cv:
        Fold count for the cross-validation at each size.
    scoring:
        Scorer name (default: estimator's ``score``).
    train_sizes:
        Absolute or fractional sizes; default ``10 % … 100 %`` in 8 steps.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.model_selection import learning_curve as sklearn_learning_curve

    if cv < 2:
        raise DataValidationError(f"cv must be >= 2, got {cv}")
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    sizes, train_scores, val_scores = sklearn_learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=None,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = new_figure(figsize=(8.4, 5.4))
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=PALETTE[0], linewidth=0)
    ax.plot(sizes, train_mean, marker="o", lw=1.9, color=PALETTE[0],
            label=f"training score ({train_mean[-1]:.3g})")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=PALETTE[1], linewidth=0)
    ax.plot(sizes, val_mean, marker="s", lw=1.9, color=PALETTE[1],
            label=f"CV score ({val_mean[-1]:.3g})")

    gap = float(train_mean[-1] - val_mean[-1])
    verdict = (
        f"gap ≈ {gap:.2f} — overfitting dominates, more data alone will not close it"
        if gap > 0.15
        else "curves converging — additional data has diminishing returns"
    )
    ax.set_xlabel("training samples")
    ax.set_ylabel(scoring or "score")
    ax.set_title(f"{title}\n{verdict}")
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | list,
    metrics: Mapping[str, float] | None = None,
    title: str = "Predicted vs actual",
) -> Any:
    """Parity scatter for regression, with the identity reference line.

    Upgrades the legacy ``Regression/*.py`` figure: the diagonal
    ``ŷ = y`` line is now labelled, and an optional ``metrics`` mapping
    (e.g. ``{"RMSE": 0.38, "R²": 0.77}``) is rendered as a summary box.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape:
        raise DataValidationError(
            f"y_true {truth.shape} and y_pred {prediction.shape} must match"
        )
    if truth.size == 0:
        raise DataValidationError("no values to plot (empty y_true/y_pred)")

    fig, ax = new_figure(figsize=(6.4, 6.0))
    low = float(min(truth.min(), prediction.min()))
    high = float(max(truth.max(), prediction.max()))
    pad = 0.05 * (high - low or 1.0)
    limits = (low - pad, high + pad)
    ax.plot(limits, limits, linestyle="--", color="#9CA3AF", label="perfect prediction (ŷ = y)")
    ax.scatter(truth, prediction, s=28, alpha=0.75, color=PALETTE[0], edgecolor="white",
               linewidth=0.4)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted (ŷ)")
    ax.set_title(title)

    if metrics:
        text = "  ".join(f"{key}={value:.4g}" for key, value in metrics.items())
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                      "edgecolor": "#D1D5DB", "alpha": 0.9})
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_residuals(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | list,
    title: str = "Residual diagnostics",
) -> Any:
    """Residuals vs predicted (left) and residual histogram (right).

    A residual view the legacy scripts never produced — heteroscedastic
    fans and skewed error distributions become visible at a glance.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.size == 0:  # same contract as plot_predicted_vs_actual (not a
        # silent empty figure — every sibling plotter raises typed on empty)
        raise DataValidationError("no values to plot (empty y_true/y_pred)")
    residuals = truth - prediction

    fig, (ax_res, ax_hist) = new_figure(nrows=1, ncols=2, figsize=(11.0, 4.8))
    ax_res.axhline(0.0, linestyle="--", color="#9CA3AF")
    ax_res.scatter(prediction, residuals, s=26, alpha=0.75, color=PALETTE[1],
                   edgecolor="white", linewidth=0.4)
    ax_res.set_xlabel("predicted (ŷ)")
    ax_res.set_ylabel("residual (y − ŷ)")
    ax_res.set_title("Residuals vs predicted")

    ax_hist.hist(residuals, bins=min(30, max(8, len(residuals) // 4)), color=PALETTE[0],
                 alpha=0.85, edgecolor="white")
    ax_hist.axvline(0.0, linestyle="--", color="#9CA3AF")
    ax_hist.set_xlabel("residual (y − ŷ)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Residual distribution")
    fig.suptitle(title, y=1.02)
    return fig


def plot_model_comparison(
    leaderboard: pd.DataFrame,
    metric: str | None = None,
    title: str = "Cross-validated model leaderboard",
) -> Any:
    """Leaderboard as horizontal bars with per-fold std error bars.

    Turns the :func:`~pharmaml.ml.training.compare_models` DataFrame
    into a figure: models sorted by their mean CV metric, error bars
    showing fold-to-fold variability, values annotated at the bar ends.

    Parameters
    ----------
    leaderboard:
        DataFrame from ``compare_models`` — a ``model`` column plus
        ``<metric>_mean`` / ``<metric>_std`` columns.  NaN metric cells
        stay legal (they model failed CV runs); duplicated column
        names and non-numeric metric cells do not.
    metric:
        Metric name when the frame contains several; defaults to the
        first ``*_mean`` column found.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If the frame lacks a ``model`` or ``<metric>_mean`` column,
        carries duplicated column names (a duplicated ``model`` column
        used to render every y-tick as "model"), or holds non-numeric
        cells in the metric columns.
    """
    duplicated = leaderboard.columns[leaderboard.columns.duplicated()]
    if len(duplicated):
        # A duplicated 'model' column turns frame["model"] into a
        # multi-column DataFrame whose iteration yields COLUMN names, so
        # every y-tick silently read "model"; duplicated metric columns
        # break the Series selection outright.
        raise DataValidationError(
            f"duplicate column names: {list(duplicated)} — rename or drop "
            "the duplicates before plotting"
        )
    if "model" not in leaderboard.columns:
        raise DataValidationError("leaderboard needs a 'model' column (compare_models output)")
    if metric is None:
        mean_cols = [c for c in leaderboard.columns if c.endswith("_mean")]
        if not mean_cols:
            raise DataValidationError(
                "no '<metric>_mean' column found in the leaderboard frame"
            )
        metric = mean_cols[0][: -len("_mean")]

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in leaderboard.columns:
        raise DataValidationError(f"column {mean_col!r} not found in the leaderboard")
    _require_numeric_metric(leaderboard, mean_col)
    if std_col in leaderboard.columns:
        _require_numeric_metric(leaderboard, std_col)

    frame = leaderboard.sort_values(mean_col, ascending=True)
    values = frame[mean_col].to_numpy(dtype=float)
    errors = frame[std_col].to_numpy(dtype=float) if std_col in frame.columns else None

    fig, ax = new_figure(figsize=(8.6, 0.62 * len(frame) + 1.6))
    positions = np.arange(len(frame))
    ax.barh(positions, values, xerr=errors, color=PALETTE[0], alpha=0.88,
            edgecolor="white", height=0.62, error_kw={"elinewidth": 1.1, "capsize": 3})
    for pos, value in zip(positions, values):
        if np.isfinite(value):
            ax.text(value, pos, f" {value:.4g}", va="center", ha="left", fontsize=9)
    ax.set_yticks(positions, [str(name) for name in frame["model"]])
    ax.set_xlabel(f"mean CV {metric} (± std across folds)" if errors is not None
                  else f"mean CV {metric}")
    ax.set_title(title)
    return fig


def _require_numeric_metric(frame: pd.DataFrame, column: str) -> None:
    """Reject non-numeric cells in a leaderboard metric column, typed.

    NaN cells stay legal — ``compare_models`` keeps failed models as
    NaN rows.  A string cell used to leak a raw ``TypeError`` from
    ``sort_values`` (mixed '<' comparison) or a raw ``ValueError`` from
    ``to_numpy(dtype=float)`` (100x-41-B corner).
    """
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    bad = values.notna() & numeric.isna()
    if bad.any():
        offenders = ", ".join(
            f"{frame.loc[idx, 'model']}: {values.loc[idx]!r}"
            for idx in frame.index[bad][:5]
        )
        raise DataValidationError(
            f"column {column!r} contains non-numeric value(s) ({offenders}); "
            "model-comparison metrics must be numeric"
        )


def plot_class_distribution(
    class_counts: Mapping[Any, int] | Any,
    title: str = "Class distribution",
) -> Any:
    """Bar chart of class balance (accepts a mapping or a ``Dataset``)."""
    if hasattr(class_counts, "class_counts") and class_counts.class_counts is not None:
        class_counts = class_counts.class_counts
    if not isinstance(class_counts, Mapping):
        raise DataValidationError(
            "class_counts must be a mapping {label: count} or a classification Dataset"
        )
    if not class_counts:
        raise DataValidationError("class_counts is empty — nothing to plot")

    labels = [str(key) for key in class_counts]
    counts = [int(value) for value in class_counts.values()]
    total = sum(counts)

    fig, ax = new_figure(figsize=(1.9 * len(labels) + 3.4, 5.0))
    bars = ax.bar(labels, counts, color=PALETTE[: len(labels)], alpha=0.9, width=0.62,
                  edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                f"{count}\n({count / total:.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("samples")
    ax.set_title(f"{title} — {total} sample(s), {len(labels)} class(es)")
    return fig


def _extract_importances(model_or_mapping, feature_names: list[str] | None):
    """Normalise ``model`` / ``{name: value}`` inputs into (names, values)."""
    if isinstance(model_or_mapping, Mapping):
        names = [str(key) for key in model_or_mapping]
        values = np.asarray(list(model_or_mapping.values()), dtype=float)
        return names, values

    estimator = model_or_mapping
    if hasattr(estimator, "named_steps"):  # sklearn Pipeline
        for step in reversed(list(estimator.named_steps.values())):
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                estimator = step
                break
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).max(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        raise DataValidationError(
            "could not extract feature importances: pass a tree/linear model, "
            "a Pipeline containing one, or an explicit {feature: importance} mapping"
        )

    # A Pipeline's preprocessing steps (impute/scale/one-hot) know the
    # transformed feature names — use them so one-hot expanded columns
    # stay readable instead of "x0, x1, ...".
    if (
        feature_names is None
        and hasattr(model_or_mapping, "named_steps")
        and len(model_or_mapping.named_steps) > 1
    ):
        try:
            expanded = list(model_or_mapping[:-1].get_feature_names_out())
        except Exception:  # noqa: BLE001 — capability probe; transformers vary
            expanded = []
        if expanded and len(expanded) == len(values):
            feature_names = expanded
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(values))]
    if len(feature_names) != len(values):
        raise DataValidationError(
            f"{len(feature_names)} feature names for {len(values)} importance values"
        )
    return [str(name) for name in feature_names], values


def plot_feature_importance(
    model_or_mapping: Any,
    feature_names: list[str] | None = None,
    top_n: int | None = 15,
    title: str = "Feature importance",
) -> Any:
    """Top-N feature importances as horizontal bars.

    Works with tree ensembles (``feature_importances_``), linear models
    (``|coef_|``), wrapped :class:`~sklearn.pipeline.Pipeline` models,
    or a plain ``{feature: importance}`` mapping when you computed the
    scores yourself.

    Parameters
    ----------
    model_or_mapping:
        Fitted estimator, sklearn Pipeline, or ``{feature: importance}``
        mapping.
    feature_names:
        Feature labels (default: ``x0, x1, ...``).
    top_n:
        How many features to show; ``None`` shows every feature.
        Values below 1 and non-integers raise
        :class:`~pharmaml.exceptions.DataValidationError` — the old
        ``0``/``-1`` silently meant "empty figure"/"all but one" here
        while meaning "all rows" in
        :func:`~pharmaml.viz.docking_plots.plot_top_ligands`.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If importances cannot be extracted from ``model_or_mapping`` or
        ``top_n`` is not a positive integer or ``None``.
    """
    _check_top_n(top_n)
    names, values = _extract_importances(model_or_mapping, feature_names)
    # Stable argsort on the NEGATED values: descending by importance with
    # ties resolving first-declared-wins (matches plot_top_ligands); the
    # trailing reversal only flips display order so the top feature sits
    # highest on the axis. argsort(values)[::-1] would flip tied groups.
    keep = len(values) if top_n is None else min(int(top_n), len(values))
    order = np.argsort(-values, kind="stable")[:keep][::-1]

    fig, ax = new_figure(figsize=(8.4, 0.42 * len(order) + 1.7))
    positions = np.arange(len(order))
    ax.barh(positions, values[order], color=PALETTE[0], alpha=0.88, height=0.62,
            edgecolor="white")
    for pos, value in zip(positions, values[order]):
        ax.text(value, pos, f" {value:.4g}", va="center", ha="left", fontsize=9)
    ax.set_yticks(positions, [names[i] for i in order])
    ax.set_xlabel("importance")
    ax.set_title(title)
    return fig


def plot_cv_scores(
    cv_scores: Any,
    model_name: str | None = None,
    metric: str | None = None,
    title: str = "Cross-validation fold scores",
) -> Any:
    """Per-fold CV scores as bars with the mean line and ±std band.

    Accepts a :class:`~pharmaml.ml.training.TrainingResult` or a plain
    array-like of fold scores — the spread across folds is the honest
    uncertainty the legacy scripts hid behind a single mean.
    """
    if hasattr(cv_scores, "cv_scores") and cv_scores.cv_scores is not None:
        model_name = model_name or getattr(cv_scores, "name", None)
        scores = np.asarray(cv_scores.cv_scores, dtype=float)
    else:
        scores = np.asarray(cv_scores, dtype=float)
    scores = scores.reshape(-1)
    if scores.size == 0:
        raise DataValidationError("no CV fold scores to plot")

    mean = float(scores.mean())
    std = float(scores.std())
    label = model_name or "model"
    metric_label = metric or "score"

    fig, ax = new_figure(figsize=(7.6, 4.6))
    positions = np.arange(1, len(scores) + 1)
    ax.bar(positions, scores, color=PALETTE[0], alpha=0.85, width=0.6, edgecolor="white")
    ax.axhline(mean, color=PALETTE[1], lw=2.0,
               label=f"mean = {mean:.4g}")
    ax.axhspan(mean - std, mean + std, color=PALETTE[1], alpha=0.14,
               label=f"± 1 std = {std:.4g}")
    ax.set_xticks(positions, [f"fold {i}" for i in positions])
    ax.set_ylabel(metric_label)
    ax.set_title(f"{title} — {label}")
    ax.legend(fontsize=9, loc="best")
    return fig

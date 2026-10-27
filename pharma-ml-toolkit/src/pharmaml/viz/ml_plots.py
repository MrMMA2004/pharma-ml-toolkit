"""Visual diagnostics for machine-learning results.

Restores and upgrades the figures the legacy scripts hard-coded:

* ``Classification/RF.py``  → ``plt.savefig('confusion_matrix.png')`` and
  one ROC curve per class — now :func:`plot_confusion_matrix` and
  :func:`plot_roc_curves`, which share one code path for binary and
  multiclass targets.
* ``Regression/RF.py``      → ``Predicted vs Actual`` scatter — now
  :func:`plot_predicted_vs_actual`, plus a residual companion.
* ``Applicability_domain_analysis.py`` (user upload) → the combined
  reliability figure — now :func:`plot_applicability_domain` (training
  kNN-distance histogram + distance-vs-prediction scatter, inside /
  outside the domain colour-coded).

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

>>> from pharmaml.viz.ml_plots import plot_gp_uncertainty
>>> fig = plot_gp_uncertainty([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], [0.1, 0.2, 0.3])
>>> fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()
('actual', 'predicted (ŷ)')
>>> close_figure(fig)
>>> plot_gp_uncertainty([1.0], [1.0], [-0.1])   # a std can never be negative
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y_sigma must be non-negative (found min -0.1) — a predictive std can never be negative
>>> plot_gp_uncertainty([1.0, 2.0], [1.0], [0.1, 0.1])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y_true (2,), y_pred (1,) and y_sigma (2,) must match

>>> from pharmaml.viz.ml_plots import plot_training_history
>>> history = [
...     {"epoch": 1, "train_loss": 0.9, "val_loss": 1.0},
...     {"epoch": 2, "train_loss": 0.6, "val_loss": 0.8},
...     {"epoch": 3, "train_loss": 0.5, "val_loss": 0.85},
...     {"epoch": 4, "train_loss": 0.45, "val_loss": 0.9},
... ]
>>> fig = plot_training_history(history)
>>> ax = fig.axes[0]
>>> ax.get_xlabel(), ax.get_ylabel()
('epoch', 'loss')
>>> len(ax.get_legend().get_texts())       # training + validation curves
2
>>> close_figure(fig)
>>> fig = plot_training_history([{"epoch": 1, "train_loss": 0.9}])
>>> len(fig.axes[0].get_legend().get_texts())   # no val column -> one curve
1
>>> close_figure(fig)
>>> plot_training_history([])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: history is empty -- pass the fitted estimator's history_ (any DeepMLP*/ResidualMLP* fit leaves it on the estimator)
>>> plot_training_history([{"epoch": 1}])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: history is missing column(s) ['train_loss'] -- expected records with epoch, train_loss and optionally val_loss
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger
from pharmaml.utils.validation import require_columns
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = [
    "plot_applicability_domain",
    "plot_calibration_curve",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_cv_scores",
    "plot_feature_importance",
    "plot_gp_uncertainty",
    "plot_learning_curve",
    "plot_model_comparison",
    "plot_permutation_heatmap",
    "plot_permutation_importance",
    "plot_pr_curves",
    "plot_predicted_vs_actual",
    "plot_prediction_intervals",
    "plot_residuals",
    "plot_roc_curves",
    "plot_signed_correlations",
    "plot_training_history",
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

    # Per-cell annotations are only readable while cells stay inch-sized,
    # and the canvas must not grow with the class count: an
    # ID-column-as-target run (420 classes) once tried a 464-inch canvas
    # plus 176k text artists (GBs of artists) before this guard.  Beyond
    # the limit the matrix renders as a plain heat-map with small ticks.
    annotated = n <= 50
    side = 2.2 + 1.1 * min(n, 12)
    fig, ax = new_figure(figsize=(side, 2.2 + 1.0 * min(n, 12)))
    image = ax.imshow(display, cmap="YlGnBu", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if annotated:
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
    tick_labels = [str(label) for label in labels]
    tick_size = 9 if annotated else max(4, 9 - n // 40)
    ax.set_xticks(range(n), tick_labels, rotation=45, ha="right",
                  fontsize=tick_size)
    ax.set_yticks(range(n), tick_labels, fontsize=tick_size)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    suffix = " (row-normalized)" if normalize else ""
    if not annotated:
        suffix += f" — {n} classes, cells unannotated"
    ax.set_title(title + suffix)
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


def plot_pr_curves(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series | list,
    title: str = "Precision-Recall curves",
) -> Any:
    """Precision-Recall curves — binary or one-vs-rest multiclass.

    The ``xxx.ipynb`` cell 28 evaluation's PR half (the notebook drew
    OvR PR curves for every model; the toolkit shipped only the ROC
    twin).  Mirrors :func:`plot_roc_curves`'s contract exactly:

    * binary targets    → one curve for the positive class, with its
      average precision (AP) in the legend;
    * 3+ class targets  → one one-vs-rest curve per class, each with
      its AP, each in a different house colour.

    Where ROC's chance line is the diagonal, PR's honest baseline is
    the class PREVALENCE (a no-skill classifier predicts positives at
    that rate): each multiclass curve carries its own dashed baseline
    at that class's positive fraction, and the binary curve carries
    one at the positive-class prevalence — the "no skill" line every
    PR-curve explainer draws but the notebook's own plot omitted.

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
        If the model lacks ``predict_proba`` or the fitted
        ``classes_`` attribute (the same guards as
        :func:`plot_roc_curves`).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if not hasattr(model, "predict_proba"):
        raise ModelError(
            "plot_pr_curves needs a model with predict_proba; "
            "fit a probabilistic classifier (logistic_regression, random_forest, ...)"
        )
    if not hasattr(model, "classes_"):
        raise ModelError(
            "plot_pr_curves needs a FITTED model: predict_proba is present "
            "but classes_ is not (classes_ is set during fit)"
        )

    proba = np.asarray(model.predict_proba(X_test))
    classes = list(model.classes_)
    y_true = np.asarray(y_test)

    fig, ax = new_figure(figsize=(7.0, 6.0))

    if len(classes) == 2:
        scores = proba[:, 1]
        binary_truth = (y_true == classes[1]).astype(int)
        precision, recall, _ = precision_recall_curve(binary_truth, scores)
        ap = average_precision_score(binary_truth, scores)
        prevalence = float(binary_truth.mean())
        ax.axhline(prevalence, linestyle="--", color="#9CA3AF",
                   label=f"no skill ({prevalence:.2f})")
        ax.plot(recall, precision, color=PALETTE[0], lw=2.2,
                label=f"AP {classes[1]}={ap:.3f}")
    else:
        for index, cls in enumerate(classes):
            binary_truth = (y_true == cls).astype(int)
            scores = proba[:, index]
            try:
                precision, recall, _ = precision_recall_curve(binary_truth, scores)
                ap = average_precision_score(binary_truth, scores)
            except ValueError as exc:
                _LOG.warning("PR curve for class %r skipped: %s", cls, exc)
                continue
            prevalence = float(binary_truth.mean())
            color = PALETTE[index % len(PALETTE)]
            ax.axhline(prevalence, linestyle="--", color=color, alpha=0.35,
                       lw=1.0, label=f"no skill {cls} ({prevalence:.2f})")
            ax.plot(recall, precision, lw=2.0, color=color,
                    label=f"AP {cls}={ap:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=9)
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


def _normalise_history(history: Any) -> pd.DataFrame:
    """Coerce a training ``history`` into a DataFrame (typed errors).

    Accepts the per-epoch record list any :mod:`pharmaml.ml.neural`
    estimator leaves on ``history_`` (``{'epoch', 'train_loss',
    'val_loss'}`` dicts -- ``val_loss`` may be ``None``), or an
    equivalent pandas DataFrame; a missing ``val_loss`` column is added
    as all-``NaN`` so downstream code has one shape to handle.
    """
    if isinstance(history, pd.DataFrame):
        frame = history
    else:
        try:
            # dict(record) also raises ValueError on plain strings ("ab"),
            # which are iterable but not per-epoch records -- catch both.
            records = [dict(record) for record in history]
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"history must be a list of per-epoch records or a DataFrame, "
                f"got {type(history).__name__}"
            ) from exc
        frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise DataValidationError(
            "history is empty -- pass the fitted estimator's history_ "
            "(any DeepMLP*/ResidualMLP* fit leaves it on the estimator)"
        )
    missing = [
        column for column in ("epoch", "train_loss") if column not in frame.columns
    ]
    if missing:
        raise DataValidationError(
            f"history is missing column(s) {missing} -- expected records with "
            "epoch, train_loss and optionally val_loss"
        )
    if "val_loss" not in frame.columns:
        frame = frame.assign(val_loss=np.nan)
    return frame


def plot_training_history(
    history: Any,
    title: str = "Training history",
) -> Any:
    """Epoch-by-epoch loss curves with the early-stopping verdict.

    The deep estimators of :mod:`pharmaml.ml.neural` record their whole
    optimisation in ``history_`` (one ``{'epoch', 'train_loss',
    'val_loss'}`` record per epoch); this figure turns it into the two
    curves a deep-learning run deserves -- training and validation loss
    on one axis -- plus the verdict a bare number cannot give:

    * a dashed line marks the best-validation epoch (the weights that
      were actually restored);
    * training that ended *on* the best epoch reports "still improving
      -- more epochs may pay off";
    * anything else reports how many patience-window epochs ran after
      the best;
    * a run without a validation split (``val_split=0``) says so
      instead of silently showing one curve.

    Parameters
    ----------
    history:
        The ``history_`` of a fitted deep estimator -- a list of
        per-epoch dicts or an equivalent DataFrame (duck-typed like the
        other viz helpers).  ``val_loss`` records may be ``None``
        (plotted as gaps) or absent entirely.
    title:
        Figure title (left-anchored; the verdict rides on the second
        line).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        When ``history`` is empty, holds records without the
        ``epoch``/``train_loss`` keys, or carries non-numeric losses.
    """
    frame = _normalise_history(history)
    try:
        epochs = pd.to_numeric(frame["epoch"], errors="raise").to_numpy(dtype=float)
        train = pd.to_numeric(frame["train_loss"], errors="raise").to_numpy(dtype=float)
        # val_loss legally holds None (no validation split / gap epochs):
        # coerce, so missing epochs plot as gaps instead of raising.
        val = pd.to_numeric(frame["val_loss"], errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"history holds non-numeric values -- conversion failed: {exc}"
        ) from exc

    fig, ax = new_figure(figsize=(8.4, 5.0))
    ax.plot(
        epochs, train, color=PALETTE[0], lw=1.9, marker="o", ms=3.5,
        label=f"training loss (final {train[-1]:.3g})",
    )
    has_val = bool(np.isfinite(val).any())
    if has_val:
        mask = np.isfinite(val)
        best_index = int(np.nanargmin(val))
        best_epoch = int(epochs[best_index])
        ax.plot(
            epochs[mask], val[mask], color=PALETTE[1], lw=1.9, marker="s", ms=3.5,
            label=f"validation loss (best {val[best_index]:.3g} @ epoch {best_epoch})",
        )
        ax.axvline(epochs[best_index], color=PALETTE[1], lw=1.0, ls="--", alpha=0.6)
        stale = len(epochs) - 1 - best_index
        verdict = (
            "validation still improving at the final epoch — more epochs may pay off"
            if stale == 0
            else f"early stopped {stale} epoch(s) after the best (epoch {best_epoch})"
        )
    else:
        verdict = "no validation split — early stopping was disabled (val_split = 0)"
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{title}\n{verdict}", fontsize=11, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False)
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


def plot_gp_uncertainty(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | list,
    y_sigma: np.ndarray | list,
    z: float = 1.96,
    metrics: Mapping[str, float] | None = None,
    title: str = "Gaussian process predictions with uncertainty",
) -> Any:
    """Parity scatter with ``± z·σ`` error bars — the GP's honest readout.

    The companion to :func:`plot_predicted_vs_actual` for models that
    know what they do not know: every prediction is drawn with its
    ``z``-sigma error bar (``z=1.96`` — the two-sided 95% band), and
    the coverage statistic — the share of test points whose bar
    actually straddles the truth — is annotated, because a calibrated
    GP straddles ~95% of them and an over-confident one visibly less.

    Parameters
    ----------
    y_true, y_pred:
        Held-out targets and the GP's predictive means.
    y_sigma:
        The GP's per-point predictive standard deviation (e.g. from
        :func:`pharmaml.ml.predict_with_uncertainty`), same length as
        ``y_true``.
    z:
        Error-bar width in sigmas (default 1.96 — the 95% band).
    metrics:
        Optional summary box content, same contract as
        :func:`plot_predicted_vs_actual`.

    Raises
    ------
    DataValidationError
        On length mismatches, empty input or a negative sigma.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(y_sigma, dtype=float)
    if truth.shape != prediction.shape or truth.shape != sigma.shape:
        raise DataValidationError(
            f"y_true {truth.shape}, y_pred {prediction.shape} and y_sigma "
            f"{sigma.shape} must match"
        )
    if truth.size == 0:
        raise DataValidationError("no values to plot (empty y_true/y_pred)")
    if float(sigma.min()) < 0.0:
        raise DataValidationError(
            f"y_sigma must be non-negative (found min {sigma.min():.4g}) — "
            "a predictive std can never be negative"
        )
    if not z > 0.0:
        raise DataValidationError(f"z must be positive, got {z!r}")

    fig, ax = new_figure(figsize=(6.4, 6.0))
    low = float(min(truth.min(), prediction.min()) - z * float(sigma.max()))
    high = float(max(truth.max(), prediction.max()) + z * float(sigma.max()))
    pad = 0.05 * (high - low or 1.0)
    limits = (low - pad, high + pad)
    ax.plot(limits, limits, linestyle="--", color="#9CA3AF", label="perfect prediction (ŷ = y)")
    covered = np.abs(truth - prediction) <= z * sigma
    coverage = float(covered.mean())
    ax.errorbar(
        truth,
        prediction,
        yerr=z * sigma,
        fmt="none",
        ecolor=PALETTE[2],
        elinewidth=1.2,
        capsize=2.5,
        alpha=0.85,
        label=f"±{z:g}·σ error bar",
    )
    ax.scatter(truth, prediction, s=28, alpha=0.9, color=PALETTE[0],
               edgecolor="white", linewidth=0.4, zorder=3,
               label="test predictions")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted (ŷ)")
    ax.set_title(title)

    lines = [f"coverage ±{z:g}·σ = {coverage * 100:.1f}%"]
    if metrics:
        lines.append("  ".join(f"{key}={value:.4g}" for key, value in metrics.items()))
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "#D1D5DB", "alpha": 0.9},
    )
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_prediction_intervals(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | list,
    y_lower: np.ndarray | list,
    y_upper: np.ndarray | list,
    alpha: float = 0.1,
    metrics: Mapping[str, float] | None = None,
    title: str = "Prediction intervals (quantile regression)",
) -> Any:
    """Parity scatter with the quantile-regression prediction band.

    The honest twin of :func:`plot_gp_uncertainty` for models whose
    uncertainty is a bracket, not a sigma: the median predictions
    scatter against the truth, the interval endpoints draw a band
    around them, and the coverage statistic — the share of test points
    whose interval actually straddles the truth — is annotated, because
    a calibrated quantile trio straddles ~``1 - alpha`` of them and a
    miscalibrated one visibly less.

    The band is drawn through the test rows sorted by actual value —
    the ``D2.py`` upload's deliberate calibration-style choice, kept:
    in the parity plane the ribbon stays visually coherent instead of
    zig-zagging with row order.

    Parameters
    ----------
    y_true, y_pred:
        Held-out targets and the median quantile predictions.
    y_lower, y_upper:
        The interval endpoints per test row (``y_lower <= y_upper``).
    alpha:
        The interval's tail mass, used only for the legend label
        (default 0.1 — a 90% band).
    metrics:
        Optional summary box content, same contract as
        :func:`plot_predicted_vs_actual`.

    Raises
    ------
    DataValidationError
        On length mismatches, empty input, or crossed bounds
        (``y_lower > y_upper`` — the quantile record repairs crossings,
        so raw crossed input reaching this plot is a caller bug).
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    lower = np.asarray(y_lower, dtype=float)
    upper = np.asarray(y_upper, dtype=float)
    if truth.shape != prediction.shape or truth.shape != lower.shape \
            or truth.shape != upper.shape:
        raise DataValidationError(
            f"y_true {truth.shape}, y_pred {prediction.shape}, y_lower "
            f"{lower.shape} and y_upper {upper.shape} must match"
        )
    if truth.size == 0:
        raise DataValidationError("no values to plot (empty y_true/y_pred)")
    if np.any(lower > upper):
        bad = int((lower > upper).sum())
        raise DataValidationError(
            f"y_lower exceeds y_upper on {bad} row(s) -- prediction "
            "intervals must be ordered bounds"
        )
    if not 0.0 < alpha < 1.0:
        raise DataValidationError(f"alpha must be in (0, 1), got {alpha!r}")

    fig, ax = new_figure(figsize=(6.4, 6.0))
    low = float(min(truth.min(), prediction.min(), lower.min()))
    high = float(max(truth.max(), prediction.max(), upper.max()))
    pad = 0.05 * (high - low or 1.0)
    limits = (low - pad, high + pad)
    ax.plot(limits, limits, linestyle="--", color="#9CA3AF",
            label="perfect prediction (ŷ = y)")
    order = np.argsort(truth, kind="stable")
    ax.fill_between(
        truth[order],
        lower[order],
        upper[order],
        alpha=0.25,
        color=PALETTE[2],
        linewidth=0,
        label=f"{(1.0 - alpha) * 100:.0f}% prediction interval",
    )
    covered = (truth >= lower) & (truth <= upper)
    coverage = float(covered.mean())
    mean_width = float((upper - lower).mean())
    ax.scatter(truth, prediction, s=28, alpha=0.85, color=PALETTE[0],
               edgecolor="white", linewidth=0.4, zorder=3,
               label="median prediction")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted (ŷ)")
    ax.set_title(title)

    lines = [
        f"coverage = {coverage * 100:.1f}%",
        f"mean width = {mean_width:.4g}",
    ]
    if metrics:
        lines.append("  ".join(f"{key}={value:.4g}" for key, value in metrics.items()))
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "#D1D5DB", "alpha": 0.9},
    )
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


def _extract_permutation(importance: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Names / means / stds out of a record or its ``as_frame`` twin.

    Accepts the :class:`~pharmaml.ml.evaluation.PermutationImportance`
    record (attribute access), any object with the same three
    attributes, or the DataFrame the record's ``as_frame`` returns
    (``feature`` / ``importance_mean`` / ``importance_std`` columns).
    """
    names: Any = None
    means: Any = None
    stds: Any = None
    if isinstance(importance, pd.DataFrame):
        missing = [
            column
            for column in ("feature", "importance_mean", "importance_std")
            if column not in importance.columns
        ]
        if missing:
            raise DataValidationError(
                "permutation frame must carry the columns feature / "
                "importance_mean / importance_std (missing: "
                + ", ".join(missing)
                + ") -- pass the PermutationImportance record or its "
                "as_frame() output"
            )
        names = importance["feature"].tolist()
        means = importance["importance_mean"].to_numpy(dtype=float)
        stds = importance["importance_std"].to_numpy(dtype=float)
    else:
        names = getattr(importance, "names", None)
        means = getattr(importance, "means", None)
        stds = getattr(importance, "stds", None)
    if names is None or means is None or stds is None:
        raise DataValidationError(
            "importance must be a PermutationImportance record (names/"
            "means/stds attributes) or its as_frame() DataFrame -- got "
            f"{type(importance).__name__}"
        )
    names_list = [str(name) for name in list(names)]
    means_array = np.asarray(means, dtype=float)
    stds_array = np.asarray(stds, dtype=float)
    if not (len(names_list) == len(means_array) == len(stds_array)):
        raise DataValidationError(
            f"names ({len(names_list)}), means ({len(means_array)}) and "
            f"stds ({len(stds_array)}) must carry one entry per feature"
        )
    return names_list, means_array, stds_array


def plot_permutation_importance(
    importance: Any,
    top_n: int | None = 15,
    title: str = "Permutation importance",
) -> Any:
    """Top-N permutation importances as signed bars with error bars.

    The figure ``D2.py`` drew from its raw
    ``sklearn.inspection.permutation_importance`` result, upgraded with
    the spread the original threw away: every bar carries its
    ``±1 std`` whisker, and NEGATIVE means (shuffling the column HELPED
    -- a noise feature) render in the palette's red instead of teal so
    the eye cannot miss them.

    Parameters
    ----------
    importance:
        A :class:`~pharmaml.ml.evaluation.PermutationImportance` record
        (or any object with ``names`` / ``means`` / ``stds``), or a
        DataFrame with ``feature`` / ``importance_mean`` /
        ``importance_std`` columns (the record's own ``as_frame``
        output).
    top_n:
        How many features to show (largest ``|mean|`` first); ``None``
        shows every feature.  Validated by :func:`_check_top_n`.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``importance`` carries neither the record attributes nor
        the DataFrame columns, the arrays disagree in length, or
        ``top_n`` is invalid.

    Examples
    --------
    >>> from pharmaml.ml.evaluation import PermutationImportance
    >>> import numpy as np
    >>> record = PermutationImportance(
    ...     names=["a", "b", "c"], means=np.array([0.5, -0.2, 0.1]),
    ...     stds=np.array([0.05, 0.02, 0.01]), scoring="r2", n_repeats=5,
    ...     n_samples=30)
    >>> fig = plot_permutation_importance(record)
    >>> ax = fig.axes[0]
    >>> [tick.get_text() for tick in ax.get_yticklabels()]  # bottom -> top
    ['c', 'b', 'a']
    >>> ax.get_xlabel()
    'importance (Δ score when shuffled)'
    >>> plot_permutation_importance(record, top_n=0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: top_n must be a positive integer or None, got 0
    """
    _check_top_n(top_n)
    names, means, stds = _extract_permutation(importance)
    keep = len(means) if top_n is None else min(int(top_n), len(means))
    # Largest |mean| first (stable, first-declared-wins ties), then
    # reversed for display so the strongest feature sits at the top.
    order = np.argsort(-np.abs(means), kind="stable")[:keep][::-1]

    fig, ax = new_figure(figsize=(8.4, 0.42 * len(order) + 1.7))
    positions = np.arange(len(order))
    colors = [PALETTE[3] if means[int(index)] < 0 else PALETTE[0] for index in order]
    ax.barh(
        positions,
        means[order],
        xerr=stds[order],
        color=colors,
        alpha=0.88,
        height=0.62,
        edgecolor="white",
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#525252"},
    )
    for pos, value in zip(positions, means[order]):
        ax.text(value, pos, f" {value:+.4g}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9)
    ax.axvline(0.0, color="#525252", lw=0.8, ls=":")
    ax.set_yticks(positions, [names[int(index)] for index in order])
    ax.set_xlabel("importance (Δ score when shuffled)")
    ax.set_title(title)
    return fig


def plot_permutation_heatmap(
    importances: Mapping[str, Any],
    *,
    top: int | None = 10,
    title: str = "",
) -> Any:
    """Cross-model permutation-importance heatmap (features × models).

    The matrix figure ``D2.py`` drew after looping
    ``permutation_importance`` over its base models and meta-learner
    (the audit's GAP 3): which features does EVERY model lean on, and
    where do the meta and the bases disagree?  Rows are features —
    the ``top`` largest mean absolute importances ACROSS models —
    columns are models, cells carry the model's own signed mean
    importance for that feature.  A feature a model never saw (a
    name absent from its record) renders light grey, honestly
    "no vote" instead of a fake zero.

    Parameters
    ----------
    importances:
        Model display name → a
        :class:`~pharmaml.ml.evaluation.PermutationImportance` record
        (or ``as_frame()`` DataFrame) — the direct output of
        :func:`~pharmaml.ml.evaluation.cross_model_importances`, or a
        hand-assembled panel.
    top:
        How many feature rows to show (largest cross-model mean
        ``|importance|`` first); ``None`` shows every feature.
        Validated by :func:`_check_top_n`.
    title:
        Chart title; empty falls back to ``"Cross-model permutation
        importance"``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``importances`` is empty, a value carries neither the
        record attributes nor the DataFrame columns (the
        :func:`_extract_permutation` contract), or the assembled
        matrix is entirely undefined.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.evaluation import PermutationImportance
    >>> def rec(means):
    ...     return PermutationImportance(
    ...         names=["signal", "noise"], means=np.asarray(means),
    ...         stds=np.zeros(2), scoring="r2", n_repeats=5, n_samples=30)
    >>> panel = {"ols": rec([0.5, 0.0]), "ridge": rec([0.4, -0.1])}
    >>> fig = plot_permutation_heatmap(panel)
    >>> ax = fig.axes[0]
    >>> [tick.get_text() for tick in ax.get_yticklabels()]   # rows
    ['signal', 'noise']
    >>> [tick.get_text() for tick in ax.get_xticklabels()]   # columns
    ['ols', 'ridge']
    >>> plot_permutation_heatmap({})
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: importances is empty — pass at least one model's PermutationImportance record (cross_model_importances builds the panel for you)
    """
    _check_top_n(top)
    if not importances:
        raise DataValidationError(
            "importances is empty — pass at least one model's "
            "PermutationImportance record (cross_model_importances builds "
            "the panel for you)"
        )

    model_names = [str(name) for name in importances]
    parsed = {
        name: _extract_permutation(record) for name, record in importances.items()
    }

    # Feature order: every name in first-appearance order, ranked by
    # the cross-model mean |importance| (stable, first-declared wins
    # ties — the module's sort convention).
    appearance: list[str] = []
    magnitudes: dict[str, list[float]] = {}
    for name in model_names:
        names, means, _ = parsed[name]
        for feature, mean in zip(names, means):
            if feature not in magnitudes:
                appearance.append(feature)
                magnitudes[feature] = []
            magnitudes[feature].append(abs(float(mean)))
    cross_mean = {
        feature: sum(magnitudes[feature]) / len(magnitudes[feature])
        for feature in appearance
    }
    ordered = sorted(appearance, key=lambda f: -cross_mean[f])
    keep = len(ordered) if top is None else min(int(top), len(ordered))
    ordered = ordered[:keep]

    # The matrix: rows = features (display order: strongest at the
    # TOP, the bar-chart convention), columns = models, NaN where a
    # model's record lacks the feature.
    matrix = np.full((len(ordered), len(model_names)), np.nan)
    for col, name in enumerate(model_names):
        names, means, _ = parsed[name]
        lookup = {feature: float(mean) for feature, mean in zip(names, means)}
        for row, feature in enumerate(ordered):
            if feature in lookup:
                matrix[row, col] = lookup[feature]
    if not np.isfinite(matrix).any():
        raise DataValidationError(
            "the assembled importance matrix is entirely undefined — no "
            "model's record carries any of the selected features"
        )

    annotated = len(ordered) <= 12 and len(model_names) <= 12
    rows = len(ordered)
    fig, ax = new_figure(
        figsize=(2.6 + 1.15 * len(model_names), 1.6 + 0.62 * rows)
    )
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "pharmaml_signed", [PALETTE[3], "#FFFFFF", PALETTE[0]]
    )
    cmap.set_bad("#E5E7EB")  # features the model never saw
    finite_max = float(np.nanmax(np.abs(matrix)))
    bound = max(finite_max, 1e-12)  # a symmetric, zero-centred scale
    image = ax.imshow(matrix, cmap=cmap, vmin=-bound, vmax=bound,
                      aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="importance (Δ score when shuffled)")

    if annotated:
        for row in range(rows):
            for col in range(len(model_names)):
                value = matrix[row, col]
                if np.isnan(value):
                    continue  # the grey cell already says "no vote"
                ax.text(
                    col, row, f"{value:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.6 * bound else "#1F2937",
                )
    ax.set_xticks(range(len(model_names)), model_names, rotation=30,
                  ha="right", fontsize=9)
    ax.set_yticks(range(rows), ordered, fontsize=9)
    suffix = "" if annotated else f" — {rows} features, cells unannotated"
    ax.set_title((title or "Cross-model permutation importance") + suffix)
    ax.grid(False)
    return fig


def plot_signed_correlations(
    correlations: pd.Series,
    *,
    top: int | None = 10,
    title: str = "",
    ax: Any = None,
    metric_label: str = "Pearson r",
) -> Any:
    """Top-N signed feature-target correlations as a horizontal bar chart.

    The ``D2.py`` upload's ``Top10_Correlation_Direction`` figure, in
    the toolkit's visual language: the strongest correlations sit at
    the TOP of the chart, POSITIVE bars render in the palette's teal
    and NEGATIVE ones in its red (the upload used green/red — the
    colour semantics are identical, the hues now match every other
    figure), and every bar carries its signed value as text.

    The whisker-free twin of :func:`plot_permutation_importance`:
    a correlation has no ±std to draw, so the bars carry no error
    whiskers — everything else (edge colours, the zero marker line,
    the per-bar value labels, the largest-|value|-at-the-top display
    order) mirrors that sibling exactly.

    The Series is trusted, not re-sorted: the FIRST ``top`` entries
    are plotted in the caller's order, reversed only for display (so
    entry 0 — the strongest, if the Series came from
    :func:`pharmaml.chem.interpret.signed_feature_correlations` —
    lands at the top).  Pass a Series sorted by ``|r|`` descending;
    an unsorted one plots its head, not its top.  NaN entries (an
    undefined correlation) are dropped with a warning before the
    head is taken, mirroring the computation side's honest drop.

    Parameters
    ----------
    correlations:
        Signed correlations indexed by feature name, ``|r|``
        descending (the ``signed_feature_correlations`` output).
    top:
        How many entries to show; ``None`` shows every entry.
        Validated by :func:`_check_top_n`.
    title:
        Chart title; empty falls back to ``"Top {n} features by
        |{metric_label}|"``.
    ax:
        Draw onto an existing axes instead of a fresh figure (the
        caller's axes are used as given — no restyling); the function
        returns that axes' figure.
    metric_label:
        The x-axis label — ``"Pearson r"`` (default) or ``"Spearman ρ"``
        for the rank flavour.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``correlations`` is not a ``pandas.Series``, its values
        are not numeric, it is empty, every entry is NaN, or ``top``
        is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.viz import close_figure
    >>> corr = pd.Series({"logp": 0.9, "mw": -0.8, "tpsa": 0.3})
    >>> fig = plot_signed_correlations(corr)
    >>> chart = fig.axes[0]
    >>> chart.get_xlabel()
    'Pearson r'
    >>> [tick.get_text() for tick in chart.get_yticklabels()]  # bottom -> top
    ['tpsa', 'mw', 'logp']
    >>> len(chart.patches)                                     # every bar
    3
    >>> close_figure(fig)
    >>> fig = plot_signed_correlations(corr, top=2, metric_label="Spearman ρ")
    >>> (len(fig.axes[0].patches), fig.axes[0].get_xlabel())
    (2, 'Spearman ρ')
    >>> close_figure(fig)
    >>> plot_signed_correlations(pd.Series(dtype=float))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: no correlations to plot (empty Series)
    >>> plot_signed_correlations(pd.Series({"logp": float("nan")}))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: every correlation is NaN (undefined -- constant columns?); nothing to plot
    """
    if not isinstance(correlations, pd.Series):
        raise DataValidationError(
            "correlations must be a pandas.Series sorted by |r| (the "
            "signed_feature_correlations output), got "
            f"{type(correlations).__name__}"
        )
    _check_top_n(top)
    try:
        values_all = correlations.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"correlation values must be numeric ({exc})"
        ) from exc
    values_all = values_all.reshape(-1)
    if values_all.size == 0:
        raise DataValidationError("no correlations to plot (empty Series)")
    finite = np.isfinite(values_all)
    if not finite.all():
        _LOG.warning(
            "Dropping %d NaN correlation(s) before plotting (undefined "
            "correlations are not a zero)",
            int((~finite).sum()),
        )
    values = values_all[finite]
    names = [str(name) for name, ok in zip(correlations.index, finite) if ok]
    if values.size == 0:
        raise DataValidationError(
            "every correlation is NaN (undefined -- constant columns?); "
            "nothing to plot"
        )

    keep = values.size if top is None else min(int(top), values.size)
    # The caller's order is trusted (head, not top); reversed for
    # display so the strongest entry sits at the top of the chart —
    # the plot_permutation_importance display convention.
    shown_values = values[:keep][::-1]
    shown_names = names[:keep][::-1]

    if ax is None:
        fig, ax = new_figure(figsize=(8.4, 0.42 * keep + 1.7))
    else:
        fig = ax.figure
    positions = np.arange(keep)
    colors = [PALETTE[3] if value < 0 else PALETTE[0] for value in shown_values]
    ax.barh(
        positions,
        shown_values,
        color=colors,
        alpha=0.88,
        height=0.62,
        edgecolor="white",
    )
    for pos, value in zip(positions, shown_values):
        ax.text(value, pos, f" {value:+.4g}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9)
    ax.axvline(0.0, color="#525252", lw=0.8, ls=":")
    ax.set_yticks(positions, shown_names)
    ax.set_xlabel(metric_label)
    ax.set_title(title or f"Top {keep} features by |{metric_label}|")
    return fig


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    title: str = "",
) -> Any:
    """Feature × feature correlation heatmap (the seaborn-free port).

    The ``D2.py`` upload's ``Feature_Correlation_Heatmap_with_pEC50``
    figure: the upload called ``sns.heatmap`` on the top-10 features
    plus the target; this port draws the same matrix in pure
    matplotlib (``imshow`` + colour bar + tick labels) so the toolkit
    keeps its matplotlib-only viz contract.  Pass the top-``top``
    columns of :func:`pharmaml.chem.interpret.signed_feature_correlations`
    plus the target column to reproduce the upload's exact figure.

    Two deliberate deviations from the upload: the FULL matrix is
    drawn (the original masked the upper triangle — a seaborn
    styling habit; the symmetric mirror and the trivial 1.0 diagonal
    cost nothing and aid scanning), and the diverging colormap is
    built from the house palette (red → white → teal, mirroring
    :func:`plot_signed_correlations`'s signed-bar colours) instead of
    seaborn's ``coolwarm`` — pinned to the honest correlation range
    ``[-1, 1]`` so the centre IS zero.  Cells whose correlation is
    undefined (a constant column's NaN row) render light grey.

    Parameters
    ----------
    frame:
        Frame carrying the columns to correlate (NaN cells are
        handled pairwise, the ``pandas.corr`` convention).
    columns:
        The columns to correlate, in display order; ``None`` uses
        every NUMERIC column (non-numeric columns are skipped with a
        warning, the ``clean_numeric_frame`` precedent — name them
        explicitly and a non-numeric column raises instead).
    method:
        ``"pearson"`` (default) or ``"spearman"``.
    title:
        Chart title; empty falls back to ``"Feature correlation
        heatmap"``.  The method (``" (spearman)"``) and, past 12
        columns, an ``"cells unannotated"`` note are appended to the
        effective title — the ``plot_confusion_matrix`` convention.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``frame`` is not a DataFrame, ``method`` is unknown,
        ``columns`` names a missing or non-numeric column, fewer than
        two numeric columns remain, or every correlation is undefined
        (all-constant input).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.viz import close_figure
    >>> rng = np.random.default_rng(42)
    >>> frame = pd.DataFrame(rng.uniform(size=(60, 4)),
    ...                       columns=["a", "b", "c", "d"])
    >>> fig = plot_correlation_heatmap(frame)
    >>> chart = fig.axes[0]
    >>> [tick.get_text() for tick in chart.get_yticklabels()]
    ['a', 'b', 'c', 'd']
    >>> len(chart.texts)                       # 4x4 cells annotated
    16
    >>> close_figure(fig)
    >>> wide = pd.DataFrame(rng.uniform(size=(30, 14)),
    ...                     columns=[f"f{i}" for i in range(14)])
    >>> fig = plot_correlation_heatmap(wide)   # >12 columns: no annotations
    >>> len(fig.axes[0].texts)
    0
    >>> "unannotated" in fig.axes[0].get_title()
    True
    >>> close_figure(fig)
    >>> plot_correlation_heatmap(frame, columns=["a", "missing"])
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: Missing required column(s) ['missing']. Available columns: ['a', 'b', 'c', 'd']
    """
    if method not in ("pearson", "spearman"):
        raise DataValidationError(
            f"method must be 'pearson' or 'spearman', got {method!r}"
        )
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    if columns is not None:
        column_list = list(columns)
        require_columns(frame, column_list)
        non_numeric = [
            name
            for name in column_list
            if not pd.api.types.is_numeric_dtype(frame[name])
        ]
        if non_numeric:
            raise DataValidationError(
                f"column(s) {non_numeric} are not numeric -- a correlation "
                "heatmap needs numeric columns; drop or coerce them first"
            )
    else:
        column_list = [
            name
            for name in frame.columns
            if pd.api.types.is_numeric_dtype(frame[name])
        ]
        skipped = [name for name in frame.columns if name not in column_list]
        if skipped:
            _LOG.warning(
                "Skipping %d non-numeric column(s): %s", len(skipped), skipped[:10]
            )
    if len(column_list) < 2:
        raise DataValidationError(
            "need at least 2 numeric columns for a correlation heatmap, got "
            f"{len(column_list)}: {column_list}"
        )

    matrix = frame.loc[:, column_list].corr(method=method)
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise DataValidationError(
            "every correlation is undefined (all columns constant?) -- "
            "nothing to heat-map"
        )

    n = len(column_list)
    # Per-cell annotations are only readable while cells stay
    # inch-sized (the plot_confusion_matrix guard: a 464-class canvas
    # once tried 176k text artists); past 12 columns the matrix
    # renders as a plain heat-map with small ticks.
    annotated = n <= 12
    side = 2.2 + 1.1 * min(n, 12)
    fig, ax = new_figure(figsize=(side, 2.2 + 1.0 * min(n, 12)))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "pharmaml_signed", [PALETTE[3], "#FFFFFF", PALETTE[0]]
    )
    cmap.set_bad("#E5E7EB")  # undefined (constant-column) cells
    image = ax.imshow(values, cmap=cmap, vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=f"{method} r")

    if annotated:
        for i in range(n):
            for j in range(n):
                value = values[i, j]
                if np.isnan(value):
                    continue  # the grey cell already says "undefined"
                ax.text(
                    j, i, f"{value:+.2f}", ha="center", va="center",
                    fontsize=9 if n <= 8 else 8,
                    color="white" if abs(value) > 0.6 else "#1F2937",
                )
    tick_labels = [str(name) for name in column_list]
    tick_size = 9 if annotated else max(4, 9 - n // 40)
    ax.set_xticks(range(n), tick_labels, rotation=45, ha="right",
                  fontsize=tick_size)
    ax.set_yticks(range(n), tick_labels, fontsize=tick_size)
    suffix = " (spearman)" if method != "pearson" else ""
    if not annotated:
        suffix += f" — {n} features, cells unannotated"
    ax.set_title((title or "Feature correlation heatmap") + suffix)
    ax.grid(False)
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


def plot_applicability_domain(
    result: Any,
    y_label: str = "predicted value",
    title: str = "Applicability domain",
    bins: int = 40,
) -> Any:
    """The uploaded AD workflow's combined two-panel reliability figure.

    Restores ``Applicability_domain_analysis.py``'s ``AD_Combined_
    Figure.jpeg`` (2026-10 upload) as the toolkit's honest readout of a
    QSAR prediction's trustworthiness:

    * **(a) training kNN-distance histogram** — the descriptor-space
      density the model was fitted on, with the percentile threshold
      that separates inside from outside;
    * **(b) distance vs prediction scatter** — every query compound at
      its mean kNN distance against the ensemble's predicted value,
      green inside the domain, red outside, the same threshold line.

    When the analysis ran without the bootstrap half (a kNN-only quick
    pass) panel (b) falls back to plotting the per-row distances alone
    — the ``y`` axis becomes the row index, honestly labelled.

    Parameters
    ----------
    result:
        An :class:`~pharmaml.ml.applicability.ApplicabilityDomainResult`
        (from ``applicability_domain`` or ``knn_applicability_domain``).
    y_label:
        Axis label for the prediction axis (e.g. ``"predicted pIC50"``).
    title:
        Figure title.
    bins:
        Histogram bin count for panel (a).

    Raises
    ------
    DataValidationError
        When ``result`` is not an applicability-domain result object.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import Ridge
    >>> from pharmaml.ml import applicability_domain
    >>> from pharmaml.viz import close_figure
    >>> x = pd.DataFrame({"a": np.linspace(0.0, 10.0, 60)})
    >>> y = pd.Series(2.0 * x["a"] + 1.0)
    >>> model = Ridge().fit(x, y)
    >>> res = applicability_domain(model, x, y,
    ...                            pd.DataFrame({"a": [0.5, 100.0]}), n_models=4)
    >>> fig = plot_applicability_domain(res, y_label="predicted pIC50")
    >>> len(fig.axes)
    2
    >>> ax_a, ax_b = fig.axes
    >>> ax_a.get_xlabel()
    'mean kNN distance (training set, standardised)'
    >>> ax_b.get_ylabel()
    'predicted pIC50'
    >>> [text.get_text() for text in ax_b.get_legend().get_texts()]
    ['inside domain', 'outside domain', 'threshold = 0.127']
    >>> plot_applicability_domain({"not": "a result"})
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: result must be an ApplicabilityDomainResult, got dict
    >>> close_figure(fig)
    """
    needed = ("train_knn_mean", "test_knn_mean", "inside_ad", "distance_threshold")
    if any(not hasattr(result, name) for name in needed):
        raise DataValidationError(
            "result must be an ApplicabilityDomainResult, got "
            f"{type(result).__name__}"
        )
    train_dist = np.asarray(result.train_knn_mean, dtype=float).reshape(-1)
    test_dist = np.asarray(result.test_knn_mean, dtype=float).reshape(-1)
    inside = np.asarray(result.inside_ad, dtype=bool).reshape(-1)
    threshold = float(result.distance_threshold)
    if test_dist.size != inside.size:
        raise DataValidationError(
            f"test_knn_mean ({test_dist.size}) and inside_ad ({inside.size}) "
            "must match"
        )
    k = int(getattr(result, "k", 5))
    percentile = float(getattr(result, "percentile", 95.0))
    n_inside = int(inside.sum())

    fig, (ax_a, ax_b) = new_figure(1, 2, figsize=(12.4, 5.2))
    fig.subplots_adjust(wspace=0.28)

    # panel (a): training distance distribution + threshold
    ax_a.hist(train_dist, bins=bins, color=PALETTE[0], alpha=0.85,
              edgecolor="white", linewidth=0.4)
    ax_a.axvline(threshold, color=PALETTE[3], linestyle="--", lw=1.8,
                 label=f"threshold = {threshold:.3g} "
                       f"(p{percentile:g}, k={k})")
    ax_a.set_xlabel("mean kNN distance (training set, standardised)")
    ax_a.set_ylabel("compounds")
    ax_a.set_title(f"(a) training distance density — {train_dist.size} compound(s)")
    ax_a.legend(fontsize=9)

    # panel (b): distance vs prediction, colour-coded by the verdict
    pred = getattr(result, "pred_mean", None)
    if pred is not None and np.asarray(pred).size == test_dist.size:
        pred = np.asarray(pred, dtype=float).reshape(-1)
        ax_b.scatter(test_dist[inside], pred[inside], s=30, alpha=0.8,
                     color=PALETTE[4], edgecolor="white", linewidth=0.4,
                     label="inside domain")
        if (~inside).any():
            ax_b.scatter(test_dist[~inside], pred[~inside], s=44, alpha=0.9,
                         color=PALETTE[3], edgecolor="white", linewidth=0.5,
                         marker="D", label="outside domain")
        ax_b.set_ylabel(y_label)
        y_lo = float(np.nanmin(pred))
        y_hi = float(np.nanmax(pred))
        pad = 0.06 * (y_hi - y_lo or 1.0)
        ax_b.set_ylim(y_lo - pad, y_hi + pad)
    else:
        positions = np.arange(test_dist.size)
        ax_b.scatter(test_dist[inside], positions[inside], s=30, alpha=0.8,
                     color=PALETTE[4], edgecolor="white", linewidth=0.4,
                     label="inside domain")
        if (~inside).any():
            ax_b.scatter(test_dist[~inside], positions[~inside], s=44, alpha=0.9,
                         color=PALETTE[3], edgecolor="white", linewidth=0.5,
                         marker="D", label="outside domain")
        ax_b.set_ylabel("query row")
        ax_b.set_ylim(-1, test_dist.size)
    ax_b.axvline(threshold, color=PALETTE[1], linestyle="--", lw=1.8,
                 label=f"threshold = {threshold:.3g}")
    ax_b.set_xlabel("mean kNN distance (query compounds, standardised)")
    ax_b.set_title(f"(b) distance vs prediction — {n_inside}/{test_dist.size} "
                   "inside the domain")
    ax_b.legend(fontsize=9, loc="best")
    fig.suptitle(title, fontsize=12)
    return fig
